"""Odd-first chunk planning: geometry, slots, fill detection, next chunk.

Every frame is sampled exactly once: roots are independent first-pass
generations, bridges are single passes pinned by both neighbours. These
tests pin the arithmetic that makes that true -- AV-grid legality, the
stride, coverage, and the timeline-reading that decides what is next.
"""
import importlib
import json
import os
import sys
import types
import unittest
from unittest.mock import patch

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pack = sys.modules.setdefault("obvpm_tl_test", types.ModuleType("obvpm_tl_test"))
pack.__path__ = [ROOT]
cp = importlib.import_module("obvpm_tl_test.chunkplan")
ta = importlib.import_module("obvpm_tl_test.timeline_audio")
fr = importlib.import_module("obvpm_tl_test.frames")
av = importlib.import_module("obvpm_tl_test.avpack")


def track(**changes):
    return {"file": "voice.wav", "start_frame": 0, "source_start": 0,
            "duration": 60.0, "denoise": .5, **changes}


def entry(name, at=0, anchor=None, frames=192):
    header = {"self_id": name, "pinned_head_frames": 0,
              "delivered_frames": frames}
    if anchor is not None:
        header[ta.MARKER] = json.dumps({"id": "track", "raw_start": anchor})
    return dict(clip=name, timeline_start=at, enter=0, header=header)


class Geometry(unittest.TestCase):
    def test_default_eight_second_chunk_over_sixty_seconds(self):
        g = cp.geometry(1440, 8 * 24, 39)
        self.assertEqual(g["chunk_frames"], 192)
        self.assertEqual(g["bridge_frames"], 192)
        self.assertEqual(g["stride"], 306)          # 2*(192-39)
        self.assertEqual(g["n_odd"], 6)             # ceil((1440-192)/306)+1
        self.assertEqual(g["n_runs"], 11)           # 6 roots + 5 bridges
        self.assertGreaterEqual(g["coverage"], 1440)

    def test_short_audio_is_a_single_root(self):
        for length in (1, 100, 192):
            g = cp.geometry(length, 192, 39)
            self.assertEqual((g["n_odd"], g["n_runs"]), (1, 1), length)

    def test_one_frame_past_the_chunk_needs_a_bridge(self):
        self.assertEqual(cp.geometry(193, 192, 39)["n_odd"], 2)

    def test_chunk_length_snaps_to_the_shared_av_grid(self):
        self.assertEqual(cp.geometry(1440, 100, 39)["chunk_frames"], 90)

    def test_everything_stays_on_the_av_grids(self):
        g = cp.geometry(1440, 192, 39)
        self.assertEqual(g["chunk_frames"] % 17, 5)   # 17k+5 raw ladder
        self.assertEqual(g["stride"] % 51, 0)         # roots stay AV-aligned
        self.assertEqual((g["chunk_frames"] - 39) % 51, 0)  # masked-legal

    def test_off_grid_bridge_window_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "AV grid"):
            cp.geometry(1440, 192, 22)

    def test_chunk_must_exceed_twice_the_window(self):
        with self.assertRaisesRegex(ValueError, "twice"):
            cp.geometry(1440, 39, 39)

    def test_audio_frames_reads_the_row_duration(self):
        self.assertEqual(cp.audio_frames(track(duration=2.5)), 60)


class Slots(unittest.TestCase):
    def setUp(self):
        self.g = cp.geometry(1440, 192, 39)
        self.sl = cp.slots(0, self.g)

    def test_roots_step_by_the_stride_bridges_sit_between(self):
        roots = [s for s in self.sl if s["role"] == "root"]
        bridges = [s for s in self.sl if s["role"] == "bridge"]
        self.assertEqual([s["raw_start"] for s in roots],
                         [0, 306, 612, 918, 1224, 1530])
        self.assertEqual([s["raw_start"] for s in bridges],
                         [153, 459, 765, 1071, 1377])
        for b in bridges:
            self.assertEqual(b["right"] - b["left"], self.g["stride"])
            self.assertEqual(b["raw_start"], b["left"] + 192 - 39)

    def test_indices_are_odd_for_roots_even_for_bridges(self):
        self.assertEqual([s["index"] for s in self.sl], list(range(1, 12)))

    def test_generation_order_interleaves_bridges_after_their_roots(self):
        order = cp.generation_order(self.sl)
        self.assertEqual([s["index"] for s in order],
                         [1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10])

    def test_generation_order_thirteen_chunks(self):
        # The user-facing example: 13 chunks run as
        # 1, 3, 2(bridge 1-3), 5, 4(bridge 3-5), 7, 6, 9, 8, 11, 10,
        # 13, 12(bridge 11-13).
        g = cp.geometry(1900, 192, 39)
        self.assertEqual(g["n_odd"], 7)
        order = cp.generation_order(cp.slots(0, g))
        self.assertEqual([s["index"] for s in order],
                         [1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12])

    def test_generation_order_never_bridges_before_both_roots(self):
        g = cp.geometry(1900, 192, 39)
        order = cp.generation_order(cp.slots(0, g))
        seen = set()
        for s in order:
            if s["role"] == "bridge":
                self.assertIn(s["left"], seen)
                self.assertIn(s["right"], seen)
            seen.add(s["raw_start"])


class Plan(unittest.TestCase):
    def _plan(self, entries, total, **tk):
        with patch.object(ta, "track_id", return_value="track"), \
             patch.object(ta, "sequence_positions",
                          return_value=(entries, total)):
            return cp.plan(track(**tk), "sequence", 8.0, "39")

    def test_empty_timeline_starts_with_root_one(self):
        p = self._plan([], 0)
        self.assertEqual(p["base"], 0)
        self.assertEqual(p["next"]["index"], 1)
        self.assertEqual(p["next"]["role"], "root")
        self.assertIn("chunk 1 (root)", p["status"])
        self.assertIn("6 roots + 5 bridges", p["status"])

    def test_plan_is_anchored_by_its_clips_not_the_growing_timeline(self):
        # Root 1 butt-joined at the timeline end: total is 192, but the
        # plan must keep root 2 at base+stride, not drift by the total.
        p = self._plan([entry("A", anchor=0)], 192)
        self.assertEqual(p["base"], 0)
        self.assertEqual(p["next"]["index"], 3)
        self.assertEqual(p["next"]["raw_start"], 306)

    def test_bridge_is_next_once_its_two_roots_exist(self):
        # Interleaved order: roots 1 and 3 filled -> bridge 2 runs next,
        # long before the later roots are generated.
        roots = [entry("R%d" % j, at=j * 192, anchor=j * 306)
                 for j in range(2)]
        p = self._plan(roots, 2 * 192)
        self.assertEqual(p["next"]["index"], 2)
        self.assertEqual(p["next"]["role"], "bridge")
        self.assertEqual(p["next"]["raw_start"], 153)

    def test_all_roots_filled_still_bridges_in_order(self):
        roots = [entry("R%d" % j, at=j * 192, anchor=j * 306)
                 for j in range(6)]
        p = self._plan(roots, 6 * 192)
        self.assertEqual(p["next"]["index"], 2)
        self.assertEqual(p["next"]["role"], "bridge")
        self.assertEqual(p["next"]["raw_start"], 153)

    def test_first_bridge_anchor_reanchors_the_base(self):
        # Root 1 deleted, everything else kept: the earliest anchor is
        # bridge 2 at 153, which still says the plan starts at raw 0.
        kept = [entry("R%d" % j, anchor=j * 306) for j in range(1, 6)]
        kept.append(entry("B2", anchor=153))
        p = self._plan(kept, 6 * 192)
        self.assertEqual(p["base"], 0)
        self.assertEqual(p["next"]["index"], 1)
        self.assertEqual(p["next"]["raw_start"], 0)

    def test_complete_plan_reports_complete(self):
        filled = [entry("R%d" % j, anchor=j * 306) for j in range(6)]
        filled += [entry("B%d" % j, anchor=j * 306 + 153) for j in range(5)]
        p = self._plan(filled, 6 * 192)
        self.assertIsNone(p["next"])
        self.assertIn("complete", p["status"])

    def test_a_track_is_required(self):
        with self.assertRaisesRegex(ValueError, "soundtrack"):
            cp.plan(None, "", 8.0, "39")


class TimelineEmit(unittest.TestCase):
    """The Timeline node's chunk_plan toggle: each Run emits the NEXT
    chunk -- a forced-slot root, or a two-pin bridge."""

    SOURCE = {"waveform": torch.arange(4800, dtype=torch.float32)[None, None],
              "sample_rate": 24}

    def _emit(self, timeline, entries, total, **kw):
        na = importlib.import_module("obvpm_tl_test.nodes_assemble")

        def specs_for(clip, role, window, cut, mode, mask_shape=None,
                      guide_window=None):
            spec = {"source_kind": "clip", "source": {
                "self_id": clip,
                "video_latent": torch.zeros(1, 24, fr.frames_to_latents(192), 2, 2),
                "audio_latent": torch.zeros(1, 32, 2, fr.audio_total(192)),
                "meta": {"raw_frames": 192, "delivered_frames": 192}},
                "take_from": "tail" if role == "extend" else "head",
                "place": "before" if role == "extend" else "after",
                "requested_window": int(window)}
            if guide_window:
                spec["guide_window"] = guide_window
            return [spec]

        with patch.object(ta, "track_id", return_value="track"), \
             patch.object(ta, "load_audio", return_value=self.SOURCE), \
             patch.object(ta, "sequence_positions",
                          return_value=(entries, total)), \
             patch.object(timeline, "_specs_for", side_effect=specs_for):
            return timeline.emit(duration_seconds=8.0,
                                 audio_track=json.dumps(track()), **kw)

    def test_root_carries_no_pin_and_a_forced_audio_slot(self):
        na = importlib.import_module("obvpm_tl_test.nodes_assemble")
        specs, length, _, _, _, root, status = self._emit(
            na.H3Timeline(), [], 0, chunk_plan=True)
        self.assertEqual(length, 192)
        self.assertEqual(len(specs), 1)          # the audio marker only
        self.assertTrue(specs[0]["forced"])
        self.assertEqual(specs[0]["raw_start"], 0)
        self.assertIs(root, True)                # chunk 1: a root
        self.assertIn("chunk 1 (root)", status)

    def test_second_root_forces_the_next_stride_slot(self):
        na = importlib.import_module("obvpm_tl_test.nodes_assemble")
        specs, _, _, _, _, root, status = self._emit(
            na.H3Timeline(), [entry("A", anchor=0)], 192, chunk_plan=True)
        self.assertEqual(specs[0]["raw_start"], 306)
        self.assertIs(root, True)                # chunk 3: still a root
        self.assertIn("chunk 3 (root)", status)

    def test_bridge_pins_both_neighbours_and_aligns_by_construction(self):
        na = importlib.import_module("obvpm_tl_test.nodes_assemble")
        # Interleaved order: roots 1 and 3 exist -> bridge 2 is next.
        roots = [entry("R%d" % j, anchor=j * 306) for j in range(2)]
        specs, length, _, _, _, root, status = self._emit(
            na.H3Timeline(), roots, 2 * 192, chunk_plan=True)
        self.assertEqual(length, 192)
        self.assertEqual(len(specs), 3)          # extend + prepend + audio
        self.assertNotIn("forced", specs[-1])
        # Left tail at 192-39=153 and right head at 306-(192-39)=153:
        # the two pins agree, so the bridge audio is exact.
        self.assertEqual(specs[-1]["raw_start"], 153)
        self.assertIs(root, False)               # chunk 2: a bridge
        self.assertIn("chunk 2 (bridge)", status)
        # The arrival pin rides an edge-only guide row: the 10-frame
        # ramp plus lookahead, not the whole 39-frame window.
        self.assertEqual(specs[1]["guide_window"], 22)
        self.assertNotIn("guide_window", specs[0])  # masked extend: no row

    def test_plan_mode_ignores_a_manual_pin(self):
        na = importlib.import_module("obvpm_tl_test.nodes_assemble")
        manual = json.dumps({"role": "extend", "source": "A", "window": "39"})
        specs, _, _, _, _, root, status = self._emit(
            na.H3Timeline(), [], 0, chunk_plan=True, pin_state=manual)
        self.assertEqual(len(specs), 1)
        self.assertIs(root, True)
        self.assertIn("chunk 1 (root)", status)

    def test_plan_off_leaves_the_old_outputs_untouched(self):
        na = importlib.import_module("obvpm_tl_test.nodes_assemble")
        out = self._emit(na.H3Timeline(), [], 0)
        self.assertIs(out[5], True)              # unplanned: root-like
        self.assertEqual(out[6], "")             # plan_status empty
        self.assertNotIn("forced", out[0][-1])

    def test_upscale_skips_the_plan(self):
        na = importlib.import_module("obvpm_tl_test.nodes_assemble")
        specs, _, _, up, audio, root, status = self._emit(
            na.H3Timeline(), [], 0, chunk_plan=True, upscaling=True)
        self.assertTrue(up)
        self.assertIsNone(audio)
        self.assertIs(root, True)                # upscale: no planned chunk
        self.assertEqual(status, "")


if __name__ == "__main__":
    unittest.main()
