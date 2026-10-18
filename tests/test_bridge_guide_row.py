"""The bridge's arrival row rides the join-facing edge, not the whole window.

A "both" pin pays for its keyframe row in packed-attention tokens: core
packs the row's full-resolution steps into the SAME joint attention as
the target (12 steps at window 39, 27 at 90, against a 57-step
192-frame target), and attention is quadratic -- reported 2026-10:
plan bridges ran 50-60% slower than roots for exactly this. The mask
half is free (it writes into the target latent, which is already in the
sequence), and the deep window is held at mask 0 either way, so only
the ramped edge needs the trained-in steering row.

These tests pin the arithmetic: the row is shorter, anchored where it
always was, sliced from the join-facing edge of the same phase-0 slice,
and the mask window is untouched.

Run from the pack folder (with the Python that runs ComfyUI):
  python tests/test_bridge_guide_row.py
"""
import importlib
import os
import sys
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = os.path.abspath(os.path.join(ROOT, "..", ".."))
if COMFY not in sys.path:
    sys.path.insert(0, COMFY)
pack = sys.modules.get("obvpm_tl_test")
if pack is None:
    pack = types.ModuleType("obvpm_tl_test")
    pack.__path__ = [ROOT]
    sys.modules[pack.__name__] = pack

import torch  # noqa: E402

fr = importlib.import_module("obvpm_tl_test.frames")  # noqa: E402


class GuideWindow(unittest.TestCase):
    """arrival_guide_window: smallest ladder window past the ramp."""

    def test_covers_the_ramp_and_caps_at_the_mask_window(self):
        na = importlib.import_module("obvpm_tl_test.nodes_assemble")
        # The plan's 10-frame ramp -> 22 frames (7 steps) whatever the
        # mask window: the whole point is that the row stops costing
        # more as the window grows.
        self.assertEqual(na.arrival_guide_window(39, 10), 22)
        self.assertEqual(na.arrival_guide_window(90, 10), 22)
        # No ramp: the minimal trained-in anchor.
        self.assertEqual(na.arrival_guide_window(39, 0), 5)
        # A big ramp still never exceeds the window it steers into.
        self.assertEqual(na.arrival_guide_window(22, 10), 22)
        self.assertEqual(na.arrival_guide_window(39, 30), 39)


def bundle(frames=192):
    """A source bundle whose video steps are identifiable by index."""
    steps = fr.frames_to_latents(frames)
    video = torch.zeros(1, 24, steps, 2, 2)
    video[0, 0, :, 0, 0] = torch.arange(steps, dtype=torch.float32)
    return {"self_id": "SRC",
            "video_latent": video,
            "audio_latent": torch.zeros(1, 32, 2, fr.audio_total(frames)),
            "meta": {"raw_frames": frames, "delivered_frames": frames}}


class GuideSlice(unittest.TestCase):
    """_prepare_pins cuts the row from the join-facing edge."""

    def _pin(self, place, guide=22, mode="both", window=39):
        np = importlib.import_module("obvpm_tl_test.nodes_pins")
        spec = {"source_kind": "clip", "source": bundle(),
                "source_id": "SRC", "source_kind": "clip",
                "take_from": "head" if place == "after" else "tail",
                "take_from_frame": 0, "place": place,
                "requested_window": window, "audio_window": 0,
                "mode": mode, "mask_ramp_frames": 10,
                "mask_ramp_edge": 0.4, "mask_hold": 0.0}
        if guide:
            spec["guide_window"] = guide
        return np._prepare_pins([spec], True)[0]

    def test_after_pin_row_is_the_window_head(self):
        pin = self._pin("after")
        self.assertEqual(pin["steps"], fr.frames_to_latents(39))   # 12
        self.assertEqual(pin["guide_steps"], fr.frames_to_latents(22))  # 7
        # The arrival window is the source's head; the row is its first
        # 22 frames -- the edge the free content must flow into.
        self.assertEqual(pin["guide_video"][0, 0, :, 0, 0].tolist(),
                         list(range(7)))

    def test_before_pin_row_is_the_window_tail(self):
        pin = self._pin("before")
        # A before-pin faces the join with its END: the row is the
        # window's last 22 frames, and 39-22 is a whole 17-frame group,
        # so the slice stays on the phase-0 grid.
        self.assertEqual(pin["guide_video"][0, 0, :, 0, 0].tolist(),
                         list(range(50, 57)))

    def test_masked_mode_never_carries_a_row(self):
        pin = self._pin("after", mode="masked")
        self.assertIsNone(pin["guide_video"])
        self.assertEqual(pin["guide_steps"], 0)

    def test_full_window_guide_is_no_guide(self):
        # guide == window: the row would duplicate the whole slice;
        # keep the single-slice path instead of a pointless clone.
        pin = self._pin("after", guide=39)
        self.assertIsNone(pin["guide_video"])


class KeyframeRow(unittest.TestCase):
    """Apply attaches the short row at the window's own anchor."""

    def test_row_is_short_anchor_is_the_window_base_mask_is_full(self):
        np = importlib.import_module("obvpm_tl_test.nodes_pins")
        av = importlib.import_module("obvpm_tl_test.avpack")
        target_video = torch.zeros(1, 24, fr.frames_to_latents(192), 2, 2)
        target_audio = torch.zeros(1, 32, 2, fr.audio_total(192))
        latent = av.pack_av(target_video, target_audio)

        def spec(guide):
            s = {"source_kind": "clip", "source": bundle(),
                 "source_id": "SRC",
                 "take_from": "head", "take_from_frame": 0,
                 "place": "after", "requested_window": 39,
                 "audio_window": 0, "mode": "both",
                 "mask_ramp_frames": 10, "mask_ramp_edge": 0.4,
                 "mask_hold": 0.0}
            if guide:
                s["guide_window"] = guide
            return s

        node = np.H3MCtxApplyPins()
        # The core probes import comfy.ldm.minimax -- heavy and optional
        # here. The mask machinery itself is pure torch; the anchor rule
        # is core's, checked live at run time, not in this test.
        from unittest.mock import patch
        with patch.object(np.nodes_masked, "core_masks_available",
                          return_value=True), \
             patch.object(np, "_ensure_native_anchors", lambda: None):
            out, latent_out, pins = node.apply(
                [[None, {}]], latent, False, pin_specs=[spec(22)])
            # Control: no guide_window -> the whole window rides, as
            # before.
            out2, _, _ = node.apply([[None, {}]], latent, False,
                                    pin_specs=[spec(None)])
        kfs = out[0][1]["minimax_keyframes"]
        video_rows = [k for k in kfs if k.get("latent") is not None]
        self.assertEqual(len(video_rows), 1)
        self.assertEqual(video_rows[0]["latent"].shape[2],
                         fr.frames_to_latents(22))     # 7, not 12
        # Anchored where the full-window row sat: the window's base.
        self.assertEqual(video_rows[0]["resolved_frame_index"], 192 - 39)
        # The mask window is untouched: trim and hold still see 39.
        self.assertEqual(pins[0]["covered"], 39)
        self.assertIsNotNone(latent_out.get("noise_mask"))

        rows2 = [k for k in out2[0][1]["minimax_keyframes"]
                 if k.get("latent") is not None]
        self.assertEqual(rows2[0]["latent"].shape[2],
                         fr.frames_to_latents(39))


if __name__ == "__main__":
    unittest.main()
