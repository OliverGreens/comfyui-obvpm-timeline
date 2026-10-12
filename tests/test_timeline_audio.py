"""Soundtrack coordinates and AV conditioning, without loading model weights."""
import importlib
import json
import os
import sys
import types
import unittest
from unittest.mock import Mock, patch

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pack = sys.modules.setdefault("obvpm_tl_test", types.ModuleType("obvpm_tl_test"))
pack.__path__ = [ROOT]
ta = importlib.import_module("obvpm_tl_test.timeline_audio")
fr = importlib.import_module("obvpm_tl_test.frames")
av = importlib.import_module("obvpm_tl_test.avpack")


def track(**changes):
    return {"file": "voice.wav", "start_frame": 0, "source_start": 0,
            "duration": 0, "denoise": .5, **changes}


def entry(name, at=0, head=0, enter=0, anchor=None):
    header = {"self_id": name, "pinned_head_frames": head}
    if anchor is not None:
        header[ta.MARKER] = json.dumps({"id": "track", "raw_start": anchor})
    return dict(clip=name, timeline_start=at, enter=enter, header=header)


def pin(name, place, start, covered=39):
    return {"source_id": name, "place": place, "covered": covered,
            "spec": {"source_start": start, "source_frames": covered}}


class Coordinates(unittest.TestCase):
    def test_root_starts_at_end_of_timeline(self):
        self.assertEqual(ta.raw_start([], [], 0, 243, "track"), 0)
        self.assertEqual(ta.raw_start([], [entry("A")], 243, 243, "track"), 243)

    def test_extend_includes_context_and_resolved_cut(self):
        # Source has a trimmed head; it enters at delivered frame 10.
        entries = [entry("A", at=200, head=39, enter=10)]
        self.assertEqual(ta.raw_start([pin("A", "before", 136)],
                                     entries, 400, 243, "track"), 287)

    def test_prepend_reads_before_original_root(self):
        self.assertEqual(ta.raw_start([pin("A", "after", 0)],
                                     [entry("A")], 243, 141, "track"), -102)

    def test_saved_anchor_survives_prepend_and_save_reload(self):
        entries = [entry("P", 0, anchor=-102), entry("A", 102, anchor=0)]
        self.assertEqual(ta.raw_start([pin("A", "before", 204)],
                                     entries, 345, 243, "track"), 204)
        self.assertEqual(ta.raw_start([], entries, 345, 243, "track"), 243)

    def test_bridge_uses_both_raw_windows_not_handover_duration(self):
        entries = [entry("A"), entry("B", at=170, head=39, enter=25)]
        # B raw start lies at timeline 106; raw 51 is at 157.
        pins = [pin("A", "before", 4), pin("B", "after", 51)]
        self.assertEqual(ta.raw_start(pins, entries, 500, 192, "track"), 4)

    def test_bridge_mismatch_reports_required_raw_length(self):
        with self.assertRaisesRegex(ValueError, r"require 192"):
            ta.raw_start([pin("A", "before", 4), pin("B", "after", 51)],
                         [entry("A"), entry("B", 170, 39, 25)], 500, 243, "track")

    def test_pixel_source_uses_path(self):
        p = pin("", "before", 45)
        p["spec"]["source_path"] = "import.mp4"
        self.assertEqual(ta.raw_start([p], [entry("import.mp4", 90)], 500, 243, "track"), 135)

    def test_repeated_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly once"):
            ta.raw_start([pin("A", "before", 0)], [entry("A"), entry("A", 243)], 486, 243, "track")


class WaveformWindows(unittest.TestCase):
    def test_offset_and_placement(self):
        audio = {"waveform": torch.arange(1000)[None, None], "sample_rate": 24}
        got = ta.waveform_window(audio, track(source_start=10, start_frame=48), 48, 39)
        torch.testing.assert_close(got["waveform"], audio["waveform"][..., 240:279])

    def test_prepend_uses_earlier_source_audio(self):
        audio = {"waveform": torch.arange(1000)[None, None], "sample_rate": 24}
        got = ta.waveform_window(audio, track(source_start=10), -102, 141)
        torch.testing.assert_close(got["waveform"], audio["waveform"][..., 138:279])

    def test_silence_before_and_after_file(self):
        audio = {"waveform": torch.ones(1, 2, 30), "sample_rate": 24}
        got = ta.waveform_window(audio, track(), -5, 39)["waveform"]
        self.assertEqual(got.shape, (1, 2, 39))
        self.assertEqual(got[..., :5].sum(), 0)
        self.assertEqual(got[..., 5:35].sum(), 60)
        self.assertEqual(got[..., 35:].sum(), 0)

    def test_source_duration_trims_and_pads(self):
        audio = {"waveform": torch.ones(1, 1, 100), "sample_rate": 24}
        got = ta.waveform_window(audio, track(duration=1), 0, 39)["waveform"]
        self.assertEqual(got.sum(), 24)

    def test_absolute_sample_boundaries_join_without_drift(self):
        audio = {"waveform": torch.arange(20000)[None, None], "sample_rate": 1000}
        config = track(source_start=1)
        # Repeated 17-frame advances are not integral milliseconds.
        chunks = [ta.waveform_window(audio, config, i * 17, 17)["waveform"] for i in range(20)]
        expected = ta.waveform_window(audio, config, 0, 340)["waveform"]
        torch.testing.assert_close(torch.cat(chunks, -1), expected)

    def test_delivered_trim_uses_absolute_sample_phase(self):
        audio = {"waveform": torch.arange(5000)[None, None], "sample_rate": 1000}
        config = track()
        sound = ta.waveform_window(audio, config, 17, 73)
        samples = {ta.LATENT_KEY: {"audio": sound, "record": {
            "raw_start": 17, "track": config}}}
        got = ta.source_sound(samples, 39, 22)
        expected = ta.waveform_window(audio, config, 56, 22)
        torch.testing.assert_close(got["waveform"], expected["waveform"])


class Conditioning(unittest.TestCase):
    def test_saver_uses_original_audio_and_records_coordinates(self):
        ns = importlib.import_module("obvpm_tl_test.nodes_save")
        video = torch.zeros(1, 24, fr.frames_to_latents(39), 2, 2)
        sampled = torch.zeros(1, 32, 2, 65)
        original = torch.ones_like(sampled)
        samples = av.pack_av(video, sampled)
        sound = {"waveform": torch.ones(1, 2, 3900), "sample_rate": 2400}
        record = {"id": "track", "raw_start": 0, "file": "voice.wav", "track": track()}
        samples[ta.LATENT_KEY] = {"audio": sound, "latent": original, "record": record}
        with patch.object(ns.folder_paths, "get_save_image_path", return_value=("tmp", "take", 1, "", "")), \
             patch.object(ns.H3SaveVideoWithMCtx, "_encode_mp4") as encode, \
             patch.object(ns.mctx, "hash_file", return_value="hash"), \
             patch.object(ns.mctx, "write_sidecar", return_value="sidecar") as save, \
             patch.object(ns, "_save_conditioning"), patch.object(ns, "_workflow_blobs", return_value={}):
            ns.H3SaveVideoWithMCtx().save(torch.zeros(39, 32, 32, 3), samples,
                                        "test", "take", 19)
        torch.testing.assert_close(encode.call_args.args[2]["waveform"], sound["waveform"])
        torch.testing.assert_close(save.call_args.args[2], original)
        self.assertEqual(json.loads(save.call_args.args[3][ta.MARKER]), record)

    def test_audio_embedding_preserves_video_mask_and_original_sound(self):
        import comfy.nested_tensor
        video = torch.zeros(1, 24, fr.frames_to_latents(243), 2, 2)
        old_audio = torch.zeros(1, 32, 2, fr.audio_total(243))
        latent = av.pack_av(video, old_audio)
        vm = torch.ones(1, 1, video.shape[2], 2, 2)
        vm[:, :, :12] = 0
        latent["noise_mask"] = comfy.nested_tensor.NestedTensor((vm, torch.ones_like(old_audio)))
        p = pin("A", "before", 204)
        p["audio"] = torch.full((1, 32, 2, 65), 7.)
        audio = {"waveform": torch.ones(1, 2, 24000), "sample_rate": 2400}

        class VAE:
            audio_sample_rate = 2400

            def encode(self, wave):
                self.received = wave
                return torch.full((1, 32, 2, 405), 3.)

        vae = VAE()
        cond = [[None, {"minimax_keyframes": [
            {"resolved_frame_index": 0, "latent": video},
            {"resolved_frame_index": 0, "audio_latent": p["audio"]}]}]]
        with patch.object(ta, "track_id", return_value="track"), \
             patch.object(ta, "load_audio", return_value=audio), \
             patch.object(ta, "sequence_positions", return_value=([entry("A", anchor=0)], 243)):
            out_cond, out, _ = ta.apply_audio(cond, latent, [p], {"track": track(), "sequence": "A"}, vae)
        torch.testing.assert_close(out["noise_mask"].tensors[0], vm)
        self.assertEqual(out["noise_mask"].tensors[1][..., :65].max(), 0)
        self.assertEqual(out["noise_mask"].tensors[1][..., 65:].min(), .5)
        self.assertEqual(out["samples"].tensors[1][..., :65].min(), 7)
        self.assertEqual(out["samples"].tensors[1][..., 65:].min(), 3)
        self.assertEqual(out_cond[0][1]["minimax_keyframes"][1]["audio_latent"].min(), 7)
        self.assertEqual(out[ta.LATENT_KEY]["record"]["raw_start"], 204)
        self.assertEqual(vae.received.shape, (1, 24300, 2))
        self.assertEqual(old_audio.sum(), 0)

    def test_root_audio_marker_reaches_apply_even_without_video_pins(self):
        np = importlib.import_module("obvpm_tl_test.nodes_pins")
        na = importlib.import_module("obvpm_tl_test.nodes_assemble")
        with patch.object(ta, "track_id", return_value="track"), \
             patch.object(ta, "sequence_positions", return_value=([], 0)), \
             patch.object(ta, "load_audio", return_value={
                 "waveform": torch.ones(1, 1, 2400), "sample_rate": 24}):
            specs, length, _, _, chunk, _, _ = na.H3Timeline().emit(audio_track=json.dumps(track()))
        self.assertEqual(len(specs), 1)
        self.assertIs(specs[0]["audio"], chunk)
        node = np.H3MCtxApplyPins()
        with patch.object(node, "_apply", return_value=("cond", "latent", [])) as apply, \
             patch.object(np.nodes_masked, "core_masks_available", return_value=True), \
             patch.object(ta, "apply_audio", return_value=("new", "audio", [])) as custom:
            self.assertEqual(node.apply("cond", {}, True, pin_specs=specs, audio_vae="vae"), ("new", "audio", []))
            self.assertEqual(apply.call_args.kwargs["pin_specs"], [])
            custom.assert_called_once()
        with self.assertRaisesRegex(ValueError, "audio_vae"):
            node.apply("cond", {}, True, pin_specs=specs)

    def test_old_workflow_emits_no_audio_marker(self):
        na = importlib.import_module("obvpm_tl_test.nodes_assemble")
        self.assertEqual(na.H3Timeline().emit()[0], [])
        self.assertEqual(na.H3Timeline().emit(upscaling=True, audio_track=json.dumps(track()))[0], [])
        self.assertIsNone(na.H3Timeline().emit()[4])
        self.assertIsNone(na.H3Timeline().emit(upscaling=True, audio_track=json.dumps(track()))[4])


class ChunkAudioOutput(unittest.TestCase):
    def test_asr_and_vae_receive_identical_audio_for_every_generation_role(self):
        na = importlib.import_module("obvpm_tl_test.nodes_assemble")
        np = importlib.import_module("obvpm_tl_test.nodes_pins")
        video = torch.zeros(1, 24, fr.frames_to_latents(243), 2, 2)
        source_latent = torch.zeros(1, 32, 2, fr.audio_total(243))
        source = {"waveform": torch.arange(2400, dtype=torch.float32)[None, None],
                  "sample_rate": 24}

        def specs_for(clip, role, *args, **kwargs):
            return [{"source_kind": "clip", "source": {
                "self_id": clip, "video_latent": video, "audio_latent": source_latent,
                "meta": {"raw_frames": 243, "delivered_frames": 243}},
                "take_from": "tail" if role == "extend" else "head",
                "place": "before" if role == "extend" else "after",
                "requested_window": 39}]

        class VAE:
            audio_sample_rate = 24

            def encode(self, wave):
                self.received = wave
                return source_latent.clone()

        for role, start in (("root", 0), ("extend", 204), ("prepend", -204), ("bridge", 204)):
            with self.subTest(role=role):
                state = {"role": role, "source": "A", "window": "39"}
                entries, total = ([entry("A")], 243)
                if role == "bridge":
                    state["source2"] = "B"
                    entries.append(entry("B", at=408))
                    total = 651
                elif role == "root":
                    entries, total = [], 0
                timeline = na.H3Timeline()
                with patch.object(timeline, "_specs_for", side_effect=specs_for), \
                     patch.object(ta, "track_id", return_value="track"), \
                     patch.object(ta, "load_audio", return_value=source) as load, \
                     patch.object(ta, "sequence_positions", return_value=(entries, total)):
                    specs, length, _, _, sound, _, _ = timeline.emit(
                        pin_state="" if role == "root" else json.dumps(state),
                        duration_seconds=243 / 24,
                        audio_track=json.dumps(track(source_start=10)))
                    expected = ta.waveform_window(source, track(source_start=10), start, 243)
                    self.assertEqual(length, 243)
                    torch.testing.assert_close(sound["waveform"], expected["waveform"])
                    pins = np._prepare_pins(specs[:-1], True) if specs[:-1] else []
                    vae = VAE()
                    _, result, _ = ta.apply_audio([[None, {}]], av.pack_av(video, source_latent),
                                                pins, specs[-1], vae)
                    self.assertIs(result[ta.LATENT_KEY]["audio"], sound)
                    torch.testing.assert_close(vae.received.movedim(-1, 1), sound["waveform"])
                    load.assert_called_once()  # ASR's slice is reused by the encoder.

    def test_pixel_timing_does_not_require_a_vae_or_decode_video(self):
        np = importlib.import_module("obvpm_tl_test.nodes_pins")
        # Probe is I/O; isolate it from Comfy's optional GPU memory runtime.
        ne = types.ModuleType("obvpm_tl_test.nodes_encode")
        ne.probe_clip = Mock(return_value={"frames": 30})
        ne.decode_window = Mock()
        nl = importlib.import_module("obvpm_tl_test.nodes_load")
        spec = {"source_kind": "clip_pixels", "source_path": "import.mp4",
                "take_from": "tail", "place": "before", "requested_window": 39}
        package = sys.modules["obvpm_tl_test"]
        with patch.object(nl, "resolve_clip_path", return_value="import.mp4"), \
             patch.dict(sys.modules, {"obvpm_tl_test.nodes_encode": ne}), \
             patch.object(package, "nodes_encode", ne, create=True):
            resolved, = np._prepare_pins([spec], True, timing_only=True)
            self.assertEqual(resolved["covered"], 22)
            self.assertEqual(resolved["spec"]["source_start"], 8)
            ne.decode_window.assert_not_called()

    def test_mismatched_generation_length_is_rejected_before_encoding(self):
        source = {"waveform": torch.ones(1, 1, 2400), "sample_rate": 24}
        with patch.object(ta, "track_id", return_value="track"), \
             patch.object(ta, "load_audio", return_value=source), \
             patch.object(ta, "sequence_positions", return_value=([], 0)):
            descriptor = ta.prepare_chunk_audio([], track(), "", 243)
            target = av.pack_av(torch.zeros(1, 24, fr.frames_to_latents(192), 2, 2),
                                torch.zeros(1, 32, 2, fr.audio_total(192)))
            with self.assertRaisesRegex(ValueError, "chunk_audio no longer matches"):
                ta.apply_audio([[None, {}]], target, [], descriptor, object())


class PlannedRootAudio(unittest.TestCase):
    """A chunk-plan root owns its slot: the timeline end is not the answer
    while other planned chunks are still missing from it."""

    def test_forced_start_places_the_root_at_its_plan_slot(self):
        source = {"waveform": torch.arange(4800, dtype=torch.float32)[None, None],
                  "sample_rate": 24}
        with patch.object(ta, "track_id", return_value="track"), \
             patch.object(ta, "load_audio", return_value=source), \
             patch.object(ta, "sequence_positions",
                          return_value=([entry("A", anchor=0)], 192)):
            natural = ta.prepare_chunk_audio([], track(), "", 192)
            self.assertEqual(natural["raw_start"], 192)  # the timeline end
            self.assertNotIn("forced", natural)
            forced = ta.prepare_chunk_audio([], track(), "", 192, forced_start=384)
        self.assertEqual(forced["raw_start"], 384)
        self.assertTrue(forced["forced"])
        expected = ta.waveform_window(source, track(), 384, 192)
        torch.testing.assert_close(forced["audio"]["waveform"], expected["waveform"])

    def test_forced_start_rejects_pinned_specs(self):
        spec = {"source_kind": "clip", "source": {
            "self_id": "A",
            "video_latent": torch.zeros(1, 24, fr.frames_to_latents(243), 2, 2),
            "audio_latent": torch.zeros(1, 32, 2, fr.audio_total(243)),
            "meta": {"raw_frames": 243, "delivered_frames": 243}},
            "take_from": "tail", "place": "before", "requested_window": 39}
        with patch.object(ta, "track_id", return_value="track"), \
             patch.object(ta, "load_audio", return_value={
                 "waveform": torch.ones(1, 1, 4800), "sample_rate": 24}), \
             patch.object(ta, "sequence_positions", return_value=([], 0)):
            with self.assertRaisesRegex(ValueError, "unpinned root"):
                ta.prepare_chunk_audio([spec], track(), "", 243, forced_start=384)

    def test_apply_audio_trusts_a_forced_descriptor_and_records_the_slot(self):
        source = {"waveform": torch.ones(1, 1, 4800), "sample_rate": 24}
        video = torch.zeros(1, 24, fr.frames_to_latents(192), 2, 2)
        latent = torch.zeros(1, 32, 2, fr.audio_total(192))

        class VAE:
            audio_sample_rate = 24

            def encode(self, wave):
                return latent.clone()

        with patch.object(ta, "track_id", return_value="track"), \
             patch.object(ta, "load_audio", return_value=source), \
             patch.object(ta, "sequence_positions",
                          return_value=([entry("A", anchor=0)], 192)):
            descriptor = ta.prepare_chunk_audio([], track(), "", 192, forced_start=384)
            _, out, _ = ta.apply_audio([[None, {}]], av.pack_av(video, latent),
                                       [], descriptor, VAE())
        # The natural start (192) would have failed the match check; the
        # forced slot passes it and is saved for the plan to find later.
        self.assertEqual(out[ta.LATENT_KEY]["record"]["raw_start"], 384)


if __name__ == "__main__":
    unittest.main()
