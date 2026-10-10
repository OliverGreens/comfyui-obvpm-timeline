"""Custom soundtrack windows in the timeline's 24 fps timebase.

The pin recipe is authoritative: raw windows include the scaffolding which
Trim removes later. Saved audio coordinates keep a prepend from retiming
the soundtrack of all the takes that already exist.
"""
import hashlib
import json
import logging
import math
import os

from . import frames as fr

MARKER = "timeline_audio"
LATENT_KEY = "obvpm_timeline_audio"
_LOG = logging.getLogger("obvpm.h3")


def parse_track(value):
    track = json.loads(value or "{}") if isinstance(value, str) else value
    if not track:
        return None
    if not isinstance(track, dict) or not isinstance(track.get("file"), str):
        raise ValueError("Timeline audio must name an uploaded audio file.")
    out = {"file": track["file"], "start_frame": int(track.get("start_frame", 0)),
           "source_start": float(track.get("source_start", 0)),
           "duration": float(track.get("duration", 0)),
           "denoise": float(track.get("denoise", 0.5))}
    if (not all(math.isfinite(v) for v in list(out.values())[1:])
            or out["source_start"] < 0 or out["duration"] < 0
            or not 0 <= out["denoise"] <= 1):
        raise ValueError("Invalid timeline audio offset, duration or lip-sync strength.")
    return out


def audio_path(filename):
    import folder_paths
    root = os.path.realpath(folder_paths.get_input_directory())
    path = os.path.realpath(os.path.join(root, filename))
    if os.path.commonpath([root, path]) != root or not os.path.isfile(path):
        raise ValueError("Timeline audio file is missing or outside ComfyUI/input: " + filename)
    return path


def track_id(track):
    # Denoise does not change the soundtrack's coordinates.
    data = {k: v for k, v in track.items() if k != "denoise"}
    stat = os.stat(audio_path(track["file"]))
    data["revision"] = [stat.st_size, stat.st_mtime_ns]
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def saved_anchor(header, identity):
    try:
        record = json.loads((header or {}).get(MARKER, "{}"))
        return int(record["raw_start"]) if record.get("id") == identity else None
    except (ValueError, TypeError, KeyError):
        return None


def sequence_positions(sequence):
    from .nodes_assemble import resolve_sequence, EmptySequence
    from .nodes_encode import probe_clip
    try:
        entries = resolve_sequence(sequence)
    except EmptySequence:
        return [], 0
    cursor = 0
    for entry in entries:
        entry["timeline_start"] = cursor
        if entry.get("gap"):
            cursor += int(entry["gap"])
        else:
            header = entry.get("header") or {}
            end = entry["exit"]
            if end is None:
                end = (int(header["delivered_frames"]) if header
                       else probe_clip(entry["path"])["frames"])
            cursor += max(0, end - entry["enter"])
    return entries, cursor


def timeline_origin(entries, identity):
    """Raw frame under delivered timeline frame 0, from the first saved
    anchor; 0 when no clip on the timeline carries this track's anchors."""
    for e in entries:
        anchor = saved_anchor(e.get("header"), identity)
        if anchor is not None:
            header = e.get("header") or {}
            return (anchor + int(header.get("pinned_head_frames", 0))
                    + e["enter"] - e["timeline_start"])
    return 0


def raw_start(pins, entries, total, frames, identity):
    """Absolute frame under raw frame zero, using resolved source windows."""
    origin = timeline_origin(entries, identity)
    starts = []
    for pin in pins:
        spec = pin["spec"]
        matches = [e for e in entries if not e.get("gap") and (
            (pin.get("source_id") and (e.get("header") or {}).get("self_id") == pin["source_id"])
            or e["clip"] == spec.get("source_path"))]
        if len(matches) != 1:
            raise ValueError("Custom audio needs each pinned source exactly once in the timeline.")
        e = matches[0]
        header = e.get("header") or {}
        base = saved_anchor(header, identity)
        if base is None:
            base = (origin + e["timeline_start"] - e["enter"]
                    - int(header.get("pinned_head_frames", 0)))
        start = base + int(spec["source_start"])
        if pin["place"] == "after":
            start -= frames - pin["covered"]
        starts.append(start)
    if len(starts) == 2 and starts[0] != starts[1]:
        required = frames + starts[1] - starts[0]
        raise ValueError(
            "Custom audio bridge does not align with both source clips: "
            "this run has %d raw frames; the audio anchors require %d (%.3fs). "
            "Adjust the gap/run length to a legal H3 length before generating."
            % (frames, required, required / fr.FPS))
    return starts[0] if starts else origin + total


def waveform_window(audio, track, start, frames):
    """Absolute sample boundaries; silence outside the placed source range."""
    sr = int(audio["sample_rate"])
    wave = audio["waveform"]
    if wave.ndim == 2:
        wave = wave.unsqueeze(0)
    source_zero = round(track["source_start"] * sr)
    lo = source_zero + round((start - track["start_frame"]) * sr / fr.FPS)
    hi = source_zero + round((start + frames - track["start_frame"]) * sr / fr.FPS)
    result = wave.new_zeros((1, wave.shape[1], hi - lo))
    limit = wave.shape[-1]
    if track["duration"]:
        limit = min(limit, source_zero + round(track["duration"] * sr))
    a, b = max(lo, 0), min(hi, limit)
    # Negative raw time is intentional: a prepend may use audio BEFORE
    # the source offset selected for the original root take.
    if b > a:
        result[..., a - lo:b - lo] = wave[:1, ..., a:b]
    return {"waveform": result, "sample_rate": sr}


def load_audio(filename):
    from .preview_route import _decode_audio
    audio = _decode_audio(audio_path(filename))
    if audio is None:
        raise ValueError("The uploaded file contains no audio stream.")
    return audio


def prepare_chunk_audio(specs, track, sequence, frames, forced_start=None):
    """One waveform shared by upstream ASR and downstream VAE encoding.

    forced_start places an UNPINNED root chunk at an absolute slot on the
    audio grid. The chunk plan needs this because a planned root's slot
    is fixed by the plan, while the natural start (end of the delivered
    timeline) is wrong while even chunks are still missing from it.
    """
    from .nodes_pins import _prepare_pins
    pins = _prepare_pins(specs, True, timing_only=True) if specs else []
    if sum(p["covered"] for p in pins) >= frames:
        raise ValueError("Custom audio: pin windows fill the generation chunk; increase its length.")
    identity = track_id(track)
    entries, total = sequence_positions(sequence)
    if forced_start is not None:
        if pins:
            raise ValueError("Custom audio: a forced chunk start is only legal for an unpinned root chunk.")
        start = int(forced_start)
    else:
        start = raw_start(pins, entries, total, frames, identity)
    sound = waveform_window(load_audio(track["file"]), track, start, frames)
    out = {"source_kind": MARKER, "track": track, "sequence": sequence,
           "audio": sound, "raw_start": start, "frames": frames, "identity": identity}
    if forced_start is not None:
        out["forced"] = True
    return out


def apply_audio(conditioning, latent, pins, descriptor, audio_vae):
    import torch
    import comfy.nested_tensor
    from .avpack import unpack_av, pack_av
    if audio_vae is None:
        raise ValueError("Custom timeline audio needs the H3 audio VAE connected to Apply Pins' audio_vae input.")
    track = parse_track(descriptor["track"])
    identity = track_id(track)
    video, previous = unpack_av(latent)
    frames = fr.pixel_frames(video.shape[2])
    entries, total = sequence_positions(descriptor["sequence"])
    if descriptor.get("forced") and not pins:
        # Planned root chunk: the plan owns its slot on the audio grid;
        # the end of the delivered timeline is not the answer while
        # other planned chunks are still missing from it.
        start = int(descriptor["raw_start"])
    else:
        start = raw_start(pins, entries, total, frames, identity)
    _LOG.info("obvpm.h3: custom audio %s, source %.6fs for %.6fs "
              "(%d raw frames including all pin overlaps)", track["file"],
              track["source_start"] + (start - track["start_frame"]) / fr.FPS,
              frames / fr.FPS, frames)
    if "audio" in descriptor:
        if (descriptor["frames"] != frames or descriptor["raw_start"] != start
                or descriptor["identity"] != identity):
            raise ValueError(
                "Timeline chunk_audio no longer matches this generation. Wire Timeline's "
                "length to the generation latent and use the same pin_specs for Apply Pins.")
        sound = descriptor["audio"]
    else:
        # Accept descriptors emitted by an older/custom caller.
        sound = waveform_window(load_audio(track["file"]), track, start, frames)
    wave = sound["waveform"]
    sr = int(getattr(audio_vae, "audio_sample_rate", 32000))
    if sr != sound["sample_rate"]:
        import torchaudio
        wave = torchaudio.functional.resample(wave, sound["sample_rate"], sr)
    encoded = audio_vae.encode(wave.movedim(1, -1))
    if encoded.ndim == 3:
        encoded = encoded.unsqueeze(0)
    ticks = previous.shape[-1]
    if encoded.shape[-1] < ticks:
        encoded = torch.cat((encoded, encoded[..., -1:].expand(
            *encoded.shape[:-1], ticks - encoded.shape[-1])), dim=-1)
    encoded = encoded[..., :ticks].to(previous.device, previous.dtype)
    if encoded.shape[1:-1] != previous.shape[1:-1]:
        raise ValueError("Custom audio VAE does not produce H3 audio latents.")
    encoded = encoded.expand(video.shape[0], *encoded.shape[1:]).clone()
    mask = latent.get("noise_mask")
    vm = (mask.tensors[0].clone() if mask is not None else
          torch.ones((1, 1) + tuple(video.shape[2:]), device=video.device))
    am = torch.full((1, 1) + tuple(encoded.shape[2:]), track["denoise"],
                    device=encoded.device, dtype=torch.float32)
    for pin in pins:
        count = fr.audio_total(pin["covered"])
        a, b = (0, count) if pin["place"] == "before" else (ticks - count, ticks)
        am[..., a:b] = 0
        matching = next((e for e in entries if pin.get("source_id")
                         and (e.get("header") or {}).get("self_id") == pin["source_id"]), None)
        if (matching and saved_anchor(matching.get("header"), identity) is not None
                and pin.get("audio") is not None and pin["audio"].shape[-1] == count):
            encoded[..., a:b] = pin["audio"].to(encoded.device, encoded.dtype)
    out = dict(latent)
    out.update(pack_av(video, encoded))
    out["noise_mask"] = comfy.nested_tensor.NestedTensor((vm, am))
    out[LATENT_KEY] = {"audio": sound, "latent": encoded,
                       "record": {"id": identity, "raw_start": start,
                                  "file": track["file"], "track": track}}
    # Replace audio guides as well: an arrival in 'both' mode must not
    # condition the new soundtrack on a different, generated voice.
    cond = []
    for embedding, values in conditioning:
        values = dict(values)
        guides = []
        for guide in values.get("minimax_keyframes", []):
            guide = dict(guide)
            if guide.get("audio_latent") is not None:
                a = round(float(guide["resolved_frame_index"]) * fr.FRAME_RESCALE)
                n = guide["audio_latent"].shape[-1]
                a = max(0, min(ticks - n, a))
                guide["audio_latent"] = encoded[..., a:a + n]
            guides.append(guide)
        if guides:
            values["minimax_keyframes"] = guides
        cond.append([embedding, values])
    return cond, out, pins


def source_sound(samples, head=0, frames=None):
    record = samples.get(LATENT_KEY)
    if not record:
        return None
    sound = record["audio"]
    sr = sound["sample_rate"]
    meta = record["record"]
    start = meta["raw_start"] - meta["track"]["start_frame"]
    base = round(start * sr / fr.FPS)
    lo = round((start + head) * sr / fr.FPS) - base
    hi = None if frames is None else round((start + head + frames) * sr / fr.FPS) - base
    return {"waveform": sound["waveform"][..., lo:hi], "sample_rate": sr}
