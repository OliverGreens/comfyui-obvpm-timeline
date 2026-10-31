"""Odd-first chunk planning: roots are the odd chunks, bridges fill gaps.

Why this exists: every masked extend re-samples its parent's tail window
in a fresh pass, so a chain of extends accumulates sampling/VAE drift --
each generation looks a little flatter and sharper than the last. The
plan removes chains entirely. The odd chunks (1, 3, 5, ...) are
independent first-pass ROOT generations; the even chunks (2, 4, ...) each
bridge one gap in exactly ONE pass, pinned at the head by the left
neighbour's tail (masked extend) and at the tail by the right neighbour's
head (both-mode arrival). Every delivered frame is therefore sampled
exactly once, no matter how long the soundtrack is, and every join is a
measured-seamless bridge rather than a chain link.

Geometry (all raw frames on the 24 fps grid; W, C and L live on the
shared AV grid so masked runs and their audio ticks stay legal):

    W  bridge window, pinned on BOTH sides of every bridge
    C  odd chunk length (also the bridge length L, so stride is symmetric)
    S = C + L - 2W      odd-to-odd stride (2*(C-W) when L == C)

    root   j (0-based): raw_start = base + j*S,           frames = C
    bridge j (roots j, j+1): raw_start = base + j*S+C-W,  frames = L

A bridge's before-pin takes the left root's last W raw frames (its
source_start is C-W), and its after-pin takes the right root's first W
raw frames (source_start 0); both anchors agree exactly when the right
root sits S frames after the left one -- which is what the slots encode.

Coverage: (n_odd - 1)*S + C >= audio frames. The LAST root, though, is
cut to the audio it actually has to cover instead of a full chunk:
needed = audio - (n_odd-1)*S frames (its first W frames are the bridge's
motion context, already inside that count), plus one second of headroom,
floored at 5 s and capped at C -- under 5 s of audio left generate 5 s,
5-9 s generate needed+1, 9 s and up generate the normal chunk. The
shortened root still carries the W head frames its bridge's after-pin
takes, and coverage never drops below the end of the soundtrack.

Generation ORDER interleaves the two roles: root 1, root 2, bridge 2,
then root 3, bridge 4, root 4, bridge 6, ... -- each bridge runs as soon
as its two roots exist, so the timeline closes each gap right after
opening it instead of leaving every hole until the end. A bridge can
never be scheduled before its roots. A slot counts as filled
when a clip on the timeline carries a saved audio anchor at exactly its
raw_start -- the same anchors Apply Pins writes into every take's
sidecar, so "what is done" is read from the timeline itself and the plan
survives reloads, reorders and partial sessions.
"""
import logging
import math

from . import frames as fr

_LOG = logging.getLogger("obvpm.h3")

# Bridge windows that are legal masked windows AND offered in the UI.
WINDOW_CHOICES = ("39", "90")


def audio_frames(track):
    """Frames of soundtrack to cover: the row's duration, else the file."""
    if track["duration"] > 0:
        return int(round(track["duration"] * fr.FPS))
    from .timeline_audio import audio_path
    import av
    seconds = 0.0
    with av.open(audio_path(track["file"])) as container:
        if container.duration:
            seconds = container.duration / 1_000_000.0
        else:
            for stream in container.streams.audio:
                if stream.duration and stream.time_base:
                    seconds = float(stream.duration * stream.time_base)
                break
    if not seconds > 0 or not math.isfinite(seconds):
        raise ValueError("chunk plan: cannot read the duration of %s; set "
                         "the audio row's duration s instead." % track["file"])
    return int(round(seconds * fr.FPS))


def last_root_frames(needed, C, window):
    """Length of the FINAL root: only the audio it still has to cover.

    ``needed`` is the audio from the last root's slot to the end of the
    soundtrack -- the root's first W frames are the bridge's motion
    context and already inside that count. Generating a full chunk there
    samples silence, so: under 5 s left generate 5 s; up to 9 s generate
    one second more than needed; past that the normal chunk. The result
    stays on the shared AV grid, never uncovers the soundtrack, never
    drops below the window its bridge's after-pin takes, and never
    exceeds the normal chunk length.
    """
    from . import nodes_masked
    if needed >= 9 * fr.FPS:
        return C
    target = 5 * fr.FPS if needed < 5 * fr.FPS else needed + fr.FPS
    last = int(nodes_masked.shared_av_snap_nearest(target))
    lo = nodes_masked.shared_av_snap_down(needed)
    floor = (nodes_masked.SHARED_AV_GRID_BASE if lo is None
             else lo if lo >= needed
             else lo + nodes_masked.SHARED_AV_GRID_STEP)
    return min(max(last, floor, window), C)


def geometry(audio_len, chunk_frames, window):
    """Slot sizes and counts for one soundtrack; pure, no I/O."""
    from . import nodes_masked
    window = int(window)
    if not nodes_masked.masked_window_ok(window):
        raise ValueError("chunk plan: bridge window %d is not on the shared "
                         "AV grid (39, 90, 141, ...)." % window)
    if audio_len <= 0:
        raise ValueError("chunk plan: the soundtrack has no length to cover.")
    C = int(nodes_masked.shared_av_snap_nearest(chunk_frames))
    if C <= 2 * window:
        raise ValueError("chunk plan: chunk length %d frames must exceed "
                         "twice the bridge window (%d); raise the chunk "
                         "duration or use window 39." % (C, window))
    L = C
    S = C + L - 2 * window
    n_odd = 1 if audio_len <= C else -(-(audio_len - C) // S) + 1
    last = last_root_frames(audio_len - (n_odd - 1) * S, C, window)
    return {"audio_frames": int(audio_len), "chunk_frames": C,
            "bridge_frames": L, "window": window, "stride": S,
            "n_odd": n_odd, "n_runs": 2 * n_odd - 1,
            "last_root_frames": last,
            "coverage": (n_odd - 1) * S + last}


def slots(base, g):
    """Every slot in the plan, in timeline order (odd, even, odd, ...).

    The final root is the short one (see last_root_frames): it generates
    only the audio it still has to cover, not a full chunk of silence.
    Its raw_start and audio anchor are unchanged, so the bridge before
    it -- which takes just its first W frames -- aligns exactly the same.
    """
    C, L, W, S, n = (g["chunk_frames"], g["bridge_frames"], g["window"],
                     g["stride"], g["n_odd"])
    last = g.get("last_root_frames", C)
    out = []
    for j in range(n):
        out.append({"index": 2 * j + 1, "role": "root",
                    "raw_start": base + j * S,
                    "frames": last if j == n - 1 else C})
    for j in range(n - 1):
        out.append({"index": 2 * j + 2, "role": "bridge",
                    "raw_start": base + j * S + C - W, "frames": L,
                    "left": base + j * S, "right": base + (j + 1) * S})
    return sorted(out, key=lambda s: s["raw_start"])


def generation_order(sl):
    """Root 1, root 2, bridge 2, root 3, bridge 4, root 4, ...

    Two roots open the first gap, then every bridge runs right after the
    root that opens the next one: 1, 3, 2, 5, 4, 7, 6, 9, 8, ... Each
    bridge is still only scheduled once BOTH of its roots precede it.
    """
    roots = [s for s in sl if s["role"] == "root"]
    bridges = [s for s in sl if s["role"] == "bridge"]
    out = roots[:1]
    for j in range(1, len(roots)):
        out.append(roots[j])
        out.append(bridges[j - 1])
    return out


def filled_anchors(sequence, identity):
    """{saved raw_start: clip} for this track, plus (origin, total)."""
    from .timeline_audio import sequence_positions, saved_anchor, timeline_origin
    entries, total = sequence_positions(sequence)
    origin = timeline_origin(entries, identity)
    filled = {}
    for e in entries:
        if e.get("gap"):
            continue
        anchor = saved_anchor(e.get("header"), identity)
        if anchor is not None:
            filled[anchor] = e["clip"]
    return filled, origin, total


def _seconds(raw_start, base, frames):
    lo = (raw_start - base) / float(fr.FPS)
    return lo, lo + frames / float(fr.FPS)


def _status(g, nxt, done, base):
    head = ("chunk plan: %.2f s audio -> %d roots + %d bridges (%d runs), "
            "chunk %.2f s, bridge %.2f s, window %.2f s; %d/%d done"
            % (g["audio_frames"] / float(fr.FPS), g["n_odd"],
               g["n_odd"] - 1, g["n_runs"],
               g["chunk_frames"] / float(fr.FPS),
               g["bridge_frames"] / float(fr.FPS),
               g["window"] / float(fr.FPS), done, g["n_runs"]))
    if g.get("last_root_frames", g["chunk_frames"]) < g["chunk_frames"]:
        head += ", last root %.2f s" % (g["last_root_frames"] / float(fr.FPS))
    if nxt is None:
        return head + "; plan complete -- further runs append past the plan"
    lo, hi = _seconds(nxt["raw_start"], base, nxt["frames"])
    return head + "; next: chunk %d (%s) covering audio %.2f-%.2f s" % (
        nxt["index"], nxt["role"], lo, hi)


def plan(track, sequence, chunk_seconds, window):
    """Full plan for the current track + timeline; see module docstring.

    Returns the geometry, every slot, the filled map, the next slot in
    generation order (None when complete) and a human status line.
    """
    from .timeline_audio import track_id
    if not track:
        raise ValueError("chunk plan needs a soundtrack: add an audio file "
                         "on the Timeline's audio row.")
    identity = track_id(track)
    g = geometry(audio_frames(track), float(chunk_seconds) * fr.FPS, window)
    filled, origin, total = filled_anchors(sequence, identity)
    if filled:
        # The plan is anchored by its own clips, not by the timeline:
        # roots butt-join while bridges are still missing, so the
        # timeline's total grows past the plan and would drift every
        # slot. Saved anchors are absolute positions on the audio grid,
        # immune to where the clips sit. The earliest anchor is either a
        # root (a multiple of the stride) or the first bridge.
        a = min(filled)
        base = (a - (g["chunk_frames"] - g["window"])
                if a % g["stride"] else a)
    else:
        base = origin + total
    sl = slots(base, g)
    nxt = None
    for s in generation_order(sl):
        if s["raw_start"] not in filled:
            nxt = s
            break
    done = sum(1 for s in sl if s["raw_start"] in filled)
    g.update({"identity": identity, "base": base, "slots": sl,
              "filled": filled, "next": nxt,
              "status": _status(g, nxt, done, base)})
    return g
