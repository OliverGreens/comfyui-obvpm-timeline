# Custom audio for lip sync

Restart ComfyUI and refresh the browser after updating the pack. The Timeline
now has an audio row below the video strip. Click **+ audio** or drop an audio
file there. One soundtrack can be placed per Timeline; it is saved with the
workflow, while the uploaded file lives in `ComfyUI/input/timeline_audio`.

The supplied timeline workflow already has the necessary wires:

1. Timeline **pin_specs** → H3 MCtx Apply Pins **pin_specs**.
2. H3 audio VAE → Apply Pins **audio_vae**.
3. Apply Pins **latent** → the generation sampler's latent input, and
   **conditioning** → its guider.
4. Sampler output → H3 MCtx Trim and Save Video **samples**, with the decoded
   images and Apply Pins' **pins** connected as usual.

No manual Load Audio / VAE Encode Audio / AV concatenate chain is needed.
Apply Pins performs those audio operations using the selected generation's
actual frame count. Removing the track restores the existing generated-audio
behavior. Older workflows without an audio row value behave as before.

## Transcribe the upcoming chunk for its prompt

Connect Timeline **chunk_audio** → your ASR node's **audio** input, then
insert the ASR transcript into the video prompt. This AUDIO output is
available before prompt conditioning or VAE encoding, so it creates no
circular dependency through Apply Pins.

It contains exactly the source waveform that will be encoded for the next
generation: the full raw chunk, including head/tail context and both bridge
overlaps, source offset, and any silence padding. ASR timestamps are relative
to the beginning of that raw chunk. Apply Pins reuses this same waveform
rather than loading and cropping another copy. Keep Timeline **length**
connected to the generation latent's length; a mismatch is rejected before
audio encoding.

**chunk_audio** is `None` when no custom track is selected or when the
Timeline is in upscale mode. Bypass the ASR branch in those cases. The
existing four output sockets keep their indices; **chunk_audio**,
**root_generation** and then **plan_status** are appended.

## Chunk plan: odd-first generation

Extending a clip re-samples its parent's tail window in a fresh pass, so a
long chain of extensions slowly accumulates colour and sharpness drift:
each link adds a little. The **chunk plan** toggle removes the chain. It
splits the soundtrack into chunks and generates them so that every frame
is sampled exactly once — either as a first-pass generation or as a single
bridge pass — no matter how long the audio is.

Turn on **chunk plan** (an audio track is required; upscale mode ignores
it). The Timeline measures the track and lays out slots on the audio grid:

- **Odd chunks (1, 3, 5, …)** are independent root generations. They carry
  no pin at all: each is a plain first-pass run whose audio is forced to
  its own slot, so chunk 3 never inherits anything from chunk 1.
- **Even chunks (2, 4, …)** fill the gaps. Each is one bridge run pinned by
  *both* neighbours: a masked extension of the left chunk's tail and a
  soft arrival into the right chunk's head. The bridge is the only thing
  that ever touches a neighbour's latents, and only its overlap.

Each **Queue the Timeline** then emits the *next* chunk of the plan —
roots and bridges **interleaved**: 1, 3, 2, 5, 4, 7, 6, 9, 8… Each bridge
runs as soon as both of its roots exist, so the timeline closes each gap
right after opening it instead of leaving every hole until the end. The
runs go through the usual **pin_specs**, **length**
and **chunk_audio** outputs, so the ASR-prompt wiring keeps working per
chunk. **root_generation** (a BOOLEAN output, right under **chunk_audio**)
says what this run is: **true** for a root — an odd chunk, or no chunk
planned at all (plan off, upscale mode, or the plan complete, which is an
ordinary root-like take) — and **false** for a bridge, the even chunk.
Feed it straight to a boolean gate or a text switch to swap the prompt per
role: root wording on true, bridge wording on false, no math needed.
**plan_status** (a STRING output) describes the plan and
names the next chunk; connect it to any text display.

Why a bridge used to cost more, and what changed: the arrival is steered
by a keyframe row — the trained-in signal that tells the model what it
must reach. Core packs that row into the *same* attention as the clip
being generated, so a row spanning the whole window adds a whole window
of extra conditioning tokens, and attention is quadratic: this is why
bridges were reported running 50–60% slower than roots. The row now
covers only the **join-facing edge** of the window — the ramped region
the model may move, plus a little lookahead (22 frames) — whatever the
bridge window is. The mask still holds the whole window, so the arrival
is exactly as exact; the deep frames already sit inside the target's own
sequence, and the bridge runs at near-root speed.

The Timeline's video **duration** is the chunk length (snapped to the
shared audio-video grid; it must exceed twice the **bridge window**, 39 or
90 frames). Roots step forward by a fixed stride, and each saved take
records its audio slot, so the plan stays anchored even while roots are
still butt-joined on the timeline and the bridges are missing. Adding a
take advances the plan; deleting one frees its slot for a rerun. Bridges
are inserted between their two source chunks automatically. When every
slot is filled the status reports the plan complete, and further runs
append past the plan as ordinary generations.

In the Timeline's own strip the plan is a toolbar toggle, **chunk plan**,
just left of **upscale**, filled purple while it owns the next run. While
it is on the strip shows where each chunk goes:

- **+ add** places the next root past the last one with a hole exactly one
  undelivered bridge wide between them — the two roots visibly do *not*
  run on from each other, because they are not continuous: the bridge
  between them is still missing. A bridge take instead fills the hole
  between its two roots — with the interleaved order that is the hole
  the last two roots just opened, exactly the bridge's length.
- The dash pill at a clip's end offers **next chunk**: it marks in purple
  the exact span the next run generates — the next root's slot past the
  last root, or the hole a bridge will fill. The band shows on the ruler
  and across the waveform, and the hole itself is tinted.
- **prepend** and **extend** go grey, because the plan owns the pin, and
  the row under the strip names the next chunk instead of a pin.

Keep the same reference image and a consistent prompt across the roots:
they are independent generations, and the shared conditioning plus the
bridge overlaps are what make them read as one continuous take.

## Audio row controls

- **place s** sets the audio anchor on the original timeline. **Shift-dragging
  the timeline waveform** changes this value at video-frame precision.
- **source start s** is the source-file time at that anchor. For example, `67`
  starts the initial take from 67 seconds into a song. A prepend can then
  use the earlier part of that same file.
- **duration s** limits the audio after the source offset. `0` uses the rest
  of the file. Before the beginning or after the end of the available
  source, the generation window is padded with encoded silence.
- **lip sync** controls audio denoise during sampling, default `0.5`.
  The model can resample the audio while producing mouth motion. The pack's
  Save and Trim and Save nodes always write the original waveform and
  store the supplied audio latent for later continuation. `0` holds the
  supplied audio latent throughout sampling.
- **cursor s** seeks to an exact time in the source file. Clicking or dragging
  either waveform also seeks, without changing the generation settings.
- **▶** plays or pauses from the cursor and resumes where you paused.
- **▶ selection** starts at **source start s** and stops automatically at the
  selected end. Pausing/resuming keeps that end boundary; seeking elsewhere
  returns to auditioning freely from the new cursor.

The lower **Source** waveform shows the entire file and highlights the selected
range in green. Drag either green edge to trim the range, or **Shift-drag** across
the source waveform to select a new one. The start and duration fields update
together. The yellow cursor appears on both waveforms; the upper waveform still
uses the video timeline's scale and shows the shaded next-generation region.

For example, set **source start s** to `30` and **duration s** to `8`, then click
**▶ selection** to audition seconds 30–38. Those values are also used for future
generation windows. They do not change the video duration: choose the Timeline's
video duration separately. If a generation extends past the available selection,
its audio is padded with silence. Preview controls do not queue a generation.

## Chunk and bridge timing

All generation ranges use the 24 fps H3 frame grid and the **resolved** pin
recipe, after source cuts and window snapping. An extension begins its raw
audio window inside the previous clip's overlap. A prepend includes the
destination overlap at its tail. A bridge includes both. The existing
handover/trim calculation removes the scaffolding from picture and sound
together, including soft arrival ramps.

Audio sample boundaries are calculated from absolute positions, avoiding
rounding drift across a chain of chunks. Saved takes record their source
audio coordinates in their sidecars. Inserting a prepend therefore keeps
the existing clips on their original part of the soundtrack; the audio row
adjusts its display to those saved coordinates. Changing the file, source
offset, placement or duration deliberately starts a new audio mapping.

For a bridge, both pinned sources must agree on the raw audio window. If
they do not, Apply Pins reports the required raw frame count and duration
before sampling. Adjust the gap or generation length to a legal H3 length.
Moving existing clips or cutting arbitrary amounts of speech cannot always
be joined by one continuous source-audio window. A wrap bridge to the start
of a non-looping soundtrack likewise cannot satisfy both audio anchors.

Existing clips keep the sound and mouth motion they were generated with;
adding an audio file affects subsequent generations. To maintain lip sync
through a whole sequence, start its generations with the intended track.
The upscale branch continues to use the saved audio latents as before.

Use the pack's Save or Trim and Save node for automatic original-audio
delivery. Third-party savers and latent transformations that discard extra
latent metadata require explicit source-audio wiring; they cannot recover
the original waveform from the denoised latent alone.
