# comfyui-obvpm-timeline

This fork adds **custom audio for lip-sync generation** to the original obvpm timeline. See [what changed](#custom-audio-in-this-fork) and [how to use it](#using-custom-audio).

A MiniMax H3 ComfyUI workflow and nodes that allow generating, extending, prepending, bridging and editing videos in an easy to use UI.

The timeline node is like a mini video editor.

And it also supports creating seamless looping videos.

<img src="assets/timeline-node.png" title="" alt="The Timeline node: three clips on the timeline, with the first one set to be prepended" width="804">

## Custom audio in this fork

- **Audio row below the video timeline:** upload or drop one soundtrack, seek with a visible audio cursor, select a source range, and audition that range before generating.
- **Audio conditioning for lip sync:** the matching source audio is encoded into H3's audio latent during generation. The pack's Save / Trim and Save nodes write the original waveform into the saved video.
- **Automatic chunk timing:** generate, extend, prepend, and bridge operations select their own audio start time and duration from the actual video frame range. Pinned overlaps are included, and picture and sound are trimmed together at the joins.
- **Audio output for transcription:** Timeline's new **chunk_audio** output provides the exact waveform used for the upcoming chunk, including its overlaps, so an ASR model can transcribe it before the video prompt is encoded.
- **Chunk plan (odd-first generation):** with a soundtrack loaded, the Timeline splits it into chunks and emits one planned run per Queue: independent first-pass roots (the odd chunks 1, 3, 5, …) and single-pass bridges (the even chunks 2, 4, …) pinned by both neighbours, run interleaved — 1, 3, 2, 5, 4, 7, 6… — each bridge right after the root that completes its pair. Every frame is sampled exactly once, so long narrations no longer degrade with each extension. See [chunk plan](docs/custom-audio.md#chunk-plan-odd-first-generation).
- **Existing workflows remain supported:** removing the custom track restores generated-audio behavior. The original four Timeline output sockets keep their positions.

## Updates

### Fork: custom audio

- Added custom-audio conditioning, waveform controls, and the **chunk_audio** output.
- Added the **chunk plan** toggle, the **bridge window** setting, and the **root_generation** and **plan_status** outputs for drift-free long audio-driven generations. **root_generation** (BOOLEAN) is **true** when the run about to happen is a root (odd chunk — or no plan at all) and **false** when it is a bridge (even chunk), so the prompt can branch on the role with no math.
- Updated the included timeline workflow with audio-to-ASR prompt wiring and generation mode enabled for an empty timeline.

### 0.1.0 (2026-09-21)

- First release.

## Watch the Tutorial

The best way to learn how to use the Timeline and workflow is by watching the YouTube tutorial video:

[![Watch the tutorial on YouTube](https://img.youtube.com/vi/kqP09NfJXaQ/hqdefault.jpg)](https://youtu.be/kqP09NfJXaQ)

**Watch on YouTube: [https://youtu.be/kqP09NfJXaQ](https://youtu.be/kqP09NfJXaQ)**

## Features

The main node is the **Timeline** node: the clips you generate are placed on it, and it is where you arrange, cut and preview them, and pick which clip the next generation continues from.

Generate a clip, see if you like it, add it to the timeline, then extend from it. Any part you don't like can be redone later.

Every generated clip is saved together with a motion context file (`.mctx`) that holds its latents. Extensions continue from those original latents using latent masks, not from re-encoded pixels, so there are no lighting changes or flickers at the joins. And because the files sit next to the clips, a clip can still be extended at full quality days or months later.

**Generating**

- **Extend** a clip seamlessly.
- **Prepend**: generate what happens *before* a clip
- **Bridge**: generate what happens between the end of one clip and the beginning of another. This lets you redo any section of the video, and bridging onto a fresh generation brings the quality back if it has drifted.
- **Loop**: bridge the last clip back to the first clip to make a seamless looping video.
- **Extend from a cut**: if a clip ends badly, cut off the bad part and extend from the good part. Snap cuts keeps cuts on the latent frame grid.
- Clips that have no motion context (external clips, clips from other workflows) can be extended too, with seam improvements applied automatically.
- **Result preview**: plays each new clip joined to its neighbours and rates how seamless the join is. From there you add it to the timeline, delete it or dismiss it.
- **Model preview override**: Integrated into the result preview. Lets you quickly see if a generation is going wrong.

**Timeline**

- **Custom audio for lip sync**: upload or drop a soundtrack onto the audio row below the video strip. Generation, extension, prepending and bridges automatically encode the matching audio window, including pinned overlap. See [custom audio setup and timing](docs/custom-audio.md).
- **Chunk plan**: with a soundtrack, one toggle turns long audio into a queue of drift-free runs — independent roots and single-pass bridges interleaved (1, 3, 2, 5, 4, 7, 6…), each frame sampled once. The strip leaves a bridge-sized hole between roots and marks in purple exactly where the next chunk lands. See [chunk plan](docs/custom-audio.md#chunk-plan-odd-first-generation).
- **Mini Video Editor**: Drag in clips from anywhere, reorder them by dragging, trim the ends, cut left / cut right at the playhead, undo cuts, and open gaps.
- **Quick preview**: plays through the clips right away. **Full preview** assembles them into a single video file, losslessly where possible. **Export** saves the finished video.
- **Load Settings, Prompts from Clips**: restores the prompt, seed, settings and references that a clip was generated with, so you don't have to keep track of them yourself.
- **Swap takes**: Swap between previous takes generated for the same extension, so you can go back to them and swap between them on the timeline.
- The sequence is also available as text, for copying a timeline into another workflow.

**Upscaling**

- A latent upscale and refine of the **whole sequence in one pass**, so details stay consistent across the joins and the result is still seamless.
- Sampling is done in windows, so long or high resolution videos still fit in VRAM.
- Generate at low resolution for fast iteration, then upscale once at the end. This is optional, you can also generate the first pass at final quality.
- Each clip's conditioning is saved with it, so the upscale uses the same references the clip was generated with.

**Workflow**

- Organized for fast and easy use. All inputs are in one area. No pan all over the workflow to change things.
- **Settings Presets**: Allows you to save frequently used settings such as steps, turbo, and sampling settings and control them in one place
- Up to four image references, plus video and audio references. (More can be added)

## Dependencies

The **workflows** use these custom node packs, so install them too (ComfyUI Manager's *Install Missing Custom Nodes* finds them):

| Custom node pack                                                                                       | Used for                                                             |
| ------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------- |
| [comfyui-obvpm](https://github.com/chanon/comfyui-obvpm)                                               | Bundles, Settings Presets, switches and gates, Load Images & Compose |
| [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)                                            | Set / Get nodes, model preview override, Sage attention patch        |
| [Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler) | The latent upscaler used by the upscale pass                         |
| [ComfyUI-MiniMax-H3-Turbo](https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo)                       | Turbo LoRA loader for larryvrh turbo                                 |
| [ComfyUI-Spectrum-MiniMax-H3](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3)                   | Spectrum acceleration                                                |
| [rgthree-comfy](https://github.com/rgthree/rgthree-comfy)                                              | Power Lora Loader                                                    |

The MiniMax H3 nodes themselves are part of ComfyUI.

## Installation

Clone (or copy) this folder into `ComfyUI/custom_nodes`:

```
cd ComfyUI/custom_nodes
git clone https://github.com/omegasrsw/comfyui-obvpm-timeline.git
```

Restart ComfyUI. No extra Python dependencies are required. A ComfyUI
from 2026-08-13 or later is needed.

If you already installed this fork, run `git pull` inside its folder, restart ComfyUI, and hard-refresh the browser with **Ctrl+F5**. Keep only one installed copy of the timeline pack.

## Using custom audio

1. Open the [included timeline workflow](workflows/h3_obvpm_timeline_r2v_v0.1.1-004.json). Use ComfyUI Manager's **Install Missing Custom Nodes** for any missing nodes. Its prompt branch uses Qwen3-ASR and LM Studio nodes; configure their models and your LM Studio connection, or replace that branch with your own ASR and prompt setup.
2. On **H3 MCtx Timeline**, leave **upscaling** off when generating your first clip. Upscaling requires an existing sequence.
3. Click **+ audio** below the video timeline, or drop your audio file onto that row. Upload your own file even if the example workflow shows a saved filename: workflow JSON stores the track settings, not the audio file itself.
4. Set the audio controls below, then choose your video duration and prompt. Keep Timeline's **length** output connected to the generation latent's length so audio and video use the same actual frame count.
5. Generate and add the result to the timeline. Select an extension, prepend, or bridge as usual; the corresponding audio window is calculated automatically for each run. Keep the intended soundtrack selected across takes.

| Audio control | What it does |
| --- | --- |
| **place s** | Places the audio anchor on the original timeline. You can also **Shift-drag the timeline waveform**. |
| **source start s** | Selects the source-file time at that anchor. For example, `30` starts the initial take at 30 seconds into the file, leaving earlier audio available for prepends. |
| **duration s** | Limits the source duration after the offset; `0` uses the rest of the file. Windows outside the available source are padded with silence. |
| **lip sync** | Sets audio denoise during sampling; default `0.5`. `0` holds the supplied audio latent fixed. The pack's saver still delivers the original waveform. |
| **cursor s / ▶** | Seeks to an exact source time / plays or pauses from the cursor without restarting. |
| **▶ selection** | Plays from the selected source start and automatically stops at the selected end. |
| **×** | Removes the custom track. |

Click or drag on either waveform to seek. The lower **Source** waveform shows the whole file: drag its green selection edges, or **Shift-drag** across it to select a new range. The **source start s** and **duration s** fields update to match. For example, set start to `30` and duration to `8`, then click **▶ selection** to check seconds 30–38. Use **▶** to pause/resume that preview; seeking elsewhere lets you audition from the new cursor.

Seeking only changes the preview cursor. Changing the selection changes the audio used for future generations. The selected audio duration does not change the video generation duration; set the Timeline's video duration separately.

### Wiring another workflow

The included workflow already connects the audio-conditioning path. For your own workflow, connect:

- Timeline **pin_specs** → **H3 MCtx Apply Pins** → **pin_specs**.
- H3 audio VAE → Apply Pins **audio_vae**.
- Apply Pins **latent** → generation sampler's latent input, and **conditioning** → its guider.
- Sampler output → **H3 MCtx Trim and Save Video** **samples**, with decoded images and Apply Pins **pins** connected as usual.

Apply Pins crops and encodes the source audio automatically; no separate manual audio-crop or audio-encode chain is needed. Use this pack's Save or Trim and Save node to preserve the original soundtrack in the output.

### Use the chunk transcript in your prompt

Connect **Timeline → chunk_audio** to your ASR node's **audio** input, then insert its transcript into the video prompt before text conditioning. For example, combine your scene description with `The speaker says: <ASR transcript>`.

The output contains the full upcoming generation window, including head/tail context and both bridge overlaps. It is available before Apply Pins and uses the same waveform that will be encoded, so this wiring does not create a dependency cycle. ASR timestamps start at the beginning of that raw chunk.

**Bypass the ASR branch when no custom track is selected or when upscaling:** `chunk_audio` is `None` in those modes.

### Avoid drift on long audio: the chunk plan

Chained extensions re-sample the previous clip's tail on every pass, so each extension in a long chain adds a little colour and sharpness drift. Turn on **chunk plan** (with a soundtrack selected) and the Timeline instead splits the audio into chunks and emits the next planned run on every Queue: the **odd chunks** are independent first-pass generations with no pin at all, the **even chunks** are each a single bridge pass pinned by both neighbours, and the two roles run **interleaved** — 1, 3, 2, 5, 4, 7, 6, 9, 8… — so a bridge runs as soon as both of its roots exist and the timeline closes each gap right after opening it. Every frame is sampled exactly once, so quality no longer depends on how many runs the video needed. The **root_generation** BOOLEAN output (right under **chunk_audio**) is **true** when this run generates a root (an odd chunk — or no chunk is planned at all, which is an ordinary root-like take) and **false** when it generates a bridge (an even chunk), so a simple boolean gate can swap the prompt per role with no math; the **plan_status** string output explains the plan and which chunk runs next; the video **duration** sets the chunk length, and **bridge window** the overlap each bridge borrows. Keep the same reference image and a consistent prompt across the runs. See [chunk plan](docs/custom-audio.md#chunk-plan-odd-first-generation) for the full mechanics.

### Timing and existing clips

Adding or changing a track affects future generations; it does not redo existing clips' sound or mouth motion. Uploaded audio is stored in `ComfyUI/input/timeline_audio`; copy that file too when moving a workflow to another machine.

Extensions include the previous clip's overlap, prepends include the destination overlap, and bridges include both. Saved audio coordinates keep existing takes aligned when prepending. A bridge must match both source clips' audio positions: if it reports incompatible timing, adjust the gap or generation length to the legal H3 length indicated by the error. A non-looping soundtrack cannot automatically wrap into an audio loop.

See [custom audio setup and timing](docs/custom-audio.md) for the full details, including source offsets, bridge alignment, and third-party savers.

## Workflow

The workflow is available in this fork's [workflows](workflows) folder.

## Other Notes

Every node in this pack is listed with **(obvpm)** after its name, so
searching the node menu for `obvpm` finds all of them. The names used
throughout this documentation leave that suffix off.

**AI Generated docs:**

**[docs/h3.md](docs/h3.md)** 
**[docs/h3-nodes.md](docs/h3-nodes.md)**

## License

[GPL-3.0](LICENSE). See
[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md) for acknowledgments.
