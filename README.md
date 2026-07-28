# Video Scene Detector

Find the exact timestamps of a scene you describe in plain language, across one video or an entire folder of them — then cut the matching clips automatically.

Runs entirely on your own machine with local open-weight models. Nothing is uploaded anywhere.

```
"a man lying on the ground"
        │
        ▼
  ┌───────────────────────────────────────────────────┐
  │  Phase 1  Coarse scan   2-min windows             │  ← recall
  │  Phase 2  Medium scan   30-sec sub-windows        │  ← precision
  │  Phase 3  Binary search boundaries to ~4s         │  ← exact edges
  └───────────────────────────────────────────────────┘
        │
        ▼
  Scene 1: 00:14:22 → 00:15:08   ✂ clip saved
  Scene 2: 00:41:07 → 00:41:53   ✂ clip saved
```

---

## How it works

Feeding a whole movie to a vision model at once is slow and imprecise. This tool narrows the search hierarchically instead, spending GPU time only where a match is plausible.

### The three-phase pipeline

| Phase | Window | Frames | Purpose |
|---|---|---|---|
| **1 — Coarse** | 2 min (configurable 1–5), 30 s overlap | 64 | Scan the whole video for candidate regions. Tuned for recall — a false positive here is cheap, a miss is not. |
| **2 — Medium** | 30 s, 50 % overlap | 16 | Subdivide surviving regions to pin down which half-minute actually matches. |
| **3 — Fine** | binary search | 12 | Bisect the leading and trailing edges until the boundary is known to ~4 seconds. |

Scenes ending and restarting within 10 seconds of each other are merged into one continuous clip.

### Dual-model classification (default)

Rather than asking one vision model "does this clip match my prompt?", the work is split in two:

1. **Vision captioner** watches the window and writes an objective description of who is present, where they are, and what they are doing. It never sees your search prompt — so it cannot be led toward a false positive.
2. **Text LLM matcher** compares that description against your prompt and returns a **match probability (0–100 %)**. Anything at or above your threshold (default 50 %) counts as a hit.

The payoff is throughput: descriptions are generated once per window, then all of them are classified in a single batched LLM call (micro-batches of 16). Both system prompts are editable and persist to disk — see [Tuning the prompts](#tuning-the-prompts).

`--single-model` skips step 2 and asks the vision model for a direct YES/NO instead.

### Works with any vision-language model

The vision model is loaded generically via `AutoModelForImageTextToText`, and whether it gets
true multi-frame video or a batch of individual frames is decided by inspecting the loaded
**processor class** — not by matching the model name against a hardcoded list.

- If the processor accepts a `video_processor` argument (Qwen2-VL/2.5-VL/3-VL, VideoLlava,
  LLaVA-NeXT-Video, InternVL, …), the whole window is passed through as native video.
- Otherwise (plain LLaVA, LLaVA-NeXT, and other single-image architectures) frames are
  sampled and captioned as a **batch** — up to `IMAGE_ONLY_BATCH_FRAMES` (8 by default) frames
  per window, each captioned independently in one batched `generate()` call, then fused into a
  single temporally-ordered description (`[t=12.3s] ...`) so the matcher downstream sees the
  same one-description-per-window shape regardless of which path produced it.

Batching trades VRAM for wall-clock time: each frame's activations are held concurrently during
the batched forward pass, so memory scales with batch size, while generation time stays close to
1× since encoding parallelizes across the batch (only autoregressive decoding stays sequential
per frame). `IMAGE_ONLY_BATCH_FRAMES` bounds that memory cost independently of how many frames a
given phase asks for — Phase 1 requests 64 frames per window, but an image-only model still only
batches 8 of them.

## Requirements

- **Python 3.10+**
- **NVIDIA GPU strongly recommended.** Developed against an RTX 4080 SUPER with a ~14 GB VRAM budget. With 4-bit NF4 quantization a 7B vision model plus a 7B text model fit together in that envelope. CPU inference works but is impractically slow for long videos.
- **CUDA 12.x** (tested on torch 2.6.0+cu124)
- **FFmpeg** — bundled automatically via `imageio-ffmpeg`, no system install needed

## Install

```bash
git clone <your-repo-url>
cd video-scene-detector

python -m venv .venv
.\.venv\Scripts\Activate.ps1      # Windows PowerShell
# source .venv/bin/activate       # Linux / macOS

pip install -r requirements.txt
```

Install PyTorch matching your CUDA version first if the default wheel doesn't suit your setup — see [pytorch.org](https://pytorch.org/get-started/locally/).

No model ships with the tool or downloads automatically — you choose what to run. You need at least one **vision** model and, for the default dual-model mode, one **text** model. Reasonable starting points are `Qwen/Qwen2.5-VL-7B-Instruct` (vision) and `Qwen/Qwen2.5-7B-Instruct` (text), but any HuggingFace vision-language / causal-LM model works — see [Works with any vision-language model](#works-with-any-vision-language-model).

Get a model onto disk either by:
- passing its HuggingFace ID to `--model` / `--text-model` once — it downloads to `models/` and loads directly from there on subsequent runs, or
- using the web UI's *Download Model from HuggingFace* button, which downloads to `models_cache/hub/`.

Once something is installed, omitting `--model`/`--text-model` auto-selects the first installed model of each type — see [CLI reference](#cli-reference). Expect a large one-time download per model (7B models run roughly 15 GB each). Both `models/` and `models_cache/` are gitignored.

---

## Quick start

### Web client

```bash
python cli.py --web            # or just: python cli.py
```

Open <http://localhost:8000>. Set a port with `--port 9000`.

The interface has two tabs:

- **Live** — point it at a video file or a folder, enter your prompt, and watch phase-by-phase logs stream in over Server-Sent Events with live VRAM readouts. Stop the run at any time.
- **Gallery** — browse and play the clips that were cut, straight from the output folder.

Also available in the UI: an inline model downloader (paste any HuggingFace model ID), editable captioner and matcher system prompts, prompt-quality analysis, and a **single-frame debug tool** that runs one timestamp through the pipeline and shows you the raw caption, the matcher's reasoning, and the timing breakdown. That debug view is the fastest way to work out why a scene was or wasn't matched.

### Command line — one video

```bash
python cli.py \
  --video "D:/Videos/movie.mp4" \
  --prompt "two people shaking hands in an office" \
  --cut-clip \
  --output results.json
```

### Command line — a whole folder

```bash
python cli.py \
  --input-dir "D:/Videos/library" \
  --clip-dir "D:/Videos/matches" \
  --prompt "a red car driving at night"
```

Recurses through the folder for `.mp4`, `.mkv`, `.avi`, `.mov`, `.webm`, and `.m4v`. Batch mode always cuts clips and always writes a consolidated `batch_detection_report.json` to the output folder.

### Search only part of a video

```bash
python cli.py --video movie.mp4 --prompt "..." --start 00:20:00 --end 00:35:00
```

---

## CLI reference

Neither `--model` nor `--text-model` has a hardcoded default. Leave them out and the first installed
vision/text model in `models_cache/hub/` is used instead (alphabetically, by whatever's classified as
that type — see [Works with any vision-language model](#works-with-any-vision-language-model)). If
nothing of that type is installed yet, the command exits with an error suggesting where to get one.

| Flag | Default | Description |
|---|---|---|
| `--video` | — | Path to a single video file |
| `--input-dir` | — | Folder to scan recursively (batch mode) |
| `--prompt` | — | Scene description. Required for detection. |
| `--model` | *(none — see below)* | Vision model — HF ID, local folder, or `.safetensors` file |
| `--text-model` | *(none — see below)* | Text LLM used by the matcher |
| `--quantization` | `4bit` | `4bit` (NF4), `none`, or `bfloat16`. Only `4bit` quantizes; the other two load at full precision. |
| `--dual-model` / `--single-model` | dual | Two-stage caption→match, or direct YES/NO from the vision model |
| `--similarity-threshold`, `--threshold` | `50.0` | Match probability % required to count as a hit |
| `--coarse-window-min` | `2.0` | Phase 1 window length in minutes (clamped 1.0–5.0) |
| `--coarse-frames` | `64` | Frames sampled per coarse window (clamped 8–64) |
| `--start-time`, `--start` | start of video | Restrict search — `00:01:30` or `90.0` |
| `--end-time`, `--end` | end of video | Restrict search |
| `--cut-clip` | off | Extract matched scenes as MP4 (single-video mode; batch always cuts) |
| `--clip-dir` | `./extracted_clips` | Where clips and logs are written |
| `--output` | — | Write results to a JSON file (single-video mode) |
| `--sage-attn` | off | SageAttention 2 INT8 attention kernels (requires `sageattention`) |
| `--smart-motion` | off | Reserved — see [Known limitations](#known-limitations) |
| `--web`, `--gui` | — | Launch the web client instead of detecting |
| `--port` | `8000` | Web client port |

Running `python cli.py` with no `--video` and no `--input-dir` launches the web client.

---

## Output

Clips are named `{video}_scene{n}_{start}s-{end}s.mp4` and cut with an FFmpeg stream copy — lossless, and effectively instant since nothing is re-encoded.

Every processed video also gets a `{video}_summary.log` next to its clips, opening with a timing breakdown:

```
Filename: movie.mp4
Total Time: 4m 12.3s (252.3s)
Total GPU Time: 3m 48.1s (228.1s)
Active Frame Extraction Time: 11.2s
Vision Inference Time: 2m 51.0s (171.0s)
LLM Inference Time: 57.1s
Clips Extracted: 2
Clip Names: movie_scene1_862s-908s.mp4, movie_scene2_2467s-2513s.mp4
```

followed by the full phase-by-phase log including every window verdict and the matcher's reasoning. "Active Frame Extraction Time" counts only decode time that wasn't hidden behind GPU work — frame extraction runs on a background thread and prefetches the next window while the current one is on the GPU, so most of it costs nothing.

`SceneMatch` records serialize to JSON with `video_path`, `prompt`, `found`, `start_time`, `end_time`, `start_seconds`, `end_seconds`, `explanation`, `clip_path`, and `scene_index`.

---

## Tuning the prompts

Two system prompts drive the whole pipeline, both editable in the web UI and persisted as plain text:

| File | Used by | Role |
|---|---|---|
| `prompts/captioner_prompt.txt` | Vision model | What to describe and how to structure it |
| `prompts/matcher_prompt.txt` | Text LLM | How to score similarity; must emit `PROBABILITY: <0-100>%` |

The shipped captioner prompt asks for two labelled sections — people and spatial positions, then actions and movements — because the matcher scores much more reliably against consistently structured descriptions. The matcher prompt asks for a dual evaluation (phrasing overlap, then visual/semantic alignment) and a numeric probability.

If you rewrite the matcher prompt, keep the `PROBABILITY: <n>%` line. Parsing falls back to any `<n>%` in the response, then to a bare YES (scored 85 %), then to 0 — so dropping the format degrades matching quietly rather than loudly.

**Writing good search prompts:** be visually concrete. Name body positions, spatial relationships between subjects, and camera perspective. `"a person standing behind another, both facing the camera"` outperforms `"an interaction"` by a wide margin. The analyzer will flag prompts that lack these anchors.

**Tuning the threshold:** too many false positives, raise it toward 65–75; missing scenes you know are there, lower it toward 35–40 and check the reasoning in the log to see what score the scene actually got.

---

## HTTP API

The web client is a thin front-end over a documented FastAPI backend — usable directly if you want to script it.

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/start-batch` | Start detection on a file or folder |
| `POST` | `/api/stop-batch` | Cancel the running job |
| `GET` | `/api/progress-stream` | SSE stream of progress, logs, and VRAM stats |
| `POST` | `/api/analyze-prompt` | Prompt quality score and improvement suggestions |
| `POST` | `/api/check-path` | Validate a path, count videos found |
| `POST` | `/api/debug-frame` | Run one timestamp through the pipeline |
| `GET` | `/api/clips` | List cut clips in the output folder |
| `GET` | `/api/stream-clip`, `/api/stream-video` | Serve video to the player |
| `GET` / `POST` | `/api/prompts` | Read / update the two system prompts |
| `GET` | `/api/models` | List locally installed models |
| `POST` | `/api/models/download` | Download a model from HuggingFace |

One job runs at a time; starting a second returns HTTP 400.

> **Note on binding:** the server listens on `0.0.0.0` with permissive CORS, and `/api/stream-video` will serve **any** file path the host can read. It is built for single-user localhost use — don't expose it to an untrusted network as-is.

---

## Python API

```python
from detector import VideoSceneDetector

detector = VideoSceneDetector(
    model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct",  # or omit to use the first installed vision model
    quantization="4bit",
    dual_model=True,
    similarity_threshold=50.0,
    coarse_window_minutes=2.0,
)

scenes = detector.detect_scenes(
    video_path="movie.mp4",
    prompt="two people shaking hands in an office",
    cut_clips=True,
    clip_output_dir="./extracted_clips",
)

for s in scenes:
    print(f"{s.start_time} → {s.end_time}  ({s.clip_path})")
```

Loaded models are cached globally by their configuration (model, quantization, device, dtype, attention backend). Repeated calls reuse the weights already in VRAM; changing any of those settings unloads the old model and frees VRAM before loading the new one. `BatchProcessor` wraps this for folder runs and accepts a `progress_callback(progress, log_line)`.

---

## Project layout

```
cli.py                      CLI entrypoint; also launches the web client
app.py                      Legacy Gradio UI — superseded, see limitations
detector/
  scene_detector.py         Three-phase pipeline, boundary search, export
  qwen_model.py             Vision model loading + inference; fast OpenCV frame reader
  text_llm.py               Text LLM matcher, batched classification
  batch_processor.py        Folder orchestration and progress reporting
  prompt_analyzer.py        Prompt quality scoring and suggestions
  prompts.py                Editable system prompts, persisted to prompts/
  video_utils.py            Metadata, timestamps, clip cutting, model resolution
  sage_patcher.py           Optional SageAttention 2 SDPA monkey-patch
web/
  app.py                    FastAPI server
  static/                   Front-end (index.html, app.js, style.css)
prompts/                    Editable captioner + matcher prompts
extracted_clips/            Default output (gitignored)
models/, models_cache/      Downloaded weights (gitignored)
```

---

## Performance notes

Speed comes from a handful of deliberate choices:

- **Hierarchical search.** Only regions that survive Phase 1 get expensive fine-grained analysis.
- **Prefetched frame extraction.** Frames for window *n+1* decode on a worker thread while window *n* runs on the GPU, so decode time is mostly hidden. The log reports how much wasn't.
- **Batched classification.** All window descriptions go to the text LLM in micro-batches of 16 rather than one at a time.
- **Short-circuit evaluation.** A multi-condition prompt stops at the first failing condition.
- **4-bit NF4 quantization** keeps two 7B models resident together in ~14 GB.
- **TF32 + bfloat16 tensor cores**, FlashAttention-2 when installed (auto-detected), PyTorch SDPA otherwise, or SageAttention 2 INT8 with `--sage-attn`.
- **Custom OpenCV/FFmpeg video reader** registered into `qwen_vl_utils`, which seeks across a 45-minute file in a couple of seconds.
- **Stream-copy clip cutting** — no re-encode.

Runtime scales with video length and how much of it survives Phase 1. A prompt that matches nothing finishes in roughly Phase 1 time alone.

---

## Known limitations

Documented honestly so you know what you're getting:

- **`--smart-motion` is a no-op.** The flag is plumbed through every layer but `filter_motion_keyframes()` is never actually called.
- **`--coarse-frames` only affects video-native models.** Image-only models always batch `IMAGE_ONLY_BATCH_FRAMES` (8) frames per window regardless of what a phase requests, to keep VRAM bounded — see [Works with any vision-language model](#works-with-any-vision-language-model).
- **`app.py` (Gradio) is legacy.** The web client under `web/` replaced it. Its export path calls `export_json` with a single `SceneMatch` where a list is expected, so downloads from that UI will fail. Use `cli.py --web`.
- **Boundary precision is ~4 seconds**, set by `FINE_PRECISION_SEC`. Lower it for tighter edges at the cost of more binary-search steps.
- **Windows-first.** Paths and console encoding are handled with Windows in mind; Linux and macOS should work but are less exercised.

---

## Content and licensing

You are responsible for having the rights to the video you process, and for complying with the license of whichever vision and text models you configure — check each model's card on HuggingFace before use.

---

## License

GPLv3 — see [LICENSE](LICENSE).
