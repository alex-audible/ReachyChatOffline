# reachy_chat.vision — webcam frame → Gemma 4 multimodal

The **vision path** for ReachyChatOffline: feed a single camera frame plus a
text question into the Gemma 4 E4B QAT multimodal model and get back text, so
"what Reachy sees" can enter the conversation.

- **Model:** `mlx-community/gemma-4-E4B-it-qat-4bit` (QAT int4, ~3 GB resident,
  image + text in, text out), the same weights used for the reasoning core. See
  `docs/research/01-llm-audio-native.md`.
- **Runtime:** [`mlx-vlm`](https://github.com/Blaizzy/mlx-vlm) (installed
  `0.6.3`) on Apple Silicon / Metal.

## Files

| File | Purpose |
|---|---|
| `vlm.py` | Loads the model and runs image+text → text. `VLM` class + `describe()` / `stream_describe()` + a one-shot `describe(image, prompt)` convenience. |
| `frame_source.py` | `FrameSource` abstraction: `FileFrameSource` (working, used by the smoke test) and `ReachyCameraFrameSource` (documented hook for the live robot camera). `to_pil()` normalizes PIL / numpy / path / URL / bytes → RGB PIL. |
| `__init__.py` | Public exports. |
| `../../../scripts/smoke_vision.py` | End-to-end smoke test (generate a test image, describe it, print the answer + one rough timing). |

## Quick start

```python
from reachy_chat.vision import VLM

vlm = VLM()  # loads mlx-community/gemma-4-E4B-it-qat-4bit (slow, do once)
answer = vlm.describe("photo.jpg", "What do you see?")
print(answer)
```

One-shot convenience (caches a process-wide model):

```python
from reachy_chat.vision import describe
print(describe(frame, "What is the person doing?"))
```

`image` accepts a `PIL.Image`, a `numpy` RGB uint8 array (a camera frame), a
file path / URL / data-URI string, or raw encoded `bytes`. PIL images are passed
to mlx-vlm in memory (no temp-file round-trip).

Streaming (for later TTS hand-off):

```python
for chunk in vlm.stream_describe(frame, "Describe the scene."):
    print(chunk, end="", flush=True)
```

## The mlx-vlm API that works for Gemma 4 (0.6.x)

```python
from mlx_vlm import load, generate, stream_generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("mlx-community/gemma-4-E4B-it-qat-4bit")
prompt = apply_chat_template(processor, model.config, "What do you see?", num_images=1)

# image may be a PIL.Image, a path/URL string, or a list of those.
result = generate(model, processor, prompt, image=pil_image, max_tokens=120, temperature=0.0)
text = result.text                      # GenerationResult.text

# or stream:
for chunk in stream_generate(model, processor, prompt, image=pil_image):
    text += chunk.text
```

Notes:
- `load()` returns `(model, processor)`; pass `model.config` (not the dict) to
  `apply_chat_template`.
- `apply_chat_template(..., num_images=1)` inserts the image placeholder.
- mlx-vlm's `process_image` accepts a `PIL.Image` directly, so frames don't need
  to be written to disk.

## Live Reachy camera (integration hook — not exercised yet)

The daemon owns the camera; a local client reads the latest RGB frame from the
SDK (`.reference/reachy_mini/docs/source/SDK/media-architecture.md`,
`.reference/reachy_mini_conversation_app/.../camera_worker.py`):

```python
from reachy_mini import ReachyMini
from reachy_chat.vision import ReachyCameraFrameSource, describe

mini = ReachyMini()                       # auto LOCAL backend, daemon @ localhost
src  = ReachyCameraFrameSource(mini=mini) # or frame_provider=camera_worker
frame = src.get_frame()                   # RGB PIL image
print(describe(frame, "What do you see?"))
```

`mini.media.get_frame()` returns an RGB `uint8` numpy array (or `None` while the
pipeline warms up). The conversation app polls it ~every 40 ms on a worker
thread and exposes `get_latest_frame()`; `ReachyCameraFrameSource` accepts
either that worker (`frame_provider=`) or a `ReachyMini` (`mini=`). This class
does not create or own the robot connection — the main app does.

## Smoke test

```bash
.venv/bin/python scripts/smoke_vision.py            # synthetic red-circle image
.venv/bin/python scripts/smoke_vision.py my.jpg     # your own image
```

Verified end-to-end: on the synthetic image the model answered
*"This image is a simple, solid red circle centered on a white background."* —
correct.

## Preliminary timing (rough, GPU was contended — NOT a benchmark)

A single shot on the synthetic 512px image:

- **Image-prefill TTFT ≈ ~0.8 s** (time to first token, includes loading/encoding
  the image + prompt prefill).
- Decode ≈ ~46 tok/s (research baseline ~57 tok/s on M4 Pro).
- Total image+text call (≤120 tokens) ≈ 1.6 s.

⚠️ **Preliminary only.** Another process was actively benchmarking on the Metal
GPU during this measurement, so these numbers are inflated and not
representative. Careful, serial latency measurement is done separately by the
main thread. The research note's expectation is that one 768px frame adds well
under 100 ms of *image prefill* to an otherwise-warm pipeline.

## Notes / gotchas

- Importing `mlx` requires a real Metal device; it fails under the build sandbox
  with `No Metal device available`. Run with the sandbox disabled.
- mlx-vlm pulls in the audio feature extractor; you may see a benign
  `mel filter ... all zero values` warning from `transformers` on load. The
  vision path is unaffected.
- The model is already in the HF cache; set `HF_HUB_OFFLINE=1` to avoid any
  network access.
