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
| `frame_source.py` | `FrameSource` abstraction + a robust capture chain. `ReachyCameraFrameSource` (live daemon camera, **working** against localhost:8000), `WebcamFrameSource` (Mac webcam fallback), `SyntheticFrameSource` (last-resort image), `FileFrameSource` (file), `FallbackFrameSource` / `best_available_frame_source()` (tries them in order). `to_pil(arr, bgr=...)` normalizes PIL / numpy / path / URL / bytes → RGB PIL; `bgr_to_rgb()` flips channel order. |
| `responder.py` | **Vision turn handler.** `VisionResponder.maybe_answer(transcript)` (lazy-loaded VLM + frame source), `vision_reply(...)`, `is_visual_query(...)`. |
| `__init__.py` | Public exports. |
| `../../../scripts/smoke_vision.py` | End-to-end smoke test (generate a test image, describe it, print the answer + one rough timing). |
| `../../../scripts/smoke_vision_turn.py` | Vision *turn* smoke test (capture a real frame, run `maybe_answer` on a vision + a non-vision transcript). |

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

## Live Reachy camera (WORKING — verified against the live daemon)

The daemon owns the camera; on the same machine the SDK's **LOCAL** backend
(`GStreamerCamera`) reads the latest frame from the daemon's GStreamer IPC
endpoint. There is **no plain REST endpoint** that returns a frame — the SDK
media manager is the supported path (the daemon's REST API only exposes
`/api/camera/specs` and `/api/media/status`).

```python
from reachy_mini import ReachyMini
from reachy_chat.vision import ReachyCameraFrameSource, describe

mini = ReachyMini(media_backend="default")  # auto -> LOCAL on-device
src  = ReachyCameraFrameSource(mini=mini)   # or frame_provider=camera_worker
frame = src.get_frame()                      # RGB PIL image (BGR->RGB applied)
print(describe(frame, "What do you see?"))
```

**Verified:** against the running daemon (`localhost:8000`, `--mockup-sim`,
`no_media=false`) this returns a real `1280x720` frame. `ReachyCameraFrameSource`
defaults to lazily creating its own `ReachyMini(media_backend="default")` if you
pass neither `mini=` nor `frame_provider=`.

**⚠️ Colour order:** `mini.media.get_frame()` returns a **BGR** `uint8` numpy
array (so does OpenCV) — *not* RGB. `ReachyCameraFrameSource` /
`WebcamFrameSource` flip it to RGB via `to_pil(arr, bgr=True)`. If you read
frames yourself, convert with `bgr_to_rgb()` before describing them, or the
model sees swapped colours.

The conversation app polls the camera ~every 40 ms on a worker thread and
exposes `get_latest_frame()`; pass that worker as `frame_provider=` to avoid
double-reading the pipeline. This class does not own the robot connection unless
it had to create one — the main app should pass `mini=`.

### Robust capture with fallbacks

`best_available_frame_source()` returns a `FallbackFrameSource` that tries, in
order: **Reachy daemon camera → Mac webcam (`cv2.VideoCapture(0)`) → synthetic
image**, so `get_frame()` always returns a valid RGB PIL image. Inspect
`.last_used.name` afterwards to see which path produced the frame.

```python
from reachy_chat.vision import best_available_frame_source

src = best_available_frame_source(mini=mini)  # mini optional
frame = src.get_frame()
print("frame from:", src.last_used.name)       # e.g. "reachy:camera"
```

Against the live mockup-sim daemon the **`reachy:camera`** path is the one that
works (the daemon proxies the host webcam through its simulation camera).

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

## Vision turn (answering "what do you see?" in the conversation loop)

`responder.py` turns a user utterance into a spoken answer about what Reachy
sees — but only when the utterance is actually a *visual* question. The main app
calls this **before** the text LLM on each turn.

### API

```python
class VisionResponder:
    def __init__(self, frame_source=None, *, mini=None, frame_provider=None,
                 model_name=None, max_tokens=64): ...
    def maybe_answer(self, transcript: str) -> str | None: ...
    @property
    def vlm_loaded(self) -> bool: ...      # True once a visual query has loaded the VLM

# Functional equivalents:
def is_visual_query(transcript: str) -> bool: ...
def vision_reply(transcript, *, vlm=None, frame_source=None,
                 max_tokens=64) -> str | None: ...
```

`maybe_answer(transcript)`:
- returns **`None`** if the transcript is not a visual query (heuristics:
  "what do you see", "look at…", "describe…", "what is this/that", "in front of
  you", "can you see", "what am I holding", "what colour…", "how many… do you
  see", "read this", etc.) → let your text LLM handle the turn;
- otherwise captures one frame, runs the **Gemma 4 E4B** VLM, and returns a
  **short spoken-style answer** (1–2 sentences, markdown/asterisks stripped).

**Lazy load:** the ~6 GB VLM loads on the *first* visual query only. Frame source
defaults to `best_available_frame_source()` (Reachy camera → webcam → synthetic).
`maybe_answer` never raises on capture/inference failure — it returns a brief
spoken apology so the turn loop stays robust.

### Wiring it into the turn loop

Construct one responder per session, pass the live `ReachyMini` so it shares the
app's connection, and call it just before the text LLM:

```python
from reachy_chat.vision import VisionResponder

responder = VisionResponder(mini=mini)        # or frame_provider=camera_worker

# ...in the per-turn handler, after you have the user transcript:
spoken = responder.maybe_answer(transcript)
if spoken is not None:
    speak(spoken)                              # vision answered this turn
else:
    speak(text_llm(transcript))               # normal text path
```

The first visual query pays a one-time VLM load; warm it at startup with
`responder._get_vlm()` (or just accept the lazy load) if you prefer.

### Verified example

Against the live daemon, "What do you see?" produced (from the real camera
frame):

> *"I see a bright ceiling light over a dark dresser in a room with light-colored
> walls and blinds. There is a small potted plant sitting on top of the dresser."*

…and "What time is it in Paris right now?" returned `None` (routed to the text
LLM). Run it yourself:

```bash
.venv/bin/python scripts/smoke_vision_turn.py
```

### Preliminary vision-turn timing (rough — NOT a benchmark)

Single shot via the live `reachy:camera` source (`1280x720` frame), GPU
contended by another process:

- Frame capture (first read, includes pipeline warm-up): **~1.6 s** the first
  time; subsequent reads are much cheaper (the SDK keeps the pipeline open).
- VLM load (once per session, lazy): **~6.5 s**.
- Inference (frame prefill + ≤64 decoded tokens): **~1.6 s**.

⚠️ Preliminary, single shot, contended GPU. The research target is well under
100 ms of *image prefill* on an otherwise-warm pipeline; the main thread does
careful serial latency measurement separately. To stay near the latency budget,
prefer passing `frame_provider=` (a running camera worker) so capture is a cheap
buffer read, keep `max_tokens` small (64 here), and warm the VLM at startup.

## Notes / gotchas

- Importing `mlx` requires a real Metal device; it fails under the build sandbox
  with `No Metal device available`. Run with the sandbox disabled.
- mlx-vlm pulls in the audio feature extractor; you may see a benign
  `mel filter ... all zero values` warning from `transformers` on load. The
  vision path is unaffected.
- The model is already in the HF cache; set `HF_HUB_OFFLINE=1` to avoid any
  network access.
