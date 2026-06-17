# ReachyChatOffline

A fully-local, low-latency, English voice-to-voice conversational app for the **Reachy Mini**
robot, on Apple Silicon (MLX/Metal). Streaming **cascade-with-overlap**: VAD/endpoint → STT → LLM
→ TTS, with semantic turn-taking, barge-in, wake-word, and embodied "look-at-speaker" via the
robot's mic direction-of-arrival.

> **Status:** components built & benchmarked; integrated app runs end-to-end. **Voice-to-voice
> first-audio: p50 271 ms (compute pipeline, under the 500 ms budget).** Full path incl. endpoint
> detection ~590 ms today, being optimized toward ~350–450 ms. See [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md).

## The stack (all local, MLX)

| Stage | Choice | Warm latency |
|---|---|---|
| VAD + endpoint | Silero VAD v6 + Smart-Turn v3 (eagerness-gated) | ~100–200 ms (tunable) |
| STT | parakeet-tdt-0.6b (streaming) | finalize ~120 ms |
| LLM | Gemma 4 **E2B** (latency) / **E4B** (quality+vision) QAT-4bit, thinking off | TTFT 29 / 53 ms |
| TTS | Kokoro-82M (latency-first); emotive options under eval | TTFA ~83 ms |
| Embodiment | DOA look-at-speaker + tool-call gestures (decoupled, off latency path) | — |

## Layout

```
plan.md                     # living plan & decisions
docs/architecture.md        # stack rationale + measured budget
docs/PERFORMANCE.md         # consolidated performance report
docs/FUTURE_WORK.md         # deferred experiments (emotive TTS, etc.)
docs/research/01..06-*.md   # grounding research (LLM, STT, TTS, VAD, e2e, semantic-VAD)
docs/experiments/latency-experiments.md   # every measurement, incl. overshoots
src/reachy_chat/
  pipeline/   # engines (STT/LLM/TTS) + ConversationApp orchestrator + audio I/O
  audio/      # Silero VAD, Smart-Turn endpointer, wake-word, interaction state machine
  robot/      # Reachy daemon REST client, motion queue, look-at-speaker, tool-calls
  vision/     # Gemma 4 E4B webcam-frame path (mlx-vlm)
  tts/        # Kokoro istftnet bug fix (#786) + emotive engine loaders
benchmarks/   # harness + per-component + end-to-end latency benchmarks
audio_samples/ # test prompt WAVs + generated TTS/response samples
```

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python numpy soundfile sounddevice mlx mlx-lm parakeet-mlx mlx-audio "misaki[en]"
# models download from HF on first use (hf CLI logged in)
```

## Run

```bash
# offline (file-driven; measures full stop-talking → first-audio incl. endpointing)
PYTHONPATH=src .venv/bin/python -m reachy_chat.pipeline.app --mode wav --wav audio_samples/prompts/p3_vision.wav

# live mic (needs a microphone); --wake for "Hey Reachy"
PYTHONPATH=src .venv/bin/python -m reachy_chat.pipeline.app --mode live [--wake]

# benchmarks
.venv/bin/python benchmarks/bench_e2e.py --llm mlx-community/gemma-4-E2B-it-qat-4bit --n 30
.venv/bin/python benchmarks/bench_tts.py kokoro
```

Requires the Reachy Mini daemon/simulator at `http://localhost:8000` for robot control (the
voice pipeline itself runs without it). Built & verified against the Reachy Mini SDK (Python 3.12).
