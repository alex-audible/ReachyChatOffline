# Reachy Mini Offline Voice Assistant — Performance Report

**Date:** 2026-06-17 · **Hardware:** Apple M3 Max (30-core GPU, 96 GB), macOS 15.7.5 ·
**Target deploy:** M3 / 24 GB (working set ≤ ~20 GB) · **Runtime:** MLX/Metal, Python 3.12.

Full methodology + every experiment (incl. overshoots) in
[`docs/experiments/latency-experiments.md`](experiments/latency-experiments.md);
architecture rationale in [`docs/architecture.md`](architecture.md);
grounding research in [`docs/research/`](research/).

---

## 1. Executive summary

A fully-local, English, streaming **cascade-with-overlap** voice assistant for Reachy Mini.

- **Compute pipeline (detected end-of-speech → first audio): p50 271 ms (E2B) / 331 ms (E4B)** —
  comfortably **under the 500 ms budget**. This is the responsiveness once the turn end is known.
- **Full pipeline (stop-talking → first audio, incl. endpoint detection): p50 ~590 ms today** —
  the endpoint silence window adds ~300 ms. **Being optimized** toward a realistic **~350–450 ms**
  floor via eagerness-gated Smart-Turn + speculative LLM start (semantic-VAD research).
- All models run locally on Metal within ~7.5–12.5 GB (well under 20 GB).
- The latency metric is measured the right way: **wall-clock time-to-first-audio from t0 =
  end of user speech**, warm/in-conversation — not the additive sum of components (which overstates).

## 2. Component results (warm / in-conversation, M3 Max)

| Stage | Choice | Measured | Memory | Notes |
|---|---|---|---|---|
| STT | parakeet-tdt-0.6b (MLX), streaming ctx(256,64), 320 ms chunk | finalize **~116–130 ms**, RTF 0.20 | ~1–2 GB | near-perfect English; smaller chunk = lower finalize |
| LLM | Gemma 4 **E2B** QAT-4bit (latency) | **TTFT 29 ms (warm)**, 91 tok/s | 3.6 GB | persona cached; `enable_thinking=False` |
| LLM | Gemma 4 **E4B** QAT-4bit (quality+vision) | **TTFT 53 ms (warm)**, 50 tok/s | 6.0 GB | mixed 4/8-bit MLP recipe → ~50 tok/s |
| TTS | **Kokoro-82M** (latency-first) | **TTFA 79–83 ms**, RTF 0.04 | 1.4 GB | fast; flat affect |
| Endpoint | Silero VAD v6 + Smart-Turn v3 | ~160 ms (isolated); ~300 ms in live pipeline | <0.5 GB | dominant latency; being tuned |
| Vision | Gemma 4 E4B via mlx-vlm | image-prefill ~0.8 s (contended; clean TBD) | shared w/ LLM | webcam frame → conversation (≤200 ms budget) |
| Embodiment | DOA look-at-speaker (REST `/api/state/doa`) | decoupled, off latency path | — | verified vs live simulator (81 emotion moves) |

## 3. End-to-end latency (full variance spreads)

**Compute pipeline** (from detected endpoint; EXP-E2E-1, n=30, Kokoro TTS):

| Pipeline | min | **p50** | p90 | p95 | p99 | max | std |
|---|---:|---:|---:|---:|---:|---:|---:|
| E2B + Kokoro | 240 | **271** | 377 | 387 | 407 | 415 | 49 |
| E4B + Kokoro | 313 | **331** | 478 | 497 | 524 | 534 | 62 |

**Full pipeline** incl. endpoint detection (EXP-E2E-2, E2B+Kokoro, n=15): min 475 · **p50 590** ·
p90 703 · p95 712 · max 717 · std 84 ms. Endpoint detection ≈ the ~300 ms gap (silence window +
Smart-Turn commit). Variance is dominated by the LLM first-clause stage + environmental load
(Reachy sim daemon ~15% CPU, present in production too).

## 4. Memory budget (resident, ≤20 GB target on 24 GB M3)

| Config | STT | LLM | TTS | VAD/turn | Total |
|---|---:|---:|---:|---:|---:|
| Speed (E2B + Kokoro) | ~2 | 3.6 | 1.4 | <0.5 | **~7.5 GB** ✅ |
| Quality (E4B + emotive 3B) | ~2 | 6.0 | ~4 | <0.5 | **~12.5 GB** ✅ |

## 5. Key findings & fixes (the non-obvious ones)

- **Gemma 4 thinking mode is ON by default** and emits a long `<think>` block before answering —
  fatal for latency. Fix: `apply_chat_template(enable_thinking=False)` + a no-stage-directions persona.
- **Warm vs cold:** naive benchmarking (fresh cache each call) showed E4B TTFT 275 ms; the real
  in-conversation path (persona prefix cached) is 53 ms. Always measure warm/primed.
- **Kokoro istftnet crash** (`broadcast_shapes`) is upstream mlx-audio bug #786 (`math.ceil` float
  rounding); patched in `src/reachy_chat/tts/kokoro_fix.py` (chose patch over downgrade per #784).
- **Emotive TTS is hard locally on M3:** Kokoro is fast but flat; **Orpheus-3B is unusable
  (RTF 2.23)**; **Sesame CSM is gated** (needs access to `sesame/csm-1b`); **Chatterbox-Turbo q4**
  needs the `chatterbox_turbo` backend + a reference voice (integration in progress).
- **Endpoint detection is the latency bottleneck**, not compute. Realistic floor ~135–165 ms; a
  150–250 ms window is partly *desirable* (human turn gaps cluster ~200 ms). Path to ≤500 ms full:
  eagerness-gated Smart-Turn + speculative LLM start on stabilized partials.

## 6. Status & next steps

**Done:** research (6 streams); shared benchmark harness; component benchmarks; cascade pipeline +
reusable engines (`src/reachy_chat/pipeline/`); integrated `ConversationApp` (offline driver
validated, full-pipeline measured); robot integration (verified vs simulator); VAD/turn-taking/
wake-word/barge-in module; vision path; thorough docs.

**In flight:** endpoint-latency optimization (eagerness + speculative start → target full p50 ≤500 ms);
Chatterbox-Turbo emotive TTS integration; clean vision latency measurement.

**Remaining:** live mic run + barge-in on hardware; on-simulator end-to-end; final tuning.

## 7. How to run

```bash
# offline (file-driven, measures full stop-talking → first-audio incl. endpointing)
PYTHONPATH=src .venv/bin/python -m reachy_chat.pipeline.app --mode wav --wav audio_samples/prompts/p3_vision.wav
# live mic (needs a microphone)
PYTHONPATH=src .venv/bin/python -m reachy_chat.pipeline.app --mode live           # always-on
PYTHONPATH=src .venv/bin/python -m reachy_chat.pipeline.app --mode live --wake    # "Hey Reachy"
# component benchmarks
.venv/bin/python benchmarks/bench_e2e.py --llm mlx-community/gemma-4-E2B-it-qat-4bit --n 30
.venv/bin/python benchmarks/bench_tts.py kokoro
```
