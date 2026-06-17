# Architecture & Component Decisions

Synthesis of the five research reports (`docs/research/`) + measured benchmarks
(`docs/experiments/latency-experiments.md`). All components are **local**, English, and run on
**MLX/Metal** (PyTorch/MPS only where no MLX path exists). Hardware target: **M3 / 24 GB**
(working set ≤ ~20 GB); dev on M3 Max / 96 GB.

## Recommended stack

| Stage | Choice | Repo / package | Measured (warm, M3 Max) | Mem | Notes |
|---|---|---|---|---|---|
| Audio I/O | `sounddevice` (dev) · Reachy daemon **LOCAL** GStreamer backend (robot) | — | — | — | 20–40 ms blocks; barge-in flushes output |
| VAD | **Silero VAD v6** | `silero-vad` 6.x + onnxruntime | <1 ms/frame (research) | ~0 | strict 512-sample @16 kHz frames |
| Endpointing | **Smart-Turn v3** (semantic) + silence window | `pipecat-ai/smart-turn-v3` | ~15 ms model (research) | small | dominant tunable = silence window |
| Wake word | **livekit-wakeword** ("Hey Reachy") | `livekit-wakeword` 0.2.x | ~few ms (research) | small | needs Py≥3.11 (we're 3.12 ✓) |
| STT | **Parakeet-TDT-0.6B** (MLX), streaming | `mlx-community/parakeet-tdt-0.6b-v2` | 165–288 ms batch; streaming finalize TBD | ~1–2 GB | best English WER 1.69%; Nemotron streaming = alt |
| LLM | **Gemma 4 E4B QAT** (quality+vision) / **E2B** (speed) | `mlx-community/gemma-4-E4B-it-qat-4bit` / `…E2B…` | **TTFT 53 / 29 ms**, 50 / 91 tok/s | 6.0 / 3.6 GB | persona prefix cached warm; 128K ctx; Apache-2.0 |
| Vision | same **Gemma 4 E4B** via mlx-vlm | `mlx-community/gemma-4-E4B-it-qat-4bit` | ~+100 ms/frame (research) | shared w/ LLM | webcam frame → conversation (600 ms budget) |
| TTS | **Kokoro-82M** (latency-first) / **Orpheus-3B** or **Chatterbox-Turbo** (emotive) | `mlx-community/Kokoro-82M-bf16` / `…orpheus-3b…` / `…Chatterbox-TTS-fp16` | Kokoro **TTFA 83 ms, RTF 0.04** | 1.4 GB / ~3–6 GB | emotive picks benchmarking now |
| Embodiment | **DOA look-at-speaker** + presence cues | Reachy `/api/state/doa` + face track | decoupled loop | — | off the latency path |

## Verdict: cascade-with-overlap (not full-duplex)

Confirmed by research + measurement. Full-duplex Moshi has lower raw latency but a sealed weak
brain, fixed non-emotive voice, no vision, no swappable TTS. The streaming cascade meets the
budget while keeping full control of the emotive TTS and vision — and fits ≤20 GB. `moshi_mlx`
kept as a documented Plan-B "ultra-low-latency mode."

## Projected voice-to-voice budget (warm / in-conversation)

t=0 = end of user speech. STT runs **during** speech (streaming), so only the tail finalizes at t0.
TTS starts on the **first speakable clause**; later clauses synthesize while audio plays.

| Stage | E4B-quality | E2B-fast | Notes |
|---|---:|---:|---|
| Endpoint confirm (silence + Smart-Turn) | 180 | 130 | dominant tunable; aggressive = lower but more cut-offs |
| STT finalize (tail only) | 90 | 90 | most already transcribed; EXP-STT-2 to confirm |
| LLM TTFT (warm) | 53 | 29 | measured |
| LLM → first speakable clause | ~120 | ~70 | decode of first ~6–10 tokens; **to measure (EXP-E2E-1)** |
| TTS time-to-first-audio | 83 | 83 | Kokoro measured; emotive engines TBD |
| Output buffering | 30 | 30 | |
| **Total to first audio** | **~556** | **~432** | E2B fits; E4B tight → tune endpoint + clause trigger |

**MEASURED (EXP-E2E-1, N=30, warm, Kokoro TTS):** end-to-end first-audio **p50 = 271 ms (E2B) /
331 ms (E4B)**, p95 = 387 / 497 ms — **both under 500 ms**. The measured wall-clock is well below
the additive sum above because STT runs during speech and TTS overlaps later LLM tokens (the whole
point of streaming). Stage p50 (E2B/E4B): STT 125/128 · LLM TTFT+clause 75/140 · TTS 65/67.
Optimizations applied: STT streaming chunk 320 ms, first-clause cap (~5 words), Gemma
`enable_thinking=False`, no-asterisk persona. **E2B = latency-first default; E4B = quality + vision
option** (headroom for the ≤600 ms — preferably ≤100 ms but ≤200 ms acceptable — vision budget).
Emotive TTS (Orpheus/Chatterbox) has higher TTFA than Kokoro — being quantified (EXP-TTS-2); any
overshoot logged with its quality trade-off per the spec. Full variance spreads in the experiments log.

## Memory budget (resident, simultaneous) — target ≤20 GB on 24 GB M3

| Config | STT | LLM | TTS | VAD/turn | Total |
|---|---:|---:|---:|---:|---:|
| Speed (E2B + Kokoro) | ~2 | 3.6 | 1.4 | <0.5 | **~7.5 GB** |
| Quality (E4B + Orpheus-3B) | ~2 | 6.0 | ~4 | <0.5 | **~12.5 GB** |

Both fit comfortably under 20 GB, leaving headroom for KV cache growth + vision frames.

## Interaction modes
- **Always-on open mic:** Silero VAD gates audio → Smart-Turn endpoints → pipeline. Barge-in via
  Reachy XVF3800 hardware AEC (robot) / pyaec (Mac dev) + output-buffer flush + cooperative cancel.
- **Wake-word:** idle until "Hey Reachy" (livekit-wakeword), then same pipeline; snap gaze to wake
  direction (DOA).

## Embodiment (decoupled control loop, off latency path)
DOA (`/api/state/doa`) → look-at-speaker (body_yaw + head yaw, minjerk, dead-zoned); fuse with
face tracking; active-listening nods; "thinking" gaze during LLM latency; emotion-linked antennas.

## Engineering rules (from project skills + research)
- All MLX inference in `asyncio.to_thread` to avoid event-loop starvation; 3 loops (in/orchestrate/out).
- Prime everything at startup (load + persona prefill + dummy STT/TTS pass) so first real turn is warm.
- Sentence-chunked streaming TTS (also dodges mlx-audio's multi-sentence batching bug).
- Python 3.12; MPS PYTORCH_ENABLE_MPS_FALLBACK where needed.
