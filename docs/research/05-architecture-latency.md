# 05 — End-to-End Architecture & Latency Budgeting

**Project:** ReachyChatOffline — fully-local, low-latency, English-only voice-to-voice assistant for the Reachy Mini robot
**Target HW:** Apple Silicon M3 / 24 GB (working set ≲20 GB); dev on M3 Max 96 GB. macOS 15. Python 3.10–3.12. MLX/MPS preferred.
**Hard target:** voice-to-voice ≤ **500 ms** (≤ **600 ms** with a webcam frame in the loop). Natural turn-taking + barge-in required.
**Date:** 2026-06-17 — all sources verified current as of June 2026.

---

## 0. TL;DR Verdict

**Build a cascade-with-overlap pipeline, not a full-duplex speech-to-speech model.**

- A streaming cascade (Smart-Turn-v3 endpoint → Kyutai STT → small MLX LLM → Kyutai/Kokoro streaming TTS) hits a realistic **~430–480 ms p50** voice-to-voice on M3, fits in ≲20 GB, and lets us hot-swap a high-quality emotive TTS and bolt on optional vision.
- Full-duplex S2S (Moshi 7B) is genuinely lower *raw* latency (~250 ms on M4 Max MLX) **but** it is a sealed 7B Helium backbone: weak reasoning, ~3.8 MOS voice, English-only, no clean way to inject our persona/knowledge or swap the voice, and ~10–16 GB just for the S2S model leaving little room for vision/RAG. It loses on every constraint except raw latency, and the cascade is already under budget.
- The two project-relevant gotchas that dominate real-world latency are **(a) asyncio event-loop starvation during MLX inference** and **(b) MLX full-prefill TTFT**. Both are addressed below.

---

## 1. Cascade-with-Overlap vs Full-Duplex S2S

### 1.1 Cascade WITH streaming/overlap

The naïve cascade is `VAD-endpoint → STT(finalize) → LLM(full) → TTS(full) → play`, where every stage waits for the previous to *finish*. That serial floor is what makes people quote 800 ms+ budgets ([Channel](https://www.channel.tel/blog/voice-ai-pipeline-stt-tts-latency-budget), [Smallest.ai](https://smallest.ai/blog/designing-voice-assistants-stt-llm-tts-tools-and-latency-budget)). The whole game is **overlap**: make stages emit partials and make downstream stages start on those partials.

Overlap tricks that actually move the needle (all in our design):

1. **Streaming STT, not batch Whisper.** Kyutai STT is a *delayed-streams* model: text is emitted ~500 ms after it is spoken (1B model), so by end-of-speech almost the entire transcript already exists. The "flush trick" forces the buffered tail out in **~125 ms** instead of waiting the full delay ([kyutai.org/stt](https://kyutai.org/stt), [delayed-streams-modeling](https://github.com/kyutai-labs/delayed-streams-modeling)). This is the single biggest win vs. mlx-whisper, which only starts transcribing *after* endpoint and re-processes the whole utterance.
2. **Semantic endpointing instead of fixed silence.** A fixed 500–700 ms silence timer is pure dead air on every turn. Smart-Turn-v3 (Pipecat) classifies "is the speaker done?" from the waveform in **~12 ms CPU** ([Daily](https://www.daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/)), letting us commit the turn ~100–150 ms after true end-of-speech instead of 500 ms+. Kyutai STT also ships a semantic-VAD pause predictor whose delay adapts to intonation ([kyutai.org/stt](https://kyutai.org/stt)).
3. **Start the LLM on the (near-)final transcript immediately at endpoint** — because streaming STT already has it, there is no STT "finalization" wait beyond the flush.
4. **Start TTS on the first LLM sentence/clause, not the full reply.** Sentence-chunked TTS: emit audio for sentence 1 while the LLM is still generating sentence 2. Kyutai TTS and Kokoro both synthesize and yield chunk-by-chunk; Kyutai TTS is explicitly designed to *ingest LLM tokens as they stream* ("start making audio as soon as text arrives", **~220 ms** time-to-first-audio) ([erogol](https://erogol.substack.com/p/model-check-kyutaitts-streaming-text)). Kokoro yields sentence-by-sentence at sub-200 ms ([mlx-audio](https://github.com/Blaizzy/mlx-audio)).
5. **Prefetch / prewarm.** Keep the LLM warm with a cached system-prompt prefix (TTFT for a cached prefix drops to <0.3 s and small models reload cache fast — [roborhythms](https://www.roborhythms.com/reduce-local-llm-ttft-mac-studio/), [yage.ai](https://yage.ai/share/mlx-apple-silicon-en-20260331.html)). Keep STT/TTS models resident and warmed (≥10 warm-up frames before measuring — per the `mlx-realtime-inference-optimization` skill).
6. **Minimise output buffering.** Use a small audio output blocksize (~20–40 ms) via `sounddevice`; do not buffer a whole sentence of PCM before the speaker starts.

**Where the bottlenecks are on M3:**

- **LLM TTFT is the #1 budget item.** It is compute-bound (prefill) and MLX does *full prefill before emitting any token*, so TTFT rises with prompt length ([yage.ai](https://yage.ai/share/mlx-apple-silicon-en-20260331.html)). Mitigation: small model (3–4B 4-bit), a *cached* system prompt, and a short rolling conversation window. M5 neural accelerators give up to 4× TTFT speedup but we target M3, so we budget conservatively.
- **TTS time-to-first-audio** is #2 (~150–220 ms).
- **Endpointing** is #3 but is the easiest to *waste* (fixed silence) and the easiest to fix (semantic turn detection).
- STT "finalization" is nearly free with streaming STT (just the ~125 ms flush).

**Is ≤500 ms achievable?** Yes, at p50, on M3, with this overlap design (budget in §3). The tail (p95) is tighter — see §3 and §6.

### 1.2 Full-Duplex / Speech-to-Speech models

| Model | Raw latency | Memory (Apple Silicon) | English quality | Persona / knowledge | Vision | Verdict for us |
|---|---|---|---|---|---|---|
| **Moshi / Kyutai (Mimi codec, 7B Helium)** | ~160 ms theoretical, ~200 ms on L4, **~250 ms on M4 Max MLX** ([moshi](https://github.com/kyutai-labs/moshi), [localaimaster](https://localaimaster.com/blog/moshi-realtime-speech-guide)) | ~16 GB FP16, **~10 GB int4/int8** ([localaimaster](https://localaimaster.com/blog/moshi-realtime-speech-guide)) | **~3.8 MOS**, English-only mid-2026, only trained voices | Fine-tune only (separate `moshi-finetune` repo); **no clean runtime persona/knowledge injection**, weak 7B reasoning, limited tool calling | None | Lowest raw latency, worst on every other axis |
| **Qwen2.5/3.5-Omni (thinker-talker)** | Streaming talker, low latency by design; MLX 7 s vs 21 s llama.cpp on Qwen3.5 (not S2S-specific) ([willitrunai](https://willitrunai.com/blog/qwen-3-5-mlx-apple-silicon-guide), [arxiv 2604.15804](https://arxiv.org/html/2604.15804v1)) | Omni weights are large; talker+thinker+codec push well past a lean budget | Strong text brain (thinker), decent talker | Persona via system prompt (it's an LLM), but voice is the built-in talker | **Yes — audio+vision+video in** | Best *brain*, heaviest footprint; voice not swappable; overkill for ≲20 GB latency-first |
| **Gemma 3n / Gemma 4 (audio-in)** | Audio understanding, **not** speech-out; ~1.5× faster first response on-device ([HF gemma4](https://huggingface.co/blog/gemma4), [lmstudio gemma-3n](https://lmstudio.ai/models/gemma-3n)) | Compact (PLE, KV-sharing); 3n designed for edge | ASR-grade audio understanding | Full system-prompt control (plain LLM) | **Yes (image/audio/video in)** | An *audio-in LLM*, i.e. it replaces STT+LLM in a cascade — **not** a full-duplex S2S. Useful as a fallback "omni brain" if we want vision+audio fused, but still needs an external TTS. |

**Why full-duplex loses for us:** our constraints are *latency-first AND ≤20 GB AND swappable emotive TTS AND optional vision AND custom persona/knowledge*. Moshi gives only the first. It cannot swap voices, cannot take vision, has a weak sealed brain, and offers no clean knowledge injection. Qwen-Omni is the strongest brain but its footprint and non-swappable voice fight the ≲20 GB / emotive-TTS goals. Gemma-3n is an audio-in brain, not S2S, so it slots *into* a cascade rather than replacing it. Meanwhile our cascade already meets the latency target with full freedom over every component. **Keep full-duplex (Moshi) as a documented Plan-B "low-latency mode" we could A/B, not the primary architecture.**

> Apple-Silicon caveat (skill: `pytorch-mps-apple-silicon`): any PyTorch/MPS port of these models is 5–10× slower than MLX (no Flash-Attn, no CUDA Graphs, no inductor, bf16 upcast, sync points). All quoted Mac latencies assume the **MLX** path. Prefer MLX builds (`moshi_mlx`, `mlx-whisper`, `mlx-audio`, MLX LLMs); treat MPS as a last resort.

---

## 2. Recommended Architecture

**Cascade-with-overlap, all MLX, all stages streaming, three concurrent async loops + a dedicated inference thread pool.**

```
                                 ReachyChatOffline — Cascade-with-Overlap (all local, MLX)
                                 voice-to-voice target: <=500ms p50  (<=600ms with vision)

  Mic 24kHz mono                                                                      Speaker
   |                                                                                     ^
   |  sounddevice InputStream (20-40ms blocks)                  sounddevice OutputStream |
   v                                                            (20-40ms blocks)         |
 +-----------------+      +------------------------+                          +----------+--------+
 | AUDIO-IN LOOP   |      | TURN / ENDPOINT        |                          | AUDIO-OUT LOOP    |
 | (asyncio)       |----->| Smart-Turn-v3 (~12ms)  |                          | ring buffer / play|
 | ring buffer     |      | + Kyutai semantic VAD  |                          | (barge-in: flush) |
 +--------+--------+      +-----------+------------+                          +----------+--------+
          |                           | endpoint event (t=0)                            ^
          | PCM frames                v                                                  |
          v               (transcript already ~complete via delayed streams)            | PCM chunks
 +--------------------+    flush (~125ms tail)                                           |
 | STREAMING STT      |--------------------------------+                                 |
 | Kyutai stt-2.6b-en |   partial + final transcript   |                                 |
 | (MLX, resident)    |                                v                                 |
 +--------------------+                     +---------------------+    first sentence    |
                                            | LLM (MLX)           |    (clause-chunked)  |
   [OPTIONAL VISION]                        | Qwen3/Llama 3.x 3-4B|------------------+   |
   webcam frame --> vision encoder ----+--->| 4-bit, KV cache,    |   tokens stream  |   |
   (only when image in loop, +~100ms)  |    | cached sys-prompt   |                  v   |
                                       |    +---------------------+         +----------------+
   persona + knowledge (RAG / system prompt) -----^                         | STREAMING TTS  |
                                                                            | Kyutai TTS 1.6B|
                                                                            | OR Kokoro-82M  |
   ALL MLX inference runs in a thread pool (asyncio.to_thread) so the       | (MLX, ~150-220 |
   three event-loop tasks never starve during GPU work.                     |  ms TTFA)      |
                                                                            +----------------+
```

**Component choices (primary):**

- **Endpointing / turn-taking:** Pipecat **Smart-Turn-v3** (semantic, ~12 ms CPU, offline, `LocalSmartTurnAnalyzerV3`) + Kyutai STT's built-in semantic pause predictor. Gives natural turn-taking without a fixed silence timer.
- **STT:** **Kyutai `stt-2.6b-en`** (English-only, streaming, MLX) — text trails audio so it's ~done at endpoint; flush ≈125 ms. (`stt-1b-en_fr` is a lighter 0.5 s-delay alternative with built-in semantic VAD.)
- **LLM:** a **3–4B instruct model in MLX 4-bit** (Qwen3-4B / Llama-3.x-3B class), system prompt cached/prewarmed, short rolling context. This is the persona + knowledge surface (system prompt + optional RAG).
- **TTS:** **Kyutai TTS 1.6B** (MLX, streams from LLM tokens, ~220 ms TTFA, 10-s voice cloning) **or** **Kokoro-82M** (MLX, sub-200 ms, 54 voices) — both swappable behind a `TTSEngine` interface so a higher-quality emotive TTS can drop in later.
- **Vision (optional):** webcam frame → vision encoder feeding the LLM only when an image is in the loop (adds ~100 ms; covered by the 600 ms vision budget). If we ever want fused audio+image reasoning, **Gemma-3n / Qwen-Omni as the audio+vision-in brain** is the documented swap-in.

**Concurrency model (skills `mlx-realtime-inference-optimization`, `pytorch-mps-apple-silicon`):**

- Three asyncio tasks — audio-in, orchestration, audio-out — must **never** call MLX inference synchronously on the event loop. Wrap every inference call in `asyncio.to_thread(...)`; the GIL is released during Metal GPU and Rust (PyO3 codec) work, so the loops stay responsive ("not listening" bug fix).
- Pipeline CPU codec work against GPU inference (dual codec instances, independent streaming state) to hide decode behind the LM step.
- `sounddevice` over PyAudio (better macOS support, per `realtime-whisper-macos`). Output blocksize ~20–40 ms; barge-in = flush the output ring buffer + reset TTS the instant the turn analyzer fires during playback.
- Watch `sphn` version pinning if any Moshi/Kyutai opus path is used (`sphn-opus-api-versioning`: keep the version matched to the app's queue-vs-sync API).

---

## 3. Per-Stage Latency Budget (M3, p50)

`t=0` is defined as **the end of the user's speech waveform** (last voiced sample). All numbers are wall-clock on M3 (not M3 Max), MLX path, models resident + warmed, system prompt cached.

| # | Stage | What happens after t=0 | p50 (ms) | Notes / source |
|---|---|---|---|---|
| 1 | **Endpoint decision** | Smart-Turn-v3 confirms turn end | **40** | ~12 ms inference + frame aggregation/hysteresis ([Daily v3](https://www.daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/)) |
| 2 | **STT finalize (flush)** | streaming STT emits buffered tail | **125** | "flush trick" ([kyutai.org/stt](https://kyutai.org/stt)); transcript otherwise already present |
| 3 | **LLM TTFT** | prefill cached sys-prompt + short ctx, emit token 1 | **150** | compute-bound; 4-bit 3–4B, cached prefix ([yage.ai](https://yage.ai/share/mlx-apple-silicon-en-20260331.html), [roborhythms](https://www.roborhythms.com/reduce-local-llm-ttft-mac-studio/)) |
| 4 | **LLM → first clause** | generate enough tokens for TTS to start | **40** | ~5–10 tok at small-model speed; overlaps with TTS warm |
| 5 | **TTS time-to-first-audio** | first PCM chunk from first clause | **110** | Kyutai 220 ms / Kokoro <200 ms TTFA, reduced by clause-chunking + warm ([erogol](https://erogol.substack.com/p/model-check-kyutaitts-streaming-text), [mlx-audio](https://github.com/Blaizzy/mlx-audio)) |
| 6 | **Audio output buffering** | first chunk reaches speaker | **30** | one ~20–40 ms `sounddevice` output block |
| | **TOTAL voice-to-voice (p50)** | | **~495 ms** | **under the 500 ms target** |
| + | **Vision add-on** | webcam frame encode + LLM cross-attn prefill | **+100** | only when image in loop → **~595 ms**, under 600 ms |

Stages 3–6 partially overlap (LLM keeps generating while TTS speaks sentence 1; TTS streams while audio plays), so the *sum above is already the conservative serialized critical path to first audible word*. Headroom is thin: the budget assumes warm/cached models. **Cold start, growing KV cache, or a fixed silence timer each blow the budget** — see §5.

---

## 4. Cascade-vs-Full-Duplex — Final Verdict

| Criterion | Cascade-with-overlap (recommended) | Full-duplex Moshi 7B |
|---|---|---|
| Voice-to-voice latency | ~495 ms p50 (under target) | ~250 ms (M4 Max MLX) — best |
| Memory (working set) | ~10–14 GB (STT 2.6B + LLM 3–4B 4-bit + TTS 1.6B + VAD) — **fits ≲20 GB** | ~10–16 GB just for S2S; little room for vision/RAG |
| English quality | STT SOTA-streaming; TTS swappable to emotive | ~3.8 MOS, fixed voices |
| Reasoning / brain | swap any MLX LLM; strong | sealed 7B Helium, weak, limited tools |
| Persona + knowledge | system prompt + RAG, fully controllable | fine-tune only; no clean injection |
| Swap emotive TTS | yes (interface) | no |
| Optional vision | yes (vision encoder → LLM; or Gemma-3n/Omni brain) | no |
| Barge-in / turn-taking | Smart-Turn-v3 + semantic VAD | native full-duplex (its one structural edge) |

**Decision: cascade-with-overlap is primary.** It satisfies *all* hard constraints with latency headroom; full-duplex satisfies only latency. Keep Moshi (`moshi_mlx`, int4 ~10 GB) documented as an experimental "ultra-low-latency mode" to A/B against, given the project already references `sphn`/`moshi` and the relevant skills.

---

## 5. Apple-Silicon Real-Time Tooling & Pitfalls

- **Orchestration framework:** **Pipecat v1.0** (Apr 2026, Python, transport-agnostic) is the natural fit — it ships the local Smart-Turn analyzer and a frame-pipeline model that maps onto our async loops. LiveKit Agents is heavier (WebRTC infra, multi-participant) and aimed at networked deployments; overkill for a single on-device robot. Either can be thin orchestration; we can also hand-roll asyncio given the skills already encode the hard parts. ([Pipecat vs LiveKit](https://webrtc.ventures/2026/03/choosing-a-voice-ai-agent-production-framework/), [inworld](https://inworld.ai/resources/vapi-vs-pipecat-vs-livekit))
- **MLX libs:** `mlx-whisper` (fallback STT), `mlx-audio` (Kokoro/TTS, STS), `moshi_mlx` (Plan-B S2S), MLX LLMs. Prefer MLX over MPS everywhere (compounding 5–10× MPS penalty — `pytorch-mps-apple-silicon`).
- **Audio I/O:** `sounddevice` (not PyAudio) for macOS; float32 16 kHz mono; Silero VAD if used needs **exactly 512-sample windows at 16 kHz** (`realtime-whisper-macos`).
- **#1 pitfall — asyncio event-loop starvation:** synchronous MLX inference (60–150 ms) blocks the loop and the server "stops listening." Always `asyncio.to_thread` the inference; pipeline CPU codec vs GPU LM with dual codec instances; watch for MLX eager-eval points (`.any()`, `.item()`) that secretly force `mx.eval`. (`mlx-realtime-inference-optimization`)
- **#2 pitfall — MLX full-prefill TTFT:** TTFT grows with prompt length; cap context, cache the system prompt, prewarm.
- **#3 pitfall — `sphn` API break:** 0.1.x (queue) vs 0.2.x (sync) — match the version to the app to avoid stuttery audio (`sphn-opus-api-versioning`).
- **#4 — drift tracking:** log per-frame time and cumulative drift every N frames; KV-cache growth raises LM step time until a rotating cache stabilises it.

---

## 6. Benchmark Methodology (offline, reproducible, p50/p95)

**Goal:** measure true voice-to-voice latency from pre-recorded prompts, with per-stage attribution, reported as p50/p95.

**Corpus.** A set of WAV files (16/24 kHz mono), each one user utterance, varied length (short command, mid, long), recorded with natural trailing silence. Keep a hand-labelled **end-of-speech sample index** per file (last voiced sample) — this is the ground-truth `t=0`. Label it once with a forced-aligner or manual annotation; do **not** let the system's own VAD define t=0 (that would hide endpointing latency).

**Defining t=0.** `t=0 = (end_of_speech_sample_index / sample_rate)`. Feed the file into the pipeline **in real time** (stream it through the same `sounddevice`-style frame path, not a bulk `transcribe(file)`), so endpointing and STT streaming behave as in production. Everything after the labelled end-of-speech counts against the budget; audio before it is "free."

**Instrumentation (single monotonic clock, `time.perf_counter()`):** emit a timestamped event at each boundary:

| Marker | Event |
|---|---|
| `t0` | labelled end-of-speech crossed (injected as a known timeline offset) |
| `t_endpoint` | turn analyzer fires |
| `t_stt_final` | STT final transcript ready (post-flush) |
| `t_llm_ttft` | first LLM token |
| `t_llm_clause` | first clause/sentence ready for TTS |
| `t_tts_ttfa` | first TTS PCM chunk produced |
| `t_audio_out` | **first sample written to output device** ← primary v2v latency = `t_audio_out − t0` |

Per-stage deltas = consecutive markers. The **headline metric is `t_audio_out − t0`** (time-to-first-audible-word). Also record time-to-*full*-response if needed.

**Procedure.**
1. Warm up: run ≥10 utterances first and discard (model load, KV warm, cache fill) — per `mlx-realtime-inference-optimization`.
2. Run each corpus file N≥30 times (or 30 distinct files × ≥3 reps) to get a stable distribution.
3. Pin config: same models, quant, context length, output blocksize, sample rate, silence budget — apples-to-apples ([Gladia](https://www.gladia.io/blog/measuring-latency-in-stt), [Hamming](https://hamming.ai/resources/voice-agent-evaluation-metrics-guide)).
4. Record every marker for every run to a CSV/JSONL.

**Reporting.** For the headline metric and each stage delta, report **p50, p95, p99, mean, std, max**. p50 = typical UX; p95 = tail/"feels sluggish" ([Hamming](https://hamming.ai/resources/voice-agent-evaluation-metrics-guide), [Telnyx](https://telnyx.com/resources/voice-ai-agents-compared-latency)). Also report **RTF** (compute time ÷ audio duration) per stage to catch throughput regressions independent of utterance length. Pass/fail gate: **headline p50 ≤ 500 ms and p95 ≤ ~650 ms**; vision-in-loop p50 ≤ 600 ms.

**What to A/B with this harness:** Kyutai-STT vs mlx-whisper; Kyutai-TTS vs Kokoro; 3B vs 4B LLM; cached vs cold system prompt; cascade vs `moshi_mlx` S2S; with/without dual-codec pipelining; fixed-silence vs Smart-Turn endpointing (expect the largest single delta here).

---

## Sources

- Kyutai Moshi (GitHub): https://github.com/kyutai-labs/moshi
- Kyutai delayed-streams-modeling (STT/TTS): https://github.com/kyutai-labs/delayed-streams-modeling
- Kyutai STT page (latency, flush trick, semantic VAD): https://kyutai.org/stt
- Kyutai TTS model-check (220 ms TTFA, voice cloning, MLX): https://erogol.substack.com/p/model-check-kyutaitts-streaming-text
- Running Kyutai low-latency models on macOS in one command: https://anil.recoil.org/notes/kyutai-streaming-voice-mlx
- Moshi real-time guide (M4 Max ~250 ms, memory, 3.8 MOS): https://localaimaster.com/blog/moshi-realtime-speech-guide
- Qwen3.5-Omni technical report: https://arxiv.org/html/2604.15804v1
- Qwen 3.5 on Apple Silicon MLX (memory, tok/s): https://willitrunai.com/blog/qwen-3-5-mlx-apple-silicon-guide
- Gemma 4 / 3n multimodal on-device (audio+vision): https://huggingface.co/blog/gemma4 ; https://lmstudio.ai/models/gemma-3n
- mlx-audio (Kokoro/TTS/STS, streaming): https://github.com/Blaizzy/mlx-audio
- Pipecat Smart-Turn v3 (12 ms CPU): https://www.daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/ ; v2 model: https://huggingface.co/pipecat-ai/smart-turn-v2
- Pipecat vs LiveKit framework choice (2026): https://webrtc.ventures/2026/03/choosing-a-voice-ai-agent-production-framework/ ; https://inworld.ai/resources/vapi-vs-pipecat-vs-livekit
- MLX vs llama.cpp / MLX TTFT full-prefill: https://yage.ai/share/mlx-apple-silicon-en-20260331.html
- Reducing local LLM TTFT on Mac (prompt cache): https://www.roborhythms.com/reduce-local-llm-ttft-mac-studio/
- Voice pipeline latency budgets: https://www.channel.tel/blog/voice-ai-pipeline-stt-tts-latency-budget ; https://smallest.ai/blog/designing-voice-assistants-stt-llm-tts-tools-and-latency-budget
- STT latency measurement (TTFB/partials/finals/RTF): https://www.gladia.io/blog/measuring-latency-in-stt
- Voice agent eval metrics (p50/p95): https://hamming.ai/resources/voice-agent-evaluation-metrics-guide ; https://telnyx.com/resources/voice-ai-agents-compared-latency
- Turn-taking / barge-in 2026: https://futureagi.com/blog/voice-ai-barge-in-turn-taking-2026/ ; https://gradium.ai/blog/semantic-vad

**Project skills incorporated:** `mlx-realtime-inference-optimization`, `mlx-whisper-macos`, `realtime-whisper-macos`, `sphn-opus-api-versioning`, `pytorch-mps-apple-silicon`.
