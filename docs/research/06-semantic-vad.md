# Semantic VAD / Smart Endpointing for Faster Turn Commitment — Research

**Project:** ReachyChatOffline — fully-local, low-latency, English-only voice-to-voice for Reachy Mini
**Target hardware:** Apple Silicon M3 / 24 GB, macOS 15, Python 3.10–3.12
**Current turn stack:** Silero VAD v6 + Pipecat Smart-Turn v3
**Problem:** endpoint detection (knowing the user has truly FINISHED, not just paused) is the dominant remaining latency. A fixed silence window either cuts people off (short) or feels sluggish (long, 200–300 ms+). We want approaches that commit *faster* on genuinely-complete utterances while *waiting* on incomplete ones — local, low-compute, on M3.
**Research date:** June 2026

> See also `04-vad-turntaking.md` (broader VAD/turn/barge-in/wake-word stack) and `05-architecture-latency.md`. This doc drills specifically into semantic endpointing and the latency floor.

---

## 0. TL;DR

- **Keep Pipecat Smart-Turn v3 (now v3.2 by default).** It is still the best *local, audio-based, no-STT-dependency* semantic endpointer available in mid-2026: ~12–20 ms CPU inference, 23 languages, BSD-2-Clause, and a recent Pipecat refactor dropped its import RSS from ~566 MB → ~60 MB and cold-start ~5.0 s → ~0.3 s. Nothing else is both this fast *and* fully local on Apple Silicon without dragging STT onto the critical path. [v3.2 blog](https://www.daily.co/blog/smart-turn-v3-2-handling-noisy-environments-and-short-responses/) [pipecat CHANGELOG](https://github.com/pipecat-ai/pipecat/blob/main/CHANGELOG.md)
- **The biggest available win is not a better model — it is the *control logic* around it: a dual-threshold + speculative-start scheme.** Run Smart-Turn continuously; the moment it fires with high confidence on a stabilized partial transcript, *speculatively* start STT-final → LLM prefill; cancel if the user resumes within a short grace window. This is exactly what Deepgram "Eager EOT" and OpenAI `semantic_vad` do server-side, and it is replicable locally. It buys ~150–250 ms on correct predictions. [Deepgram eager EOT](https://developers.deepgram.com/docs/flux/voice-agent-eager-eot)
- **Realistic "stop-talking → first-audio" floor on M3: ~350–450 ms**, of which endpoint detection is ~150–250 ms (silence confirmation + Smart-Turn). You cannot reliably go below a ~120–200 ms *acoustic* commit window without raising false-cut rate, because human turn-yielding gaps themselves cluster around 200 ms — that is the conversational floor, not an engineering one.
- **Do NOT adopt** NVIDIA Parakeet-EOU (NeMo/CUDA, Linux-only), LiveKit turn-detector as the *primary* (needs STT text first → adds latency), Krisp/Cartesia/AssemblyAI/Speechmatics/Deepgram (cloud/proprietary, violate local constraint). Vogent-Turn-80M is the one *watch-list* model worth tracking (multimodal, Apache-2.0) once it has a CPU/ONNX path.

---

## 1. What "semantic VAD" actually means

"Semantic VAD" is a marketing umbrella for **turn-completion prediction** that uses more than audio energy + a silence timer. Across vendors it splits into three architectures:

| Architecture | Decision input | Examples | Local fit |
|---|---|---|---|
| **Audio-based semantic** | raw waveform → P(turn complete); learns prosody (falling pitch, deceleration, filler/hesitation acoustics) directly | Pipecat Smart-Turn v3, Vogent-Turn (audio branch), Krisp, NVIDIA Parakeet-EOU | **Best** — no STT on the path |
| **Text/transcript-based** | recent transcript tokens → P(complete); LLM-style linguistic completeness | LiveKit turn-detector (Qwen2.5-0.5B), TEN Turn Detection (8B), Turnsense | Needs STT *first* → adds STT latency before the endpoint decision |
| **Fused / model-native** | single model emits both transcript **and** an EOU/turn signal | Deepgram Flux, AssemblyAI Universal-3, Cartesia Ink-2, NVIDIA Parakeet-EOU (`<EOU>` token), Nemotron VoiceChat | Cloud (DG/AAI/Cartesia) or GPU-only (NVIDIA) — not local on Mac |

**OpenAI Realtime `semantic_vad`** is the canonical definition to replicate: a turn-detection model runs *alongside* a plain VAD and "dynamically sets a timeout based on" the probability the user is done — i.e. it does not replace the silence timer, it *modulates* it. The `eagerness` knob (low/medium/high) just tunes the max-wait timeout: high = respond sooner / shorter wait, low = let the user ramble. This is precisely the "adaptive silence window driven by a completion probability" pattern. [OpenAI Realtime VAD](https://platform.openai.com/docs/guides/realtime-vad) [server-events ref](https://developers.openai.com/api/reference/resources/realtime/server-events)

**Inworld's framing** is the clearest taxonomy: semantic VAD scores completion from (1) linguistic completeness, (2) prosodic cues, (3) conversational context (question vs statement), (4) filler tokens, (5) hesitation/restart patterns — and their own product is literally "Silero VAD + a Smart Turn detector + session context," i.e. the same architecture we already run. [Inworld semantic VAD](https://inworld.ai/resources/what-is-semantic-vad)

**Takeaway:** we already have a credible local semantic-VAD (Silero + Smart-Turn). The realistic gains are (a) tuning the probability→timeout mapping (an eagerness curve) and (b) speculative start — not swapping the model.

---

## 2. Model / library survey (comparison table)

| Model / lib | Type | Local on M3? | Size | Decision latency | Accuracy | Languages | License |
|---|---|---|---|---|---|---|---|
| **Pipecat Smart-Turn v3.2** ✅ | Audio (waveform→P) | **Yes** (ONNX CPU, arm64; bundled in `pipecat-ai`) | 8 MB int8 / 32 MB fp32 (~8M params, Whisper-tiny enc + linear head) | **~12 ms x86, ~15 ms Graviton; ~10–20 ms M3** | ~94.7% EN @8MB / 95.6% @32MB (v3.1); v3.2 +40% on short utterances & noise | 23 | BSD-2-Clause (weights+data+train) |
| **LiveKit turn-detector v0.4.1-intl** | Text (transcript) | **Yes** but needs STT text first | ~396 MB on disk (Qwen2.5-0.5B distilled→INT8 ONNX), ~400 MB RAM | ~25 ms CPU **+ STT latency** | TPR 99.3–99.4%; −39% false interruptions vs v0.3 | 14 | Apache-2.0 code / custom model license |
| **Vogent-Turn-80M** 👀 | **Multimodal** (Whisper enc audio + text) | Inference code yes (Apache-2.0); **no CPU/ONNX path yet** | 80M | ~7 ms **on T4 GPU** (no CPU number) | 94.1% (claims SoTA) | EN-focused | Apache-2.0 (code) |
| **NVIDIA Parakeet-realtime-EOU-120M** ❌ | Fused ASR + `<EOU>` token (audio) | **No** — NeMo 2.5.3+, CUDA, Linux; NVIDIA Open Model License | 120M | EOU p50 160 ms / p90 280 ms / p95 320 ms | (TTS-eval) | EN only | NVIDIA Open Model License |
| **NVIDIA Nemotron VoiceChat 12B** ❌ | End-to-end S2S w/ built-in VAD+EOU | No (GPU) | 12B | — | — | — | NVIDIA |
| **Kyutai STT 1B (semantic VAD)** ⚠️ | Audio semantic VAD inside streaming STT | Yes via `moshi-mlx` on M3, but heavy | 1B (also 2.6B) | VAD fires then **+500 ms STT lookahead delay** (1B); 2.5 s (2.6B) | — | EN/FR (1B) | code MIT/Apache, weights CC-BY |
| **Krisp VIVA Turn-Taking v2** ❌ | Audio | Proprietary SDK (.kef weights), Pipecat `krisp_viva_turn` hook | — | — | claims SoTA | — | Commercial |
| **TEN Turn Detection** ❌ | Text (8B Qwen2.5) | Technically yes, but 8B → seconds on CPU | 8B | seconds (CPU) | 3-class finished/unfinished/**wait** | EN/ZH | Apache-2.0 |
| **Turnsense (135M)** ⚠️ | Text | Yes (Apache-2.0 edge classifier) | 135M | low | weak (2k-sample train set) | EN | Apache-2.0 |
| **Deepgram Flux (Eager EOT)** ❌ | Fused ASR + native turn detection | **Cloud only** | — | EOT p50 <300 ms; EagerEOT 150–250 ms earlier | acoustic+semantic+context | EN (+) | Commercial cloud |
| **AssemblyAI Universal-3 Pro Streaming** ❌ | Acoustic turn + semantic endpointing | Cloud only | — | ~150 ms P50 after VAD endpoint | 6.3% WER | EN domains | Commercial cloud |
| **Cartesia Ink-2** ❌ | Streaming STT + built-in turn detection | Cloud only | — | — | noise-robust | — | Commercial cloud |
| **OpenAI Realtime `semantic_vad`** ❌ | Hosted turn model + VAD; `eagerness` knob | Cloud only | — | dynamic timeout | — | many | Commercial cloud |

Legend: ✅ recommended/primary · 👀 watch-list · ⚠️ usable but compromised · ❌ not viable locally.

---

## 3. Local-runnability verdicts (M3 / fully offline)

- **Smart-Turn v3.2 — VERDICT: keep as primary.** Only model here that is audio-based, sub-20 ms, fully offline, permissively licensed, and already integrated. The v3.2 default in Pipecat plus the numpy-vendored feature extractor (no `transformers` import, ~60 MB RSS, ~0.3 s cold start) makes it cheap to keep resident. [pipecat local_smart_turn_v3 ref](https://reference-server.pipecat.ai/en/stable/api/pipecat.audio.turn.smart_turn.local_smart_turn_v3.html)
- **LiveKit turn-detector — VERDICT: optional second-stage gate only.** It is genuinely local and very accurate, but it is *text-based*: it cannot decide until STT has emitted tokens, so it adds STT latency *before* the endpoint commit. Use it only to *suppress* false commits on ambiguous endings, never as the first trigger. ~400 MB RAM is also non-trivial alongside STT+LLM+TTS on 24 GB.
- **Vogent-Turn-80M — VERDICT: watch-list.** Conceptually the best fit (multimodal audio+text, Apache-2.0, claims to beat Smart-Turn), but published latency is GPU-only (~7 ms T4) with no CPU/ONNX export and EN-centric. Re-evaluate if/when an ONNX-CPU build appears. [vogent-turn repo](https://github.com/vogent/vogent-turn)
- **Kyutai STT semantic VAD — VERDICT: only if you also adopt Kyutai STT.** Its semantic VAD is *coupled* to the STT model and carries a ~500 ms STT lookahead delay before EOU is trustworthy — that is *more* endpoint latency than Smart-Turn, not less. Not worth it unless the STT choice changes. [Kyutai STT](https://kyutai.org/stt) [delayed-streams-modeling](https://github.com/kyutai-labs/delayed-streams-modeling)
- **Parakeet-EOU / Nemotron — VERDICT: no.** NeMo + CUDA + Linux. (Parakeet *ASR* has `parakeet-rs` / `parakeet.cpp` CPU ports, but those do **not** ship the EOU head.) [Parakeet-EOU card](https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1)
- **All cloud vendors (Deepgram/AAI/Cartesia/OpenAI/Krisp) — VERDICT: no.** Violate the fully-local constraint; listed only as design references for the *technique* (eager EOT, eagerness curves).

---

## 4. Reducing endpoint latency WITHOUT a (new) model

These are the highest-leverage moves and most are free given our current stack.

1. **Dual-threshold endpointing (the core pattern).** Don't treat Smart-Turn as a single binary at one silence window. Run two paths:
   - **Eager commit:** if Smart-Turn P(complete) is *very high* (e.g. ≥0.85) AND a short ~100–120 ms silence has elapsed → commit immediately.
   - **Conservative commit:** if P is *medium* (0.5–0.85) → wait the full ~200–300 ms silence window (the user is probably mid-thought).
   - **Hold:** if P is low → extend the window / keep listening.
   This is OpenAI's `eagerness` and Deepgram's `eot_threshold` re-implemented locally over Smart-Turn's probability. [Deepgram EOT config](https://developers.deepgram.com/docs/flux/configuration)

2. **Speculative LLM start on stabilized partials (biggest single win).** When Smart-Turn fires *and* the streaming STT partial has been stable for ~150–200 ms, start the LLM **prefill/first-token speculatively** before the final transcript. If the user resumes within a grace window (~250–300 ms), cancel the speculative generation (cooperative `asyncio` cancel — we already have this pattern in §3.3 of `04-vad-turntaking.md`). Reported ~150–250 ms saved when the prediction holds (correct 80–90% of the time), at the cost of ~50–70% extra LLM calls — cheap with a *local* LLM. [Deepgram eager EOT](https://developers.deepgram.com/docs/flux/voice-agent-eager-eot) [latency engineering](https://medium.com/@reveorai/solving-voice-ai-latency-from-5-seconds-to-sub-1-second-responses-d0065e520799)

3. **Adaptive (per-session) silence window.** Calibrate the base silence window to the *observed speaker cadence* in the first few turns — slow talkers get a longer window, fast talkers a shorter one — rather than one global constant. Cheap, no model. [adaptive endpointing (bandits)](https://arxiv.org/pdf/2303.13407)

4. **Prosody cues as a free prior.** Falling pitch, energy roll-off and speech-rate deceleration are strong turn-completion signals; Smart-Turn already learns these from the waveform, but a trivial pitch/energy slope check (from the same audio frames) can *shorten* the silence window when the contour is clearly terminal and *lengthen* it on a rising/level contour (a held turn). Prosodic endpointing is decades-old, well validated. [prosody endpoint patent](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/6873953) [Inworld](https://inworld.ai/resources/what-is-semantic-vad)

5. **Linguistic-completeness fast path.** If a streaming STT partial is already syntactically complete and ends with a clear terminal (e.g. a question), you can drop the silence window further. This is the LiveKit-style text signal applied *opportunistically* without making it a hard dependency.

6. **Overlap STT with the silence window.** Endpoint detection latency is partly hideable: keep STT decoding *through* the silence-confirmation window so that the final transcript is ready the instant the endpoint commits (we already do streaming STT per `05-architecture-latency.md`).

---

## 5. The latency floor & concrete recommendation for our stack

### 5.1 What "stop-talking → first-audio" decomposes into

| Stage | Realistic M3 budget | Notes |
|---|---|---|
| Acoustic silence confirmation | **100–250 ms** | the dominant, tunable knob; cannot reliably go <~120 ms without false cuts |
| Smart-Turn v3.2 inference | ~15 ms | audio-based, no STT needed |
| STT finalization (overlapped) | ~0–80 ms incremental | mostly hidden if streaming |
| LLM first token (TTFT, local) | ~100–200 ms | **speculative start hides most of this** |
| TTS first chunk (TTFA, local) | ~80–150 ms | sentence-chunked streaming |
| **Total perceived** | **~350–500 ms** | speculative start pulls toward the low end |

### 5.2 The floor

- **Engineering floor (endpoint portion):** ~120–150 ms silence + ~15 ms Smart-Turn ≈ **~135–165 ms** before you can confidently commit on a *clearly* complete utterance with eager thresholds. Push silence below ~120 ms and false-cut rate climbs.
- **Conversational floor:** human turn-transition gaps cluster around ~200 ms (and people *expect* a beat). Committing faster than ~150 ms can feel like the robot is interrupting. So ~150–250 ms endpoint latency is not just a limitation — part of it is *desirable* naturalness.
- **End-to-end floor with speculative start:** **~350–450 ms** stop-talking→first-audio is realistic and reliable on M3. Sub-350 ms is achievable only on the subset of turns where eager commit + speculative LLM prefill both fire correctly.

### 5.3 Recommendation

1. **Keep Smart-Turn v3.2 as the primary endpointer** (it is the right local model; do not swap it). Ensure you're on the Pipecat build with the v3.2 default + numpy-vendored extractor.
2. **Add a dual-threshold "eagerness" layer over Smart-Turn's probability** (eager ≥0.85 → 100–120 ms window; conservative 0.5–0.85 → 200–300 ms window; <0.5 → hold). This is the single cleanest, model-free latency reduction.
3. **Add speculative LLM start** on stabilized STT partials + high Smart-Turn confidence, with cooperative cancel on resume (reuse the barge-in cancel machinery already documented in `04-vad-turntaking.md` §3.3). Cheap because the LLM is local.
4. **Keep an absolute silence-timeout fallback (~0.6 s)** so the pipeline never hangs on a missed endpoint.
5. **Optionally** layer the LiveKit text turn-detector *only* as a false-commit suppressor on ambiguous endings — opt-in, never on the first trigger.
6. **Watch Vogent-Turn-80M** for a CPU/ONNX release; it is the only model that could plausibly beat Smart-Turn locally.

Net: the marginal model upgrade is small; the marginal *control-logic* upgrade (eagerness curve + speculative start) is where the sub-450 ms experience comes from.

---

## Sources

**Smart-Turn:** [v3 (12ms) blog](https://www.daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/) · [v3.1 blog](https://www.daily.co/blog/improved-accuracy-in-smart-turn-v3-1/) · [v3.2 blog](https://www.daily.co/blog/smart-turn-v3-2-handling-noisy-environments-and-short-responses/) · [smart-turn repo](https://github.com/pipecat-ai/smart-turn) · [smart-turn-v3 HF](https://huggingface.co/pipecat-ai/smart-turn-v3) · [Pipecat Smart Turn overview](https://docs.pipecat.ai/api-reference/server/utilities/turn-detection/smart-turn-overview) · [local_smart_turn_v3 ref](https://reference-server.pipecat.ai/en/stable/api/pipecat.audio.turn.smart_turn.local_smart_turn_v3.html) · [pipecat CHANGELOG](https://github.com/pipecat-ai/pipecat/blob/main/CHANGELOG.md)

**LiveKit turn-detector:** [HF model](https://huggingface.co/livekit/turn-detector) · [plugin docs](https://docs.livekit.io/agents/build/turns/turn-detector/) · [EOU −39% blog](https://livekit.com/blog/improved-end-of-turn-model-cuts-voice-ai-interruptions-39) · [transformer EOU blog](https://blog.livekit.io/using-a-transformer-to-improve-end-of-turn-detection) · [PyPI](https://pypi.org/project/livekit-plugins-turn-detector/) · [Turns overview](https://docs.livekit.io/agents/logic/turns/)

**Other models:** [Vogent-Turn-80M HF](https://huggingface.co/vogent/Vogent-Turn-80M) · [vogent-turn repo](https://github.com/vogent/vogent-turn) · [Parakeet-EOU-120M HF](https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1) · [Nemotron Voice Agent](https://build.nvidia.com/nvidia/nemotron-voice-agent) · [NeMo voice_agent README](https://github.com/NVIDIA-NeMo/NeMo/blob/main/examples/voice_agent/README.md) · [Kyutai STT](https://kyutai.org/stt) · [delayed-streams-modeling](https://github.com/kyutai-labs/delayed-streams-modeling)

**Semantic VAD definitions / vendor APIs:** [OpenAI Realtime VAD](https://platform.openai.com/docs/guides/realtime-vad) · [OpenAI server-events ref](https://developers.openai.com/api/reference/resources/realtime/server-events) · [Inworld: what is semantic VAD](https://inworld.ai/resources/what-is-semantic-vad) · [AssemblyAI turn detection](https://www.assemblyai.com/docs/universal-streaming/turn-detection) · [AssemblyAI endpointing blog](https://www.assemblyai.com/blog/turn-detection-endpointing-voice-agent) · [Cartesia Ink](https://www.cartesia.ai/ink/)

**Eager EOT / speculative / model-free techniques:** [Deepgram eager EOT (voice agent)](https://developers.deepgram.com/docs/flux/voice-agent-eager-eot) · [Deepgram Flux config](https://developers.deepgram.com/docs/flux/configuration) · [Deepgram: evaluating EOT models](https://deepgram.com/learn/evaluating-end-of-turn-detection-models) · [latency: 5s→sub-1s](https://medium.com/@reveorai/solving-voice-ai-latency-from-5-seconds-to-sub-1-second-responses-d0065e520799) · [adaptive endpointing (contextual bandits)](https://arxiv.org/pdf/2303.13407) · [prosody-based endpoint (patent)](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/6873953) · [predicting initiation points](https://arxiv.org/pdf/2208.03812) · [LiveKit voice agent architecture](https://livekit.com/blog/voice-agent-architecture-stt-llm-tts-pipelines-explained)
