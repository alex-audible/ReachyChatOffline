# 02 — Speech-to-Text (STT/ASR) Research

**Project:** ReachyChatOffline — fully-local, low-latency, English-only voice-to-voice for Reachy Mini
**Target deployment:** Apple Silicon M3 / 24 GB RAM (working set ≲20 GB); dev on M3 Max 96 GB
**Constraints:** macOS 15, Python 3.10–3.12, prefer MLX/MPS, voice-to-voice ≤500 ms total
**STT budget:** first partial < ~150 ms after speech, low finalization latency, streaming strongly preferred
**Researched:** 2026-06-17

---

## TL;DR — Ranked recommendation

For a **sub-150 ms-partial, English-only, streaming** STT on M3, the architecture matters more than the brand: you want a **cache-aware streaming encoder (FastConformer/Conformer-RNNT)** that emits partials per small chunk, *not* a chunked-Whisper that buffers ~15 s windows behind a `LocalAgreement` policy.

| Rank | Choice | Why | Streaming partial latency | English WER | Mem | License |
|---|---|---|---|---|---|---|
| **1** | **Nemotron-ASR streaming 0.6B via MLX** (`dboris/nemotron-asr-mlx`, src `nvidia/nemotron-speech-streaming-en-0.6b`) | True cache-aware frame-by-frame streaming, MLX-native, ~112× RT on M-series, best accuracy/latency tradeoff | ~80–160 ms (chunk-tunable) | 2.70% LS-clean, 5.52% LS-other | ~3.4 GB peak | Apache-2.0 (code) / NVIDIA Open Model (weights) |
| **2 (fallback)** | **parakeet-mlx** (`mlx-community/parakeet-tdt-0.6b-v3` or `-v2`) | Mature, easy `transcribe_stream()`, Apache-2.0 tooling, ~24–117× RT on M4; great non-streaming fallback and good streaming with small chunks | ~150–500 ms (chunk-dependent) | v2: 1.69% LS-clean / 3.19% other; v3: 1.93% / 3.59% | ~0.9–2 GB | code Apache-2.0; weights CC-BY-4.0 |
| 3 | **Moonshine streaming-tiny/base** (`UsefulSensors/moonshine-streaming-tiny`) | Lowest theoretical latency (80 ms lookahead), tiny memory; accuracy weaker | ~80 ms lookahead | 4.49% LS-clean, 12.09% other | <1 GB | MIT |
| 4 | **WhisperKit (Argmax) large-v3-turbo, d750** | Best Whisper-family streaming on Apple, ANE/CoreML; but ~0.45 s partial — over budget | ~0.45 s hypothesis | 2.2–2.3% LS-clean | ~0.4 GB | MIT (WhisperKit) |
| — | **Kyutai STT** (`kyutai/stt-1b-en_fr` / `stt-2.6b-en-mlx`) | Genuinely streaming + semantic VAD, but 1B delay is 500 ms and 2.6B is 2.5 s | 500 ms / 2500 ms | n/a published here | 1B ~2–4 GB; 2.6B larger | CC-BY-4.0 |

**Bottom line:** Adopt **Nemotron-ASR-streaming-0.6B (MLX)** as primary for ≤150 ms partials. Keep **parakeet-mlx** as the safe fallback (most mature MLX ASR tooling, you can run it streaming with small chunks or as a high-accuracy batch finalizer). The existing **mlx-whisper / faster-whisper** skills are fine for offline/file transcription but cannot hit a 150 ms partial.

> Important nuance: many headline RTF numbers (Qwen3-ASR 0.015, Parakeet 3000+ RTFx) are **batch/file throughput**, NOT streaming first-partial latency. A model can be 100× real-time in batch and still have 500 ms+ user-perceived partial latency if it buffers. For voice-to-voice, optimize the *streaming* latency column, not RTF.

---

## Why architecture decides this

Two families:

1. **Chunked autoregressive (Whisper family).** Whisper is non-streaming by design (30 s windows, encoder-decoder). "Streaming" is emulated by feeding overlapping buffers and reconciling with a `LocalAgreement` policy (WhisperKit) or sliding windows. Even the best Apple-optimized variant (WhisperKit large-v3-turbo) lands at **~0.45 s** per-word hypothesis latency on M3 Max — that is *finalization-ish*, not a <150 ms partial. Good accuracy, wrong latency class for our target.

2. **Cache-aware streaming encoders (Conformer/FastConformer + RNNT/TDT/CTC).** Each audio frame is encoded **exactly once**; state is carried in fixed-size ring buffers. Chunk size controls *when* the model sees context, not *how much*. These emit partials every chunk (configurable down to ~80 ms). This is the only family that natively meets <150 ms partials. Parakeet, Nemotron-ASR, Moonshine-streaming, and Kyutai all live here.

For a 500 ms voice-to-voice budget shared with LLM + TTS, STT realistically gets ~100–200 ms of latency budget for the first usable partial and must finalize fast on end-of-speech. That points squarely at family (2).

---

## Detailed comparison

### NVIDIA Parakeet (TDT 0.6B v2 / v3) and parakeet-mlx

- **Repo IDs:**
  - Upstream weights: `nvidia/parakeet-tdt-0.6b-v2` (English-only), `nvidia/parakeet-tdt-0.6b-v3` (25 EU languages incl. English)
  - MLX: `mlx-community/parakeet-tdt-0.6b-v3` (default in parakeet-mlx), plus the `mlx-community/parakeet` collection (v2/v3, INT8/bf16 variants)
  - Tooling: `senstella/parakeet-mlx` (PyPI `parakeet-mlx`), Apache-2.0
- **Architecture:** FastConformer encoder + TDT (Token-and-Duration Transducer) decoder. Also CTC/RNNT variants supported by parakeet-mlx (`ParakeetTDT/RNNT/CTC/TDTCTC`).
- **English WER:** v2 = **1.69%** LS test-clean / **3.19%** test-other / **6.05%** Open-ASR avg. v3 = **1.93%** / **3.59%** / **6.34%** avg. (v2 is *better* for English-only — prefer v2 since the project is English-only.)
- **Speed on Apple Silicon:** batch RTFx ~3380 on datacenter GPU; on Mac, ~**0.042 RTF (~24× RT) on M4**, reported up to **117× RT** for English-only in third-party Apple-Silicon benches; in one M4 large-model shootout parakeet-mlx averaged **0.50 s** vs mlx-whisper 1.02 s.
- **Streaming:** parakeet-mlx exposes `transcribe_stream()` with `(left_context, right_context)` attention frames and `transcriber.add_audio(chunk)` + live results. The base v2/v3 TDT checkpoints were trained full-attention (24-min single pass), so streaming with these is "buffered streaming" — partial latency depends on your chunk (commonly ~5 s default; can be reduced, but small chunks hurt accuracy because these aren't cache-aware checkpoints). For true low-latency, use the dedicated streaming/cache-aware checkpoints below.
- **Memory:** ~0.9 GB (INT8) to ~2 GB (bf16); needs ≥2 GB RAM. Comfortably within budget.
- **License:** code Apache-2.0; **weights CC-BY-4.0** (attribution; commercially usable).
- **Verdict:** Best **fallback** and best **high-accuracy batch finalizer**. Most mature MLX ASR path. As a pure <150 ms streamer it's second-best because the v2/v3 checkpoints aren't cache-aware.

### NVIDIA Nemotron-ASR streaming 0.6B (cache-aware) + MLX port  ← RECOMMENDED

- **Repo IDs:**
  - Upstream: `nvidia/nemotron-speech-streaming-en-0.6b` (cache-aware streaming FastConformer + RNN-T, English; part of "Nemotron 3.5 ASR" family, multilingual variants exist for 40 langs)
  - MLX port: `dboris/nemotron-asr-mlx` (GitHub `199-biotechnologies/nemotron-asr-mlx`), Apache-2.0 code, Python 3.10+, deps MLX + huggingface-hub + numpy (+ sounddevice for live mic, websockets for demo)
- **Architecture:** Cache-aware streaming Conformer/FastConformer (8× downsampling), each frame encoded once, fixed-size ring-buffer state. This is the canonical low-latency streaming ASR architecture.
- **English WER (MLX port, measured):** **2.70%** LS test-clean, **5.52%** LS test-other.
- **Speed on Apple Silicon:** **112× real-time on M4 Max** (full 5.4 h LS test-clean in 173 s) after v0.2.0 mel-frontend/decoder fixes (was 76×). Constant memory streaming.
- **Latency:** cache-aware design targets **~80–160 ms** chunk latency (tunable via chunk size; chunk size affects *when* context is seen, not accuracy of context). Meets the <150 ms partial target.
- **Memory:** ~**3.4 GB** peak (MLX port). Fits 24 GB target with headroom for LLM+TTS if managed.
- **License:** MLX port code Apache-2.0; NVIDIA weights under NVIDIA Open Model License (commercially usable; review terms).
- **Caveats:** MLX port is newer/smaller-community than parakeet-mlx (verify build, mic demo, and exact streaming-partial latency on your M3 before committing). Note the MLX port README states it "runs in batch mode" while using the cache-aware architecture — confirm true incremental streaming behavior in `stt_from_mic` path.
- **Verdict:** Best fit for the stated goal: MLX-native, cache-aware streaming, strong WER, sane memory.

### Related NVIDIA streaming checkpoints worth knowing

- `nvidia/parakeet_realtime_eou_120m-v1` — **120M** FastConformer-RNNT, **80–160 ms** latency, emits an `<EOU>` end-of-utterance token (EOU detection p50 **160 ms**, p90 280 ms, p95 320 ms). WER avg 9.30% (LS-clean **3.61%**). English-only, no punctuation/caps. NVIDIA Open Model License. **Very interesting for voice-to-voice turn-taking** — the EOU token can drive your endpointing/barge-in logic and shave finalization latency. Consider running it alongside a higher-accuracy model, or evaluate standalone if 120M WER is acceptable for short commands.
- `nvidia/parakeet-unified-en-0.6b`, `nvidia/multitalker-parakeet-streaming-0.6b-v1` — other 2025–26 streaming Parakeet variants (unified, multi-speaker). Not needed for single-speaker robot use.
- No official MLX port of the EOU model found yet (2026-06); would need porting or run via NeMo/ONNX on MPS.

### Whisper family on Apple Silicon

- **whisper-large-v3-turbo** (`openai/whisper-large-v3-turbo`, MLX: `mlx-community/whisper-large-v3-turbo`): 809M params (4-layer decoder), ~2.2–2.5% WER LS-clean, 8×+ RT. Non-streaming natively.
- **distil-whisper** (`distil-whisper/distil-large-v3`, English-focused): ~756M, within ~1% WER of large-v3, ~5–6× faster than large-v3. English-only. MIT. Still non-streaming.
- **mlx-whisper** (`mlx-community/*`, e.g. `mlx-community/whisper-large-v3-turbo`): GPU on Apple Silicon. ~1.02 s for "large" in the M4 shootout. **Project already has an mlx-whisper skill.** Good for file/offline.
- **whisper.cpp** (CoreML): ~1.23 s in the same M4 shootout. Portable C++.
- **faster-whisper** (CTranslate2): ~6.96 s on M4 in that shootout (CPU-bound on Mac; far better on CUDA). **Project already has a faster-whisper skill.** Use for CPU fallback, not Mac low-latency.
- **WhisperKit (Argmax)** (`argmaxinc/whisperkit-coreml`; Swift): ANE/CoreML. Streaming via `LocalAgreement` over 15 s block-diagonal-attention chunks (d750). On **M3 Max** with **large-v3-turbo**: **~0.45 s** per-word hypothesis latency, **2.2%** WER (2.30% for d750-compressed). Best Whisper streaming on Apple, but **0.45 s partial is ~3× over our 150 ms target**, and it's Swift-first (Python integration is awkward for this app).
- **Verdict:** Whisper family is the wrong latency class for <150 ms partials. Keep mlx-whisper (existing skill) as an offline/high-accuracy re-transcription option only.

### Moonshine (Useful Sensors)

- **Repo IDs:** `UsefulSensors/moonshine-streaming-tiny` (**34M**), `UsefulSensors/moonshine-streaming-medium`; non-streaming `UsefulSensors/moonshine/{tiny,base}`. Moonshine v2 ("Ergodic Streaming Encoder") announced 2026.
- **Architecture:** ~50 Hz frontend + sliding-window Transformer encoder; streaming-tiny uses **80 ms lookahead**, contexts (16,4)/(16,0). RTFx 847.
- **WER (streaming-tiny):** **4.49%** LS-clean, **12.09%** LS-other, 12.01% Open-ASR avg — clearly behind Parakeet/Nemotron.
- **Memory:** sub-1 GB budgets (smallest model ~27–44 MB). Runtimes: ONNX, Transformers (HF note: "does not yet implement fully efficient streaming"), Keras/torch; no first-class MLX yet.
- **License:** MIT (most permissive here).
- **Verdict:** Lowest theoretical latency and footprint, MIT license — attractive for the *absolute* tightest latency or a tiny always-on wake/command path, but English WER is notably worse than Parakeet/Nemotron, and MLX support is weak. Good niche/last-resort streamer.

### Kyutai STT (delayed-streams-modeling)

- **Repo IDs:** `kyutai/stt-1b-en_fr` (1B, EN/FR, **500 ms delay**), `kyutai/stt-2.6b-en` (2.6B, English, **2.5 s delay**), MLX: `kyutai/stt-2.6b-en-mlx`, Rust: `kyutai/stt-2.6b-en-candle`.
- **Strengths:** genuinely streaming; **semantic VAD** that predicts end-of-turn from content+intonation (currently only in the Rust server) — excellent for conversational turn-taking; word-level timestamps + punctuation. Runs on Mac via `moshi-mlx` (`uvx --with moshi-mlx python scripts/stt_from_mic_mlx.py`); 1B tested on iPhone 16 Pro.
- **Latency:** 1B = **500 ms** built-in delay, 2.6B = **2.5 s**. The 500 ms 1B is over our 150 ms partial target; 2.6B is far over.
- **License:** weights CC-BY-4.0.
- **Verdict:** Conceptually ideal for voice agents (streaming + semantic endpointing) but the published delays exceed our partial-latency budget, and the English-best model (2.6B) is too slow. Watch the semantic-VAD idea — replicate the EOU concept via Nemotron's `<EOU>` model instead.

### 2025–2026 newcomers

- **Qwen3-ASR (MLX)** — `mlx-community/Qwen3-ASR-1.7B-8bit` / `-bf16`, `mlx-community/Qwen3-ASR-0.6B-*`; tooling `qwen3-asr-mlx` (PyPI), `Blaizzy/mlx-audio`, Apache-2.0, Python 3.10–3.13, MLX 0.31+. **Best accuracy on Apple Silicon**: 1.52–1.82% WER (8-bit), RTF 0.012–0.033 (≈30–80× RT) on M5 Pro, 1.0–2.7 GB. **But this is batch/file decoding** — no native low-latency streaming partials documented. Use as an offline accuracy champion, not a <150 ms streamer.
- **Omnilingual CTC 300M (MLX)** — `mlx-community` 4-bit, ~222× RT, 384 MB, 1672 languages, ~4.26% WER. Throughput leader; multilingual; CTC could stream but not the English accuracy leader.
- **Canary-1B-v2** (`nvidia/canary-1b-v2`) — strong multilingual ASR/AST, larger and not streaming-first.
- **WhisperRT** (arXiv 2508.12301) — research: turns Whisper into a causal streaming model. Not productionized for Apple.

---

## Soniqo Apple-Silicon benchmark (M5 Pro, 48 GB, LibriSpeech test-clean) — for cross-reference

| Model | WER % | RTF | Peak RSS |
|---|---|---|---|
| Qwen3-ASR 1.7B MLX 8-bit | 1.52 | 0.033 | 2.7 GB |
| WhisperKit large-v3-turbo FP16 | 1.71 | 0.084 | 0.4 GB |
| Qwen3-ASR 0.6B MLX 8-bit | 1.82 | 0.015 | 1.3 GB |
| Qwen3-ASR 0.6B MLX 4-bit | 2.20 | 0.012 | 1.0 GB |
| Parakeet TDT v3 INT8 | 2.37 | 0.009 | 0.9 GB |
| Nemotron Streaming INT8 | 2.82 | 0.058 | 961 MB |
| Omnilingual CTC 300M MLX 4-bit | 4.26 | 0.005 | 0.4 GB |

(These are **batch RTF**, not streaming partial latency — see caveat above. M5 Pro is faster than the M3 target; expect roughly 1.5–2× higher RTF / latency on M3.)

---

## Recommendation for ReachyChatOffline

1. **Primary (streaming, <150 ms partial):** **Nemotron-ASR streaming 0.6B via MLX** — `dboris/nemotron-asr-mlx` (src `nvidia/nemotron-speech-streaming-en-0.6b`). Cache-aware, ~80–160 ms chunk latency, 2.70% LS-clean, ~3.4 GB, Apache-2.0 code. **Validate true incremental streaming + measured first-partial latency on the actual M3** before locking in (the port advertises batch mode).
2. **Fallback / high-accuracy finalizer:** **parakeet-mlx** with `mlx-community/parakeet-tdt-0.6b-v2` (English-only, 1.69% WER). Most mature MLX ASR tooling; run streaming with small chunks, or as a batch finalizer on the captured utterance for best-quality final transcript.
3. **Turn-taking / endpointing:** evaluate `nvidia/parakeet_realtime_eou_120m-v1` for its `<EOU>` token (p50 160 ms EOU) to drive low-latency end-of-turn detection and barge-in, complementing the primary STT. (No MLX port yet — port or run via NeMo/ONNX-MPS.)
4. **Offline/file transcription:** keep the existing **mlx-whisper** skill (`mlx-community/whisper-large-v3-turbo`); optionally **Qwen3-ASR 0.6B MLX 8-bit** (`mlx-community/Qwen3-ASR-0.6B-8bit`) for best-accuracy non-real-time passes.
5. **De-prioritize for this target:** WhisperKit (0.45 s partial, Swift), Kyutai (500 ms+ delay), faster-whisper on Mac (CPU-slow). Moonshine streaming-tiny only if you need the absolute smallest footprint and can accept higher WER.

**Memory sanity (24 GB / ≲20 GB working set):** Nemotron MLX ~3.4 GB or Parakeet ~1–2 GB leaves ample room to co-host the LLM and TTS. Prefer INT8/bf16 quantized variants.

---

## Sources

- senstella/parakeet-mlx — https://github.com/senstella/parakeet-mlx
- parakeet-mlx (PyPI) — https://pypi.org/project/parakeet-mlx/
- nvidia/parakeet-tdt-0.6b-v2 — https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2
- nvidia/parakeet-tdt-0.6b-v3 — https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
- Canary-1B-v2 & Parakeet-TDT-0.6B-v3 paper — https://arxiv.org/html/2509.14128v1
- nemotron-asr-mlx (MLX port) — https://github.com/199-biotechnologies/nemotron-asr-mlx
- NVIDIA Nemotron cache-aware streaming ASR — https://huggingface.co/blog/nvidia/nemotron-speech-asr-scaling-voice-agents
- nvidia/nemotron-speech-streaming-en-0.6b — https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b
- nvidia/parakeet_realtime_eou_120m-v1 — https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1
- WhisperKit paper (arXiv 2507.10860) — https://arxiv.org/html/2507.10860v1 / https://arxiv.org/abs/2507.10860
- WhisperKit (Argmax) overview — https://whipscribe.com/tools/whisperkit
- mac-whisper-speedtest — https://github.com/anvanvan/mac-whisper-speedtest
- Soniqo Apple Silicon speech benchmarks — https://soniqo.audio/benchmarks
- Qwen3-ASR MLX (moona3k) — https://github.com/moona3k/mlx-qwen3-asr/
- mlx-community/Qwen3-ASR-1.7B-8bit — https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-8bit
- Blaizzy/mlx-audio — https://github.com/Blaizzy/mlx-audio
- UsefulSensors/moonshine-streaming-tiny — https://huggingface.co/UsefulSensors/moonshine-streaming-tiny
- Moonshine paper — https://arxiv.org/pdf/2410.15608
- Moonshine v2 paper — https://arxiv.org/html/2602.12241
- Kyutai delayed-streams-modeling — https://github.com/kyutai-labs/delayed-streams-modeling
- Kyutai STT — https://kyutai.org/stt
- Kyutai on macOS (Anil Madhavapeddy) — https://anil.recoil.org/notes/kyutai-streaming-voice-mlx
- Best open-source STT 2026 (Northflank) — https://northflank.com/blog/best-open-source-speech-to-text-stt-model-in-2026-benchmarks
- Open ASR Leaderboard (HF) — https://huggingface.co/blog/open-asr-leaderboard
