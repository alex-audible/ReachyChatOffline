# 01 — LLM / Reasoning-Core & Audio-Native Multimodal Options

**Project:** ReachyChatOffline — fully-local, low-latency, English-only voice-to-voice for Reachy Mini
**Target deployment:** Apple Silicon M3 / 24 GB (working set ≲ 20 GB); dev box M3 Max / 96 GB
**macOS 15, Python 3.10–3.12, MLX-first (MPS fallback)**
**Latency budget:** ≤ 500 ms voice-to-voice (≤ 600 ms with vision)
**Research date:** 2026-06-17 — all claims recency-checked against sources published Apr–Jun 2026.

---

## 0. TL;DR / Recommendation

1. **"Gemma 4" is real.** The user was correct, not confused. Google released **Gemma 4** on **2 April 2026** (Apache-2.0), with edge models **E2B / E4B**, an MoE **26B-A4B**, a dense **31B**, and — added **3 June 2026** — a unified dense **12B**. "Gemma 4 12B QAT" and "Gemma 4 E2B/E4B" both exist. Gemma 3 (1B/4B/12B/27B + QAT int4) and Gemma 3n (E2B/E4B MatFormer) also still exist but are the **previous** generation.

2. **Do NOT collapse STT into the LLM for the production path.** Gemma 4 E2B/E4B *can* take raw audio natively (ASR / translation / understanding, 16 kHz mono, ≤ 30 s), but:
   - **WER is not Whisper-grade** — E4B ≈ 3.05 % on LibriSpeech-clean (beats `whisper-base.en`) but blows up on noisy/spontaneous speech (AMI ≈ 19 %, vs Whisper-large-v3 ≈ 16 %), and the audio path hard-caps at **30 s**, with a non-trivial refusal/hallucination rate. A robot in a room is a noisy-speech scenario.
   - The MLX audio path was **broken until early April 2026** (gibberish output) and is still fragile (see §2.4).
   - Audio is **25 tokens/s of audio** in Gemma 4 (4× the prefill cost of Gemma 3n's 6.25 tok/s), which *hurts* TTFT — the opposite of latency-first.

   → **Keep STT separate** (a dedicated streaming ASR such as `mlx-community/whisper-large-v3-turbo` or Parakeet-MLX), feed text into a small fast MLX LLM. This is faster, more robust, and lets you stream partials.

3. **Core reasoning LLM (cascaded): the QAT int4 Gemma 4 E4B — `mlx-community/gemma-4-E4B-it-qat-4bit`** (≈ 57 tok/s, ~3 GB resident, bf16-grade quality from QAT) as the primary, with **`mlx-community/Qwen3.5-4B-MLX-4bit`** as the alternate if you want stronger reasoning/tool-calling. Both are English-strong and leave headroom for ASR+TTS+vision inside 20 GB. **Prefer the QAT build over the plain `-4bit` post-training quant** — QAT recovers most of the int4 quality loss for free at the same memory/speed (see §1b).

4. **Vision: the same `gemma-4-E4B-it-qat-4bit` (image input) via mlx-vlm.** One weight set covers reasoning + text **and** webcam frames. A single 768px frame adds ≈ 256 image tokens to prefill — well inside the +100 ms vision budget.

5. **Keep audio-native Gemma 4 E4B QAT as an experimental "single-model" fallback** (audio+image+text → text) for the quiet-room demo case, but it is not the production STT.

---

## 1. The Gemma lineup as of mid-2026 (recency-checked)

| Family | Released | Sizes | Multimodal in | License | Notes |
|---|---|---|---|---|---|
| Gemma 1 | 2024-02-21 | 2B, 7B | text | Gemma TOU | — |
| Gemma 2 | 2024-06-27 | 2B, 9B, 27B | text | Gemma TOU | — |
| Gemma 3 | 2025-03-12 | 1B, 4B, 12B, 27B | text + **image** | Gemma TOU (source-available) | QAT int4 added 2025-09-04 |
| Gemma 3n | 2025-06 | **E2B, E4B** (MatFormer) | text + image + **audio** + video | Gemma TOU | First Gemma with audio; 6.25 audio-tok/s |
| **Gemma 4** | **2026-04-02** | **E2B, E4B, 26B-A4B (MoE, ~3.8B active), 31B dense** | text + image + video; **audio on E2B/E4B** | **Apache-2.0** | New license. 140+ langs, 128K (edge)/256K (large) ctx |
| **Gemma 4 12B "Unified"** | **2026-06-03** | **12B dense** | text + image + video + **audio** | Apache-2.0 | Encoder-free; raw audio/image flow straight into decoder. 256K ctx |

**Verdict on the user's terms:**
- **"Gemma 4 12B QAT"** → exists: `google/gemma-4-12B-it-qat-q4_0-gguf` (and the 12B Unified added June 3). ✔
- **"Gemma 4 E2B / E4B"** → exists: `google/gemma-4-E2B-it`, `google/gemma-4-E4B-it`. These are the Gemma 4 *edge* models — the successors to Gemma 3n's E2B/E4B, NOT the same artifacts. ✔
- Gemma 3 (1B/4B/12B/27B + QAT int4) and Gemma 3n (E2B/E4B) are the prior generation and are not deprecated, but Gemma 4 supersedes them for new work.

### Exact repo IDs (Gemma 4)

**Instruction-tuned (bf16, source for conversion / mlx-vlm):**
`google/gemma-4-E2B-it`, `google/gemma-4-E4B-it`, `google/gemma-4-12B-it`, `google/gemma-4-26B-A4B-it`, `google/gemma-4-31B-it`

**Official QAT int4 GGUF (Q4_0)** — every size has one (verified via HF API 2026-06-17):
`google/gemma-4-E2B-it-qat-q4_0-gguf`, `google/gemma-4-E4B-it-qat-q4_0-gguf` (file 5.15 GB), `google/gemma-4-12B-it-qat-q4_0-gguf`, `google/gemma-4-26B-A4B-it-qat-q4_0-gguf`, `google/gemma-4-31B-it-qat-q4_0-gguf`
(+ `…-qat-q4_0-unquantized` half-precision QAT checkpoints for custom compile; `…-qat-mobile-transformers` / `…-qat-mobile-ct` / `…-qat-w4a16-ct` deployment-targeted builds; `litert-community/gemma-4-E4B-it-litert-lm` for LiteRT/on-device.)

**MLX QAT ports (recommended) — `mlx-community`, verified to exist:**
- `mlx-community/gemma-4-E4B-it-qat-4bit` ← **primary pick**
- `mlx-community/gemma-4-E2B-it-qat-4bit` (faster, smaller)
- `mlx-community/gemma-4-12B-it-qat-4bit`
- `mlx-community/gemma-4-26B-A4B-it-qat-4bit`, `mlx-community/gemma-4-31B-it-qat-4bit`
- Higher-fidelity QAT options: `…-qat-6bit`, `…-qat-8bit`, `…-qat-mxfp4` (Apple-friendly fp4), `…-qat-bf16`, and OptiQ mixed-precision `mlx-community/gemma-4-e4b-it-qat-OptiQ-4bit` / `mlx-community/gemma-4-12B-it-qat-OptiQ-4bit`.

**MLX plain (post-training quant) — community:**
`mlx-community/gemma-4-e4b-it-4bit`, `mlx-community/gemma-4-e2b-4bit`, `lmstudio-community/gemma-4-E4B-it-MLX-4bit`, `lmstudio-community/gemma-4-E4B-it-MLX-8bit`, `unsloth/gemma-4-E4B-it-UD-MLX-4bit`. Text-only LM-head builds also exist (`…-text-…`, `…-lm-…`) if you don't need vision/audio.
⚠ **Audio caveat** — see §2.4 before trusting *any* MLX quant for audio input (vision/text are fine).

Sources: [HF API gemma-4 query], Google blog [gemma-4](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/), [Gemma releases](https://ai.google.dev/gemma/docs/releases), [Wikipedia: Gemma](https://en.wikipedia.org/wiki/Gemma_(language_model)), [Gemma 3 QAT blog](https://developers.googleblog.com/en/gemma-3-quantized-aware-trained-state-of-the-art-ai-to-consumer-gpus/), [HF google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it), [HF google/gemma-4-E4B-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf), [Gemma 4 12B Unified guide](https://www.buildfastwithai.com/blogs/gemma-4-12b-guide).

---

## 1b. Gemma 4 QAT — the centerpiece (deep dive)

**What QAT is:** Google fine-tunes the model *while simulating int4* so the int4 deployment checkpoint keeps ~bf16 quality at 4× smaller memory — confirmed on the official card: QAT "preserves similar quality to bfloat16 while dramatically reducing the memory requirements." For a latency-first 24 GB box this is the ideal default: full int4 speed/footprint with minimal quality loss.

**Variants released (all on 2026-04-02, 12B on 2026-06-03), each with an official QAT int4 build:**

| Variant | Eff. / total params | Modalities IN | Audio? | Ctx | QAT int4 file size | Fits 24 GB M3? | Role |
|---|---|---|---|---|---|---|---|
| **E2B QAT** | ~2.3B / ~5B | text + image + **audio** | ✔ | 128K | ~3 GB | ✔ comfortably | Fastest edge model |
| **E4B QAT** | ~4.5B / ~8B | text + image + **audio** | ✔ | 128K | **5.15 GB** (GGUF) / ~3 GB resident MLX 4-bit | ✔ comfortably | **★ our pick** |
| **12B QAT** | 12B dense | text + image + **audio** (Unified) | ✔ | 256K | ~8 GB | ✔ (tighter) | Most capable trimodal that still fits |
| 26B-A4B QAT | 26B MoE / 3.8B active | text + image (no audio) | ✘ | 256K | ~16 GB | borderline; ~2 tok/s on M4 Pro (memory pressure) | too slow for latency |
| 31B QAT | 31B dense | text + image (no audio) | ✘ | 256K | ~18 GB | no headroom; ~5 tok/s | not viable |

Key per-variant facts for us:
- **Audio input is native on E2B / E4B / 12B only** — the 26B and 31B are text+image. So an "audio-native + vision + low-latency on 24 GB" model is necessarily **E2B, E4B, or 12B**.
- **License: Apache-2.0** across the board (a real improvement over Gemma 3/3n's source-available Gemma Terms — matters if you ship the robot commercially).
- **MLX & MPS:** MLX via `mlx-lm` (text) / `mlx-vlm` (vision+audio); MPS via HF `transformers` (slower, more fragile on Apple — prefer MLX, consistent with the project's MLX-first stance and known PyTorch-MPS limitations).
- **Memory at QAT int4 on M3:** E4B ≈ 3 GB resident MLX 4-bit (GGUF artifact 5.15 GB); E2B ≈ 2–3 GB; 12B ≈ 8 GB. All inside the 20 GB working-set ceiling with room for ASR+TTS.
- **Speed (M4 Pro / 24 GB, community):** E2B ~95 tok/s, E4B ~57 tok/s decode; E2B ~158 tok/s on M5 Max. M3 (lower bandwidth) will be proportionally slower (~0.5–0.7×), but a 4B at ~35–55 tok/s on M3 is fine for conversational turns.
- **TTFT:** MLX does full prefill before the first token, so TTFT ∝ prompt length. With a short system prompt + trimmed history, TTFT for E4B int4 is tens of ms on M-series; adding one 768px webcam frame (~256 tokens) adds well under 100 ms.

### Gemma 4 QAT vs Gemma 3n (E2B/E4B) — head-to-head for our use case

| Criterion | **Gemma 4 E4B QAT** | Gemma 3n E4B |
|---|---|---|
| Release | 2026-04-02 (current gen) | 2025-06 (prior gen) |
| License | **Apache-2.0** | Gemma Terms (source-available) |
| Audio input | ✔ native, ~300M encoder (50% smaller, 40 ms frame) | ✔ native, USM conformer |
| Audio token cost | 25 tok/s of audio (heavier prefill) | **6.25 tok/s of audio** (lighter prefill) |
| ASR quality (LS-clean) | **~3.05 % WER** | worse (older encoder) |
| Vision input | ✔ SigLIP2 | ✔ (MobileNet-style) |
| Reasoning / instruction-following | **stronger** (built from Gemini-3-era research, configurable thinking) | weaker |
| MLX support | ✔ mature, QAT + many quants in `mlx-community` | ✔ (`mlx-community/gemma-3n-E4B-it-4bit`, incl. `-text-4bit-dwq`) |
| Memory (int4, M3) | ~3 GB | ~3 GB |
| Verdict | **Use this.** Newer, better reasoning + ASR, permissive license. | Only consider if the 6.25 audio-tok/s prefill advantage matters for very long audio-native prompts — but we're keeping STT separate anyway. |

**Bottom line on QAT:** make **`mlx-community/gemma-4-E4B-it-qat-4bit`** the single multimodal reasoning core (text + webcam image), preferring the QAT build over the plain `-4bit` PTQ build for free quality. Gemma 4 QAT cleanly supersedes Gemma 3n for this project.

---

## 2. Gemma 3n / Gemma 4 audio-native capability

### 2.1 Does it take raw audio natively? Yes.
Both Gemma 3n (E2B/E4B) and Gemma 4 (E2B/E4B + 12B Unified) accept **raw audio as native input** and can do **ASR, automatic speech translation (AST), and general audio understanding** — i.e. audio in → text answer out, collapsing STT+LLM into one model. Output is **text only** (no speech output from any Gemma).

### 2.2 Encoder, format, length, languages

| Property | Gemma 3n | Gemma 4 (E2B/E4B/12B) |
|---|---|---|
| Audio encoder | USM-style conformer | ~300M-param encoder (Gemma 4 edge encoder is **50 % smaller** than 3n's, **40 ms** frame) |
| Audio token cost | **6.25 tokens / s** of audio | **25 tokens / s** of audio |
| Max audio length | 30 s | **30 s** |
| Input format | 16 kHz, mono, 32-bit float [-1,1] | same (resample with `scipy.signal.resample`) |
| Tasks | ASR / AST / understanding | ASR / AST / understanding |
| Languages | multilingual | 35+ out-of-box, pre-trained on 140+ |

> Note the regression for latency: Gemma 4 uses **25 audio-tok/s** vs Gemma 3n's 6.25 — a 10 s utterance is 250 prefill tokens on Gemma 4 vs 63 on Gemma 3n. Better transcription, worse TTFT.

### 2.3 Quality (WER) — the deciding numbers
From an independent Open-ASR-Leaderboard-style benchmark (2026-06-04):

| Dataset | Gemma 4 E2B | Gemma 4 E4B | Gemma 4 12B | Whisper-large-v3 |
|---|---|---|---|---|
| LibriSpeech clean | 3.70 % | **3.05 %** | 3.85 % | 2.01 % |
| LibriSpeech other | 8.69 % | 7.61 % | 14.75 % | 3.91 % |
| AMI (meeting/noisy) | 20.81 % | 18.95 % | **104 %** | 15.95 % |
| Earnings22 | 15.44 % | 13.98 % | 50.40 % | 11.29 % |
| GigaSpeech | 11.81 % | 11.11 % | 23.91 % | 10.02 % |

- E4B beats `whisper-base.en` (4.25 %) on clean read speech, **but trails Whisper-large-v3 everywhere** and **degrades badly on noisy/spontaneous speech** (the robot-in-a-room case).
- The **12B Unified is *worse* at ASR** than E4B (104 % WER on AMI, 7.4 % refusal rate) — do not pick 12B for transcription.
- Benchmark author's verdict: **"Don't swap it in for a dedicated ASR model."**

### 2.4 Running on MLX + latency/memory (M-series)
- Tooling: **`mlx-vlm` ≥ 0.4.3** (Day-0 Gemma 4 support, vision+audio). CLI: `python -m mlx_vlm.generate --model google/gemma-4-e4b-it --audio clip.wav --prompt "Transcribe this audio" --max-tokens 500`.
- ⚠ **Known audio bug (mlx-vlm #903):** all early mlx-community Gemma 4 E2B/E4B quantizations produced **gibberish** on audio — two causes: (1) missing `feature_extractor` in `processor_config.json`, (2) audio-tower embedding propagation. **Closed via PR #931 (2026-04-03).** Use mlx-vlm ≥ the post-#931 release and verify the model's `processor_config.json` carries the feature extractor (`sampling_rate: 16000`, `feature_size: 128`, `audio_ms_per_token: 40`). Some community quants reportedly still mishandle PLE (per-layer-embedding) quantization — `github.com/FakeRocket543/mlx-gemma4` packages validated trimodal quants. **Vision and text were never affected.**
- Latency on M4 Pro / 24 GB (community): **E4B English transcription ≈ 1.0 s** for a short clip (French 1.6 s, Arabic 6.0 s). This is end-to-end including prefill — i.e. ~1 s just for STT, which alone nearly eats the 500 ms budget. A streaming Whisper-turbo is faster for short partials.
- Memory: E2B/E4B 4-bit ≈ **5 GB** (16-bit ≈ 15 GB).

Sources: [Gemma audio docs](https://ai.google.dev/gemma/docs/capabilities/audio), [mlx-vlm Gemma 4 README](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/gemma4/README.md), [mlx-vlm issue #903](https://github.com/Blaizzy/mlx-vlm/issues/903), [ASR benchmark (twango.dev)](https://twango.dev/writing/gemma4-asr-benchmark), [local Mac benchmark (kartit.net)](https://kartit.net/blog/gemma4-local-benchmark.html), [transcribing with Gemma 3n](https://www.gemma-3n.net/blog/transcribing-speech-with-gemma-3n/).

---

## 3. Vision / image input

| Model | Image input | Vision module | Repo (MLX) |
|---|---|---|---|
| Gemma 3 4B/12B/27B (+QAT) | ✔ | SigLIP encoder | `mlx-community/gemma-3-4b-it-qat-4bit` etc. |
| Gemma 3n E2B/E4B | ✔ | MobileNet-style | mlx-vlm |
| **Gemma 4 E2B/E4B** | ✔ | SigLIP2 | `mlx-community/gemma-4-e4b-it-4bit` |
| Gemma 4 12B Unified | ✔ | encoder-free, ~35M patch→token module | `google/gemma-4-12B-it` via mlx-vlm |
| Gemma 4 26B/31B | ✔ (no audio) | SigLIP2 | too big / too slow on M3 |

- **Recommended: reuse `gemma-4-E4B-it` for vision** — one model covers reasoning + image, so no extra resident weights.
- Cost of one webcam frame: a 768×768 image is ≈ **256 image tokens** of prefill. On E4B at ~57 tok/s decode and fast prefill this adds well under 100 ms — inside the ≤ 600 ms vision budget. Memory unchanged (same 4-bit weights, ~3 GB; transient image tensors are small).
- For higher-fidelity OCR/charts, Gemma 4 12B or Qwen3.5-VL are stronger but heavier; not needed for "what am I looking at" robot vision.

Sources: [Gemma core docs](https://ai.google.dev/gemma/docs/core), [Gemma 4 12B Unified guide](https://www.buildfastwithai.com/blogs/gemma-4-12b-guide), [mlx-vlm](https://pypi.org/project/mlx-vlm/).

---

## 4. Alternatives — multimodal low-latency local assistants on Apple Silicon

| Model | Repo ID | In | Out | Params | MLX | License | Released | Notes for us |
|---|---|---|---|---|---|---|---|---|
| **Qwen3-Omni-30B-A3B-Instruct** | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | text+audio+image+video | **text + speech** | 30B MoE (3B active) | partial (mlx-vlm vision; talker/audio-out not first-class on MLX) | Apache-2.0 | 2025-09-22 | Only listed option with **native speech OUT** (real-time Thinker-Talker, 10 langs out / 18 in). 30B too big for 24 GB at good quant; better on M3 Max. 32K ctx. |
| Qwen3-Omni-30B-A3B-Thinking | `Qwen/Qwen3-Omni-30B-A3B-Thinking` | text+audio+image+video | text | 30B MoE | partial | Apache-2.0 | 2025-09-22 | Reasoning variant, text-only out. |
| **Phi-4-multimodal-instruct** | `microsoft/Phi-4-multimodal-instruct` | text+audio+image | text only | 5.6B | no first-class MLX quant found | MIT | 2025-02 | LoRA modality adapters; **#1 OpenASR ~6.14 % WER** at release — best ASR of the small multimodals, but **no speech out** and **no MLX path** yet (would run via MPS/transformers). Older (Feb 2025). |
| Gemma 4 E4B (audio-native) | `mlx-community/gemma-4-e4b-it-4bit` | text+audio+image | text only | 4B | ✔ (mlx-vlm) | Apache-2.0 | 2026-04 | Our combined-model candidate (see §2). No speech out. |

**Speech-OUTPUT note:** Among local options, **only Qwen3-Omni generates speech**. No Gemma and no Phi-4 emit audio. For ReachyChatOffline you will therefore need a **separate TTS** stage regardless of LLM choice (e.g. Kokoro-MLX / Qwen3-TTS-MLX via `mlx-audio`) — unless you adopt Qwen3-Omni, which is likely too heavy for the 24 GB target. So the realistic architecture is **cascaded ASR → LLM → TTS**, with audio-native Gemma as an optional shortcut for the STT+LLM half.

Sources: [Qwen3-Omni GitHub](https://github.com/QwenLM/Qwen3-Omni), [HF Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct), [mlx-audio](https://github.com/Blaizzy/mlx-audio), [Qwen on Apple Silicon (2026)](https://codersera.com/blog/apple-silicon-llms-complete-guide-2026/).

---

## 5. Best small fast English text LLM on MLX (cascaded reasoning stage)

| Model | Repo ID (MLX 4-bit) | ~Mem (4-bit) | tok/s (M-series) | Ctx | License | Released | Verdict |
|---|---|---|---|---|---|---|---|
| **Gemma 4 E4B (QAT)** | `mlx-community/gemma-4-E4B-it-qat-4bit` | ~3 GB | **~57 tok/s (M4 Pro)** | 128K | Apache-2.0 | 2026-04 | **Primary.** QAT = ~bf16 quality at int4. Same weights also do vision+audio → one model for the whole multimodal core. |
| Gemma 4 E2B (QAT) | `mlx-community/gemma-4-E2B-it-qat-4bit` | ~2 GB | **~95 tok/s (M4 Pro)**; ~158 tok/s (M5 Max) | 128K | Apache-2.0 | 2026-04 | Fastest; use if E4B latency is too high and quality acceptable. |
| Gemma 4 12B (QAT) | `mlx-community/gemma-4-12B-it-qat-4bit` | ~8 GB | slower (12B dense) | 256K | Apache-2.0 | 2026-06 | Most capable trimodal that still fits 24 GB; use on M3 Max dev box for quality headroom. |
| **Qwen3.5-4B** | `mlx-community/Qwen3.5-4B-MLX-4bit` (or `…OptiQ-4bit`) | ~3 GB | ~50–60 tok/s (M4 Pro est.) | long | Apache-2.0 | 2026-02 | **Alternate.** Stronger reasoning/tool-calling; verify MLX 0.25.2+ to avoid the early-2026 Qwen3.5 MLX latency regression. |
| Qwen3-4B | `Qwen/Qwen3-4B-MLX-4bit` | ~3 GB | ~50 tok/s | 32K+ | Apache-2.0 | 2025 | Stable predecessor to 3.5. |
| Gemma 3 4B QAT | `mlx-community/gemma-3-4b-it-qat-4bit` | ~3 GB | ~55–60 tok/s | 128K | Gemma TOU | 2025-03 | Solid, but older license + superseded by Gemma 4 E4B. |
| Llama 3.2 3B | `mlx-community/Llama-3.2-3B-Instruct-4bit` | ~2 GB | ~70–80 tok/s | 128K | Llama 3.2 | 2024-09 | Fast & lean but weaker reasoning than Gemma 4 / Qwen3.5; oldest option. |

**TTFT caveat (all MLX models):** MLX does a **full prefill before emitting any token**, so TTFT scales linearly with prompt length. Keep the system prompt short and the conversation window trimmed to protect the 500 ms budget. With a short prompt, TTFT for a 4B 4-bit model on M-series is on the order of tens of ms.

Sources: [llmcheck.net Apple Silicon benchmarks](https://llmcheck.net/benchmarks), [local Mac benchmark (kartit.net)](https://kartit.net/blog/gemma4-local-benchmark.html), [Qwen3.5 MLX guide](https://willitrunai.com/blog/qwen-3-5-mlx-apple-silicon-guide), [Qwen3→3.5 MLX latency note](https://medium.com/@aejaz.sheriff/from-qwen-3-to-qwen-3-5-on-apple-silicon-a-14x-latency-regression-and-how-mlx-got-us-back-0ed9ed21fa68), [HF Qwen3.5-4B-MLX-4bit](https://huggingface.co/mlx-community/Qwen3.5-4B-MLX-4bit), [mlx-lm Qwen docs](https://qwen.readthedocs.io/en/latest/run_locally/mlx-lm.html).

---

## 6. Final recommendation

**Architecture: cascaded ASR → LLM(+vision) → TTS. Do NOT collapse STT into the LLM for production.**

Rationale recap: audio-native Gemma is latency-unfriendly (25 audio-tok/s prefill, ~1 s for a short clip), noise-fragile (AMI ~19 % WER vs Whisper-large-v3 ~16 %), capped at 30 s, can't stream partials, and only emits text anyway (you still need TTS). A dedicated streaming Whisper-turbo / Parakeet ASR is faster, more robust on room audio, and supports partial hypotheses for sub-500 ms barge-in.

**Concrete picks (working set comfortably < 20 GB on 24 GB M3):**

- **(a) Core reasoning LLM:** **`mlx-community/gemma-4-E4B-it-qat-4bit`** (~3 GB, ~57 tok/s, 128K ctx, Apache-2.0; QAT int4 = ~bf16 quality). Alternate for stronger reasoning/tools: `mlx-community/Qwen3.5-4B-MLX-4bit`. Faster fallback: `mlx-community/gemma-4-E2B-it-qat-4bit` (~95 tok/s). Quality headroom on the M3 Max dev box: `mlx-community/gemma-4-12B-it-qat-4bit`.
- **(b) Vision:** reuse the **same** `gemma-4-E4B-it-qat-4bit` via `mlx-vlm` for webcam frames (~256 tokens/frame, <100 ms added). No extra resident model.
- **ASR (separate):** `mlx-community/whisper-large-v3-turbo` (or Parakeet-MLX) — recency-check in the dedicated STT research note.
- **TTS (separate):** Kokoro-MLX / Qwen3-TTS via `mlx-audio` — see TTS research note.
- **Optional experiment:** keep audio-native `mlx-community/gemma-4-E4B-it-qat-4bit` (audio+image+text → text) on the shelf for a quiet-room single-model demo, gated behind mlx-vlm ≥ post-PR-#931 and a `processor_config.json` feature-extractor check.

Budget check: Whisper-turbo (~1.5 GB) + Gemma 4 E4B (~3 GB) + Kokoro TTS (<1 GB) + KV cache/buffers ≈ **6–8 GB resident**, leaving wide headroom under the 20 GB ceiling on the 24 GB M3.
