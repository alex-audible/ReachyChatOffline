# VAD, Turn-Taking, Barge-In, Wake-Word & DOA Fusion — Research & Recommended Stack

**Project:** ReachyChatOffline — fully-local, low-latency, English-only voice-to-voice for Reachy Mini
**Target hardware:** Apple Silicon M3 / 24 GB, macOS 15, Python 3.10–3.12
**Latency goal:** voice-to-voice ≤ 500 ms
**Interaction modes:** (a) always-on open mic, (b) wake-word ("Hey Reachy")
**Research date:** June 2026 (all versions/dates verified current as of mid-2026; items I could not confirm are flagged)

---

## 0. TL;DR — Recommended Stack

| Component | Recommendation | Package / Repo ID | Latency contribution | License |
|---|---|---|---|---|
| **VAD** | Silero VAD v6 (ONNX CPU) | `silero-vad` 6.2.1 + `onnxruntime` ([repo](https://github.com/snakers4/silero-vad)) | ~0.2–1 ms / 32 ms frame | MIT |
| **Semantic turn / endpointing** | Pipecat Smart Turn v3 (audio-based) | `pipecat-ai/smart-turn` → HF `pipecat-ai/smart-turn-v3` | ~12–20 ms (M3) | BSD-2-Clause |
| **Barge-in framework** | Pipecat interruption pattern (or hand-rolled asyncio) | `pipecat-ai` | ~35–130 ms total cancel | Apache-2.0 |
| **AEC** | XVF3800 hardware AEC (primary); `pyaec` (SpeexDSP) for Mac-only testing | hardware / `pyaec` 1.0.1 | ~0 ms hw / 1–2 ms per 10 ms frame sw | MIT (pyaec) |
| **Wake word** | livekit-wakeword (custom "Hey Reachy", ONNX) | `livekit-wakeword` 0.2.1 ([repo](https://github.com/livekit/livekit-wakeword)) | <10 ms / 80 ms frame | Apache-2.0 |
| **DOA look-at-speaker** | XVF3800 azimuth → smoothed control loop (decoupled) | `reachy_mini.media` / `audio_control_utils.py DOA_VALUE` | off the voice path | — |
| **Full-duplex** | NOT now — keep modular; revisit later | (Moshi `moshi-mlx` as future option) | — | CC-BY-4.0 |

**Key numbers:** VAD ~1 ms, Smart Turn ~15 ms, endpoint silence window 150–200 ms, barge-in cancel ~50–130 ms. These leave roughly **280–330 ms** for STT + LLM first-token + TTS first-chunk inside the 500 ms budget.

> **Note on naming:** the prompt asked for "Smart Turn v2" and "Silero v5/v6". As of June 2026, **Smart Turn v3.x** (Sept–Dec 2025) supersedes v2, and **Silero VAD v6** (released Aug 2024, latest 6.2.1 Feb 2026) supersedes v5. Both are the current, recommended versions.

---

## 1. Voice Activity Detection (VAD)

VAD must be near-free in this budget: ideally sub-millisecond per frame. It is the first gate (speech present?) and feeds both endpointing and barge-in.

### 1.1 Silero VAD — **RECOMMENDED**

- **Repo:** [snakers4/silero-vad](https://github.com/snakers4/silero-vad) · **PyPI:** `silero-vad` · **License:** MIT (no keys, no telemetry)
- **Latest:** **6.2.1** (24 Feb 2026). v6.0 (Aug 2024) brought ~16% fewer errors on noisy data vs v5; v6.2.1 made `onnxruntime` an optional dependency (install it explicitly). [releases](https://github.com/snakers4/silero-vad/releases) · [PyPI](https://pypi.org/project/silero-vad/)
- **Per-frame compute:** ~189 µs per 31.25 ms chunk (ONNX, single x86 thread). On M3 CPU expect well under 1 ms. A CoreML port reports <2 ms/chunk at ~5% of one core. [perf wiki](https://github.com/snakers4/silero-vad/wiki/Performance-Metrics) · [FluidInference/silero-vad-coreml](https://huggingface.co/FluidInference/silero-vad-coreml)
- **Accuracy (FLEURS-VAD-102):** AUC-ROC 97.99%, F1 95.95%, false-alarm 9.41%. [FireRedASR2S report](https://arxiv.org/pdf/2603.10420)
- **Apple Silicon:** `pip install silero-vad onnxruntime` — native arm64 wheels, CPU path is already ~165× real-time; no MPS needed.

**STRICT chunk-size requirement (matches the project's existing skill note):** Silero only accepts fixed window sizes — **512 samples @ 16 kHz** (32 ms), or 1024/1536; **256 @ 8 kHz**. Arbitrary chunk sizes raise `Provided number of samples...`. You **must** ring-buffer mic audio into exactly 512-sample blocks. The model is stateful — call `reset_states()` between utterances.

```python
from silero_vad import load_silero_vad, VADIterator
model = load_silero_vad(onnx=True)
vad = VADIterator(model, sampling_rate=16000)   # feed EXACTLY 512 samples @16k
event = vad(chunk_512, return_seconds=True)     # None | {'start': t} | {'end': t}
```

### 1.2 Other VADs evaluated

| VAD | ID / version | Apple Silicon | Per-frame | License | Verdict |
|---|---|---|---|---|---|
| **WebRTC VAD** | `webrtcvad-wheels` 2.0.14 (Sep 2024) | arm64 wheel | <50 µs | BSD | **Outdated.** GMM, ~50% TPR @ 5% FPR. Skip. [pkg](https://pypi.org/project/webrtcvad-wheels/) |
| **TEN-VAD** | `ten-vad` 1.0.6.8 (Nov 2025) | native arm64 lib | ~0.16 ms / 10 ms | "Apache-2.0" **+ Agora non-compete clause** | Good 10 ms granularity, but **license risk** for any RTC product; FAR 15.47% (worse than Silero). [repo](https://github.com/TEN-framework/ten-vad) · [license issue #9](https://github.com/TEN-framework/ten-vad/issues/9) |
| **FireRedVAD** | `fireredvad` (Mar 2026) | macos-arm64 reqs | not published | Apache-2.0 | **Best accuracy** (AUC 99.60%, FAR 2.69%) but 3 months old, no latency numbers — **watch list**. [repo](https://github.com/FireRedTeam/FireRedVAD) |
| **Cobra (Picovoice)** | `pvcobra` 2.1 | arm64 | ~0.0004 RTF | Commercial (AccessKey) | Best accuracy but paid; same vendor as Porcupine (see §4 license note). [docs](https://picovoice.ai/docs/benchmark/vad/) |
| **pyannote/segmentation-3.0** | HF model, MLX port exists | yes | 10 s window | MIT | **Not streaming** — needs multi-second lookahead. Offline only. [HF](https://huggingface.co/pyannote/segmentation-3.0) |
| **FSMN-VAD** | `funasr` 1.3.9 | yes | ~0.0077 RTF | MIT | Only if already using FunASR ASR; heavy packaging. [PyPI](https://pypi.org/project/funasr/) |
| **NeMo MarbleNet** | NVIDIA NGC | CUDA-centric | — | — | Heavy NVIDIA dep chain. Skip on Mac. |

**Decision:** Silero VAD v6 (ONNX). Fallback / future upgrade: FireRedVAD once it has a latency track record.

---

## 2. Semantic Turn Detection / Smart Endpointing

The job: decide the user has *actually finished*, not just paused mid-sentence. Pure silence timeouts either cut users off (short timeout) or feel sluggish (long timeout). A semantic model lets you keep the silence window short (~150–200 ms) and only commit when the utterance is *complete*.

### 2.1 Pipecat Smart Turn v3 — **RECOMMENDED**

- **Repo:** [pipecat-ai/smart-turn](https://github.com/pipecat-ai/smart-turn) · **Model:** HF [pipecat-ai/smart-turn-v3](https://huggingface.co/pipecat-ai/smart-turn-v3) · **License:** **BSD-2-Clause** (weights + data + training all open)
- **Type:** **Audio-based** (raw 16 kHz waveform → P(turn complete)). No transcript needed → no STT dependency on the critical path.
- **Versions:** v2 (Jul 2025, 360 MB, 14 langs) → **v3.0** (11 Sep 2025: Whisper-Tiny encoder + linear head, **8 MB int8 ONNX**, 23 langs, ~12 ms CPU) → **v3.1** (3 Dec 2025: English 94.7% @ 8 MB / 95.6% @ 32 MB) → v3.2 bundled as Pipecat default. [v3 blog](https://www.daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/) · [v3.1 blog](https://www.daily.co/blog/improved-accuracy-in-smart-turn-v3-1/)
- **Latency:** 12.6 ms (x86 CPU), 15.2 ms (AWS Graviton ARM). **On M3 expect ~10–20 ms** via `onnxruntime` arm64 CPU EP. Optional CoreML/ANE EP exists but CPU is already fast enough.
- **Apple Silicon:** ships in `pip install pipecat-ai` (onnxruntime included); arm64 native. The old `LocalCoreMLSmartTurnAnalyzer` was removed in favor of the faster ONNX CPU path.

```python
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
analyzer = LocalSmartTurnAnalyzerV3()   # default since pipecat v0.0.102
# Pair with Silero VAD; recommended stop_secs ~0.2
```

**Standalone (no Pipecat):** the repo ships `predict_endpoint()` over an `onnxruntime.InferenceSession` on `smart-turn-v3.2-cpu.onnx`.

### 2.2 LiveKit turn-detector — strong text-based alternative / second layer

- **Pkg:** `livekit-plugins-turn-detector` 1.6.0 (Jun 2026; now folded into `livekit-agents` core) · **Model:** HF [livekit/turn-detector](https://huggingface.co/livekit/turn-detector)
- **Type:** **Text/transcript-based** (Qwen2.5-0.5B distilled → INT8 ONNX, last 4 turns). Multilingual model v0.4.1-intl (Dec 2025) cut false-positive interruptions ~39%. TPR 99.3–99.4%. [blog](https://livekit.com/blog/improved-end-of-turn-model-cuts-voice-ai-interruptions-39)
- **Latency:** ~25 ms CPU + **requires STT output first** (adds STT latency). ~396 MB on disk, <500 MB RAM. License: Apache-2.0 code / custom LiveKit model license.
- **Use:** optional semantic safety net for ambiguous endings; costs latency headroom, so make it opt-in.

### 2.3 Others evaluated

| Model | ID | Type | Why not primary |
|---|---|---|---|
| **Krisp VIVA Turn-Taking** | proprietary SDK | audio | Not open / not free; commercial license, .kef weights. Pipecat has a `krisp_viva_turn` hook but needs the SDK. [blog](https://krisp.ai/blog/krisp-turn-taking-v2-voice-ai-viva-sdk/) |
| **Vogent-Turn-80M** | HF `vogent/Vogent-Turn-80M` (Oct 2025) | audio+text | Best concept (multimodal) but English-only, no ONNX export / CPU numbers yet — too slow in FP32. Watch. |
| **TEN Turn Detection** | HF `TEN-framework/TEN_Turn_Detection` | text (8B Qwen2.5) | 8B model → seconds on CPU. Has useful 3-class finished/unfinished/**wait**, but too big. |
| **Turnsense** | `latishab/turnsense` | text (135M) | Apache-2.0 edge classifier, but tiny 2k-sample training set; LiveKit is better validated. |

**Decision:** Smart Turn v3 (audio, on the hot path) as the primary endpointer paired with Silero VAD. Keep LiveKit text-based as an optional second-stage gate if false-commit rate is too high in testing. Always keep an absolute silence-timeout fallback (e.g. `user_speech_timeout=0.6 s`) so the pipeline never hangs.

---

## 3. Barge-In / Interruption + AEC

### 3.1 The two sub-problems

1. **Detect** user speech while TTS is playing — without the mic's pickup of the robot's *own* TTS falsely triggering VAD (the echo/self-trigger loop).
2. **Cancel** the in-flight STT→LLM→TTS cleanly and start listening, fast and without artefacts.

### 3.2 How the frameworks do it

**Pipecat** ([repo](https://github.com/pipecat-ai/pipecat), Apache-2.0) uses a frame pipeline with **priority "SystemFrame" lanes**. Flow:
1. Silero VAD (or Smart Turn) fires → `UserStartedSpeakingFrame`.
2. Transport pushes an `InterruptionFrame` (SystemFrame) downstream, bypassing the normal queue.
3. Each processor cancels: LLM service cancels in-flight generation (+ filters stale responses), TTS calls `cancel_task()` (asyncio cancel w/ ~1 s guard), output transport flushes the audio buffer.
4. A pipeline-level `allow_interruptions` flag gates step 3 (set False for PTT). `UninterruptibleFrame` survives barge-in (e.g. committing a tool result).

Known caveats (mid-2026): hard-cut interruption sounds abrupt — open proposal [#3985](https://github.com/pipecat-ai/pipecat/issues/3985) for a `bounded_drain` (let 40–80 ms finish, 10–30 ms fade-out). Race when user resumes after a short pause [#2703](https://github.com/pipecat-ai/pipecat/issues/2703). Pipecat has **no built-in local AEC** ([#670](https://github.com/pipecat-ai/pipecat/issues/670)). [frames docs](https://reference-server.pipecat.ai/en/stable/api/pipecat.frames.frames.html) · [speech-input docs](https://docs.pipecat.ai/pipecat/learn/speech-input)

**LiveKit Agents** ([repo](https://github.com/livekit/agents), Apache-2.0, v1.5.x) adds an **Adaptive Interruption** model: after VAD triggers, a small CNN classifies the first 200–500 ms as genuine barge-in vs backchannel ("uh-huh") — 86% precision / 100% recall @ 500 ms, ≤30 ms inference, rejects ~51% of false VAD triggers. `start_cooldown=1.0 s` (let client AEC settle after agent starts), `end_cooldown=3.5 s`. `SpeechHandle.cancel()` propagates to the TTS synth stream. Knobs: `allow_interruptions`, `interrupt_speech_duration`, `interrupt_min_words`. **Local AEC is "handled client-side, where the speaker is"** — their NC plugins are cloud-side. [adaptive interruption docs](https://docs.livekit.io/agents/logic/turns/adaptive-interruption-handling/) · [blog](https://livekit.com/blog/adaptive-interruption-handling)

**ChipChat** (Apple, M2 Ultra, [arXiv](https://arxiv.org/html/2509.00078v1)) is the most directly relevant local pattern: ASR identifies non-silence tokens during TTS → "instantaneously signals all downstream components to halt," with word-level alignment to clear only the unvocalized LLM cache. Relies on **device hardware AEC**.

### 3.3 Canonical clean-cancellation pattern (framework-agnostic)

```python
async def handle_barge_in(self):
    self._interrupt.set()
    if self._turn_task and not self._turn_task.done():
        self._turn_task.cancel()
        try: await asyncio.wait_for(self._turn_task, timeout=0.1)
        except (asyncio.CancelledError, asyncio.TimeoutError): pass
    sd.stop()                       # flush TTS playback immediately
    self._tts_reference.clear()     # clear AEC reference buffer
    self._interrupt.clear()
    await self.start_listening()
```
Inside the turn coroutine, check `self._interrupt.is_set()` at every `async for` token/chunk (cooperative cancel) **and** re-raise `asyncio.CancelledError` after cleanup. For cloud TTS you must also close the provider stream/WebSocket; for local TTS, stopping playback + dropping the queue is enough. Production targets (2026): TTS flush ≤60 ms, LLM cancel ≤40 ms, total barge-in handle ≤150 ms. [futureagi](https://futureagi.com/blog/voice-ai-barge-in-turn-taking-2026/)

### 3.4 Acoustic Echo Cancellation on Mac

The decisive fact for this project: **Reachy Mini's mic array is a reSpeaker XVF3800 (XMOS), which has on-chip hardware AEC, beamforming, NS, AGC, VAD and DoA, always on, 16 kHz over USB.** [Reachy media stack](https://huggingface.co/blog/pollen-robotics/reachy-mini-media-stack) · [XVF3800](https://www.seeedstudio.com/ReSpeaker-XVF3800-USB-Mic-Array-p-6488.html)

So the tiered approach is:

- **Tier 1 / 2 (production — robot in the loop):** Capture the XVF3800's already-AEC'd stream. The robot's own TTS echo is removed in hardware before it reaches Python. **No software AEC needed.** This is by far the most robust path.
- **Tier 3 (Mac-only dev: built-in mic + local speaker):** use **`pyaec`** (Rust/SpeexDSP, MIT, [PyPI](https://pypi.org/project/pyaec/), arm64 wheel, Python ≥3.6):
  ```python
  clean = pyaec.cancel_echo(near=mic_pcm, far=tts_ref_pcm, sample_rate=16000, frame_size=160)
  ```
  Keep the TTS PCM you sent to the speaker as the **reference signal** (simplest, zero extra capture latency), and time-align it to the mic via a fixed measured delay (~20–40 ms; estimate once with GCC-PHAT cross-correlation since the robot/room geometry is fixed).
- **Reference capture alternatives (if you can't keep TTS PCM in-process):** `proctap` (MIT, macOS 13+, CoreAudio/ScreenCaptureKit process tap, [repo](https://github.com/m96-chan/ProcTap)) or a BlackHole virtual loopback device — both add loopback delay and setup friction.
- **macOS VoiceProcessingIO** (`kAudioUnitSubType_VoiceProcessingIO`): Apple's native AEC, but **no Python API**, only cancels within the same audio-unit graph, `AudioComponentInstanceNew` blocks ~2 s on startup, and has Bluetooth-output quirks. Needs a Swift helper — not worth it given the XVF3800.

**Avoid:** `pyspeexaec` (CPython 3.8-only wheel), raw `speexdsp-python` (compile-only), `python-webrtc-audio-processing` (does **not** expose AEC3). `pyroomacoustics` adaptive filters are pure-Python and too slow for real-time callbacks.

**Pragmatic fallbacks regardless of AEC quality:**
- **Partial ducking:** during TTS, raise the Silero threshold (e.g. 0.5 → 0.80–0.85) so only confident/loud speech interrupts. [coval.ai](https://www.coval.ai/blog/voice-ai-echo-cancellation)
- **Half-duplex gating:** disable VAD during TTS, re-enable in a 200 ms drain window after — 100% reliable, but no mid-speech barge-in. Last resort.

---

## 4. Wake Word ("Hey Reachy")

### 4.1 livekit-wakeword — **RECOMMENDED**

- **Repo:** [livekit/livekit-wakeword](https://github.com/livekit/livekit-wakeword) · **PyPI:** `livekit-wakeword[train,eval,export]` 0.2.1 (21 May 2026) · **License:** **Apache-2.0 (including trained models)**
- **Why:** spiritual successor to openWakeWord with a conv-attention classifier: **~100× fewer false positives/hr, 86% vs 69% recall** on their test set. Exports the **same ONNX format** as openWakeWord (drop-in inference / Home Assistant compatible). [blog](https://livekit.com/blog/livekit-wakeword)
- **Custom training is one command** and TTS-synthesizes its own data — no NVIDIA GPU required (CPU/MPS on Mac); deps Homebrew-installable:
  ```bash
  pip install "livekit-wakeword[train,eval,export]"
  brew install espeak-ng ffmpeg sox portaudio
  livekit-wakeword train --config hey_reachy.yaml   # → hey_reachy.onnx (~200 KB)
  ```
- **Inference:** ONNX via `onnxruntime` on arm64 (optional CoreML/ANE EP). <10 ms per 80 ms frame; ~50–80 MB RAM with the shared embedding backbone.
- **Constraint:** **Python 3.11+** for training (3.11/3.12 supported; 3.10 users either upgrade or fall back to openWakeWord for the custom model).

### 4.2 Alternatives

| Engine | ID / version | Apple Silicon | License / cost | Verdict |
|---|---|---|---|---|
| **openWakeWord** | `openwakeword` 0.6.0 (Feb 2024) | inference yes (ONNX); **training needs Linux+CUDA** | code Apache-2.0; **pre-trained models CC BY-NC-SA (non-commercial)** | Solid **fallback**; train custom model off-Mac, transfer .onnx. Maintenance-mode (no release in 16 mo); Colab notebook needs patches. ~0.2–0.5 FA/hr. [repo](https://github.com/dscripka/openWakeWord) |
| **Picovoice Porcupine** | `pvporcupine` 4.0.2 (Feb 2026) | native arm64, offline | **Free tier ends 30 Jun 2026; enterprise ~$6,000/yr** | Best accuracy (97.1% @ 1 FA/10 hr) & instant custom keyword, but **free tier effectively gone now** → disqualified for this project. [benchmark](https://picovoice.ai/docs/benchmark/wake-word/) · [shutdown notice](https://community.home-assistant.io/t/fyi-picovoice-confirmed-free-tier-accesskeys-will-stop-working-after-june-30-2026/1012744) |
| **sherpa-onnx KWS** | `sherpa-onnx` 1.13.3 (Jun 2026) | arm64 wheels | Apache-2.0 | **No training** (phoneme keyword file) is attractive, but **English GigaSpeech model has <10% recall** ([issue #2678](https://github.com/k2-fsa/sherpa-onnx/issues/2678)); new bilingual zh-en model unbenchmarked for English. Secondary experiment only. |
| **microWakeWord** | OHF-Voice (AS trainer v7, Jun 2026) | trainer on Mac, **runtime ESP32-only** | Apache-2.0 | Not a Mac runtime. Skip (relevant only for ESP32 satellites). |
| **speech-swift (Soniqo)** | `soniqo/speech-swift` 0.0.20 | CoreML/ANE, 26× RT | Apache-2.0 | Swift-only KWS Zipformer; great if you go native Swift, impractical from pure Python. |
| Snowboy / Mycroft Precise | — | — | — | Dead. Ignore. |

**Decision:** livekit-wakeword for a custom "Hey Reachy" (Apache-2.0, easy local training, ONNX). Fallback: openWakeWord (train off-Mac, inference on-Mac). **Porcupine is ruled out** by the June 30 2026 free-tier shutdown.

---

## 5. Full-Duplex Alternative (Moshi-style) — brief

Full-duplex models listen and speak simultaneously, handling turn-taking/barge-in *inside the model*, removing explicit VAD/endpointing.

- **Kyutai Moshi** — `moshi-mlx` 0.3.0 (Aug 2025), HF `kyutai/moshiko-mlx-q4`. Runs on M3: q4 ~6 GB, bf16 ~16 GB unified RAM. ~160–200 ms latency, 24 kHz, Mimi codec @ 12.5 Hz. Backbone: Helium 7B. Gotcha: pin `sphn==0.1.12` and `cmake<4` ([#278](https://github.com/kyutai-labs/moshi/issues/278)). License: code MIT/Apache, weights CC-BY-4.0. [repo](https://github.com/kyutai-labs/moshi)
- **NVIDIA PersonaPlex-7B** (Jan 2026) adds voice/role conditioning; official Linux+GPU, experimental Python MLX port (`mu-hashmi/personaplex-mlx`, no streaming mic / no echo cancel). [HF](https://huggingface.co/nvidia/personaplex-7b-v1)
- Others (SALMONN-omni, DuplexOmni, Mini-Omni2, MiniCPM-o 4.5) are research-grade and/or not MLX-runnable for streaming mic.

**Why NOT now (recommendation):** full-duplex models lock you to a fixed ~7B backbone with a ~2023 knowledge cutoff, **no tool/function calling, no system prompt/persona without fine-tuning, ~4–5 min context cap, English-only, and no natural wake-word path** — all of which a robot assistant needs. Tellingly, **Kyutai itself pivoted to a modular STT+TTS stack** (Delayed Streams Modeling / Unmute). Keep the modular pipeline; the ~300 ms latency gap is recoverable with streaming STT/LLM/TTS. **Revisit** when a full-duplex model ships with a 30B-class tool-using backbone, >30 min context, and native MLX streaming I/O. [Moshi-vs-modular tradeoffs]

---

## 6. DOA Fusion: Look-At-Speaker + Turn-Taking (Reachy Mini)

Reachy Mini exposes mic direction-of-arrival from the XVF3800 array. Per the Reachy docs, DOA/azimuth is read via `reachy_mini.media` (`python src/reachy_mini/media/audio_control_utils.py DOA_VALUE`, also `DOA_VALUE_RADIANS`), in **radians**, derived from the 4-mic array. The project additionally exposes it at the daemon REST endpoint `/api/state/doa`. [Reachy advanced media controls](https://wiki.seeedstudio.com/reachymini_platforms_reachy_mini_media_advanced_controls/) · [XVF3800 host control](https://github.com/respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY/blob/master/host_control/README.md)

**Design principle (most important):** DOA drives **two separate things** that must NOT block the latency-critical voice path:
1. **Look-at-speaker head/body motion** — purely a control-loop concern.
2. **A turn-taking / VAD *prior*** — a cheap gating signal, never a hard dependency.

Run DOA in its own thread/async loop polling `/api/state/doa`. The STT→LLM→TTS path keeps consuming the (already AEC'd, beamformed) audio stream regardless of DOA state, so even if DOA is stale or wrong, voice latency is unaffected.

### 6.1 Fusing DOA + VAD + face tracking ("who has the turn")

A robust per-speaker turn estimate combines three asynchronous evidence streams:

| Signal | Source | Rate | Strength | Weakness |
|---|---|---|---|---|
| **VAD** | Silero on XVF3800 stream | ~30 Hz | "is anyone speaking now" | no *who*/*where* |
| **DOA azimuth** | XVF3800 / `/api/state/doa` | coarse, instant | *where* the sound is | unstable in silence; reflections; single dominant source |
| **Face/body** | MediaPipe Face/Pose or YOLO on the robot camera | 15–30 Hz | *who* + visual angle, mouth-open / lip motion | needs line of sight; lighting |

**Fusion recipe (lightweight, no learning needed to start):**
1. Maintain a small set of *speaker tracks*, each an azimuth estimate (from DOA, and from face detection mapped camera-px → azimuth) with a confidence/decay.
2. Gate DOA on VAD: **only trust/update DOA azimuth while VAD says speech is active.** This is the single most valuable rule — it removes meaningless DOA jitter during silence.
3. Associate the VAD-gated DOA azimuth to the nearest face track (angular nearest-neighbor within a tolerance, e.g. ±15–20°). When DOA and a face agree, confidence is high → that track "has the turn." Lip-motion (MediaPipe FaceMesh mouth landmarks) further disambiguates which visible person is actually speaking.
4. Feed the active track's identity/azimuth to: (a) the look-at controller, and (b) optionally the endpointer as a prior (see §6.2).

This audio-visual speaker localization for attention-shifting is established prior art — e.g. robots that shift gaze in group conversation via fused audio-visual speaker localization. [USPTO 11127401](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11127401) · [AcousticFusion (audio→visual SLAM)](https://arxiv.org/pdf/2108.01246) · [SSL review 2025](https://arxiv.org/html/2507.01143v1)

### 6.2 Can DOA reduce VAD false-accepts / help endpointing?

**Yes, as a *prior*, not a hard gate:**
- **Directional gating intuition:** the XVF3800 already beamforms toward the dominant source. If you additionally know the *active speaker's* azimuth (from the fused track), you can **down-weight VAD/barge-in triggers whose instantaneous DOA points well off-axis** from the engaged speaker — i.e. a TV across the room, an HVAC vent, or a second bystander. This is exactly the structure used in multi-mic speaker-segmentation systems: **DOA + spatial filter bank + VAD**, where DOAs are triggered by VAD and estimated via TDOA. [DOA+beamforming+VAD segmentation](https://dael.euracoustics.org/confs/fa2025/data/articles/000526.pdf)
- **Multi-speaker endpointing:** in a two-person setting, a *change* in DOA mid-utterance is strong evidence of a speaker switch (turn yielded/taken) — a useful complement to Smart Turn's acoustic-completion cue. A *stable* DOA across a pause argues "same speaker, just thinking" → hold the turn.
- **Robot audition prior art:** HARK (MUSIC localization + GSS separation + per-source ASR) is the canonical framework for DOA-driven multi-speaker listening on robots. [HARK on GPU/FPGA 2024](https://www.mdpi.com/2674-0729/4/1/2) Humanoid speaker-direction estimation is well studied. [PLOS One 2023](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0296452)

**Guardrail:** because the XVF3800 reports a single dominant azimuth (not full multi-source separation), treat DOA as soft evidence. Never let an off-axis DOA *suppress a confident VAD+Smart Turn detection* — that risks deaf-spots. Use it to **raise** the barge-in threshold for off-axis sound, mirroring the partial-ducking pattern in §3.4.

### 6.3 Look-at-speaker motion (decoupled control loop)

- **Update rate:** XVF3800 azimuth is coarse and effectively instantaneous; the project's `/api/state/doa` poll rate is the practical limiter (the exact device DOA refresh rate is **not published** in the docs — measure empirically and treat ~10–30 Hz as a working assumption). [XVF3800 user guide](https://www.xmos.com/documentation/XM-014888-PC/pdf/xvf3800_user_guide_v3.2.1.pdf)
- **Anti-jitter:** (1) only update target azimuth when **VAD-active**; (2) apply a **dead-zone** (ignore azimuth changes < ~10–15°); (3) **smooth** with an exponential moving average or low-pass on the target angle; (4) add **hysteresis / dwell time** (require N consecutive consistent readings, e.g. 150–300 ms, before re-pointing) so the head doesn't snap to a brief cough or reflection; (5) rate-limit head/body velocity for natural motion.
- **Decoupling:** the control loop reads the fused active-azimuth at its own cadence and commands head/body; it shares only a small thread-safe state object (current active azimuth + confidence) with the voice pipeline. No DOA call is ever `await`ed on the STT→LLM→TTS path. Prefer face-track azimuth when available (smoother, identity-stable) and fall back to raw DOA when no face is visible.

### 6.4 DOA fusion — recommended summary

- Read `/api/state/doa` in a dedicated loop; **VAD-gate** every DOA reading.
- Associate VAD-gated DOA to MediaPipe/YOLO face tracks (+ lip motion) for "who is speaking."
- Use the active azimuth to (a) drive a **smoothed, dead-zoned, hysteretic** look-at controller, and (b) **softly raise** barge-in/VAD thresholds for off-axis sound and inform multi-speaker turn-switch detection.
- Keep it 100% off the latency-critical path: shared state only, never blocking.

---

## 7. End-to-End Architecture + Latency Budget

### 7.1 Two-mode capture front-end

```
                         ┌─────────────────────────────────────────────┐
  Reachy Mini XVF3800 ───┤ USB 16kHz PCM  (HW AEC + beamform + NS)       │
   (mic array + DOA)     └───────────────┬───────────────┬──────────────┘
                                         │ audio          │ /api/state/doa
                          ┌──────────────▼───┐     ┌──────▼───────────────┐
   MODE B (wake word):    │ livekit-wakeword │     │ DOA loop (separate)  │
   gate pipeline until    │  "Hey Reachy"    │     │  VAD-gated azimuth    │
   detect; MODE A: skip   └──────────────┬───┘     │  → face-track assoc   │
                                         │         │  → look-at controller │
                          ┌──────────────▼───┐     │  → off-axis prior     │
                          │ Silero VAD v6    │◄────┤ (soft threshold bump) │
                          │ (512-sample win) │     └──────────────────────┘
                          └──────────────┬───┘
                          ┌──────────────▼─────────┐
                          │ Smart Turn v3 (audio)  │  endpoint? (P≥0.5)
                          │  + 150–200ms silence   │  + 0.6s timeout fallback
                          └──────────────┬─────────┘
                          ┌──────────────▼─────────┐   barge-in: InterruptionFrame
                          │ STT → LLM → TTS        │   → cancel task, sd.stop(),
                          │ (streaming, cancelable)│   → flush, re-listen
                          └──────────────┬─────────┘
                                  speaker (TTS PCM kept as AEC ref for Tier-3)
```

### 7.2 Per-component latency budget (M3, local)

| Stage | Budget | Notes |
|---|---|---|
| Wake-word detect (Mode B only) | ~80 ms frame, <10 ms compute | off the response budget (pre-trigger) |
| Silero VAD | ~1 ms / 32 ms frame | negligible |
| Endpoint silence window | **150–200 ms** | tunable; the big knob for "feel" |
| Smart Turn v3 inference | **~15 ms** | audio-based, no STT needed |
| **Turn-detection subtotal** | **~165–215 ms** | before STT starts |
| STT (streaming, e.g. MLX/whisper) | ~80–150 ms to usable text | overlaps with end of speech |
| LLM first token (local, streaming) | ~100–200 ms TTFT | speculative decoding helps |
| TTS first chunk (streaming) | ~80–150 ms TTFA | sentence-chunked |
| **Voice-to-voice (perceived)** | **~300–500 ms** | achievable if STT/LLM/TTS stream |
| Barge-in cancel (when interrupted) | **~35–130 ms** | task cancel + audio flush |
| AEC (Tier-3 sw only) | ~1–2 ms / 10 ms frame | zero with XVF3800 hardware |
| DOA + look-at | decoupled | never on the response path |

The 500 ms target is feasible **only** because Smart Turn is audio-based (no STT before endpointing) and STT/LLM/TTS stream and overlap. The dominant tunable is the silence window; start at 200 ms and reduce toward 150 ms once Smart Turn's false-commit rate is acceptable.

### 7.3 Mode handling

- **Mode A (always-on):** wake-word stage disabled; Silero VAD → Smart Turn gate every utterance. Use the DOA off-axis prior + partial ducking to suppress ambient false triggers.
- **Mode B (wake word):** livekit-wakeword runs continuously at low cost; on "Hey Reachy" it opens the VAD→Smart Turn→pipeline for one turn (or a short follow-up window). Same downstream path.

---

## 8. Concrete Install Set

```bash
# core turn-taking + VAD + barge-in
pip install pipecat-ai           # bundles Smart Turn v3 + onnxruntime
pip install silero-vad onnxruntime
# wake word (Python 3.11+)
pip install "livekit-wakeword[train,eval,export]"
brew install espeak-ng ffmpeg sox portaudio
# Mac-only dev AEC (skip in production; XVF3800 does it in hardware)
pip install pyaec sounddevice
# optional second-stage semantic endpoint gate
pip install livekit-plugins-turn-detector
```

---

## 9. Open Risks / Watch List

- **FireRedVAD** (Mar 2026) — best accuracy, but verify latency before adopting over Silero.
- **Smart Turn** English accuracy ~94.7% (8 MB) — validate false-commit rate on your mic/room; keep the silence-timeout fallback.
- **livekit-wakeword** requires **Python 3.11+** — confirm the project's interpreter; else openWakeWord fallback (train off-Mac).
- **Porcupine free tier ends 30 Jun 2026** — do not build on it.
- **TEN-VAD license** (Agora non-compete) — avoid if any commercial/RTC intent.
- **`/api/state/doa` update rate** not documented — measure; tune dead-zone/hysteresis accordingly.
- **Full-duplex** — re-evaluate when a tool-using, long-context, MLX-streaming model appears.

---

## Sources

**VAD:** [silero-vad repo](https://github.com/snakers4/silero-vad) · [releases](https://github.com/snakers4/silero-vad/releases) · [PyPI](https://pypi.org/project/silero-vad/) · [perf wiki](https://github.com/snakers4/silero-vad/wiki/Performance-Metrics) · [silero CoreML](https://huggingface.co/FluidInference/silero-vad-coreml) · [ten-vad](https://github.com/TEN-framework/ten-vad) · [ten-vad license issue](https://github.com/TEN-framework/ten-vad/issues/9) · [webrtcvad-wheels](https://pypi.org/project/webrtcvad-wheels/) · [FireRedVAD](https://github.com/FireRedTeam/FireRedVAD) · [FireRedASR2S report](https://arxiv.org/pdf/2603.10420) · [Picovoice VAD benchmark](https://picovoice.ai/docs/benchmark/vad/) · [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) · [FunASR](https://pypi.org/project/funasr/)

**Turn detection:** [smart-turn repo](https://github.com/pipecat-ai/smart-turn) · [smart-turn-v3 HF](https://huggingface.co/pipecat-ai/smart-turn-v3) · [v3 blog](https://www.daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/) · [v3.1 blog](https://www.daily.co/blog/improved-accuracy-in-smart-turn-v3-1/) · [LiveKit turn-detector HF](https://huggingface.co/livekit/turn-detector) · [livekit-plugins-turn-detector](https://pypi.org/project/livekit-plugins-turn-detector/) · [LiveKit EOU blog](https://livekit.com/blog/improved-end-of-turn-model-cuts-voice-ai-interruptions-39) · [Krisp VIVA](https://krisp.ai/blog/krisp-turn-taking-v2-voice-ai-viva-sdk/) · [Vogent-Turn-80M](https://huggingface.co/vogent/Vogent-Turn-80M) · [TEN Turn Detection](https://huggingface.co/TEN-framework/TEN_Turn_Detection) · [turnsense](https://github.com/latishab/turnsense)

**Barge-in / AEC:** [Pipecat repo](https://github.com/pipecat-ai/pipecat) · [frames docs](https://reference-server.pipecat.ai/en/stable/api/pipecat.frames.frames.html) · [Pipecat #3985](https://github.com/pipecat-ai/pipecat/issues/3985) · [#2703](https://github.com/pipecat-ai/pipecat/issues/2703) · [#670 AEC](https://github.com/pipecat-ai/pipecat/issues/670) · [LiveKit agents](https://github.com/livekit/agents) · [adaptive interruption docs](https://docs.livekit.io/agents/logic/turns/adaptive-interruption-handling/) · [LiveKit noise/echo docs](https://docs.livekit.io/transport/media/noise-cancellation/) · [ChipChat](https://arxiv.org/html/2509.00078v1) · [pyaec PyPI](https://pypi.org/project/pyaec/) · [pyaec repo](https://github.com/thewh1teagle/aec) · [ProcTap](https://github.com/m96-chan/ProcTap) · [coval.ai AEC](https://www.coval.ai/blog/voice-ai-echo-cancellation) · [futureagi barge-in 2026](https://futureagi.com/blog/voice-ai-barge-in-turn-taking-2026/) · [Daily advice Jun 2025](https://www.daily.co/blog/advice-on-building-voice-ai-in-june-2025/)

**Wake word:** [livekit-wakeword repo](https://github.com/livekit/livekit-wakeword) · [livekit-wakeword blog](https://livekit.com/blog/livekit-wakeword) · [openWakeWord](https://github.com/dscripka/openWakeWord) · [pvporcupine](https://pypi.org/project/pvporcupine/) · [Porcupine benchmark](https://picovoice.ai/docs/benchmark/wake-word/) · [Picovoice free-tier shutdown](https://community.home-assistant.io/t/fyi-picovoice-confirmed-free-tier-accesskeys-will-stop-working-after-june-30-2026/1012744) · [sherpa-onnx KWS](https://k2-fsa.github.io/sherpa/onnx/kws/index.html) · [sherpa English KWS issue #2678](https://github.com/k2-fsa/sherpa-onnx/issues/2678) · [microWakeWord AS trainer](https://github.com/TaterTotterson/microWakeWord-Trainer-AppleSilicon) · [speech-swift](https://github.com/soniqo/speech-swift)

**Full-duplex:** [Moshi repo](https://github.com/kyutai-labs/moshi) · [moshi-mlx PyPI](https://pypi.org/project/moshi-mlx/) · [Moshi paper](https://arxiv.org/html/2410.00037v2) · [sphn issue #278](https://github.com/kyutai-labs/moshi/issues/278) · [PersonaPlex-7B HF](https://huggingface.co/nvidia/personaplex-7b-v1) · [personaplex-mlx](https://github.com/mu-hashmi/personaplex-mlx) · [Kyutai delayed-streams-modeling](https://github.com/kyutai-labs/delayed-streams-modeling) · [Kyutai unmute](https://github.com/kyutai-labs/unmute)

**DOA fusion:** [Reachy Mini media controls](https://wiki.seeedstudio.com/reachymini_platforms_reachy_mini_media_advanced_controls/) · [Reachy media stack](https://huggingface.co/blog/pollen-robotics/reachy-mini-media-stack) · [XVF3800 host control](https://github.com/respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY/blob/master/host_control/README.md) · [XVF3800 user guide](https://www.xmos.com/documentation/XM-014888-PC/pdf/xvf3800_user_guide_v3.2.1.pdf) · [DOA+beamforming+VAD segmentation](https://dael.euracoustics.org/confs/fa2025/data/articles/000526.pdf) · [HARK GPU/FPGA 2024](https://www.mdpi.com/2674-0729/4/1/2) · [SSL review 2025](https://arxiv.org/html/2507.01143v1) · [Humanoid speaker direction (PLOS 2023)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0296452) · [Audio-visual attention shifting (USPTO)](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11127401) · [AcousticFusion](https://arxiv.org/pdf/2108.01246)
