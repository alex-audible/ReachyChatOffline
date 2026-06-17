# `reachy_chat.audio` — VAD, semantic endpointing, wake word & barge-in

The capture-side front end of the offline voice assistant. Turns a 16 kHz mic
stream into clean turn boundaries and drives the IDLE → LISTENING → THINKING →
SPEAKING interaction loop (both always-on and wake-word modes), with barge-in.

Grounded in [`docs/research/04-vad-turntaking.md`](../../../docs/research/04-vad-turntaking.md).

## Modules

| File | What it does |
|---|---|
| `vad.py` | **Silero VAD v6** (ONNX/CPU) wrapper. Strict 512-sample/16 kHz frames. Per-frame speech probability + a Schmitt-trigger speech/silence state machine (start/end thresholds, min-speech, hangover). ~0.13 ms/frame on M3. |
| `endpointer.py` | Semantic turn-end: Silero silence window **+ Pipecat Smart-Turn v3.2** (audio-based, ~13 ms). **Dual-threshold "eagerness"** over Smart-Turn's `P(complete)`: very-confident (`≥0.85`, sustained) commits after a short ~200 ms silence window; ordinary-confident (`≥0.5`) after ~300 ms; unsure holds to the `max_silence_ms`=700 ms timeout. Exposes an `on_speculative` hook (fires when `P≥0.75` *before* commit) so the pipeline can speculatively start the LLM and cancel-on-resume. Degrades to a pure VAD silence-window endpointer if Smart-Turn can't load (hook preserved). |
| `wakeword.py` | **"Hey Reachy"** via `livekit.wakeword` (ONNX). Real detector when a trained `hey_reachy.onnx` is present; otherwise a transparent energy/cadence **stub** so the state machine is testable end-to-end. |
| `interaction.py` | The state machine. Mode A (always-on, VAD-gated) and Mode B (wake-word + follow-up window). Barge-in: confident user speech during SPEAKING cancels the in-flight reply and re-listens. |

All components consume the same unit — **512-sample (32 ms) mono float32 frames at
16 kHz** — so they run identically from the live mic loop and the offline benchmark.

## Packages: clean installs vs fallbacks

| Component | Package | Status |
|---|---|---|
| Silero VAD v6 | `silero-vad==6.2.1` + `onnxruntime` | ✅ clean. ONNX model loads, ~0.13 ms/frame. |
| Smart-Turn v3 | HF `pipecat-ai/smart-turn-v3` (`smart-turn-v3.2-cpu.onnx`, 8 MB) + `transformers` `WhisperFeatureExtractor` | ✅ clean **standalone** (no Pipecat dep). Output is already a sigmoid probability; preprocessing matches the upstream repo (last 8 s → 80×800 log-mel, `do_normalize=True`). |
| Wake word | `livekit-wakeword==0.2.1` (imports as `livekit.wakeword`) | ⚠️ installs cleanly on **Python 3.12**; inference **backbone runs** (bundled mel + embedding ONNX). Missing only a *trained* `hey_reachy.onnx` classifier (requires the offline training toolchain). Falls back to an energy/cadence stub until trained. |

### Training the real "Hey Reachy" model

```bash
pip install "livekit-wakeword[train,eval,export]"
brew install espeak-ng ffmpeg sox portaudio
python -m livekit.wakeword train --config hey_reachy.yaml   # → hey_reachy.onnx (~200 KB)
```

Drop `hey_reachy.onnx` next to `wakeword.py` (or pass `model_path=`) and the real
conv-attention detector replaces the stub automatically.

## Interaction modes

**Mode A — always-on (VAD-gated):** no wake word. Any confirmed Silero speech onset
opens a turn (IDLE → LISTENING); the endpointer (VAD silence + Smart-Turn) commits
the turn end (→ THINKING). The pipeline runs STT→LLM→TTS, calls `begin_speaking()`
(→ SPEAKING) and `finish_speaking()` (→ IDLE).

**Mode B — wake word:** `WakeWordDetector` runs continuously and cheaply while IDLE.
On "Hey Reachy" it opens one turn (same downstream path), then keeps a follow-up
window (default 4 s) where it behaves like Mode A before requiring the wake word
again.

**Barge-in (both modes):** while SPEAKING, a dedicated VAD runs on the mic with a
**raised start threshold** (0.85 — "partial ducking", so the robot's own TTS doesn't
self-interrupt). Confident sustained user speech fires `Callbacks.on_barge_in` —
where the pipeline cancels the in-flight turn task, flushes TTS playback, clears the
AEC reference — and transitions back to LISTENING.

> AEC note: in production the Reachy Mini XVF3800 mic array removes the robot's own
> TTS echo **in hardware** before audio reaches Python, so barge-in here only needs
> the threshold bump. For Mac-only dev (built-in mic + speaker), add software AEC
> (`pyaec`) upstream of these frames; this package is AEC-agnostic.

## Usage

```python
from reachy_chat.audio import InteractionMachine, InteractionConfig, Mode, Callbacks, iter_frames

cb = Callbacks(
    on_endpoint=lambda e: run_stt_llm_tts(),   # turn ended
    on_barge_in=cancel_and_flush,              # user interrupted TTS
)
m = InteractionMachine(InteractionConfig(mode=Mode.ALWAYS_ON), cb)
m.warmup()
for frame in mic_frames_512():                 # 512 samples @16k each
    m.process_frame(frame)
# pipeline drives: m.begin_speaking() when TTS starts, m.finish_speaking() when done
```

## Benchmark

`benchmarks/bench_endpoint.py` streams each prompt in real-time 32 ms frames and
measures **endpoint-decision latency** = wall time from ground-truth end-of-speech
(`harness.find_end_of_speech`) to the endpoint firing.

```bash
.venv/bin/python benchmarks/bench_endpoint.py            # VAD + Smart-Turn
.venv/bin/python benchmarks/bench_endpoint.py --no-smart-turn   # VAD-only baseline
```

### Measured (M3, real-time feed, n=5/prompt × 6 prompts) — **PRELIMINARY**

Isolated endpointer (`bench_endpoint.py`, eagerness layer):

| Config | Decision latency p50 | p95 | Smart-Turn compute p50 |
|---|---:|---:|---:|
| VAD + Smart-Turn v3.2 (eager) | **159 ms** | 269 ms | 13 ms |

All turns fire via `smart_turn_eager`; **zero timeouts, zero misses**. Decision
latency = the confidence-dependent silence window (200 ms eager / 300 ms medium) +
~13 ms compute. `p6_greeting` reads ~0 ms because its **first** phrase ("Good
morning.") is itself a complete turn (Smart-Turn correctly commits early) while the
harness t0 marks the *final* end of audio — a metric artefact, not a defect.

In the **full pipeline** (`ConversationApp.run_wav`, E2B LLM + Kokoro TTS):

| Stage | p50 | p95 |
|---|---:|---:|
| Endpoint detection (stop-talking → endpoint) | ~246 ms | ~325 ms |
| Compute (endpoint → first-audio: STT finalize + LLM + TTS) | ~361 ms | ~418 ms |
| **Full stop-talking → first-audio** | **~607 ms** | ~725 ms |

#### Why the live endpoint (~246 ms) is higher than the isolated bench (~159 ms)

Both use the *same* `harness.find_end_of_speech` t0, and **all turns commit via
Smart-Turn — none fall through to the timeout** (verified per-prompt). The gap is
**not** the endpointer's window; it is contention on the shared real-time loop: the
live app feeds streaming STT (`STTEngine.add_frame`, synchronous MLX decode every
~320 ms window) on the *same* thread that pumps endpoint frames. When an STT decode
lands during the trailing silence it blocks the loop right when the endpointer wants
to poll/fire Smart-Turn, adding ~85 ms. That cost lives in `engines.py`/`app.py`
(out of this module's scope), not the endpointer.

#### What the eagerness layer changed, and its safety floor

The endpointer is now as fast as is **safe** for these prompts. Their genuine
mid-turn (inter-clause) pauses run **160–192 ms**, and Smart-Turn briefly reads the
first clause of a two-clause question as "complete" during them (e.g. p4_weather at
its 160 ms pause returned P≈0.97). Committing on a 100–120 ms window there **cuts the
user off**. Two guards prevent that: (1) `eager_window_ms`=200 ms sits above the
longest mid-turn pause, so only a true turn-end (whose silence keeps growing) reaches
it; (2) `eager_consecutive`=2 requires sustained confidence — a short pause resolves
back to speech and resets the streak first. A cutoff-safety check confirms every
prompt fires **at/after** its true speech end.

So ~200 ms is the realistic floor for confident commits on these prompts; pushing
lower trades turn-taking correctness for latency. The remaining full-latency wins are
**(a)** taking STT off the endpoint-frame thread (engines/app), and **(b)** the
**speculative LLM start**: wiring `Endpointer.on_speculative` to start the LLM on the
stabilized STT partial and cancel-on-resume. The hook fires with a measured median
~109 ms (up to ~226 ms) head-start before commit in the full app — overlapping that
much LLM TTFT with endpoint detection should bring full p50 toward ~450–500 ms.

```python
# Pipeline-side speculative start (cancel-on-resume), using the endpointer hook:
ep.on_speculative = lambda prob: pipeline.start_llm_speculative(stt.partial())
# ... if VAD shows speech resumed before commit, pipeline cancels the speculative run.
```

Numbers are **preliminary**: a concurrent process may share the Metal GPU. Silero and
Smart-Turn are CPU/ONNX so are largely insulated, but treat absolutes as indicative
until run on an idle machine.
