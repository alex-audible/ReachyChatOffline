# Reachy Mini — Offline Conversational App · Plan

> Reachy convention #1: write `plan.md` first. This is the living plan; it is updated as
> research lands and experiments complete. Items marked **(provisional)** depend on research
> still in flight.

## 1. Goal

A fully **local**, **low-latency**, **English-only** voice-to-voice conversational app for the
**Reachy Mini** robot, running on **Apple Silicon**.

- **Latency target:** ≤ **500 ms** voice-to-voice (end of user speech → first audio out).
  ≤ **600 ms** when a webcam image is in the loop.
- **Natural conversation:** SOTA VAD, semantic turn-taking/endpointing, and **barge-in**
  interruption (user can cut the robot off mid-sentence).
- **TTS:** **very natural & emotional** speech is a hard requirement. Latency-first, but we
  experiment across engines and produce example audio; any engine that overshoots the budget
  is explicitly flagged.
- **Vision (secondary):** feed a **webcam frame** into the model so what Reachy sees enters
  the conversation.
- **Embodied presence:** use mic **direction-of-arrival (DOA)** to make Reachy **look at the
  active speaker**, plus other embodiment cues (active-listening nods, "thinking" gaze,
  emotion-linked antennas) for a more natural, lifelike interaction. See §11.

## 2. Confirmed decisions (from user)

| Decision | Choice |
|---|---|
| Interaction modes | **Both** always-on open-mic **and** wake-word — configurable |
| Language | **English only** |
| RAM budget | Soft **~20 GB** working set; must run on a **24 GB M3**. Dev on M3 Max/96 GB. |
| Priority | **Latency-first**; but very natural/emotional TTS is required — experiment & flag overshoots |
| Python | **3.12** (Reachy SDK is `>=3.10`, itself targets 3.12) |
| Accel | **MLX** preferred, **MPS** (PyTorch) fallback |

## 3. Environment / hardware (detected)

- Apple **M3 Max**, 30-core GPU, **96 GB** RAM, macOS 15.7.5.
- Toolchain: `uv` 0.9.20, `python@3.12` (3.12.9), `ffmpeg`, `hf` CLI (logged in, read token).
- Project venv: `.venv` (Python 3.12.9) via uv.
- Reference repos cloned to `.reference/` (gitignored): `reachy_mini`, `reachy_mini_conversation_app`.

## 4. Reachy Mini integration contract (from cloned SDK + skills)

- Daemon/simulator runs at **`http://localhost:8000`**; REST under `/api`, Swagger at `/docs`.
- Python SDK: `from reachy_mini import ReachyMini`; `with ReachyMini() as mini:`.
  - Motion: `goto_target()` (smooth ≥0.5 s), `set_target()` (real-time ≥10 Hz).
  - Motors: `body_rotation`, `stewart_1-6`, `right_antenna`, `left_antenna`.
  - Limits: head pitch/roll [-40,40]°, yaw [-180,180]°, body yaw [-160,160]°.
- **Media:** daemon owns camera+audio (`GstMediaServer`). Same-machine client gets a
  **LOCAL** GStreamer backend (no encode/decode) → `mini.media.get_frame()`. Or
  `ReachyMini(media_backend="no_media")` to release hardware and use `sounddevice`/OpenCV directly.
- `/api/state/doa` → mic **direction-of-arrival** (turn-taking + look-at-speaker).
- **AI pattern (reuse):** LLM decides → tool call → **queue** move → control loop executes
  (decouples LLM latency from smooth motion). Tools: `move_head`, `dance`, `play_emotion`,
  `camera`, `head_tracking`. **Profiles** = personalities (instructions + enabled tools).
- Reference conv-app uses **cloud** realtime (OpenAI/Gemini/HF) — **we replace that whole path**
  with our local pipeline, reusing the tool/profile/queue patterns + `vision/local_vision.py`.

## 5. Architecture hypothesis (provisional — pending research)

Two candidate architectures under evaluation:

**A. Cascade-with-overlap** (classic, heavily pipelined)
```
mic → VAD/endpoint → STT(streaming, partials) → LLM(stream tokens) → TTS(sentence-chunked, streaming) → speaker
                                   │ start LLM on stabilized partial      │ start TTS on first sentence
```
Latency comes from overlapping stages: LLM begins on a stabilized partial transcript; TTS begins
on the first completed sentence/clause; audio plays while later sentences synthesize.

**B. Full-duplex / speech-to-speech / audio-native LLM**
- e.g. **Gemma 4 QAT** (audio-native input — under research), or Moshi/Kyutai, Qwen-Omni.
- Collapses STT (+ maybe TTS) into one model → fewer stage hops, but TTS quality / voice control
  and the 20 GB budget are open questions.

**Decision driver:** whichever hits ≤500 ms p50 while keeping TTS very natural within ~20 GB.
Likely outcome: cascade-with-overlap for control over emotive TTS, with audio-native LLM
(Gemma 4 QAT) evaluated as a STT-collapse and as the vision path.

## 6. Provisional latency budget (cascade, M3) — to be validated empirically

| Stage | Budget (ms) | Notes |
|---|---:|---|
| Endpointing (VAD semantic turn-end) | 50–150 | dominant tunable; semantic turn model |
| STT finalize (on already-streamed audio) | 30–80 | streaming STT, partials precomputed |
| LLM TTFT (first token) | 80–150 | small MLX model, warm |
| LLM → first sentence | overlapped | start TTS at first clause |
| TTS time-to-first-audio | 80–150 | streaming TTS |
| Output buffering/playback start | 10–30 | |
| **Total to first audio** | **~350–500** | tight; needs overlap + warm models |

## 7. Component candidates (to be decided by research + benchmarks)

- **LLM/core:** Gemma 4 QAT (centerpiece — audio + image native?), Gemma 3n E2B/E4B, Qwen-Omni, small MLX text LLM.
- **STT:** Parakeet (parakeet-mlx), Whisper-large-v3-turbo / mlx-whisper / WhisperKit, Moonshine, Kyutai STT.
- **TTS:** Kokoro, Kyutai TTS, Orpheus, Sesame CSM, Chatterbox, XTTS-v2, Dia, Fish/OpenAudio, Higgs — benchmark shortlist + sample audio.
- **VAD/turn-taking:** Silero VAD v5/v6, TEN-VAD; Smart-Turn v2 / LiveKit turn detector; barge-in via AEC.
- **Wake word:** openWakeWord / Porcupine / microWakeWord.

## 8. Benchmark methodology

- **Canonical metric = in-conversation (warm/primed), cold-start EXCLUDED.** The 500 ms budget
  is for turns mid-conversation: models loaded + warmed, persona/RAG prefix prefilled & cached,
  only the new user turn prefilled per turn. The app **primes at startup** so the first real turn
  is already warm. Cold-start (load/first-forward) is recorded for info only, never vs budget.
- **t=0** ≡ end of user speech in a pre-recorded prompt WAV (16 kHz mono).
- Instrument each stage; report **p50/p95** over a fixed prompt set.
- Offline first (audio files), then live mic, then on-simulator.
- Test prompts generated in `audio_samples/prompts/` (varying length + a vision prompt + a turn-taking/pause case).
- Reports written to `docs/research/` and `benchmarks/` (per-engine + end-to-end).

## 9. Phases / milestones

1. **Research** (in flight): LLM/audio-native (Gemma 4 QAT), STT, TTS, VAD/turn-taking, e2e+latency. → `docs/research/*.md`
2. **Component benchmarks:** measure TTFT/TTFA/RTF/WER + memory per candidate on M3 (serial on GPU). Generate TTS sample audio for subjective judging.
3. **Pipeline v1 (offline):** cascade-with-overlap, measured against the 500 ms budget on audio files.
4. **Turn-taking + barge-in + wake-word.**
5. **Vision path** (webcam frame → model), 600 ms budget.
6. **Reachy integration:** wire to SDK/daemon, tool-calls → motion queue, run on simulator (localhost:8000).
7. **Performance report + polish.**

## 10. Open questions / risks

- Does **Gemma 4 QAT** truly take audio natively, and at what latency/memory on M3? (research)
- Can emotive TTS hit <150 ms time-to-first-audio on M3 without quality loss? (experiment)
- Echo cancellation for barge-in on macOS (capturing mic while speaker plays).
- asyncio event-loop starvation during MLX inference (known gotcha — see project skills).
- Staying ≤20 GB with LLM + STT + TTS + VAD resident simultaneously.

## 11. Embodied naturalness (look-at-speaker + presence)

Make the interaction feel alive, not like a smart speaker. Use Reachy's body + sensors. All
motion runs through the **move queue + control loop**, fully decoupled from the voice pipeline,
so embodiment **never eats the 500 ms latency budget**.

**Look-at-speaker (primary):**
- Read `/api/state/doa` (mic direction-of-arrival) → estimate speaker azimuth → map to
  `body_yaw` + head `yaw`, drive with `goto_target()` (minjerk) so Reachy turns toward whoever
  is talking. Dead-zone + rate-limit to avoid jitter.
- **Fuse DOA with face tracking** (mediapipe/YOLO already in conv-app `vision/head_tracking/`):
  DOA for instant coarse direction, vision for fine lock and when multiple faces are present.

**DOA as a conversation signal:**
- Weight/gate VAD by direction → fewer false triggers from off-axis noise; helps know *who* spoke.
- On wake-word, snap gaze toward the wake direction before responding.

**Active-listening + turn-taking cues:**
- Subtle antenna/nod backchannels while the user speaks.
- "Thinking" gaze (slight look-away + antenna droop) during LLM latency, then return gaze and
  perk antennas when starting to speak — also perceptually masks latency.
- Lean-in / antenna perk when yielding the turn (inviting the user to speak).

**Emotion-linked motion:** tie `play_emotion`/antenna pose to TTS emotional tone for congruent affect.
