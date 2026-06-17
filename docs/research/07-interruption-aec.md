# Research 07 — Reliable Barge-in / Acoustic Echo Cancellation (AEC) for the Mac dev case

> Status: DRAFT (in progress) — last updated 2026-06-17
>
> Problem: On the Reachy Mini robot, the **XVF3800** XMOS chip does hardware AEC at the mic
> input, so the barge-in VAD never hears the robot's own TTS — barge-in "just works."
> On the **Mac dev rig** we play Kokoro TTS (24 kHz) out the laptop speakers and capture the
> mic (16 kHz, resampled with soxr) via `sounddevice`/PortAudio. With open speakers, the mic
> picks up the robot's own voice and Silero VAD + Pipecat Smart-Turn v3 fire on it → false
> barge-in. We need **echo cancellation** before the barge-in VAD, runnable locally on
> macOS 15 (Sequoia) / Apple Silicon M3 / Python 3.12 venv.

## TL;DR recommendation

_(filled in at end)_

---

## Why `pip install` of the obvious AEC libs fails here

_(filled in)_

---

## Option 1 — macOS-native Voice Processing IO (CoreAudio AUVoiceProcessing)

_(filled in)_

## Option 2 — Installable Python AEC libraries that actually build

_(filled in)_

## Option 3 — How production frameworks (Pipecat / LiveKit / WebRTC) solve it

_(filled in)_

## Option 4 — Algorithmic fallback (NLMS / FDAF in numpy)

_(filled in)_

---

## Ranked options & RECOMMENDATION

_(filled in)_

## Sources

_(filled in)_
