# ReachyChatOffline

**A fully-local, low-latency, voice-to-voice conversational app for the [Reachy Mini](https://www.pollen-robotics.com/reachy-mini/) robot — running entirely on Apple Silicon (MLX/Metal). No cloud, no API keys, no data leaving your Mac.**

Talk to Reachy and it talks back, in under a second. Feels natural. Should work on most modern Macs with M2, M3, M4, M5 processors with 16+GB of RAM. 

## Quick start

After the one-time [setup](#getting-started), from the project directory:

```bash
cd ReachyChatOffline

# Talk to Reachy — live mic, vision on, turbo voice, + robot embodiment:
PYTHONPATH=src .venv/bin/python -m reachy_chat.pipeline.app --robot
```

`--robot` expects the **Reachy Mini daemon at `http://localhost:8000`**. Start a simulator daemon in another terminal first:

(For a physical **Wireless** robot, point at it instead with `--robot-url http://reachy-mini.local:8000`.)

Common variations:

```bash
# Voice only, no robot:
PYTHONPATH=src .venv/bin/python -m reachy_chat.pipeline.app

# Lowest-latency voice (Kokoro ~150 ms) instead of the turbo voice:
PYTHONPATH=src .venv/bin/python -m reachy_chat.pipeline.app --tts kokoro

# Low-RAM machine — skip the second (vision) model:
PYTHONPATH=src .venv/bin/python -m reachy_chat.pipeline.app --no-vision

Run it in a **real terminal** (Terminal.app / iTerm) so macOS grants microphone access. First launch downloads models and warms up (STT + LLM + TTS + vision), so judge responsiveness from the second turn on. Mid-conversation, say **"Hey Reachy, can you clone my voice?"** to switch to a clone of your voice.

## Features

- **Voice-to-voice, fully local** — speech in, speech out, with everything (STT, LLM, TTS, VAD) running on your Mac's GPU via MLX. Nothing is sent to a server.
- **Low latency** — streaming **cascade-with-overlap** pipeline: STT runs *during* your speech, the LLM starts on the finalized transcript, and TTS begins on the first speakable clause while later clauses synthesize. Measured first-audio (from detected end-of-speech): **p50 ~271 ms** (E2B) — well under the 500 ms budget.
- **The stack** — Silero VAD v6 + Smart-Turn v3 (endpointing) → Parakeet-TDT-0.6B (STT) → Gemma 4 E2B/E4B QAT-4bit (LLM) → Kokoro-82M (TTS), all MLX.
- **Natural turn-taking** — semantic endpointing decides when you've actually finished a thought, not just paused. Always-on open-mic *and* "Hey Reachy" wake-word modes.
- **Embodied presence** — uses the robot's mic **direction-of-arrival** to **look at the active speaker**, plus emotion-linked gestures matched to its reply. All motion runs on a decoupled control loop, so it never eats the latency budget.
- **Camera vision** — ask "what do you see?" and Reachy answers from a live webcam frame via the same Gemma 4 model (adds ~1 s for vision turns).
- **Fits the hardware** — ~7.5 GB working set in the speed config, comfortably under a 24 GB M3.

## Architecture

```
mic → VAD/endpoint → STT(finalize) → LLM(stream clauses) → TTS(stream) → speaker
         (Silero +     (Parakeet)      (Gemma 4)            (Kokoro)
          Smart-Turn)
                                  └── robot look-at-speaker + emotion (decoupled loop)
```

## Getting Started

```bash
git clone <this-repo> ReachyChatOffline
cd ReachyChatOffline

# Create an isolated venv with the correct Python (3.12, auto-downloaded) and install:
uv venv                 # reads .python-version → CPython 3.12
uv pip install -e .     # installs reachy-chat + all runtime deps into .venv
```

That installs the `reachy-chat` command. Verify:

```bash
uv run reachy-chat --help
```

Now choose a setup path below depending on whether you have a physical robot.

---

## Setup Path A — Physical Reachy Mini

1. Install **[Reachy Mini Control](https://hf.co/reachy-mini/#/download)** to bring it online and update its firmware. See `.reference/reachy_mini/docs/source/platforms/`.

2. **Find the daemon URL:**
   - **Reachy Mini Lite** (USB-tethered, daemon runs on *your Mac*): `http://localhost:8000`.
   - **Reachy Mini Wireless** (onboard Raspberry Pi runs the daemon): `http://reachy-mini.local:8000` (mDNS). If `reachy-mini.local` doesn't resolve on your network, use the robot's IP, e.g. `http://192.168.1.42:8000`.

3. **Run the app** pointing at that daemon:
   ```bash
   uv run reachy-chat --mode live --robot --robot-url http://reachy-mini.local:8000 --vision
   ```

## Setup Path B — Reachy Mini Simulator (no hardware)

You can run the **full** app — including `--robot` (look-at-speaker, emotions) against a simulated robot using the **Reachy Mini Control** desktop app ([download](https://hf.co/reachy-mini/#/download)). 

With a daemon running on `localhost:8000`, run the app with `--robot` (and optionally `--vision`) — see **Running it** below. (The voice pipeline itself runs fine with **no** daemon; you only need it for `--robot`/`--vision`.)

---

## Running it

After install, run with the `reachy-chat` command (no `PYTHONPATH` needed):

```bash
# Live mic, voice-to-voice (always-on open mic)
uv run reachy-chat --mode live

# Live mic + robot embodiment + camera vision (needs a daemon at the robot-url)
uv run reachy-chat --mode live --robot --vision

# Wake-word mode: idle until "Hey Reachy"
uv run reachy-chat --mode live --wake

# Smarter, slower model (better answers, enables a richer vision path)
uv run reachy-chat --mode live --llm mlx-community/gemma-4-E4B-it-qat-4bit
```

There is also an **offline file driver** (`--mode wav`) used for latency measurement. It needs the `benchmarks/` harness, so it only works from a source checkout (not the installed package):

```bash
uv run reachy-chat --mode wav --wav audio_samples/prompts/p3_vision.wav --speak
```

### Flags

| Flag | Default | What it does |
|---|---|---|
| `--mode {live,wav}` | `live` | `live` = mic + speaker conversation (default); `wav` = offline file driver (measures full stop-talking → first-audio; source checkout only) |
| `--llm <repo>` | `mlx-community/gemma-4-E2B-it-qat-4bit` | Gemma 4 model for chat **and** vision. Default **E2B** = latency-first. Pass **`mlx-community/gemma-4-E4B-it-qat-4bit`** for a smarter (slower) model |
| `--tts {kokoro,chatterbox-4bit,chatterbox-turbo-8bit}` | `chatterbox-turbo-8bit` | TTS model (the preset name **is** the model). **chatterbox-turbo-8bit** (default) = best-sounding + voice-clonable (~0.7–0.8 s TTFA); **chatterbox-4bit** = emotive (exaggeration knob) + clonable; **kokoro** = lowest latency (~150 ms) for the strict ≤500 ms budget |
| `--voice <wav>` | off | (chatterbox only) clone the voice from a reference WAV; omit for the built-in voice. You can also say "Hey Reachy, clone my voice" live |
| `--exaggeration <0..~1.5>` | `0.5` | (chatterbox-4bit only) emotion intensity; turbo ignores it |
| `--robot` | off | Drive the Reachy daemon/sim: look-at-speaker + emotion gestures (decoupled from the voice path) |
| `--robot-url <url>` | `http://localhost:8000` | Daemon address (e.g. `http://reachy-mini.local:8000` for a Wireless unit) |
| `--vision` / `--no-vision` | on | Answer "what do you see?" from the camera via Gemma 4 vision; the model is loaded at startup. On by default; `--no-vision` skips it (saves a second ~4–6 GB model — use on low-RAM machines) |
| `--wake` | off | Wake-word mode — idle until "Hey Reachy" |
| `--wav <path>` | `audio_samples/prompts/p3_vision.wav` | (wav mode) prompt WAV — must be 16 kHz mono |
| `--speak` | off | (wav mode) play the generated response aloud |

## Voice cloning

The Chatterbox voices (`chatterbox-turbo-8bit`, the default, and `chatterbox-4bit`) can **clone a voice** from a short reference clip. There are two ways to use it:

**1. Mid-conversation — just ask.** While talking, say **"Hey Reachy, can you clone my voice?"** Reachy will:
1. ask you to talk for ~10 seconds (your name, what you like doing for fun, favourite subject + foods),
2. record you, clone your voice, then
3. carry on the conversation **in your voice**.

This works with any `chatterbox-*` voice; if you launched with `--tts kokoro`, it loads a Chatterbox model on the fly to do the clone.

**2. From the command line — `--voice`.** Start already speaking in a cloned voice, from a reference WAV (~8–12 s of clean speech):

```bash
PYTHONPATH=src .venv/bin/python -m reachy_chat.pipeline.app --voice path/to/reference.wav
```

Omit `--voice` for the model's built-in default voice.

**Standalone demo (no conversation).** Record yourself and hear the clone read a few sentences back, without launching the full app:

```bash
PYTHONPATH=src .venv/bin/python scripts/clone_voice_demo.py --play
#   --ref some.wav    clone from a file instead of the mic
#   --seconds 10      record longer than the default 8 s
#   --model mlx-community/chatterbox-4bit   use the emotive 4-bit model instead
```

**Tips:** ~8–12 s of clean, expressive speech clones best — talk *with feeling*, since the clone copies your energy. On a Mac, run from a real terminal so the microphone works.

## First-run notes

- **Models download from Hugging Face on first use** (Parakeet STT, Gemma 4 LLM, Kokoro TTS, Silero VAD, Smart-Turn) — **several GB total, one-time**, cached in `~/.cache/huggingface`. Subsequent runs are offline.
- **Warmup ~30–60 s** at startup: the app loads every model and runs a dummy pass so the **first real turn is already warm** (this is why the latency numbers are measured in-conversation, not cold).
- On startup the app prints a **model banner** listing exactly which STT/LLM/TTS/Vision models are in use, and reminds you how to switch to the E4B model.
- **macOS microphone permission:** the first time you run `--mode live`, macOS will prompt for mic access. The permission is granted to the **terminal app that launches the process**, so run it from a real Terminal/iTerm (one that has, or can be granted, microphone permission under *System Settings → Privacy & Security → Microphone*). Launching from a context without mic permission will produce silent/empty audio.

## Troubleshooting

- **No transcript / empty audio in live mode** → microphone permission. Grant your terminal app mic access in *System Settings → Privacy & Security → Microphone*, then restart it. Set `REACHY_DEBUG_AUDIO=1` to dump per-turn captured audio to `/tmp/reachy_turn_*.wav` and confirm the mic is actually being heard.
- **`Reachy Chat needs Python 3.10–3.12…`** → you're on the wrong interpreter. Recreate the venv: `uv venv --python 3.12 && uv pip install -e .`.
- **Robot/vision does nothing / connection errors** → the daemon isn't reachable at `--robot-url`. Start a daemon (`uv run reachy-mini-daemon --mockup-sim` for the sim) and confirm `http://localhost:8000/docs` loads. For a Wireless unit, check `reachy-mini.local` resolves or use the IP.
- **Reachy interrupts itself / hears its own voice** → on a Mac there's no hardware echo cancellation. **Use headphones.** (On the physical robot the XVF3800 handles this.)
- **MuJoCo sim won't open on macOS** → use `mjpython` (not plain `python`), or fall back to `--mockup-sim`. See the simulator note above.

## Project layout

```
src/reachy_chat/
  pipeline/   # streaming engines (STT/LLM/TTS) + ConversationApp orchestrator + audio I/O + app.py entry point
  audio/      # Silero VAD, Smart-Turn endpointer, wake-word, interaction state machine
  robot/      # Reachy daemon REST client, motion queue, look-at-speaker, emotion/tool-calls
  vision/     # Gemma 4 webcam-frame path (mlx-vlm) + frame sources
  tts/        # Kokoro istftnet bug fix + emotive engine loaders
benchmarks/   # latency harness + per-component + end-to-end benchmarks
audio_samples/prompts/   # 16 kHz test prompt WAVs (p1..p6)
docs/         # architecture, performance report, research, experiments
plan.md       # living plan & decisions
```

## Status & known limitations

- **Working end-to-end:** the voice pipeline, turn-taking, wake-word, robot embodiment (verified against the simulator), and the vision path all run. The speed config (E2B + Kokoro) is the default.
- **Latency:** compute pipeline first-audio is **p50 ~271 ms** (under the 500 ms budget). The *full* path including endpoint silence detection is ~590 ms today; the endpoint window (~300 ms) is the bottleneck, being optimized toward ~350–450 ms (eagerness-gated Smart-Turn + speculative LLM start).
- **Barge-in** (interrupting Reachy mid-sentence) is **parked on the `feature/barge-in` branch** — it needs STT/LLM/TTS on a single persistent MLX worker thread (MLX streams are thread-local), plus AEC. It works on the physical robot (hardware AEC); Mac dev needs software AEC.
- **Vision** adds ~1 s on vision turns (image prefill); a clean serial measurement is pending.
- **Emotive TTS** is an open quality gap: Kokoro is fast but flat. Expressive engines (Orpheus, Chatterbox-Turbo, Sesame CSM) are either too slow for real-time on M3 or not yet integrated — `chatterbox` is wired but parked. See `docs/FUTURE_WORK.md`.
- **`--mode wav`** (offline driver) requires the `benchmarks/` harness and only runs from a source checkout.

## License

Apache-2.0.
