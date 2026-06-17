# ReachyChatOffline

**A fully-local, low-latency, voice-to-voice conversational app for the [Reachy Mini](https://www.pollen-robotics.com/reachy-mini/) robot — running entirely on Apple Silicon (MLX/Metal). No cloud, no API keys, no data leaving your Mac.**

Talk to Reachy and it talks back, in about half a second, with the robot turning to look at whoever is speaking and reacting with emotion — all computed on-device.

## Quick start

After the one-time [setup](#getting-started), from the project directory:

```bash
cd ReachyChatOffline

# Talk to Reachy — live mic, vision on, turbo voice, + robot embodiment:
PYTHONPATH=src .venv/bin/python -m reachy_chat.pipeline.app --robot
```

`--robot` expects the **Reachy Mini daemon at `http://localhost:8000`**. Start a simulator daemon in another terminal first:

```bash
.venv/bin/reachy-mini-daemon --mockup-sim      # headless mock daemon on localhost:8000
```

(For a physical **Wireless** robot, point at it instead with `--robot-url http://reachy-mini.local:8000`.)

Common variations:

```bash
# Voice only, no robot:
PYTHONPATH=src .venv/bin/python -m reachy_chat.pipeline.app

# Lowest-latency voice (Kokoro ~150 ms) instead of the turbo voice:
PYTHONPATH=src .venv/bin/python -m reachy_chat.pipeline.app --tts kokoro

# Low-RAM machine — skip the second (vision) model:
PYTHONPATH=src .venv/bin/python -m reachy_chat.pipeline.app --no-vision

# Wake-word mode — idle until "Hey Reachy":
PYTHONPATH=src .venv/bin/python -m reachy_chat.pipeline.app --wake
```

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

The app is a streaming cascade where each stage overlaps the next; robot embodiment is fully decoupled from the voice path. For the full rationale, component choices, and measured latency/memory budgets, see:

- **[`docs/architecture.md`](docs/architecture.md)** — stack rationale and the measured voice-to-voice budget.
- **[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md)** — consolidated performance report (per-component + end-to-end p50/p95).
- **[`plan.md`](plan.md)** — the living plan and design decisions.
- **[`docs/FUTURE_WORK.md`](docs/FUTURE_WORK.md)** — deferred experiments (emotive TTS, speculative LLM start, etc.).

```
mic → VAD/endpoint → STT(finalize) → LLM(stream clauses) → TTS(stream) → speaker
         (Silero +     (Parakeet)      (Gemma 4)            (Kokoro)
          Smart-Turn)
                                  └── robot look-at-speaker + emotion (decoupled loop)
```

## Prerequisites

- **macOS on Apple Silicon** (M1/M2/M3/M4). The MLX wheels are arm64-mac only — this will not run on Intel Macs or other platforms.
- **~16–24 GB RAM.** The speed config (E2B + Kokoro) needs ~7.5 GB resident; the quality config (E4B + vision) needs ~12.5 GB. A 24 GB machine is the deploy target; 16 GB works for the speed config.
- **[`uv`](https://docs.astral.sh/uv/)** — the installer/runner (`brew install uv`). It downloads the correct Python automatically.
- **A microphone and speakers/headphones.** Headphones are recommended for cleaner audio on a Mac (no hardware echo cancellation; see Troubleshooting).
- **Hugging Face access** — models download from the Hub on first run (several GB, one-time). Most are public; logging in avoids rate limits: `uv run hf auth login` (or `huggingface-cli login`).
- **`ffmpeg`** (optional) — handy for audio handling: `brew install ffmpeg`.

## Getting Started

The project ships a pinned Python version (`.python-version` = 3.12) and a console entry point, so `uv` selects and downloads the right interpreter automatically:

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

> **Wrong-Python guard:** the app requires Python 3.10–3.12 (verified on 3.12). On any other interpreter it exits immediately with a clear message instead of a cryptic import error:
> ```
> Reachy Chat needs Python 3.10–3.12; you have 3.13.
> Recreate the environment with the right interpreter:
>     uv venv --python 3.12 && uv pip install -e .
> ```

Now choose a setup path below depending on whether you have a physical robot.

---

## Setup Path A — Reachy Mini Simulator (no hardware)

You can run the **full** app — including `--robot` (look-at-speaker, emotions) and `--vision` — against a simulated robot. The robot daemon serves a REST API on `http://localhost:8000`; the app just needs *something* answering there.

There are two simulator options:

**1. Headless mock daemon (fastest, no GUI, no MuJoCo)** — perfect for exercising the robot/vision code paths:

```bash
uv run reachy-mini-daemon --mockup-sim
```

This starts the daemon on `localhost:8000` with no 3D window and no MuJoCo dependency. Robot motion commands are accepted (and acknowledged) so look-at-speaker and emotion calls succeed without hardware.

**2. Full MuJoCo physics simulator (3D viewer)** — see the robot actually move:

```bash
# Install the simulator extra once:
uv pip install -e ".[sim]"     # pulls in reachy-mini[mujoco]

# On macOS, MuJoCo's GUI needs the mjpython launcher (a plain `--sim` may not open the window):
uv run mjpython -m reachy_mini.daemon.app.main --sim
# (add --scene minimal for a table with objects)
```

> macOS note: `uv` can be finicky with MuJoCo wheels. If the 3D sim misbehaves, the Pollen docs recommend installing the `mujoco` package with plain `pip`, or just use `--mockup-sim` above. See `.reference/reachy_mini/docs/source/platforms/simulation/get_started.md`.

Alternatively, the **Reachy Mini Control** desktop app ([download](https://hf.co/reachy-mini/#/download)) can host a local simulated daemon on `localhost:8000` too.

With a daemon running on `localhost:8000`, run the app with `--robot` (and optionally `--vision`) — see **Running it** below. (The voice pipeline itself runs fine with **no** daemon; you only need it for `--robot`/`--vision`.)

---

## Setup Path B — Physical Reachy Mini

1. **Assemble and connect your robot** (Lite or Wireless) following the Pollen Robotics guides, and install **[Reachy Mini Control](https://hf.co/reachy-mini/#/download)** to bring it online and update its firmware. See `.reference/reachy_mini/docs/source/platforms/`.

2. **Find the daemon URL:**
   - **Reachy Mini Lite** (USB-tethered, daemon runs on *your Mac*): `http://localhost:8000`.
   - **Reachy Mini Wireless** (onboard Raspberry Pi runs the daemon): `http://reachy-mini.local:8000` (mDNS). If `reachy-mini.local` doesn't resolve on your network, use the robot's IP, e.g. `http://192.168.1.42:8000`.

3. **Run the app** pointing at that daemon:
   ```bash
   uv run reachy-chat --mode live --robot --robot-url http://reachy-mini.local:8000 --vision
   ```

**What differs from the simulator:**
- **Real motion and a real camera** — look-at-speaker physically turns the head/body, and `--vision` uses the robot's onboard camera frame.
- **Hardware echo cancellation / barge-in** — the robot's **XVF3800** audio front-end does hardware AEC, so the mic doesn't hear Reachy's own TTS. On the physical robot, barge-in (cutting Reachy off mid-sentence) is feasible. On a bare Mac there is no hardware AEC, so use **headphones** to avoid the speaker self-triggering the mic. (Software AEC for Mac dev is future work; see `docs/FUTURE_WORK.md`.)

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
