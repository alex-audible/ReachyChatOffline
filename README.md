# ReachyChatOffline

**A fully-local, low-latency, voice-to-voice conversational app for the [Reachy Mini](https://www.pollen-robotics.com/reachy-mini/) robot — running entirely on Apple Silicon (MLX/Metal). No cloud, no API keys, no data leaving your Mac.**

Talk to Reachy and it talks back, in under a second. Runs on Apple Silicon Macs (M2/M3/M4/M5) with 16 GB+ RAM.

Will also clone your voice if you ask it - "Hey Reachy, can you clone my voice?"

## Setup

Needs an Apple Silicon Mac, the **[Reachy Mini Control](https://hf.co/reachy-mini/#/download)** app and [uv](https://docs.astral.sh/uv/). Install uv if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then clone and install:

```bash
git clone https://github.com/alex-audible/ReachyChatOffline.git
cd ReachyChatOffline
uv venv                   # creates .venv with Python 3.12
uv pip install -e .       # installs all dependencies into .venv
.venv/bin/hf auth login   # log in to Hugging Face (models download from there)
```

The models (Parakeet, Gemma 4, Kokoro/Chatterbox, Silero, Smart-Turn) download from Hugging Face on first run — **several GB, one-time**, cached in `~/.cache/huggingface`. You must be logged in (a free [account](https://huggingface.co/join) + [token](https://huggingface.co/settings/tokens)); some Gemma models are gated, so accept the license once on the model page if prompted.

## Run

Start the Reachy Mini Control app, connected to a real robot or running the simulator. Then activate ReachyChatOffline with a command  below:

```bash
# Talk to Reachy — default voice + camera vision and Gemma 4 E4B model (best quality):
PYTHONPATH=src .venv/bin/python -m reachy_chat.pipeline.app --robot --llm mlx-community/gemma-4-E4B-it-qat-4bit

# For computers with slower CPUs and less RAM, you can use a less capable LLM (defaults to Gemma 4 E2B):
PYTHONPATH=src .venv/bin/python -m reachy_chat.pipeline.app --robot

# And for very low-spec machines (voice cloning wont work) 
PYTHONPATH=src .venv/bin/python -m reachy_chat.pipeline.app --robot --no-vision --tts kokoro
```

Run it from a **real Terminal** (Terminal.app / iTerm) so macOS can grant microphone access. The first launch downloads models (several GB, one-time) and warms up — judge responsiveness from the **second** turn on.

`--robot` needs the **Reachy Mini daemon at `http://localhost:8000`**: install **[Reachy Mini Control](https://hf.co/reachy-mini/#/download)**, launch it, and activate the robot (or its simulator). For a physical Wireless unit, point elsewhere with `--robot-url http://reachy-mini.local:8000`.

Common options:

```bash
--tts kokoro                                     # lowest-latency voice (~150 ms) but no voice cloning
--no-vision                                      # disable the camera (saves memory)
--llm mlx-community/gemma-4-E4B-it-qat-4bit       # smarter, slower model
```

Mid-conversation, say **"Hey Reachy, can you clone my voice?"** to switch to a clone of your voice (see [Voice cloning](#voice-cloning)). Full flag reference is in [Flags](#flags).

## Features

- **Voice-to-voice, fully local** — STT, LLM, TTS and VAD all run on your Mac's GPU via MLX. Nothing is sent to a server.
- **Low latency** — a streaming **cascade-with-overlap** pipeline (STT during your speech → LLM on the finalized transcript → TTS on the first speakable clause). Measured first-audio **p50 ~271 ms** (E2B), under the 500 ms budget.
- **Natural turn-taking** — semantic endpointing (Silero VAD v6 + Smart-Turn v3) decides when you've actually finished, not just paused. Always-on and "Hey Reachy" wake-word modes.
- **Embodied presence** — uses mic **direction-of-arrival** to look at the active speaker, plus emotion-linked gestures, all on a decoupled loop that never eats the latency budget.
- **Camera vision** — ask "what do you see?" and Reachy answers from a webcam frame via Gemma 4 (`--no-vision` to disable).

## Architecture

```
mic → VAD/endpoint → STT(finalize) → LLM(stream clauses) → TTS(stream) → speaker
         (Silero +     (Parakeet)      (Gemma 4)            (Kokoro/
          Smart-Turn)                                        Chatterbox)
                                  └── robot look-at-speaker + emotion (decoupled loop)
```

## Flags

| Flag | Default | What it does |
|---|---|---|
| `--mode {live,wav}` | `live` | `live` = mic + speaker conversation; `wav` = offline file driver for latency measurement (source checkout only) |
| `--llm <repo>` | `mlx-community/gemma-4-E2B-it-qat-4bit` | LLM. Default **E2B** is latency-first; pass **`mlx-community/gemma-4-E4B-it-qat-4bit`** for a smarter, slower model |
| `--tts {kokoro,chatterbox-4bit,chatterbox-turbo-8bit}` | `chatterbox-turbo-8bit` | Voice. **chatterbox-turbo-8bit** = best-sounding + clonable (~0.7–0.8 s); **chatterbox-4bit** = emotive + clonable; **kokoro** = lowest latency (~150 ms) |
| `--voice <wav>` | off | (chatterbox) clone the voice from a reference WAV; omit for the built-in voice |
| `--exaggeration <0..~1.5>` | `0.5` | (chatterbox-4bit only) emotion intensity; turbo ignores it |
| `--robot` | off | Drive the daemon/sim: look-at-speaker + emotion gestures |
| `--robot-url <url>` | `http://localhost:8000` | Daemon address (e.g. `http://reachy-mini.local:8000` for a Wireless unit) |
| `--vision` / `--no-vision` | on | Camera vision via Gemma 4, loaded at startup; `--no-vision` skips it (saves memory on low-RAM machines) |
| `--wake` | off | Wake-word mode — idle until "Hey Reachy" |
| `--wav <path>` | `audio_samples/prompts/p3_vision.wav` | (wav mode) prompt WAV — 16 kHz mono |
| `--speak` | off | (wav mode) play the generated response aloud |

## Voice cloning

The Chatterbox voices can **clone a voice** from a short clip (~8–12 s of clean, expressive speech). Two ways:

1. **Mid-conversation** — say **"Hey Reachy, can you clone my voice?"** Reachy asks you to talk for ~10 s, records you, clones your voice, and carries on speaking in it.
2. **From the command line** — `--voice path/to/reference.wav` starts already speaking in that cloned voice.

Standalone demo (no conversation):

```bash
PYTHONPATH=src .venv/bin/python scripts/clone_voice_demo.py --play
```

## Troubleshooting

- **No transcript / empty audio** → microphone permission. Grant your terminal mic access in *System Settings → Privacy & Security → Microphone*, then restart it. `REACHY_DEBUG_AUDIO=1` dumps per-turn audio to `/tmp/reachy_turn_*.wav`.
- **`needs Python 3.10–3.12`** → wrong interpreter. Recreate: `uv venv --python 3.12 && uv pip install -e .`.
- **`--robot` does nothing / connection errors** → the daemon isn't reachable at `--robot-url`. Launch Reachy Mini Control and confirm `http://localhost:8000/docs` loads.
- **Reachy hears its own voice** → no hardware echo cancellation on a Mac. **Use headphones.** (The physical robot handles this in hardware.)

## Project layout

```
src/reachy_chat/
  pipeline/   # streaming engines (STT/LLM/TTS) + ConversationApp + audio I/O + app.py entry point
  audio/      # Silero VAD, Smart-Turn endpointer, wake-word, interaction state machine
  robot/      # Reachy daemon REST client, motion queue, look-at-speaker, emotion gestures
  vision/     # Gemma 4 webcam-frame path (mlx-vlm) + frame sources
  tts/        # Kokoro fix + Chatterbox emotive/cloning engine
benchmarks/   # latency harness + per-component + end-to-end benchmarks
audio_samples/prompts/   # 16 kHz test prompt WAVs
docs/         # architecture, performance report, research, experiments
```

## Status

- **Working end-to-end:** voice pipeline, turn-taking, wake-word, robot embodiment (verified against the simulator), and the vision path.
- **Latency:** compute first-audio **p50 ~271 ms**. The full path including endpoint silence detection is ~590 ms; the endpoint window is the bottleneck, being optimized.
- **Barge-in** (interrupting mid-sentence) is parked on the `feature/barge-in` branch — needs a single persistent MLX worker thread plus echo cancellation.

## License

Apache-2.0.
