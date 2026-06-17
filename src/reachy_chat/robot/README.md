# reachy_chat.robot

Robot-integration layer for the local voice assistant. Talks to the Reachy Mini
daemon over its REST API (default `http://localhost:8000`) and keeps all motion
**decoupled** from the voice pipeline — head movement may trail the audio and is
never on the speech latency path.

## Modules

| Module | Responsibility |
|---|---|
| `client.py` | `ReachyClient` — typed REST wrapper (goto, set_target, recorded moves, state, DoA, motors). Enforces motion limits and handles unit conversion. |
| `motion_queue.py` | `MotionQueue` — thread-safe command queue + control-loop worker thread. The LLM enqueues `GotoCommand` / `PlayMoveCommand`; the loop executes them smoothly and serially. |
| `look_at_speaker.py` | `LookAtSpeaker` — DoA-driven look-at controller in its own thread, with dead-zone, exponential smoothing and rate-limiting. |
| `tools.py` | `RobotTools` — LLM tool handlers (`move_head`, `look_at`, `play_emotion`, `dance`, `nod`, `head_tracking`) + OpenAI-style `TOOL_SCHEMAS`. Each enqueues and returns a short string. |
| `presence.py` | `RobotPresence` — embodiment/presence layer: maps conversation states (`idle`/`listening`/`thinking`/`speaking`) to subtle, lifelike head + antenna cues via a background loop on top of `MotionQueue` + `LookAtSpeaker`. Fully decoupled from voice latency; no-ops if the daemon is down. |

## Architecture

```
LLM -> tool call -> MotionQueue.enqueue() -> control loop -> ReachyClient -> daemon
                       LookAtSpeaker (own loop, polls DoA) -> ReachyClient -> daemon
```

## Units & conventions (confirmed against the live daemon)

- Head pose `roll`/`pitch`/`yaw` and `body_yaw` are **radians** on the wire.
  `ReachyClient` / `HeadPose` accept **degrees** and convert.
- Motion limits (clamped, never rejected): head pitch/roll ±40°, head yaw ±180°,
  body yaw ±160°.
- **DoA** (`/api/state/doa`) returns `{"angle": float_radians, "speech_detected": bool}`
  or **`null`** (no mic array / no source — the bare simulator returns `null`).
  DoA frame: `0`=left, `π/2`=front/back, `π`=right. `doa_angle_to_azimuth_deg`
  remaps this to a signed azimuth (0=front, +left, −right); front/back ambiguity
  is resolved toward front.
- `goto` returns `{"uuid": ...}`. While any goto/recorded move runs, `set_target`
  is **ignored** (`{"status":"ignored","reason":"move_running"}`), so the look-at
  controller uses `set_target` for real-time tracking and naturally yields to
  emotions/dances.

## Confirmed REST endpoints

- `POST /api/move/goto` — `{head_pose: XYZRPYPose, body_yaw, antennas, duration, interpolation}`
- `POST /api/move/set_target` — `{target_head_pose, target_body_yaw, target_antennas}`
- `POST /api/move/play/recorded-move-dataset/{dataset}/{move}`
- `POST /api/move/play/wake_up`, `POST /api/move/play/goto_sleep`
- `GET  /api/move/running`, `POST /api/move/stop` (`{uuid}`)
- `GET  /api/move/recorded-move-datasets/list/{dataset}`
- `GET  /api/state/full`, `/state/present_head_pose`, `/state/present_body_yaw`, `/state/doa`
- `GET  /api/motors/status`, `POST /api/motors/set_mode/{mode}`

Emotion/dance dataset: `pollen-robotics/reachy-mini-emotions-library` (81 moves
on the live simulator).

## Quick start

```python
from reachy_chat.robot import ReachyClient, MotionQueue, LookAtSpeaker, RobotTools

client = ReachyClient("http://localhost:8000")
look_at = LookAtSpeaker(client); look_at.start()
with MotionQueue(client) as queue:
    tools = RobotTools(client, queue, look_at)
    # expose tools.schemas to the LLM; dispatch tool calls:
    print(tools.dispatch("nod", {"times": 2}))
    print(tools.dispatch("play_emotion", {"emotion": "happy"}))
```

## Presence / embodiment (`presence.py`)

`RobotPresence` is the "make it feel alive" layer. It ties the `InteractionMachine`
states to subtle motion via a single background loop that sits on top of the
existing `MotionQueue` + `LookAtSpeaker` — so it inherits the decoupling guarantee:
**`on_state()` only flips a flag and returns; the loop does the motion.** Nothing
is on the speech latency path, and if the daemon is unreachable it logs and
no-ops (verified — a dead-port client neither blocks nor crashes the caller).

State → cue (all `minjerk`, small amplitudes, inside the limits):

| State | Embodiment | Pose / move used |
|---|---|---|
| `listening` | orient toward speaker + attentive tilt + antennas perk; enable DoA tracking | `goto` head `roll +7°, pitch −3°, yaw = DoA azimuth` (capped ±45°), antennas `(+0.35,+0.35)`, 0.6 s |
| `thinking` | brief "thinking beat": glance up & away, antennas relax, tracking paused | `goto` head `pitch −14° (up), yaw ±16° (random side)`, antennas `(−0.25,−0.25)`, 0.7 s |
| `speaking` | re-center, then a gentle nod every ~2.4 s with a small antenna wiggle | re-center `goto` (neutral, 0.8 s); per nod: `pitch +9°` → `pitch −3°` → neutral, antennas `(+0.18,−0.18)` then neutral, 0.32 s each |
| `idle` | return to neutral, then occasional subtle micro-motion (~every 6 s, jittered) so it never looks frozen | neutral `goto` (0.8 s); micro: random `yaw ±12°, pitch ±6°` + antenna twitch then drift back, 1.2 s |

`look_at(azimuth_deg)` points at a sound source (0 = front, + = left, − = right) by
delegating to `LookAtSpeaker.look_at_azimuth` (splits body + head, one smooth `goto`).
On hardware, feed it the DoA azimuth; on the bare sim DoA is `null`, so pass a
manual azimuth.

### API the live app should call

```python
from reachy_chat.robot.presence import RobotPresence

presence = RobotPresence(base_url="http://localhost:8000")  # owns its queue + look_at
presence.start()
# from the interaction driver, on every State transition:
presence.on_state("listening")   # "idle" / "listening" / "thinking" / "speaking"
presence.look_at(30)             # optional: point at a sound source (deg, +=left)
presence.stop()                  # returns to neutral, tears down owned helpers
```

Pass `queue=` / `look_at=` to share the app's existing `MotionQueue` /
`LookAtSpeaker` instead of having `RobotPresence` create+own its own (it only
starts/stops the helpers it created). `on_state` also accepts an Enum-like object
with a `.value` (e.g. the project's `State`).

## Reply → expression (`expression.py`)

The voice pipeline has **no structured LLM function-calling**, so the model can't
ask for a move directly. `expression.py` is a lightweight, deterministic
keyword/sentiment heuristic (no ML, no network) that maps an assistant **reply's
text** to one recorded emotion move, so the robot reacts expressively to its own
spoken words. Call it right after a reply is produced; it's off the speech
latency path and the actual move runs through the decoupled `MotionQueue`.

It is grounded in the **real 81 moves** of
`pollen-robotics/reachy-mini-emotions-library` (listed live from the running
daemon, not guessed). First matching rule wins; rules are ordered
most-specific-first; triggers match on whole words (so `hi`≠`this`, `no`≠`now`).

| Reply sentiment | Trigger examples | Move | Type |
|---|---|---|---|
| farewell | "goodbye", "bye", "see you", "take care" | `go_away1` | goodbye |
| greeting | "hello", "hi", "welcome", "how can I help" | `welcoming2` | wave/welcome |
| apology | "sorry", "my mistake", "oops" | `oops1` | small oops |
| grateful | "thank you", "appreciate" | `grateful1` | grateful |
| sad | "unfortunately", "bad news", "I can't help" | `sad1` | sad |
| excited | "amazing", "fantastic", "can't wait", "thrilled" | `enthusiastic1` | upbeat |
| happy | "happy", "glad", "sounds good", "of course" | `cheerful1` | cheerful |
| success | "done", "fixed", "it worked", "got it" | `success1` | proud success |
| agreement | "yes", "sure", "correct", "exactly" | `yes1` | nod/yes |
| negation | "no", "nope", "I disagree" | `no1` | no |
| surprised | "wow", "no way", "unbelievable" | `amazed1` | amazed |
| curious | "what would you", "tell me more", "I wonder" | `curious1` | curious tilt |
| thinking | "let me think", "hmm", "maybe", "it depends" | `thoughtful1` | thoughtful |
| confused | "I don't understand", "what do you mean" | `confused1` | confused |
| *fallback* | bare `?` → curious, bare `!` → cheerful | `curious1`/`cheerful1` | — |
| neutral | (nothing matched) | `None` | no move |

`map_reply()` returns `Expression(move, category, rationale)` (rationale = the
trigger that fired, handy for logging/tuning). All emittable move names are
asserted to exist in `EMOTION_LIBRARY`.

### API the live app should call

```python
from reachy_chat.robot.expression import reply_to_move
from reachy_chat.robot import PlayMoveCommand

move = reply_to_move(reply_text)          # -> "welcoming2" | ... | None
if move:
    queue.enqueue(PlayMoveCommand(move_name=move), interrupt=True)   # decoupled
```

Use `map_reply(reply_text)` instead of `reply_to_move` if you also want the
category + rationale for logging. Deterministic and fast (pure string matching),
so it is safe to call synchronously after each reply.

## Smoke test

```bash
.venv/bin/python scripts/smoke_robot.py            # against localhost:8000
.venv/bin/python scripts/smoke_presence.py         # presence layer, localhost:8000
.venv/bin/python scripts/smoke_robot_conversation.py  # reply→expression, localhost:8000
```

`smoke_robot.py` prints state + DoA, lists recorded moves, then runs
nod / look / emotion / reset. `smoke_presence.py` starts the presence loop, cycles
idle→listening→thinking→speaking→idle with realistic dwell times, fires `look_at`
at mock azimuths, and polls `/api/move/running` + present head pose to confirm
motion actually executes (**verified: every state drove the sim, no exceptions**).
`smoke_robot_conversation.py` runs a scripted conversation (greeting, excited,
question-back, agreement, thinking, farewell), maps each reply with
`reply_to_move`, glances the head, plays the chosen move, and confirms each
move's UUID appears in `/api/move/running` (**verified: 6/6 moves drove the sim
end to end**; add `--use-queue` to also exercise the `MotionQueue` app path).
(Under the command sandbox, localhost HTTP needs the sandbox disabled.)

If no daemon is listening on :8000, start the bundled mock simulator:

```bash
.venv/bin/reachy-mini-daemon --mockup-sim --headless --no-media \
    --no-wake-up-on-start --no-goto-sleep-on-stop --fastapi-port 8000
```
```
