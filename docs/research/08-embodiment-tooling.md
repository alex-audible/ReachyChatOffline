# 08 — Embodiment Tooling: Driving Reachy's Body From Our Local LLM

**Status:** Research only (no code changes). **Date:** June 2026.
**Question:** How do we make Reachy Mini react emotionally/physically to what our LLM
says, and obey commands like "dance", "nod", "shake your head" — given that our LLM is a
*very basic* local **Gemma 4 E2B** (mlx-lm), text-only, streamed via `stream_clauses()`,
with **no** structured function-calling?

**TL;DR recommendation.** Adopt **inline embodiment tags** the LLM emits in its text
(e.g. `I'm really sad <reachyemote:sad>` / `<reachydo:dance>`), parsed out of the streaming
clause output with a small **state-machine / partial-tag buffer** so tags never reach TTS and
the move fires at the spoken position. Keep the existing **`reply_to_move` sentiment heuristic
as the always-on backstop** so the robot is expressive even when E2B forgets to tag. Do *not*
adopt JSON function-calling for E2B (it is unreliable below ~7–10B and adds latency). This is a
hybrid: tags are the *opt-in command/emphasis* channel; sentiment mapping is the *guaranteed*
baseline. Both feed the existing decoupled `MotionQueue`.

---

## 1. What the reference conversation app does

Source: `.reference/reachy_mini_conversation_app/` (Pollen Robotics' official app). It is built
around **OpenAI/Gemini Realtime function-calling** with a large cloud model — the opposite end of
the spectrum from our setup, but its *motion vocabulary* and *queue architecture* are directly
reusable.

### 1.1 How the LLM invokes the body — function calling

- Tools are `Tool` subclasses with a `name`, `description`, and JSON-Schema
  `parameters_schema`, exposed to the model as function specs
  (`tools/core_tools.py:68` `Tool.spec()` → `{"type":"function","name":...,"parameters":...}`).
- The Realtime backend streams `response.function_call_arguments.done` events; the app parses the
  JSON args and dispatches to the tool (`base_realtime.py:840`, `core_tools.py:dispatch_tool_call`).
- Tools **never touch motors directly**. They `movement_manager.queue_move(...)` onto a 60 Hz
  control loop (`moves.py` `MovementManager`), exactly mirroring our `MotionQueue`. Primary moves
  (emotions/dances/goto) run sequentially; secondary offsets (speech sway, face tracking) are
  additive on top (`moves.py:135` `combine_full_body`). `needs_response=False` tools skip the
  spoken follow-up so a dance doesn't trigger chatter (`core_tools.py:62`).
- A `BackgroundToolManager` runs tool calls off the audio path so the voice stream is never blocked.

This is the same "LLM → tool → queue → control loop" indirection documented in
`skills/ai-integration.md` and already implemented in our `robot/motion_queue.py` + `robot/tools.py`.

### 1.2 The emotion vocabulary (`tools/play_emotion.py`)

The app uses the **same dataset we do**: `pollen-robotics/reachy-mini-emotions-library`
(`play_emotion.py:18`). It does **not** expose 81 raw move IDs to the LLM. Instead it exposes a
compact **intent enum** (`EMOTION_INTENTS`, `play_emotion.py:26`) and maps each intent to one or
more real move IDs (`_INTENT_TO_MOVES`, `play_emotion.py:126`):

> `random, happy, excited, loving, grateful, success, thinking, attentive, confused,
> uncertain, sad, downcast, lonely, angry, irritated, displeased, disgusted, scared,
> anxious, surprised, amazed, calming, relief, impatient, embarrassed, bored, tired,
> sleepy, yes, yes_understanding, no, no_sad, no_excited, no_firm, welcoming, greeting,
> goodbye, go_away, helpful, dance, electric, dying`

Notable design points worth copying:
- **Nuanced yes/no** intents (`no_sad`, `no_excited`, `no_firm`, `yes_understanding`) because a
  flat "no" loses meaning (`play_emotion.py:172` `_KEYWORD_INTENTS`).
- A **curated default pool** (`_EXCELLENT_MOVES` + `_OK_CLEAR_MOVES`, `play_emotion.py:71`) used
  for `random` so unrecognised requests still look good rather than picking an ugly move.
- Resolution normalises accents/case and falls back to a random *curated* emotion if nothing
  matches (`resolve_emotion_name` / `random_curated_emotion`). **Our `robot/tools.py`
  `_EMOTION_INTENTS` is already a subset of this — we are aligned.**

### 1.3 The dance vocabulary (`tools/dance.py` + `reachy_mini_dances_library`)

Dances come from a *separate* library, `reachy_mini_dances_library` (symbolic, math-defined
moves — see `skills/symbolic-motion.md`), not the emotions dataset. Confirmed live from the
installed package, `AVAILABLE_MOVES` has 20 named dances:

> `simple_nod, head_tilt_roll, side_to_side_sway, dizzy_spin, stumble_and_recover,
> headbanger_combo, interwoven_spirals, sharp_side_tilt, side_peekaboo, yeah_nod,
> uh_huh_tilt, neck_recoil, chin_lead, groovy_sway_and_roll, chicken_peck,
> side_glance_flick, polyrhythm_combo, grid_snap, pendulum_swing, jackson_square`

The `dance` tool's enum is the full list with descriptions; `move` omitted → random
(`dance.py:43`). **Important for us:** this library may not be installed, and our daemon plays
moves by name from a HF *dataset* via REST. Our `expression.py`/`tools.py` instead use the
**dance-flavoured moves inside the emotions dataset** (`dance1/2/3`). For an offline build we should
keep using `dance2`/`dance3` from the emotions dataset (already wired in `tools.py:_DANCE_MOVES`)
rather than depend on `reachy_mini_dances_library`.

### 1.4 Gestures (`tools/move_head.py`, `head_tracking.py`)

- `move_head(direction)` — enum `left/right/up/down/front`, mapped to fixed head deltas
  (`move_head.py:32`; left/right = ±40° yaw, up/down = ∓30° pitch). **Identical to our
  `tools.py:_HEAD_DIRECTIONS`.**
- `head_tracking(start: bool)` — toggles camera/DoA face tracking.
- There is **no dedicated "nod" or "shake head" tool** in the reference — those live as
  `simple_nod`/`yeah_nod`/`uh_huh_tilt` *dances*, or as the emotions `yes1`/`no1`. Our `tools.py`
  already adds a synthesized `nod()` from goto commands; we should add a symmetric `shake()`.

### 1.5 Profiles (personality + enabled tools)

Each profile is a directory with `instructions.txt` (system prompt, supports `[include]`
expansion from a shared `prompts/` library, `prompts.py:18`) and `tools.txt` (newline list of
enabled tool names, `core_tools.py:319`). The default profile enables:

> `dance, stop_dance, play_emotion, stop_emotion, camera, idle_do_nothing, head_tracking,
> move_head, remember, forget`

The **default prompt's movement guidance is minimal** — it does *not* enumerate emotions or teach
a syntax; it relies on the model being smart enough to call tools ("The head can move
(left/right/up/down/front). Enable head tracking when looking at a person"). The character lives
almost entirely in tone/identity text (see `mars_rover`, `noir_detective`, `bored_teenager`
profiles). **Lesson for us:** with a *large* model you can under-specify and it figures out tool
use. With **E2B we must do the opposite — give an explicit, tiny, example-rich tag grammar.**

---

## 2. Our current building blocks (already implemented)

| Piece | File | Role |
|---|---|---|
| `reply_to_move(text)` | `robot/expression.py` | Deterministic keyword/sentiment → 1 of 81 real emotion moves. Already the backstop. |
| `RobotPresence` | `robot/presence.py` | State cues (listening/thinking/speaking/idle) + nods/idle micro-motion + DoA look-at. Background loop. |
| `MotionQueue` + `PlayMoveCommand`/`GotoCommand` | `robot/motion_queue.py` | Thread-safe serial executor, `interrupt=True` pre-empts. The single sink for all motion. |
| `RobotTools` | `robot/tools.py` | `move_head/look_at/play_emotion/dance/nod/head_tracking` handlers + OpenAI-style `TOOL_SCHEMAS` (unused by E2B today). |
| `ReachyClient` | `robot/client.py` | REST wrapper; `play_recorded_move(name, dataset)`, `goto`, limits, units. |
| **Pipeline** | `pipeline/app.py` `_RobotLink`, `pipeline/engines.py` `stream_clauses` | The voice loop. `_RobotLink.express(reply)` already calls `reply_to_move` after each turn; `on_state` drives presence. |

Key facts about our streaming path (from `engines.py`):
- `stream_clauses()` yields **whole speakable clauses** (sentence-end, or comma/clause break when
  long) — not raw tokens. A tag like `<reachydo:dance>` will arrive *inside* a clause string.
- `run_turn()` (`app.py:166`) iterates clauses; for each clause it calls `tts.synth_stream(clause)`
  and plays audio. After the whole reply, it calls `robot.express(reply)`.
- Everything robot-side is **best-effort and off the latency path** (`_RobotLink` wraps all calls
  in `try/except`, motion runs on `MotionQueue`'s worker thread).

This is the ideal insertion point: we already have a per-clause loop and a post-reply hook.

---

## 3. The three options, compared (for E2B specifically)

### (a) Inline embodiment tags parsed from the text stream  ← RECOMMENDED

The LLM writes ordinary text and drops short literal markers where a move should happen, e.g.
`Hi there! <reachyemote:happy> Want to see something? <reachydo:dance>`. We strip the tags before
TTS and enqueue the matching move at that point in the speech.

**Pros**
- **Plays to the model's strengths.** LLMs see vast amounts of HTML/code, so angle-bracket
  markers interleaved with prose are *in-distribution* and don't fight the model's tendencies.
  The "When Robots Get Chatty" system chose angle-bracket inline action calls for exactly this
  reason and reports higher robustness than fighting the model with rigid formats.
- **No extra inference round-trips** → no added latency. JSON tool-calling typically chains a
  second model call (decide → call → observe → speak); inline tags are emitted *in the single
  speech stream we already generate*.
- **Natural time-sync for free.** Because the tag sits between words, we can fire the move when we
  reach that clause — "sad" emote lands on the word "sad".
- **Trivial to extend.** New action = one line in the prompt + one mapping entry.
- **Graceful degradation.** A malformed/unknown tag is just stripped → silence, never a crash.
- **Composable with the backstop.** Tags handle *commands* and *emphasis*; `reply_to_move` still
  guarantees a baseline reaction.

**Cons**
- Small models emit malformed/hallucinated tags some of the time (quantified in §3 risks). Must be
  defensive: whitelist, fuzzy-match, strip-on-fail.
- Tag can be split across clause/token boundaries → need a tiny buffer (solved, §4.3).
- Pollutes the text channel slightly (mitigated by strict stripping; if a raw `<reachy...>`
  ever leaks to TTS it must be inaudible — we strip *anything* matching the tag shape).

### (b) Structured / JSON function-calling

Hand E2B the `TOOL_SCHEMAS` and parse tool-call JSON (the reference app's approach).

**Pros:** schema-validated args; same code path as the reference; clean separation.
**Cons (decisive for E2B):**
- **Reliability collapses on small models.** Local models <~10B parameters suffer *eager
  invocation*, *wrong tool selection*, and *invalid arguments* (Docker's practical eval; the
  promptlayer/Fowler write-ups). mlx-lm's Gemma E2B has no first-class tool-calling harness and
  E2B was not selected for tool use.
- **Latency.** Tool-calling generally chains multiple inference rounds — antithetical to our
  <500 ms first-audio budget.
- **Streaming friction.** Our pipeline streams *speech clauses*; interleaving structured tool
  calls into that single stream is awkward (the reference relies on a Realtime API's separate
  function-call event channel, which mlx-lm does not provide).

### (c) Keyword/sentiment mapping we already have (`reply_to_move`)

**Pros:** zero model dependence, deterministic, instant, already wired into `_RobotLink.express`.
Grounded in the real 81-move dataset. Cannot break.
**Cons:** cannot follow explicit *commands* ("do a dance now", "shake your head"); reacts only to
*its own words*; no fine timing (fires once after the whole reply); coarse (one move per turn).

### Verdict

(b) is wrong for E2B. (c) is a great *floor* but can't obey commands. **(a) layered on top of (c)**
gives us command-following + emphasis with no latency cost and a guaranteed fallback. This matches
the literature's trajectory: GenEM/Haru show LLMs *can* compose expressive behavior from a small
action API given few-shot examples, while the small-model gesture study (below) shows you must not
*rely* on small models to always emit valid markers — hence keep the heuristic backstop.

---

## 4. Recommended design — inline embodiment tags

### 4.1 Tag syntax

Two tag families, deliberately short, lowercase, angle-bracketed, colon-separated, with a fixed
prefix `reachy` so the strip regex is unambiguous and a stray `<...>` in normal prose is never
caught:

```
<reachyemote:NAME>     # play an emotion move (reaction / feeling)
<reachydo:COMMAND>     # perform a physical command (gesture / dance / look)
```

- One token per tag, `[a-z_]+` only. No arguments, no nesting, no JSON — the simplest thing E2B
  can reliably produce. (Repetition like "nod twice" → emit two `<reachydo:nod>` or just map
  `nod` to a double-nod.)
- Tags may appear anywhere in the text; placement = timing. Convention taught to the model:
  *put the tag immediately after the word it relates to.*

### 4.2 Command + emotion vocabulary

Keep it **small** (small models pick badly from long enums — the dance lib's 20-name enum is
already too much for E2B). Proposed minimal set, every entry mapped to something that already
exists in our stack:

**`<reachyemote:…>`** (→ `reply_to_move`-style → real emotion-library move):

| tag | move (emotions dataset) |
|---|---|
| `happy` | `cheerful1` / `laughing2` |
| `sad` | `sad1` |
| `excited` | `enthusiastic1` / `dance3` |
| `curious` | `curious1` |
| `surprised` | `amazed1` / `surprised1` |
| `confused` | `confused1` |
| `grateful` | `grateful1` |
| `proud` / `success` | `success1` |
| `scared` | `scared1` |
| `angry` | `irritated1` |
| `love` | `loving1` |
| `bored` | `boredom2` |
| `tired` | `exhausted1` |

**`<reachydo:…>`** (→ `RobotTools` handler / `MotionQueue` command):

| tag | action |
|---|---|
| `nod` | `RobotTools.nod()` (yes) |
| `shake` | **new** `RobotTools.shake()` (no) — symmetric to `nod`, or play `no1` |
| `dance` | `RobotTools.dance()` (random `dance2`/`dance3`) |
| `look_left` / `look_right` / `look_up` / `look_down` / `look_front` | `RobotTools.move_head(dir)` |
| `wave` / `hello` | `welcoming2` |
| `bye` | `go_away1` |

This reuses `tools.py` almost verbatim; the only new handler is `shake()`
(`HeadPose(yaw=±18°)` oscillation, mirroring the existing `nod()` build). A single resolver maps
both tag families onto either a `PlayMoveCommand` or a `GotoCommand` sequence.

### 4.3 Parsing/stripping on the STREAMING clause output

We must (i) never let a tag reach TTS, (ii) fire the move at the right spoken position, (iii)
handle a tag split across two clauses. Use the **buffered state-machine pattern** LiveKit
documents for stripping `<think>` blocks from a streaming LLM before TTS — proven for exactly this
"remove inline markup mid-stream, keep partial-tag buffer" problem.

Sketch (sits between `stream_clauses` and `tts.synth_stream` in `run_turn`):

```python
TAG_RE = re.compile(r"<reachy(emote|do):([a-z_]{1,20})>")
# anything that *might* be the start of a tag, held back until we can decide:
PARTIAL = re.compile(r"<r?e?a?c?h?y?\??.{0,24}$")  # conservative trailing-prefix guard

class TagStripper:
    def __init__(self, fire):       # fire(kind, name) -> enqueue move (non-blocking)
        self._buf = ""; self._fire = fire
    def feed(self, clause: str) -> str:
        self._buf += clause
        clean = []
        while True:
            m = TAG_RE.search(self._buf)
            if not m: break
            clean.append(self._buf[:m.start()])
            self._fire(m.group(1), m.group(2))      # fire at spoken position
            self._buf = self._buf[m.end():]
        # hold back only a possible partial tag at the very end; emit the rest
        keep = 0
        pm = PARTIAL.search(self._buf)
        if pm: keep = len(self._buf) - pm.start()
        out = "".join(clean) + self._buf[:len(self._buf)-keep]
        self._buf = self._buf[len(self._buf)-keep:]
        return out
    def flush(self) -> str:
        out, self._buf = self._buf, ""
        # final safety: strip any leftover tag-shaped junk so it never reaches TTS
        return TAG_RE.sub("", out)
```

In `run_turn` the loop becomes: for each `clause` from `stream_clauses`, `spoken =
stripper.feed(clause)`; if `spoken.strip()`, `tts.synth_stream(spoken)`. After the loop,
`tail = stripper.flush()` and speak it. `fire(kind,name)` resolves the tag to a move and calls
`self._queue.enqueue(...)` — **non-blocking, on the motion thread**, so the voice path is untouched.

**Timing nuance:** because `synth_stream` for a clause runs *after* the tag in that clause is
parsed, enqueuing the move when we fire gives "move starts as the clause begins to speak" — close
enough to "synced to where the tag appears" for a desk robot. If we want tighter sync we can fire
the emote with `interrupt=True` (it pre-empts ambient nods) and let commands (`dance`,`nod`) queue
normally so they don't fight each other. (We deliberately do *not* try sub-clause word-level sync —
not worth the complexity given clause-granular TTS.)

### 4.4 Mapping tags → existing moves / MotionQueue

`fire(kind, name)` does:
- `emote` → look up the emotion table (§4.2); reuse `expression.EMOTION_LIBRARY` validation so we
  never enqueue a non-existent move; `queue.enqueue(PlayMoveCommand(move), interrupt=True)`.
- `do` → dispatch through the existing `RobotTools` handler (`nod/shake/dance/move_head/…`), which
  already enqueues `GotoCommand`/`PlayMoveCommand`.
- Unknown/garbled `name` → **fuzzy-match** against the whitelist (e.g. `difflib.get_close_matches`);
  if still unmatched, drop silently. This absorbs the small-model error modes (§3).

### 4.5 System-prompt snippet to teach E2B (drop into `DEFAULT_SYSTEM_PROMPT`)

E2B needs explicit, example-rich, *short* instruction (unlike the reference's terse prompt). Note
this **replaces** the current prompt's blanket ban on stage directions — we now *want* a controlled
form of them:

```
You control a small desk robot's body with short inline tags placed directly in your
reply. Use them sparingly and naturally — they are silent (never spoken aloud).

Emotions (show how you feel):  <reachyemote:NAME>
  NAME = happy | sad | excited | curious | surprised | confused | grateful |
         proud | scared | angry | love | bored | tired
Actions (do a movement):       <reachydo:NAME>
  NAME = nod | shake | dance | wave | bye |
         look_left | look_right | look_up | look_down | look_front

Rules:
- Put the tag right after the words it matches. Example:
  "That's wonderful news! <reachyemote:happy>"
  "No, I don't think so. <reachydo:shake>"
- When the user asks you to move ("dance", "nod", "shake your head", "look left"),
  ALWAYS emit the matching <reachydo:...> tag.
- Use at most one or two tags per reply. Only use tag names from the lists above.
- Everything outside the tags is spoken normally — no asterisks, emojis, or markdown.
```

(Keep the rest of `DEFAULT_SYSTEM_PROMPT` as-is. The persona is prefilled once into the KV cache,
`engines.py:_rebuild`, so this costs nothing per turn.)

### 4.6 How it plugs into the current pipeline

```
stream_clauses(transcript)  ─► TagStripper.feed(clause) ─► spoken text ─► tts.synth_stream ─► speaker
                                      │ fire(kind,name)
                                      ▼
                            _RobotLink resolve ─► MotionQueue.enqueue(PlayMove/Goto)  (decoupled thread)
                                      ▲
            (after reply)  reply_to_move(full_reply) ───────────────┘   ← UNCHANGED backstop
```

Concretely, in `pipeline/app.py`:
- `_RobotLink` gains a `make_stripper()` returning a `TagStripper` whose `fire` resolves+enqueues.
- `run_turn` wraps each clause with `stripper.feed(...)` before TTS, and speaks `stripper.flush()`
  at the end.
- `_RobotLink.express(reply)` (the `reply_to_move` call) **stays** — it now only fires when no
  `<reachyemote:…>` tag fired this turn (track a flag), so we don't double-emote. Commands and the
  sentiment backstop coexist.

No change to `motion_queue.py`, `client.py`, `presence.py`, or `expression.py` is required; the
only new code is `TagStripper`, a `shake()` handler in `tools.py`, the resolver table, and the
prompt edit. All additive, all off the latency path.

---

## 5. Risks & mitigations

1. **Will E2B reliably emit well-formed tags?** Partly. The most directly comparable study —
   *Simultaneous text and gesture generation for social robots with small language models* — had
   models emit **inline gesture tags** (`[GEST] … [\GEST]`) and measured error rates of
   **~42% (1B), ~29% (3B), ~27% (8B)** (non-existent gesture names, impossible parameters, etc.).
   E2B (~2B effective) sits in the worst band. **Mitigations:** (i) tiny whitelist + fuzzy-match +
   silent-drop so bad tags are harmless; (ii) keep the deterministic `reply_to_move` backstop so
   *some* expression always happens; (iii) no-argument tags (the study's errors were largely
   *parameter* errors — we have no parameters); (iv) few-shot examples in the prompt (GenEM shows
   in-context examples sharply improve small-model behavior composition). That paper's ultimate fix
   was a constrained "gesture head" decoder — overkill for us, but the takeaway is *constrain the
   output space and never trust it blindly*, which our whitelist+backstop does.

2. **Tag leaks to TTS (audible "less-than reachy do dance").** Prevented by `TagStripper.flush()`
   doing a final `TAG_RE.sub("", …)`, plus the partial-prefix hold-back so a tag split across
   clauses is never half-spoken. Defense-in-depth: a broad `<reachy...>`-shaped cleanup before
   any text hits `synth_stream`.

3. **Latency / blocking the voice path.** None added: tag parsing is pure-Python regex on strings
   already in hand; move firing is `queue.enqueue(...)` onto the existing background `MotionQueue`,
   wrapped in `_RobotLink`'s try/except. The decoupling guarantee in `presence.py`/`motion_queue.py`
   is preserved.

4. **Moves fighting each other** (emote vs ambient nod vs command). Use the existing
   `interrupt=True` for emotes (pre-empts ambient presence nods), let `<reachydo:…>` commands queue
   serially. `RobotPresence` speaking-state nods already yield to enqueued moves.

5. **Dance library availability.** Don't depend on `reachy_mini_dances_library` for an offline
   build; `<reachydo:dance>` → `dance2`/`dance3` from the emotions dataset (already in
   `tools.py:_DANCE_MOVES`).

6. **Over-tagging / spammy motion.** Prompt caps at 1–2 tags/reply; resolver can rate-limit
   (ignore a 3rd tag per turn).

---

## 6. Sources

Reference code (local):
- `.reference/reachy_mini_conversation_app/src/reachy_mini_conversation_app/tools/play_emotion.py`
  (intent enum, `_INTENT_TO_MOVES`, curated pools, resolver), `tools/dance.py`,
  `tools/move_head.py`, `tools/head_tracking.py`, `tools/core_tools.py` (Tool spec/dispatch),
  `moves.py` (`MovementManager`, 60 Hz queue, primary/secondary fusion),
  `dance_emotion_moves.py`, `prompts.py`, `prompts/default_prompt.txt`,
  `profiles/default/{instructions,tools}.txt`.
- `.reference/reachy_mini/skills/ai-integration.md`, `symbolic-motion.md`, `motion-philosophy.md`,
  `interaction-patterns.md`, `rest-api.md`, `AGENTS.md`.
- Dance vocabulary confirmed live from installed `reachy_mini_dances_library.collection.dance.AVAILABLE_MOVES`.

Our code (local): `src/reachy_chat/robot/{expression,presence,motion_queue,tools,client}.py`,
`src/reachy_chat/pipeline/{engines,app}.py`.

Web / prior art:
- When Robots Get Chatty (inline angle-bracket action calls; chosen because in-distribution for
  LLMs) — https://arxiv.org/abs/2407.00518
- Simultaneous text and gesture generation for social robots with **small** language models
  (inline `[GEST]` tags; 1B/3B/8B error rates; gesture-head fix) —
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12122315/
- GenEM — Generative Expressive Robot Behaviors using LLMs (few-shot → robot API composition) —
  https://arxiv.org/abs/2401.14673 / https://generative-expressive-motion.github.io/
- Ain't Misbehavin' — LLM-generated expressive behavior for the Haru robot —
  https://dl.acm.org/doi/10.1145/3610978.3640562
- LiveKit — stripping inline `<think>` tags from a streaming LLM before TTS (buffered state-machine,
  partial-tag hold-back) — https://docs.livekit.io/reference/recipes/replacing_llm_output/
- Docker — Tool Calling with Local LLMs: practical eval (small-model failure modes: eager
  invocation, wrong tool, invalid args) —
  https://www.docker.com/blog/local-llm-tool-calling-a-practical-evaluation/
- LLM Agents vs Function Calling (multi-round latency of tool calling) —
  https://blog.promptlayer.com/llm-agents-vs-function-calling/
- W3C EmotionML / Azure SSML emotion styles (markup precedent for emotion in the text/speech
  channel) — https://www.w3.org/TR/emotionml/ ,
  https://medium.com/@brijeshrn/ssml-the-practical-standard-for-controlling-speech-synthesis-c52940314ffa
- Reachy Mini conversation app (function-calling architecture overview) —
  https://github.com/pollen-robotics/reachy_mini_conversation_app ;
  fully-local fork — https://github.com/dwain-barnes/reachy_mini_conversation_app_local
