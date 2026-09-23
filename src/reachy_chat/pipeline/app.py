"""ConversationApp — the real-time voice loop tying everything together.

Flow (see docs/architecture.md): mic/file → InteractionMachine (Silero VAD + Smart-Turn
endpoint, both interaction modes, barge-in) → STTEngine (streaming) → LLMEngine (warm,
thinking off, clause streaming) → TTSEngine (streaming) → speaker. Robot motion (look-at-
speaker, gestures) is optional and runs in its own decoupled loop — never on the latency path.

Two drivers:
  * ``run_wav(path)``  — offline, real-time-paced; measures the FULL "stop-talking → first-audio"
    latency INCLUDING endpoint detection. Testable without a mic. (EXP-E2E-2)
  * ``run_live()``     — mic + speaker, turns run in a worker thread so the frame pump keeps
    feeding the barge-in detector. Needs a microphone.
"""

from __future__ import annotations

import argparse
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]  # project root (…/ReachyChatOffline) when run from source
# `harness` (used only by the offline run_wav driver) lives in benchmarks/, which exists in a
# source checkout but NOT in an installed package. Add it to sys.path only when present so the
# live app (`reachy-chat --mode live`) works after `uv pip install -e .` without the benchmarks dir.
_BENCH = ROOT / "benchmarks"
if _BENCH.is_dir():
    sys.path.insert(0, str(_BENCH))

from reachy_chat.audio import Callbacks, InteractionConfig, InteractionMachine, Mode, State  # noqa: E402
from reachy_chat.pipeline.engines import DEFAULT_SYSTEM_PROMPT, LLMEngine, STTEngine, TTSEngine  # noqa: E402

FRAME = 512  # 32 ms @ 16 kHz

# TTS presets: preset name -> (repo, generate_kwargs). Preset names mirror the model repo
# (minus the mlx-community/ prefix) so --tts says exactly which model you get.
#   kokoro               — Kokoro-82M-bf16: fast (~150 ms TTFA), named voices (af_heart, …).
#   chatterbox-4bit      — chatterbox-4bit: emotive + voice-CLONABLE (~0.9-1.1 s TTFA);
#                          exaggeration/cfg_weight are real emotion knobs. Default voice or --voice.
#   chatterbox-turbo-8bit — chatterbox-turbo-8bit: MeanFlow few-step, ~25% faster (TTFA ~0.7-0.8 s,
#                          RTF ~0.22), peak-normalized, best-sounding clone. Ignores the emotion
#                          knob. Also the clone model, so cloning reuses it (no reload).
# All chatterbox* presets go through ChatterboxTTS; kokoro through the generic TTSEngine.
TTS_PRESETS = {
    "kokoro": ("mlx-community/Kokoro-82M-bf16", {"voice": "af_heart", "lang_code": "a"}),
    "chatterbox-4bit": ("mlx-community/chatterbox-4bit", {"exaggeration": 0.5, "cfg_weight": 0.5}),
    "chatterbox-turbo-8bit": ("mlx-community/chatterbox-turbo-8bit", {}),
}

# --- Audio device selection -------------------------------------------------------------
# The wired Reachy Mini exposes "Reachy Mini Audio" (USB, XMOS XVF3800: 16 kHz, onboard echo
# cancellation + DoA) with both a mic array and a speaker. Talking through it — rather than the
# Mac's mic/speakers — is what you want whenever the robot is plugged in, so "auto" picks it.
REACHY_AUDIO_NAME = "Reachy Mini Audio"


def resolve_audio_device(spec: str | None) -> int | None:
    """Map --audio-device to a sounddevice index, or None for the system default.

    spec: "auto" (Reachy Mini Audio if present, else system default) | "default" |
    a device index | a case-insensitive substring of the device name."""
    import sounddevice as sd
    if spec is None or spec == "default":
        return None
    devs = sd.query_devices()
    if spec == "auto":
        spec = REACHY_AUDIO_NAME
        strict = False
    else:
        strict = True
    if spec.isdigit():
        return int(spec)
    for i, d in enumerate(devs):
        if spec.lower() in d["name"].lower() and d["max_input_channels"] > 0 \
                and d["max_output_channels"] > 0:
            return i
    if strict:
        names = ", ".join(f"[{i}] {d['name']}" for i, d in enumerate(devs))
        raise SystemExit(f"--audio-device {spec!r}: no device with both input and output "
                         f"matches. Available: {names}")
    return None


def audio_device_label(device: int | None) -> str:
    import sounddevice as sd
    if device is None:
        i, o = sd.default.device
        return f"{sd.query_devices(i)['name']} (in) / {sd.query_devices(o)['name']} (out)  [system default]"
    return f"{sd.query_devices(device)['name']} (in+out)"


# --- Children's-library persona + catalogue (--context) -----------------------------------
# The system prompt is prefilled into the LLM's KV cache ONCE at startup, so a few hundred lines
# of catalogue cost a one-off prefill and nothing per turn.
LIBRARY_SYSTEM_PROMPT = (
    "You are Reachy, a friendly little robot who lives in the children's library and helps kids "
    "find great books to read. You are talking out loud with children, so keep it simple, warm and "
    "fun: short sentences, easy words, and usually one to three sentences per reply. "
    "When a kid asks for a book, recommend one straight away from the catalogue below: say the "
    "title and who wrote it, then one exciting sentence about why they'll like it. Don't ask "
    "questions before recommending. If you don't know their age or what they like, just pick a "
    "popular book that fits whatever they said. Only ask a question if you truly cannot choose, and "
    "never more than one. If they want another book, or didn't like your pick, give a different one "
    "without fuss. Only recommend books that are in the catalogue; if none fit, say so kindly and "
    "offer the closest one. If they ask about something other than books, chat kindly and briefly, "
    "and come back to books when it feels natural. Remember what was said earlier in the "
    "conversation. Speak plainly: never use asterisks, emojis, markdown, bullet points, headings, "
    "stage directions, or action descriptions like *(tilts head)* — express everything in spoken "
    "words."
)


def build_system_prompt(context_path: str | None) -> str:
    """No context → the generic desk-robot persona. With --context → the children's-library
    book-recommender persona with the catalogue file appended (one line per book)."""
    if not context_path:
        return DEFAULT_SYSTEM_PROMPT
    text = Path(context_path).read_text(encoding="utf-8").strip()
    return LIBRARY_SYSTEM_PROMPT + "\n\nBook catalogue:\n" + text


# --- Voice cloning by voice command ("Hey Reachy, can you clone my voice?") --------------
# Matches "clone/copy/mimic/imitate ... [my] voice" or "sound/talk/speak (just) like me".
_CLONE_RE = re.compile(
    r"\b(clone|copy|mimic|imitate)\b[\w\s]{0,20}?\bvoice\b"
    r"|\b(sound|talk|speak)\s+(just\s+)?like\s+me\b",
    re.IGNORECASE,
)
# Chatterbox model loaded for the clone when the active TTS isn't already Chatterbox (e.g.
# launched with Kokoro). Turbo is fast and was the preferred-sounding clone in testing.
CLONE_REPO = "mlx-community/chatterbox-turbo-8bit"
CLONE_SECONDS = 12.0  # >= the 10 s the user asked for, with headroom
CLONE_EXAGGERATION = 0.6  # a touch lively so the cloned voice has some life
# Spoken prompts for the clone flow (the first two in the CURRENT voice, the last in the NEW one).
CLONE_PROMPT = (
    "I would love to! When I say go, just talk to me for about ten seconds. "
    "Tell me your name, what you like to do for fun, your favourite subject at school, "
    "and your favourite foods. Okay — go!"
)
CLONE_WORKING = "Great, thank you! Give me just a moment while I learn your voice."
CLONE_DONE = "All done! This is what I sound like now. Pretty cool, right?"
CLONE_FAILED = "Hmm, I could not quite catch your voice that time. Let's try again later."


class _RobotLink:
    """Drives RobotPresence (state cues + DOA look-at) + reply→emotion moves on the Reachy
    daemon. Everything runs on the presence/motion-queue background threads, fully OFF the
    voice latency path; every call is best-effort (never raises into the pipeline)."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        from reachy_chat.robot import MotionQueue, PlayMoveCommand, ReachyClient
        from reachy_chat.robot.expression import reply_to_move
        from reachy_chat.robot.presence import RobotPresence
        self._client = ReachyClient(base_url=base_url)
        self._queue = MotionQueue(self._client)
        self._presence = RobotPresence(client=self._client, queue=self._queue)
        self._PlayMove = PlayMoveCommand
        self._reply_to_move = reply_to_move

    def start(self) -> None:
        self._queue.start()
        self._presence.start()

    def stop(self) -> None:
        try:
            self._presence.stop()
        finally:
            self._queue.stop()

    def on_state(self, state: str) -> None:
        try:
            self._presence.on_state(state)
        except Exception:
            pass

    def express(self, reply: str) -> None:
        """Play an emotion move matching the reply (interrupts ambient nods)."""
        try:
            mv = self._reply_to_move(reply)
            if mv:
                self._queue.enqueue(self._PlayMove(move_name=mv), interrupt=True)
        except Exception:
            pass


class ConversationApp:
    def __init__(self, llm_repo: str = "mlx-community/gemma-4-E2B-it-qat-4bit",
                 tts: str = "chatterbox-turbo-8bit", mode: Mode = Mode.ALWAYS_ON,
                 robot: bool = False, robot_url: str = "http://localhost:8000",
                 vision: bool = True, vision_model: str | None = None,
                 voice: str | None = None, exaggeration: float | None = None,
                 system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                 audio_device: int | None = None):
        repo, kwargs = TTS_PRESETS.get(tts, TTS_PRESETS["kokoro"])
        self.audio_device = audio_device  # sounddevice index for mic+speaker; None = system default
        vision_model = vision_model or llm_repo  # vision uses the SAME Gemma 4 model as chat
        self.llm_repo = llm_repo
        self.stt = STTEngine()
        self.llm = LLMEngine(repo=llm_repo, system_prompt=system_prompt)
        # Chatterbox is a different engine (emotive + voice-cloning); Kokoro et al. go through
        # the generic mlx-audio TTSEngine. Both expose the same synth_stream/synth/.sr contract.
        if tts.startswith("chatterbox"):
            from reachy_chat.tts.chatterbox import ChatterboxTTS
            exa = exaggeration if exaggeration is not None else kwargs.get("exaggeration", 0.5)
            self.tts = ChatterboxTTS(repo=repo, ref_audio=voice, exaggeration=exa,
                                     cfg_weight=kwargs.get("cfg_weight", 0.5))
            tts_voice = f"cloned: {voice}" if voice else "built-in default"
        else:
            self.tts = TTSEngine(repo=repo, generate_kwargs=kwargs)
            tts_voice = kwargs.get("voice", "default")
        self.robot = _RobotLink(robot_url) if robot else None
        self.vision = None
        if vision:
            try:
                # Light import (the heavy mlx-vlm load happens in warmup(), not here).
                from reachy_chat.vision.responder import VisionResponder
                self.vision = VisionResponder(model_name=vision_model)
            except Exception as e:  # vision is optional — never block the voice app
                print(f"[vision] disabled: {type(e).__name__}: {e}", flush=True)
        self._model_info = {
            "STT": "mlx-community/parakeet-tdt-0.6b-v2",
            "LLM": llm_repo,
            "TTS": f"{repo}  (voice: {tts_voice})",
            "Vision": (f"{vision_model}  (separate mlx-vlm model)" if self.vision is not None
                       else "off (--no-vision)"),
            "Audio": audio_device_label(audio_device),
        }
        self.machine = InteractionMachine(
            InteractionConfig(mode=mode),
            Callbacks(on_listen_start=self._on_listen_start,
                      on_endpoint=lambda evt: None,   # state→THINKING handled by the driver
                      on_barge_in=self._on_barge_in),
        )
        self._cancel = threading.Event()
        self._capturing = False

    # -- callbacks -----------------------------------------------------------
    def _on_listen_start(self) -> None:
        self.stt.start()
        self._capturing = True
        if self.robot is not None:
            self.robot.on_state("listening")  # orient to speaker + attentive cue (decoupled)

    def _on_barge_in(self) -> None:
        self._cancel.set()

    def print_models(self) -> None:
        print("Models in use:", flush=True)
        for role, name in self._model_info.items():
            print(f"  {role:7s} {name}", flush=True)
        if "E2B" in self.llm_repo:
            print("  Using the fast Gemma 4 E2B. For a smarter (slower) model, relaunch with:",
                  flush=True)
            print("    --llm mlx-community/gemma-4-E4B-it-qat-4bit", flush=True)

    def warmup(self) -> None:
        self.print_models()
        self.stt.warmup()
        for _ in self.llm.stream_clauses("Hello"):
            pass
        list(self.tts.synth_stream("Hello there."))
        # Load the vision model now (not lazily on the first visual query) so nothing stalls
        # mid-conversation. Degrade to no-vision if it can't load.
        if self.vision is not None:
            try:
                print("  loading vision model…", flush=True)
                self.vision.warmup()
            except Exception as e:
                print(f"[vision] warmup failed, disabling: {type(e).__name__}: {e}", flush=True)
                self.vision = None
        self.machine.warmup()

    # -- one turn: STT.finalize -> LLM clauses -> TTS -> play ----------------
    def run_turn(self, speaker=None, mic=None) -> dict:
        self._cancel.clear()
        self._capturing = False
        transcript = self.stt.finalize()
        if len(transcript.strip()) < 3:  # ignore noise / empty endpoints (don't reply to nothing)
            self.machine.finish_speaking()
            return {"transcript": transcript, "reply": "", "first_audio": None, "audio": []}
        # "Hey Reachy, can you clone my voice?" — capture the speaker and switch the TTS to a
        # clone of their voice. Needs mic + speaker (live mode only).
        if mic is not None and speaker is not None and _CLONE_RE.search(transcript):
            return self._clone_voice(transcript, speaker, mic)
        first_audio: float | None = None
        clauses: list[str] = []
        chunks: list[np.ndarray] = []
        # Visual queries ("what do you see?") are answered from a camera frame via the VLM, not the
        # text LLM. maybe_answer() is a fast heuristic that returns None for non-visual turns (no
        # VLM load), so it adds negligible overhead otherwise. (A vision turn adds ~1s.)
        vision_text = self.vision.maybe_answer(transcript) if self.vision is not None else None
        clause_source = ([vision_text] if vision_text is not None
                         else self.llm.stream_clauses(transcript, self._cancel))
        for clause in clause_source:
            if self._cancel.is_set():
                break
            clauses.append(clause)
            for audio in self.tts.synth_stream(clause, self._cancel):
                if self._cancel.is_set():
                    break
                if first_audio is None:
                    first_audio = time.perf_counter()
                    self.machine.begin_speaking()
                    if self.robot is not None:
                        self.robot.on_state("speaking")
                if speaker is not None:
                    speaker.play(audio)
                else:
                    chunks.append(audio)
        reply = " ".join(clauses)
        if self.robot is not None and reply:
            self.robot.express(reply)  # emotion move matching the reply (decoupled)
        self.machine.finish_speaking()
        return {"transcript": transcript, "reply": reply,
                "first_audio": first_audio, "audio": chunks}

    # -- voice cloning by voice command --------------------------------------
    def _speak(self, text: str, speaker) -> None:
        """Synthesize ``text`` with the current TTS and play it (used for clone prompts)."""
        for chunk in self.tts.synth_stream(text):
            speaker.play(chunk)

    def _clone_voice(self, transcript: str, speaker, mic) -> dict:
        """Prompt the user, record ~10 s, clone their voice, and swap the TTS to the clone.

        Runs inline on the pipeline thread (MLX is thread-bound). The two prompts before the
        swap are spoken in the CURRENT voice; the confirmation after the swap is the FIRST
        thing spoken in the cloned voice."""
        import tempfile
        import soundfile as sf

        self.machine.begin_speaking()
        if self.robot is not None:
            self.robot.on_state("speaking")
        # 1) ask the user to speak (in the current voice), then wait for it to finish
        self._speak(CLONE_PROMPT, speaker)
        speaker.wait()
        # 2) record the reference straight from the mic (the frames() loop is paused here)
        mic.drain()
        print(f"  [clone] recording {CLONE_SECONDS:.0f}s …", flush=True)
        ref = mic.record(CLONE_SECONDS)
        tmp = f"{tempfile.gettempdir()}/reachy_voice_ref.wav"
        sf.write(tmp, ref, mic.target_sr)
        # 3) acknowledge (still the old voice) while the model loads/clones
        self._speak(CLONE_WORKING, speaker)
        speaker.wait()
        ok = self._load_cloned_tts(tmp)
        # 4) confirm — the first line in the NEW cloned voice (or report failure, old voice)
        self._speak(CLONE_DONE if ok else CLONE_FAILED, speaker)
        speaker.wait()
        self.machine.finish_speaking()
        return {"transcript": transcript, "reply": (CLONE_DONE if ok else CLONE_FAILED),
                "first_audio": None, "audio": [], "cloned": ok}

    def _load_cloned_tts(self, ref_path: str) -> bool:
        """Point the TTS at a clone of ``ref_path``. Reuses the current Chatterbox model if one
        is already loaded; otherwise loads CLONE_REPO and swaps it in (e.g. replacing Kokoro).
        All our TTS engines share a 24 kHz rate, so the open Speaker stays valid. Never raises."""
        try:
            from reachy_chat.tts.chatterbox import ChatterboxTTS
            if isinstance(self.tts, ChatterboxTTS):
                return self.tts.set_voice(ref_path)
            new = ChatterboxTTS(repo=CLONE_REPO, exaggeration=CLONE_EXAGGERATION)
            if not new.set_voice(ref_path):
                return False
            self.tts = new  # Kokoro → cloned Chatterbox (same 24 kHz, Speaker unaffected)
            return True
        except Exception as e:
            print(f"  [clone] failed: {type(e).__name__}: {e}", flush=True)
            return False

    # -- offline driver (testable, measures full latency incl. endpointing) --
    def run_wav(self, path: str, realtime: bool = True) -> dict:
        import soundfile as sf
        try:
            from harness import find_end_of_speech
        except ImportError as e:  # benchmarks/ is only present in a source checkout
            raise RuntimeError(
                "--mode wav needs the benchmarks/ harness, which ships only with the source "
                "checkout (not the installed package). Run from a clone of the repo, or use "
                "`reachy-chat --mode live`."
            ) from e
        a, sr = sf.read(path, dtype="float32")
        if a.ndim > 1:
            a = a.mean(1)
        assert sr == 16000, "prompt must be 16 kHz"
        eos_n = int(find_end_of_speech(a, sr) * sr)
        self.machine.reset()
        start = time.perf_counter()
        t_eos: float | None = None
        for i in range(0, len(a) - FRAME, FRAME):
            frame = a[i:i + FRAME]
            if realtime:
                target = start + i / sr
                dt = target - time.perf_counter()
                if dt > 0:
                    time.sleep(dt)
            if t_eos is None and i + FRAME >= eos_n:
                t_eos = time.perf_counter()
            state = self.machine.process_frame(frame)
            if self._capturing:
                self.stt.add_frame(frame)
            if state is State.THINKING:
                res = self.run_turn(speaker=None)
                res["endpoint_to_first_audio_ms"] = None
                if res["first_audio"] is not None and t_eos is not None:
                    res["full_latency_ms"] = (res["first_audio"] - t_eos) * 1000
                return res
        return {"transcript": "", "reply": "", "first_audio": None, "audio": [],
                "note": "no endpoint fired"}

    # -- live driver (mic + speaker) -----------------------------------------
    def run_live(self) -> None:  # pragma: no cover (needs a microphone)
        from reachy_chat.pipeline.audio_io import MicStream, Speaker
        import os
        dbg = os.environ.get("REACHY_DEBUG_AUDIO")  # dump per-turn 16 kHz STT audio to /tmp
        dbg_buf: list[np.ndarray] = []
        turn_n = 0
        print("Listening… (Ctrl-C to stop)", flush=True)
        prev = self.machine.state
        with MicStream(device=self.audio_device) as mic, \
                Speaker(sr=self.tts.sr, device=self.audio_device) as speaker:
            if self.robot is not None:
                self.robot.start()
            for frame in mic.frames():
                state = self.machine.process_frame(frame)
                if state is not prev:
                    print(f"  [{prev.value} -> {state.value}]", flush=True)
                    if self.robot is not None:
                        self.robot.on_state(state.value)
                    prev = state
                if self._capturing:
                    self.stt.add_frame(frame)
                    if dbg:
                        dbg_buf.append(frame.copy())
                if state is State.THINKING:
                    # Run the turn INLINE on this thread. MLX streams are thread-local, so MLX
                    # inference must run on the same thread the models were loaded on — spawning a
                    # worker thread per turn raises "no Stream(gpu, N) in current thread". (Barge-in
                    # is deferred until STT/LLM/TTS run on a single persistent MLX worker thread.)
                    res = self.run_turn(speaker=speaker, mic=mic)
                    print(f"  you    > {res['transcript']!r}", flush=True)
                    print(f"  reachy < {res['reply']!r}", flush=True)
                    if dbg and dbg_buf:
                        import soundfile as sf
                        p = f"/tmp/reachy_turn_{turn_n}.wav"
                        sf.write(p, np.concatenate(dbg_buf), 16000)
                        print(f"  [debug: STT heard -> {p}]", flush=True)
                    dbg_buf = []
                    turn_n += 1
                    speaker.wait()   # block until Reachy finishes speaking...
                    mic.drain()      # ...THEN flush the mic so it doesn't hear its own voice
                    prev = self.machine.state
                    if self.robot is not None:
                        self.robot.on_state("idle")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["live", "wav"], default="live",
                    help="live = mic+speaker conversation (default); wav = offline file driver "
                         "for benchmarking")
    ap.add_argument("--wav", default=str(ROOT / "audio_samples/prompts/p3_vision.wav"))
    ap.add_argument("--llm", default="mlx-community/gemma-4-E2B-it-qat-4bit",
                    help="Gemma 4 model for BOTH chat and vision (default: E2B, fast; "
                         "pass mlx-community/gemma-4-E4B-it-qat-4bit for a smarter, slower model)")
    ap.add_argument("--tts", default="chatterbox-turbo-8bit", choices=list(TTS_PRESETS),
                    help="speech model (default chatterbox-turbo-8bit): kokoro (lowest latency, "
                         "~150 ms) | chatterbox-4bit (emotive, clonable) | "
                         "chatterbox-turbo-8bit (best-sounding + clonable, ~0.7-0.8 s TTFA)")
    ap.add_argument("--voice", default=None,
                    help="(chatterbox only) reference WAV to clone the voice from; "
                         "omit for the built-in default voice")
    ap.add_argument("--exaggeration", type=float, default=None,
                    help="(chatterbox only) emotion intensity 0..~1.5 (default 0.5)")
    ap.add_argument("--wake", action="store_true", help="wake-word mode ('Hey Reachy')")
    ap.add_argument("--speak", action="store_true", help="(wav mode) play the response aloud")
    ap.add_argument("--robot", action="store_true", help="drive the Reachy daemon/sim (look-at + emotions)")
    ap.add_argument("--robot-url", default="http://localhost:8000")
    ap.add_argument("--vision", action=argparse.BooleanOptionalAction, default=True,
                    help="camera vision ('what do you see?') via Gemma 4, loaded at startup. "
                         "On by default; --no-vision skips it (saves a second ~4-6 GB model — "
                         "use on low-RAM machines)")
    ap.add_argument("--audio-device", default="auto", metavar="NAME|INDEX",
                    help="mic+speaker device: 'auto' (default: Reachy Mini Audio when plugged in, "
                         "else the system default), 'default' (Mac mic/speakers), a device index, "
                         "or a name substring")
    ap.add_argument("--context", default=None, metavar="FILE",
                    help="text file appended to the persona (e.g. context/kids_books.txt, a "
                         "library catalogue for book recommendations); prefilled once at startup")
    args = ap.parse_args()

    app = ConversationApp(llm_repo=args.llm, tts=args.tts,
                          mode=Mode.WAKE_WORD if args.wake else Mode.ALWAYS_ON,
                          robot=args.robot, robot_url=args.robot_url,
                          vision=args.vision,  # vision uses the same --llm model
                          voice=args.voice, exaggeration=args.exaggeration,
                          system_prompt=build_system_prompt(args.context),
                          audio_device=resolve_audio_device(args.audio_device))
    if args.context:
        print(f"Context: {args.context}")
    print("Warming up…")
    app.warmup()
    if args.mode == "live":
        try:
            app.run_live()
        finally:
            if app.robot is not None:
                app.robot.stop()
    else:
        res = app.run_wav(args.wav)
        print(f"\ntranscript: {res['transcript']!r}")
        print(f"reply     : {res['reply']!r}")
        if res.get("full_latency_ms"):
            print(f"\nFULL stop-talking → first-audio (incl. endpoint detection): "
                  f"{res['full_latency_ms']:.0f} ms")
        if args.speak and res.get("audio"):
            import time
            from reachy_chat.pipeline.audio_io import Speaker
            print("playing response…", flush=True)
            with Speaker(sr=app.tts.sr, device=app.audio_device) as sp:
                for ch in res["audio"]:
                    sp.play(ch)
                dur = sum(len(c) for c in res["audio"]) / app.tts.sr
                time.sleep(dur + 0.8)  # let playback drain before closing the stream


if __name__ == "__main__":
    main()
