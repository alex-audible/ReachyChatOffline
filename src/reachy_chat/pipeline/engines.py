"""Reusable streaming engines for the live conversation pipeline.

Generalized from ``benchmarks/bench_e2e.py`` (which proved <500 ms first-audio, EXP-E2E-1).
All MLX inference here is synchronous; the async app runs each call via ``asyncio.to_thread``
so the event loop never starves (see the mlx-realtime-inference-optimization lessons).

- :class:`STTEngine`  — parakeet streaming; accept 16 kHz frames during speech, ``finalize()`` at
  the endpoint returns the transcript (only the final window is on the critical path).
- :class:`LLMEngine`  — Gemma 4 with the persona prefix cached warm and thinking disabled;
  ``stream_clauses()`` yields speakable text chunks (first chunk capped for low TTFA), cancellable.
- :class:`TTSEngine`  — mlx-audio engine (Kokoro default); ``synth_stream()`` yields audio chunks
  per sentence so playback starts on the first clause.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Iterator

import mlx.core as mx
import numpy as np

from reachy_chat.tts import apply_kokoro_fixes

apply_kokoro_fixes()

DEFAULT_SYSTEM_PROMPT = (
    "You are Reachy, a friendly desk robot having a natural spoken conversation. "
    "Reply warmly and conversationally — usually two to four sentences, and go longer when the "
    "question genuinely calls for more detail, but stay focused and don't ramble. "
    "Remember what was said earlier in the conversation and stay on topic. "
    "Speak plainly: never use asterisks, emojis, markdown, bullet points, headings, stage "
    "directions, or action descriptions like *(tilts head)* — express everything in spoken words."
)
_CLAUSE_RE = re.compile(r"[,.!?;:]")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


# ---------------------------------------------------------------------------
# STT
# ---------------------------------------------------------------------------
class STTEngine:
    """Parakeet STT. Buffers 16 kHz mono float32 frames during the user's turn, then
    BATCH-transcribes the whole utterance at the endpoint.

    Why batch, not streaming: parakeet-mlx's ``transcribe_stream`` exposes only the text within
    its rolling context window, so utterances longer than that window lose their beginning
    (verified: a 9 s utterance streamed to "They and can you say a time as a fifteen minutes?"
    while batch gave the full sentence). Batch ``transcribe`` handles any length via internal
    overlapped chunking and is accurate. Bonus: ``add_frame`` is now a cheap buffer append with
    NO MLX on the caller's thread, so it never blocks the endpoint frame-pump (fixes the live
    STT/endpoint contention). Cost: STT runs at the endpoint (~165–290 ms for typical turns)
    instead of overlapping speech — a worthwhile trade for correct transcripts."""

    def __init__(self, repo: str = "mlx-community/parakeet-tdt-0.6b-v2", sr: int = 16000):
        from parakeet_mlx import from_pretrained
        self.model = from_pretrained(repo)
        self.sr = sr
        self._buf: list[np.ndarray] = []
        self._tmp = f"/tmp/_reachy_stt_{id(self)}.wav"

    def start(self) -> None:
        self._buf = []

    def add_frame(self, frame: np.ndarray) -> None:
        """Cheap buffer append (no inference) — safe to call from the frame-pump thread."""
        self._buf.append(np.asarray(frame, dtype=np.float32))

    def finalize(self) -> str:
        """Batch-transcribe the buffered utterance and return the transcript."""
        if not self._buf:
            return ""
        import soundfile as sf
        audio = np.concatenate(self._buf)
        self._buf = []
        sf.write(self._tmp, audio, self.sr)
        return self.model.transcribe(self._tmp).text.strip()

    def warmup(self) -> None:
        self.start()
        self.add_frame(np.zeros(self.sr // 2, np.float32))
        self.finalize()


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
class LLMEngine:
    """Gemma 4 chat model with CONVERSATION MEMORY and natural sentence-level streaming.

    - Persona prefilled once; each turn EXTENDS the same KV cache incrementally (no trim) so the
      model remembers the conversation. History is capped to bound prefill/cache growth.
    - ``stream_clauses`` yields speakable chunks at NATURAL boundaries — full sentences (.!?), or a
      comma/clause break only when a sentence runs long — so TTS gets natural prosody, not mid-word
      fragments. Thinking is disabled for low latency.
    """

    def __init__(self, repo: str = "mlx-community/gemma-4-E2B-it-qat-4bit",
                 system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                 min_clause_chars: int = 8, long_clause_chars: int = 48,
                 max_reply_tokens: int = 400, max_history_turns: int = 12):
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache
        self.model, self.tok = load(repo)
        self.system_prompt = system_prompt
        self.min_clause_chars = min_clause_chars
        self.long_clause_chars = long_clause_chars
        self.max_reply_tokens = max_reply_tokens
        self.max_history_turns = max_history_turns
        self.stop_ids = set(getattr(self.tok, "eos_token_ids", None) or {self.tok.eos_token_id})
        self._eot = self.tok.eos_token_id
        self._make_cache = make_prompt_cache
        self._sys = [{"role": "system", "content": system_prompt}]
        self.messages: list[dict] = []  # conversation history (excludes system)
        self._rebuild()

    def _ids(self, messages, add_gen):
        return list(self.tok.apply_chat_template(
            messages, add_generation_prompt=add_gen, tokenize=True, enable_thinking=False))

    def _rebuild(self) -> None:
        """(Re)create the KV cache holding persona + current self.messages (no generation prompt)."""
        self.cache = self._make_cache(self.model)
        ids = self._ids(self._sys + self.messages, add_gen=False)
        mx.eval(self.model(mx.array(ids)[None], cache=self.cache))

    def reset(self) -> None:
        """Forget the conversation (keep the persona)."""
        self.messages = []
        self._rebuild()

    def _cut(self, new: str):
        """Index of a natural speech boundary in `new` (sentence end, or clause break if long)."""
        m = re.search(r"[.!?]", new)
        if m is None and len(new) >= self.long_clause_chars:
            m = re.search(r"[,;:]", new)
        if m is not None and m.end() >= self.min_clause_chars:
            return m.end()
        return None

    def stream_clauses(self, transcript: str, cancel: threading.Event | None = None) -> Iterator[str]:
        # Bound history so prefill/cache stay small; rebuild the cache for the trimmed window.
        if len(self.messages) > 2 * self.max_history_turns:
            self.messages = self.messages[-2 * self.max_history_turns:]
            self._rebuild()
        # Append the new user turn as INCREMENTAL tokens onto the existing cache (template diff:
        # `full` minus `base` share the same history prefix, so the slice is exactly the user turn).
        base = self._ids(self._sys + self.messages, add_gen=False)
        self.messages.append({"role": "user", "content": transcript})
        full = self._ids(self._sys + self.messages, add_gen=True)
        user_block = full[len(base):]
        logits = self.model(mx.array(user_block)[None], cache=self.cache)
        y = mx.argmax(logits[:, -1, :], axis=-1); mx.eval(y)
        tokid = int(y.item())
        out: list[int] = []
        emitted = 0
        n = 0
        while n < self.max_reply_tokens and tokid not in self.stop_ids:
            if cancel is not None and cancel.is_set():
                break
            out.append(tokid); n += 1
            text = self.tok.decode(out, skip_special_tokens=True)
            cut = self._cut(text[emitted:])
            if cut is not None:
                seg = text[emitted:emitted + cut].strip()
                if seg:
                    yield seg
                    emitted += cut
            logits = self.model(y[None], cache=self.cache)
            y = mx.argmax(logits[:, -1, :], axis=-1); mx.eval(y)
            tokid = int(y.item())
        reply = self.tok.decode(out, skip_special_tokens=True).strip()
        tail = reply[emitted:].strip()
        if tail:
            yield tail
        # Close the assistant turn in the cache so the next user turn appends cleanly; keep memory.
        mx.eval(self.model(mx.array([self._eot])[None], cache=self.cache))
        self.messages.append({"role": "assistant", "content": reply})


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------
class TTSEngine:
    """mlx-audio TTS. ``synth_stream`` yields float32 audio chunks (per sentence) so the
    caller can start playback on the first clause. Kokoro is the low-latency default."""

    def __init__(self, repo: str = "mlx-community/Kokoro-82M-bf16",
                 generate_kwargs: dict | None = None, sr: int = 24000):
        from mlx_audio.tts.utils import get_model_path, load_model
        p = get_model_path(repo)
        self.model = load_model(p[0] if isinstance(p, tuple) else p)
        self.kwargs = generate_kwargs or {"voice": "af_heart", "lang_code": "a"}
        self.sr = getattr(self.model, "sample_rate", sr)

    def synth_stream(self, text: str, cancel: threading.Event | None = None) -> Iterator[np.ndarray]:
        for sent in (s for s in _SENT_RE.split(text.strip()) if s.strip()):
            if cancel is not None and cancel.is_set():
                return
            for seg in self.model.generate(text=sent, **self.kwargs):
                if cancel is not None and cancel.is_set():
                    return
                au = getattr(seg, "audio", seg)
                mx.eval(au)
                yield np.array(au).reshape(-1).astype(np.float32)

    def synth(self, text: str) -> np.ndarray:
        chunks = list(self.synth_stream(text))
        return np.concatenate(chunks) if chunks else np.zeros(1, np.float32)
