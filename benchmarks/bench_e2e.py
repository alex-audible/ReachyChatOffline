"""END-TO-END voice-to-voice FIRST-AUDIO benchmark — the metric that actually matters.

Measures wall-clock time from **t0 (end of user speech)** to the **first audio sample** of the
response, through the real streaming cascade WITH overlap — not the sum of component latencies.

Pipeline (warm / in-conversation):
  * STT (parakeet streaming) runs DURING speech; at t0 only the final window is processed.
  * LLM (Gemma 4, persona prefix cached) generates from the transcript; we start TTS on the
    FIRST speakable clause (first punctuation boundary or ~min chars), not a full sentence.
  * TTS (Kokoro) synthesizes that first clause → first audio.

t0 is taken AFTER feeding all-but-the-final STT window (those were consumed live while the user
talked); the timed critical path is: final-STT-window → LLM TTFT+first-clause → TTS first audio.

Usage:
  python benchmarks/bench_e2e.py                 # default E2B + Kokoro
  python benchmarks/bench_e2e.py --llm mlx-community/gemma-4-E4B-it-qat-4bit --n 20
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from harness import find_end_of_speech  # noqa: E402
from reachy_chat.tts import apply_kokoro_fixes  # noqa: E402

apply_kokoro_fixes()  # patch mlx-audio Kokoro istftnet length-mismatch crash

from mlx_audio.tts.utils import get_model_path, load_model as load_tts  # noqa: E402
from mlx_lm import load as load_llm  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache  # noqa: E402
from parakeet_mlx import from_pretrained  # noqa: E402

SYSTEM_PROMPT = (
    "You are Reachy, a friendly desk robot. Reply in one or two short, warm, natural spoken "
    "sentences. Speak plainly: never use asterisks, emojis, markdown, stage directions, or "
    "action descriptions like *(tilts head)* — express everything through spoken words only."
)
STT_REPO = "mlx-community/parakeet-tdt-0.6b-v2"
TTS_REPO = "mlx-community/Kokoro-82M-bf16"
CHUNK_MS = 480.0
CTX = (256, 64)
CLAUSE_RE = re.compile(r"[,.!?;:]")
MIN_CLAUSE_CHARS = 5  # first speakable chunk can be short ("Hello!", "Oh,") to minimise TTFA
FIRST_CHUNK_MAXCHARS = 24  # cap first chunk to a short prosodic unit (~5 words) to bound TTFA;
# later chunks stream behind it. Lower = lower latency, slightly choppier onset.


def pctl(xs, p):
    return float(np.percentile(xs, p)) if xs else float("nan")


def spread(xs: list[float]) -> str:
    a = np.array(xs)
    return (f"min {a.min():.0f} | p50 {np.percentile(a,50):.0f} | p90 {np.percentile(a,90):.0f} | "
            f"p95 {np.percentile(a,95):.0f} | p99 {np.percentile(a,99):.0f} | max {a.max():.0f} | "
            f"std {a.std():.0f}")


class Pipeline:
    def __init__(self, llm_repo: str, max_clause_tokens: int = 32,
                 stt_chunk_ms: float = CHUNK_MS, first_chunk_chars: int = FIRST_CHUNK_MAXCHARS):
        self.stt_chunk_ms = stt_chunk_ms
        self.first_chunk_chars = first_chunk_chars
        print(f"Loading STT {STT_REPO} ...")
        self.stt = from_pretrained(STT_REPO)
        print(f"Loading LLM {llm_repo} ...")
        self.model, self.tok = load_llm(llm_repo)
        self.bos = getattr(self.tok, "bos_token_id", None)
        print(f"Loading TTS {TTS_REPO} ...")
        p = get_model_path(TTS_REPO)
        self.tts = load_tts(p[0] if isinstance(p, tuple) else p)
        self.max_clause_tokens = max_clause_tokens
        # Warm persona cache (prefill once)
        self.cache = make_prompt_cache(self.model)
        sys_ids = mx.array(self.tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}], add_generation_prompt=False,
            tokenize=True, enable_thinking=False))  # thinking OFF: low-latency direct replies
        mx.eval(self.model(sys_ids[None], cache=self.cache))
        self.base_off = self.cache[0].offset

    # --- stages ---------------------------------------------------------
    def stt_stream(self, audio, sr, eos_n):
        """Feed all-but-final window live (untimed); return a closure that, when called,
        processes the final window and returns the transcript (this call is the timed part)."""
        step = int(sr * self.stt_chunk_ms / 1000)
        n_full = max(1, eos_n // step)
        ctx = self.stt.transcribe_stream(context_size=CTX)
        tx = ctx.__enter__()
        for k in range(n_full - 1):
            tx.add_audio(mx.array(audio[k * step:(k + 1) * step]))
            _ = tx.result.text
        last = n_full - 1

        def finalize():
            tx.add_audio(mx.array(audio[last * step:(last + 1) * step]))
            text = tx.result.text
            ctx.__exit__(None, None, None)
            return text
        return finalize

    def llm_first_clause(self, transcript: str):
        ids = self.tok.apply_chat_template(
            [{"role": "user", "content": transcript}], add_generation_prompt=True,
            tokenize=True, enable_thinking=False)
        if self.bos is not None and ids and ids[0] == self.bos:
            ids = ids[1:]
        stop_ids = set(getattr(self.tok, "eos_token_ids", None) or {self.tok.eos_token_id})
        uids = mx.array(ids)
        logits = self.model(uids[None], cache=self.cache)
        y = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(y)
        out: list[int] = []
        tokid = int(y.item())
        clause: str | None = None
        while len(out) < self.max_clause_tokens and tokid not in stop_ids:
            out.append(tokid)
            text = self.tok.decode(out, skip_special_tokens=True)
            m = CLAUSE_RE.search(text)
            if m and m.end() >= MIN_CLAUSE_CHARS:
                clause = text[: m.end()].strip()  # clean chunk up to the boundary, no mid-word cut
                break
            if len(text) >= self.first_chunk_chars:  # cap: cut at last word boundary to bound TTFA
                sp = text.rfind(" ")
                if sp >= MIN_CLAUSE_CHARS:
                    clause = text[:sp].strip()
                    break
            logits = self.model(y[None], cache=self.cache)
            y = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(y)
            tokid = int(y.item())
        if clause is None:
            clause = self.tok.decode(out, skip_special_tokens=True).strip()
        trim_prompt_cache(self.cache, self.cache[0].offset - self.base_off)
        return clause or "Okay!"

    def tts_first_audio(self, clause: str):
        for seg in self.tts.generate(text=clause, voice="af_heart", lang_code="a"):
            au = getattr(seg, "audio", seg)
            mx.eval(au)
            return np.array(au).reshape(-1)
        return np.zeros(1, np.float32)

    # --- one full turn --------------------------------------------------
    def run_turn(self, audio, sr, eos_n):
        finalize = self.stt_stream(audio, sr, eos_n)  # live feed (during speech)
        t0 = time.perf_counter()                       # === END OF USER SPEECH ===
        transcript = finalize()
        t_stt = time.perf_counter()
        clause = self.llm_first_clause(transcript)
        t_llm = time.perf_counter()
        _ = self.tts_first_audio(clause)
        t_audio = time.perf_counter()
        return {
            "first_audio_ms": (t_audio - t0) * 1000,
            "stt_ms": (t_stt - t0) * 1000,
            "llm_ms": (t_llm - t_stt) * 1000,
            "tts_ms": (t_audio - t_llm) * 1000,
            "transcript": transcript,
            "clause": clause,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="mlx-community/gemma-4-E2B-it-qat-4bit")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--stt-chunk-ms", type=float, default=CHUNK_MS)
    ap.add_argument("--first-chunk-chars", type=int, default=FIRST_CHUNK_MAXCHARS)
    args = ap.parse_args()

    pipe = Pipeline(args.llm, stt_chunk_ms=args.stt_chunk_ms, first_chunk_chars=args.first_chunk_chars)
    prompts = []
    for p in sorted(Path("audio_samples/prompts").glob("*.wav")):
        a, sr = sf.read(str(p), dtype="float32")
        if a.ndim > 1:
            a = a.mean(1)
        eos_n = int(find_end_of_speech(a, sr) * sr)
        prompts.append((p.stem, a, sr, eos_n))

    print("\nWarmup ...")
    for i in range(args.warmup):
        name, a, sr, eos_n = prompts[i % len(prompts)]
        pipe.run_turn(a, sr, eos_n)
    mx.eval()

    print(f"Measuring ({args.n}) ...  LLM={args.llm}")
    fa, stt, llm, tts = [], [], [], []
    sample = None
    for i in range(args.n):
        name, a, sr, eos_n = prompts[i % len(prompts)]
        r = pipe.run_turn(a, sr, eos_n)
        fa.append(r["first_audio_ms"]); stt.append(r["stt_ms"]); llm.append(r["llm_ms"]); tts.append(r["tts_ms"])
        if i < len(prompts):
            print(f"  {name:14s} first_audio={r['first_audio_ms']:.0f}ms "
                  f"(stt {r['stt_ms']:.0f} | llm {r['llm_ms']:.0f} | tts {r['tts_ms']:.0f})  "
                  f"resp={r['clause']!r}")

    peak = mx.get_peak_memory() / 1e9
    print("\n=== END-TO-END VOICE-TO-VOICE (first audio, from t0=end of speech) ===")
    print(f"LLM={args.llm}  stt_chunk={args.stt_chunk_ms:.0f}ms  first_chunk={args.first_chunk_chars}chars  n={args.n}")
    print(f"FIRST-AUDIO:  {spread(fa)}")
    print(f"  STT stage:  {spread(stt)}")
    print(f"  LLM stage:  {spread(llm)}")
    print(f"  TTS stage:  {spread(tts)}")
    print(f"peak GPU memory: {peak:.2f} GB")
    verdict = "✅ UNDER" if pctl(fa, 50) <= 500 else "❌ OVER"
    print(f"budget: p50={pctl(fa,50):.0f}ms {verdict} 500ms  |  p95={pctl(fa,95):.0f}ms")


if __name__ == "__main__":
    main()
