"""TTFT + decode-speed benchmark for MLX LLMs (mlx-lm 0.31), production-accurate.

The LLM stage budget assumes the **persona/system prefix is cached once** and only the
new user turn is prefilled per request (this is how the live app runs). Measuring with a
fresh cache every call (re-prefilling the persona) massively overstates TTFT.

We therefore report BOTH:
  * **warm TTFT** — persona prefilled once into a persistent cache; per turn we prefill only
    the new user-turn tokens, produce the first token, then trim the cache back. This is the
    number that lands in the voice-to-voice budget (target ~150 ms p50).
  * **cold TTFT** — fresh cache, full persona+user prefilled each call (worst case / no cache).
  * **decode tok/s** — sustained greedy decode after first token.

Usage:
    python benchmarks/bench_llm.py mlx-community/gemma-4-E4B-it-qat-4bit --n 30
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

SYSTEM_PROMPT = (
    "You are Reachy, a friendly desk robot. Keep replies short, warm, and natural — "
    "one or two spoken sentences. You can move your head and antennas to express yourself."
)
USER_TURNS = [
    "Hey Reachy, what's your name?",
    "Can you tell me a quick joke?",
    "What do you see in front of you right now?",
    "What's the weather like today, and should I bring an umbrella?",
    "Good morning! How are you feeling today?",
]


def pctl(xs: list[float], p: float) -> float:
    return float(np.percentile(xs, p)) if xs else float("nan")


def _ids(tokenizer, messages, add_generation_prompt):
    ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=add_generation_prompt, tokenize=True
    )
    return list(ids)


def bench(repo: str, n: int, warmup: int, max_tokens: int) -> None:
    print(f"Loading {repo} ...")
    t = time.perf_counter()
    model, tok = load(repo)
    print(f"  load: {time.perf_counter() - t:.2f}s")

    bos = getattr(tok, "bos_token_id", None)

    # Persistent cache with the persona prefix prefilled once.
    cache = make_prompt_cache(model)
    sys_ids = mx.array(_ids(tok, [{"role": "system", "content": SYSTEM_PROMPT}], False))
    mx.eval(model(sys_ids[None], cache=cache))
    base_off = cache[0].offset

    def user_ids(user: str) -> mx.array:
        ids = _ids(tok, [{"role": "user", "content": user}], True)
        if bos is not None and ids and ids[0] == bos:
            ids = ids[1:]  # persona turn already carried BOS
        return mx.array(ids)

    def warm_once(user: str) -> tuple[float, float, int]:
        uids = user_ids(user)
        t0 = time.perf_counter()
        logits = model(uids[None], cache=cache)
        y = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(y)
        ttft = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        ntok = 1
        for _ in range(max_tokens - 1):
            logits = model(y[None], cache=cache)
            y = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(y)
            ntok += 1
        decode_s = time.perf_counter() - t1
        trim_prompt_cache(cache, cache[0].offset - base_off)  # back to persona-only
        tps = (ntok - 1) / decode_s if decode_s > 0 and ntok > 1 else float("nan")
        return ttft, tps, ntok

    def cold_once(user: str) -> float:
        c = make_prompt_cache(model)
        ids = mx.array(_ids(tok, [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ], True))
        t0 = time.perf_counter()
        logits = model(ids[None], cache=c)
        y = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(y)
        return (time.perf_counter() - t0) * 1000

    print(f"Warmup ({warmup}) ...")
    for i in range(warmup):
        warm_once(USER_TURNS[i % len(USER_TURNS)])
        cold_once(USER_TURNS[i % len(USER_TURNS)])

    print(f"Measuring ({n}) ...")
    warm_ttft: list[float] = []
    cold_ttft: list[float] = []
    tpss: list[float] = []
    for i in range(n):
        u = USER_TURNS[i % len(USER_TURNS)]
        wt, tps, ntok = warm_once(u)
        ct = cold_once(u)
        warm_ttft.append(wt)
        cold_ttft.append(ct)
        if not np.isnan(tps):
            tpss.append(tps)
        if (i + 1) % max(1, n // 10) == 0:
            print(f"  [{i + 1}/{n}] warm_ttft={wt:.0f}ms  cold_ttft={ct:.0f}ms  decode={tps:.0f}tok/s")

    peak_gb = mx.get_peak_memory() / 1e9
    print("\n=== RESULTS ===")
    print(f"model:              {repo}")
    print(f"WARM TTFT p50/p95:  {pctl(warm_ttft, 50):.0f} / {pctl(warm_ttft, 95):.0f} ms   <- production (persona cached)")
    print(f"COLD TTFT p50/p95:  {pctl(cold_ttft, 50):.0f} / {pctl(cold_ttft, 95):.0f} ms   (fresh cache, persona re-prefilled)")
    print(f"decode    p50/p5:   {pctl(tpss, 50):.0f} / {pctl(tpss, 5):.0f} tok/s")
    print(f"peak GPU memory:    {peak_gb:.2f} GB")
    target = 150
    print(f"budget check:       WARM TTFT p50 {'<=' if pctl(warm_ttft,50) <= target else '>'} {target} ms")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=48)
    args = ap.parse_args()
    bench(args.repo, args.n, args.warmup, args.max_tokens)
