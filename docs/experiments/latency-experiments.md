# Latency Experiments Log

> Living record of **every** latency measurement, with special attention to **budget
> overshoots** and how they were resolved. Each entry: hypothesis → method → result →
> verdict vs budget → resolution. Hardware: **Apple M3 Max, 96 GB, macOS 15.7.5**, MLX on
> Metal GPU. Target deployment: M3 / 24 GB (working set ≤ ~20 GB). Budget: **≤500 ms**
> voice-to-voice (≤600 ms with vision).

### Measurement convention (canonical) — IN-CONVERSATION, not cold-start

The 500 ms budget is for **turns during an ongoing conversation**, so every headline number
is measured **warm / primed** and **cold-start is explicitly excluded from budget verdicts**:

1. **Models are loaded and warmed** before measurement (≥6 warmup iterations; weights resident,
   Metal kernels/graphs compiled).
2. **Persona/system (+ any RAG) prefix is prefilled once and kept cached**; each measured turn
   prefills only the new user turn — exactly the live in-conversation path.
3. Reported as **p50 / p95** over N≥20 warm runs.
4. **Cold-start** costs (process launch, model load, first-ever forward, first persona prefill)
   are recorded **for information only** and never counted against the 500 ms budget.

**Design implication — startup priming:** the app primes at launch (load all models + prefill the
persona cache + run one dummy forward/STT/TTS pass) so the *first real user turn* is already warm.
A "warming up…" state can cover the one-time cost before the conversation starts.

For LLMs the WARM number above = persona cached, only the new user turn prefilled. The **COLD**
column in tables (fresh cache, persona re-prefilled) is diagnostic context, **not** a budget figure.

---

## Running component budget (measured vs target)

| Stage | Target p50 | Measured p50 | Source | Status |
|---|---:|---:|---|---|
| Turn-detection (Silero v6 + Smart-Turn v3) ‡ | ~165–215 | **~160** | EXP-TURN-1 | ✅ under |
| STT finalize (streaming, ctx 256/64) | 125 | **~116** | EXP-STT-2 | ✅ under |
| LLM TTFT (warm, E4B) | 150 | **53** | EXP-LLM-2 | ✅ under |
| LLM TTFT (warm, E2B) | 150 | **29** | EXP-LLM-2 | ✅ under |
| TTS TTFA (Kokoro, latency-first) | 110 | **83** | EXP-TTS-1 | ✅ under |
| TTS TTFA (emotive engines) | 110 | _benchmarking_ | EXP-TTS-2 | ⏳ |
| Output buffering | 30 | _pending impl_ | — | ⏳ |

¹ EXP-STT-1 measured *full-file batch* transcribe, not streaming finalize. Streaming finalize
(incremental, audio already consumed by end-of-speech) is expected far lower; to be measured.
‡ Turn-detection is a SEPARATE stage that fires AT t0 (end of speech); the e2e first-audio metric
below is measured FROM t0, so turn-detection is not double-counted in it.

> **HEADLINE (EXP-E2E-1):** measured end-to-end voice-to-voice **first-audio p50 = 271 ms (E2B) /
> 331 ms (E4B)**, p95 = 387 / 497 ms — both **under the 500 ms budget**. The overlapped wall-clock
> is far below the additive component sum (~432 ms), which is the whole point of streaming.

---

## EXP-LLM-1 — Gemma 4 E4B cold TTFT **OVERSHOOT** (and root cause)

- **Date:** 2026-06-17
- **Hypothesis:** Gemma 4 E4B QAT-4bit hits the ~150 ms TTFT budget on M3.
- **Method (initial, flawed):** `bench_llm.py` v1 — `stream_generate` with a **fresh
  `make_prompt_cache` every call**, re-prefilling the full persona + user prompt each request.
  N=20, max_tokens=64.
- **Result:** **TTFT p50 = 275 ms, p95 = 319 ms** ❌ (decode 49 tok/s, 6.0 GB peak).
  **OVERSHOOT: 275 ms vs 150 ms target (+83%).**
- **Diagnosis (prefill-bound vs fixed overhead):** measured TTFT vs prefill length, fresh cache:

  | Prefill | tokens | TTFT |
  |---|---:|---:|
  | minimal (`stream_generate`, fresh cache) | 17 | 234 ms |
  | sys+user (fresh cache) | 58 | 266 ms |
  | sys+RAG (fresh cache) | 418 | 492 ms |
  | **user-turn, persona already cached (direct forward)** | 22 | **52 ms** |

  → The cost was **re-prefilling the persona prompt every call** + `stream_generate` per-call
  setup overhead, **not** intrinsic first-token latency. With the persona prefix cached, a real
  user turn prefills in ~52 ms.
- **Resolution:** rewrote `bench_llm.py` to the **production path** — persona prefilled once into
  a persistent cache; per turn prefill only the new user-turn tokens, then `trim_prompt_cache`
  back. App must keep a warm persistent persona/RAG cache. See EXP-LLM-2.
- **Verdict:** overshoot was a **measurement artifact**; resolved by correct caching.

## EXP-LLM-2 — Gemma 4 E4B / E2B production TTFT (corrected)

- **Date:** 2026-06-17
- **Method:** `bench_llm.py` v2. Persona (~50 tok) prefilled once; per turn prefill only the new
  user turn + first token (WARM); also fresh-cache persona+user (COLD) for contrast. Greedy
  decode for tok/s. N=25, warmup=6. 5 rotating conversational user turns.
- **Results:**

  | Model | WARM TTFT p50/p95 | COLD TTFT p50/p95 | decode p50 | peak GPU |
  |---|---:|---:|---:|---:|
  | `gemma-4-E4B-it-qat-4bit` | **53 / 53 ms** | 92 / 129 ms | 45 tok/s | 6.07 GB |
  | `gemma-4-E2B-it-qat-4bit` | **29 / 30 ms** | 48 / 66 ms | 77 tok/s | 3.59 GB |

- **Verdict:** ✅ both **well under** the 150 ms LLM TTFT budget on the warm path. Even COLD
  (no cache) is under budget. E4B chosen for quality (also the vision model); E2B is the
  ultra-low-latency / headroom option. Decode 45–77 tok/s easily streams clauses to TTS faster
  than real-time speech (~3 words/s ≈ ~4 tok/s needed).
- **Decode speed:** manual per-token loop gave 45/76 tok/s; mlx-lm native `generation_tps` gives
  **E4B 50 / E2B 91 tok/s** (the per-token `mx.eval` sync was only ~10% pessimistic). Root cause
  of "only 50 tok/s" investigated in EXP-LLM-3. Either way, 50 tok/s ≫ the ~4 tok/s needed to
  stream clauses to TTS faster than real-time speech, so decode is not a bottleneck.

## EXP-LLM-3 — Why Gemma 4 E4B decodes at ~50 tok/s (not bandwidth-limited by hardware)

- **Date:** 2026-06-17 · **Trigger:** 50 tok/s looked slow for an M3 Max.
- **Method:** inspected `mx.device_info()` and the model's quantization config + vocab.
- **Findings:**
  - Hardware is healthy: **M3 Max, 77 GB GPU working set**, 2048² matmul 35 ms, **prefill 415 tok/s**.
  - The `*-qat-4bit` build is **mixed precision**: per the `quantization` config, **every
    transformer layer's MLP `gate_proj`/`up_proj`/`down_proj` is 8-bit**, only attention is 4-bit.
    The FFN dominates per-token weight traffic, so decode runs near **8-bit memory bandwidth**, not
    4-bit → ~50 tok/s and ~6 GB resident (a pure-4-bit build would be ~3 GB and faster).
  - **262,144-token vocab** → the final logits projection is a large matmul every decode step
    (a well-known Gemma decode tax).
- **Verdict:** ~50 tok/s is expected for *this QAT recipe* (a quality-preserving choice), not a
  hardware or measurement fault. E2B (91 tok/s) is the faster option. **Follow-up (EXP-LLM-4):**
  test a pure-4-bit / `mxfp4` E4B variant to quantify the speed↔quality trade — expected faster
  decode at some quality cost.

## EXP-STT-1 — Parakeet-TDT-0.6B (MLX) full-file transcribe

- **Date:** 2026-06-17
- **Method:** `parakeet-mlx` `from_pretrained("mlx-community/parakeet-tdt-0.6b-v2")`, full-file
  `transcribe()` on the 6 prompt WAVs (16 kHz mono), 1 warmup.
- **Results:** latency 165–288 ms for 1.5–5.5 s clips; transcripts essentially perfect (only the
  coined name "Reachy"→"Riachi"). Model load 145 s (included first-time download).
- **Verdict:** 🟡 batch transcribe latency is fine, but this is **not** the streaming-finalize
  number. Next: measure incremental streaming finalize (audio consumed during speech, only the
  tail finalized at end-of-speech) — expected well under 125 ms. Also evaluate Nemotron streaming
  0.6B MLX (research-stt primary) for true partials.

---

## EXP-TTS-1 — Kokoro-82M (MLX) time-to-first-audio + sample audio

- **Date:** 2026-06-17 · **Engine:** `mlx-community/Kokoro-82M-bf16` via `mlx-audio`, voice `af_heart`.
- **Method:** `bench_tts.py`, **sentence-chunked** synth (production path). TTFA = synth of first
  short clause ("Oh wow, that's wonderful!"); RTF on a longer line; warm (3 warmups), N=8.
- **Results:** **TTFA p50 = 83 ms, p95 = 85 ms; RTF = 0.04 (≈25× real-time); peak 1.40 GB; 24 kHz.**
  Samples: `audio_samples/tts_outputs/kokoro_{neutral,emotive}.wav`.
- **Verdict:** ✅ well under the 110 ms TTFA budget and tiny memory. **Caveat:** Kokoro is fast but
  **flat/neutral** in affect (research + audible in samples) — likely insufficient for the "very
  natural & emotional" requirement on its own. It is the latency-first floor; emotive engines next.
- **Gotcha (root cause found later — see EXP-TTS-fix):** the `broadcast_shapes` crash is upstream
  bug #786 (a `math.ceil` float64 rounding error in `interpolate.py`), deterministic for certain
  clause lengths — not "multi-sentence" as first guessed. Fixed via `src/reachy_chat/tts/kokoro_fix.py`.

## EXP-STT-2 — Parakeet streaming finalize latency (+ accuracy/latency sweep)

- **Date:** 2026-06-17 · `bench_stt_stream.py`, 480 ms windows, `context_size=(left, right)`.
- **Finding:** finalize cost = processing the final streaming window ≈ **97–116 ms**. Right-context
  is the accuracy↔latency knob — a sweep on a clear sentence: **right_ctx ≥ 64 → perfect transcript**
  (last window 116 ms); right_ctx ≤ 32 → empty/garbage (decoder starved). Per-window RTF ≈ 0.20.
- **Chosen config:** `context_size=(256, 64)`, 320 ms chunk in the e2e pipeline.
- **Verdict:** ✅ STT finalize ~116–130 ms in-pipeline, under the 125 ms budget.

## EXP-TURN-1 — Endpoint decision latency (Silero v6 + Smart-Turn v3)

- **Date:** 2026-06-17 · `benchmarks/bench_endpoint.py`. Real-time 32 ms feed; latency from
  harness ground-truth end-of-speech to endpoint-fired.
- **Results (CPU/ONNX → GPU-insulated):** overall **p50 ≈ 160 ms, p95 ≈ 210 ms**; Smart-Turn v3
  compute p50 ≈ 12–15 ms; VAD-only baseline p50 ≈ 145 ms. 30/30 fired via smart_turn, 0 misses.
- **Verdict:** ✅ within the ~165–215 ms turn-detection budget. This stage PRECEDES t0 in live use;
  the e2e first-audio metric (below) starts AT t0 (end of speech).

## EXP-LLM-5 — Gemma 4 "thinking" mode must be DISABLED (latency-critical)

- Gemma 4's chat template injects `<|think|>` by default → the model emits a long
  `<|channel|>thought… Thinking Process:` block before the spoken answer — catastrophic for
  first-audio latency (TTS would also speak the reasoning).
- **Fix:** `apply_chat_template(..., enable_thinking=False)` (verified: strips the think block).
  Plus the persona prompt forbids stage-directions/asterisks (the model otherwise emits
  `*(tilts head)*`, which TTS would speak and which delays the first clause).

## EXP-TTS-fix — Kokoro istftnet crash: root cause + fix (supersedes EXP-TTS-1 gotcha)

- The `broadcast_shapes` crash is upstream mlx-audio bug **#786**: v0.4.4 swapped
  `mx.ceil`→`math.ceil` in `interpolate.py`; float64 rounds `988.0000…1`→989, so `989×300=296700`
  vs `296400` — an off-by-one-frame (Δ = upsample_scale = 300) mismatch between `sine_waves` and
  `uv`. **Deterministic for certain clause lengths** (not "multi-sentence" as first guessed).
- **Decision:** patch, not downgrade (#784) — downgrade risks the emotive engines + vision on 0.4.4.
  `src/reachy_chat/tts/kokoro_fix.py` monkeypatches `interpolate` (epsilon-round before ceil = the
  #786 fix) + a defensive SineGen length-align. Kokoro now robust on arbitrary LLM text.

## EXP-E2E-1 — END-TO-END voice-to-voice first-audio (THE headline metric) ✅

- **Date:** 2026-06-17 · `benchmarks/bench_e2e.py`. Wall-clock from **t0 = end of user speech** to
  **first audio sample**, through the real streaming cascade (STT streams during speech → only the
  final window at t0; LLM warm persona cache, thinking off; TTS on the first capped clause).
  Warm/in-conversation, N=30. Config: STT chunk 320 ms / ctx (256,64), first-chunk cap 16 chars,
  Kokoro TTS. Clean machine (after removing a hung background download).
- **Results (full spread, ms):**

  | Pipeline | min | **p50** | p90 | p95 | p99 | max | std |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | **E2B + Kokoro** | 240 | **271** | 377 | 387 | 407 | 415 | 49 |
  | **E4B + Kokoro** | 313 | **331** | 478 | 497 | 524 | 534 | 62 |

  Stage p50 (E2B / E4B): STT 125/128 · LLM TTFT+clause 75/140 · TTS 65/67. Peak GPU ~6 / ~8.4 GB.
- **Proves the overlap point:** measured first-audio (271 ms E2B) is far below the additive
  component sum (~432 ms) — streaming overlap, not addition, is what counts.
- **Variance:** std ~50–62 ms; the LLM first-clause stage is the dominant variance source
  (std 32–42, depends on tokens-to-boundary). Environmental jitter from the Reachy sim daemon
  (~15% CPU, also present in production) + multi-user load; a hung download earlier inflated E4B
  p50 to 549 (removed). Min column (E2B 240 / E4B 313) ≈ uncontended capability.
- **Optimization findings:** (1) smaller STT streaming chunk 480→320 ms cut STT finalize ~217→~128
  p50; (2) first-chunk cap (start TTS on ~first 5 words / first punctuation) cut LLM+TTS on the
  critical path and tightened p95; (3) thinking-off + no-asterisk persona removed huge outliers.
- **Verdict:** ✅ **Both configs meet ≤500 ms p50; E2B meets it at p95 (387) too.** E2B = latency-
  first default; E4B = quality/vision option (≤600 ms vision budget; p95 497).

## EXP-TTS-2 — Emotive engines (latency screening)

- **Date:** 2026-06-17 · `bench_tts.py` via mlx-audio.

  | Engine | repo | TTFA p50 | RTF | Verdict |
  |---|---|---:|---:|---|
  | Kokoro-82M | `mlx-community/Kokoro-82M-bf16` | **79–83 ms** | 0.04 | ✅ fast, but flat affect |
  | Orpheus-3B | `mlx-community/orpheus-3b-0.1-ft-bf16` | **8128 ms** | **2.23** | ❌ unusable (slower than real-time) |
  | Sesame CSM-1B | `mlx-community/csm-1b{,-fp16}` | 2780 | **1.18–1.72** | ❌ too slow to stream even ungated → [FUTURE_WORK](../FUTURE_WORK.md) (quantize to int4/8) |
  | Chatterbox-Turbo q4 | `mlx-community/chatterbox-turbo-mlx-q4` | (0.38 RTF) | — | ❌ **broken**: S3Gen vocoder layout mismatch (~41% weights) → near-silence; see FUTURE_WORK |

- **Finding:** emotive TTS is the open gap on M3 — both "quality-max" picks are **slower than
  real-time** in mlx-audio (Orpheus RTF 2.23, Sesame CSM RTF 1.18), so they can't stream. Only
  **Kokoro** clears RTF<1 but is flat. Emotive hopes: Chatterbox-Turbo (integrating) and a
  **quantized CSM-1B** (deferred to [FUTURE_WORK.md](../FUTURE_WORK.md)). Sample WAVs in
  `audio_samples/tts_outputs/` for subjective judging.

## EXP-E2E-2 — FULL pipeline incl. endpoint detection ("stop-talking → first-audio")

- **Date:** 2026-06-17 · `ConversationApp.run_wav` (real-time paced) through the **InteractionMachine**
  (Silero VAD + Smart-Turn endpoint) + STT + LLM + TTS. t0 = harness end-of-speech. E2B + Kokoro, n=15.
- **Result:** stop-talking → first-audio **min 475 | p50 590 | p90 703 | p95 712 | max 717 | std 84 ms**.
- **Interpretation:** the **compute pipeline** (from the *detected* endpoint → first audio) is
  **271 ms** (EXP-E2E-1) ✅; **endpoint detection adds ~300 ms** — the silence window needed to be
  sure the user finished + Smart-Turn commit. This is the dominant remaining latency and an inherent
  turn-taking tradeoff (shorter window = faster but cuts people off).
- **Eagerness re-measure (n=15):** with the eagerness-gated endpointer (commit at ~100 ms if
  P(complete)≥0.85; max_silence 700 ms): min 474 · **p50 607** · p95 646 · max 651 · std 63 ms.
  The timeout tail shrank (max 717→651, std 84→63) but **p50 did not improve** — because:
  - **(measured, corrects earlier guess) Smart-Turn fires EAGER (P=0.93–0.98) at ~224 ms trailing
    silence** even on the synthetic prompts — the "synthetic under-triggers" worry was WRONG.
    ~200 ms silence is an **inherent floor**: Smart-Turn must SEE silence to judge completion
    (partly desirable — human turn-gaps cluster ~200 ms).
  - **Decomposition (per-prompt, real-time):** endpoint ~156–330 ms (p50 ~244) **+** app-compute
    ~300–558 ms. The app-compute stage is **higher/noisier than the isolated bench (271 ms)** —
    extra STT silence-window buffering during LISTENING (the ~224 ms of post-speech silence frames
    get streamed) + LLM first-clause length variance. **This app-compute gap is the real tuning
    target**, not the endpoint.
  - **(lever) speculative LLM start is not wired yet** — the endpointer exposes `on_speculative`
    but the pipeline doesn't act on it. Wiring it (start LLM on the stabilized STT partial, cancel
    on resume) overlaps the ~224 ms endpoint wait → ~150–250 ms win toward the ~400 ms floor.
- **Status:** endpoint optimization in progress (eagerness done; speculative-start + human-speech
  validation remaining). Compute pipeline already meets budget (271 ms).
- **Caveat found+fixed:** the endpointer's VAD hangover is deliberately large (=max_silence) so the
  VAD stays in-speech while the endpointer tracks silence itself; shrinking it breaks firing (n=0).

## EXP-STT-3 — Streaming parakeet loses long utterances → switched to batch-on-finalize

- **Date:** 2026-06-17 · Found via a live mic transcription test (user read a known sentence).
- **Symptom:** live STT mis-transcribed badly. Same recorded audio: **batch** parakeet =
  near-perfect ("Hello Richie, my name is Alex. What's the weather like in Sydney today and can
  you set a timer for 15 minutes?"); **streaming STTEngine** = garbage ("They and can you say a
  time as a fifteen minutes?").
- **Root cause:** parakeet-mlx `transcribe_stream`'s `.result.text` only exposes text within its
  rolling **context window**. Short prompts fit (so EXP-STT-2 looked fine); a 9 s two-sentence
  utterance scrolled its beginning out → only the tail survived. NOT resampling (per-block vs
  stateful both transcribe perfectly), NOT levels (peak 0.21/rms 0.015 fine), NOT the model.
- **Fix:** `STTEngine` now **buffers the utterance and batch-transcribes at the endpoint**
  (`model.transcribe` handles any length via internal overlapped chunking). Verified on the live
  recording → near-perfect. **Finalize latency: ~153 ms (2 s turn) / 208 ms (9 s).** Bonus:
  `add_frame` is a cheap buffer append (no MLX) → no longer contends with the endpoint frame pump
  (also resolves the live STT/endpoint contention behind issue (a)).
- **Trade:** STT now runs at the endpoint (not overlapped during speech), +~30–80 ms vs the
  (broken) streaming finalize — worth it for correct transcripts; revisit a proper streaming
  decoder (Nemotron) later for overlap.

## Open / pending experiments

- **EXP-TTS-2b:** Chatterbox-Turbo q4 TTFA/RTF + emotive samples (download finishing).
- **EXP-VISION-1:** Gemma 4 E4B image-frame prefill TTFT, serial/clean (vision ≤200 ms budget).
- **EXP-E2E-3:** endpoint-optimized full pipeline (target full p50 → ≤500 ms) + barge-in live.
