"""Latency benchmark harness for the Reachy offline voice pipeline.

Methodology (plan.md §8, docs/research/05-architecture-latency.md):

  * **t=0 ≡ end of user speech** in a pre-recorded prompt WAV (16 kHz mono).
  * Prompts are streamed through the *real* frame path (not bulk `transcribe`), paced
    to wall-clock real time, so STT has already consumed most audio by the time the
    user stops — exactly as in live use.
  * A single ``time.perf_counter`` clock records stage markers. The **headline metric**
    is ``first_audio - t0`` (first output sample handed to the device, relative to the
    moment the user stopped speaking).
  * Warm up ``warmup`` runs, measure ``n`` runs; report p50 / p95 / p99 + RTF.

The harness is component-agnostic: a component benchmark implements ``Pipeline`` and
calls ``timeline.mark(...)`` at the canonical stage boundaries.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# Canonical stage markers (a subset may be used depending on the pipeline)
# ---------------------------------------------------------------------------
T0 = "t0_end_of_speech"
ENDPOINT = "endpoint_decision"
STT_FINAL = "stt_final"
LLM_TTFT = "llm_first_token"
LLM_FIRST_CLAUSE = "llm_first_clause"
TTS_TTFA = "tts_first_audio"
FIRST_AUDIO = "first_audio_to_device"

# Default ordering used for stage-delta tables / reports.
DEFAULT_STAGES = [ENDPOINT, STT_FINAL, LLM_TTFT, LLM_FIRST_CLAUSE, TTS_TTFA, FIRST_AUDIO]


# ---------------------------------------------------------------------------
# Timeline: perf_counter markers relative to t0
# ---------------------------------------------------------------------------
@dataclass
class Timeline:
    """Records perf_counter timestamps for named stage markers within one run."""

    _marks: dict[str, float] = field(default_factory=dict)

    def mark(self, name: str, when: float | None = None) -> None:
        """Record ``name`` at ``when`` (perf_counter secs) or now."""
        self._marks[name] = time.perf_counter() if when is None else when

    def has(self, name: str) -> bool:
        return name in self._marks

    def abs(self, name: str) -> float:
        return self._marks[name]

    def delta_ms(self, name: str, base: str = T0) -> float | None:
        """Milliseconds from ``base`` to ``name`` (None if either missing)."""
        if name not in self._marks or base not in self._marks:
            return None
        return (self._marks[name] - self._marks[base]) * 1000.0

    def headline_ms(self) -> float | None:
        """Voice-to-voice latency: first audio to device, relative to t0."""
        return self.delta_ms(FIRST_AUDIO)

    def as_deltas(self, stages: list[str] = DEFAULT_STAGES) -> dict[str, float | None]:
        return {s: self.delta_ms(s) for s in stages}


# ---------------------------------------------------------------------------
# Prompts: load WAV + locate end-of-speech (t0)
# ---------------------------------------------------------------------------
@dataclass
class Prompt:
    name: str
    audio: np.ndarray  # float32 mono, [-1, 1]
    sr: int
    eos_sec: float  # end-of-speech time (t0), seconds from start

    @property
    def duration_sec(self) -> float:
        return len(self.audio) / self.sr


def find_end_of_speech(
    audio: np.ndarray,
    sr: int,
    frame_ms: float = 20.0,
    rel_threshold: float = 0.02,
    hangover_ms: float = 120.0,
) -> float:
    """Estimate end-of-speech (seconds) as the last frame whose RMS exceeds
    ``rel_threshold`` * peak-frame-RMS, plus a small hangover.

    This is a pragmatic energy heuristic for clean prompt WAVs; for noisy audio,
    pass an explicit ``eos_sec`` when constructing the Prompt instead.
    """
    if audio.size == 0:
        return 0.0
    frame = max(1, int(sr * frame_ms / 1000.0))
    n_frames = len(audio) // frame
    if n_frames == 0:
        return len(audio) / sr
    trimmed = audio[: n_frames * frame].reshape(n_frames, frame)
    rms = np.sqrt(np.mean(trimmed.astype(np.float64) ** 2, axis=1) + 1e-12)
    peak = float(rms.max())
    if peak <= 0:
        return len(audio) / sr
    voiced = np.where(rms >= rel_threshold * peak)[0]
    if voiced.size == 0:
        return len(audio) / sr
    last_voiced_frame = int(voiced[-1])
    eos = (last_voiced_frame + 1) * frame / sr + hangover_ms / 1000.0
    return min(eos, len(audio) / sr)


def load_prompt(path: str | Path, target_sr: int = 16000, eos_sec: float | None = None) -> Prompt:
    """Load a WAV as float32 mono at ``target_sr`` and locate t0 (end of speech)."""
    path = Path(path)
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = _resample_linear(audio, sr, target_sr)
        sr = target_sr
    if eos_sec is None:
        eos_sec = find_end_of_speech(audio, sr)
    return Prompt(name=path.stem, audio=audio, sr=sr, eos_sec=eos_sec)


def load_prompt_dir(directory: str | Path, target_sr: int = 16000) -> list[Prompt]:
    directory = Path(directory)
    return [load_prompt(p, target_sr) for p in sorted(directory.glob("*.wav"))]


def _resample_linear(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Dependency-light linear resample (fine for 16 kHz prompt prep)."""
    if sr_in == sr_out:
        return audio
    n_out = int(round(len(audio) * sr_out / sr_in))
    x_in = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    x_out = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_out, x_in, audio).astype(np.float32)


# ---------------------------------------------------------------------------
# Real-time streaming feeder
# ---------------------------------------------------------------------------
def stream_realtime(
    prompt: Prompt,
    timeline: Timeline,
    on_frame: Callable[[np.ndarray, bool], None],
    frame_ms: float = 20.0,
    realtime: bool = True,
) -> None:
    """Feed ``prompt`` frame-by-frame to ``on_frame(frame, is_last_speech)`` paced to
    real time. Marks T0 on the frame that crosses ``eos_sec``.

    ``on_frame`` receives the audio frame and a flag that is True on the frame where
    end-of-speech occurs (the pipeline may use this, but should normally rely on its
    own VAD/endpointer to *decide* the turn end — t0 is only the ground-truth clock).
    """
    frame = max(1, int(prompt.sr * frame_ms / 1000.0))
    eos_sample = int(prompt.eos_sec * prompt.sr)
    start = time.perf_counter()
    t0_marked = False
    for i in range(0, len(prompt.audio), frame):
        chunk = prompt.audio[i : i + frame]
        crosses_eos = (not t0_marked) and (i + frame >= eos_sample)
        if realtime:
            target = start + i / prompt.sr
            sleep = target - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
        if crosses_eos:
            timeline.mark(T0)
            t0_marked = True
        on_frame(chunk, crosses_eos)
    if not t0_marked:  # eos at/after end of file
        timeline.mark(T0)


# ---------------------------------------------------------------------------
# Pipeline protocol
# ---------------------------------------------------------------------------
class Pipeline(Protocol):
    """A component or end-to-end pipeline under test.

    ``run`` must consume ``prompt`` (typically via ``stream_realtime``), call
    ``timeline.mark(...)`` at stage boundaries — crucially ``T0`` and ``FIRST_AUDIO``
    — and return when first output audio has been produced (further synthesis may
    continue in the background but is not part of the headline metric).
    """

    name: str

    def run(self, prompt: Prompt, timeline: Timeline) -> None: ...

    def warmup(self) -> None:  # optional; default no-op via runtime check
        ...


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
@dataclass
class StageStats:
    stage: str
    samples: list[float]

    def pct(self, p: float) -> float:
        return float(np.percentile(self.samples, p)) if self.samples else float("nan")

    @property
    def p50(self) -> float:
        return self.pct(50)

    @property
    def p95(self) -> float:
        return self.pct(95)

    @property
    def p99(self) -> float:
        return self.pct(99)

    @property
    def mean(self) -> float:
        return float(np.mean(self.samples)) if self.samples else float("nan")


@dataclass
class BenchmarkResult:
    pipeline: str
    n: int
    warmup: int
    stages: dict[str, StageStats]
    per_prompt_headline: dict[str, list[float]] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def headline(self) -> StageStats:
        return self.stages[FIRST_AUDIO]

    def to_dict(self) -> dict:
        return {
            "pipeline": self.pipeline,
            "n": self.n,
            "warmup": self.warmup,
            "meta": self.meta,
            "stages": {
                s: {
                    "p50": st.p50,
                    "p95": st.p95,
                    "p99": st.p99,
                    "mean": st.mean,
                    "n": len(st.samples),
                }
                for s, st in self.stages.items()
            },
            "per_prompt_headline_p50": {
                k: float(np.percentile(v, 50)) for k, v in self.per_prompt_headline.items() if v
            },
        }


# ---------------------------------------------------------------------------
# Benchmark driver
# ---------------------------------------------------------------------------
def run_benchmark(
    pipeline: Pipeline,
    prompts: list[Prompt],
    n: int = 30,
    warmup: int = 10,
    stages: list[str] = DEFAULT_STAGES,
    progress: bool = True,
) -> BenchmarkResult:
    """Warm up, then run ``n`` measured iterations cycling through ``prompts``."""
    warm = getattr(pipeline, "warmup", None)
    if callable(warm):
        warm()

    def one(prompt: Prompt) -> Timeline:
        tl = Timeline()
        pipeline.run(prompt, tl)
        return tl

    # Warmup (not measured)
    for i in range(warmup):
        one(prompts[i % len(prompts)])

    collected: dict[str, list[float]] = {s: [] for s in stages}
    per_prompt: dict[str, list[float]] = {p.name: [] for p in prompts}
    for i in range(n):
        prompt = prompts[i % len(prompts)]
        tl = one(prompt)
        for s in stages:
            d = tl.delta_ms(s)
            if d is not None:
                collected[s].append(d)
        h = tl.headline_ms()
        if h is not None:
            per_prompt[prompt.name].append(h)
        if progress and (i + 1) % max(1, n // 10) == 0:
            hl = tl.headline_ms()
            print(f"  [{i + 1}/{n}] {prompt.name}: headline={hl:.0f} ms" if hl else f"  [{i + 1}/{n}]")

    return BenchmarkResult(
        pipeline=pipeline.name,
        n=n,
        warmup=warmup,
        stages={s: StageStats(s, collected[s]) for s in stages},
        per_prompt_headline=per_prompt,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def format_markdown(result: BenchmarkResult, budget_ms: float | None = 500.0) -> str:
    lines = [f"### {result.pipeline}", ""]
    lines.append(f"_n={result.n}, warmup={result.warmup}_")
    if result.meta:
        lines.append("")
        for k, v in result.meta.items():
            lines.append(f"- **{k}**: {v}")
    lines += ["", "| Stage (Δ from t0) | p50 ms | p95 ms | p99 ms | mean ms |", "|---|---:|---:|---:|---:|"]
    for s in result.stages.values():
        if s.samples:
            lines.append(f"| {s.stage} | {s.p50:.0f} | {s.p95:.0f} | {s.p99:.0f} | {s.mean:.0f} |")
    h = result.headline
    if h.samples and budget_ms is not None:
        verdict = "✅ under budget" if h.p50 <= budget_ms else "❌ over budget"
        lines += ["", f"**Voice-to-voice p50 = {h.p50:.0f} ms** (budget {budget_ms:.0f} ms) → {verdict}; "
                  f"p95 = {h.p95:.0f} ms."]
    return "\n".join(lines)


def write_results(result: BenchmarkResult, out_dir: str | Path, budget_ms: float | None = 500.0) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{result.pipeline}.json"
    md_path = out_dir / f"{result.pipeline}.md"
    json_path.write_text(json.dumps(result.to_dict(), indent=2))
    md_path.write_text(format_markdown(result, budget_ms))
    return json_path, md_path


# ---------------------------------------------------------------------------
# Self-test: a fabricated pipeline validates stats + reporting with no models.
# ---------------------------------------------------------------------------
class _DummyPipeline:
    """Fabricates plausible stage latencies to validate the harness end-to-end."""

    name = "dummy-selftest"

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def run(self, prompt: Prompt, timeline: Timeline) -> None:
        # Don't actually stream in the self-test; just synthesize markers off t0.
        t0 = time.perf_counter()
        timeline.mark(T0, t0)
        budget = {ENDPOINT: 40, STT_FINAL: 125, LLM_TTFT: 150, LLM_FIRST_CLAUSE: 190,
                  TTS_TTFA: 300, FIRST_AUDIO: 330}
        for stage, base in budget.items():
            jitter = float(self._rng.normal(0, base * 0.12))
            timeline.mark(stage, t0 + max(1.0, base + jitter) / 1000.0)


def _selftest() -> None:
    here = Path(__file__).resolve().parent.parent
    prompt_dir = here / "audio_samples" / "prompts"
    if prompt_dir.exists() and any(prompt_dir.glob("*.wav")):
        prompts = load_prompt_dir(prompt_dir)
        print(f"Loaded {len(prompts)} prompts:")
        for p in prompts:
            print(f"  {p.name}: dur={p.duration_sec:.2f}s  eos(t0)={p.eos_sec:.2f}s  sr={p.sr}")
    else:
        # Fabricate a prompt if none on disk.
        sr = 16000
        prompts = [Prompt("synthetic", np.zeros(sr, dtype=np.float32), sr, 0.8)]
        print("No prompt WAVs found; using a synthetic silent prompt.")
    print("\nRunning dummy pipeline benchmark (validates stats + report)...")
    result = run_benchmark(_DummyPipeline(), prompts, n=30, warmup=5, progress=False)
    print()
    print(format_markdown(result))


if __name__ == "__main__":
    _selftest()
