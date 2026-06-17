"""Reachy Chat Offline — fully-local low-latency voice-to-voice for Reachy Mini.

See ``plan.md`` for the architecture and ``docs/research/`` for the grounding research.
"""

import sys

__version__ = "0.0.1"

# Supported Python range (kept in sync with pyproject's requires-python). The MLX +
# Reachy Mini stack is verified on 3.12; we accept 3.10–3.12 and fail loudly otherwise so
# a user on the wrong interpreter gets a clear instruction instead of a cryptic import error.
_MIN_PY = (3, 10)
_MAX_PY = (3, 12)  # inclusive


def _check_python() -> None:
    cur = sys.version_info[:2]
    if cur < _MIN_PY or cur > _MAX_PY:
        have = f"{cur[0]}.{cur[1]}"
        msg = (
            f"Reachy Chat needs Python {_MIN_PY[0]}.{_MIN_PY[1]}–{_MAX_PY[0]}.{_MAX_PY[1]}; "
            f"you have {have}. Recreate the environment with the right interpreter:\n"
            f"    uv venv --python 3.12 && uv pip install -e .\n"
            f"(or `uv sync`, which reads the pinned 3.12 from .python-version)."
        )
        raise RuntimeError(msg)


_check_python()
