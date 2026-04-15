from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter


@dataclass
class _ProfilerFrame:
    name: str
    start_time: float


_PROFILE_TOTALS: dict[str, float] = {}
_PROFILE_COUNTS: dict[str, int] = {}
_PROFILE_STACK: list[_ProfilerFrame] = []


def reset_runtime_profile() -> None:
    _PROFILE_TOTALS.clear()
    _PROFILE_COUNTS.clear()
    _PROFILE_STACK.clear()


@contextmanager
def profile_runtime_section(name: str):
    now = perf_counter()
    if _PROFILE_STACK:
        parent = _PROFILE_STACK[-1]
        _PROFILE_TOTALS[parent.name] = _PROFILE_TOTALS.get(parent.name, 0.0) + (
            now - parent.start_time
        )
    _PROFILE_STACK.append(_ProfilerFrame(name=name, start_time=now))
    try:
        yield
    finally:
        end_time = perf_counter()
        frame = _PROFILE_STACK.pop()
        _PROFILE_TOTALS[frame.name] = _PROFILE_TOTALS.get(frame.name, 0.0) + (
            end_time - frame.start_time
        )
        _PROFILE_COUNTS[frame.name] = _PROFILE_COUNTS.get(frame.name, 0) + 1
        if _PROFILE_STACK:
            _PROFILE_STACK[-1].start_time = end_time


def runtime_profile_snapshot() -> dict[str, dict[str, float | int]]:
    return {
        name: {
            "seconds": round(seconds, 6),
            "count": int(_PROFILE_COUNTS.get(name, 0)),
        }
        for name, seconds in sorted(
            _PROFILE_TOTALS.items(),
            key=lambda item: (-item[1], item[0]),
        )
    }


def format_runtime_profile(snapshot: dict[str, dict[str, float | int]]) -> str:
    if not snapshot:
        return "No runtime profile captured."

    lines = ["Runtime profile:"]
    for name, payload in snapshot.items():
        lines.append(
            f"  {name}: {float(payload['seconds']):.3f}s across {int(payload['count'])} call(s)"
        )
    return "\n".join(lines)
