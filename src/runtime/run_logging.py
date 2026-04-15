from __future__ import annotations

import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping

from src.runtime.app import AppRuntimeSettings


@dataclass(frozen=True)
class RunDescriptor:
    mode: str
    action: str
    run_id: str | None = None
    log_path: Path | None = None


class TeeWriter:
    def __init__(self, *targets):
        self._targets = targets

    def write(self, data: str) -> int:
        for target in self._targets:
            target.write(data)
        return len(data)

    def flush(self) -> None:
        for target in self._targets:
            target.flush()

    def isatty(self) -> bool:
        return any(getattr(target, "isatty", lambda: False)() for target in self._targets)


@contextmanager
def tee_console_to_log(log_path: Path | None):
    if log_path is None:
        yield
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        stdout_tee = TeeWriter(sys.stdout, log_handle)
        stderr_tee = TeeWriter(sys.stderr, log_handle)
        with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee):
            print(f"[log] Writing detailed log to {log_path}")
            yield


def timestamp_token() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_run_log_path(
    *,
    mode: str,
    action: str,
    run_id: str | None,
    write_detailed_log_file: bool,
    run_log_dir: Path,
) -> Path | None:
    if not write_detailed_log_file:
        return None
    if mode == "online" and run_id is not None:
        return Path("artifacts/online_runs") / run_id / "main_run.log"
    return run_log_dir / f"{mode}_{action}_{timestamp_token()}.log"


def render_run_header_lines(
    *,
    descriptor: RunDescriptor,
    settings: AppRuntimeSettings,
    started_at: datetime,
) -> tuple[str, ...]:
    parallel_label, worker_label = _format_parallel_summary(
        settings.motion_settings.local_parallel_workers
    )
    lines = [
        f"[run] start={_format_datetime(started_at)}",
        (
            f"[run] mode={descriptor.mode} action={descriptor.action} "
            f"backend={settings.motion_settings.ik_backend} "
            f"parallel={parallel_label} workers={worker_label} "
            f"min_batch={settings.motion_settings.local_parallel_min_batch_size}"
        ),
        (
            f"[run] robot={settings.robot_name} frame={settings.frame_name} "
            f"program={settings.program_name}"
        ),
        f"[run] csv_in={settings.validation_centerline_csv}",
        f"[run] csv_out={settings.tool_poses_frame2_csv}",
        (
            "[run] target_origin_mm="
            f"{_format_float_tuple(settings.target_frame_origin_mm)}"
        ),
        (
            "[run] target_rotation_xyz_deg="
            f"{_format_float_tuple(settings.target_frame_rotation_xyz_deg)}"
        ),
    ]
    if descriptor.run_id is not None:
        lines.append(f"[run] run_id={descriptor.run_id}")
    if descriptor.log_path is not None:
        lines.append(f"[run] detail_log={descriptor.log_path}")
    return tuple(lines)


def render_run_footer_lines(
    *,
    status: str,
    finished_at: datetime,
    duration_seconds: float,
    error: BaseException | None = None,
) -> tuple[str, ...]:
    lines = [
        (
            f"[run] status={status} end={_format_datetime(finished_at)} "
            f"duration={duration_seconds:.3f}s"
        )
    ]
    if error is not None:
        lines.append(f"[run] error={type(error).__name__}: {error}")
    return tuple(lines)


def print_result_paths(title: str, paths: Mapping[str, Path | None]) -> None:
    print(title)
    for label, path in paths.items():
        if path is None:
            continue
        print(f"  {label}: {path}")


def emit_lines(lines: Iterable[str]) -> None:
    for line in lines:
        print(line)


def _format_parallel_summary(configured_workers: int) -> tuple[str, str]:
    if configured_workers == 1:
        return "off", "1"
    if configured_workers == 0:
        return "on(auto)", "auto"
    return "on", str(configured_workers)


def _format_float_tuple(values: tuple[float, ...]) -> str:
    return "(" + ", ".join(f"{float(value):.1f}" for value in values) + ")"


def _format_datetime(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")
