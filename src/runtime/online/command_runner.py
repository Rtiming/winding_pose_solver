from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter, sleep


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class CommandLogOptions:
    log_path: Path | None = None
    show_command_details: bool = True
    show_worker_output: bool = True


def _append_log(log_path: Path | None, message: str) -> None:
    if log_path is None:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(message)
        if not message.endswith("\n"):
            handle.write("\n")


def _console_safe_text(text: str) -> str:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def _safe_print(text: str) -> None:
    print(_console_safe_text(text))


def _render_command(args: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in args)


def _run_logged_command(
    args: list[str],
    *,
    cwd: Path | None = None,
    display_prefix: str,
    log_options: CommandLogOptions,
    echo_output: bool,
) -> subprocess.CompletedProcess[str]:
    rendered_command = _render_command(args)
    if log_options.show_command_details:
        print(f"{display_prefix} {rendered_command}")
    _append_log(log_options.log_path, f"{display_prefix} {rendered_command}")

    result = subprocess.run(
        args,
        cwd=str(cwd or REPO_ROOT),
        text=True,
        encoding="utf-8",
        errors="backslashreplace",
        capture_output=True,
        check=False,
    )
    stdout_text = result.stdout or ""
    stderr_text = result.stderr or ""
    if stdout_text:
        _append_log(log_options.log_path, "STDOUT:\n" + stdout_text)
    if stderr_text:
        _append_log(log_options.log_path, "STDERR:\n" + stderr_text)

    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            args,
            output=result.stdout,
            stderr=result.stderr,
        )

    if echo_output:
        if stdout_text.strip():
            _safe_print(stdout_text.strip())
        if stderr_text.strip():
            _safe_print(stderr_text.strip())
    return result


def _run_local_command(
    args: list[str],
    *,
    cwd: Path | None = None,
    log_options: CommandLogOptions,
    echo_output: bool,
) -> subprocess.CompletedProcess[str]:
    return _run_logged_command(
        args,
        cwd=cwd,
        display_prefix="LOCAL >",
        log_options=log_options,
        echo_output=echo_output,
    )


def _run_local_command_with_retries(
    args: list[str],
    *,
    cwd: Path | None = None,
    log_options: CommandLogOptions,
    echo_output: bool,
    attempts: int | None = None,
    delay_seconds: float | None = None,
) -> subprocess.CompletedProcess[str]:
    resolved_attempts = (
        max(1, int(attempts))
        if attempts is not None
        else _positive_int_from_env("WPS_LOCAL_CMD_RETRY_ATTEMPTS", 3)
    )
    resolved_delay_seconds = (
        max(0.0, float(delay_seconds))
        if delay_seconds is not None
        else _non_negative_float_from_env("WPS_LOCAL_CMD_RETRY_DELAY_SECONDS", 1.0)
    )
    last_error: subprocess.CalledProcessError | None = None
    for attempt_index in range(1, resolved_attempts + 1):
        try:
            return _run_local_command(
                args,
                cwd=cwd,
                log_options=log_options,
                echo_output=echo_output,
            )
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if attempt_index >= resolved_attempts:
                break
            message = (
                f"[retry] local command failed on attempt {attempt_index}/{resolved_attempts}; "
                f"retrying in {resolved_delay_seconds:.1f}s."
            )
            print(message)
            _append_log(log_options.log_path, message)
            sleep(resolved_delay_seconds)
    assert last_error is not None
    raise last_error


def _run_remote_bash(
    host: str,
    script: str,
    *,
    log_options: CommandLogOptions,
    echo_output: bool,
) -> subprocess.CompletedProcess[str]:
    command = ["ssh", host, f"bash -lc {shlex.quote(script)}"]
    _append_log(log_options.log_path, f"REMOTE SCRIPT [{host}]:\n{script}")
    return _run_logged_command(
        command,
        display_prefix=f"REMOTE[{host}] >",
        log_options=log_options,
        echo_output=echo_output,
    )


def _run_remote_bash_with_retries(
    host: str,
    script: str,
    *,
    log_options: CommandLogOptions,
    echo_output: bool,
    attempts: int | None = None,
    delay_seconds: float | None = None,
) -> subprocess.CompletedProcess[str]:
    resolved_attempts = (
        max(1, int(attempts))
        if attempts is not None
        else _positive_int_from_env("WPS_REMOTE_CMD_RETRY_ATTEMPTS", 3)
    )
    resolved_delay_seconds = (
        max(0.0, float(delay_seconds))
        if delay_seconds is not None
        else _non_negative_float_from_env("WPS_REMOTE_CMD_RETRY_DELAY_SECONDS", 1.0)
    )
    last_error: subprocess.CalledProcessError | None = None
    for attempt_index in range(1, resolved_attempts + 1):
        try:
            return _run_remote_bash(
                host,
                script,
                log_options=log_options,
                echo_output=echo_output,
            )
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if attempt_index >= resolved_attempts:
                break
            message = (
                f"[retry] remote command failed on attempt {attempt_index}/{resolved_attempts}; "
                f"retrying in {resolved_delay_seconds:.1f}s."
            )
            print(message)
            _append_log(log_options.log_path, message)
            sleep(resolved_delay_seconds)
    assert last_error is not None
    raise last_error


def _summarize_subprocess_failure(exc: subprocess.CalledProcessError) -> str:
    stderr_text = (exc.stderr or "").strip()
    stdout_text = (exc.output or "").strip()
    for text in (stderr_text, stdout_text):
        if text:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if lines:
                return lines[-1]
    return f"command failed with exit code {exc.returncode}"


def _subprocess_text(exc: subprocess.CalledProcessError) -> str:
    return "\n".join(text for text in (exc.stderr, exc.output) if text)


def _positive_int_from_env(name: str, default: int) -> int:
    fallback = max(1, int(default))
    raw_value = os.getenv(name)
    if raw_value is None:
        return fallback
    try:
        parsed_value = int(str(raw_value).strip())
    except ValueError:
        return fallback
    return max(1, parsed_value)


def _non_negative_float_from_env(name: str, default: float) -> float:
    fallback = max(0.0, float(default))
    raw_value = os.getenv(name)
    if raw_value is None:
        return fallback
    try:
        parsed_value = float(str(raw_value).strip())
    except ValueError:
        return fallback
    return max(0.0, parsed_value)


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return bool(default)
    normalized = str(raw_value).strip().lower()
    if normalized in {"", "0", "false", "no", "off"}:
        return False
    if normalized in {"1", "true", "yes", "on"}:
        return True
    return bool(default)


def _looks_like_missing_remote_artifact(exc: subprocess.CalledProcessError) -> bool:
    text = _subprocess_text(exc).lower()
    missing_markers = (
        "no such file",
        "not a regular file",
        "cannot stat",
        "could not stat",
        "does not exist",
    )
    return any(marker in text for marker in missing_markers)


def _run_stage(
    stage_name: str,
    func,
    *,
    log_options: CommandLogOptions,
):
    print(stage_name)
    _append_log(log_options.log_path, stage_name)
    started = perf_counter()
    try:
        result = func()
    except subprocess.CalledProcessError as exc:
        elapsed_seconds = perf_counter() - started
        summary = _summarize_subprocess_failure(exc)
        _append_log(log_options.log_path, f"{stage_name} failed after {elapsed_seconds:.2f}s")
        raise RuntimeError(f"{stage_name} failed after {elapsed_seconds:.2f}s: {summary}") from exc
    elapsed_seconds = perf_counter() - started
    timing_message = f"{stage_name} done in {elapsed_seconds:.2f}s"
    print(timing_message)
    _append_log(log_options.log_path, timing_message)
    return result


def _can_import_module(python_executable: str, module_name: str) -> bool:
    try:
        subprocess.run(
            [python_executable, "-c", f"import {module_name}"],
            check=True,
            text=True,
            encoding="utf-8",
            errors="backslashreplace",
            capture_output=True,
        )
        return True
    except Exception:
        return False
