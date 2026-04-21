from __future__ import annotations

import argparse
import hashlib
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

from src.core.collab_models import (
    EvaluationBatchRequest,
    EvaluationBatchResult,
    ProfileEvaluationRequest,
    RemoteSearchRequest,
    load_json_file,
    write_json_file,
)
from src.runtime.delivery import (
    load_handoff_package,
    result_is_strictly_valid,
    result_quality_summary,
    write_handoff_package,
)
from src.runtime.remote_search import propose_candidates, summarize_results
from src.runtime.request_builder import build_profile_evaluation_request, build_remote_search_request


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SERVER_DIR = "/home/tzwang/program/winding_pose_solver"
DEFAULT_ENV_NAME = "winding_pose_solver"
SYNC_GUARD_FILES: tuple[str, ...] = (
    "app_settings.py",
    "main.py",
    "src/core/motion_settings.py",
    "src/core/types.py",
    "src/runtime/delivery.py",
    "src/runtime/main_entrypoint.py",
    "src/runtime/origin_search_runner.py",
    "src/runtime/origin_sweep.py",
    "scripts/sweep_target_origin_yz.py",
    "src/robodk_runtime/eval_worker.py",
    "src/search/global_search.py",
    "src/search/ik_collection.py",
    "src/search/path_optimizer.py",
    "src/search/local_repair.py",
    "src/search/parallel_profile_eval.py",
    "src/runtime/online/roundtrip.py",
    "online_roundtrip.py",
)
SYNC_MODE_OFF = "off"
SYNC_MODE_GUARD = "guard"
SYNC_MODE_PUSH = "push"
SYNC_BUNDLE_TREE_PATHS: tuple[str, ...] = (
    "src",
    "scripts",
)
SYNC_BUNDLE_ROOT_FILES: tuple[str, ...] = (
    "online_roundtrip.py",
    "online_requester.py",
    "online_worker.py",
    "main.py",
    "app_settings.py",
    "requirements.shared.txt",
    "requirements.server.txt",
    "environment.server.yml",
    "README.md",
    "AGENTS.md",
)
SYNC_BUNDLE_HASH_RELATIVE_PATH = "artifacts/tmp/runtime_bundle.sha256"
DEFAULT_ONLINE_RETRY_CANDIDATE_LIMIT = 4
DEFAULT_ONLINE_RETRY_REPAIR_LIMIT = 2
DEFAULT_ONLINE_RETRY_MAX_ROUNDS = 1


@dataclass(frozen=True)
class CommandLogOptions:
    log_path: Path | None = None
    show_command_details: bool = True
    show_worker_output: bool = True


@dataclass(frozen=True)
class RoundtripArtifacts:
    run_id: str
    local_run_dir: Path
    request_path: Path
    candidates_path: Path
    results_path: Path
    summary_path: Path
    profile_retry_summary_path: Path | None
    final_request_path: Path | None
    final_result_path: Path | None
    handoff_package_path: Path | None
    debug_final_request_path: Path | None
    debug_handoff_package_path: Path | None
    optimized_pose_csv_path: Path | None
    final_program_name: str | None
    best_request_id: str | None
    result_status: str
    log_path: Path | None


@dataclass(frozen=True)
class ServerRoleArtifacts:
    run_id: str
    run_dir: Path
    request_path: Path
    candidates_path: Path
    results_path: Path
    summary_path: Path
    profile_retry_summary_path: Path | None
    final_request_path: Path | None
    handoff_package_path: Path | None
    debug_final_request_path: Path | None
    debug_handoff_package_path: Path | None
    best_request_id: str | None
    result_status: str
    log_path: Path | None


@dataclass(frozen=True)
class ReceiverRoleArtifacts:
    run_id: str
    run_dir: Path
    handoff_package_path: Path
    receiver_request_path: Path
    result_path: Path
    optimized_pose_csv_path: Path | None
    final_program_name: str | None
    result_status: str
    log_path: Path | None


@dataclass(frozen=True)
class WorkerEvalArtifacts:
    request_path: Path
    result_path: Path
    log_path: Path | None


def _default_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _local_artifact_dir(run_id: str) -> Path:
    return REPO_ROOT / "artifacts" / "online_runs" / run_id


def _remote_dir_for_shell(server_dir: str) -> str:
    if server_dir.startswith("~/"):
        return server_dir.replace("~/", "$HOME/", 1)
    if server_dir == "~":
        return "$HOME"
    return server_dir


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _sync_bundle_local_paths() -> tuple[Path, ...]:
    return tuple(
        (REPO_ROOT / relative_path)
        for relative_path in (
            *SYNC_BUNDLE_TREE_PATHS,
            *SYNC_BUNDLE_ROOT_FILES,
        )
    )


def _iter_sync_bundle_local_files() -> tuple[Path, ...]:
    collected_files: list[Path] = []
    for path in _sync_bundle_local_paths():
        if not path.exists():
            raise FileNotFoundError(f"Sync bundle path does not exist: {path}")
        if path.is_file():
            collected_files.append(path)
            continue
        if path.is_dir():
            collected_files.extend(
                child_path
                for child_path in sorted(path.rglob("*"))
                if child_path.is_file()
            )
            continue
        raise RuntimeError(f"Unsupported sync bundle path type: {path}")
    return tuple(collected_files)


def _compute_sync_bundle_sha256() -> str:
    digest = hashlib.sha256()
    seen_relative_paths: set[str] = set()
    for local_path in _iter_sync_bundle_local_files():
        relative_path = local_path.relative_to(REPO_ROOT).as_posix()
        if relative_path in seen_relative_paths:
            continue
        seen_relative_paths.add(relative_path)
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        with local_path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _read_remote_sync_bundle_sha256(
    *,
    host: str,
    server_dir: str,
    log_options: CommandLogOptions,
) -> str | None:
    remote_shell_dir = _remote_dir_for_shell(server_dir)
    remote_script = f"""
set -e
cd "{remote_shell_dir}"
if [ -f "{SYNC_BUNDLE_HASH_RELATIVE_PATH}" ]; then
    cat "{SYNC_BUNDLE_HASH_RELATIVE_PATH}"
fi
"""
    result = _run_remote_bash_with_retries(
        host,
        remote_script,
        log_options=log_options,
        echo_output=False,
    )
    digest = (result.stdout or "").strip().lower()
    if len(digest) == 64 and all(char in "0123456789abcdef" for char in digest):
        return digest
    return None


def _write_remote_sync_bundle_sha256(
    *,
    host: str,
    server_dir: str,
    digest: str,
    log_options: CommandLogOptions,
) -> None:
    remote_shell_dir = _remote_dir_for_shell(server_dir)
    quoted_digest = shlex.quote(str(digest).strip())
    remote_script = f"""
set -e
cd "{remote_shell_dir}"
mkdir -p "$(dirname "{SYNC_BUNDLE_HASH_RELATIVE_PATH}")"
printf '%s\\n' {quoted_digest} > "{SYNC_BUNDLE_HASH_RELATIVE_PATH}"
"""
    _run_remote_bash_with_retries(
        host,
        remote_script,
        log_options=log_options,
        echo_output=False,
    )


def _remote_file_sha256(
    *,
    host: str,
    remote_path: str,
    log_options: CommandLogOptions,
) -> str | None:
    remote_script = f"""
set -e
if [ -f "{remote_path}" ]; then
    sha256sum "{remote_path}" | awk '{{print $1}}'
fi
"""
    result = _run_remote_bash_with_retries(
        host,
        remote_script,
        log_options=log_options,
        echo_output=False,
    )
    digest = (result.stdout or "").strip()
    return digest or None


def _remote_files_sha256(
    *,
    host: str,
    server_dir: str,
    relative_paths: tuple[str, ...],
    log_options: CommandLogOptions,
) -> dict[str, str | None]:
    """Return SHA-256 for several remote files through one SSH connection.

    The Windows coordinator can see intermittent SSH resets on master.  Running
    one short remote shell for all key files keeps the sync guard strict while
    avoiding a burst of separate SSH sessions before every online run.
    """
    remote_shell_dir = _remote_dir_for_shell(server_dir)
    script_lines = ["set -e", f'cd "{remote_shell_dir}"']
    for relative_path in relative_paths:
        quoted_path = shlex.quote(relative_path)
        script_lines.append(
            "if [ -f {path} ]; then "
            "printf '%s\\t' {path}; "
            "sha256sum {path} | awk '{{print $1}}'; "
            "else "
            "printf '%s\\tMISSING\\n' {path}; "
            "fi".format(path=quoted_path)
        )
    result = _run_remote_bash_with_retries(
        host,
        "\n".join(script_lines),
        log_options=log_options,
        echo_output=False,
    )
    hashes: dict[str, str | None] = {}
    for raw_line in (result.stdout or "").splitlines():
        line = raw_line.strip()
        if not line or "\t" not in line:
            continue
        relative_path, digest = line.split("\t", 1)
        hashes[relative_path] = None if digest == "MISSING" else digest
    return hashes


def _enforce_remote_sync_guard(
    *,
    host: str,
    server_dir: str,
    log_options: CommandLogOptions,
) -> None:
    mismatches: list[dict[str, str]] = []
    missing_local: list[str] = []
    missing_remote: list[str] = []
    remote_hashes = _remote_files_sha256(
        host=host,
        server_dir=server_dir,
        relative_paths=SYNC_GUARD_FILES,
        log_options=log_options,
    )
    for relative_path in SYNC_GUARD_FILES:
        local_path = REPO_ROOT / relative_path
        if not local_path.is_file():
            missing_local.append(relative_path)
            continue
        local_hash = _sha256_file(local_path)
        remote_hash = remote_hashes.get(relative_path)
        if remote_hash is None:
            missing_remote.append(relative_path)
            continue
        if local_hash != remote_hash:
            mismatches.append(
                {
                    "path": relative_path,
                    "local_sha256": local_hash,
                    "remote_sha256": remote_hash,
                }
            )

    if not mismatches and not missing_local and not missing_remote:
        print("[online/coordinator] Remote sync guard passed.")
        return

    message_lines = [
        "Remote sync guard blocked online run: local/remote key files are inconsistent.",
        f"Remote root: {server_dir}",
    ]
    if missing_local:
        message_lines.append("Missing local files: " + ", ".join(sorted(missing_local)))
    if missing_remote:
        message_lines.append("Missing remote files: " + ", ".join(sorted(missing_remote)))
    if mismatches:
        message_lines.append("Hash mismatches:")
        for item in mismatches:
            message_lines.append(
                f"  - {item['path']}: local={item['local_sha256'][:12]} remote={item['remote_sha256'][:12]}"
            )
    message_lines.append(
        f"Please sync to {DEFAULT_SERVER_DIR} before rerunning online mode."
    )
    raise RuntimeError("\n".join(message_lines))


def _normalize_sync_mode(raw_value: str) -> str:
    normalized = str(raw_value).strip().lower()
    aliases = {
        "check": SYNC_MODE_GUARD,
        "verify": SYNC_MODE_GUARD,
        "sync": SYNC_MODE_PUSH,
        "none": SYNC_MODE_OFF,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {SYNC_MODE_OFF, SYNC_MODE_GUARD, SYNC_MODE_PUSH}:
        raise ValueError(
            f"Unsupported remote sync mode={raw_value!r}; expected off/guard/push."
        )
    return normalized


def _sync_source_tree_to_server(
    *,
    host: str,
    server_dir: str,
    log_options: CommandLogOptions,
) -> None:
    """Upload runtime source files to server without touching conda environment."""
    remote_shell_dir = _remote_dir_for_shell(server_dir)
    _run_stage(
        "[online/sync] Ensuring remote project directory exists...",
        lambda: _run_remote_bash_with_retries(
            host,
            f'mkdir -p "{remote_shell_dir}"',
            log_options=log_options,
            echo_output=False,
        ),
        log_options=log_options,
    )
    local_bundle_hash = _compute_sync_bundle_sha256()
    remote_bundle_hash = _read_remote_sync_bundle_sha256(
        host=host,
        server_dir=server_dir,
        log_options=log_options,
    )
    force_bundle_sync = _env_flag("WPS_FORCE_BUNDLE_SYNC", default=False)
    if not force_bundle_sync and remote_bundle_hash == local_bundle_hash:
        message = (
            "[online/sync] Runtime bundle unchanged; upload skipped "
            f"(hash={local_bundle_hash[:12]})."
        )
        print(message)
        _append_log(log_options.log_path, message)
        return
    if force_bundle_sync:
        force_message = "[online/sync] WPS_FORCE_BUNDLE_SYNC=1; forcing runtime bundle upload."
        print(force_message)
        _append_log(log_options.log_path, force_message)

    upload_sources = [
        *(str(REPO_ROOT / relative_path) for relative_path in SYNC_BUNDLE_TREE_PATHS),
        *(str(REPO_ROOT / relative_path) for relative_path in SYNC_BUNDLE_ROOT_FILES),
    ]
    _run_stage(
        "[online/sync] Uploading runtime bundle ...",
        lambda: _run_local_command_with_retries(
            [
                "scp",
                "-r",
                *upload_sources,
                f"{host}:{server_dir}/",
            ],
            log_options=log_options,
            echo_output=False,
            attempts=4,
            delay_seconds=1.5,
        ),
        log_options=log_options,
    )
    _run_stage(
        "[online/sync] Recording runtime bundle hash ...",
        lambda: _write_remote_sync_bundle_sha256(
            host=host,
            server_dir=server_dir,
            digest=local_bundle_hash,
            log_options=log_options,
        ),
        log_options=log_options,
    )


def _run_coordinator_sync_preflight(
    *,
    host: str,
    server_dir: str,
    enforce_remote_sync_guard: bool,
    remote_sync_mode: str,
    log_options: CommandLogOptions,
) -> None:
    sync_mode = _normalize_sync_mode(remote_sync_mode)
    if sync_mode == SYNC_MODE_PUSH:
        _sync_source_tree_to_server(
            host=host,
            server_dir=server_dir,
            log_options=log_options,
        )
    if sync_mode == SYNC_MODE_OFF:
        if enforce_remote_sync_guard:
            print(
                "[online/coordinator] Remote sync mode=off; SHA guard is bypassed "
                "for this run."
            )
        else:
            print("[online/coordinator] Remote sync mode=off.")
        return
    if enforce_remote_sync_guard:
        _run_stage(
            "[online/coordinator] Checking local/remote sync guard...",
            lambda: _enforce_remote_sync_guard(
                host=host,
                server_dir=server_dir,
                log_options=log_options,
            ),
            log_options=log_options,
        )
    else:
        print("[online/coordinator] Remote sync guard disabled by configuration.")


def _remote_server_env_export_block() -> str:
    override_names = (
        "WPS_SERVER_PARTITION",
        "WPS_SERVER_CPUS",
        "WPS_SERVER_MEM_MB",
        "WPS_OFFLINE_BATCH_WORKERS",
        "WPS_SERVER_PROFILE_WORKERS",
        "WPS_SERVER_PROFILE_WORKERS_MAX",
        "WPS_SERVER_PROFILE_MIN_BATCH_SIZE",
        "WPS_SERVER_PROFILE_MIN_BATCH_SIZE_CAP",
        "WPS_SERVER_PROFILE_ALLOW_HIGH_MIN_BATCH",
        "WPS_LOCAL_RETRY_REPAIR_EVAL_MODE",
        "WPS_RUNTIME_PROFILE_LEVEL",
        "WPS_ALLOW_LOGIN_NODE_SERVER",
    )
    export_lines: list[str] = []
    for name in override_names:
        value = os.getenv(name)
        if value is None:
            continue
        stripped = str(value).strip()
        if not stripped:
            continue
        export_lines.append(f"export {name}={shlex.quote(stripped)}")
    return "\n".join(export_lines)


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


def resolve_local_worker_python(explicit_path: str | None = None) -> str:
    if explicit_path:
        return explicit_path

    candidates = [sys.executable]
    default_conda_worker = Path.home() / "anaconda3" / "envs" / "winding_pose_solver" / "python.exe"
    if default_conda_worker.is_file():
        candidates.append(str(default_conda_worker))

    for candidate in candidates:
        if _can_import_module(candidate, "robodk"):
            return candidate
    raise RuntimeError(
        "Could not find a local Python interpreter with the 'robodk' package. "
        "Use --local-python to point to your worker environment."
    )


def build_request_file(
    settings,
    request_path: str | Path,
    *,
    round_index: int,
    candidate_limit: int,
    refresh_csv: bool,
    metadata: dict[str, object] | None = None,
) -> Path:
    request_metadata = {"entrypoint": "online_roundtrip", **dict(metadata or {})}
    base_request = build_profile_evaluation_request(
        settings,
        request_id=f"round{round_index}_base",
        strategy="full_search",
        refresh_csv=refresh_csv,
        include_pose_rows_in_result=False,
        create_program=False,
        program_name=settings.program_name,
        optimized_csv_path=str(settings.tool_poses_frame2_csv),
        metadata=request_metadata,
    )
    remote_request = build_remote_search_request(
        base_request,
        round_index=round_index,
        candidate_limit=candidate_limit,
        metadata=request_metadata,
    )
    return write_json_file(request_path, remote_request.to_dict())


def _load_remote_request(path: str | Path) -> RemoteSearchRequest:
    payload = load_json_file(path)
    if "base_request" in payload:
        return RemoteSearchRequest.from_dict(payload)
    return build_remote_search_request(ProfileEvaluationRequest.from_dict(payload))


def _find_result_by_id(
    batch_result: EvaluationBatchResult,
    request_id: str,
):
    for result in batch_result.results:
        if result.request_id == request_id:
            return result
    raise KeyError(f"Request id not found in evaluation results: {request_id}")


def _result_sort_key(result) -> tuple[float, ...]:
    return (
        float(result.invalid_row_count),
        float(result.ik_empty_row_count),
        float(result.bridge_like_segments),
        float(getattr(result, "big_circle_step_count", 0)),
        float(result.worst_joint_step_deg),
        float(result.mean_joint_step_deg),
        float(result.config_switches),
        float(result.total_cost),
        float(result.timing_seconds),
    )


def _select_best_result(results: tuple) -> object | None:
    if not results:
        return None
    return sorted(results, key=_result_sort_key)[0]


def _build_receiver_request_from_result(
    base_request: ProfileEvaluationRequest,
    selected_result,
    *,
    request_id: str,
    program_name: str | None,
    optimized_csv_path: str | None,
    force_debug_program_generation: bool = False,
) -> ProfileEvaluationRequest:
    metadata = dict(base_request.metadata)
    metadata["entrypoint"] = "online_receiver_handoff"
    metadata["receiver_materialization_mode"] = "direct_handoff_import"
    metadata["source_request_id"] = str(selected_result.request_id)
    selected_pose_rows = getattr(selected_result, "pose_rows", None)
    reference_pose_rows = (
        tuple(dict(row) for row in selected_pose_rows)
        if selected_pose_rows is not None
        else tuple(dict(row) for row in base_request.reference_pose_rows)
    )
    metadata["receiver_reference_pose_rows_source"] = (
        "selected_result_pose_rows"
        if selected_pose_rows is not None
        else "base_request_reference_pose_rows"
    )
    if force_debug_program_generation:
        metadata["force_debug_program_generation"] = True
        metadata["debug_invalid_output"] = True
        if program_name and not str(program_name).startswith("Debug_"):
            program_name = f"Debug_{program_name}"
    return ProfileEvaluationRequest(
        request_id=request_id,
        robot_name=base_request.robot_name,
        frame_name=base_request.frame_name,
        motion_settings=dict(base_request.motion_settings),
        reference_pose_rows=reference_pose_rows,
        frame_a_origin_yz_profile_mm=tuple(
            (float(dy_mm), float(dz_mm))
            for dy_mm, dz_mm in selected_result.frame_a_origin_yz_profile_mm
        ),
        row_labels=tuple(str(label) for label in selected_result.row_labels or base_request.row_labels),
        inserted_flags=tuple(bool(flag) for flag in selected_result.inserted_flags or base_request.inserted_flags),
        strategy="exact_profile",
        start_joints=base_request.start_joints,
        run_window_repair=False,
        run_inserted_repair=False,
        include_pose_rows_in_result=False,
        create_program=True,
        program_name=program_name or base_request.program_name,
        optimized_csv_path=optimized_csv_path or base_request.optimized_csv_path,
        metadata=metadata,
    )


def _can_run_server_side_ik_eval(batch_request: EvaluationBatchRequest) -> bool:
    return all(
        request.motion_settings.get("ik_backend") == "six_axis_ik" and not request.create_program
        for request in batch_request.evaluations
    )


def _unlink_if_present(path: Path | None) -> None:
    if path is not None and path.exists():
        path.unlink()


def setup_server(
    *,
    host: str,
    server_dir: str,
    env_name: str,
    enable_slurm_setup: bool = False,
    log_options: CommandLogOptions | None = None,
) -> None:
    log_options = log_options or CommandLogOptions()
    remote_shell_dir = _remote_dir_for_shell(server_dir)
    _sync_source_tree_to_server(host=host, server_dir=server_dir, log_options=log_options)

    remote_setup_inner_script = f"""
set -e
cd "{remote_shell_dir}"
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
if conda env list | awk '{{print $1}}' | grep -Fxq "{env_name}"; then
    conda env update -n "{env_name}" -f environment.server.yml --prune
else
    conda env create -n "{env_name}" -f environment.server.yml
fi
conda activate "{env_name}"
python online_requester.py --help >/tmp/wps_requester_help.txt
python online_roundtrip.py --help >/tmp/wps_roundtrip_help.txt
python online_worker.py --help >/tmp/wps_worker_help.txt
python scripts/sweep_target_origin_yz.py --help >/tmp/wps_sweep_help.txt
python main.py --help >/tmp/wps_main_help.txt
python - <<'PY'
import importlib.util
spec = importlib.util.find_spec("robodk")
print("robodk_spec", spec)
if spec is not None:
    raise SystemExit("Server environment unexpectedly contains robodk")
PY
"""
    remote_setup_slurm_script = f"""
set -e
if command -v srun >/dev/null 2>&1 && command -v sinfo >/dev/null 2>&1; then
    partition="${{WPS_SERVER_PARTITION:-amd96c}}"
    idle_cpu="$(sinfo -h -p "$partition" -o '%C' 2>/dev/null | awk -F/ 'NR == 1 {{print $2}}')"
    case "$idle_cpu" in
        ''|*[!0-9]*) idle_cpu=1 ;;
    esac
    cpu_req="${{WPS_SETUP_SERVER_CPUS:-4}}"
    case "$cpu_req" in
        ''|*[!0-9]*) cpu_req=4 ;;
    esac
    if [ "$cpu_req" -gt "$idle_cpu" ]; then
        cpu_req="$idle_cpu"
    fi
    if [ "$cpu_req" -lt 1 ]; then
        cpu_req=1
    fi

    node_name="$(sinfo -h -p "$partition" -N -o '%N' 2>/dev/null | head -n 1)"
    case "$node_name" in
        ''|*[!A-Za-z0-9._-]*) node_name=master ;;
    esac
    free_mem_mb="$(scontrol show node "$node_name" 2>/dev/null | sed -n 's/.*FreeMem=\\([0-9]\\+\\).*/\\1/p' | head -n 1)"
    case "$free_mem_mb" in
        ''|*[!0-9]*) free_mem_mb=16384 ;;
    esac
    mem_req_mb="${{WPS_SETUP_SERVER_MEM_MB:-16384}}"
    case "$mem_req_mb" in
        ''|*[!0-9]*) mem_req_mb=16384 ;;
    esac
    mem_cap_mb="$free_mem_mb"
    if [ "$mem_cap_mb" -gt 4096 ]; then
        mem_cap_mb=$((mem_cap_mb - 1024))
    fi
    if [ "$mem_req_mb" -gt "$mem_cap_mb" ]; then
        mem_req_mb="$mem_cap_mb"
    fi
    if [ "$mem_req_mb" -lt 2048 ]; then
        mem_req_mb=2048
    fi
    echo "[setup-server] Slurm allocation: partition=$partition cpus=$cpu_req mem=${{mem_req_mb}}M"
    srun -p "$partition" -c "$cpu_req" --mem="${{mem_req_mb}}M" bash -lc {shlex.quote(remote_setup_inner_script)}
elif [ "${{WPS_ALLOW_LOGIN_NODE_SERVER:-0}}" = "1" ]; then
    echo "[setup-server] WARNING: running environment setup on login node because WPS_ALLOW_LOGIN_NODE_SERVER=1"
    bash -lc {shlex.quote(remote_setup_inner_script)}
else
    echo "[setup-server] Slurm is required for server environment setup on master; refusing login-node run." >&2
    exit 125
fi
"""
    remote_setup_login_script = f"""
set -e
echo "[setup-server] Slurm setup is disabled by configuration; running on login node."
bash -lc {shlex.quote(remote_setup_inner_script)}
"""
    remote_setup_script = (
        remote_setup_slurm_script if enable_slurm_setup else remote_setup_login_script
    )
    _run_stage(
        "[setup-server] Creating or updating server conda environment...",
        lambda: _run_remote_bash(
            host,
            remote_setup_script,
            log_options=log_options,
            echo_output=False,
        ),
        log_options=log_options,
    )
    print(f"[setup-server] Done. Server role ready at {server_dir}")
    if log_options.log_path is not None:
        print(f"[setup-server] Detailed log: {log_options.log_path}")


def run_server_role(
    *,
    request_path: str | Path,
    run_id: str,
    final_program_name: str | None = None,
    optimized_csv_path: str | None = None,
    retry_candidate_limit: int = 4,
    retry_repair_limit: int = 0,
    retry_max_rounds: int = 1,
    allow_invalid_outputs: bool = False,
    log_options: CommandLogOptions | None = None,
) -> ServerRoleArtifacts:
    run_dir = _local_artifact_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    effective_log_options = log_options or CommandLogOptions(log_path=run_dir / "server_role.log")

    remote_request = _load_remote_request(request_path)
    if remote_request.base_request.create_program:
        raise RuntimeError("online/server refuses create_program=True requests.")

    local_request_path = write_json_file(run_dir / "request.json", remote_request.to_dict())
    candidates_path = run_dir / "candidates.json"
    results_path = run_dir / "results.json"
    summary_path = run_dir / "summary.json"
    profile_retry_summary_path = run_dir / "profile_retry_summary.json"
    final_request_path = run_dir / "final_generate_request.json"
    handoff_package_path = run_dir / "handoff_package.json"
    debug_final_request_path = run_dir / "debug_final_generate_request.json"
    debug_handoff_package_path = run_dir / "debug_handoff_package.json"
    _unlink_if_present(profile_retry_summary_path)
    _unlink_if_present(final_request_path)
    _unlink_if_present(handoff_package_path)
    _unlink_if_present(debug_final_request_path)
    _unlink_if_present(debug_handoff_package_path)

    candidate_batch = _run_stage(
        "[online/server] Generating candidate batch...",
        lambda: propose_candidates(remote_request),
        log_options=effective_log_options,
    )
    write_json_file(candidates_path, candidate_batch.to_dict())
    if not candidate_batch.evaluations:
        empty_results = EvaluationBatchResult(results=())
        write_json_file(results_path, empty_results.to_dict())
        summary = summarize_results(())
        write_json_file(
            summary_path,
            {
                **summary.to_dict(),
                "server_role": {
                    "status": "invalid",
                    "delivery_ready": False,
                    "strict_delivery_ready": False,
                    "selected_request_id": None,
                },
                "server_final_selection": None,
            },
        )
        print("[online/server] No candidates were produced; diagnostics only.")
        return ServerRoleArtifacts(
            run_id=run_id,
            run_dir=run_dir,
            request_path=local_request_path,
            candidates_path=candidates_path,
            results_path=results_path,
            summary_path=summary_path,
            profile_retry_summary_path=None,
            final_request_path=None,
            handoff_package_path=None,
            debug_final_request_path=None,
            debug_handoff_package_path=None,
            best_request_id=None,
            result_status="invalid",
            log_path=effective_log_options.log_path,
        )

    if not _can_run_server_side_ik_eval(candidate_batch):
        raise RuntimeError(
            "online/server only supports offline SixAxisIK evaluation "
            "(ik_backend='six_axis_ik', create_program=False)."
        )

    selected_search_result_holder: dict[str, Any] = {"value": None}

    def _evaluate_candidates() -> EvaluationBatchResult:
        from src.robodk_runtime.eval_worker import (
            evaluate_batch_request,
            evaluate_single_request,
        )

        if len(candidate_batch.evaluations) == 1:
            single_result, single_search_result = evaluate_single_request(
                candidate_batch.evaluations[0]
            )
            selected_search_result_holder["value"] = single_search_result
            return EvaluationBatchResult(results=(single_result,))
        return evaluate_batch_request(candidate_batch)

    batch_result = _run_stage(
        "[online/server] Evaluating candidates with offline SixAxisIK...",
        _evaluate_candidates,
        log_options=effective_log_options,
    )
    write_json_file(results_path, batch_result.to_dict())

    summary = summarize_results(batch_result.results)
    selected_result = (
        _find_result_by_id(batch_result, summary.best_request_id)
        if summary.best_request_id is not None
        else _select_best_result(batch_result.results)
    )
    best_request_id = None if selected_result is None else str(selected_result.request_id)
    if selected_result is not None:
        print(
            "[online/server] Best candidate selected: "
            f"{selected_result.request_id} "
            f"(status={selected_result.status}, "
            f"ik_empty_rows={selected_result.ik_empty_row_count}, "
            f"config_switches={selected_result.config_switches}, "
            f"bridge_like_segments={selected_result.bridge_like_segments}, "
            f"big_circle_step_count={int(getattr(selected_result, 'big_circle_step_count', 0))})"
        )

    retry_candidate_budget = max(0, int(retry_candidate_limit))
    retry_repair_budget = max(0, int(retry_repair_limit))
    retry_round_budget = max(0, int(retry_max_rounds))
    retry_enabled = retry_candidate_budget > 0 and retry_round_budget > 0
    retry_summary_path: Path | None = None
    if selected_result is not None and not result_is_strictly_valid(selected_result) and retry_enabled:
        def _retry():
            from src.runtime.local_retry import run_local_profile_retry
            baseline_search_result = selected_search_result_holder.get("value")
            selected_request = next(
                (
                    request
                    for request in candidate_batch.evaluations
                    if str(request.request_id) == str(selected_result.request_id)
                ),
                None,
            )
            if selected_request is not None and baseline_search_result is None:
                try:
                    from src.robodk_runtime.eval_worker import evaluate_single_request

                    _replayed_result, baseline_search_result = evaluate_single_request(
                        selected_request
                    )
                except Exception as exc:
                    print(
                        "[online/server] Retry baseline exact-search reuse unavailable: "
                        f"{exc}"
                    )
                    baseline_search_result = None

            return run_local_profile_retry(
                remote_request.base_request,
                selected_result,
                candidate_limit=retry_candidate_budget,
                repair_retry_limit=retry_repair_budget,
                max_rounds=retry_round_budget,
                baseline_search_result=baseline_search_result,
            )

        retry_outcome = _run_stage(
            "[online/server] Running server-side profile retry/repair...",
            _retry,
            log_options=effective_log_options,
        )
        retry_summary_path = write_json_file(
            profile_retry_summary_path,
            {
                **retry_outcome.payload,
                "budget": {
                    "candidate_limit": retry_candidate_budget,
                    "repair_retry_limit": retry_repair_budget,
                    "max_rounds": retry_round_budget,
                },
            },
        )
        if _result_sort_key(retry_outcome.best_result) < _result_sort_key(selected_result):
            selected_result = retry_outcome.best_result
            best_request_id = str(selected_result.request_id)
        print(
            "[online/server] Retry best: "
            f"status={retry_outcome.best_result.status}, "
            f"worst_joint_step={retry_outcome.best_result.worst_joint_step_deg:.3f}"
        )
    elif selected_result is not None and not result_is_strictly_valid(selected_result):
        print(
            "[online/server] Server-side profile retry/repair skipped by retry budget "
            f"(candidate_limit={retry_candidate_budget}, "
            f"repair_limit={retry_repair_budget}, max_rounds={retry_round_budget})."
        )

    strict_ready = bool(selected_result is not None and result_is_strictly_valid(selected_result))
    selection_summary = (
        result_quality_summary(selected_result)
        if selected_result is not None
        else None
    )
    summary_payload: dict[str, Any] = summary.to_dict()
    summary_payload["server_role"] = {
        "status": "valid" if strict_ready else "invalid",
        "delivery_ready": strict_ready,
        "strict_delivery_ready": strict_ready,
        "selected_request_id": best_request_id,
        "gate_tier": None if selection_summary is None else selection_summary.get("gate_tier"),
        "block_reasons": [] if selection_summary is None else selection_summary.get("block_reasons", []),
    }
    summary_payload["server_final_selection"] = (
        None
        if selected_result is None
        else {**selection_summary}
    )
    write_json_file(summary_path, summary_payload)
    if summary.conclusion:
        print(f"[online/server] {summary.conclusion}")
    for note in summary.notes:
        print(f"[online/server] {note}")
    if selected_result is not None:
        print(
            "[online/server] Final selection after retry/repair: "
            f"{selected_result.request_id} "
            f"(status={selected_result.status}, "
            f"ik_empty_rows={selected_result.ik_empty_row_count}, "
            f"config_switches={selected_result.config_switches}, "
            f"bridge_like_segments={selected_result.bridge_like_segments}, "
            f"big_circle_step_count={int(getattr(selected_result, 'big_circle_step_count', 0))}, "
            f"worst_joint_step={selected_result.worst_joint_step_deg:.3f} deg)."
        )

    final_request_path_output: Path | None = None
    handoff_package_path_output: Path | None = None
    debug_final_request_path_output: Path | None = None
    debug_handoff_package_path_output: Path | None = None
    result_status = "invalid"
    if selected_result is not None and strict_ready:
        receiver_request = _build_receiver_request_from_result(
            remote_request.base_request,
            selected_result,
            request_id=f"{selected_result.request_id}_receiver_generate",
            program_name=final_program_name,
            optimized_csv_path=optimized_csv_path,
        )
        final_request_path_output = write_json_file(final_request_path, receiver_request.to_dict())
        handoff_package_path_output = write_handoff_package(
            handoff_package_path,
            run_id=run_id,
            receiver_request=receiver_request,
            selected_result=selected_result,
            artifacts={
                "request": local_request_path,
                "candidates": candidates_path,
                "results": results_path,
                "summary": summary_path,
                "receiver_request": final_request_path_output,
            },
            diagnostics_artifacts={
                "request": local_request_path,
                "candidates": candidates_path,
                "results": results_path,
                "summary": summary_path,
                "profile_retry_summary": retry_summary_path,
                "server_log": effective_log_options.log_path,
            },
        )
        result_status = "valid"
        print(f"[online/server] Deliverable handoff package: {handoff_package_path_output}")
    else:
        _unlink_if_present(final_request_path)
        _unlink_if_present(handoff_package_path)
        if selected_result is not None and allow_invalid_outputs:
            receiver_request = _build_receiver_request_from_result(
                remote_request.base_request,
                selected_result,
                request_id=f"{selected_result.request_id}_debug_receiver_generate",
                program_name=final_program_name,
                optimized_csv_path=optimized_csv_path,
                force_debug_program_generation=True,
            )
            debug_final_request_path_output = write_json_file(
                debug_final_request_path,
                receiver_request.to_dict(),
            )
            debug_handoff_package_path_output = write_handoff_package(
                debug_handoff_package_path,
                run_id=run_id,
                receiver_request=receiver_request,
                selected_result=selected_result,
                artifacts={
                    "request": local_request_path,
                    "candidates": candidates_path,
                    "results": results_path,
                    "summary": summary_path,
                    "debug_receiver_request": debug_final_request_path_output,
                },
                diagnostics_artifacts={
                    "request": local_request_path,
                    "candidates": candidates_path,
                    "results": results_path,
                    "summary": summary_path,
                    "profile_retry_summary": retry_summary_path,
                    "server_log": effective_log_options.log_path,
                },
                allow_invalid=True,
                package_kind="debug_invalid_handoff",
            )
            print(
                "[online/server] Delivery gate blocked official handoff; "
                f"debug handoff written: {debug_handoff_package_path_output}"
            )
        else:
            print("[online/server] Delivery gate blocked handoff; diagnostics were kept.")

    print(f"[online/server] Artifacts: {run_dir}")
    if effective_log_options.log_path is not None:
        print(f"[online/server] Detailed log: {effective_log_options.log_path}")

    return ServerRoleArtifacts(
        run_id=run_id,
        run_dir=run_dir,
        request_path=local_request_path,
        candidates_path=candidates_path,
        results_path=results_path,
        summary_path=summary_path,
        profile_retry_summary_path=retry_summary_path,
        final_request_path=final_request_path_output,
        handoff_package_path=handoff_package_path_output,
        debug_final_request_path=debug_final_request_path_output,
        debug_handoff_package_path=debug_handoff_package_path_output,
        best_request_id=best_request_id,
        result_status=result_status,
        log_path=effective_log_options.log_path,
    )


def run_receiver_role(
    *,
    handoff_path: str | Path,
    run_id: str,
    local_python: str | None,
    allow_invalid_handoff: bool = False,
    log_options: CommandLogOptions | None = None,
) -> ReceiverRoleArtifacts:
    run_dir = _local_artifact_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    effective_log_options = log_options or CommandLogOptions(log_path=run_dir / "receiver_role.log")

    handoff_package_path = Path(handoff_path)
    handoff_payload = load_handoff_package(
        handoff_package_path,
        allow_invalid=allow_invalid_handoff,
    )
    selection_payload = dict(handoff_payload.get("selection", {}))
    if not allow_invalid_handoff and not bool(selection_payload.get("strictly_valid")):
        raise RuntimeError("Receiver refused a handoff package without a deliverable selection.")
    if allow_invalid_handoff and not bool(selection_payload.get("strictly_valid")):
        print("[online/receiver] Debug mode: accepting non-deliverable handoff.")

    receiver_request = ProfileEvaluationRequest.from_dict(dict(handoff_payload["receiver_request"]))
    if not receiver_request.create_program:
        raise RuntimeError("Receiver request must set create_program=True.")
    receiver_metadata = dict(receiver_request.metadata)
    receiver_metadata["entrypoint"] = "online_receiver_handoff"
    receiver_metadata["receiver_materialization_mode"] = "direct_handoff_import"
    receiver_request = replace(
        receiver_request,
        run_window_repair=False,
        run_inserted_repair=False,
        metadata=receiver_metadata,
    )

    receiver_request_path = write_json_file(
        run_dir / "receiver_request.json",
        receiver_request.to_dict(),
    )
    result_path = run_dir / "final_generate_result.json"

    if local_python:
        _append_log(
            effective_log_options.log_path,
            "[online/receiver] local_python was provided but direct handoff import runs "
            "in the current process; no worker interpreter is needed.",
        )

    def _direct_import_handoff():
        from src.core.collab_models import EvaluationBatchResult
        from src.robodk_runtime.result_import import (
            import_profile_result_to_robodk,
            profile_result_from_handoff_payload,
            write_optimized_pose_csv_from_result,
        )

        started = perf_counter()
        selected_result = profile_result_from_handoff_payload(handoff_payload)
        import_summary = import_profile_result_to_robodk(
            selected_result,
            robot_name=receiver_request.robot_name,
            frame_name=receiver_request.frame_name,
            program_name=receiver_request.program_name,
            prefix=receiver_request.program_name or f"WPS_{run_id}",
            clear_prefix=True,
            create_program=True,
            create_cartesian_markers=False,
            move_type=str(receiver_request.motion_settings.get("move_type", "MoveJ")),
        )

        optimized_output = None
        if receiver_request.optimized_csv_path:
            optimized_output = write_optimized_pose_csv_from_result(
                receiver_request.optimized_csv_path,
                selected_result,
            )

        materialized_program = import_summary.program_name
        direct_valid = result_is_strictly_valid(selected_result) and bool(materialized_program)
        metadata = dict(selected_result.metadata)
        metadata.update(
            {
                "program_name": materialized_program,
                "receiver_direct_import": True,
                "receiver_marker_count": int(import_summary.marker_count),
                "receiver_program_target_count": int(import_summary.program_target_count),
                "receiver_optimized_pose_csv": None
                if optimized_output is None
                else str(optimized_output),
            }
        )
        final_result = replace(
            selected_result,
            status="valid" if direct_valid else "invalid",
            timing_seconds=perf_counter() - started,
            metadata=metadata,
            diagnostics=None
            if direct_valid
            else "Direct handoff import did not produce a deliverable RoboDK program.",
        )
        write_json_file(result_path, EvaluationBatchResult(results=(final_result,)).to_dict())
        return final_result, import_summary, optimized_output

    final_result, import_summary, optimized_pose_csv_output = _run_stage(
        "[online/receiver] Importing server-selected path directly into RoboDK...",
        _direct_import_handoff,
        log_options=effective_log_options,
    )

    materialized_program_name = import_summary.program_name
    receiver_valid = result_is_strictly_valid(final_result) and bool(materialized_program_name)

    if receiver_valid:
        print(f"[online/receiver] Program generated successfully: {materialized_program_name}")
        final_program_name = materialized_program_name
        result_status = "valid"
    else:
        diagnostics = (
            final_result.diagnostics
            or final_result.error_message
            or "unknown receiver failure"
        )
        if allow_invalid_handoff and materialized_program_name:
            print(
                "[online/receiver] Debug program generated despite invalid path: "
                f"{materialized_program_name}"
            )
            final_program_name = materialized_program_name
        else:
            final_program_name = None
        print(
            "[online/receiver] Delivery gate blocked official receiver outputs. "
            f"Reason: {diagnostics}"
        )
        optimized_pose_csv_output = None
        result_status = "invalid"

    print(f"[online/receiver] Artifacts: {run_dir}")
    if effective_log_options.log_path is not None:
        print(f"[online/receiver] Detailed log: {effective_log_options.log_path}")

    return ReceiverRoleArtifacts(
        run_id=run_id,
        run_dir=run_dir,
        handoff_package_path=handoff_package_path,
        receiver_request_path=receiver_request_path,
        result_path=result_path,
        optimized_pose_csv_path=optimized_pose_csv_output,
        final_program_name=final_program_name,
        result_status=result_status,
        log_path=effective_log_options.log_path,
    )


def _try_download_remote_file(
    *,
    host: str,
    remote_path: str,
    local_path: Path,
    log_options: CommandLogOptions,
) -> Path | None:
    _unlink_if_present(local_path)
    command = ["scp", f"{host}:{remote_path}", str(local_path)]
    attempts = 3
    for attempt_index in range(1, attempts + 1):
        try:
            _run_local_command(
                command,
                log_options=log_options,
                echo_output=False,
            )
            return local_path
        except subprocess.CalledProcessError as exc:
            if _looks_like_missing_remote_artifact(exc):
                _append_log(log_options.log_path, f"Optional remote artifact missing: {remote_path}")
                return None
            if attempt_index >= attempts:
                _append_log(log_options.log_path, f"Optional remote artifact unavailable: {remote_path}")
                return None
            message = (
                f"[retry] optional artifact download failed on attempt {attempt_index}/{attempts}; "
                "retrying in 1.0s."
            )
            _append_log(log_options.log_path, message)
            sleep(1.0)
    return local_path


def run_online_coordinator(
    *,
    host: str,
    server_dir: str,
    env_name: str,
    request_path: str | Path,
    run_id: str,
    local_python: str | None,
    generate_final_program: bool = True,
    final_program_name: str | None = None,
    optimized_csv_path: str | None = None,
    retry_candidate_limit: int = DEFAULT_ONLINE_RETRY_CANDIDATE_LIMIT,
    retry_repair_limit: int = DEFAULT_ONLINE_RETRY_REPAIR_LIMIT,
    retry_max_rounds: int = DEFAULT_ONLINE_RETRY_MAX_ROUNDS,
    allow_invalid_outputs: bool = False,
    enforce_remote_sync_guard: bool = True,
    remote_sync_mode: str = SYNC_MODE_GUARD,
    log_options: CommandLogOptions | None = None,
) -> RoundtripArtifacts:
    local_run_dir = _local_artifact_dir(run_id)
    local_run_dir.mkdir(parents=True, exist_ok=True)
    effective_log_options = log_options or CommandLogOptions(
        log_path=local_run_dir / "online_coordinator.log",
    )

    remote_request = _load_remote_request(request_path)
    local_request_path = write_json_file(local_run_dir / "request.json", remote_request.to_dict())
    local_candidates_path = local_run_dir / "candidates.json"
    local_results_path = local_run_dir / "results.json"
    local_summary_path = local_run_dir / "summary.json"
    local_profile_retry_summary_path = local_run_dir / "profile_retry_summary.json"
    local_final_request_path = local_run_dir / "final_generate_request.json"
    local_final_result_path = local_run_dir / "final_generate_result.json"
    local_handoff_package_path = local_run_dir / "handoff_package.json"
    local_debug_final_request_path = local_run_dir / "debug_final_generate_request.json"
    local_debug_handoff_package_path = local_run_dir / "debug_handoff_package.json"
    for stale_path in (
        local_candidates_path,
        local_results_path,
        local_summary_path,
        local_profile_retry_summary_path,
        local_final_request_path,
        local_final_result_path,
        local_handoff_package_path,
        local_debug_final_request_path,
        local_debug_handoff_package_path,
        local_run_dir / "server_role.log",
    ):
        _unlink_if_present(stale_path)

    remote_shell_dir = _remote_dir_for_shell(server_dir)
    _run_coordinator_sync_preflight(
        host=host,
        server_dir=server_dir,
        enforce_remote_sync_guard=bool(enforce_remote_sync_guard),
        remote_sync_mode=remote_sync_mode,
        log_options=effective_log_options,
    )
    remote_run_dir = f"{remote_shell_dir}/artifacts/online_runs/{run_id}"
    remote_run_dir_scp = f"{server_dir}/artifacts/online_runs/{run_id}"
    remote_request_arg = f"artifacts/online_runs/{run_id}/request.json"
    remote_log_arg = f"artifacts/online_runs/{run_id}/server_role.log"

    _run_stage(
        "[online/coordinator] Creating remote run directory...",
        lambda: _run_remote_bash_with_retries(
            host,
            f'mkdir -p "{remote_run_dir}"',
            log_options=effective_log_options,
            echo_output=False,
        ),
        log_options=effective_log_options,
    )
    _run_stage(
        "[online/coordinator] Uploading request to server...",
        lambda: _run_local_command_with_retries(
            ["scp", str(local_request_path), f"{host}:{remote_run_dir_scp}/request.json"],
            log_options=effective_log_options,
            echo_output=False,
        ),
        log_options=effective_log_options,
    )

    remote_server_args = [
        "python",
        "online_roundtrip.py",
        "run-server",
        "--request",
        remote_request_arg,
        "--run-id",
        run_id,
        "--retry-candidate-limit",
        str(max(0, int(retry_candidate_limit))),
        "--retry-repair-limit",
        str(max(0, int(retry_repair_limit))),
        "--retry-max-rounds",
        str(max(0, int(retry_max_rounds))),
        "--log-file",
        remote_log_arg,
    ]
    if allow_invalid_outputs:
        remote_server_args.append("--allow-invalid-outputs")
    if final_program_name:
        remote_server_args.extend(["--program-name", final_program_name])
    if optimized_csv_path:
        remote_server_args.extend(["--optimized-csv-path", optimized_csv_path])

    remote_env_exports = _remote_server_env_export_block()
    remote_env_exports_block = (
        f"{remote_env_exports}\n" if remote_env_exports else ""
    )

    remote_server_script = f"""
set -e
cd "{remote_shell_dir}"
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "{env_name}"
{remote_env_exports_block}
export OMP_NUM_THREADS="${{OMP_NUM_THREADS:-1}}"
export OPENBLAS_NUM_THREADS="${{OPENBLAS_NUM_THREADS:-1}}"
export MKL_NUM_THREADS="${{MKL_NUM_THREADS:-1}}"
export NUMEXPR_NUM_THREADS="${{NUMEXPR_NUM_THREADS:-1}}"
if command -v srun >/dev/null 2>&1 && command -v sinfo >/dev/null 2>&1; then
    partition="${{WPS_SERVER_PARTITION:-amd96c}}"
    idle_cpu="$(sinfo -h -p "$partition" -o '%C' 2>/dev/null | awk -F/ 'NR == 1 {{print $2}}')"
    case "$idle_cpu" in
        ''|*[!0-9]*) idle_cpu=1 ;;
    esac
    cpu_req="${{WPS_SERVER_CPUS:-16}}"
    case "$cpu_req" in
        ''|*[!0-9]*) cpu_req=16 ;;
    esac
    if [ "$cpu_req" -gt "$idle_cpu" ]; then
        cpu_req="$idle_cpu"
    fi
    if [ "$cpu_req" -lt 1 ]; then
        cpu_req=1
    fi

    node_name="$(sinfo -h -p "$partition" -N -o '%N' 2>/dev/null | head -n 1)"
    case "$node_name" in
        ''|*[!A-Za-z0-9._-]*) node_name=master ;;
    esac
    free_mem_mb="$(scontrol show node "$node_name" 2>/dev/null | sed -n 's/.*FreeMem=\\([0-9]\\+\\).*/\\1/p' | head -n 1)"
    case "$free_mem_mb" in
        ''|*[!0-9]*) free_mem_mb=65536 ;;
    esac
    mem_req_mb="${{WPS_SERVER_MEM_MB:-65536}}"
    case "$mem_req_mb" in
        ''|*[!0-9]*) mem_req_mb=65536 ;;
    esac
    mem_cap_mb="$free_mem_mb"
    if [ "$mem_cap_mb" -gt 8192 ]; then
        mem_cap_mb=$((mem_cap_mb - 2048))
    fi
    if [ "$mem_req_mb" -gt "$mem_cap_mb" ]; then
        mem_req_mb="$mem_cap_mb"
    fi
    if [ "$mem_req_mb" -lt 4096 ]; then
        mem_req_mb=4096
    fi

    profile_workers="${{WPS_SERVER_PROFILE_WORKERS:-$cpu_req}}"
    case "$profile_workers" in
        ''|*[!0-9]*) profile_workers="$cpu_req" ;;
    esac
    profile_workers_cap="${{WPS_SERVER_PROFILE_WORKERS_MAX:-16}}"
    case "$profile_workers_cap" in
        ''|*[!0-9]*) profile_workers_cap=16 ;;
    esac
    if [ "$profile_workers_cap" -lt 1 ]; then
        profile_workers_cap=1
    fi
    if [ "$profile_workers_cap" -gt "$cpu_req" ]; then
        profile_workers_cap="$cpu_req"
    fi
    if [ "$profile_workers" -gt "$cpu_req" ]; then
        profile_workers="$cpu_req"
    fi
    if [ "$profile_workers" -gt "$profile_workers_cap" ]; then
        profile_workers="$profile_workers_cap"
    fi
    if [ "$profile_workers" -lt 1 ]; then
        profile_workers=1
    fi
    profile_min_batch="${{WPS_SERVER_PROFILE_MIN_BATCH_SIZE:-4}}"
    case "$profile_min_batch" in
        ''|*[!0-9]*) profile_min_batch=4 ;;
    esac
    if [ "$profile_min_batch" -lt 1 ]; then
        profile_min_batch=1
    fi
    profile_min_batch_cap="${{WPS_SERVER_PROFILE_MIN_BATCH_SIZE_CAP:-4}}"
    case "$profile_min_batch_cap" in
        ''|*[!0-9]*) profile_min_batch_cap=4 ;;
    esac
    if [ "$profile_min_batch_cap" -lt 1 ]; then
        profile_min_batch_cap=1
    fi
    if [ "${{WPS_SERVER_PROFILE_ALLOW_HIGH_MIN_BATCH:-0}}" != "1" ] && [ "$profile_min_batch" -gt "$profile_min_batch_cap" ]; then
        echo "[online/coordinator] Clamping WPS_SERVER_PROFILE_MIN_BATCH_SIZE from $profile_min_batch to $profile_min_batch_cap"
        profile_min_batch="$profile_min_batch_cap"
    fi
    export WPS_OFFLINE_BATCH_WORKERS="${{WPS_OFFLINE_BATCH_WORKERS:-$cpu_req}}"
    export WPS_SERVER_PROFILE_WORKERS="$profile_workers"
    export WPS_SERVER_PROFILE_MIN_BATCH_SIZE="$profile_min_batch"
    export WPS_LOCAL_RETRY_REPAIR_EVAL_MODE="${{WPS_LOCAL_RETRY_REPAIR_EVAL_MODE:-exact_reuse}}"
    echo "[online/coordinator] Slurm allocation: partition=$partition cpus=$cpu_req mem=${{mem_req_mb}}M profile_workers=$WPS_SERVER_PROFILE_WORKERS"
    set +e
    srun -p "$partition" -c "$cpu_req" --mem="${{mem_req_mb}}M" {_render_command(remote_server_args)}
    ROLE_STATUS=$?
    set -e
elif [ "${{WPS_ALLOW_LOGIN_NODE_SERVER:-0}}" = "1" ]; then
    echo "[online/coordinator] WARNING: running server role on login node because WPS_ALLOW_LOGIN_NODE_SERVER=1"
    set +e
    {_render_command(remote_server_args)}
    ROLE_STATUS=$?
    set -e
else
    echo "[online/coordinator] Slurm is required for server compute on master; refusing login-node run." >&2
    exit 125
fi
if [ "$ROLE_STATUS" -ne 0 ] && [ "$ROLE_STATUS" -ne 2 ]; then
    exit "$ROLE_STATUS"
fi
exit 0
"""
    _run_stage(
        "[online/coordinator] Server running compute role...",
        lambda: _run_remote_bash_with_retries(
            host,
            remote_server_script,
            log_options=effective_log_options,
            echo_output=effective_log_options.show_worker_output,
            attempts=3,
            delay_seconds=2.0,
        ),
        log_options=effective_log_options,
    )

    for artifact_name, local_path in (
        ("candidates.json", local_candidates_path),
        ("results.json", local_results_path),
        ("summary.json", local_summary_path),
        ("profile_retry_summary.json", local_profile_retry_summary_path),
        ("final_generate_request.json", local_final_request_path),
        ("handoff_package.json", local_handoff_package_path),
        ("debug_final_generate_request.json", local_debug_final_request_path),
        ("debug_handoff_package.json", local_debug_handoff_package_path),
        ("server_role.log", local_run_dir / "server_role.log"),
    ):
        _try_download_remote_file(
            host=host,
            remote_path=f"{remote_run_dir_scp}/{artifact_name}",
            local_path=local_path,
            log_options=effective_log_options,
        )

    final_request_path = local_final_request_path if local_final_request_path.exists() else None
    final_result_path: Path | None = None
    handoff_package_path = local_handoff_package_path if local_handoff_package_path.exists() else None
    debug_final_request_path = (
        local_debug_final_request_path if local_debug_final_request_path.exists() else None
    )
    debug_handoff_package_path = (
        local_debug_handoff_package_path if local_debug_handoff_package_path.exists() else None
    )
    optimized_pose_csv_output: Path | None = None
    final_program_name_output: str | None = None
    best_request_id: str | None = None
    result_status = "valid" if handoff_package_path is not None else "invalid"
    profile_retry_summary_path = (
        local_profile_retry_summary_path if local_profile_retry_summary_path.exists() else None
    )

    if local_summary_path.exists():
        summary_payload = load_json_file(local_summary_path)
        conclusion = summary_payload.get("conclusion")
        if conclusion:
            print(f"[online/coordinator] {conclusion}")
        role_payload = summary_payload.get("server_role")
        if isinstance(role_payload, dict) and role_payload.get("selected_request_id") is not None:
            best_request_id = str(role_payload.get("selected_request_id"))
        final_selection_payload = summary_payload.get("server_final_selection")
        if isinstance(final_selection_payload, dict):
            print(
                "[online/coordinator] Final server selection: "
                f"{final_selection_payload.get('request_id')} "
                f"(status={final_selection_payload.get('status')}, "
                f"ik_empty_rows={final_selection_payload.get('ik_empty_row_count')}, "
                f"config_switches={final_selection_payload.get('config_switches')}, "
                f"bridge_like_segments={final_selection_payload.get('bridge_like_segments')}, "
                f"big_circle_step_count={final_selection_payload.get('big_circle_step_count')}, "
                f"worst_joint_step={float(final_selection_payload.get('worst_joint_step_deg', 0.0)):.3f} deg)."
            )

    if handoff_package_path is None:
        if allow_invalid_outputs and debug_handoff_package_path is not None:
            print(
                "[online/coordinator] Server did not produce a deliverable handoff; "
                "using debug handoff for receiver diagnostics."
            )
            if generate_final_program:
                receiver_artifacts = run_receiver_role(
                    handoff_path=debug_handoff_package_path,
                    run_id=run_id,
                    local_python=local_python,
                    allow_invalid_handoff=True,
                    log_options=effective_log_options,
                )
                final_result_path = receiver_artifacts.result_path
                optimized_pose_csv_output = receiver_artifacts.optimized_pose_csv_path
                final_program_name_output = receiver_artifacts.final_program_name
                result_status = receiver_artifacts.result_status
        else:
            print(
                "[online/coordinator] Server did not produce a deliverable handoff package; "
                "receiver role will not run."
            )
    elif generate_final_program:
        receiver_artifacts = run_receiver_role(
            handoff_path=handoff_package_path,
            run_id=run_id,
            local_python=local_python,
            log_options=effective_log_options,
        )
        final_result_path = receiver_artifacts.result_path
        optimized_pose_csv_output = receiver_artifacts.optimized_pose_csv_path
        final_program_name_output = receiver_artifacts.final_program_name
        result_status = receiver_artifacts.result_status
    else:
        print("[online/coordinator] Final receiver generation skipped by configuration.")

    print(f"[online/coordinator] Artifacts: {local_run_dir}")
    if effective_log_options.log_path is not None:
        print(f"[online/coordinator] Detailed log: {effective_log_options.log_path}")

    return RoundtripArtifacts(
        run_id=run_id,
        local_run_dir=local_run_dir,
        request_path=local_request_path,
        candidates_path=local_candidates_path,
        results_path=local_results_path,
        summary_path=local_summary_path,
        profile_retry_summary_path=profile_retry_summary_path,
        final_request_path=final_request_path,
        final_result_path=final_result_path,
        handoff_package_path=handoff_package_path,
        debug_final_request_path=debug_final_request_path,
        debug_handoff_package_path=debug_handoff_package_path,
        optimized_pose_csv_path=optimized_pose_csv_output,
        final_program_name=final_program_name_output,
        best_request_id=best_request_id,
        result_status=result_status,
        log_path=effective_log_options.log_path,
    )


def run_round(
    *,
    host: str,
    server_dir: str,
    env_name: str,
    request_path: str | Path,
    run_id: str,
    local_python: str | None,
    server_eval_when_possible: bool = False,
    generate_final_program: bool = True,
    final_program_name: str | None = None,
    optimized_csv_path: str | None = None,
    retry_candidate_limit: int = DEFAULT_ONLINE_RETRY_CANDIDATE_LIMIT,
    retry_repair_limit: int = DEFAULT_ONLINE_RETRY_REPAIR_LIMIT,
    retry_max_rounds: int = DEFAULT_ONLINE_RETRY_MAX_ROUNDS,
    allow_invalid_outputs: bool = False,
    enforce_remote_sync_guard: bool = True,
    remote_sync_mode: str = SYNC_MODE_GUARD,
    log_options: CommandLogOptions | None = None,
) -> RoundtripArtifacts:
    if not server_eval_when_possible:
        print(
            "[run-round] Compatibility note: online now uses the server role for "
            "offline SixAxisIK evaluation by default."
        )
    return run_online_coordinator(
        host=host,
        server_dir=server_dir,
        env_name=env_name,
        request_path=request_path,
        run_id=run_id,
        local_python=local_python,
        generate_final_program=generate_final_program,
        final_program_name=final_program_name,
        optimized_csv_path=optimized_csv_path,
        retry_candidate_limit=retry_candidate_limit,
        retry_repair_limit=retry_repair_limit,
        retry_max_rounds=retry_max_rounds,
        allow_invalid_outputs=allow_invalid_outputs,
        enforce_remote_sync_guard=enforce_remote_sync_guard,
        remote_sync_mode=remote_sync_mode,
        log_options=log_options,
    )


def run_worker_eval(
    *,
    request_path: str | Path,
    result_path: str | Path,
    local_python: str | None,
    log_options: CommandLogOptions | None = None,
) -> WorkerEvalArtifacts:
    effective_log_options = log_options or CommandLogOptions()
    request_path = Path(request_path)
    result_path = Path(result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    worker_python = resolve_local_worker_python(local_python)

    _run_stage(
        "[worker-eval] Local RoboDK worker evaluating request...",
        lambda: _run_local_command(
            [
                worker_python,
                str(REPO_ROOT / "online_worker.py"),
                "eval",
                "--request",
                str(request_path),
                "--result",
                str(result_path),
            ],
            log_options=effective_log_options,
            echo_output=effective_log_options.show_worker_output,
        ),
        log_options=effective_log_options,
    )
    print(f"[worker-eval] Result: {result_path}")
    if effective_log_options.log_path is not None:
        print(f"[worker-eval] Detailed log: {effective_log_options.log_path}")
    return WorkerEvalArtifacts(
        request_path=request_path,
        result_path=result_path,
        log_path=effective_log_options.log_path,
    )


def _add_coordinator_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default="master")
    parser.add_argument("--server-dir", default=DEFAULT_SERVER_DIR)
    parser.add_argument("--env", default=DEFAULT_ENV_NAME)
    parser.add_argument("--request", required=True)
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--local-python")
    parser.add_argument("--log-file")
    parser.add_argument("--show-command-details", action="store_true")
    parser.add_argument("--quiet-worker-output", action="store_true")
    parser.add_argument("--skip-final-generate", action="store_true")
    parser.add_argument("--program-name")
    parser.add_argument("--optimized-csv-path")
    parser.add_argument(
        "--retry-candidate-limit",
        type=int,
        default=DEFAULT_ONLINE_RETRY_CANDIDATE_LIMIT,
    )
    parser.add_argument(
        "--retry-repair-limit",
        type=int,
        default=DEFAULT_ONLINE_RETRY_REPAIR_LIMIT,
    )
    parser.add_argument(
        "--retry-max-rounds",
        type=int,
        default=DEFAULT_ONLINE_RETRY_MAX_ROUNDS,
    )
    parser.add_argument(
        "--remote-sync-mode",
        choices=(SYNC_MODE_OFF, SYNC_MODE_GUARD, SYNC_MODE_PUSH),
        default=SYNC_MODE_GUARD,
        help=(
            "Coordinator sync policy before remote compute: "
            "off=skip check, guard=hash-check only, push=upload source then hash-check."
        ),
    )
    parser.add_argument(
        "--disable-sync-guard",
        action="store_true",
        help=(
            "Compatibility flag. Skip local/remote SHA256 guard. "
            "Equivalent to remote-sync-mode=off when used alone."
        ),
    )
    parser.add_argument(
        "--allow-invalid-outputs",
        action="store_true",
        help="Write debug_* artifacts and optionally run receiver diagnostics even when the delivery gate fails.",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Role-based online runner for Windows coordinator, server, and receiver flows."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup_parser = subparsers.add_parser(
        "setup-server",
        help="Sync code and create/update the server conda environment.",
    )
    setup_parser.add_argument("--host", default="master")
    setup_parser.add_argument("--server-dir", default=DEFAULT_SERVER_DIR)
    setup_parser.add_argument("--env", default=DEFAULT_ENV_NAME)
    setup_parser.add_argument(
        "--enable-slurm-setup",
        action="store_true",
        help="Run setup-server environment stage inside Slurm (default: disabled).",
    )
    setup_parser.add_argument("--log-file")
    setup_parser.add_argument("--show-command-details", action="store_true")

    build_parser = subparsers.add_parser(
        "build-request",
        help="Build a RemoteSearchRequest from current local settings.",
    )
    build_parser.add_argument("--request", required=True)
    build_parser.add_argument("--round-index", type=int, default=1)
    build_parser.add_argument("--candidate-limit", type=int, default=8)
    build_parser.add_argument("--skip-pose-solver", action="store_true")

    online_parser = subparsers.add_parser(
        "run-online",
        help="Run Windows coordinator: server compute then local receiver.",
    )
    _add_coordinator_args(online_parser)

    round_parser = subparsers.add_parser(
        "run-round",
        help="Compatibility alias for run-online/coordinator.",
    )
    _add_coordinator_args(round_parser)
    round_parser.add_argument("--server-eval-when-possible", action="store_true")

    server_parser = subparsers.add_parser("run-server", help="Run pure compute server role only.")
    server_parser.add_argument("--request", required=True)
    server_parser.add_argument("--run-id", default=_default_run_id())
    server_parser.add_argument("--log-file")
    server_parser.add_argument("--show-command-details", action="store_true")
    server_parser.add_argument("--program-name")
    server_parser.add_argument("--optimized-csv-path")
    server_parser.add_argument(
        "--retry-candidate-limit",
        type=int,
        default=DEFAULT_ONLINE_RETRY_CANDIDATE_LIMIT,
    )
    server_parser.add_argument(
        "--retry-repair-limit",
        type=int,
        default=DEFAULT_ONLINE_RETRY_REPAIR_LIMIT,
    )
    server_parser.add_argument(
        "--retry-max-rounds",
        type=int,
        default=DEFAULT_ONLINE_RETRY_MAX_ROUNDS,
    )
    server_parser.add_argument("--allow-invalid-outputs", action="store_true")

    receiver_parser = subparsers.add_parser("run-receiver", help="Run Windows receiver role only.")
    receiver_parser.add_argument("--handoff", required=True)
    receiver_parser.add_argument("--run-id", default=_default_run_id())
    receiver_parser.add_argument("--local-python")
    receiver_parser.add_argument("--log-file")
    receiver_parser.add_argument("--show-command-details", action="store_true")
    receiver_parser.add_argument("--quiet-worker-output", action="store_true")
    receiver_parser.add_argument("--allow-invalid-handoff", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        log_options = CommandLogOptions(
            log_path=None if getattr(args, "log_file", None) is None else Path(args.log_file),
            show_command_details=bool(getattr(args, "show_command_details", False)),
            show_worker_output=not bool(getattr(args, "quiet_worker_output", False)),
        )
        if args.command == "setup-server":
            setup_server(
                host=args.host,
                server_dir=args.server_dir,
                env_name=args.env,
                enable_slurm_setup=bool(args.enable_slurm_setup),
                log_options=log_options,
            )
            return 0
        if args.command == "build-request":
            from main import APP_RUNTIME_SETTINGS

            output_path = build_request_file(
                APP_RUNTIME_SETTINGS,
                args.request,
                round_index=args.round_index,
                candidate_limit=args.candidate_limit,
                refresh_csv=not args.skip_pose_solver,
            )
            print(f"Wrote request: {output_path}")
            return 0
        if args.command == "run-server":
            artifacts = run_server_role(
                request_path=args.request,
                run_id=args.run_id,
                final_program_name=args.program_name,
                optimized_csv_path=args.optimized_csv_path,
                retry_candidate_limit=args.retry_candidate_limit,
                retry_repair_limit=args.retry_repair_limit,
                retry_max_rounds=args.retry_max_rounds,
                allow_invalid_outputs=bool(args.allow_invalid_outputs),
                log_options=log_options,
            )
            return 0 if artifacts.result_status == "valid" else 2
        if args.command == "run-receiver":
            artifacts = run_receiver_role(
                handoff_path=args.handoff,
                run_id=args.run_id,
                local_python=args.local_python,
                allow_invalid_handoff=bool(args.allow_invalid_handoff),
                log_options=log_options,
            )
            return 0 if artifacts.result_status == "valid" else 2

        if args.command == "run-round":
            sync_mode = (
                SYNC_MODE_OFF if bool(args.disable_sync_guard) else str(args.remote_sync_mode)
            )
            artifacts = run_round(
                host=args.host,
                server_dir=args.server_dir,
                env_name=args.env,
                request_path=args.request,
                run_id=args.run_id,
                local_python=args.local_python,
                server_eval_when_possible=bool(args.server_eval_when_possible),
                generate_final_program=not bool(args.skip_final_generate),
                final_program_name=args.program_name,
                optimized_csv_path=args.optimized_csv_path,
                retry_candidate_limit=args.retry_candidate_limit,
                retry_repair_limit=args.retry_repair_limit,
                retry_max_rounds=args.retry_max_rounds,
                allow_invalid_outputs=bool(args.allow_invalid_outputs),
                enforce_remote_sync_guard=not bool(args.disable_sync_guard),
                remote_sync_mode=sync_mode,
                log_options=log_options,
            )
        else:
            sync_mode = (
                SYNC_MODE_OFF if bool(args.disable_sync_guard) else str(args.remote_sync_mode)
            )
            artifacts = run_online_coordinator(
                host=args.host,
                server_dir=args.server_dir,
                env_name=args.env,
                request_path=args.request,
                run_id=args.run_id,
                local_python=args.local_python,
                generate_final_program=not bool(args.skip_final_generate),
                final_program_name=args.program_name,
                optimized_csv_path=args.optimized_csv_path,
                retry_candidate_limit=args.retry_candidate_limit,
                retry_repair_limit=args.retry_repair_limit,
                retry_max_rounds=args.retry_max_rounds,
                allow_invalid_outputs=bool(args.allow_invalid_outputs),
                enforce_remote_sync_guard=not bool(args.disable_sync_guard),
                remote_sync_mode=sync_mode,
                log_options=log_options,
            )
        return 0 if artifacts.result_status == "valid" else 2
    except Exception as exc:
        _safe_print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
