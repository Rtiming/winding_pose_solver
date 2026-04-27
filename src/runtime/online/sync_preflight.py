from __future__ import annotations

import hashlib
import shlex
from pathlib import Path

from src.runtime.online.command_runner import (
    CommandLogOptions,
    _append_log,
    _env_flag,
    _run_local_command_with_retries,
    _run_remote_bash_with_retries,
    _run_stage,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SERVER_DIR = "/home/tzwang/program/winding_pose_solver"
SYNC_MODE_OFF = "off"
SYNC_MODE_GUARD = "guard"
SYNC_MODE_PUSH = "push"
SYNC_GUARD_FILES: tuple[str, ...] = (
    "app_settings.py",
    "main.py",
    "src/core/motion_settings.py",
    "src/core/types.py",
    "src/runtime/delivery.py",
    "src/runtime/local_profile.py",
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
    "src/runtime/online/command_runner.py",
    "src/runtime/online/roundtrip.py",
    "src/runtime/online/sync_preflight.py",
    "online_roundtrip.py",
)
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


def _remote_dir_for_shell(server_dir: str) -> str:
    if server_dir.startswith("~/"):
        return server_dir.replace("~/", "$HOME/", 1)
    if server_dir == "~":
        return "$HOME"
    return server_dir


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


def _remote_files_sha256(
    *,
    host: str,
    server_dir: str,
    relative_paths: tuple[str, ...],
    log_options: CommandLogOptions,
) -> dict[str, str | None]:
    """Return SHA-256 for several remote files through one SSH connection."""
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
