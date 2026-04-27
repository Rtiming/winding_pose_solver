from __future__ import annotations

import csv
import math
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.core.collab_models import load_json_file, write_json_file
from src.runtime.online.roundtrip import build_request_file
from src.runtime.run_logging import timestamp_token


REPO_ROOT = Path(__file__).resolve().parents[2]
ONLINE_RUNS_DIR = REPO_ROOT / "artifacts" / "online_runs"
SINGLE_RUNS_DIR = REPO_ROOT / "artifacts" / "local_runs"
STATE_DIR = REPO_ROOT / ".runtime" / "winding_runs"


@dataclass(frozen=True)
class _EntrypointDefaults:
    app_runtime_settings: Any
    online_host: str
    online_server_dir: str
    online_env_name: str
    remote_sync_mode: str


@lru_cache(maxsize=1)
def _entrypoint_defaults() -> _EntrypointDefaults:
    try:
        from src.runtime.main_entrypoint import (
            APP_RUNTIME_SETTINGS,
            ONLINE_ENV_NAME,
            ONLINE_HOST,
            ONLINE_SERVER_DIR,
            REMOTE_SYNC_MODE,
        )
    except Exception as exc:
        raise WindingRunError(
            "runtime_defaults_unavailable",
            "Failed to load runtime defaults from main_entrypoint.",
            detail={"error": f"{type(exc).__name__}: {exc}"},
            suggestion=(
                "Ensure winding_pose_solver runtime dependencies are installed, "
                "then retry."
            ),
            status_code=500,
        ) from exc
    return _EntrypointDefaults(
        app_runtime_settings=APP_RUNTIME_SETTINGS,
        online_host=str(ONLINE_HOST),
        online_server_dir=str(ONLINE_SERVER_DIR),
        online_env_name=str(ONLINE_ENV_NAME),
        remote_sync_mode=str(REMOTE_SYNC_MODE),
    )


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(value: datetime | None = None) -> str:
    dt = value or _now_utc()
    return dt.isoformat().replace("+00:00", "Z")


def _request_id() -> str:
    return uuid.uuid4().hex


def _as_float(value: Any, *, name: str) -> float:
    try:
        parsed = float(value)
    except Exception as exc:
        raise WindingRunError(
            "invalid_request",
            f"{name} must be numeric.",
            detail={"field": name, "value": value},
            suggestion="Send finite numeric values.",
            status_code=400,
        ) from exc
    if not math.isfinite(parsed):
        raise WindingRunError(
            "invalid_request",
            f"{name} must be finite.",
            detail={"field": name, "value": value},
            suggestion="Send finite numeric values.",
            status_code=400,
        )
    return parsed


def _normalize_vec3(values: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = math.sqrt(values[0] ** 2 + values[1] ** 2 + values[2] ** 2)
    if norm <= 1e-9:
        raise WindingRunError(
            "invalid_request",
            "Vector norm is too small.",
            detail={"vector": list(values)},
            suggestion="Use non-zero vectors for tangent/normal.",
            status_code=400,
        )
    return (values[0] / norm, values[1] / norm, values[2] / norm)


def _safe_run_id(raw: Any | None) -> str:
    if raw is None:
        return f"api_{timestamp_token()}_{uuid.uuid4().hex[:6]}"
    text = str(raw).strip()
    if not text:
        return f"api_{timestamp_token()}_{uuid.uuid4().hex[:6]}"
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)
    cleaned = cleaned.strip("_")
    return cleaned or f"api_{timestamp_token()}_{uuid.uuid4().hex[:6]}"


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _as_int(value: Any, *, default: int, minimum: int = 0) -> int:
    if value is None:
        return max(minimum, int(default))
    try:
        parsed = int(value)
    except Exception:
        return max(minimum, int(default))
    return max(minimum, parsed)


def _triplet(value: Any, *, field_name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise WindingRunError(
            "invalid_request",
            f"{field_name} must be [x, y, z].",
            detail={"field": field_name, "value": value},
            suggestion="Provide exactly 3 numeric values.",
            status_code=400,
        )
    return (
        _as_float(value[0], name=f"{field_name}[0]"),
        _as_float(value[1], name=f"{field_name}[1]"),
        _as_float(value[2], name=f"{field_name}[2]"),
    )


def _candidate_request_path(payload: dict[str, Any]) -> Path | None:
    direct = payload.get("request_path")
    if isinstance(direct, str) and direct.strip():
        return Path(direct)
    source = payload.get("request_source")
    if isinstance(source, dict):
        raw = source.get("request_path")
        if isinstance(raw, str) and raw.strip():
            return Path(raw)
    return None


def _candidate_handoff_path(payload: dict[str, Any]) -> Path | None:
    direct = payload.get("handoff_path")
    if isinstance(direct, str) and direct.strip():
        return Path(direct)
    source = payload.get("request_source")
    if isinstance(source, dict):
        raw = source.get("handoff_path")
        if isinstance(raw, str) and raw.strip():
            return Path(raw)
    return None


def _resolve_user_path(raw_path: Path) -> Path:
    return raw_path if raw_path.is_absolute() else (REPO_ROOT / raw_path)


def _resolve_centerline_path(
    *,
    payload: dict[str, Any],
    run_dir: Path,
) -> Path:
    source = payload.get("centerline_source")
    if source is None:
        defaults = _entrypoint_defaults()
        return _resolve_user_path(Path(defaults.app_runtime_settings.validation_centerline_csv))
    if not isinstance(source, dict):
        raise WindingRunError(
            "invalid_request",
            "centerline_source must be an object.",
            detail={"centerline_source_type": type(source).__name__},
            suggestion="Use centerline_source.file_path or centerline_source.inline_points.",
            status_code=400,
        )
    file_path = source.get("file_path")
    if isinstance(file_path, str) and file_path.strip():
        resolved = _resolve_user_path(Path(file_path).expanduser())
        if not resolved.is_file():
            raise WindingRunError(
                "centerline_not_found",
                "centerline_source.file_path does not exist.",
                detail={"file_path": str(resolved)},
                suggestion="Check the file path or export centerline CSV first.",
                status_code=404,
            )
        return resolved
    inline_points = source.get("inline_points")
    if inline_points is None:
        raise WindingRunError(
            "invalid_request",
            "centerline_source requires file_path or inline_points.",
            detail={"centerline_source_keys": sorted(source.keys())},
            suggestion="Provide file_path or inline_points.",
            status_code=400,
        )
    output_path = run_dir / "validation_centerline.csv"
    _write_inline_centerline_csv(inline_points, output_path)
    return output_path


def _pick_default_normal(tangent: tuple[float, float, float]) -> tuple[float, float, float]:
    tx, ty, tz = tangent
    if abs(tz) < 0.9:
        base = (0.0, 0.0, 1.0)
    else:
        base = (0.0, 1.0, 0.0)
    projection = tx * base[0] + ty * base[1] + tz * base[2]
    normal_raw = (
        base[0] - projection * tx,
        base[1] - projection * ty,
        base[2] - projection * tz,
    )
    return _normalize_vec3(normal_raw)


def _write_inline_centerline_csv(inline_points: Any, output_path: Path) -> None:
    if not isinstance(inline_points, list) or not inline_points:
        raise WindingRunError(
            "invalid_request",
            "centerline_source.inline_points must be a non-empty array.",
            detail={"inline_points_type": type(inline_points).__name__},
            suggestion="Provide a list of points.",
            status_code=400,
        )

    parsed: list[dict[str, float | int | None]] = []
    for idx, raw in enumerate(inline_points):
        if isinstance(raw, dict):
            x = _as_float(raw.get("x"), name=f"inline_points[{idx}].x")
            y = _as_float(raw.get("y"), name=f"inline_points[{idx}].y")
            z = _as_float(raw.get("z"), name=f"inline_points[{idx}].z")
            tx_raw = raw.get("tx")
            ty_raw = raw.get("ty")
            tz_raw = raw.get("tz")
            nx_raw = raw.get("nx")
            ny_raw = raw.get("ny")
            nz_raw = raw.get("nz")
        elif isinstance(raw, (list, tuple)) and len(raw) >= 3:
            x = _as_float(raw[0], name=f"inline_points[{idx}][0]")
            y = _as_float(raw[1], name=f"inline_points[{idx}][1]")
            z = _as_float(raw[2], name=f"inline_points[{idx}][2]")
            tx_raw = raw[3] if len(raw) > 3 else None
            ty_raw = raw[4] if len(raw) > 4 else None
            tz_raw = raw[5] if len(raw) > 5 else None
            nx_raw = raw[6] if len(raw) > 6 else None
            ny_raw = raw[7] if len(raw) > 7 else None
            nz_raw = raw[8] if len(raw) > 8 else None
        else:
            raise WindingRunError(
                "invalid_request",
                f"inline_points[{idx}] has unsupported shape.",
                detail={"point": raw},
                suggestion="Use object points (x,y,z,tx,ty,tz,nx,ny,nz) or numeric arrays.",
                status_code=400,
            )
        parsed.append(
            {
                "index": idx,
                "x": x,
                "y": y,
                "z": z,
                "tx": None if tx_raw is None else _as_float(tx_raw, name=f"inline_points[{idx}].tx"),
                "ty": None if ty_raw is None else _as_float(ty_raw, name=f"inline_points[{idx}].ty"),
                "tz": None if tz_raw is None else _as_float(tz_raw, name=f"inline_points[{idx}].tz"),
                "nx": None if nx_raw is None else _as_float(nx_raw, name=f"inline_points[{idx}].nx"),
                "ny": None if ny_raw is None else _as_float(ny_raw, name=f"inline_points[{idx}].ny"),
                "nz": None if nz_raw is None else _as_float(nz_raw, name=f"inline_points[{idx}].nz"),
            }
        )

    def _xyz(i: int) -> tuple[float, float, float]:
        row = parsed[i]
        return (float(row["x"]), float(row["y"]), float(row["z"]))

    total = len(parsed)
    arc_length = 0.0
    previous_xyz: tuple[float, float, float] | None = None
    output_rows: list[dict[str, float | int]] = []

    for idx, row in enumerate(parsed):
        current = _xyz(idx)
        if previous_xyz is not None:
            dx = current[0] - previous_xyz[0]
            dy = current[1] - previous_xyz[1]
            dz = current[2] - previous_xyz[2]
            arc_length += math.sqrt(dx * dx + dy * dy + dz * dz)
        previous_xyz = current

        if row["tx"] is not None and row["ty"] is not None and row["tz"] is not None:
            tangent = _normalize_vec3((float(row["tx"]), float(row["ty"]), float(row["tz"])))
        else:
            left = _xyz(idx - 1) if idx > 0 else current
            right = _xyz(idx + 1) if idx < total - 1 else current
            tangent = _normalize_vec3((right[0] - left[0], right[1] - left[1], right[2] - left[2]))

        if row["nx"] is not None and row["ny"] is not None and row["nz"] is not None:
            normal_hint = _normalize_vec3((float(row["nx"]), float(row["ny"]), float(row["nz"])))
            projection = (
                tangent[0] * normal_hint[0]
                + tangent[1] * normal_hint[1]
                + tangent[2] * normal_hint[2]
            )
            normal_raw = (
                normal_hint[0] - projection * tangent[0],
                normal_hint[1] - projection * tangent[1],
                normal_hint[2] - projection * tangent[2],
            )
            try:
                normal = _normalize_vec3(normal_raw)
            except WindingRunError:
                normal = _pick_default_normal(tangent)
        else:
            normal = _pick_default_normal(tangent)

        side = (
            tangent[1] * normal[2] - tangent[2] * normal[1],
            tangent[2] * normal[0] - tangent[0] * normal[2],
            tangent[0] * normal[1] - tangent[1] * normal[0],
        )
        side = _normalize_vec3(side)
        output_rows.append(
            {
                "index": idx,
                "arc_length": arc_length,
                "parameter_s": arc_length,
                "x": current[0],
                "y": current[1],
                "z": current[2],
                "tx": tangent[0],
                "ty": tangent[1],
                "tz": tangent[2],
                "nx": normal[0],
                "ny": normal[1],
                "nz": normal[2],
                "sx": side[0],
                "sy": side[1],
                "sz": side[2],
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "index",
                "arc_length",
                "parameter_s",
                "x",
                "y",
                "z",
                "tx",
                "ty",
                "tz",
                "nx",
                "ny",
                "nz",
                "sx",
                "sy",
                "sz",
            ),
        )
        writer.writeheader()
        for item in output_rows:
            writer.writerow(item)


class WindingRunError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        detail: Any | None = None,
        suggestion: str | None = None,
        status_code: int = 400,
    ) -> None:
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)
        self.detail = detail
        self.suggestion = suggestion
        self.status_code = int(status_code)


@dataclass(frozen=True)
class RunPlan:
    run_id: str
    run_mode: str
    online_role: str | None
    single_action: str | None
    command: tuple[str, ...]
    cwd: Path
    run_dir: Path
    request_path: Path | None
    summary_path: Path | None
    results_path: Path | None
    handoff_path: Path | None
    stdout_log_path: Path
    stderr_log_path: Path
    timeout_sec: int
    dry_run: bool
    build_request_before_launch: bool
    build_request_round_index: int
    build_request_candidate_limit: int
    build_request_refresh_csv: bool
    build_request_settings: dict[str, Any]


@dataclass
class RunState:
    plan: RunPlan
    payload: dict[str, Any]
    request_id: str
    created_at: str
    status: str = "queued"
    stage: str = "queued"
    progress: float = 0.0
    started_at: str | None = None
    finished_at: str | None = None
    pid: int | None = None
    exit_code: int | None = None
    semantic_status: str | None = None
    cancel_requested: bool = False
    error_code: str | None = None
    error_detail: str | None = None
    error_suggestion: str | None = None
    worker_thread: threading.Thread | None = None
    process: subprocess.Popen[Any] | None = None


def _runtime_options(payload: dict[str, Any]) -> dict[str, Any]:
    options = payload.get("options")
    if not isinstance(options, dict):
        return {}
    return dict(options)


def _program_name(payload: dict[str, Any]) -> str | None:
    process_params = payload.get("process_params")
    if isinstance(process_params, dict):
        value = process_params.get("program_name")
        if isinstance(value, str) and value.strip():
            return value.strip()
    options = _runtime_options(payload)
    value = options.get("program_name")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _build_plan(payload: dict[str, Any]) -> RunPlan:
    run_id = _safe_run_id(payload.get("run_id"))
    mode = str(payload.get("run_mode", "online")).strip().lower()
    if mode not in {"online", "single", "origin_search"}:
        raise WindingRunError(
            "invalid_request",
            "run_mode must be one of: online, single, origin_search.",
            detail={"run_mode": mode},
            suggestion="Use run_mode=online for workbench integration.",
            status_code=400,
        )

    options = _runtime_options(payload)
    timeout_sec = _as_int(options.get("timeout_sec"), default=0, minimum=0)
    dry_run = _as_bool(options.get("dry_run"), False)
    allow_invalid_outputs = _as_bool(options.get("allow_invalid_outputs"), False)

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    stdout_log_path = STATE_DIR / f"{run_id}.out.log"
    stderr_log_path = STATE_DIR / f"{run_id}.err.log"

    build_request_before_launch = False
    request_path: Path | None = None
    build_request_settings: dict[str, Any] = {}
    summary_path: Path | None = None
    results_path: Path | None = None
    handoff_path: Path | None = None
    run_dir: Path
    command: list[str]
    online_role: str | None = None
    single_action: str | None = None

    if mode == "online":
        defaults = _entrypoint_defaults()
        run_dir = ONLINE_RUNS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        online_role = str(payload.get("online_role", "coordinator")).strip().lower()
        if online_role not in {"coordinator", "server", "receiver"}:
            raise WindingRunError(
                "invalid_request",
                "online_role must be one of: coordinator, server, receiver.",
                detail={"online_role": online_role},
                suggestion="Use coordinator unless you specifically need server/receiver only.",
                status_code=400,
            )

        if online_role == "receiver":
            handoff_path = _candidate_handoff_path(payload)
            if handoff_path is None:
                raise WindingRunError(
                    "invalid_request",
                    "receiver mode requires handoff_path.",
                    detail={"run_mode": mode, "online_role": online_role},
                    suggestion="Provide request_source.handoff_path or handoff_path.",
                    status_code=400,
                )
            handoff_path = _resolve_user_path(handoff_path)
            if not handoff_path.is_file():
                raise WindingRunError(
                    "handoff_not_found",
                    "handoff file was not found.",
                    detail={"handoff_path": str(handoff_path)},
                    suggestion="Run server/coordinator first and pass the generated handoff path.",
                    status_code=404,
                )
            command = [
                sys.executable,
                "online_roundtrip.py",
                "run-receiver",
                "--handoff",
                str(handoff_path),
                "--run-id",
                run_id,
            ]
            local_python = options.get("local_python")
            if isinstance(local_python, str) and local_python.strip():
                command.extend(["--local-python", local_python.strip()])
            if allow_invalid_outputs:
                command.append("--allow-invalid-handoff")
            results_path = run_dir / "final_generate_result.json"
        else:
            request_path = _candidate_request_path(payload)
            if request_path is None:
                request_path = run_dir / "request.json"
                build_request_before_launch = True
            else:
                request_path = _resolve_user_path(request_path)
            if not build_request_before_launch and not request_path.is_file():
                raise WindingRunError(
                    "request_not_found",
                    "request_path was not found.",
                    detail={"request_path": str(request_path)},
                    suggestion="Provide an existing request JSON or omit request_path to auto-build.",
                    status_code=404,
                )

            process_params = payload.get("process_params")
            target_origin_mm = None
            target_rotation_xyz_deg = None
            if isinstance(process_params, dict):
                if process_params.get("target_frame_origin_mm") is not None:
                    target_origin_mm = _triplet(
                        process_params.get("target_frame_origin_mm"),
                        field_name="process_params.target_frame_origin_mm",
                    )
                if process_params.get("target_frame_rotation_xyz_deg") is not None:
                    target_rotation_xyz_deg = _triplet(
                        process_params.get("target_frame_rotation_xyz_deg"),
                        field_name="process_params.target_frame_rotation_xyz_deg",
                    )
            build_request_settings = {
                "centerline_path": None,
                "target_origin_mm": target_origin_mm,
                "target_rotation_xyz_deg": target_rotation_xyz_deg,
                "program_name": _program_name(payload),
                "selection": payload.get("selection", {}),
                "process_params": process_params if isinstance(process_params, dict) else {},
                "round_index": _as_int(options.get("round_index"), default=1, minimum=1),
                "candidate_limit": _as_int(options.get("candidate_limit"), default=4, minimum=1),
                "refresh_csv": not _as_bool(options.get("skip_pose_solver"), False),
            }
            build_request_settings["centerline_path"] = str(
                _resolve_centerline_path(payload=payload, run_dir=run_dir)
            )

            if online_role == "coordinator":
                remote_sync_mode = str(
                    options.get("remote_sync_mode", defaults.remote_sync_mode)
                ).strip().lower()
                if remote_sync_mode not in {"off", "guard", "push"}:
                    raise WindingRunError(
                        "invalid_request",
                        "options.remote_sync_mode must be off/guard/push.",
                        detail={"remote_sync_mode": remote_sync_mode},
                        suggestion="Use guard for safe default behavior.",
                        status_code=400,
                    )
                command = [
                    sys.executable,
                    "online_roundtrip.py",
                    "run-online",
                    "--host",
                    str(options.get("host", defaults.online_host)),
                    "--server-dir",
                    str(options.get("server_dir", defaults.online_server_dir)),
                    "--env",
                    str(options.get("env_name", defaults.online_env_name)),
                    "--request",
                    str(request_path),
                    "--run-id",
                    run_id,
                    "--remote-sync-mode",
                    remote_sync_mode,
                ]
                if _as_bool(options.get("disable_sync_guard"), False):
                    command.append("--disable-sync-guard")
                if _as_bool(options.get("skip_final_generate"), False):
                    command.append("--skip-final-generate")
                local_python = options.get("local_python")
                if isinstance(local_python, str) and local_python.strip():
                    command.extend(["--local-python", local_python.strip()])
            else:
                command = [
                    sys.executable,
                    "online_roundtrip.py",
                    "run-server",
                    "--request",
                    str(request_path),
                    "--run-id",
                    run_id,
                ]

            program_name = build_request_settings.get("program_name")
            if isinstance(program_name, str) and program_name.strip():
                command.extend(["--program-name", program_name])
            optimized_csv_path = run_dir / "tool_poses_frame2.csv"
            command.extend(["--optimized-csv-path", str(optimized_csv_path)])
            retry_candidate_limit = _as_int(options.get("retry_candidate_limit"), default=4, minimum=0)
            retry_repair_limit = _as_int(options.get("retry_repair_limit"), default=2, minimum=0)
            retry_max_rounds = _as_int(options.get("retry_max_rounds"), default=1, minimum=0)
            command.extend(["--retry-candidate-limit", str(retry_candidate_limit)])
            command.extend(["--retry-repair-limit", str(retry_repair_limit)])
            command.extend(["--retry-max-rounds", str(retry_max_rounds)])
            if allow_invalid_outputs:
                command.append("--allow-invalid-outputs")

            summary_path = run_dir / "summary.json"
            results_path = run_dir / "results.json"
            handoff_path = run_dir / "handoff_package.json"

    elif mode == "single":
        if payload.get("centerline_source") is not None:
            raise WindingRunError(
                "unsupported_for_single_mode",
                "single mode currently uses project-local centerline configuration only.",
                detail={"run_mode": mode},
                suggestion="Use run_mode=online when passing centerline_source.",
                status_code=400,
            )
        run_dir = SINGLE_RUNS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        single_action = str(payload.get("single_action", "program")).strip().lower()
        if single_action not in {"solve", "program", "visualize"}:
            raise WindingRunError(
                "invalid_request",
                "single_action must be solve/program/visualize.",
                detail={"single_action": single_action},
                suggestion="Use single_action=program unless you only need solve output.",
                status_code=400,
            )
        command = [
            sys.executable,
            "main.py",
            "--mode",
            "single",
            "--single-action",
            single_action,
        ]
        if single_action != "visualize":
            command.extend(["--run-id", run_id])
        if allow_invalid_outputs:
            command.append("--allow-invalid-outputs")
        results_path = run_dir / "eval_result.json"
    else:
        run_dir = ONLINE_RUNS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "main.py",
            "--mode",
            "origin_search",
            "--run-id",
            run_id,
        ]
        if _as_bool(options.get("allow_invalid_outputs"), False):
            command.append("--allow-invalid-outputs")
        results_path = run_dir / "origin_yz_search_results.json"

    return RunPlan(
        run_id=run_id,
        run_mode=mode,
        online_role=online_role,
        single_action=single_action,
        command=tuple(command),
        cwd=REPO_ROOT,
        run_dir=run_dir,
        request_path=request_path,
        summary_path=summary_path,
        results_path=results_path,
        handoff_path=handoff_path,
        stdout_log_path=stdout_log_path,
        stderr_log_path=stderr_log_path,
        timeout_sec=timeout_sec,
        dry_run=dry_run,
        build_request_before_launch=build_request_before_launch,
        build_request_round_index=_as_int(
            build_request_settings.get("round_index"),
            default=1,
            minimum=1,
        ),
        build_request_candidate_limit=_as_int(
            build_request_settings.get("candidate_limit"),
            default=4,
            minimum=1,
        ),
        build_request_refresh_csv=bool(build_request_settings.get("refresh_csv", True)),
        build_request_settings=build_request_settings,
    )


def _artifact_manifest(plan: RunPlan) -> dict[str, str | None]:
    return {
        "run_dir": str(plan.run_dir),
        "request": None if plan.request_path is None else str(plan.request_path),
        "summary": None if plan.summary_path is None else str(plan.summary_path),
        "results": None if plan.results_path is None else str(plan.results_path),
        "handoff_package": None if plan.handoff_path is None else str(plan.handoff_path),
        "stdout_log": str(plan.stdout_log_path),
        "stderr_log": str(plan.stderr_log_path),
    }


class WindingRunManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: dict[str, RunState] = {}

    def create_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise WindingRunError(
                "invalid_request",
                "Request body must be a JSON object.",
                detail={"payload_type": type(payload).__name__},
                suggestion="Send an object payload.",
                status_code=400,
            )
        plan = _build_plan(payload)
        request_id = _request_id()
        state = RunState(
            plan=plan,
            payload=dict(payload),
            request_id=request_id,
            created_at=_iso_utc(),
        )
        with self._lock:
            if plan.run_id in self._runs:
                raise WindingRunError(
                    "run_id_conflict",
                    "run_id already exists.",
                    detail={"run_id": plan.run_id},
                    suggestion="Use a new run_id or omit run_id for auto-generation.",
                    status_code=409,
                )
            self._runs[plan.run_id] = state

        if plan.dry_run:
            with self._lock:
                state.status = "succeeded"
                state.stage = "dry_run"
                state.progress = 1.0
                state.started_at = _iso_utc()
                state.finished_at = _iso_utc()
                state.semantic_status = "dry_run"
            self._persist_state(plan.run_id)
            return self.get_run(plan.run_id)

        worker = threading.Thread(
            target=self._worker_main,
            args=(plan.run_id,),
            name=f"winding-run-{plan.run_id}",
            daemon=True,
        )
        with self._lock:
            state.worker_thread = worker
            state.status = "queued"
            state.stage = "queued"
            state.progress = 0.0
        worker.start()
        self._persist_state(plan.run_id)
        return self.get_run(plan.run_id)

    def list_runs(self, *, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            states = list(self._runs.values())
        states.sort(key=lambda item: item.created_at, reverse=True)
        return [self._public_state(item) for item in states[: max(1, int(limit))]]

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            state = self._runs.get(str(run_id))
        if state is not None:
            return self._public_state(state)
        persisted = self._load_persisted_state(str(run_id))
        if persisted is not None:
            return persisted
        synthesized = self._synthesize_existing_run(str(run_id))
        if synthesized is not None:
            return synthesized
        raise WindingRunError(
            "run_not_found",
            "Run id was not found.",
            detail={"run_id": str(run_id)},
            suggestion="Check run_id or create a new run first.",
            status_code=404,
        )

    def cancel_run(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            state = self._runs.get(str(run_id))
            if state is None:
                raise WindingRunError(
                    "run_not_found",
                    "Run id was not found.",
                    detail={"run_id": str(run_id)},
                    suggestion="Check run_id or create a new run first.",
                    status_code=404,
                )
            if state.status in {"succeeded", "failed", "canceled"}:
                return self._public_state(state)
            state.cancel_requested = True
            state.stage = "cancel_requested"
            state.error_code = "cancel_requested"
            process = state.process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except Exception:
                pass
        self._persist_state(str(run_id))
        return self.get_run(str(run_id))

    def load_artifact(self, run_id: str, artifact_name: str) -> dict[str, Any]:
        run_data = self.get_run(run_id)
        artifacts = run_data.get("artifacts", {})
        if not isinstance(artifacts, dict):
            artifacts = {}
        candidate_path = artifacts.get(artifact_name)
        if not isinstance(candidate_path, str) or not candidate_path:
            default = self._default_artifact_path(run_id, artifact_name)
            candidate_path = None if default is None else str(default)
        if not candidate_path:
            raise WindingRunError(
                "artifact_not_found",
                f"Artifact '{artifact_name}' is not available for this run.",
                detail={"run_id": run_id, "artifact": artifact_name},
                suggestion="Ensure the run mode produces this artifact.",
                status_code=404,
            )
        path = Path(candidate_path)
        if not path.is_file():
            raise WindingRunError(
                "artifact_not_found",
                f"Artifact '{artifact_name}' file is missing.",
                detail={"run_id": run_id, "artifact": artifact_name, "path": str(path)},
                suggestion="Wait for run completion or inspect stderr log.",
                status_code=404,
            )
        return {
            "run_id": run_id,
            "artifact": artifact_name,
            "path": str(path),
            "content": load_json_file(path),
        }

    def latest_summary(self) -> dict[str, Any]:
        ONLINE_RUNS_DIR.mkdir(parents=True, exist_ok=True)
        candidates: list[tuple[float, Path, str]] = []
        for run_dir in ONLINE_RUNS_DIR.iterdir():
            if not run_dir.is_dir():
                continue
            summary_path = run_dir / "summary.json"
            if not summary_path.is_file():
                continue
            candidates.append((summary_path.stat().st_mtime, summary_path, run_dir.name))
        if not candidates:
            raise WindingRunError(
                "latest_summary_not_found",
                "No summary.json was found under artifacts/online_runs.",
                detail={"root": str(ONLINE_RUNS_DIR)},
                suggestion="Create an online run first.",
                status_code=404,
            )
        candidates.sort(key=lambda item: item[0], reverse=True)
        _, summary_path, run_id = candidates[0]
        content = load_json_file(summary_path)
        return {
            "run_id": run_id,
            "summary_path": str(summary_path),
            "summary": content,
        }

    def capabilities(self) -> dict[str, Any]:
        return {
            "api_version": "2026-04-22",
            "run_modes": ["online", "single", "origin_search"],
            "online_roles": ["coordinator", "server", "receiver"],
            "centerline_source": {
                "file_path": True,
                "inline_points": True,
            },
            "artifacts": {
                "summary": "artifacts/online_runs/<run_id>/summary.json",
                "results": "artifacts/online_runs/<run_id>/results.json",
                "handoff_package": "artifacts/online_runs/<run_id>/handoff_package.json",
            },
            "lifecycle_states": [
                "queued",
                "running",
                "succeeded",
                "failed",
                "canceled",
                "cancel_requested",
            ],
            "notes": [
                "run_mode=online supports centerline_source.file_path and inline_points.",
                "run_mode=single reuses existing project-local control panel configuration.",
                "return code 2 from solver process is treated as completed with semantic_status=invalid.",
            ],
        }

    def _default_artifact_path(self, run_id: str, artifact_name: str) -> Path | None:
        online_dir = ONLINE_RUNS_DIR / str(run_id)
        single_dir = SINGLE_RUNS_DIR / str(run_id)
        mapping = {
            "summary": online_dir / "summary.json",
            "results": online_dir / "results.json",
            "handoff_package": online_dir / "handoff_package.json",
            "final_results": online_dir / "final_generate_result.json",
            "eval_result": single_dir / "eval_result.json",
        }
        return mapping.get(str(artifact_name))

    def _public_state(self, state: RunState) -> dict[str, Any]:
        plan = state.plan
        return {
            "run_id": plan.run_id,
            "run_mode": plan.run_mode,
            "online_role": plan.online_role,
            "single_action": plan.single_action,
            "status": state.status,
            "stage": state.stage,
            "progress": float(state.progress),
            "request_id": state.request_id,
            "created_at": state.created_at,
            "started_at": state.started_at,
            "finished_at": state.finished_at,
            "pid": state.pid,
            "exit_code": state.exit_code,
            "semantic_status": state.semantic_status,
            "cancel_requested": bool(state.cancel_requested),
            "error": {
                "code": state.error_code,
                "detail": state.error_detail,
                "suggestion": state.error_suggestion,
            }
            if state.error_code is not None
            else None,
            "artifacts": _artifact_manifest(plan),
            "command": list(plan.command),
        }

    def _persist_state(self, run_id: str) -> None:
        try:
            payload = self.get_run(run_id)
            write_json_file(STATE_DIR / f"{run_id}.json", payload)
            run_dir = Path(payload["artifacts"]["run_dir"])
            if run_dir.exists():
                write_json_file(run_dir / "api_run_manifest.json", payload)
        except Exception:
            return

    def _load_persisted_state(self, run_id: str) -> dict[str, Any] | None:
        state_path = STATE_DIR / f"{run_id}.json"
        if state_path.is_file():
            return load_json_file(state_path)
        online_manifest = ONLINE_RUNS_DIR / run_id / "api_run_manifest.json"
        if online_manifest.is_file():
            return load_json_file(online_manifest)
        single_manifest = SINGLE_RUNS_DIR / run_id / "api_run_manifest.json"
        if single_manifest.is_file():
            return load_json_file(single_manifest)
        return None

    def _synthesize_existing_run(self, run_id: str) -> dict[str, Any] | None:
        online_dir = ONLINE_RUNS_DIR / run_id
        single_dir = SINGLE_RUNS_DIR / run_id
        if not online_dir.is_dir() and not single_dir.is_dir():
            return None

        if online_dir.is_dir():
            run_dir = online_dir
            run_mode = "online"
            summary_path = run_dir / "summary.json"
            results_path = run_dir / "results.json"
            handoff_path = run_dir / "handoff_package.json"
            request_path = run_dir / "request.json"
        else:
            run_dir = single_dir
            run_mode = "single"
            summary_path = None
            results_path = run_dir / "eval_result.json"
            handoff_path = None
            request_path = run_dir / "request.json"

        mtime = run_dir.stat().st_mtime
        finished_at = _iso_utc(datetime.fromtimestamp(mtime, tz=timezone.utc))
        semantic_status = "unknown"
        status = "failed"
        if results_path is not None and results_path.is_file():
            status = "succeeded"
            semantic_status = "valid"
        if summary_path is not None and summary_path.is_file():
            try:
                summary_payload = load_json_file(summary_path)
                selection = summary_payload.get("server_final_selection", {})
                allowed = bool(selection.get("official_delivery_allowed", False))
                semantic_status = "valid" if allowed else "invalid"
            except Exception:
                semantic_status = "unknown"
            status = "succeeded"

        return {
            "run_id": run_id,
            "run_mode": run_mode,
            "online_role": None,
            "single_action": None,
            "status": status,
            "stage": "historical",
            "progress": 1.0,
            "request_id": None,
            "created_at": finished_at,
            "started_at": None,
            "finished_at": finished_at,
            "pid": None,
            "exit_code": None,
            "semantic_status": semantic_status,
            "cancel_requested": False,
            "error": None,
            "artifacts": {
                "run_dir": str(run_dir),
                "request": str(request_path) if request_path.is_file() else None,
                "summary": str(summary_path) if summary_path is not None and summary_path.is_file() else None,
                "results": str(results_path) if results_path is not None and results_path.is_file() else None,
                "handoff_package": (
                    str(handoff_path) if handoff_path is not None and handoff_path.is_file() else None
                ),
                "stdout_log": None,
                "stderr_log": None,
            },
            "command": [],
        }

    def _build_request_for_run(self, state: RunState) -> None:
        plan = state.plan
        if not plan.build_request_before_launch:
            return
        if plan.request_path is None:
            raise WindingRunError(
                "request_build_failed",
                "Internal error: request_path is missing for request build.",
                detail={"run_id": plan.run_id},
                suggestion="Retry the run.",
                status_code=500,
            )

        defaults = _entrypoint_defaults()
        settings = defaults.app_runtime_settings
        cfg = dict(plan.build_request_settings)
        centerline_path = Path(str(cfg.get("centerline_path")))
        settings = replace(
            settings,
            validation_centerline_csv=centerline_path,
            tool_poses_frame2_csv=plan.run_dir / "tool_poses_frame2.csv",
        )
        program_name = cfg.get("program_name")
        if isinstance(program_name, str) and program_name.strip():
            settings = replace(settings, program_name=program_name.strip())
        target_origin_mm = cfg.get("target_origin_mm")
        if target_origin_mm is not None:
            settings = replace(settings, target_frame_origin_mm=tuple(target_origin_mm))
        target_rotation_xyz_deg = cfg.get("target_rotation_xyz_deg")
        if target_rotation_xyz_deg is not None:
            settings = replace(
                settings,
                target_frame_rotation_xyz_deg=tuple(target_rotation_xyz_deg),
            )

        metadata = {
            "entrypoint": "winding_runtime_api",
            "api_created_at": state.created_at,
            "run_id": plan.run_id,
            "selection": cfg.get("selection", {}),
            "process_params": cfg.get("process_params", {}),
        }
        build_request_file(
            settings,
            plan.request_path,
            round_index=int(plan.build_request_round_index),
            candidate_limit=int(plan.build_request_candidate_limit),
            refresh_csv=bool(plan.build_request_refresh_csv),
            metadata=metadata,
        )

    def _worker_main(self, run_id: str) -> None:
        with self._lock:
            state = self._runs.get(run_id)
            if state is None:
                return
            state.status = "running"
            state.stage = "preparing"
            state.progress = 0.05
            state.started_at = _iso_utc()
            plan = state.plan

        try:
            if plan.build_request_before_launch:
                with self._lock:
                    state = self._runs[run_id]
                    state.stage = "building_request"
                    state.progress = 0.12
                self._build_request_for_run(state)
                self._persist_state(run_id)

            with self._lock:
                state = self._runs[run_id]
                state.stage = "launching"
                state.progress = 0.2

            plan.stdout_log_path.parent.mkdir(parents=True, exist_ok=True)
            creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
            stdout_handle = plan.stdout_log_path.open("w", encoding="utf-8", errors="replace")
            stderr_handle = plan.stderr_log_path.open("w", encoding="utf-8", errors="replace")
            try:
                process = subprocess.Popen(
                    list(plan.command),
                    cwd=str(plan.cwd),
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    creationflags=creationflags,
                )
            finally:
                stdout_handle.close()
                stderr_handle.close()

            with self._lock:
                state = self._runs[run_id]
                state.process = process
                state.pid = int(process.pid)
                state.stage = "running_solver"
                state.progress = 0.35

            started_monotonic = time.monotonic()
            timed_out = False
            termination_requested_at: float | None = None
            while True:
                return_code = process.poll()
                if return_code is not None:
                    break
                with self._lock:
                    state = self._runs[run_id]
                    cancel_requested = bool(state.cancel_requested)
                    timeout_sec = int(plan.timeout_sec)
                if cancel_requested:
                    try:
                        process.terminate()
                    except Exception:
                        pass
                    if termination_requested_at is None:
                        termination_requested_at = time.monotonic()
                if timeout_sec > 0 and time.monotonic() - started_monotonic > timeout_sec:
                    timed_out = True
                    try:
                        process.terminate()
                    except Exception:
                        pass
                    if termination_requested_at is None:
                        termination_requested_at = time.monotonic()
                if (
                    termination_requested_at is not None
                    and time.monotonic() - termination_requested_at > 5.0
                ):
                    try:
                        process.kill()
                    except Exception:
                        pass
                time.sleep(0.4)

            return_code = int(process.returncode or 0)
            with self._lock:
                state = self._runs[run_id]
                state.process = None
                state.exit_code = return_code
                state.finished_at = _iso_utc()
                state.progress = 1.0
                if timed_out:
                    state.status = "failed"
                    state.stage = "timeout"
                    state.semantic_status = "failed"
                    state.error_code = "run_timeout"
                    state.error_detail = (
                        f"Run exceeded timeout_sec={plan.timeout_sec}."
                    )
                    state.error_suggestion = "Increase timeout_sec or simplify the run."
                elif state.cancel_requested:
                    state.status = "canceled"
                    state.stage = "canceled"
                    state.semantic_status = "canceled"
                elif return_code in {0, 2}:
                    state.status = "succeeded"
                    state.stage = "completed"
                    state.semantic_status = "valid" if return_code == 0 else "invalid"
                else:
                    state.status = "failed"
                    state.stage = "failed"
                    state.semantic_status = "failed"
                    state.error_code = "run_process_failed"
                    state.error_detail = f"Process exited with code {return_code}."
                    state.error_suggestion = (
                        "Inspect stdout/stderr logs and run artifacts for details."
                    )
            self._persist_state(run_id)
        except WindingRunError as exc:
            with self._lock:
                state = self._runs[run_id]
                state.status = "failed"
                state.stage = "failed"
                state.progress = 1.0
                state.finished_at = _iso_utc()
                state.semantic_status = "failed"
                state.error_code = exc.code
                state.error_detail = exc.message
                state.error_suggestion = exc.suggestion
            self._persist_state(run_id)
        except Exception as exc:
            with self._lock:
                state = self._runs[run_id]
                state.status = "failed"
                state.stage = "failed"
                state.progress = 1.0
                state.finished_at = _iso_utc()
                state.semantic_status = "failed"
                state.error_code = "run_internal_error"
                state.error_detail = f"{type(exc).__name__}: {exc}"
                state.error_suggestion = "Retry the run and inspect stderr log."
            self._persist_state(run_id)


DEFAULT_WINDING_RUN_MANAGER = WindingRunManager()
