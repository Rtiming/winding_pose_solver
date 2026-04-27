from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from time import perf_counter

from app_settings import build_app_runtime_settings
from src.core.collab_models import (
    EvaluationBatchRequest,
    EvaluationBatchResult,
    load_json_file,
    write_json_file,
)
from src.runtime.app import run_visualization
from src.runtime.delivery import (
    STRICT_DELIVERY_GATE,
    result_has_continuity_warnings as _result_has_continuity_warnings,
    result_is_strictly_valid as _result_meets_delivery_gate,
    result_payload_is_strictly_valid as _result_payload_meets_delivery_gate,
    result_semantic_status as _result_semantic_status,
    summary_metrics_delivery_gate_details,
)
from src.runtime.run_logging import (
    RunDescriptor,
    build_run_log_path,
    emit_lines,
    print_result_paths,
    render_run_footer_lines,
    render_run_header_lines,
    tee_console_to_log,
    timestamp_token,
)
from src.runtime.origin_search_runner import (
    OriginSearchServerSettings,
    OriginSearchSettings,
    origin_search_run_dir,
    run_origin_search,
)
from src.runtime.local_profile import (
    LOCAL_PROFILE_CHOICES,
    local_profile_metadata,
    local_profile_status_text,
    resolve_local_profile,
)


# ---------------------------------------------------------------------------
# Main business parameters
# Keep the most frequently changed project inputs here.
# Detailed optimizer and RoboDK tuning lives in `app_settings.py`.
# ---------------------------------------------------------------------------

VALIDATION_CENTERLINE_CSV = Path("data/validation_centerline.csv")
TOOL_POSES_FRAME2_CSV = Path("data/tool_poses_frame2.csv")
APPEND_CENTERLINE_START_AS_TERMINAL = True

# Fallback default kept aligned with root main.py.  Current value is the
# verified clean origin from smart-square + online/RoboDK smoke.
TARGET_FRAME_A_ORIGIN_IN_FRAME2_MM = (1126.0, -647.5, 1477.5)
TARGET_FRAME_A_ROTATION_IN_FRAME2_XYZ_DEG = (0, 0, -180.0)
ENABLE_CUSTOM_SMOOTHING_AND_POSE_SELECTION = True
ROBOT_NAME = "KUKA"
FRAME_NAME = "Frame 2"
PROGRAM_NAME = "Path_From_CSV"
ENABLE_LOCAL_MULTIPROCESS_PARALLEL = True
LOCAL_PARALLEL_WORKERS = 0  # 0 = auto
LOCAL_PARALLEL_MIN_BATCH_SIZE = 4
SINGLE_RUN_WINDOW_REPAIR = False
SINGLE_RUN_INSERTED_REPAIR = False
ENABLE_FIXED_POINT_PATH_FALLBACK = True
FIXED_POINT_PATH_FALLBACK_MAX_CANDIDATES_PER_CONFIG = 12
FIXED_POINT_PATH_FALLBACK_DISABLE_GUIDED_PATH = True
FIXED_POINT_PATH_FALLBACK_RUN_WINDOW_REPAIR = True
FIXED_POINT_PATH_FALLBACK_RUN_INSERTED_REPAIR = False
LOCK_FRAME_A_ORIGIN_YZ_PROFILE_ENDPOINTS = True
ENABLE_LOCAL_PROFILE_RETRY = False
LOCAL_PROFILE_RETRY_CANDIDATE_LIMIT = 4
LOCAL_PROFILE_RETRY_REPAIR_LIMIT = 2
LOCAL_PROFILE_RETRY_MAX_ROUNDS = 1

# IK 后端选择：
#   "robodk"      — 使用 RoboDK 内置 IK 求解器（原有行为）
#   "six_axis_ik" — 使用内置本地 POE 模型求解器（不需要 RoboDK 授权许可，多解枚举更全面）
IK_BACKEND = "six_axis_ik"


# ---------------------------------------------------------------------------
# Run mode configuration
# Edit these few variables, then run `python main.py`.
# The script will choose the correct local / online flow automatically.
# ---------------------------------------------------------------------------

RUN_MODE = "online"  # "single" | "online" | "origin_search"
LOCAL_MACHINE_PROFILE = "auto"  # "auto" | "mac" | "windows" | "linux"
SINGLE_ACTION = "program"  # "solve" | "program" | "visualize"
SINGLE_RESULT_ARCHIVE_DIR = Path("artifacts/local_runs")

ONLINE_ROLE = "coordinator"  # "coordinator" | "server" | "receiver"
# Compatibility alias for older scripts. "roundtrip" maps to "coordinator".
ONLINE_ACTION = "coordinator"
ONLINE_REQUEST_SOURCE = "build"  # "build" | "existing"
ONLINE_REQUEST_PATH = Path("artifacts/online_runs/main_request.json")
ONLINE_HANDOFF_PACKAGE_PATH = Path("artifacts/online_runs/main_handoff_package.json")
ONLINE_RUN_ID: str | None = None
ONLINE_HOST = "master"
ONLINE_SERVER_DIR = "/home/tzwang/program/winding_pose_solver"
ONLINE_ENV_NAME = "winding_pose_solver"
ONLINE_LOCAL_PYTHON: str | None = None
ONLINE_ROUND_INDEX = 1
ONLINE_CANDIDATE_LIMIT = 4
ONLINE_SKIP_POSE_SOLVER_WHEN_BUILDING_REQUEST = False
ONLINE_AUTO_SETUP_SERVER = False
ONLINE_SETUP_SERVER_ENABLE_SLURM = False
ONLINE_SERVER_EVAL_WHEN_POSSIBLE = False
ONLINE_FINAL_GENERATE_PROGRAM = True
# Server-side online continuity retry budget. This keeps heavy repair on master
# while the local coordinator only handles transfer and the final RoboDK receiver import.
#   WPS_ONLINE_RETRY_CANDIDATE_LIMIT / WPS_ONLINE_RETRY_REPAIR_LIMIT / WPS_ONLINE_RETRY_MAX_ROUNDS
ONLINE_PROFILE_RETRY_CANDIDATE_LIMIT = 4
ONLINE_PROFILE_RETRY_REPAIR_LIMIT = 2
ONLINE_PROFILE_RETRY_MAX_ROUNDS = 1
ENFORCE_REMOTE_SYNC_GUARD = True
REMOTE_SYNC_MODE = "guard"  # "off" | "guard" | "push"
ALLOW_INVALID_DEBUG_OUTPUTS = False

BIG_CIRCLE_STEP_DEG_THRESHOLD = 170.0
BRANCH_FLIP_RATIO_THRESHOLD = 8.0
DEBUG_FALLBACK_IMPORT_COUNT = 0

# Optional TARGET_FRAME_A_ORIGIN_IN_FRAME2_MM Y/Z search.
# Set ENABLE_TARGET_ORIGIN_YZ_SEARCH=True or run `python main.py --mode origin_search`.
# The default search box is a 600 x 600 mm square centered on the configured
# TARGET_FRAME_A_ORIGIN_IN_FRAME2_MM.  The smart-square algorithm evaluates a
# small beam of promising neighborhoods instead of exhaustive gridding.
ENABLE_TARGET_ORIGIN_YZ_SEARCH = False
TARGET_ORIGIN_YZ_SEARCH_USE_SERVER = True
TARGET_ORIGIN_YZ_SEARCH_SQUARE_SIZE_MM = 600.0
TARGET_ORIGIN_YZ_SEARCH_INITIAL_STEP_MM = 150.0
TARGET_ORIGIN_YZ_SEARCH_MIN_STEP_MM = 10.0
TARGET_ORIGIN_YZ_SEARCH_MAX_ITERS = 5
TARGET_ORIGIN_YZ_SEARCH_BEAM_WIDTH = 4
TARGET_ORIGIN_YZ_SEARCH_DIAGONAL_POLICY = "conditional"  # conditional, always, or never
TARGET_ORIGIN_YZ_SEARCH_POLISH_STEP_MM = 5.0
TARGET_ORIGIN_YZ_SEARCH_WORKERS = 16
TARGET_ORIGIN_YZ_SEARCH_STRATEGY = "full_search"
TARGET_ORIGIN_YZ_SEARCH_RUN_WINDOW_REPAIR = False
TARGET_ORIGIN_YZ_SEARCH_RUN_INSERTED_REPAIR = False
TARGET_ORIGIN_YZ_SEARCH_VALIDATION_GRID_STEP_MM = 0.0
TARGET_ORIGIN_YZ_SEARCH_TOP_K = 12
TARGET_ORIGIN_YZ_SEARCH_MIN_SEPARATION_MM = 20.0
TARGET_ORIGIN_YZ_SEARCH_USABLE_COUNT = 5
TARGET_ORIGIN_YZ_OUTSIDE_FALLBACK_COUNT = 3
TARGET_ORIGIN_YZ_OUTSIDE_FALLBACK_MAX_RINGS = 6
TARGET_ORIGIN_YZ_OUTSIDE_FALLBACK_RING_STEP_MM = 25.0
TARGET_ORIGIN_YZ_OUTSIDE_FALLBACK_EDGE_STEP_MM = 75.0
# Post-search execution policy:
#   "none"          : only report usable origins
#   "single_action" : run local flow with SINGLE_ACTION on best N usable origins
#   "online_role"   : run online flow with ONLINE_ROLE on best N usable origins
TARGET_ORIGIN_YZ_SEARCH_POST_DISPATCH = "online_role"
TARGET_ORIGIN_YZ_SEARCH_POST_TOP_N = 1

WRITE_DETAILED_LOG_FILE = True
SHOW_DETAILED_TERMINAL_LOGS = True
SHOW_COMMAND_DETAILS = False
RUN_LOG_DIR = Path("artifacts/run_logs")
STRICT_EXIT_ON_INVALID = False


def _build_runtime_settings(
    *,
    target_frame_origin_mm: tuple[float, float, float] | None = None,
    tool_poses_frame2_csv: Path = TOOL_POSES_FRAME2_CSV,
):
    origin_mm = (
        tuple(float(value) for value in TARGET_FRAME_A_ORIGIN_IN_FRAME2_MM)
        if target_frame_origin_mm is None
        else tuple(float(value) for value in target_frame_origin_mm)
    )
    runtime_settings = build_app_runtime_settings(
        validation_centerline_csv=VALIDATION_CENTERLINE_CSV,
        tool_poses_frame2_csv=tool_poses_frame2_csv,
        append_start_as_terminal=APPEND_CENTERLINE_START_AS_TERMINAL,
        target_frame_origin_mm=origin_mm,
        target_frame_rotation_xyz_deg=TARGET_FRAME_A_ROTATION_IN_FRAME2_XYZ_DEG,
        enable_custom_smoothing_and_pose_selection=ENABLE_CUSTOM_SMOOTHING_AND_POSE_SELECTION,
        robot_name=ROBOT_NAME,
        frame_name=FRAME_NAME,
        program_name=PROGRAM_NAME,
        ik_backend=IK_BACKEND,
        local_parallel_workers=(
            LOCAL_PARALLEL_WORKERS if ENABLE_LOCAL_MULTIPROCESS_PARALLEL else 1
        ),
        local_parallel_min_batch_size=LOCAL_PARALLEL_MIN_BATCH_SIZE,
    )
    tuned_motion_settings = replace(
        runtime_settings.motion_settings,
        big_circle_step_deg_threshold=float(BIG_CIRCLE_STEP_DEG_THRESHOLD),
        branch_flip_ratio_threshold=float(BRANCH_FLIP_RATIO_THRESHOLD),
        lock_frame_a_origin_yz_profile_endpoints=bool(
            LOCK_FRAME_A_ORIGIN_YZ_PROFILE_ENDPOINTS
        ),
    )
    return replace(runtime_settings, motion_settings=tuned_motion_settings)


APP_RUNTIME_SETTINGS = _build_runtime_settings()


def refresh_runtime_settings() -> None:
    """Rebuild APP_RUNTIME_SETTINGS after top-level constants change."""
    global APP_RUNTIME_SETTINGS
    APP_RUNTIME_SETTINGS = _build_runtime_settings()


def apply_overrides(overrides: dict[str, object]) -> None:
    """
    Apply top-level runtime overrides from the thin root `main.py` control panel.

    The root entrypoint stays small and user-editable while this module remains
    the canonical implementation surface.
    """
    for name, value in overrides.items():
        if name in globals():
            globals()[name] = value
    refresh_runtime_settings()


def _read_positive_int_env(name: str, default_value: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return int(default_value)
    try:
        parsed_value = int(raw_value)
    except ValueError:
        print(f"[config] Ignoring invalid {name}={raw_value!r}; expected a positive integer.")
        return int(default_value)
    if parsed_value <= 0:
        print(f"[config] Ignoring non-positive {name}={raw_value!r}; expected a positive integer.")
        return int(default_value)
    return parsed_value


def _read_nonnegative_int_env(name: str, default_value: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return int(default_value)
    try:
        parsed_value = int(raw_value)
    except ValueError:
        print(f"[config] Ignoring invalid {name}={raw_value!r}; expected a non-negative integer.")
        return int(default_value)
    if parsed_value < 0:
        print(f"[config] Ignoring negative {name}={raw_value!r}; expected a non-negative integer.")
        return int(default_value)
    return parsed_value


def _read_first_nonnegative_int_env(names: tuple[str, ...], default_value: int) -> int:
    for name in names:
        if os.getenv(name) is not None:
            return _read_nonnegative_int_env(name, default_value)
    return int(default_value)


def _resolve_local_profile_retry_budget() -> tuple[int, int, int]:
    return (
        _read_nonnegative_int_env(
            "WPS_RETRY_CANDIDATE_LIMIT",
            LOCAL_PROFILE_RETRY_CANDIDATE_LIMIT,
        ),
        _read_nonnegative_int_env(
            "WPS_RETRY_REPAIR_LIMIT",
            LOCAL_PROFILE_RETRY_REPAIR_LIMIT,
        ),
        _read_nonnegative_int_env(
            "WPS_RETRY_MAX_ROUNDS",
            LOCAL_PROFILE_RETRY_MAX_ROUNDS,
        ),
    )


def _resolve_online_profile_retry_budget() -> tuple[int, int, int]:
    return (
        _read_first_nonnegative_int_env(
            ("WPS_ONLINE_RETRY_CANDIDATE_LIMIT", "WPS_RETRY_CANDIDATE_LIMIT"),
            ONLINE_PROFILE_RETRY_CANDIDATE_LIMIT,
        ),
        _read_first_nonnegative_int_env(
            ("WPS_ONLINE_RETRY_REPAIR_LIMIT", "WPS_RETRY_REPAIR_LIMIT"),
            ONLINE_PROFILE_RETRY_REPAIR_LIMIT,
        ),
        _read_first_nonnegative_int_env(
            ("WPS_ONLINE_RETRY_MAX_ROUNDS", "WPS_RETRY_MAX_ROUNDS"),
            ONLINE_PROFILE_RETRY_MAX_ROUNDS,
        ),
    )


def _resolve_run_id(args: argparse.Namespace | None = None) -> str:
    if args is not None and args.run_id:
        return args.run_id
    if ONLINE_RUN_ID:
        return ONLINE_RUN_ID
    return timestamp_token()


def _resolve_local_machine_profile(args: argparse.Namespace | None = None) -> str:
    configured = (
        str(args.local_profile)
        if args is not None and getattr(args, "local_profile", None)
        else LOCAL_MACHINE_PROFILE
    )
    return resolve_local_profile(configured)


def _local_machine_profile_metadata(
    args: argparse.Namespace | None = None,
) -> dict[str, object]:
    configured = (
        str(args.local_profile)
        if args is not None and getattr(args, "local_profile", None)
        else LOCAL_MACHINE_PROFILE
    )
    return local_profile_metadata(configured)


def _resolve_single_run_id(args: argparse.Namespace | None = None) -> str:
    if args is not None and args.run_id:
        return args.run_id
    return timestamp_token()


def _single_run_tool_pose_csv(run_id: str) -> Path:
    return SINGLE_RESULT_ARCHIVE_DIR / run_id / "tool_poses_frame2.csv"


def _online_run_tool_pose_csv(run_id: str) -> Path:
    return Path("artifacts/online_runs") / run_id / "tool_poses_frame2.csv"


def _runtime_settings_with_tool_pose_csv(
    runtime_settings,
    *,
    tool_pose_csv: Path,
):
    tool_pose_csv.parent.mkdir(parents=True, exist_ok=True)
    return replace(runtime_settings, tool_poses_frame2_csv=Path(tool_pose_csv))


def _build_single_artifact_paths(
    run_id: str | None,
    *,
    runtime_settings=None,
) -> dict[str, Path | None]:
    active_settings = runtime_settings or APP_RUNTIME_SETTINGS
    if run_id is None:
        return {
            "run_dir": None,
            "request": None,
            "eval_result": None,
            "archive": None,
            "joint_path_csv": None,
            "debug_joint_path_csv": None,
            "profile_retry_summary": None,
            "tool_pose_csv": active_settings.tool_poses_frame2_csv,
            "optimized_pose_csv": active_settings.tool_poses_frame2_csv.with_name(
                f"{active_settings.tool_poses_frame2_csv.stem}_optimized.csv"
            ),
        }
    run_dir = SINGLE_RESULT_ARCHIVE_DIR / run_id
    return {
        "run_dir": run_dir,
        "request": run_dir / "request.json",
        "eval_result": run_dir / "eval_result.json",
        "archive": run_dir / "run_archive.json",
        "joint_path_csv": run_dir / "selected_joint_path.csv",
        "debug_joint_path_csv": run_dir / "debug_selected_joint_path.csv",
        "profile_retry_summary": run_dir / "profile_retry_summary.json",
        "tool_pose_csv": active_settings.tool_poses_frame2_csv,
        "optimized_pose_csv": active_settings.tool_poses_frame2_csv.with_name(
            f"{active_settings.tool_poses_frame2_csv.stem}_optimized.csv"
        ),
    }


def _summarize_worker_eval_result(result_path: Path) -> None:
    payload = load_json_file(result_path)
    result_list = payload.get("results", [])
    if not result_list:
        print("[worker-eval] No result entries found.")
        return
    result = result_list[0]
    print(
        "[worker-eval] "
        f"status={result.get('status')}, "
        f"ik_empty_rows={result.get('ik_empty_row_count')}, "
        f"config_switches={result.get('config_switches')}, "
        f"bridge_like_segments={result.get('bridge_like_segments')}, "
        f"big_circle_step_count={result.get('big_circle_step_count')}, "
        f"worst_joint_step={result.get('worst_joint_step_deg')}"
    )


def _resolve_online_request_path(args: argparse.Namespace | None = None) -> Path:
    if args is not None and args.request:
        return Path(args.request)
    return ONLINE_REQUEST_PATH


def _resolve_online_command(args: argparse.Namespace | None = None) -> str:
    if args is not None and getattr(args, "online_role", None):
        return str(args.online_role)

    configured_action = (
        str(args.online_action)
        if args is not None and getattr(args, "online_action", None)
        else ONLINE_ACTION
    )
    compatibility_map = {
        "roundtrip": "coordinator",
        "coordinator": "coordinator",
        "server": "server",
        "receiver": "receiver",
    }
    if configured_action in compatibility_map:
        return compatibility_map[configured_action]
    if configured_action:
        return configured_action
    return ONLINE_ROLE


def _normalize_remote_sync_mode(raw_value: str) -> str:
    normalized = str(raw_value).strip().lower()
    aliases = {
        "check": "guard",
        "verify": "guard",
        "sync": "push",
        "auto_push": "push",
        "none": "off",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"off", "guard", "push"}:
        raise ValueError(
            f"Unsupported REMOTE_SYNC_MODE={raw_value!r}; expected off/guard/push."
        )
    return normalized


def _resolve_remote_sync_mode(args: argparse.Namespace | None = None) -> str:
    if args is not None and getattr(args, "remote_sync_mode", None):
        return _normalize_remote_sync_mode(str(args.remote_sync_mode))
    return _normalize_remote_sync_mode(REMOTE_SYNC_MODE)


def _resolve_online_handoff_path(args: argparse.Namespace | None = None) -> Path:
    if args is not None and getattr(args, "handoff", None):
        return Path(args.handoff)
    if args is not None and getattr(args, "run_id", None):
        return Path("artifacts/online_runs") / str(args.run_id) / "handoff_package.json"
    return ONLINE_HANDOFF_PACKAGE_PATH


def _allow_invalid_debug_outputs(args: argparse.Namespace | None = None) -> bool:
    return bool(
        ALLOW_INVALID_DEBUG_OUTPUTS
        or (args is not None and getattr(args, "allow_invalid_outputs", False))
    )


def _build_online_artifact_paths(args: argparse.Namespace | None = None) -> dict[str, Path | None]:
    run_id = _resolve_run_id(args)
    run_dir = Path("artifacts/online_runs") / run_id
    configured_request = _resolve_online_request_path(args)
    return {
        "configured_request": configured_request,
        "run_dir": run_dir,
        "request": run_dir / "request.json",
        "candidates": run_dir / "candidates.json",
        "eval_result": run_dir / "results.json",
        "summary": run_dir / "summary.json",
        "final_request": run_dir / "final_generate_request.json",
        "final_eval_result": run_dir / "final_generate_result.json",
        "profile_retry_summary": run_dir / "profile_retry_summary.json",
        "handoff_package": run_dir / "handoff_package.json",
        "debug_final_request": run_dir / "debug_final_generate_request.json",
        "debug_handoff_package": run_dir / "debug_handoff_package.json",
        "receiver_request": run_dir / "receiver_request.json",
        "tool_pose_csv": _online_run_tool_pose_csv(run_id),
    }


def _prepare_online_request(
    *,
    request_path: Path,
    runtime_settings=None,
    args: argparse.Namespace | None = None,
) -> Path:
    from src.runtime.online.roundtrip import build_request_file
    active_settings = runtime_settings or APP_RUNTIME_SETTINGS

    if ONLINE_REQUEST_SOURCE == "existing":
        if not request_path.is_file():
            raise FileNotFoundError(
                f"Configured existing request file was not found: {request_path}"
            )
        print(f"[online] Using existing request: {request_path}")
        return request_path

    request_path.parent.mkdir(parents=True, exist_ok=True)
    request_metadata = {
        **_fixed_point_path_fallback_metadata(),
        **_local_machine_profile_metadata(args),
    }
    built_request_path = build_request_file(
        active_settings,
        request_path,
        round_index=ONLINE_ROUND_INDEX,
        candidate_limit=ONLINE_CANDIDATE_LIMIT,
        refresh_csv=not ONLINE_SKIP_POSE_SOLVER_WHEN_BUILDING_REQUEST,
        metadata=request_metadata,
    )
    print(f"[online] Built request: {built_request_path}")
    return built_request_path


def _write_selected_joint_path_csv(result, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = [
            "row_label",
            "inserted_transition_point",
            "config_flags",
            "j1_deg",
            "j2_deg",
            "j3_deg",
            "j4_deg",
            "j5_deg",
            "j6_deg",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_label, inserted_flag, selected_entry in zip(
            result.row_labels,
            result.inserted_flags,
            result.selected_path,
        ):
            joints = tuple(float(value) for value in selected_entry.joints)
            writer.writerow(
                {
                    "row_label": str(row_label),
                    "inserted_transition_point": int(bool(inserted_flag)),
                    "config_flags": ",".join(str(int(value)) for value in selected_entry.config_flags),
                    "j1_deg": joints[0] if len(joints) > 0 else "",
                    "j2_deg": joints[1] if len(joints) > 1 else "",
                    "j3_deg": joints[2] if len(joints) > 2 else "",
                    "j4_deg": joints[3] if len(joints) > 3 else "",
                    "j5_deg": joints[4] if len(joints) > 4 else "",
                    "j6_deg": joints[5] if len(joints) > 5 else "",
                }
            )
    return output_path


def _build_single_archive_payload(
    *,
    run_id: str,
    single_action: str,
    create_program: bool,
    artifact_paths: dict[str, Path | None],
    runtime_settings=None,
    result=None,
    runtime_error: str | None = None,
    allow_invalid_debug_outputs: bool = False,
) -> dict[str, object]:
    active_settings = runtime_settings or APP_RUNTIME_SETTINGS
    delivery_ready = _result_meets_delivery_gate(result)
    diagnostic_artifacts = {
        "run_dir": artifact_paths.get("run_dir"),
        "request": artifact_paths.get("request"),
        "eval_result": artifact_paths.get("eval_result"),
        "archive": artifact_paths.get("archive"),
        "profile_retry_summary": artifact_paths.get("profile_retry_summary"),
        "tool_pose_csv": artifact_paths.get("tool_pose_csv"),
    }
    official_artifacts = {
        "joint_path_csv": artifact_paths.get("joint_path_csv") if delivery_ready else None,
        "optimized_pose_csv": (
            artifact_paths.get("optimized_pose_csv")
            if delivery_ready and create_program
            else None
        ),
    }
    debug_artifacts = {
        "debug_joint_path_csv": (
            artifact_paths.get("debug_joint_path_csv")
            if allow_invalid_debug_outputs and not delivery_ready
            else None
        ),
    }
    payload: dict[str, object] = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "single",
        "single_action": single_action,
        "create_program": bool(create_program),
        "status": "error" if runtime_error else _result_semantic_status(result),
        "runtime_error": runtime_error,
        "basic_info": {
            "target_frame_a_origin_in_frame2_mm": tuple(
                float(v) for v in active_settings.target_frame_origin_mm
            ),
            "target_frame_a_rotation_in_frame2_xyz_deg": tuple(
                float(v) for v in active_settings.target_frame_rotation_xyz_deg
            ),
            "ik_backend": IK_BACKEND,
            "robot_name": ROBOT_NAME,
            "frame_name": FRAME_NAME,
            "program_name": active_settings.program_name,
        },
        "delivery_gate": {
            **STRICT_DELIVERY_GATE,
            "objective_reachable": bool(delivery_ready),
            "official_delivery_allowed": bool(delivery_ready),
            "strictly_valid": bool(delivery_ready),
            "allow_invalid_debug_outputs": bool(allow_invalid_debug_outputs),
        },
        "diagnostic_artifacts": {
            label: (str(path) if isinstance(path, Path) else None)
            for label, path in diagnostic_artifacts.items()
        },
        "official_artifacts": {
            label: (str(path) if isinstance(path, Path) else None)
            for label, path in official_artifacts.items()
        },
        "debug_artifacts": {
            label: (str(path) if isinstance(path, Path) else None)
            for label, path in debug_artifacts.items()
        },
        "artifacts": {
            label: (str(path) if isinstance(path, Path) else None)
            for label, path in {
                **diagnostic_artifacts,
                **official_artifacts,
                **debug_artifacts,
            }.items()
        },
    }
    if result is not None:
        payload["evaluation_result"] = {
            "request_id": str(result.request_id),
            "status": str(result.status),
            "timing_seconds": float(result.timing_seconds),
            "total_candidates": int(result.total_candidates),
            "invalid_row_count": int(result.invalid_row_count),
            "ik_empty_row_count": int(result.ik_empty_row_count),
            "config_switches": int(result.config_switches),
            "bridge_like_segments": int(result.bridge_like_segments),
            "big_circle_step_count": int(getattr(result, "big_circle_step_count", 0)),
            "branch_flip_ratio": float(getattr(result, "branch_flip_ratio", 0.0)),
            "violent_branch_segments": [
                dict(item) for item in getattr(result, "violent_branch_segments", ())
            ],
            "worst_joint_step_deg": float(result.worst_joint_step_deg),
            "mean_joint_step_deg": float(result.mean_joint_step_deg),
            "total_cost": float(result.total_cost),
            "gate_tier": str(getattr(result, "gate_tier", "diagnostic")),
            "block_reasons": [
                dict(item) for item in getattr(result, "block_reasons", ())
            ],
            "diagnostics": result.diagnostics,
            "error_message": result.error_message,
        }
    return payload


def _build_followup_profile_request(
    base_request,
    *,
    request_id: str,
    frame_a_origin_yz_profile_mm: tuple[tuple[float, float], ...],
    create_program: bool,
):
    from src.core.collab_models import ProfileEvaluationRequest

    return ProfileEvaluationRequest(
        request_id=request_id,
        robot_name=base_request.robot_name,
        frame_name=base_request.frame_name,
        motion_settings=dict(base_request.motion_settings),
        reference_pose_rows=tuple(dict(row) for row in base_request.reference_pose_rows),
        frame_a_origin_yz_profile_mm=tuple(
            (float(dy_mm), float(dz_mm))
            for dy_mm, dz_mm in frame_a_origin_yz_profile_mm
        ),
        row_labels=tuple(base_request.row_labels),
        inserted_flags=tuple(base_request.inserted_flags),
        strategy="exact_profile",
        start_joints=base_request.start_joints,
        run_window_repair=bool(SINGLE_RUN_WINDOW_REPAIR),
        run_inserted_repair=bool(SINGLE_RUN_INSERTED_REPAIR),
        include_pose_rows_in_result=False,
        create_program=create_program,
        program_name=base_request.program_name if create_program else None,
        optimized_csv_path=base_request.optimized_csv_path,
        metadata={
            **dict(base_request.metadata),
            **_fixed_point_path_fallback_metadata(),
            "entrypoint": "main_single_followup",
            "requested_create_program": bool(create_program),
        },
    )


def _fixed_point_path_fallback_metadata() -> dict[str, object]:
    return {
        "enable_fixed_point_path_fallback": bool(ENABLE_FIXED_POINT_PATH_FALLBACK),
        "fixed_point_fallback_max_candidates_per_config": int(
            FIXED_POINT_PATH_FALLBACK_MAX_CANDIDATES_PER_CONFIG
        ),
        "fixed_point_fallback_disable_guided_path": bool(
            FIXED_POINT_PATH_FALLBACK_DISABLE_GUIDED_PATH
        ),
        "fixed_point_fallback_run_window_repair": bool(
            FIXED_POINT_PATH_FALLBACK_RUN_WINDOW_REPAIR
        ),
        "fixed_point_fallback_run_inserted_repair": bool(
            FIXED_POINT_PATH_FALLBACK_RUN_INSERTED_REPAIR
        ),
    }


def _run_single_profile_evaluation(
    *,
    run_id: str,
    single_action: str,
    create_program: bool,
    allow_invalid_debug_outputs: bool,
    runtime_settings=None,
) -> tuple[dict[str, Path | None], str]:
    from src.runtime.request_builder import build_profile_evaluation_request
    from src.robodk_runtime.eval_worker import evaluate_batch_request, evaluate_single_request

    base_settings = runtime_settings or APP_RUNTIME_SETTINGS
    run_scoped_tool_pose_csv = _single_run_tool_pose_csv(run_id)
    active_settings = _runtime_settings_with_tool_pose_csv(
        base_settings,
        tool_pose_csv=run_scoped_tool_pose_csv,
    )
    artifact_paths = _build_single_artifact_paths(run_id, runtime_settings=active_settings)
    run_dir = artifact_paths["run_dir"]
    assert run_dir is not None
    run_dir.mkdir(parents=True, exist_ok=True)

    request = build_profile_evaluation_request(
        active_settings,
        request_id=f"single_{single_action}_{run_id}",
        strategy="full_search",
        refresh_csv=True,
        run_window_repair=bool(SINGLE_RUN_WINDOW_REPAIR),
        run_inserted_repair=bool(SINGLE_RUN_INSERTED_REPAIR),
        include_pose_rows_in_result=False,
        create_program=create_program,
        program_name=active_settings.program_name if create_program else None,
        optimized_csv_path=str(active_settings.tool_poses_frame2_csv),
        metadata={
            "entrypoint": "main_single",
            "single_action": single_action,
            "create_program": bool(create_program),
            **_fixed_point_path_fallback_metadata(),
        },
    )
    write_json_file(artifact_paths["request"], request.to_dict())

    runtime_error: str | None = None
    batch_result = None
    result = None
    retry_candidate_limit, retry_repair_limit, retry_max_rounds = (
        _resolve_local_profile_retry_budget()
    )
    try:
        initial_result, initial_search_result = evaluate_single_request(request)
        batch_result = EvaluationBatchResult(results=(initial_result,))
        write_json_file(artifact_paths["eval_result"], batch_result.to_dict())
        if batch_result.results:
            result = batch_result.results[0]
            delivery_ready = _result_meets_delivery_gate(result)
            if (
                (not delivery_ready or _result_has_continuity_warnings(result))
                and ENABLE_LOCAL_PROFILE_RETRY
                and IK_BACKEND == "six_axis_ik"
            ):
                from src.runtime.local_retry import run_local_profile_retry

                retry_outcome = run_local_profile_retry(
                    request,
                    result,
                    candidate_limit=retry_candidate_limit,
                    repair_retry_limit=retry_repair_limit,
                    max_rounds=retry_max_rounds,
                    baseline_search_result=initial_search_result,
                )
                write_json_file(
                    artifact_paths["profile_retry_summary"],
                    {
                        **retry_outcome.payload,
                        "budget": {
                            "candidate_limit": retry_candidate_limit,
                            "repair_retry_limit": retry_repair_limit,
                            "max_rounds": retry_max_rounds,
                        },
                    },
                )
                print(
                    "[single-eval] local profile retry "
                    f"best_status={retry_outcome.best_result.status}, "
                    f"worst_joint_step={retry_outcome.best_result.worst_joint_step_deg:.3f}"
                )
                if (
                    retry_outcome.best_result.request_id != result.request_id
                    or retry_outcome.best_result.status != result.status
                    or abs(
                        float(retry_outcome.best_result.worst_joint_step_deg)
                        - float(result.worst_joint_step_deg)
                    )
                    > 1e-9
                ):
                    if create_program and retry_outcome.best_result.status == "valid":
                        followup_request = _build_followup_profile_request(
                            request,
                            request_id=f"{request.request_id}_retry_program",
                            frame_a_origin_yz_profile_mm=retry_outcome.best_result.frame_a_origin_yz_profile_mm,
                            create_program=True,
                        )
                        followup_result = evaluate_batch_request(
                            EvaluationBatchRequest(evaluations=(followup_request,))
                        ).results[0]
                        result = followup_result
                    else:
                        result = retry_outcome.best_result

            write_json_file(
                artifact_paths["eval_result"],
                EvaluationBatchResult(results=(result,)).to_dict(),
            )
            delivery_ready = _result_meets_delivery_gate(result)
            if delivery_ready:
                _write_selected_joint_path_csv(result, artifact_paths["joint_path_csv"])
                if (
                    artifact_paths["debug_joint_path_csv"] is not None
                    and artifact_paths["debug_joint_path_csv"].exists()
                ):
                    artifact_paths["debug_joint_path_csv"].unlink()
            elif artifact_paths["joint_path_csv"] is not None and artifact_paths["joint_path_csv"].exists():
                artifact_paths["joint_path_csv"].unlink()
            if (
                not delivery_ready
                and allow_invalid_debug_outputs
                and artifact_paths["debug_joint_path_csv"] is not None
            ):
                _write_selected_joint_path_csv(result, artifact_paths["debug_joint_path_csv"])
                print(
                    "[single-eval] Debug invalid output written: "
                    f"{artifact_paths['debug_joint_path_csv']}"
                )
            if not delivery_ready:
                print(
                    "[single-eval] Delivery gate blocked official outputs; "
                    "keeping diagnostic artifacts only."
                )
            print(
                "[single-eval] "
                f"status={result.status}, "
                f"ik_empty_rows={result.ik_empty_row_count}, "
                f"config_switches={result.config_switches}, "
                f"bridge_like_segments={result.bridge_like_segments}, "
                f"big_circle_step_count={int(getattr(result, 'big_circle_step_count', 0))}, "
                f"worst_joint_step={result.worst_joint_step_deg}, "
                f"branch_flip_ratio={float(getattr(result, 'branch_flip_ratio', 0.0)):.3f}"
            )
    except Exception as exc:
        runtime_error = f"{type(exc).__name__}: {exc}"
        archive_payload = _build_single_archive_payload(
            run_id=run_id,
            single_action=single_action,
            create_program=create_program,
            artifact_paths=artifact_paths,
            runtime_settings=active_settings,
            result=result,
            runtime_error=runtime_error,
            allow_invalid_debug_outputs=allow_invalid_debug_outputs,
        )
        write_json_file(artifact_paths["archive"], archive_payload)
        raise

    archive_payload = _build_single_archive_payload(
        run_id=run_id,
        single_action=single_action,
        create_program=create_program,
        artifact_paths=artifact_paths,
        runtime_settings=active_settings,
        result=result,
        runtime_error=runtime_error,
        allow_invalid_debug_outputs=allow_invalid_debug_outputs,
    )
    write_json_file(artifact_paths["archive"], archive_payload)

    delivery_ready = _result_meets_delivery_gate(result)
    optimized_csv = artifact_paths["optimized_pose_csv"]
    if (
        optimized_csv is None
        or not create_program
        or not delivery_ready
        or not optimized_csv.exists()
    ):
        optimized_csv = None

    return {
        "run_dir": run_dir,
        "request": artifact_paths["request"],
        "eval_result": artifact_paths["eval_result"] if artifact_paths["eval_result"].exists() else None,
        "archive": artifact_paths["archive"],
        "joint_path_csv": (
            artifact_paths["joint_path_csv"]
            if delivery_ready
            and artifact_paths["joint_path_csv"] is not None
            and artifact_paths["joint_path_csv"].exists()
            else None
        ),
        "debug_joint_path_csv": (
            artifact_paths["debug_joint_path_csv"]
            if allow_invalid_debug_outputs
            and not delivery_ready
            and artifact_paths["debug_joint_path_csv"] is not None
            and artifact_paths["debug_joint_path_csv"].exists()
            else None
        ),
        "profile_retry_summary": (
            artifact_paths["profile_retry_summary"]
            if artifact_paths["profile_retry_summary"].exists()
            else None
        ),
        "tool_pose_csv": artifact_paths["tool_pose_csv"],
        "optimized_pose_csv": optimized_csv,
    }, _result_semantic_status(result)


def _run_single_mode(
    single_action: str,
    *,
    run_id: str | None,
    allow_invalid_debug_outputs: bool,
    runtime_settings=None,
) -> tuple[dict[str, Path | None], str]:
    active_settings = runtime_settings or APP_RUNTIME_SETTINGS
    if single_action == "visualize":
        print("Running single-machine mode: visualization")
        run_visualization(active_settings)
        return {
            "tool_pose_csv": active_settings.tool_poses_frame2_csv,
            "optimized_pose_csv": None,
        }, "success"

    if run_id is None:
        raise ValueError("run_id is required for single solve/program actions.")

    if single_action == "program":
        print("Running single-machine mode: solve + RoboDK program generation")
        return _run_single_profile_evaluation(
            run_id=run_id,
            single_action=single_action,
            create_program=True,
            allow_invalid_debug_outputs=allow_invalid_debug_outputs,
            runtime_settings=active_settings,
        )

    if single_action == "solve":
        print("Running single-machine mode: solve only (skip RoboDK program generation)")
        return _run_single_profile_evaluation(
            run_id=run_id,
            single_action=single_action,
            create_program=False,
            allow_invalid_debug_outputs=allow_invalid_debug_outputs,
            runtime_settings=active_settings,
        )

    raise ValueError(f"Unsupported single action: {single_action}")


def _run_online_mode(
    args: argparse.Namespace | None = None,
    *,
    runtime_settings=None,
    run_id_override: str | None = None,
    request_path_override: Path | None = None,
    online_command_override: str | None = None,
) -> tuple[dict[str, Path | None], str]:
    from src.runtime.online.roundtrip import (
        CommandLogOptions,
        run_online_coordinator,
        run_receiver_role,
        run_server_role,
        run_worker_eval,
        setup_server,
    )

    base_settings = runtime_settings or APP_RUNTIME_SETTINGS
    online_command = (
        str(online_command_override)
        if online_command_override is not None
        else _resolve_online_command(args)
    )
    run_id = str(run_id_override) if run_id_override is not None else _resolve_run_id(args)
    run_scoped_tool_pose_csv = _online_run_tool_pose_csv(run_id)
    active_settings = _runtime_settings_with_tool_pose_csv(
        base_settings,
        tool_pose_csv=run_scoped_tool_pose_csv,
    )
    request_path = (
        Path(request_path_override)
        if request_path_override is not None
        else _resolve_online_request_path(args)
    )
    roundtrip_log_path = build_run_log_path(
        mode="online",
        action=online_command,
        run_id=run_id,
        write_detailed_log_file=WRITE_DETAILED_LOG_FILE,
        run_log_dir=RUN_LOG_DIR,
    )
    command_log_options = CommandLogOptions(
        log_path=roundtrip_log_path,
        show_command_details=SHOW_COMMAND_DETAILS,
        show_worker_output=SHOW_DETAILED_TERMINAL_LOGS,
    )
    allow_invalid_debug_outputs = _allow_invalid_debug_outputs(args)
    local_profile = _resolve_local_machine_profile(args)
    print(
        "[profile] Local machine profile: "
        f"{local_profile_status_text(local_profile)}."
    )

    if online_command == "setup_server":
        print("Running online mode: setup_server")
        setup_server(
            host=ONLINE_HOST,
            server_dir=ONLINE_SERVER_DIR,
            env_name=ONLINE_ENV_NAME,
            enable_slurm_setup=ONLINE_SETUP_SERVER_ENABLE_SLURM,
            log_options=command_log_options,
        )
        return {"detail_log": roundtrip_log_path}, "success"

    if online_command == "receiver":
        print("Running online mode: receiver role (local RoboDK finalization)")
        artifacts = run_receiver_role(
            handoff_path=_resolve_online_handoff_path(args),
            run_id=run_id,
            local_python=(
                args.local_python
                if args is not None and getattr(args, "local_python", None)
                else ONLINE_LOCAL_PYTHON
            ),
            local_profile=local_profile,
            allow_invalid_handoff=allow_invalid_debug_outputs,
            log_options=command_log_options,
        )
        if artifacts.final_program_name:
            print(f"[online/receiver] Final program: {artifacts.final_program_name}")
        return {
            "handoff_package": artifacts.handoff_package_path,
            "receiver_request": artifacts.receiver_request_path,
            "final_eval_result": artifacts.result_path,
            "optimized_pose_csv": artifacts.optimized_pose_csv_path,
            "detail_log": artifacts.log_path,
        }, artifacts.result_status

    prepared_request_path = _prepare_online_request(
        request_path=request_path,
        runtime_settings=active_settings,
        args=args,
    )

    if online_command == "build_request":
        print("Running online mode: build_request")
        return {
            "request": prepared_request_path,
            "tool_pose_csv": active_settings.tool_poses_frame2_csv,
            "detail_log": roundtrip_log_path,
        }, "success"

    if online_command == "worker_eval":
        print("Running online mode: local worker_eval")
        result_path = Path("artifacts/online_runs") / run_id / "eval_result.json"
        artifacts = run_worker_eval(
            request_path=prepared_request_path,
            result_path=result_path,
            local_python=(
                args.local_python
                if args is not None and getattr(args, "local_python", None)
                else ONLINE_LOCAL_PYTHON
            ),
            local_profile=local_profile,
            log_options=command_log_options,
        )
        _summarize_worker_eval_result(artifacts.result_path)
        worker_result_payload = load_json_file(artifacts.result_path)
        worker_results = worker_result_payload.get("results", [])
        worker_status = (
            "success"
            if not worker_results
            else (
                "valid"
                if _result_payload_meets_delivery_gate(worker_results[0])
                else "invalid"
            )
        )
        return {
            "request": artifacts.request_path,
            "eval_result": artifacts.result_path,
            "tool_pose_csv": active_settings.tool_poses_frame2_csv,
            "detail_log": artifacts.log_path,
        }, worker_status

    retry_candidate_limit, retry_repair_limit, retry_max_rounds = (
        _resolve_online_profile_retry_budget()
    )
    if retry_max_rounds <= 0 or retry_candidate_limit <= 0:
        print(
            "[online] Fast delivery mode: server retry/repair is disabled. "
            "Set WPS_ONLINE_RETRY_MAX_ROUNDS=1 for continuity refinement."
        )
    else:
        print(
            "[online] Server retry/repair enabled: "
            f"candidate_limit={retry_candidate_limit}, "
            f"repair_limit={retry_repair_limit}, "
            f"max_rounds={retry_max_rounds}."
        )

    if online_command == "server":
        print("Running online mode: server role (pure SixAxisIK compute, no RoboDK)")
        artifacts = run_server_role(
            request_path=prepared_request_path,
            run_id=run_id,
            final_program_name=active_settings.program_name,
            optimized_csv_path=str(active_settings.tool_poses_frame2_csv),
            retry_candidate_limit=retry_candidate_limit,
            retry_repair_limit=retry_repair_limit,
            retry_max_rounds=retry_max_rounds,
            allow_invalid_outputs=allow_invalid_debug_outputs,
            log_options=command_log_options,
        )
        return {
            "request": artifacts.request_path,
            "candidates": artifacts.candidates_path,
            "eval_result": artifacts.results_path,
            "summary": artifacts.summary_path,
            "profile_retry_summary": artifacts.profile_retry_summary_path,
            "final_request": artifacts.final_request_path,
            "handoff_package": artifacts.handoff_package_path,
            "debug_final_request": artifacts.debug_final_request_path,
            "debug_handoff_package": artifacts.debug_handoff_package_path,
            "tool_pose_csv": active_settings.tool_poses_frame2_csv,
            "detail_log": artifacts.log_path,
        }, artifacts.result_status

    print("Running online mode: coordinator role (local machine orchestrates server + optional receiver)")
    if ONLINE_AUTO_SETUP_SERVER:
        print("[online] Auto setup enabled; preparing server first...")
        setup_server(
            host=ONLINE_HOST,
            server_dir=ONLINE_SERVER_DIR,
            env_name=ONLINE_ENV_NAME,
            enable_slurm_setup=ONLINE_SETUP_SERVER_ENABLE_SLURM,
            log_options=command_log_options,
        )

    artifacts = run_online_coordinator(
        host=ONLINE_HOST,
        server_dir=ONLINE_SERVER_DIR,
        env_name=ONLINE_ENV_NAME,
        request_path=prepared_request_path,
        run_id=run_id,
        local_python=(
            args.local_python
            if args is not None and getattr(args, "local_python", None)
            else ONLINE_LOCAL_PYTHON
        ),
        local_profile=local_profile,
        generate_final_program=(
            ONLINE_FINAL_GENERATE_PROGRAM
            and not bool(args is not None and getattr(args, "skip_final_generate", False))
        ),
        final_program_name=active_settings.program_name,
        optimized_csv_path=str(active_settings.tool_poses_frame2_csv),
        retry_candidate_limit=retry_candidate_limit,
        retry_repair_limit=retry_repair_limit,
        retry_max_rounds=retry_max_rounds,
        allow_invalid_outputs=allow_invalid_debug_outputs,
        enforce_remote_sync_guard=bool(ENFORCE_REMOTE_SYNC_GUARD),
        remote_sync_mode=_resolve_remote_sync_mode(args),
        log_options=command_log_options,
    )
    if artifacts.final_program_name:
        print(f"[online] Final program: {artifacts.final_program_name}")
    return {
        "request": artifacts.request_path,
        "candidates": artifacts.candidates_path,
        "eval_result": artifacts.results_path,
        "summary": artifacts.summary_path,
        "profile_retry_summary": artifacts.profile_retry_summary_path,
        "final_request": artifacts.final_request_path,
        "final_eval_result": artifacts.final_result_path,
        "handoff_package": artifacts.handoff_package_path,
        "debug_final_request": artifacts.debug_final_request_path,
        "debug_handoff_package": artifacts.debug_handoff_package_path,
        "optimized_pose_csv": artifacts.optimized_pose_csv_path,
        "tool_pose_csv": active_settings.tool_poses_frame2_csv,
        "detail_log": artifacts.log_path,
    }, artifacts.result_status


def _build_origin_search_settings() -> OriginSearchSettings:
    return OriginSearchSettings(
        target_frame_a_origin_in_frame2_mm=tuple(
            float(value) for value in TARGET_FRAME_A_ORIGIN_IN_FRAME2_MM
        ),
        use_server=TARGET_ORIGIN_YZ_SEARCH_USE_SERVER,
        square_size_mm=TARGET_ORIGIN_YZ_SEARCH_SQUARE_SIZE_MM,
        initial_step_mm=TARGET_ORIGIN_YZ_SEARCH_INITIAL_STEP_MM,
        min_step_mm=TARGET_ORIGIN_YZ_SEARCH_MIN_STEP_MM,
        max_iters=TARGET_ORIGIN_YZ_SEARCH_MAX_ITERS,
        beam_width=TARGET_ORIGIN_YZ_SEARCH_BEAM_WIDTH,
        diagonal_policy=TARGET_ORIGIN_YZ_SEARCH_DIAGONAL_POLICY,
        polish_step_mm=TARGET_ORIGIN_YZ_SEARCH_POLISH_STEP_MM,
        workers=TARGET_ORIGIN_YZ_SEARCH_WORKERS,
        strategy=TARGET_ORIGIN_YZ_SEARCH_STRATEGY,
        run_window_repair=TARGET_ORIGIN_YZ_SEARCH_RUN_WINDOW_REPAIR,
        run_inserted_repair=TARGET_ORIGIN_YZ_SEARCH_RUN_INSERTED_REPAIR,
        validation_grid_step_mm=TARGET_ORIGIN_YZ_SEARCH_VALIDATION_GRID_STEP_MM,
        top_k=TARGET_ORIGIN_YZ_SEARCH_TOP_K,
        min_separation_mm=TARGET_ORIGIN_YZ_SEARCH_MIN_SEPARATION_MM,
        outside_fallback_count=TARGET_ORIGIN_YZ_OUTSIDE_FALLBACK_COUNT,
        outside_fallback_max_rings=TARGET_ORIGIN_YZ_OUTSIDE_FALLBACK_MAX_RINGS,
        outside_fallback_ring_step_mm=TARGET_ORIGIN_YZ_OUTSIDE_FALLBACK_RING_STEP_MM,
        outside_fallback_edge_step_mm=TARGET_ORIGIN_YZ_OUTSIDE_FALLBACK_EDGE_STEP_MM,
        remote_sync_mode=REMOTE_SYNC_MODE,
        enforce_remote_sync_guard=ENFORCE_REMOTE_SYNC_GUARD,
    )


def _build_origin_search_server_settings() -> OriginSearchServerSettings:
    return OriginSearchServerSettings(
        host=ONLINE_HOST,
        server_dir=ONLINE_SERVER_DIR,
        env_name=ONLINE_ENV_NAME,
    )


def _build_runtime_settings_for_origin(
    origin_mm: tuple[float, float, float],
):
    return _build_runtime_settings(
        target_frame_origin_mm=tuple(float(value) for value in origin_mm),
    )


def _origin_candidate_sort_key(
    candidate: dict[str, object],
) -> tuple[float, ...]:
    outside_distance = candidate.get("outside_square_distance_mm")
    try:
        outside_distance_value = float(outside_distance)
    except (TypeError, ValueError):
        outside_distance_value = math.inf
    return (
        0.0 if bool(candidate.get("official_delivery_allowed", False)) else 1.0,
        float(candidate.get("invalid_row_count", 0)),
        float(candidate.get("ik_empty_row_count", 0)),
        outside_distance_value,
        float(candidate.get("bridge_like_segments", 0)),
        float(candidate.get("big_circle_step_count", 0)),
        float(candidate.get("branch_flip_ratio", 0.0)),
        float(candidate.get("worst_joint_step_deg", 0.0)),
        float(candidate.get("mean_joint_step_deg", 0.0)),
        float(candidate.get("config_switches", 0)),
        float(candidate.get("distance_from_seed_yz_mm", 0.0)),
    )


def _evaluate_origin_candidate_gate(row: dict[str, object]) -> dict[str, object]:
    return summary_metrics_delivery_gate_details(row)


def _collect_origin_candidates(
    payload: dict[str, object],
) -> list[dict[str, object]]:
    preferred_rows = payload.get("recommended", [])
    fallback_rows = payload.get("results", [])
    candidate_rows: list[object] = []
    if isinstance(preferred_rows, list):
        candidate_rows.extend(preferred_rows)
    if isinstance(fallback_rows, list):
        candidate_rows.extend(fallback_rows)

    candidates: list[dict[str, object]] = []
    seen: set[tuple[float, float, float]] = set()
    for row in candidate_rows:
        if not isinstance(row, dict):
            continue
        try:
            candidate = {
                "x_mm": float(row.get("x_mm")),
                "y_mm": float(row.get("y_mm")),
                "z_mm": float(row.get("z_mm")),
                "status": str(row.get("status", "invalid")),
                "invalid_row_count": int(row.get("invalid_row_count", 0)),
                "ik_empty_row_count": int(row.get("ik_empty_row_count", 0)),
                "config_switches": int(row.get("config_switches", 0)),
                "bridge_like_segments": int(row.get("bridge_like_segments", 0)),
                "big_circle_step_count": int(row.get("big_circle_step_count", 0)),
                "branch_flip_ratio": float(row.get("branch_flip_ratio", 0.0)),
                "worst_joint_step_deg": float(row.get("worst_joint_step_deg", 0.0)),
                "mean_joint_step_deg": float(row.get("mean_joint_step_deg", 0.0)),
                "distance_from_seed_yz_mm": float(row.get("distance_from_seed_yz_mm", 0.0)),
            }
            if row.get("outside_square_distance_mm") is not None:
                candidate["outside_square_distance_mm"] = float(
                    row.get("outside_square_distance_mm")
                )
        except (TypeError, ValueError):
            continue
        origin_key = (
            round(float(candidate["x_mm"]), 6),
            round(float(candidate["y_mm"]), 6),
            round(float(candidate["z_mm"]), 6),
        )
        if origin_key in seen:
            continue
        seen.add(origin_key)
        candidate.update(_evaluate_origin_candidate_gate(candidate))
        candidates.append(candidate)
    candidates.sort(key=_origin_candidate_sort_key)
    return candidates


def _resolve_origin_search_post_dispatch(args: argparse.Namespace | None = None) -> str:
    value = (
        getattr(args, "origin_search_dispatch", None)
        if args is not None
        else None
    )
    dispatch = str(value) if value is not None else str(TARGET_ORIGIN_YZ_SEARCH_POST_DISPATCH)
    dispatch = dispatch.strip().lower()
    if dispatch not in {"none", "single_action", "online_role"}:
        raise ValueError(
            "TARGET_ORIGIN_YZ_SEARCH_POST_DISPATCH must be one of: "
            "none, single_action, online_role."
        )
    return dispatch


def _resolve_origin_search_post_top_n(args: argparse.Namespace | None = None) -> int:
    value = (
        getattr(args, "origin_search_post_top_n", None)
        if args is not None
        else None
    )
    if value is None:
        return max(1, int(TARGET_ORIGIN_YZ_SEARCH_POST_TOP_N))
    return max(1, int(value))


def _resolve_origin_search_usable_count(args: argparse.Namespace | None = None) -> int:
    value = (
        getattr(args, "origin_search_usable_count", None)
        if args is not None
        else None
    )
    if value is None:
        return max(1, int(TARGET_ORIGIN_YZ_SEARCH_USABLE_COUNT))
    return max(1, int(value))


def _resolve_origin_search_debug_fallback_count(
    args: argparse.Namespace | None = None,
) -> int:
    value = (
        getattr(args, "origin_search_debug_fallback_count", None)
        if args is not None
        else None
    )
    if value is None:
        return max(0, int(DEBUG_FALLBACK_IMPORT_COUNT))
    return max(0, int(value))


def _sanitize_program_token(text: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in {"_", "-"} else "_"
        for character in str(text)
    )
    return safe.strip("_") or "candidate"


def _origin_debug_tag(candidate: dict[str, object]) -> str:
    block_reasons = candidate.get("block_reasons", [])
    if isinstance(block_reasons, list) and block_reasons:
        first_reason = block_reasons[0]
        if isinstance(first_reason, dict):
            code = str(first_reason.get("code", "blocked")).upper()
        else:
            code = str(first_reason).upper()
    else:
        code = "BLOCKED"
    bridge = int(candidate.get("bridge_like_segments", 0))
    big_circle = int(candidate.get("big_circle_step_count", 0))
    worst = int(round(float(candidate.get("worst_joint_step_deg", 0.0))))
    return _sanitize_program_token(f"{code}_BR{bridge}_BC{big_circle}_WS{worst}")


def _build_origin_program_name(
    *,
    base_program_name: str,
    candidate_index: int,
    candidate: dict[str, object],
    debug_tier: bool,
) -> str:
    y_mm = float(candidate["y_mm"])
    z_mm = float(candidate["z_mm"])
    y_token = f"Y{int(round(y_mm))}"
    z_token = f"Z{int(round(z_mm))}"
    prefix = "DBG" if debug_tier else "ORI"
    suffix = _origin_debug_tag(candidate) if debug_tier else "PASS"
    return _sanitize_program_token(
        f"{prefix}_{candidate_index:02d}_{base_program_name}_{y_token}_{z_token}_{suffix}"
    )


def _run_origin_search_mode(
    *,
    run_id: str,
    args: argparse.Namespace | None = None,
) -> tuple[dict[str, Path | None], str]:
    print("[origin-search] algorithm=smart-square beam search (non-exhaustive).")
    paths, _status = run_origin_search(
        run_id=run_id,
        settings=_build_origin_search_settings(),
        server_settings=_build_origin_search_server_settings(),
        show_command_details=SHOW_COMMAND_DETAILS,
    )
    result_path = paths.get("origin_search_result")
    if not isinstance(result_path, Path) or not result_path.exists():
        return paths, "failed"

    payload = load_json_file(result_path)
    candidates = _collect_origin_candidates(payload)
    all_good_candidates = [
        candidate
        for candidate in candidates
        if bool(candidate.get("official_delivery_allowed", False))
    ]
    display_candidates = all_good_candidates[: _resolve_origin_search_usable_count(args)]
    if display_candidates:
        print(f"[origin-search] official-gate pass candidates (top {len(display_candidates)} shown):")
        for index, item in enumerate(display_candidates, start=1):
            outside_note = (
                f" outside={float(item['outside_square_distance_mm']):.3f}mm"
                if item.get("outside_square_distance_mm") is not None
                else ""
            )
            print(
                f"[origin-search] #{index} origin=({float(item['x_mm']):.3f}, "
                f"{float(item['y_mm']):.3f}, {float(item['z_mm']):.3f}) "
                f"switches={int(item['config_switches'])} "
                f"bridge={int(item['bridge_like_segments'])} "
                f"big_circle={int(item['big_circle_step_count'])} "
                f"worst={float(item['worst_joint_step_deg']):.3f} "
                f"mean={float(item['mean_joint_step_deg']):.6f}"
                f"{outside_note}"
            )
    else:
        print("[origin-search] no candidate passed official delivery gate.")

    dispatch_mode = _resolve_origin_search_post_dispatch(args)
    post_top_n = _resolve_origin_search_post_top_n(args)
    dispatch_candidates = (
        list(all_good_candidates[:post_top_n])
        if all_good_candidates
        else list(candidates[: _resolve_origin_search_debug_fallback_count(args)])
    )
    dispatch_tier = "official" if all_good_candidates else "debug"
    dispatch_top_n = len(dispatch_candidates)
    dispatch_records: list[dict[str, object]] = []
    if dispatch_mode != "none" and dispatch_candidates:
        print(
            f"[origin-search] post-dispatch mode={dispatch_mode}, "
            f"tier={dispatch_tier}, candidates={dispatch_top_n}"
        )
        for candidate_index, item in enumerate(dispatch_candidates, start=1):
            origin_mm = (
                float(item["x_mm"]),
                float(item["y_mm"]),
                float(item["z_mm"]),
            )
            candidate_run_id = f"{run_id}_cand{candidate_index:02d}"
            candidate_program_name = _build_origin_program_name(
                base_program_name=PROGRAM_NAME,
                candidate_index=candidate_index,
                candidate=item,
                debug_tier=(dispatch_tier != "official"),
            )
            candidate_settings = replace(
                _build_runtime_settings_for_origin(origin_mm),
                program_name=candidate_program_name,
            )
            print(
                "[origin-search] dispatch "
                f"#{candidate_index}: origin=({origin_mm[0]:.3f}, {origin_mm[1]:.3f}, {origin_mm[2]:.3f}), "
                f"tier={item.get('gate_tier')}, program={candidate_program_name}"
            )
            if dispatch_mode == "single_action":
                if SINGLE_ACTION == "visualize":
                    record = {
                        "candidate_index": candidate_index,
                        "origin_mm": list(origin_mm),
                        "program_name": candidate_program_name,
                        "gate_tier": item.get("gate_tier"),
                        "block_reasons": list(item.get("block_reasons", [])),
                        "dispatch_mode": dispatch_mode,
                        "status": "skipped",
                        "reason": "SINGLE_ACTION=visualize does not produce joint path/program artifacts.",
                    }
                    dispatch_records.append(record)
                    print("[origin-search] skipped: SINGLE_ACTION=visualize")
                    continue
                candidate_paths, candidate_status = _run_single_mode(
                    SINGLE_ACTION,
                    run_id=candidate_run_id,
                    allow_invalid_debug_outputs=_allow_invalid_debug_outputs(args),
                    runtime_settings=candidate_settings,
                )
                dispatch_records.append(
                    {
                        "candidate_index": candidate_index,
                        "origin_mm": list(origin_mm),
                        "program_name": candidate_program_name,
                        "gate_tier": item.get("gate_tier"),
                        "block_reasons": list(item.get("block_reasons", [])),
                        "candidate_metrics": dict(item),
                        "dispatch_mode": dispatch_mode,
                        "single_action": SINGLE_ACTION,
                        "status": candidate_status,
                        "artifacts": {
                            key: (str(value) if isinstance(value, Path) else None)
                            for key, value in candidate_paths.items()
                        },
                    }
                )
                continue

            online_command = ONLINE_ROLE.strip().lower()
            if online_command == "receiver":
                print(
                    "[origin-search] ONLINE_ROLE=receiver requires a handoff package; "
                    "falling back to coordinator for origin dispatch."
                )
                online_command = "coordinator"
            candidate_request_path = (
                origin_search_run_dir(run_id)
                / f"candidate_{candidate_index:02d}_request.json"
            )
            candidate_paths, candidate_status = _run_online_mode(
                args,
                runtime_settings=candidate_settings,
                run_id_override=candidate_run_id,
                request_path_override=candidate_request_path,
                online_command_override=online_command,
            )
            dispatch_records.append(
                {
                    "candidate_index": candidate_index,
                    "origin_mm": list(origin_mm),
                    "program_name": candidate_program_name,
                    "gate_tier": item.get("gate_tier"),
                    "block_reasons": list(item.get("block_reasons", [])),
                    "candidate_metrics": dict(item),
                    "dispatch_mode": dispatch_mode,
                    "online_role": ONLINE_ROLE,
                    "effective_online_command": online_command,
                    "status": candidate_status,
                    "artifacts": {
                        key: (str(value) if isinstance(value, Path) else None)
                        for key, value in candidate_paths.items()
                    },
                }
            )
    elif dispatch_mode != "none":
        print("[origin-search] no candidate available for post-dispatch.")

    selection_summary_path = origin_search_run_dir(run_id) / "origin_search_selection.json"
    manifest_path = origin_search_run_dir(run_id) / "origin_search_import_manifest.json"
    best_origin = all_good_candidates[0] if all_good_candidates else (candidates[0] if candidates else None)
    manifest_payload = {
        "run_id": run_id,
        "dispatch_mode": dispatch_mode,
        "dispatch_tier": dispatch_tier if dispatch_candidates else "none",
        "official_candidate_count": len(all_good_candidates),
        "debug_fallback_count": 0 if all_good_candidates else len(dispatch_candidates),
        "best_origin": None if best_origin is None else dict(best_origin),
        "entries": [
            {
                "candidate_index": int(record.get("candidate_index", 0)),
                "origin_mm": list(record.get("origin_mm", [])),
                "program_name": record.get("program_name"),
                "gate_tier": record.get("gate_tier"),
                "block_reasons": list(record.get("block_reasons", [])),
                "status": record.get("status"),
                "artifacts": dict(record.get("artifacts", {})),
                "candidate_metrics": dict(record.get("candidate_metrics", {})),
                "best_origin": bool(
                    best_origin is not None
                    and list(record.get("origin_mm", []))
                    == [
                        float(best_origin["x_mm"]),
                        float(best_origin["y_mm"]),
                        float(best_origin["z_mm"]),
                    ]
                ),
            }
            for record in dispatch_records
        ],
    }
    write_json_file(manifest_path, manifest_payload)
    selection_summary_payload = {
        "run_id": run_id,
        "dispatch_mode": dispatch_mode,
        "dispatch_tier": dispatch_tier if dispatch_candidates else "none",
        "dispatch_top_n": dispatch_top_n,
        "official_candidates": all_good_candidates,
        "all_candidates": candidates,
        "dispatch_records": dispatch_records,
        "import_manifest": str(manifest_path),
    }
    write_json_file(selection_summary_path, selection_summary_payload)

    merged_paths = dict(paths)
    merged_paths["origin_search_selection"] = selection_summary_path
    merged_paths["origin_search_import_manifest"] = manifest_path
    status = "success"
    if not all_good_candidates:
        status = "invalid"
    elif dispatch_records and any(
        str(record.get("status")) in {"invalid", "failed"}
        for record in dispatch_records
    ):
        status = "invalid"
    return merged_paths, status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unified local entrypoint for single-machine and online requester/worker flows. "
            "By default, the top-level config in main.py controls the run."
        )
    )
    parser.add_argument("--mode", choices=("single", "online", "origin_search"))
    parser.add_argument(
        "--origin-search",
        action="store_true",
        help="Run smart-square Y/Z origin search around TARGET_FRAME_A_ORIGIN_IN_FRAME2_MM.",
    )
    parser.add_argument(
        "--no-origin-search",
        action="store_true",
        help="Do not enter origin_search automatically even when ENABLE_TARGET_ORIGIN_YZ_SEARCH=True.",
    )
    parser.add_argument(
        "--origin-search-dispatch",
        choices=("none", "single_action", "online_role"),
        help=(
            "Post origin-search action: none=only report candidates; "
            "single_action=run local SINGLE_ACTION; online_role=run online ONLINE_ROLE."
        ),
    )
    parser.add_argument(
        "--origin-search-post-top-n",
        type=int,
        help="How many usable origin candidates to run in post-dispatch.",
    )
    parser.add_argument(
        "--origin-search-usable-count",
        type=int,
        help="How many usable origin candidates to report in origin_search mode.",
    )
    parser.add_argument(
        "--origin-search-debug-fallback-count",
        type=int,
        help="When no official candidate exists, how many debug candidates to dispatch/import.",
    )
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument(
        "--single-action",
        choices=("solve", "program", "visualize"),
    )
    parser.add_argument(
        "--online-action",
        choices=(
            "coordinator",
            "server",
            "receiver",
            "roundtrip",
            "worker_eval",
            "setup_server",
            "build_request",
        ),
        help=(
            "Compatibility option. 'roundtrip' now maps to the coordinator role; "
            "setup_server/build_request/worker_eval keep their old behavior."
        ),
    )
    parser.add_argument(
        "--online-role",
        choices=("coordinator", "server", "receiver"),
        help="Role-based online entry. Defaults to coordinator.",
    )
    parser.add_argument(
        "--local-profile",
        choices=LOCAL_PROFILE_CHOICES,
        help="Local coordinator/receiver machine profile. Defaults to LOCAL_MACHINE_PROFILE.",
    )
    parser.add_argument("--request")
    parser.add_argument("--handoff")
    parser.add_argument("--local-python")
    parser.add_argument(
        "--skip-final-generate",
        action="store_true",
        help="In online coordinator mode, skip the local receiver/RoboDK final generation step.",
    )
    parser.add_argument(
        "--allow-invalid-outputs",
        action="store_true",
        help="Write debug_* outputs even when the target-reachability delivery gate fails.",
    )
    parser.add_argument(
        "--remote-sync-mode",
        choices=("off", "guard", "push"),
        help="Online coordinator preflight sync policy: off | guard | push.",
    )
    parser.add_argument("--run-id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_mode = (
        "origin_search"
        if bool(args.origin_search or (ENABLE_TARGET_ORIGIN_YZ_SEARCH and not args.no_origin_search))
        else (args.mode or RUN_MODE)
    )
    single_action = (
        "visualize"
        if args.visualize
        else (args.single_action if args.single_action is not None else SINGLE_ACTION)
    )
    if run_mode == "single":
        action_name = single_action
    elif run_mode == "online":
        action_name = _resolve_online_command(args)
    elif run_mode == "origin_search":
        action_name = "origin_search"
    else:
        raise ValueError(f"Unsupported run mode: {run_mode}")
    run_id = (
        _resolve_run_id(args)
        if run_mode in {"online", "origin_search"}
        else (_resolve_single_run_id(args) if single_action != "visualize" else None)
    )
    if run_id is not None and run_mode in {"online", "origin_search"}:
        # Keep later artifact reporting on the same run id.  Without this, an
        # exception path that calls _resolve_run_id(args) can mint a fresh
        # timestamp and point the user at the wrong directory.
        args.run_id = run_id
    log_path = build_run_log_path(
        mode=run_mode,
        action=action_name,
        run_id=run_id,
        write_detailed_log_file=WRITE_DETAILED_LOG_FILE,
        run_log_dir=RUN_LOG_DIR,
    )
    descriptor = RunDescriptor(mode=run_mode, action=action_name, run_id=run_id, log_path=log_path)
    started_at = datetime.now()
    started_perf = perf_counter()
    header_settings = APP_RUNTIME_SETTINGS
    if run_mode == "single" and run_id is not None and single_action != "visualize":
        header_settings = _runtime_settings_with_tool_pose_csv(
            APP_RUNTIME_SETTINGS,
            tool_pose_csv=_single_run_tool_pose_csv(run_id),
        )
    elif run_mode == "online" and run_id is not None:
        header_settings = _runtime_settings_with_tool_pose_csv(
            APP_RUNTIME_SETTINGS,
            tool_pose_csv=_online_run_tool_pose_csv(run_id),
        )

    with tee_console_to_log(log_path):
        try:
            emit_lines(
                render_run_header_lines(
                    descriptor=descriptor,
                    settings=header_settings,
                    started_at=started_at,
                )
            )
            if run_mode == "single":
                result_paths, semantic_status = _run_single_mode(
                    single_action,
                    run_id=run_id,
                    allow_invalid_debug_outputs=_allow_invalid_debug_outputs(args),
                )
            elif run_mode == "online":
                result_paths, semantic_status = _run_online_mode(args)
            else:
                if run_id is None:
                    raise ValueError("run_id is required for origin_search mode.")
                result_paths, semantic_status = _run_origin_search_mode(
                    run_id=run_id,
                    args=args,
                )
            result_paths_with_log = dict(result_paths)
            if log_path is not None and "detail_log" not in result_paths_with_log:
                result_paths_with_log["detail_log"] = log_path
            print_result_paths("Run outputs:", result_paths_with_log)
            emit_lines(
                render_run_footer_lines(
                    status=semantic_status,
                    finished_at=datetime.now(),
                    duration_seconds=perf_counter() - started_perf,
                )
            )
        except KeyboardInterrupt:
            print("\nCancelled.")
            emit_lines(
                render_run_footer_lines(
                    status="cancelled",
                    finished_at=datetime.now(),
                    duration_seconds=perf_counter() - started_perf,
                )
            )
            if log_path is not None:
                print(f"Detailed log: {log_path}")
            return 1
        except Exception as exc:
            print(f"Error: {exc}")
            emit_lines(
                render_run_footer_lines(
                    status="failed",
                    finished_at=datetime.now(),
                    duration_seconds=perf_counter() - started_perf,
                    error=exc,
                )
            )
            if run_mode == "single":
                single_paths = _build_single_artifact_paths(run_id)
                print_result_paths(
                    "Available local outputs:",
                    {
                        **single_paths,
                        "detail_log": log_path,
                    },
                )
            elif run_mode == "online":
                print_result_paths(
                    "Available online outputs:",
                    {
                        **_build_online_artifact_paths(args),
                        "detail_log": log_path,
                    },
                )
            else:
                origin_run_dir = origin_search_run_dir(run_id or timestamp_token())
                print_result_paths(
                    "Available origin-search outputs:",
                    {
                        "run_dir": origin_run_dir,
                        "origin_search_result": origin_run_dir / "origin_yz_search_results.json",
                        "detail_log": log_path,
                    },
                )
            return 1

        if semantic_status == "invalid":
            print("Completed, but no target-reachable deliverable was produced.")
            return 2 if bool(STRICT_EXIT_ON_INVALID) else 0

        print("Done.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
