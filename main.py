from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

from app_settings import build_app_runtime_settings
from src.app_runner import run_robodk_program_generation, run_visualization
from src.collab_models import load_json_file


# ---------------------------------------------------------------------------
# Main business parameters
# Keep the most frequently changed project inputs here.
# Detailed optimizer and RoboDK tuning lives in `app_settings.py`.
# ---------------------------------------------------------------------------

VALIDATION_CENTERLINE_CSV = Path("data/validation_centerline.csv")
TOOL_POSES_FRAME2_CSV = Path("data/tool_poses_frame2.csv")

TARGET_FRAME_A_ORIGIN_IN_FRAME2_MM = (1126.0, -400.0, 1200.0)
TARGET_FRAME_A_ROTATION_IN_FRAME2_XYZ_DEG = (-180.0, -14.0, -180.0)

ENABLE_CUSTOM_SMOOTHING_AND_POSE_SELECTION = True
ROBOT_NAME = "KUKA"
FRAME_NAME = "Frame 2"
PROGRAM_NAME = "Path_From_CSV"


# ---------------------------------------------------------------------------
# Run mode configuration
# Edit these few variables, then run `python main.py`.
# The script will choose the correct local / online flow automatically.
# ---------------------------------------------------------------------------

RUN_MODE = "online"  # "single" | "online"
SINGLE_ACTION = "program"  # "program" | "visualize"

ONLINE_ACTION = "roundtrip"  # "roundtrip" | "worker_eval" | "setup_server" | "build_request"
ONLINE_REQUEST_SOURCE = "build"  # "build" | "existing"
ONLINE_REQUEST_PATH = Path("artifacts/online_runs/main_request.json")
ONLINE_RUN_ID: str | None = None
ONLINE_HOST = "master"
ONLINE_SERVER_DIR = "~/apps/winding_pose_solver"
ONLINE_ENV_NAME = "winding_pose_solver"
ONLINE_LOCAL_PYTHON: str | None = None
ONLINE_ROUND_INDEX = 1
ONLINE_CANDIDATE_LIMIT = 4
ONLINE_SKIP_POSE_SOLVER_WHEN_BUILDING_REQUEST = False
ONLINE_AUTO_SETUP_SERVER = False
ONLINE_FINAL_GENERATE_PROGRAM = True

WRITE_DETAILED_LOG_FILE = True
SHOW_DETAILED_TERMINAL_LOGS = True
SHOW_COMMAND_DETAILS = False
RUN_LOG_DIR = Path("artifacts/run_logs")


APP_RUNTIME_SETTINGS = build_app_runtime_settings(
    validation_centerline_csv=VALIDATION_CENTERLINE_CSV,
    tool_poses_frame2_csv=TOOL_POSES_FRAME2_CSV,
    target_frame_origin_mm=TARGET_FRAME_A_ORIGIN_IN_FRAME2_MM,
    target_frame_rotation_xyz_deg=TARGET_FRAME_A_ROTATION_IN_FRAME2_XYZ_DEG,
    enable_custom_smoothing_and_pose_selection=ENABLE_CUSTOM_SMOOTHING_AND_POSE_SELECTION,
    robot_name=ROBOT_NAME,
    frame_name=FRAME_NAME,
    program_name=PROGRAM_NAME,
)


class _TeeWriter:
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
def _tee_console_to_log(log_path: Path | None):
    if log_path is None:
        yield
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        stdout_tee = _TeeWriter(sys.stdout, log_handle)
        stderr_tee = _TeeWriter(sys.stderr, log_handle)
        with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee):
            print(f"[log] Writing detailed log to {log_path}")
            yield


def _timestamp_token() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolve_run_id(args: argparse.Namespace | None = None) -> str:
    if args is not None and args.run_id:
        return args.run_id
    if ONLINE_RUN_ID:
        return ONLINE_RUN_ID
    return _timestamp_token()


def _build_run_log_path(*, mode: str, action: str, run_id: str | None = None) -> Path | None:
    if not WRITE_DETAILED_LOG_FILE:
        return None
    if mode == "online" and run_id is not None:
        return Path("artifacts/online_runs") / run_id / "main_run.log"
    return RUN_LOG_DIR / f"{mode}_{action}_{_timestamp_token()}.log"


def _print_result_paths(title: str, paths: dict[str, Path | None]) -> None:
    print(title)
    for label, path in paths.items():
        if path is None:
            continue
        print(f"  {label}: {path}")


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
        f"worst_joint_step={result.get('worst_joint_step_deg')}"
    )


def _resolve_online_request_path(args: argparse.Namespace | None = None) -> Path:
    if args is not None and args.request:
        return Path(args.request)
    return ONLINE_REQUEST_PATH


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
        "tool_pose_csv": APP_RUNTIME_SETTINGS.tool_poses_frame2_csv,
    }


def _prepare_online_request(
    *,
    request_path: Path,
) -> Path:
    from online_roundtrip import build_request_file

    if ONLINE_REQUEST_SOURCE == "existing":
        if not request_path.is_file():
            raise FileNotFoundError(
                f"Configured existing request file was not found: {request_path}"
            )
        print(f"[online] Using existing request: {request_path}")
        return request_path

    request_path.parent.mkdir(parents=True, exist_ok=True)
    built_request_path = build_request_file(
        APP_RUNTIME_SETTINGS,
        request_path,
        round_index=ONLINE_ROUND_INDEX,
        candidate_limit=ONLINE_CANDIDATE_LIMIT,
        refresh_csv=not ONLINE_SKIP_POSE_SOLVER_WHEN_BUILDING_REQUEST,
    )
    print(f"[online] Built request: {built_request_path}")
    return built_request_path


def _run_single_mode(single_action: str) -> dict[str, Path | None]:
    if single_action == "visualize":
        print("Running single-machine mode: visualization")
        run_visualization(APP_RUNTIME_SETTINGS)
        return {
            "tool_pose_csv": APP_RUNTIME_SETTINGS.tool_poses_frame2_csv,
            "optimized_pose_csv": None,
        }

    print("Running single-machine mode: full solve + RoboDK program generation")
    run_robodk_program_generation(APP_RUNTIME_SETTINGS)
    optimized_csv = APP_RUNTIME_SETTINGS.tool_poses_frame2_csv.with_name(
        f"{APP_RUNTIME_SETTINGS.tool_poses_frame2_csv.stem}_optimized.csv"
    )
    return {
        "tool_pose_csv": APP_RUNTIME_SETTINGS.tool_poses_frame2_csv,
        "optimized_pose_csv": optimized_csv if optimized_csv.exists() else None,
    }


def _run_online_mode(args: argparse.Namespace | None = None) -> dict[str, Path | None]:
    from online_roundtrip import CommandLogOptions, run_round, run_worker_eval, setup_server

    online_action = args.online_action if args is not None and args.online_action else ONLINE_ACTION
    run_id = _resolve_run_id(args)
    request_path = _resolve_online_request_path(args)
    roundtrip_log_path = _build_run_log_path(mode="online", action=online_action, run_id=run_id)
    command_log_options = CommandLogOptions(
        log_path=roundtrip_log_path,
        show_command_details=SHOW_COMMAND_DETAILS,
        show_worker_output=SHOW_DETAILED_TERMINAL_LOGS,
    )

    if online_action == "setup_server":
        print("Running online mode: setup_server")
        setup_server(
            host=ONLINE_HOST,
            server_dir=ONLINE_SERVER_DIR,
            env_name=ONLINE_ENV_NAME,
            log_options=command_log_options,
        )
        return {"detail_log": roundtrip_log_path}

    prepared_request_path = _prepare_online_request(request_path=request_path)

    if online_action == "build_request":
        print("Running online mode: build_request")
        return {
            "request": prepared_request_path,
            "tool_pose_csv": APP_RUNTIME_SETTINGS.tool_poses_frame2_csv,
            "detail_log": roundtrip_log_path,
        }

    if online_action == "worker_eval":
        print("Running online mode: local worker_eval")
        result_path = Path("artifacts/online_runs") / run_id / "eval_result.json"
        artifacts = run_worker_eval(
            request_path=prepared_request_path,
            result_path=result_path,
            local_python=ONLINE_LOCAL_PYTHON,
            log_options=command_log_options,
        )
        _summarize_worker_eval_result(artifacts.result_path)
        return {
            "request": artifacts.request_path,
            "eval_result": artifacts.result_path,
            "tool_pose_csv": APP_RUNTIME_SETTINGS.tool_poses_frame2_csv,
            "detail_log": artifacts.log_path,
        }

    print("Running online mode: requester + local worker + server summarize")
    if ONLINE_AUTO_SETUP_SERVER:
        print("[online] Auto setup enabled; preparing server first...")
        setup_server(
            host=ONLINE_HOST,
            server_dir=ONLINE_SERVER_DIR,
            env_name=ONLINE_ENV_NAME,
            log_options=command_log_options,
        )

    artifacts = run_round(
        host=ONLINE_HOST,
        server_dir=ONLINE_SERVER_DIR,
        env_name=ONLINE_ENV_NAME,
        request_path=prepared_request_path,
        run_id=run_id,
        local_python=ONLINE_LOCAL_PYTHON,
        generate_final_program=ONLINE_FINAL_GENERATE_PROGRAM,
        final_program_name=APP_RUNTIME_SETTINGS.program_name,
        optimized_csv_path=str(APP_RUNTIME_SETTINGS.tool_poses_frame2_csv),
        log_options=command_log_options,
    )
    if artifacts.final_program_name:
        print(f"[online] Final program: {artifacts.final_program_name}")
    return {
        "request": artifacts.request_path,
        "candidates": artifacts.candidates_path,
        "eval_result": artifacts.results_path,
        "summary": artifacts.summary_path,
        "final_request": artifacts.final_request_path,
        "final_eval_result": artifacts.final_result_path,
        "optimized_pose_csv": artifacts.optimized_pose_csv_path,
        "tool_pose_csv": APP_RUNTIME_SETTINGS.tool_poses_frame2_csv,
        "detail_log": artifacts.log_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unified local entrypoint for single-machine and online requester/worker flows. "
            "By default, the top-level config in main.py controls the run."
        )
    )
    parser.add_argument("--mode", choices=("single", "online"))
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument(
        "--online-action",
        choices=("roundtrip", "worker_eval", "setup_server", "build_request"),
    )
    parser.add_argument("--request")
    parser.add_argument("--run-id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_mode = args.mode or RUN_MODE
    single_action = "visualize" if args.visualize else SINGLE_ACTION
    run_id = _resolve_run_id(args) if run_mode == "online" else None
    log_path = _build_run_log_path(
        mode=run_mode,
        action=single_action if run_mode == "single" else (args.online_action or ONLINE_ACTION),
        run_id=run_id,
    )

    try:
        with _tee_console_to_log(log_path):
            if run_mode == "single":
                result_paths = _run_single_mode(single_action)
            else:
                result_paths = _run_online_mode(args)
    except KeyboardInterrupt:
        print("\nCancelled.")
        if log_path is not None:
            print(f"Detailed log: {log_path}")
        return 1
    except Exception as exc:
        print(f"Error: {exc}")
        if run_mode == "single":
            _print_result_paths(
                "Available local outputs:",
                {
                    "tool_pose_csv": APP_RUNTIME_SETTINGS.tool_poses_frame2_csv,
                    "detail_log": log_path,
                },
            )
        else:
            _print_result_paths(
                "Available online outputs:",
                {
                    **_build_online_artifact_paths(args),
                    "detail_log": log_path,
                },
            )
        return 1

    _print_result_paths("Run outputs:", result_paths)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
