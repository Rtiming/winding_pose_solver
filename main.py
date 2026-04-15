from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from time import perf_counter

from app_settings import build_app_runtime_settings
from src.core.collab_models import EvaluationBatchRequest, load_json_file, write_json_file
from src.runtime.app import run_visualization
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


# ---------------------------------------------------------------------------
# Main business parameters
# Keep the most frequently changed project inputs here.
# Detailed optimizer and RoboDK tuning lives in `app_settings.py`.
# ---------------------------------------------------------------------------

VALIDATION_CENTERLINE_CSV = Path("data/validation_centerline.csv")
TOOL_POSES_FRAME2_CSV = Path("data/tool_poses_frame2.csv")

TARGET_FRAME_A_ORIGIN_IN_FRAME2_MM = (1126.0, -650.0, 1130.0)
TARGET_FRAME_A_ROTATION_IN_FRAME2_XYZ_DEG = (0, 0, -180.0)

ENABLE_CUSTOM_SMOOTHING_AND_POSE_SELECTION = True
ROBOT_NAME = "KUKA"
FRAME_NAME = "Frame 2"
PROGRAM_NAME = "Path_From_CSV"
ENABLE_LOCAL_MULTIPROCESS_PARALLEL = False
LOCAL_PARALLEL_WORKERS = 0  # 0 = auto
LOCAL_PARALLEL_MIN_BATCH_SIZE = 8

# IK 后端选择：
#   "robodk"      — 使用 RoboDK 内置 IK 求解器（原有行为）
#   "six_axis_ik" — 使用内置本地 POE 模型求解器（不需要 RoboDK 授权许可，多解枚举更全面）
IK_BACKEND = "six_axis_ik"


# ---------------------------------------------------------------------------
# Run mode configuration
# Edit these few variables, then run `python main.py`.
# The script will choose the correct local / online flow automatically.
# ---------------------------------------------------------------------------

RUN_MODE = "single"  # "single" | "online"
SINGLE_ACTION = "solve"  # "solve" | "program" | "visualize"
SINGLE_RESULT_ARCHIVE_DIR = Path("artifacts/local_runs")

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
ONLINE_SERVER_EVAL_WHEN_POSSIBLE = False
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
    ik_backend=IK_BACKEND,
    local_parallel_workers=(
        LOCAL_PARALLEL_WORKERS if ENABLE_LOCAL_MULTIPROCESS_PARALLEL else 1
    ),
    local_parallel_min_batch_size=LOCAL_PARALLEL_MIN_BATCH_SIZE,
)


def _resolve_run_id(args: argparse.Namespace | None = None) -> str:
    if args is not None and args.run_id:
        return args.run_id
    if ONLINE_RUN_ID:
        return ONLINE_RUN_ID
    return timestamp_token()


def _resolve_single_run_id(args: argparse.Namespace | None = None) -> str:
    if args is not None and args.run_id:
        return args.run_id
    return timestamp_token()


def _build_single_artifact_paths(run_id: str | None) -> dict[str, Path | None]:
    if run_id is None:
        return {
            "run_dir": None,
            "request": None,
            "eval_result": None,
            "archive": None,
            "joint_path_csv": None,
            "tool_pose_csv": APP_RUNTIME_SETTINGS.tool_poses_frame2_csv,
            "optimized_pose_csv": APP_RUNTIME_SETTINGS.tool_poses_frame2_csv.with_name(
                f"{APP_RUNTIME_SETTINGS.tool_poses_frame2_csv.stem}_optimized.csv"
            ),
        }
    run_dir = SINGLE_RESULT_ARCHIVE_DIR / run_id
    return {
        "run_dir": run_dir,
        "request": run_dir / "request.json",
        "eval_result": run_dir / "eval_result.json",
        "archive": run_dir / "run_archive.json",
        "joint_path_csv": run_dir / "selected_joint_path.csv",
        "tool_pose_csv": APP_RUNTIME_SETTINGS.tool_poses_frame2_csv,
        "optimized_pose_csv": APP_RUNTIME_SETTINGS.tool_poses_frame2_csv.with_name(
            f"{APP_RUNTIME_SETTINGS.tool_poses_frame2_csv.stem}_optimized.csv"
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
    result=None,
    runtime_error: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "single",
        "single_action": single_action,
        "create_program": bool(create_program),
        "status": "error" if runtime_error else "completed",
        "runtime_error": runtime_error,
        "basic_info": {
            "target_frame_a_origin_in_frame2_mm": tuple(float(v) for v in TARGET_FRAME_A_ORIGIN_IN_FRAME2_MM),
            "target_frame_a_rotation_in_frame2_xyz_deg": tuple(
                float(v) for v in TARGET_FRAME_A_ROTATION_IN_FRAME2_XYZ_DEG
            ),
            "ik_backend": IK_BACKEND,
            "robot_name": ROBOT_NAME,
            "frame_name": FRAME_NAME,
            "program_name": PROGRAM_NAME,
        },
        "artifacts": {
            label: (str(path) if isinstance(path, Path) else None)
            for label, path in artifact_paths.items()
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
            "worst_joint_step_deg": float(result.worst_joint_step_deg),
            "mean_joint_step_deg": float(result.mean_joint_step_deg),
            "total_cost": float(result.total_cost),
            "diagnostics": result.diagnostics,
            "error_message": result.error_message,
        }
    return payload


def _run_single_profile_evaluation(
    *,
    run_id: str,
    single_action: str,
    create_program: bool,
) -> dict[str, Path | None]:
    from src.runtime.request_builder import build_profile_evaluation_request
    from src.robodk_runtime.eval_worker import evaluate_batch_request

    artifact_paths = _build_single_artifact_paths(run_id)
    run_dir = artifact_paths["run_dir"]
    assert run_dir is not None
    run_dir.mkdir(parents=True, exist_ok=True)

    request = build_profile_evaluation_request(
        APP_RUNTIME_SETTINGS,
        request_id=f"single_{single_action}_{run_id}",
        strategy="full_search",
        refresh_csv=True,
        run_window_repair=True,
        run_inserted_repair=True,
        include_pose_rows_in_result=False,
        create_program=create_program,
        program_name=APP_RUNTIME_SETTINGS.program_name if create_program else None,
        optimized_csv_path=str(APP_RUNTIME_SETTINGS.tool_poses_frame2_csv),
        metadata={
            "entrypoint": "main_single",
            "single_action": single_action,
            "create_program": bool(create_program),
        },
    )
    write_json_file(artifact_paths["request"], request.to_dict())

    batch_request = EvaluationBatchRequest(evaluations=(request,))
    runtime_error: str | None = None
    batch_result = None
    result = None
    try:
        batch_result = evaluate_batch_request(batch_request)
        write_json_file(artifact_paths["eval_result"], batch_result.to_dict())
        if batch_result.results:
            result = batch_result.results[0]
            _write_selected_joint_path_csv(result, artifact_paths["joint_path_csv"])
            print(
                "[single-eval] "
                f"status={result.status}, "
                f"ik_empty_rows={result.ik_empty_row_count}, "
                f"config_switches={result.config_switches}, "
                f"bridge_like_segments={result.bridge_like_segments}, "
                f"worst_joint_step={result.worst_joint_step_deg}"
            )
    except Exception as exc:
        runtime_error = f"{type(exc).__name__}: {exc}"
        archive_payload = _build_single_archive_payload(
            run_id=run_id,
            single_action=single_action,
            create_program=create_program,
            artifact_paths=artifact_paths,
            result=result,
            runtime_error=runtime_error,
        )
        write_json_file(artifact_paths["archive"], archive_payload)
        raise

    archive_payload = _build_single_archive_payload(
        run_id=run_id,
        single_action=single_action,
        create_program=create_program,
        artifact_paths=artifact_paths,
        result=result,
        runtime_error=runtime_error,
    )
    write_json_file(artifact_paths["archive"], archive_payload)

    optimized_csv = artifact_paths["optimized_pose_csv"]
    if optimized_csv is not None and not optimized_csv.exists():
        optimized_csv = None

    return {
        "run_dir": run_dir,
        "request": artifact_paths["request"],
        "eval_result": artifact_paths["eval_result"] if artifact_paths["eval_result"].exists() else None,
        "archive": artifact_paths["archive"],
        "joint_path_csv": artifact_paths["joint_path_csv"] if artifact_paths["joint_path_csv"].exists() else None,
        "tool_pose_csv": artifact_paths["tool_pose_csv"],
        "optimized_pose_csv": optimized_csv,
    }


def _run_single_mode(
    single_action: str,
    *,
    run_id: str | None,
) -> dict[str, Path | None]:
    if single_action == "visualize":
        print("Running single-machine mode: visualization")
        run_visualization(APP_RUNTIME_SETTINGS)
        return {
            "tool_pose_csv": APP_RUNTIME_SETTINGS.tool_poses_frame2_csv,
            "optimized_pose_csv": None,
        }

    if run_id is None:
        raise ValueError("run_id is required for single solve/program actions.")

    if single_action == "program":
        print("Running single-machine mode: solve + RoboDK program generation")
        return _run_single_profile_evaluation(
            run_id=run_id,
            single_action=single_action,
            create_program=True,
        )

    if single_action == "solve":
        print("Running single-machine mode: solve only (skip RoboDK program generation)")
        return _run_single_profile_evaluation(
            run_id=run_id,
            single_action=single_action,
            create_program=False,
        )

    raise ValueError(f"Unsupported single action: {single_action}")


def _run_online_mode(args: argparse.Namespace | None = None) -> dict[str, Path | None]:
    from online_roundtrip import CommandLogOptions, run_round, run_worker_eval, setup_server

    online_action = args.online_action if args is not None and args.online_action else ONLINE_ACTION
    run_id = _resolve_run_id(args)
    request_path = _resolve_online_request_path(args)
    roundtrip_log_path = build_run_log_path(
        mode="online",
        action=online_action,
        run_id=run_id,
        write_detailed_log_file=WRITE_DETAILED_LOG_FILE,
        run_log_dir=RUN_LOG_DIR,
    )
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
        server_eval_when_possible=ONLINE_SERVER_EVAL_WHEN_POSSIBLE,
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
        "--single-action",
        choices=("solve", "program", "visualize"),
    )
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
    single_action = (
        "visualize"
        if args.visualize
        else (args.single_action if args.single_action is not None else SINGLE_ACTION)
    )
    action_name = single_action if run_mode == "single" else (args.online_action or ONLINE_ACTION)
    run_id = (
        _resolve_run_id(args)
        if run_mode == "online"
        else (_resolve_single_run_id(args) if single_action != "visualize" else None)
    )
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

    with tee_console_to_log(log_path):
        try:
            emit_lines(
                render_run_header_lines(
                    descriptor=descriptor,
                    settings=APP_RUNTIME_SETTINGS,
                    started_at=started_at,
                )
            )
            if run_mode == "single":
                result_paths = _run_single_mode(single_action, run_id=run_id)
            else:
                result_paths = _run_online_mode(args)
            result_paths_with_log = dict(result_paths)
            if log_path is not None and "detail_log" not in result_paths_with_log:
                result_paths_with_log["detail_log"] = log_path
            print_result_paths("Run outputs:", result_paths_with_log)
            emit_lines(
                render_run_footer_lines(
                    status="success",
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
            else:
                print_result_paths(
                    "Available online outputs:",
                    {
                        **_build_online_artifact_paths(args),
                        "detail_log": log_path,
                    },
                )
            return 1

        print("Done.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
