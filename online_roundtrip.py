from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
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
    result_has_continuity_warnings,
    result_is_strictly_valid,
    result_payload_is_strictly_valid,
    write_handoff_package,
)
from src.runtime.remote_search import propose_candidates, summarize_results
from src.runtime.request_builder import build_profile_evaluation_request, build_remote_search_request


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SERVER_DIR = "~/apps/winding_pose_solver"
DEFAULT_ENV_NAME = "winding_pose_solver"


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
        errors="replace",
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
            print(stdout_text.strip())
        if stderr_text.strip():
            print(stderr_text.strip())
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


def _summarize_subprocess_failure(exc: subprocess.CalledProcessError) -> str:
    stderr_text = (exc.stderr or "").strip()
    stdout_text = (exc.output or "").strip()
    for text in (stderr_text, stdout_text):
        if text:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if lines:
                return lines[-1]
    return f"command failed with exit code {exc.returncode}"


def _run_stage(
    stage_name: str,
    func,
    *,
    log_options: CommandLogOptions,
):
    print(stage_name)
    _append_log(log_options.log_path, stage_name)
    try:
        return func()
    except subprocess.CalledProcessError as exc:
        summary = _summarize_subprocess_failure(exc)
        raise RuntimeError(f"{stage_name} failed: {summary}") from exc


def _can_import_module(python_executable: str, module_name: str) -> bool:
    try:
        subprocess.run(
            [python_executable, "-c", f"import {module_name}"],
            check=True,
            text=True,
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
) -> Path:
    base_request = build_profile_evaluation_request(
        settings,
        request_id=f"round{round_index}_base",
        strategy="full_search",
        refresh_csv=refresh_csv,
        include_pose_rows_in_result=False,
        create_program=False,
        program_name=settings.program_name,
        optimized_csv_path=str(settings.tool_poses_frame2_csv),
        metadata={"entrypoint": "online_roundtrip"},
    )
    remote_request = build_remote_search_request(
        base_request,
        round_index=round_index,
        candidate_limit=candidate_limit,
        metadata={"entrypoint": "online_roundtrip"},
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
        float(result.config_switches),
        float(result.bridge_like_segments),
        float(result.worst_joint_step_deg),
        float(result.mean_joint_step_deg),
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
    metadata["source_request_id"] = str(selected_result.request_id)
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
        reference_pose_rows=tuple(dict(row) for row in base_request.reference_pose_rows),
        frame_a_origin_yz_profile_mm=tuple(
            (float(dy_mm), float(dz_mm))
            for dy_mm, dz_mm in selected_result.frame_a_origin_yz_profile_mm
        ),
        row_labels=tuple(str(label) for label in selected_result.row_labels or base_request.row_labels),
        inserted_flags=tuple(bool(flag) for flag in selected_result.inserted_flags or base_request.inserted_flags),
        strategy="exact_profile",
        start_joints=base_request.start_joints,
        run_window_repair=True,
        run_inserted_repair=True,
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
    log_options: CommandLogOptions | None = None,
) -> None:
    log_options = log_options or CommandLogOptions()
    remote_shell_dir = _remote_dir_for_shell(server_dir)

    _run_stage(
        "[setup-server] Ensuring remote project directory exists...",
        lambda: _run_remote_bash(
            host,
            f'mkdir -p "{remote_shell_dir}"',
            log_options=log_options,
            echo_output=False,
        ),
        log_options=log_options,
    )
    _run_stage(
        "[setup-server] Syncing source tree to server...",
        lambda: _run_local_command(
            ["scp", "-r", str(REPO_ROOT / "src"), f"{host}:{server_dir}/"],
            log_options=log_options,
            echo_output=False,
        ),
        log_options=log_options,
    )
    _run_stage(
        "[setup-server] Uploading entrypoints and dependency files...",
        lambda: _run_local_command(
            [
                "scp",
                str(REPO_ROOT / "online_roundtrip.py"),
                str(REPO_ROOT / "online_requester.py"),
                str(REPO_ROOT / "online_worker.py"),
                str(REPO_ROOT / "app_settings.py"),
                str(REPO_ROOT / "requirements.shared.txt"),
                str(REPO_ROOT / "requirements.server.txt"),
                str(REPO_ROOT / "environment.server.yml"),
                str(REPO_ROOT / "README.md"),
                f"{host}:{server_dir}/",
            ],
            log_options=log_options,
            echo_output=False,
        ),
        log_options=log_options,
    )

    remote_setup_script = f"""
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
python - <<'PY'
import importlib.util
spec = importlib.util.find_spec("robodk")
print("robodk_spec", spec)
if spec is not None:
    raise SystemExit("Server environment unexpectedly contains robodk")
PY
"""
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
    retry_repair_limit: int = 1,
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

    def _evaluate_candidates() -> EvaluationBatchResult:
        from src.robodk_runtime.eval_worker import evaluate_batch_request

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
            f"bridge_like_segments={selected_result.bridge_like_segments})"
        )

    retry_summary_path: Path | None = None
    if selected_result is not None and (
        not result_is_strictly_valid(selected_result)
        or result_has_continuity_warnings(selected_result)
    ):
        def _retry():
            from src.runtime.local_retry import run_local_profile_retry

            return run_local_profile_retry(
                remote_request.base_request,
                selected_result,
                candidate_limit=max(1, int(retry_candidate_limit)),
                repair_retry_limit=max(1, int(retry_repair_limit)),
                max_rounds=max(1, int(retry_max_rounds)),
                baseline_search_result=None,
            )

        retry_outcome = _run_stage(
            "[online/server] Running local profile retry/repair...",
            _retry,
            log_options=effective_log_options,
        )
        retry_summary_path = write_json_file(
            profile_retry_summary_path,
            {
                **retry_outcome.payload,
                "budget": {
                    "candidate_limit": int(retry_candidate_limit),
                    "repair_retry_limit": int(retry_repair_limit),
                    "max_rounds": int(retry_max_rounds),
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

    strict_ready = bool(selected_result is not None and result_is_strictly_valid(selected_result))
    summary_payload: dict[str, Any] = summary.to_dict()
    summary_payload["server_role"] = {
        "status": "valid" if strict_ready else "invalid",
        "delivery_ready": strict_ready,
        "strict_delivery_ready": strict_ready,
        "selected_request_id": best_request_id,
    }
    summary_payload["server_final_selection"] = (
        None
        if selected_result is None
        else {
            "request_id": str(selected_result.request_id),
            "status": str(selected_result.status),
            "ik_empty_row_count": int(selected_result.ik_empty_row_count),
            "config_switches": int(selected_result.config_switches),
            "bridge_like_segments": int(selected_result.bridge_like_segments),
            "worst_joint_step_deg": float(selected_result.worst_joint_step_deg),
            "objective_reachable": strict_ready,
            "official_delivery_allowed": strict_ready,
            "strictly_valid": strict_ready,
        }
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

    receiver_request_path = write_json_file(
        run_dir / "receiver_request.json",
        receiver_request.to_dict(),
    )
    result_path = run_dir / "final_generate_result.json"
    worker_python = resolve_local_worker_python(local_python)

    _run_stage(
        "[online/receiver] Local RoboDK worker validating handoff and generating program...",
        lambda: _run_local_command(
            [
                worker_python,
                str(REPO_ROOT / "online_worker.py"),
                "eval",
                "--request",
                str(receiver_request_path),
                "--result",
                str(result_path),
            ],
            log_options=effective_log_options,
            echo_output=effective_log_options.show_worker_output,
        ),
        log_options=effective_log_options,
    )

    result_payload = load_json_file(result_path)
    final_results = result_payload.get("results", [])
    if not final_results:
        raise RuntimeError(f"[online/receiver] No result entries were written: {result_path}")

    final_result = final_results[0]
    materialized_program_name = (
        final_result.get("metadata", {}).get("program_name")
        if isinstance(final_result.get("metadata"), dict)
        else None
    )
    receiver_valid = result_payload_is_strictly_valid(final_result) and bool(materialized_program_name)
    optimized_pose_csv_output: Path | None = None
    if receiver_valid and receiver_request.optimized_csv_path:
        candidate_csv = Path(receiver_request.optimized_csv_path).with_name(
            f"{Path(receiver_request.optimized_csv_path).stem}_optimized.csv"
        )
        optimized_pose_csv_output = candidate_csv if candidate_csv.exists() else None

    if receiver_valid:
        print(f"[online/receiver] Program generated successfully: {materialized_program_name}")
        final_program_name = materialized_program_name
        result_status = "valid"
    else:
        diagnostics = (
            final_result.get("diagnostics")
            or final_result.get("error_message")
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
    try:
        _run_local_command(
            ["scp", f"{host}:{remote_path}", str(local_path)],
            log_options=log_options,
            echo_output=False,
        )
    except subprocess.CalledProcessError:
        _append_log(log_options.log_path, f"Optional remote artifact missing: {remote_path}")
        return None
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
    retry_candidate_limit: int = 4,
    retry_repair_limit: int = 1,
    retry_max_rounds: int = 1,
    allow_invalid_outputs: bool = False,
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
    remote_run_dir = f"{remote_shell_dir}/artifacts/online_runs/{run_id}"
    remote_run_dir_scp = f"{server_dir}/artifacts/online_runs/{run_id}"
    remote_request_arg = f"artifacts/online_runs/{run_id}/request.json"
    remote_log_arg = f"artifacts/online_runs/{run_id}/server_role.log"

    _run_stage(
        "[online/coordinator] Creating remote run directory...",
        lambda: _run_remote_bash(
            host,
            f'mkdir -p "{remote_run_dir}"',
            log_options=effective_log_options,
            echo_output=False,
        ),
        log_options=effective_log_options,
    )
    _run_stage(
        "[online/coordinator] Uploading request to server...",
        lambda: _run_local_command(
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
        str(max(1, int(retry_candidate_limit))),
        "--retry-repair-limit",
        str(max(1, int(retry_repair_limit))),
        "--retry-max-rounds",
        str(max(1, int(retry_max_rounds))),
        "--log-file",
        remote_log_arg,
    ]
    if allow_invalid_outputs:
        remote_server_args.append("--allow-invalid-outputs")
    if final_program_name:
        remote_server_args.extend(["--program-name", final_program_name])
    if optimized_csv_path:
        remote_server_args.extend(["--optimized-csv-path", optimized_csv_path])

    remote_server_script = f"""
set -e
cd "{remote_shell_dir}"
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "{env_name}"
set +e
{_render_command(remote_server_args)}
ROLE_STATUS=$?
set -e
if [ "$ROLE_STATUS" -ne 0 ] && [ "$ROLE_STATUS" -ne 2 ]; then
    exit "$ROLE_STATUS"
fi
exit 0
"""
    _run_stage(
        "[online/coordinator] Server running compute role...",
        lambda: _run_remote_bash(
            host,
            remote_server_script,
            log_options=effective_log_options,
            echo_output=effective_log_options.show_worker_output,
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
    allow_invalid_outputs: bool = False,
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
        allow_invalid_outputs=allow_invalid_outputs,
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
    parser.add_argument("--retry-candidate-limit", type=int, default=4)
    parser.add_argument("--retry-repair-limit", type=int, default=1)
    parser.add_argument("--retry-max-rounds", type=int, default=1)
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
    server_parser.add_argument("--retry-candidate-limit", type=int, default=4)
    server_parser.add_argument("--retry-repair-limit", type=int, default=1)
    server_parser.add_argument("--retry-max-rounds", type=int, default=1)
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
                allow_invalid_outputs=bool(args.allow_invalid_outputs),
                log_options=log_options,
            )
        else:
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
                log_options=log_options,
            )
        return 0 if artifacts.result_status == "valid" else 2
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
