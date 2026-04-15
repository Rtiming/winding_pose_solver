from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.core.collab_models import (
    EvaluationBatchRequest,
    EvaluationBatchResult,
    ProfileEvaluationRequest,
    RemoteSearchRequest,
    load_json_file,
    write_json_file,
)
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
    final_request_path: Path | None
    final_result_path: Path | None
    optimized_pose_csv_path: Path | None
    final_program_name: str | None
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
        capture_output=True,
        check=False,
    )
    if result.stdout:
        _append_log(log_options.log_path, "STDOUT:\n" + result.stdout)
    if result.stderr:
        _append_log(log_options.log_path, "STDERR:\n" + result.stderr)

    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            args,
            output=result.stdout,
            stderr=result.stderr,
        )

    if echo_output:
        if result.stdout.strip():
            print(result.stdout.strip())
        if result.stderr.strip():
            print(result.stderr.strip())
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


def _find_request_by_id(
    batch_request: EvaluationBatchRequest,
    request_id: str,
) -> ProfileEvaluationRequest:
    for request in batch_request.evaluations:
        if request.request_id == request_id:
            return request
    raise KeyError(f"Request id not found in candidate batch: {request_id}")


def _find_result_by_id(
    batch_result: EvaluationBatchResult,
    request_id: str,
):
    for result in batch_result.results:
        if result.request_id == request_id:
            return result
    raise KeyError(f"Request id not found in evaluation results: {request_id}")


def _build_final_generation_request(
    candidate_request: ProfileEvaluationRequest,
    *,
    request_id: str,
    program_name: str | None,
    optimized_csv_path: str | None,
) -> ProfileEvaluationRequest:
    final_metadata = dict(candidate_request.metadata)
    final_metadata["entrypoint"] = "online_roundtrip_final_generate"
    # Final generation always runs the full repair pipeline regardless of whether
    # the batch candidate was evaluated with repairs disabled.
    return ProfileEvaluationRequest(
        request_id=request_id,
        robot_name=candidate_request.robot_name,
        frame_name=candidate_request.frame_name,
        motion_settings=dict(candidate_request.motion_settings),
        reference_pose_rows=tuple(dict(row) for row in candidate_request.reference_pose_rows),
        frame_a_origin_yz_profile_mm=tuple(candidate_request.frame_a_origin_yz_profile_mm),
        row_labels=tuple(candidate_request.row_labels),
        inserted_flags=tuple(candidate_request.inserted_flags),
        strategy=str(candidate_request.strategy),
        start_joints=candidate_request.start_joints,
        run_window_repair=True,
        run_inserted_repair=True,
        include_pose_rows_in_result=False,
        create_program=True,
        program_name=program_name or candidate_request.program_name,
        optimized_csv_path=optimized_csv_path or candidate_request.optimized_csv_path,
        metadata=final_metadata,
    )


def _can_run_server_side_ik_eval(batch_request: EvaluationBatchRequest) -> bool:
    return all(
        request.motion_settings.get("ik_backend") == "six_axis_ik" and not request.create_program
        for request in batch_request.evaluations
    )


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
        "[setup-server] Syncing requester code to server...",
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
                str(REPO_ROOT / "online_requester.py"),
                str(REPO_ROOT / "online_worker.py"),
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
    print(f"[setup-server] Done. Server requester ready at {server_dir}")
    if log_options.log_path is not None:
        print(f"[setup-server] Detailed log: {log_options.log_path}")


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
    log_options: CommandLogOptions | None = None,
) -> RoundtripArtifacts:
    local_run_dir = _local_artifact_dir(run_id)
    local_run_dir.mkdir(parents=True, exist_ok=True)
    effective_log_options = log_options or CommandLogOptions(
        log_path=local_run_dir / "roundtrip.log",
    )

    remote_request = _load_remote_request(request_path)
    local_request_path = write_json_file(local_run_dir / "request.json", remote_request.to_dict())
    local_candidates_path = local_run_dir / "candidates.json"
    local_results_path = local_run_dir / "results.json"
    local_summary_path = local_run_dir / "summary.json"
    local_final_request_path = local_run_dir / "final_generate_request.json"
    local_final_result_path = local_run_dir / "final_generate_result.json"

    remote_shell_dir = _remote_dir_for_shell(server_dir)
    remote_scp_dir = server_dir
    remote_run_dir = f"{remote_shell_dir}/artifacts/online_runs/{run_id}"
    remote_run_dir_scp = f"{remote_scp_dir}/artifacts/online_runs/{run_id}"
    remote_request_path = f"{remote_run_dir}/request.json"
    remote_candidates_path = f"{remote_run_dir}/candidates.json"
    remote_results_path = f"{remote_run_dir}/results.json"
    remote_summary_path = f"{remote_run_dir}/summary.json"
    remote_request_path_scp = f"{remote_run_dir_scp}/request.json"
    remote_candidates_path_scp = f"{remote_run_dir_scp}/candidates.json"
    remote_results_path_scp = f"{remote_run_dir_scp}/results.json"
    remote_summary_path_scp = f"{remote_run_dir_scp}/summary.json"

    _run_stage(
        "[roundtrip] Creating remote run directory...",
        lambda: _run_remote_bash(
            host,
            f'mkdir -p "{remote_run_dir}"',
            log_options=effective_log_options,
            echo_output=False,
        ),
        log_options=effective_log_options,
    )

    _run_stage(
        "[roundtrip] Uploading request to server...",
        lambda: _run_local_command(
            ["scp", str(local_request_path), f"{host}:{remote_request_path_scp}"],
            log_options=effective_log_options,
            echo_output=False,
        ),
        log_options=effective_log_options,
    )

    remote_propose_script = f"""
set -e
cd "{remote_shell_dir}"
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "{env_name}"
python online_requester.py propose --request "{remote_request_path}" --candidates "{remote_candidates_path}"
"""
    _run_stage(
        "[roundtrip] Server generating candidate batch...",
        lambda: _run_remote_bash(
            host,
            remote_propose_script,
            log_options=effective_log_options,
            echo_output=False,
        ),
        log_options=effective_log_options,
    )

    _run_stage(
        "[roundtrip] Downloading candidate batch...",
        lambda: _run_local_command(
            ["scp", f"{host}:{remote_candidates_path_scp}", str(local_candidates_path)],
            log_options=effective_log_options,
            echo_output=False,
        ),
        log_options=effective_log_options,
    )

    candidate_batch = EvaluationBatchRequest.from_dict(load_json_file(local_candidates_path))
    use_server_side_ik_eval = (
        server_eval_when_possible and _can_run_server_side_ik_eval(candidate_batch)
    )

    if use_server_side_ik_eval:
        print(
            "[roundtrip] Candidate batch is eligible for server-side SixAxisIK evaluation; "
            "skipping the local RoboDK worker for this stage."
        )
        remote_eval_script = f"""
set -e
cd "{remote_shell_dir}"
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "{env_name}"
python online_worker.py eval-batch --request "{remote_candidates_path}" --result "{remote_results_path}"
"""
        _run_stage(
            "[roundtrip] Server evaluating candidates with offline SixAxisIK...",
            lambda: _run_remote_bash(
                host,
                remote_eval_script,
                log_options=effective_log_options,
                echo_output=False,
            ),
            log_options=effective_log_options,
        )
        _run_stage(
            "[roundtrip] Downloading evaluation results...",
            lambda: _run_local_command(
                ["scp", f"{host}:{remote_results_path_scp}", str(local_results_path)],
                log_options=effective_log_options,
                echo_output=False,
            ),
            log_options=effective_log_options,
        )
    else:
        worker_python = resolve_local_worker_python(local_python)
        _run_stage(
            "[roundtrip] Local RoboDK worker evaluating candidates...",
            lambda: _run_local_command(
                [
                    worker_python,
                    str(REPO_ROOT / "online_worker.py"),
                    "eval-batch",
                    "--request",
                    str(local_candidates_path),
                    "--result",
                    str(local_results_path),
                ],
                log_options=effective_log_options,
                echo_output=effective_log_options.show_worker_output,
            ),
            log_options=effective_log_options,
        )

        _run_stage(
            "[roundtrip] Uploading evaluation results to server...",
            lambda: _run_local_command(
                ["scp", str(local_results_path), f"{host}:{remote_results_path_scp}"],
                log_options=effective_log_options,
                echo_output=False,
            ),
            log_options=effective_log_options,
        )

    remote_summarize_script = f"""
set -e
cd "{remote_shell_dir}"
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "{env_name}"
python online_requester.py summarize --results "{remote_results_path}" --summary "{remote_summary_path}"
"""
    _run_stage(
        "[roundtrip] Server summarizing results...",
        lambda: _run_remote_bash(
            host,
            remote_summarize_script,
            log_options=effective_log_options,
            echo_output=False,
        ),
        log_options=effective_log_options,
    )

    _run_stage(
        "[roundtrip] Downloading summary...",
        lambda: _run_local_command(
            ["scp", f"{host}:{remote_summary_path_scp}", str(local_summary_path)],
            log_options=effective_log_options,
            echo_output=False,
        ),
        log_options=effective_log_options,
    )

    summary_payload = load_json_file(local_summary_path)
    conclusion = summary_payload.get("conclusion")
    if conclusion:
        print(f"[roundtrip] {conclusion}")

    final_request_path: Path | None = None
    final_result_path: Path | None = None
    optimized_pose_csv_output: Path | None = None
    final_program_name_output: str | None = None

    if generate_final_program:
        best_request_id = summary_payload.get("best_request_id")
        if not best_request_id:
            raise RuntimeError(
                f"[final-generate] No best_request_id found in summary: {local_summary_path}"
            )

        candidate_batch = EvaluationBatchRequest.from_dict(load_json_file(local_candidates_path))
        candidate_request = _find_request_by_id(candidate_batch, str(best_request_id))
        batch_result = EvaluationBatchResult.from_dict(load_json_file(local_results_path))
        best_result = _find_result_by_id(batch_result, str(best_request_id))

        print(
            "[final-generate] Best candidate selected: "
            f"{best_request_id} "
            f"(status={best_result.status}, "
            f"ik_empty_rows={best_result.ik_empty_row_count}, "
            f"config_switches={best_result.config_switches}, "
            f"bridge_like_segments={best_result.bridge_like_segments})"
        )

        final_request = _build_final_generation_request(
            candidate_request,
            request_id=f"{best_request_id}_final_generate",
            program_name=final_program_name,
            optimized_csv_path=optimized_csv_path,
        )
        final_request_path = write_json_file(local_final_request_path, final_request.to_dict())
        final_result_path = local_final_result_path

        _run_stage(
            "[final-generate] Local RoboDK worker validating best candidate and generating program...",
            lambda: _run_local_command(
                [
                    resolve_local_worker_python(local_python),
                    str(REPO_ROOT / "online_worker.py"),
                    "eval",
                    "--request",
                    str(final_request_path),
                    "--result",
                    str(final_result_path),
                ],
                log_options=effective_log_options,
                echo_output=effective_log_options.show_worker_output,
            ),
            log_options=effective_log_options,
        )

        final_result_payload = load_json_file(final_result_path)
        final_results = final_result_payload.get("results", [])
        if not final_results:
            raise RuntimeError(
                f"[final-generate] No result entries were written: {final_result_path}"
            )

        final_result = final_results[0]
        final_program_name_output = (
            final_result.get("metadata", {}).get("program_name")
            if isinstance(final_result.get("metadata"), dict)
            else None
        )
        if optimized_csv_path:
            optimized_pose_csv_output = Path(optimized_csv_path).with_name(
                f"{Path(optimized_csv_path).stem}_optimized.csv"
            )
        elif candidate_request.optimized_csv_path:
            optimized_pose_csv_output = Path(candidate_request.optimized_csv_path).with_name(
                f"{Path(candidate_request.optimized_csv_path).stem}_optimized.csv"
            )

        if final_result.get("status") != "valid" or not final_program_name_output:
            diagnostics = final_result.get("diagnostics") or final_result.get("error_message") or "unknown final-generation failure"
            raise RuntimeError(
                "[final-generate] Best candidate could not generate a final RoboDK program. "
                f"summary={local_summary_path}, final_result={final_result_path}. "
                f"Reason: {diagnostics}"
            )

        print(
            "[final-generate] Program generated successfully: "
            f"{final_program_name_output}"
        )

    print(f"[roundtrip] Artifacts: {local_run_dir}")
    if effective_log_options.log_path is not None:
        print(f"[roundtrip] Detailed log: {effective_log_options.log_path}")

    return RoundtripArtifacts(
        run_id=run_id,
        local_run_dir=local_run_dir,
        request_path=local_request_path,
        candidates_path=local_candidates_path,
        results_path=local_results_path,
        summary_path=local_summary_path,
        final_request_path=final_request_path,
        final_result_path=final_result_path,
        optimized_pose_csv_path=optimized_pose_csv_output,
        final_program_name=final_program_name_output,
        log_path=effective_log_options.log_path,
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Set up the master requester environment and run online requester/worker round trips."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup_parser = subparsers.add_parser("setup-server", help="Sync requester code and create the server conda environment.")
    setup_parser.add_argument("--host", default="master")
    setup_parser.add_argument("--server-dir", default=DEFAULT_SERVER_DIR)
    setup_parser.add_argument("--env", default=DEFAULT_ENV_NAME)
    setup_parser.add_argument("--log-file")
    setup_parser.add_argument("--show-command-details", action="store_true")

    build_parser = subparsers.add_parser("build-request", help="Build a round-1 RemoteSearchRequest from current local settings.")
    build_parser.add_argument("--request", required=True)
    build_parser.add_argument("--round-index", type=int, default=1)
    build_parser.add_argument("--candidate-limit", type=int, default=8)
    build_parser.add_argument("--skip-pose-solver", action="store_true")

    round_parser = subparsers.add_parser("run-round", help="Run one requester/worker round trip.")
    round_parser.add_argument("--host", default="master")
    round_parser.add_argument("--server-dir", default=DEFAULT_SERVER_DIR)
    round_parser.add_argument("--env", default=DEFAULT_ENV_NAME)
    round_parser.add_argument("--request", required=True)
    round_parser.add_argument("--run-id", default=_default_run_id())
    round_parser.add_argument("--local-python")
    round_parser.add_argument("--log-file")
    round_parser.add_argument("--show-command-details", action="store_true")
    round_parser.add_argument("--quiet-worker-output", action="store_true")
    round_parser.add_argument("--server-eval-when-possible", action="store_true")
    round_parser.add_argument("--skip-final-generate", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "setup-server":
            setup_server(
                host=args.host,
                server_dir=args.server_dir,
                env_name=args.env,
                log_options=CommandLogOptions(
                    log_path=None if args.log_file is None else Path(args.log_file),
                    show_command_details=bool(args.show_command_details),
                    show_worker_output=True,
                ),
            )
        elif args.command == "build-request":
            from main import APP_RUNTIME_SETTINGS

            output_path = build_request_file(
                APP_RUNTIME_SETTINGS,
                args.request,
                round_index=args.round_index,
                candidate_limit=args.candidate_limit,
                refresh_csv=not args.skip_pose_solver,
            )
            print(f"Wrote request: {output_path}")
        else:
            run_round(
                host=args.host,
                server_dir=args.server_dir,
                env_name=args.env,
                request_path=args.request,
                run_id=args.run_id,
                local_python=args.local_python,
                server_eval_when_possible=bool(args.server_eval_when_possible),
                generate_final_program=not bool(args.skip_final_generate),
                log_options=CommandLogOptions(
                    log_path=None if args.log_file is None else Path(args.log_file),
                    show_command_details=bool(args.show_command_details),
                    show_worker_output=not bool(args.quiet_worker_output),
                ),
            )
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
