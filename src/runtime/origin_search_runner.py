from __future__ import annotations

import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OriginSearchSettings:
    target_frame_a_origin_in_frame2_mm: tuple[float, float, float]
    use_server: bool = True
    square_size_mm: float = 300.0
    initial_step_mm: float = 75.0
    min_step_mm: float = 10.0
    max_iters: int = 5
    beam_width: int = 4
    diagonal_policy: str = "conditional"
    polish_step_mm: float = 0.0
    workers: int = 16
    strategy: str = "full_search"
    run_window_repair: bool = False
    run_inserted_repair: bool = False
    validation_grid_step_mm: float = 0.0
    top_k: int = 12
    min_separation_mm: float = 20.0
    outside_fallback_count: int = 3
    outside_fallback_max_rings: int = 6
    outside_fallback_ring_step_mm: float = 25.0
    outside_fallback_edge_step_mm: float = 75.0
    remote_sync_mode: str = "guard"
    enforce_remote_sync_guard: bool = True


@dataclass(frozen=True)
class OriginSearchServerSettings:
    host: str = "master"
    server_dir: str = "/home/tzwang/program/winding_pose_solver"
    env_name: str = "winding_pose_solver"
    partition: str = "amd96c"
    default_cpus: int = 16
    default_mem_mb: int = 65536


def origin_search_run_dir(run_id: str) -> Path:
    return Path("artifacts/online_runs") / str(run_id)


def origin_search_output_path(run_id: str) -> Path:
    return origin_search_run_dir(run_id) / "origin_yz_search_results.json"


def run_origin_search(
    *,
    run_id: str,
    settings: OriginSearchSettings,
    server_settings: OriginSearchServerSettings,
    show_command_details: bool = False,
) -> tuple[dict[str, Path | None], str]:
    if settings.use_server:
        paths = _run_origin_search_remote(
            run_id=run_id,
            settings=settings,
            server_settings=server_settings,
            show_command_details=show_command_details,
        )
    else:
        paths = _run_origin_search_local(
            run_id=run_id,
            settings=settings,
            show_command_details=show_command_details,
        )
    output_path = paths.get("origin_search_result")
    status = "success" if isinstance(output_path, Path) and output_path.exists() else "failed"
    return paths, status


def _run_origin_search_local(
    *,
    run_id: str,
    settings: OriginSearchSettings,
    show_command_details: bool,
) -> dict[str, Path | None]:
    run_dir = origin_search_run_dir(run_id)
    output_path = origin_search_output_path(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    args = _build_origin_search_args(
        python_command=sys.executable,
        output_path=output_path,
        settings=settings,
    )
    print("[origin-search] Running smart-square search locally.")
    _run_streamed_command(args, cwd=Path.cwd(), show_command_details=show_command_details)
    return {
        "origin_search_result": output_path if output_path.exists() else None,
        "run_dir": run_dir,
    }


def _run_origin_search_remote(
    *,
    run_id: str,
    settings: OriginSearchSettings,
    server_settings: OriginSearchServerSettings,
    show_command_details: bool,
) -> dict[str, Path | None]:
    run_dir = origin_search_run_dir(run_id)
    output_path = origin_search_output_path(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    _run_origin_search_sync_preflight(
        host=server_settings.host,
        server_dir=server_settings.server_dir,
        settings=settings,
        show_command_details=show_command_details,
    )

    remote_shell_dir = _remote_dir_for_shell(server_settings.server_dir)
    remote_output_rel = f"artifacts/online_runs/{run_id}/origin_yz_search_results.json"
    remote_output_scp = f"{server_settings.server_dir}/{remote_output_rel}"
    remote_args = _build_origin_search_args(
        python_command="python",
        output_path=remote_output_rel,
        settings=settings,
    )
    remote_command = _render_shell_command(remote_args)
    remote_script = _build_remote_slurm_script(
        remote_shell_dir=remote_shell_dir,
        remote_run_id=run_id,
        env_name=server_settings.env_name,
        partition=server_settings.partition,
        default_cpus=server_settings.default_cpus,
        default_mem_mb=server_settings.default_mem_mb,
        remote_command=remote_command,
    )

    print(
        "[origin-search] Running smart-square search on "
        f"{server_settings.host}:{server_settings.server_dir} via Slurm."
    )
    _run_streamed_command(
        ["ssh", server_settings.host, f"bash -lc {shlex.quote(remote_script)}"],
        cwd=Path.cwd(),
        show_command_details=show_command_details,
    )
    _run_streamed_command(
        ["scp", f"{server_settings.host}:{remote_output_scp}", str(output_path)],
        cwd=Path.cwd(),
        show_command_details=show_command_details,
    )
    return {
        "origin_search_result": output_path if output_path.exists() else None,
        "remote_origin_search_result": Path(remote_output_scp),
        "run_dir": run_dir,
    }


def _run_origin_search_sync_preflight(
    *,
    host: str,
    server_dir: str,
    settings: OriginSearchSettings,
    show_command_details: bool,
) -> None:
    """Keep the server checkout aligned before remote smart-square compute."""
    from src.runtime.online.roundtrip import (
        CommandLogOptions,
        _run_coordinator_sync_preflight,
    )

    _run_coordinator_sync_preflight(
        host=host,
        server_dir=server_dir,
        enforce_remote_sync_guard=bool(settings.enforce_remote_sync_guard),
        remote_sync_mode=str(settings.remote_sync_mode),
        log_options=CommandLogOptions(
            log_path=None,
            show_command_details=bool(show_command_details),
            show_worker_output=True,
        ),
    )


def _build_origin_search_args(
    *,
    python_command: str,
    output_path: str | Path,
    settings: OriginSearchSettings,
) -> list[str]:
    args = [
        str(python_command),
        "scripts/sweep_target_origin_yz.py",
        "--mode",
        "smart-square",
        "--seed-origin",
        ",".join(str(float(value)) for value in settings.target_frame_a_origin_in_frame2_mm),
        "--square-size-mm",
        str(float(settings.square_size_mm)),
        "--smart-initial-step-mm",
        str(float(settings.initial_step_mm)),
        "--smart-min-step-mm",
        str(float(settings.min_step_mm)),
        "--smart-max-iters",
        str(max(1, int(settings.max_iters))),
        "--beam-width",
        str(max(1, int(settings.beam_width))),
        "--smart-diagonal-policy",
        str(settings.diagonal_policy),
        "--smart-polish-step-mm",
        str(max(0.0, float(settings.polish_step_mm))),
        "--workers",
        str(max(1, int(settings.workers))),
        "--strategy",
        str(settings.strategy),
        "--validation-grid-step-mm",
        str(max(0.0, float(settings.validation_grid_step_mm))),
        "--top-k",
        str(max(1, int(settings.top_k))),
        "--min-separation-mm",
        str(max(0.0, float(settings.min_separation_mm))),
        "--outside-fallback-count",
        str(max(0, int(settings.outside_fallback_count))),
        "--outside-fallback-max-rings",
        str(max(0, int(settings.outside_fallback_max_rings))),
        "--outside-fallback-ring-step-mm",
        str(max(0.0, float(settings.outside_fallback_ring_step_mm))),
        "--outside-fallback-edge-step-mm",
        str(max(0.0, float(settings.outside_fallback_edge_step_mm))),
        "--output",
        str(output_path),
    ]
    if not settings.run_window_repair:
        args.append("--skip-window-repair")
    if not settings.run_inserted_repair:
        args.append("--skip-inserted-repair")
    return args


def _build_remote_slurm_script(
    *,
    remote_shell_dir: str,
    remote_run_id: str,
    env_name: str,
    partition: str,
    default_cpus: int,
    default_mem_mb: int,
    remote_command: str,
) -> str:
    return f"""
set -e
cd "{remote_shell_dir}"
mkdir -p "artifacts/online_runs/{remote_run_id}"
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "{env_name}"
export OMP_NUM_THREADS="${{OMP_NUM_THREADS:-1}}"
export OPENBLAS_NUM_THREADS="${{OPENBLAS_NUM_THREADS:-1}}"
export MKL_NUM_THREADS="${{MKL_NUM_THREADS:-1}}"
export NUMEXPR_NUM_THREADS="${{NUMEXPR_NUM_THREADS:-1}}"
if command -v srun >/dev/null 2>&1 && command -v sinfo >/dev/null 2>&1; then
    partition="${{WPS_SERVER_PARTITION:-{partition}}}"
    idle_cpu="$(sinfo -h -p "$partition" -o '%C' 2>/dev/null | awk -F/ 'NR == 1 {{print $2}}')"
    case "$idle_cpu" in
        ''|*[!0-9]*) idle_cpu=1 ;;
    esac
    cpu_req="${{WPS_ORIGIN_SEARCH_CPUS:-{int(default_cpus)}}}"
    case "$cpu_req" in
        ''|*[!0-9]*) cpu_req={int(default_cpus)} ;;
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
        ''|*[!0-9]*) free_mem_mb={int(default_mem_mb)} ;;
    esac
    mem_req_mb="${{WPS_ORIGIN_SEARCH_MEM_MB:-{int(default_mem_mb)}}}"
    case "$mem_req_mb" in
        ''|*[!0-9]*) mem_req_mb={int(default_mem_mb)} ;;
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
    export WPS_OFFLINE_BATCH_WORKERS="${{WPS_OFFLINE_BATCH_WORKERS:-$cpu_req}}"
    echo "[origin-search] Slurm allocation: partition=$partition cpus=$cpu_req mem=${{mem_req_mb}}M"
    srun -p "$partition" -c "$cpu_req" --mem="${{mem_req_mb}}M" {remote_command}
elif [ "${{WPS_ALLOW_LOGIN_NODE_SERVER:-0}}" = "1" ]; then
    echo "[origin-search] WARNING: running on login node because WPS_ALLOW_LOGIN_NODE_SERVER=1"
    {remote_command}
else
    echo "[origin-search] Slurm is required for server compute on master; refusing login-node run." >&2
    exit 125
fi
"""


def _remote_dir_for_shell(server_dir: str) -> str:
    if server_dir.startswith("~/"):
        return server_dir.replace("~/", "$HOME/", 1)
    if server_dir == "~":
        return "$HOME"
    return server_dir


def _render_shell_command(args: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in args)


def _run_streamed_command(
    args: list[str],
    *,
    cwd: Path | None,
    show_command_details: bool,
) -> None:
    if show_command_details:
        print(f"[origin-search] command: {_render_shell_command(args)}")
    result = subprocess.run(
        args,
        cwd=str(cwd or Path.cwd()),
        check=False,
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, args)
