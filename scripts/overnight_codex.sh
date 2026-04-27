#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/overnight_codex.sh start [options]
  scripts/overnight_codex.sh latest [runs_dir]
  scripts/overnight_codex.sh status <run_dir>
  scripts/overnight_codex.sh logs <run_dir>
  scripts/overnight_codex.sh last <run_dir>
  scripts/overnight_codex.sh handoff <run_dir>
  scripts/overnight_codex.sh session <run_dir>
  scripts/overnight_codex.sh resume <run_dir> [followup_prompt]

Commands:
  start    Launch an unattended Codex run as a detached background process.
  latest   Print the newest overnight run directory.
  status   Show service status plus key output files.
  logs     Show the captured Codex JSON/log output.
  last     Show the last assistant message file if present.
  handoff  Show the generated handoff markdown if present.
  session  Print the Codex session/thread id parsed from the log.
  resume   Resume the run with an optional follow-up prompt.

Start options:
  --workdir DIR      Working directory to run in (default: current directory)
  --task FILE        Task spec markdown file (default: <workdir>/OVERNIGHT_TASK.md)
  --runs-dir DIR     Output directory for overnight runs
                     (default: <workdir>/artifacts/codex_overnight_runs)
  --name NAME        Friendly label included in the unit description
  --model MODEL      Optional Codex model override
  --config KEY=VAL   Extra Codex config override (repeatable)
  --search           Enable Codex web search for the run
  --help             Show this help

Examples:
  scripts/overnight_codex.sh start
  scripts/overnight_codex.sh start --model gpt-5.4-mini --config model_reasoning_effort=low
  scripts/overnight_codex.sh start --search --name nightly-fix
  scripts/overnight_codex.sh latest
  scripts/overnight_codex.sh status artifacts/codex_overnight_runs/20260416-021500
  scripts/overnight_codex.sh resume artifacts/codex_overnight_runs/20260416-021500 \
    "Continue the remaining blockers from the handoff."
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

require_file() {
  [ -f "$1" ] || die "missing file: $1"
}

codex_bin() {
  if command -v codex >/dev/null 2>&1; then
    command -v codex
    return 0
  fi
  local bundled="$HOME/.vscode-server/extensions/openai.chatgpt-26.409.20454-linux-x64/bin/linux-x86_64/codex"
  [ -x "$bundled" ] || die "could not find codex binary"
  printf '%s\n' "$bundled"
}

resolve_run_dir() {
  local run_dir="$1"
  [ -d "$run_dir" ] || die "run directory not found: $run_dir"
  printf '%s\n' "$(cd "$run_dir" && pwd)"
}

latest_run_dir() {
  local runs_dir="$1"
  [ -d "$runs_dir" ] || die "runs directory not found: $runs_dir"
  local latest
  latest="$(find "$runs_dir" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
  [ -n "$latest" ] || die "no overnight runs found under: $runs_dir"
  printf '%s\n' "$latest"
}

session_id_from_log() {
  local log_file="$1"
  require_file "$log_file"
  python - <<'PY' "$log_file"
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
    line = line.strip()
    if not line.startswith("{"):
        continue
    try:
        payload = json.loads(line)
    except Exception:
        continue
    if payload.get("type") == "thread.started":
        thread_id = payload.get("thread_id")
        if thread_id:
            print(thread_id)
            raise SystemExit(0)
raise SystemExit(1)
PY
}

run_status() {
  local run_dir
  run_dir="$(resolve_run_dir "$1")"
  local meta="$run_dir/metadata.env"
  require_file "$meta"
  # shellcheck disable=SC1090
  source "$meta"

  echo "Run directory: $run_dir"
  echo "Workdir: $WORKDIR"
  if [ -n "${PID_FILE:-}" ]; then
    echo "PID file: $PID_FILE"
  fi
  if [ -n "${UNIT:-}" ]; then
    echo "Legacy unit: $UNIT"
  fi
  echo "Task copy: $TASK_COPY"
  echo "Log: $LOG_FILE"
  echo "Last message: $LAST_MESSAGE_FILE"
  echo "Handoff: $HANDOFF_FILE"

  if [ -n "${PID_FILE:-}" ] && [ -f "$PID_FILE" ]; then
    local pid
    pid="$(cat "$PID_FILE")"
    if [ -n "$pid" ] && ps -p "$pid" >/dev/null 2>&1; then
      echo
      ps -p "$pid" -o pid,ppid,stat,etime,cmd
      return 0
    fi
    echo
    echo "Process is not active."
    return 0
  fi

  if [ -n "${UNIT:-}" ] && systemctl --user --quiet is-active "$UNIT"; then
    echo
    systemctl --user status "$UNIT" --no-pager -l
  else
    echo
    echo "No active process metadata found."
  fi
}

run_logs() {
  local run_dir
  run_dir="$(resolve_run_dir "$1")"
  require_file "$run_dir/codex.jsonl"
  cat "$run_dir/codex.jsonl"
}

run_last() {
  local run_dir
  run_dir="$(resolve_run_dir "$1")"
  require_file "$run_dir/last_message.txt"
  cat "$run_dir/last_message.txt"
}

run_handoff() {
  local run_dir
  run_dir="$(resolve_run_dir "$1")"
  require_file "$run_dir/handoff.md"
  cat "$run_dir/handoff.md"
}

run_session() {
  local run_dir
  run_dir="$(resolve_run_dir "$1")"
  local log_file="$run_dir/codex.jsonl"
  local session_id
  session_id="$(session_id_from_log "$log_file")" || die "session id not found yet in $log_file"
  printf '%s\n' "$session_id"
}

run_latest() {
  local runs_dir="${1:-$(pwd)/artifacts/codex_overnight_runs}"
  runs_dir="$(cd "$runs_dir" && pwd)"
  latest_run_dir "$runs_dir"
}

run_resume() {
  local run_dir
  run_dir="$(resolve_run_dir "$1")"
  shift

  local meta="$run_dir/metadata.env"
  require_file "$meta"
  # shellcheck disable=SC1090
  source "$meta"

  local session_id
  session_id="$(session_id_from_log "$LOG_FILE")" || die "session id not found yet in $LOG_FILE"
  local codex
  codex="$(codex_bin)"

  (
    cd "$WORKDIR"
    if [ "$#" -gt 0 ]; then
      "$codex" exec resume "$session_id" \
        --disable apps \
        --disable plugins \
        --skip-git-repo-check \
        "$*"
    else
      "$codex" exec resume "$session_id" \
        --disable apps \
        --disable plugins \
        --skip-git-repo-check
    fi
  )
}

run_start() {
  local workdir
  workdir="$(pwd)"
  local runs_dir=""
  local task_file=""
  local name="overnight"
  local model=""
  local enable_search=0
  local -a extra_configs=()

  while [ "$#" -gt 0 ]; do
    case "$1" in
      --workdir)
        workdir="$2"
        shift 2
        ;;
      --task)
        task_file="$2"
        shift 2
        ;;
      --runs-dir)
        runs_dir="$2"
        shift 2
        ;;
      --name)
        name="$2"
        shift 2
        ;;
      --model)
        model="$2"
        shift 2
        ;;
      --config)
        extra_configs+=("$2")
        shift 2
        ;;
      --search)
        enable_search=1
        shift
        ;;
      --help|-h)
        usage
        exit 0
        ;;
      *)
        die "unknown option for start: $1"
        ;;
    esac
  done

  workdir="$(cd "$workdir" && pwd)"
  if [ -z "$task_file" ]; then
    task_file="$workdir/OVERNIGHT_TASK.md"
  fi
  if [ -z "$runs_dir" ]; then
    runs_dir="$workdir/artifacts/codex_overnight_runs"
  fi

  require_file "$task_file"
  require_file "$workdir/OVERNIGHT_HANDOFF_TEMPLATE.md"
  mkdir -p "$runs_dir"

  local timestamp
  timestamp="$(date '+%Y%m%d-%H%M%S')"
  local run_dir="$runs_dir/$timestamp"
  mkdir -p "$run_dir"

  local task_copy="$run_dir/task.md"
  local handoff_file="$run_dir/handoff.md"
  local last_message_file="$run_dir/last_message.txt"
  local log_file="$run_dir/codex.jsonl"
  local prompt_file="$run_dir/prompt.md"
  local launch_file="$run_dir/launch.sh"
  local meta_file="$run_dir/metadata.env"
  local pid_file="$run_dir/pid.txt"
  local codex
  codex="$(codex_bin)"

  cp "$task_file" "$task_copy"
  cp "$workdir/OVERNIGHT_HANDOFF_TEMPLATE.md" "$run_dir/handoff_template.md"

  cat >"$prompt_file" <<EOF
You are running as an unattended overnight Codex job on a remote server.

Execute the task below end-to-end using the current repository instructions, skills, and config. Work autonomously until the task is complete or a real blocker remains. Use Slurm for any heavy work as required by the existing server policy. Prefer the smallest useful validation for any change.

Before stopping, write a concise handoff markdown file to:
$handoff_file

Follow this template:
$run_dir/handoff_template.md

If you finish cleanly, make the final assistant message a short completion note that references the handoff file. If you are blocked, make the final assistant message briefly explain the blocker and still write the handoff.

Task specification follows.

---
EOF
  cat "$task_copy" >>"$prompt_file"

  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -euo pipefail\n'
    printf 'prompt=$(cat %q)\n' "$prompt_file"
    printf 'exec %q exec --disable %q --disable %q --json -o %q --full-auto --skip-git-repo-check -C %q ' \
      "$codex" "apps" "plugins" "$last_message_file" "$workdir"
    if [ "$enable_search" -eq 1 ]; then
      printf '%q ' '--search'
    fi
    if [ -n "$model" ]; then
      printf '%q %q ' '--model' "$model"
    fi
    if [ "${#extra_configs[@]}" -gt 0 ]; then
      local cfg
      for cfg in "${extra_configs[@]}"; do
        printf '%q %q ' '--config' "$cfg"
      done
    fi
    printf '"$prompt" < /dev/null > %q 2>&1\n' "$log_file"
  } >"$launch_file"
  chmod +x "$launch_file"

  cat >"$meta_file" <<EOF
RUN_DIR=$run_dir
WORKDIR=$workdir
TASK_COPY=$task_copy
HANDOFF_FILE=$handoff_file
LAST_MESSAGE_FILE=$last_message_file
LOG_FILE=$log_file
PROMPT_FILE=$prompt_file
LAUNCH_FILE=$launch_file
PID_FILE=$pid_file
CODEX_BIN=$codex
EOF

  (
    cd "$workdir"
    setsid "$launch_file" >/dev/null 2>&1 < /dev/null &
    echo "$!" >"$pid_file"
  )

  echo "Started overnight Codex run."
  echo "Run directory: $run_dir"
  echo "PID: $(cat "$pid_file")"
  echo
  echo "Useful commands:"
  echo "  scripts/overnight_codex.sh status $run_dir"
  echo "  scripts/overnight_codex.sh logs $run_dir"
  echo "  scripts/overnight_codex.sh handoff $run_dir"
  echo "  scripts/overnight_codex.sh resume $run_dir \"Continue from the handoff.\""
}

main() {
  local cmd="${1:-}"
  case "$cmd" in
    start)
      shift
      run_start "$@"
      ;;
    latest)
      shift
      [ "$#" -le 1 ] || die "latest accepts at most [runs_dir]"
      run_latest "${1:-}"
      ;;
    status)
      shift
      [ "$#" -eq 1 ] || die "status requires <run_dir>"
      run_status "$1"
      ;;
    logs)
      shift
      [ "$#" -eq 1 ] || die "logs requires <run_dir>"
      run_logs "$1"
      ;;
    last)
      shift
      [ "$#" -eq 1 ] || die "last requires <run_dir>"
      run_last "$1"
      ;;
    handoff)
      shift
      [ "$#" -eq 1 ] || die "handoff requires <run_dir>"
      run_handoff "$1"
      ;;
    session)
      shift
      [ "$#" -eq 1 ] || die "session requires <run_dir>"
      run_session "$1"
      ;;
    resume)
      shift
      [ "$#" -ge 1 ] || die "resume requires <run_dir>"
      local run_dir="$1"
      shift
      run_resume "$run_dir" "$@"
      ;;
    ""|--help|-h|help)
      usage
      ;;
    *)
      die "unknown command: $cmd"
      ;;
  esac
}

main "$@"
