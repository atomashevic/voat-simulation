#!/usr/bin/env bash
# run_30_analysis.sh - Resource-safe batch driver for 30-run simulation analysis
#
# Usage:
#   bash scripts/run_30_analysis.sh [OPTIONS]
#
# Options:
#   --max-parallel N   Max parallel workers (default: 3)
#   --skip-existing    Skip runs where metrics.json exists (default: true)
#   --force            Re-run all analyses even if outputs exist
#   --dry-run          Print commands without executing
#   --run PATTERN      Only process runs matching pattern (e.g., "run01" or "run0*")
#
set -euo pipefail

# Defaults
MAX_PARALLEL=${MAX_PARALLEL:-3}
SKIP_EXISTING=${SKIP_EXISTING:-true}
DRY_RUN=${DRY_RUN:-false}
NICE_LEVEL=${NICE_LEVEL:-10}
RUN_PATTERN=""
FORCE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-parallel)
            MAX_PARALLEL="$2"
            shift 2
            ;;
        --skip-existing)
            SKIP_EXISTING=true
            shift
            ;;
        --force)
            FORCE=true
            SKIP_EXISTING=false
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --run)
            RUN_PATTERN="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

PYTHON_CMD=("${PYTHON:-python3}")
if command -v pyenv >/dev/null 2>&1; then
    export PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)" 2>/dev/null || true
    eval "$(pyenv virtualenv-init -)" 2>/dev/null || true
    if pyenv prefix ysocial >/dev/null 2>&1; then
        export PYENV_VERSION=ysocial
        PYTHON_CMD=(pyenv exec python)
    fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
RUNS_DIR="${REPO_ROOT}/data/benchmark_runs"
RESULTS_DIR="${REPO_ROOT}/results/benchmark"
PROGRESS_LOG="${RESULTS_DIR}/progress.log"

mkdir -p "$RESULTS_DIR"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$PROGRESS_LOG"
}

# Collect runs to process
declare -a RUN_DBS=()
for db_file in "$RUNS_DIR"/run*.sqlite; do
    [[ -f "$db_file" ]] || continue
    run_name=$(basename "$db_file" .sqlite)

    # Apply pattern filter if specified
    if [[ -n "$RUN_PATTERN" ]] && [[ ! "$run_name" == $RUN_PATTERN ]]; then
        continue
    fi

    out_dir="${RESULTS_DIR}/${run_name}"
    metrics_json="${out_dir}/metrics.json"

    # Skip if metrics.json exists and not forcing
    if [[ "$SKIP_EXISTING" == "true" && -f "$metrics_json" ]]; then
        log "SKIP $run_name (metrics.json exists)"
        continue
    fi

    RUN_DBS+=("$db_file")
done

TOTAL_RUNS=${#RUN_DBS[@]}
if [[ $TOTAL_RUNS -eq 0 ]]; then
    log "No runs to process (all skipped or none matched pattern)"
    exit 0
fi

log "Starting analysis of $TOTAL_RUNS runs with max $MAX_PARALLEL parallel workers"
log "Options: skip_existing=$SKIP_EXISTING, nice=$NICE_LEVEL, dry_run=$DRY_RUN"

# Process a single run
process_run() {
    local db_path="$1"
    local run_name=$(basename "$db_path" .sqlite)
    local out_dir="${RESULTS_DIR}/${run_name}"

    log "START $run_name"

    mkdir -p "$out_dir"

    # Step 1: Export CSVs from SQLite
    if [[ ! -f "${out_dir}/posts.csv" ]] || [[ "$FORCE" == "true" ]]; then
        nice -n "$NICE_LEVEL" "${PYTHON_CMD[@]}" "$SCRIPT_DIR/sim_stats.py" "$db_path"
    fi

    # Step 2: Run the main pipeline (modified to use single output dir)
    nice -n "$NICE_LEVEL" bash "$SCRIPT_DIR/run_sim_pipeline_single.sh" "$out_dir" "$db_path"

    # Step 3: Extract unified metrics
    nice -n "$NICE_LEVEL" "${PYTHON_CMD[@]}" "$SCRIPT_DIR/extract_run_metrics.py" --run-dir "$out_dir"

    log "DONE $run_name"
}

# Track running jobs
declare -a PIDS=()
declare -a RUN_NAMES=()
COMPLETED=0
FAILED=0

cleanup() {
    log "Received interrupt, killing running jobs..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    exit 1
}
trap cleanup SIGINT SIGTERM

# Main loop with parallelism control
for db_path in "${RUN_DBS[@]}"; do
    run_name=$(basename "$db_path" .sqlite)

    # Wait if we're at max parallel
    while [[ ${#PIDS[@]} -ge $MAX_PARALLEL ]]; do
        # Check for completed jobs
        declare -a NEW_PIDS=()
        declare -a NEW_NAMES=()
        for i in "${!PIDS[@]}"; do
            pid="${PIDS[$i]}"
            name="${RUN_NAMES[$i]}"
            if kill -0 "$pid" 2>/dev/null; then
                NEW_PIDS+=("$pid")
                NEW_NAMES+=("$name")
                continue
            fi

            if wait "$pid"; then
                log "COMPLETED $name"
                COMPLETED=$((COMPLETED + 1))
            else
                status=$?
                log "FAILED $name (exit code: $status)"
                FAILED=$((FAILED + 1))
            fi
        done
        PIDS=("${NEW_PIDS[@]}")
        RUN_NAMES=("${NEW_NAMES[@]}")

        if [[ ${#PIDS[@]} -ge $MAX_PARALLEL ]]; then
            sleep 5
        fi
    done

    # Launch new job
    if [[ "$DRY_RUN" == "true" ]]; then
        log "DRY-RUN: would process $run_name"
    else
        process_run "$db_path" &
        PIDS+=($!)
        RUN_NAMES+=("$run_name")
        log "LAUNCHED $run_name (pid: ${PIDS[-1]}, running: ${#PIDS[@]}/$MAX_PARALLEL)"
    fi
done

# Wait for remaining jobs
log "Waiting for ${#PIDS[@]} remaining jobs..."
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    name="${RUN_NAMES[$i]}"
    if wait "$pid"; then
        log "COMPLETED $name"
        COMPLETED=$((COMPLETED + 1))
    else
        status=$?
        log "FAILED $name (exit code: $status)"
        FAILED=$((FAILED + 1))
    fi
done

log "========================================"
log "FINISHED: $COMPLETED completed, $FAILED failed out of $TOTAL_RUNS total"
log "========================================"

exit $FAILED
