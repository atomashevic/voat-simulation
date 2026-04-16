#!/usr/bin/env bash
# run_sensitivity_pipeline.sh - Analysis pipeline for sensitivity analysis conditions.
#
# Processes retained sensitivity SQLite files, runs analysis steps,
# aggregates per condition, and produces cross-condition comparison outputs.
#
# Usage:
#   bash scripts/run_sensitivity_pipeline.sh [all | c0 c1 c2 ...]
#       [--max-parallel N] [--skip-existing] [--force] [--dry-run]
#       [--no-compare] [--sensitivity-dir DIR] [--results-dir DIR]
#
# Examples:
#   bash scripts/run_sensitivity_pipeline.sh all
#   bash scripts/run_sensitivity_pipeline.sh c0 c1 c2 --max-parallel 3
#   bash scripts/run_sensitivity_pipeline.sh c0 --dry-run
#
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
MAX_PARALLEL=${MAX_PARALLEL:-2}
SKIP_EXISTING=${SKIP_EXISTING:-true}
DRY_RUN=${DRY_RUN:-false}
FORCE=false
NO_COMPARE=false
NICE_LEVEL=${NICE_LEVEL:-10}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
SENSITIVITY_DIR="${REPO_ROOT}/data/sensitivity_runs"
RESULTS_DIR="${REPO_ROOT}/results/sensitivity"
VOAT_PQ="${REPO_ROOT}/data/voat_windows/combined/voat_technology_madoc_7d.parquet"

# Default conditions (all)
REQUESTED_CONDITIONS=()

# ── Parse arguments ────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-parallel)   MAX_PARALLEL="$2";     shift 2 ;;
        --skip-existing)  SKIP_EXISTING=true;    shift ;;
        --force)          FORCE=true; SKIP_EXISTING=false; shift ;;
        --dry-run)        DRY_RUN=true;          shift ;;
        --no-compare)     NO_COMPARE=true;       shift ;;
        --sensitivity-dir) SENSITIVITY_DIR="$2"; shift 2 ;;
        --results-dir)     RESULTS_DIR="$2";     shift 2 ;;
        all)              REQUESTED_CONDITIONS=(); shift ;;  # empty = all
        c[0-9]*)          REQUESTED_CONDITIONS+=("$1"); shift ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 [all | c0 c1 ...] [--max-parallel N] [--skip-existing] [--force] [--dry-run] [--no-compare]" >&2
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

# ── Setup directories and logging ──────────────────────────────────────────────
mkdir -p "$RESULTS_DIR"
PROGRESS_LOG="${RESULTS_DIR}/progress.log"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$PROGRESS_LOG"
}

# ── Collect DB files to process ────────────────────────────────────────────────
declare -a RUN_DBS=()

for cond_path in "${SENSITIVITY_DIR}"/c*; do
    [[ -d "$cond_path" ]] || continue
    cond=$(basename "$cond_path")
    for db_file in "${cond_path}"/run*.sqlite; do
        [[ -f "$db_file" ]] || continue
        run_num=$(basename "$db_file" .sqlite)

        # Filter by requested conditions if specified
        if [[ ${#REQUESTED_CONDITIONS[@]} -gt 0 ]]; then
            matched=false
            for req in "${REQUESTED_CONDITIONS[@]}"; do
                [[ "$cond" == "$req" ]] && matched=true && break
            done
            [[ "$matched" == "true" ]] || continue
        fi

        out_dir="${RESULTS_DIR}/${cond}/${run_num}"

        # Skip if metrics.json already exists (unless --force)
        if [[ "$SKIP_EXISTING" == "true" && -f "${out_dir}/metrics.json" ]]; then
            log "SKIP ${cond}/${run_num} (metrics.json exists)"
            continue
        fi

        RUN_DBS+=("$db_file")
    done
done

TOTAL_RUNS=${#RUN_DBS[@]}
if [[ $TOTAL_RUNS -eq 0 ]]; then
    log "No runs to process (all skipped or none found in ${SENSITIVITY_DIR})"
    exit 0
fi

log "Starting analysis of $TOTAL_RUNS runs with max $MAX_PARALLEL parallel workers"
log "sensitivity-dir=${SENSITIVITY_DIR}  results-dir=${RESULTS_DIR}"
log "skip_existing=${SKIP_EXISTING}  force=${FORCE}  dry_run=${DRY_RUN}"

# ── validate_db: check if DB has been run through YServer ─────────────────────
validate_db() {
    local db_path="$1"
    "${PYTHON_CMD[@]}" -c "
import sqlite3, sys
try:
    conn = sqlite3.connect(sys.argv[1])
    tables = [r[0] for r in conn.execute(\"SELECT name FROM sqlite_master WHERE type='table'\").fetchall()]
    conn.close()
    sys.exit(0 if 'post' in tables else 1)
except Exception:
    sys.exit(2)
" "$db_path"
}

# ── extract_csvs: inline SQL extraction (bypasses sim_stats.py path issue) ────
extract_csvs() {
    local db_path="$1"
    local out_dir="$2"
    "${PYTHON_CMD[@]}" -c "
import sqlite3, sys
try:
    import pandas as pd
except ImportError:
    import subprocess; subprocess.run([sys.executable, '-m', 'pip', 'install', 'pandas', '-q'])
    import pandas as pd

db_path, out_dir = sys.argv[1], sys.argv[2]
conn = sqlite3.connect(db_path)

try:
    pd.read_sql_query('SELECT * FROM post', conn).to_csv(f'{out_dir}/posts.csv', index=False)
    print(f'  Extracted posts.csv', flush=True)
except Exception as e:
    print(f'  ERROR extracting posts: {e}', file=sys.stderr)

try:
    pd.read_sql_query('SELECT * FROM user_mgmt', conn).to_csv(f'{out_dir}/users.csv', index=False)
    print(f'  Extracted users.csv', flush=True)
except Exception as e:
    print(f'  ERROR extracting users: {e}', file=sys.stderr)

try:
    pd.read_sql_query('SELECT * FROM articles', conn).to_csv(f'{out_dir}/news.csv', index=False)
    print(f'  Extracted news.csv', flush=True)
except Exception as e:
    print(f'  Warning: articles table missing: {e}', file=sys.stderr)

conn.close()
" "$db_path" "$out_dir"
}

# ── process_run: 7-step analysis for a single DB file ─────────────────────────
process_run() {
    local db_path="$1"
    local run_num
    local cond
    run_num=$(basename "$db_path" .sqlite)
    cond=$(basename "$(dirname "$db_path")")
    local out_dir="${RESULTS_DIR}/${cond}/${run_num}"

    log "START ${cond}/${run_num}"
    mkdir -p "$out_dir"

    # Step 0: Validate DB has simulation output
    if ! validate_db "$db_path"; then
        log "SKIP_UNRUN ${cond}/${run_num} — DB lacks 'post' table (not yet run through YServer)"
        return 0
    fi

    # Step 1: Extract CSVs
    if [[ ! -f "${out_dir}/posts.csv" || "$FORCE" == "true" ]]; then
        log "  [1/7] Extracting CSVs from ${cond}/${run_num}.sqlite"
        extract_csvs "$db_path" "$out_dir" \
            || { log "  ERROR: CSV extraction failed for ${cond}/${run_num}"; return 1; }
    else
        log "  [1/7] Skipping CSV extraction (posts.csv exists)"
    fi

    POSTS_CSV="${out_dir}/posts.csv"

    # Step 2: Core-periphery network analysis
    if [[ ! -f "${out_dir}/enhanced_network_analysis.txt" || "$FORCE" == "true" ]]; then
        log "  [2/7] Core-periphery network analysis"
        nice -n "$NICE_LEVEL" "${PYTHON_CMD[@]}" "${SCRIPT_DIR}/core_periphery_enhanced.py" \
            "$POSTS_CSV" --output-dir "$out_dir" \
            || log "  WARNING: core_periphery_enhanced.py failed for ${cond}/${run_num}"
    else
        log "  [2/7] Skipping network analysis (enhanced_network_analysis.txt exists)"
    fi

    # Step 3: Convergence entropy
    if [[ ! -f "${out_dir}/agg_stats.json" && ! -f "${out_dir}/entropy_agg.json" ]] \
       || [[ "$FORCE" == "true" ]]; then
        log "  [3/7] Convergence entropy"
        nice -n "$NICE_LEVEL" "${PYTHON_CMD[@]}" "${SCRIPT_DIR}/convergence_entropy_chains.py" \
            --posts-csv "$POSTS_CSV" --outdir "$out_dir" --device cpu \
            || log "  WARNING: convergence_entropy_chains.py failed for ${cond}/${run_num}"
        # Flatten nested convergence_entropy/ outputs to run root for extract_run_metrics.py
        if [[ -d "${out_dir}/convergence_entropy" ]]; then
            for f in agg_stats.json chains.csv pairs_all.csv; do
                [[ -f "${out_dir}/convergence_entropy/${f}" ]] && \
                    cp "${out_dir}/convergence_entropy/${f}" "${out_dir}/${f}" 2>/dev/null || true
            done
        fi
    else
        log "  [3/7] Skipping convergence entropy (agg_stats.json exists)"
    fi

    # Step 4: Toxicity scoring
    if [[ ! -f "${out_dir}/toxigen.csv" || "$FORCE" == "true" ]]; then
        log "  [4/7] Toxicity scoring (ToxiGen RoBERTa)"
        if [[ -f "$VOAT_PQ" ]]; then
            nice -n "$NICE_LEVEL" "${PYTHON_CMD[@]}" "${SCRIPT_DIR}/toxigen_roberta.py" \
                "$db_path" \
                --output-dir "$out_dir" \
                --madoc-parquet "$VOAT_PQ" \
                --device auto \
                --no-plots \
                || log "  WARNING: toxigen_roberta.py failed for ${cond}/${run_num}"
        else
            log "  WARNING: Voat parquet not found at ${VOAT_PQ}, skipping toxicity"
        fi
    else
        log "  [4/7] Skipping toxicity (toxigen.csv exists)"
    fi

    # Step 5: Topic comparison vs Voat
    if [[ ! -f "${out_dir}/summary.json" || "$FORCE" == "true" ]]; then
        log "  [5/7] Topic comparison (BERTopic vs Voat)"
        if [[ -f "$VOAT_PQ" ]]; then
            nice -n "$NICE_LEVEL" "${PYTHON_CMD[@]}" "${SCRIPT_DIR}/voat_topic_compare.py" \
                --sim2-posts-csv "$POSTS_CSV" \
                --outdir "$out_dir" \
                --min-topic-size 2 \
                --drop-header-rows \
                --min-doc-chars 25 \
                --extra-stopwords title,body,selftext,text,content,don,ve,isn \
                --remove-contraction-fragments \
                --df-threshold 0.0 \
                --vectorizer-min-df 1 \
                --topk-per-sim2 5 \
                --topic-repr hybrid \
                --repr-alpha 0.6 \
                --composite-alpha 0.85 \
                --save-heatmap \
                || log "  WARNING: voat_topic_compare.py failed for ${cond}/${run_num}"
        else
            log "  WARNING: Voat parquet not found, skipping topic comparison"
        fi
    else
        log "  [5/7] Skipping topics (summary.json exists)"
    fi

    # Step 6: (Optional) Embedding similarity scalar — lightweight, no UMAP/t-SNE
    # Skipped by default to save compute on 50 runs. Enable with --full flag if needed.

    # Step 7: Extract unified metrics
    log "  [7/7] Extracting unified metrics.json"
    nice -n "$NICE_LEVEL" "${PYTHON_CMD[@]}" "${SCRIPT_DIR}/extract_run_metrics.py" \
        --run-dir "$out_dir" \
        || log "  WARNING: extract_run_metrics.py failed for ${cond}/${run_num}"

    log "DONE ${cond}/${run_num}"
}

# ── Parallel job dispatch (from run_30_analysis.sh pattern) ───────────────────
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

for db_path in "${RUN_DBS[@]}"; do
    run_num=$(basename "$db_path" .sqlite)
    cond=$(basename "$(dirname "$db_path")")
    run_label="${cond}/${run_num}"

    # Wait if at max parallel
    while [[ ${#PIDS[@]} -ge $MAX_PARALLEL ]]; do
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
        [[ ${#PIDS[@]} -ge $MAX_PARALLEL ]] && sleep 5
    done

    if [[ "$DRY_RUN" == "true" ]]; then
        log "DRY-RUN: would process ${run_label}"
    else
        process_run "$db_path" &
        PIDS+=($!)
        RUN_NAMES+=("$run_label")
        log "LAUNCHED ${run_label} (pid: ${PIDS[-1]}, running: ${#PIDS[@]}/${MAX_PARALLEL})"
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
log "RUN ANALYSIS: $COMPLETED completed, $FAILED failed out of $TOTAL_RUNS total"
log "========================================"

if [[ "$DRY_RUN" == "true" ]]; then
    log "DRY-RUN mode — skipping aggregation and comparison steps"
    exit 0
fi

# ── Per-condition aggregation ──────────────────────────────────────────────────
# Determine which conditions were processed
declare -a CONDITIONS_PROCESSED=()
if [[ ${#REQUESTED_CONDITIONS[@]} -gt 0 ]]; then
    CONDITIONS_PROCESSED=("${REQUESTED_CONDITIONS[@]}")
else
    # Auto-detect from results directory
    for cond_dir in "${RESULTS_DIR}"/c*/; do
        [[ -d "$cond_dir" ]] || continue
        cond=$(basename "$cond_dir")
        [[ "$cond" =~ ^c[0-9]+$ ]] && CONDITIONS_PROCESSED+=("$cond")
    done
fi

for cond in "${CONDITIONS_PROCESSED[@]}"; do
    cond_dir="${RESULTS_DIR}/${cond}"
    agg_dir="${cond_dir}/aggregate"
    [[ -d "$cond_dir" ]] || continue

    log "Aggregating metrics for condition ${cond}..."
    mkdir -p "$agg_dir"
    "${PYTHON_CMD[@]}" "${SCRIPT_DIR}/aggregate_30runs.py" \
        --results-dir "$cond_dir" \
        --output-dir "$agg_dir" \
        || log "WARNING: aggregation failed for condition ${cond}"
done

# ── Cross-condition comparison ─────────────────────────────────────────────────
if [[ "$NO_COMPARE" == "true" ]]; then
    log "Skipping comparison (--no-compare)"
else
    log "Running cross-condition comparison..."
    "${PYTHON_CMD[@]}" "${SCRIPT_DIR}/compare_sensitivity_conditions.py" \
        --sensitivity-dir "$RESULTS_DIR" \
        --groups "persona:c0,c1,c2" "temperature:c0,c3,c4" \
                 "budget:c0,c5,c6" "cpr:c0,c7,c8" "churn:c0,c9,c10" \
        --output-dir "${RESULTS_DIR}/comparisons" \
        --voat-madoc-root "${REPO_ROOT}/data/voat_windows/validation" \
        --voat-parquet "$VOAT_PQ" \
        || log "WARNING: compare_sensitivity_conditions.py failed"
fi

log "========================================"
log "PIPELINE COMPLETE. Outputs under: ${RESULTS_DIR}"
log "  Per-run:        ${RESULTS_DIR}/c{0-10}/run{0-9}/"
log "  Per-condition:  ${RESULTS_DIR}/c{0-10}/aggregate/"
log "  Comparisons:    ${RESULTS_DIR}/comparisons/{persona,temperature,budget,cpr,churn}/"
log "========================================"

exit $FAILED
