#!/usr/bin/env bash
set -euo pipefail

MODE="audit"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

audit_mode() {
    echo "Retained public artifacts:"
    find "$ROOT/artifacts/final" -type f | sort
    echo
    echo "Benchmark DB count: $(find "$ROOT/data/benchmark_runs" -type f -name '*.sqlite' | wc -l)"
    echo "Sensitivity DB count: $(find "$ROOT/data/sensitivity_runs" -type f -name '*.sqlite' | wc -l)"
    echo "Voat parquet count: $(find "$ROOT/data/voat_windows" -type f -name '*.parquet' | wc -l)"
}

rerun_mode() {
    cat <<EOF
Trajectory rerun mode uses the retained benchmark and sensitivity SQLite files.

Primary retained entrypoints:
  bash scripts/run_30_analysis.sh --dry-run
  bash scripts/run_sensitivity_pipeline.sh all --dry-run
  python scripts/power_analysis_30runs.py
  python scripts/powerlaw_fit.py
  python scripts/workshop_sensitivity_tornado.py

The empirical-window side is retained as raw parquets plus frozen benchmark summaries.
Legacy window-analysis scripts expect transient per-window workspaces and should be run
outside version control if you want to rebuild those intermediates from scratch.
EOF

    if [[ "$DRY_RUN" == true ]]; then
        return 0
    fi
}

case "$MODE" in
    audit)
        audit_mode
        ;;
    rerun)
        rerun_mode
        ;;
    *)
        echo "Unknown mode: $MODE" >&2
        exit 1
        ;;
esac
