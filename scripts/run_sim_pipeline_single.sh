#!/usr/bin/env bash
# run_sim_pipeline_single.sh - Per-run analysis pipeline with all outputs in one directory
#
# Usage:
#   bash scripts/run_sim_pipeline_single.sh <output_dir> [sqlite_db]
#
# All outputs go to <output_dir>/ (no category subdirectories).
# This variant is tuned for 30-run batch processing: it prioritizes numeric
# artefacts and uses small embedding models to keep resource use bounded.
#
set -euo pipefail

run_py() {
    if command -v pyenv >/dev/null 2>&1; then
        export PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
        export PATH="$PYENV_ROOT/bin:$PATH"
        eval "$(pyenv init -)" 2>/dev/null || true
        eval "$(pyenv virtualenv-init -)" 2>/dev/null || true
        if pyenv prefix ysocial >/dev/null 2>&1; then
            export PYENV_VERSION=ysocial
            pyenv exec python "$@"
            return
        fi
    fi
    "${PYTHON:-python3}" "$@"
}

OUT_DIR="${1:-}"
SIM_DB="${2:-}"

if [[ -z "$OUT_DIR" ]]; then
    echo "Usage: $0 <output_dir> [sqlite_db]" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

# Required files
POSTS_CSV="${OUT_DIR}/posts.csv"
USERS_CSV="${OUT_DIR}/users.csv"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VOAT_PQ="${REPO_ROOT}/data/voat_windows/combined/voat_technology_madoc.parquet"

if [[ ! -f "$POSTS_CSV" ]]; then
    echo "ERROR: posts.csv not found at ${POSTS_CSV}" >&2
    exit 1
fi

RUN_NAME=$(basename "$OUT_DIR")

echo "===== Pipeline for $RUN_NAME ====="
echo "Output directory: $OUT_DIR"

# 1. Network analysis (core-periphery)
echo "[1/9] Core-periphery network analysis..."
run_py scripts/core_periphery_enhanced.py "$POSTS_CSV" --output-dir "$OUT_DIR" 2>&1 || echo "WARN: core_periphery_enhanced.py failed"

# Rename outputs for consistency
[[ -f "${OUT_DIR}/enhanced_network_analysis.txt" ]] && mv "${OUT_DIR}/enhanced_network_analysis.txt" "${OUT_DIR}/network_analysis.txt" 2>/dev/null || true
[[ -f "${OUT_DIR}/enhanced_core_periphery_membership.csv" ]] && mv "${OUT_DIR}/enhanced_core_periphery_membership.csv" "${OUT_DIR}/core_periphery.csv" 2>/dev/null || true
[[ -f "${OUT_DIR}/enhanced_core_periphery_visualization.png" ]] && mv "${OUT_DIR}/enhanced_core_periphery_visualization.png" "${OUT_DIR}/network_viz.png" 2>/dev/null || true

# 2. Convergence entropy
echo "[2/9] Convergence entropy analysis..."
run_py scripts/convergence_entropy_chains.py \
    --posts-csv "$POSTS_CSV" \
    --outdir "$OUT_DIR" \
    --device cpu 2>&1 || echo "WARN: convergence_entropy_chains.py failed"

# Rename entropy outputs (script creates subdir by default, flatten)
if [[ -d "${OUT_DIR}/convergence_entropy" ]]; then
    mv "${OUT_DIR}/convergence_entropy/agg_stats.json" "${OUT_DIR}/entropy_agg.json" 2>/dev/null || true
    mv "${OUT_DIR}/convergence_entropy/pairs_all.csv" "${OUT_DIR}/pairs_all.csv" 2>/dev/null || true
    rm -rf "${OUT_DIR}/convergence_entropy" 2>/dev/null || true
fi

echo "[3/9] Toxicity scoring..."
if [[ -n "$SIM_DB" && -f "$SIM_DB" ]]; then
    run_py scripts/toxigen_roberta.py "$SIM_DB" \
        --output-dir "$OUT_DIR" \
        --madoc-parquet "$VOAT_PQ" \
        --device auto \
        --no-plots 2>&1 || echo "WARN: toxigen_roberta.py failed"
else
    echo "SKIP: No SQLite DB provided for toxicity scoring"
fi

# 4. NER (numeric outputs only; plotting disabled by default)
echo "[4/9] NER analysis..."
TOX_CSV="${OUT_DIR}/toxigen.csv"
NER_DIR="${OUT_DIR}/ner"
if [[ -f "$VOAT_PQ" && -f "$TOX_CSV" ]]; then
    mkdir -p "$NER_DIR"
    run_py scripts/voat_ner_structure.py \
        --sim2-posts "$POSTS_CSV" \
        --sim2-tox "$TOX_CSV" \
        --mode both \
        --outdir "$NER_DIR" 2>&1 || echo "WARN: voat_ner_structure.py failed"
else
    echo "SKIP: Missing toxigen.csv or MADOC parquet for NER"
fi

# 5. Topic analysis (BERTopic, comment-level)
echo "[5/9] Topic analysis (comments)..."
if [[ -f "$VOAT_PQ" ]]; then
    run_py scripts/voat_topic_compare.py \
        --sim2-posts-csv "$POSTS_CSV" \
        --outdir "$OUT_DIR" \
        --min-topic-size 3 \
        --drop-header-rows \
        --min-doc-chars 25 \
        --extra-stopwords title,body,selftext,text,content,don,ve,isn \
        --remove-contraction-fragments \
        --df-threshold 0.0 \
        --vectorizer-min-df 1 \
        --topk-per-sim2 5 \
        --topic-repr hybrid \
        --repr-alpha 0.6 \
        --composite-alpha 0.85 2>&1 || echo "WARN: voat_topic_compare.py failed"

    # Rename topic output
    [[ -f "${OUT_DIR}/summary.json" ]] && mv "${OUT_DIR}/summary.json" "${OUT_DIR}/topic_summary.json" 2>/dev/null || true
else
    echo "SKIP: MADOC parquet not found for topic analysis"
fi

# 6. Thread-level topics (numeric outputs only)
echo "[6/9] Topic analysis (threads)..."
TOPIC_THREADS_DIR="${OUT_DIR}/topic_threads"
if [[ -f "$VOAT_PQ" ]]; then
    mkdir -p "$TOPIC_THREADS_DIR"
    run_py scripts/voat_topic_compare_threads.py \
        --sim2-posts-csv "$POSTS_CSV" \
        --outdir "$TOPIC_THREADS_DIR" \
        --min-topic-size 10 \
        --min-thread-chars 50 \
        --extra-stopwords title,body,selftext,text,content,don,ve,isn \
        --remove-contraction-fragments \
        --df-threshold 0.0 \
        --vectorizer-min-df 1 \
        --topk-per-sim2 5 \
        --topic-repr hybrid \
        --repr-alpha 0.6 \
        --composite-alpha 0.85 2>&1 || echo "WARN: voat_topic_compare_threads.py failed"
else
    echo "SKIP: MADOC parquet not found for thread-level topic analysis"
fi

# 7. Embedding similarity (small model, JSON only, no plots)
echo "[7/9] Embedding similarity..."
if [[ -f "$VOAT_PQ" && -f "$TOX_CSV" ]]; then
    run_py scripts/voat_sim_embedding_similarity.py \
        --sim2-posts "$POSTS_CSV" \
        --sim2-tox "$TOX_CSV" \
        --mode both \
        --device auto \
        --embedding-model "sentence-transformers/all-MiniLM-L6-v2" \
        --no-plot-tsne \
        --no-plot-umap \
        --out-json "${OUT_DIR}/sim_voat_embedding_similarity.json" 2>&1 || echo "WARN: voat_sim_embedding_similarity.py failed"
else
    echo "SKIP: Missing toxigen.csv or MADOC parquet for embedding similarity"
fi

# 8. Semantic diversity (numeric outputs only, plots disabled)
echo "[8/9] Semantic diversity..."
DIVERSITY_DIR="${OUT_DIR}/diversity"
if [[ -f "$USERS_CSV" ]]; then
    mkdir -p "$DIVERSITY_DIR"
    run_py scripts/semantic_diversity_analysis.py \
        --simulation-dir "$OUT_DIR" \
        --output-dir "$DIVERSITY_DIR" \
        --no-plot 2>&1 || echo "WARN: semantic_diversity_analysis.py failed"
else
    echo "SKIP: users.csv not found for semantic diversity"
fi

# 9. MADOC comparison
echo "[9/9] MADOC comparison..."
if [[ -d "${REPO_ROOT}/data/voat_windows/validation" ]]; then
    run_py scripts/compare_madoc_to_sim.py \
        --sim-dirs "$OUT_DIR" \
        --madoc-root "${REPO_ROOT}/data/voat_windows/validation" \
        --out-dir "${OUT_DIR}/madoc_compare" \
        --report-prefix "voat_vs_${RUN_NAME}" \
        --log-level WARNING 2>&1 || echo "WARN: compare_madoc_to_sim.py failed"
fi

echo "===== Pipeline complete for $RUN_NAME ====="
