#!/usr/bin/env bash
set -euo pipefail

# Runs the main analysis pipeline for a simulation directory (simulation2 or simulation3).
# Usage:
#   bash scripts/run_sim_pipeline.sh simulation2
#   bash scripts/run_sim_pipeline.sh simulation3

# Initialize pyenv in non-interactive shell and activate 'ysocial' if available
if command -v pyenv >/dev/null 2>&1; then
  export PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
  export PATH="$PYENV_ROOT/bin:$PATH"
  # Load pyenv + virtualenv shims
  eval "$(pyenv init -)"
  # virtualenv plugin (ignore errors if not installed)
  eval "$(pyenv virtualenv-init -)" 2>/dev/null || true
  # Try to activate; if that fails, at least pin version for shims
  if ! pyenv activate ysocial >/dev/null 2>&1; then
    export PYENV_VERSION=ysocial
  fi
else
  echo "pyenv not found; proceeding with system Python or existing venv" >&2
fi

SIM_DIR=${1:-}
if [[ -z "${SIM_DIR}" ]]; then
  echo "Usage: $0 <simulation_dir> (e.g., simulation2 or simulation3)" >&2
  exit 1
fi

if [[ ! -d "${SIM_DIR}" ]]; then
  echo "Simulation directory not found: ${SIM_DIR}" >&2
  exit 1
fi

POSTS_CSV="${SIM_DIR}/posts.csv"
TOX_CSV="${SIM_DIR}/toxigen.csv"
USERS_CSV="${SIM_DIR}/users.csv"

if [[ ! -f "${POSTS_CSV}" ]]; then
  echo "Missing posts.csv at ${POSTS_CSV}" >&2
  exit 1
fi

PY=${PYTHON:-python3}

echo "[1/7] Visualize simulation"
${PY} scripts/visualize_simulation2_additional.py --sim-dir "${SIM_DIR}" || echo "visualize_simulation2_additional.py failed" >&2

echo "[2/7] Additional plots"
if [[ -f "${USERS_CSV}" ]]; then
  ${PY} scripts/additional_plots.py --sim-dir "${SIM_DIR}" --out-dir "${SIM_DIR}" || echo "additional_plots.py failed" >&2
else
  echo "Skipping additional_plots.py (no users.csv in ${SIM_DIR})"
fi

echo "[3/7] Convergence entropy"
${PY} scripts/convergence_entropy_chains.py --posts-csv "${POSTS_CSV}" || echo "convergence_entropy_chains.py failed" >&2

echo "[4/7] NER (requires MADOC Voat parquet + toxigen.csv)"
VOAT_PQ="MADOC/voat-technology/voat_technology_madoc.parquet"
if [[ -f "${VOAT_PQ}" && -f "${TOX_CSV}" ]]; then
  ${PY} scripts/voat_ner_structure.py --sim2-posts "${POSTS_CSV}" --sim2-tox "${TOX_CSV}" --mode both --plot-compare --plot-bipartite || echo "voat_ner_structure.py failed" >&2
else
  echo "Skipping NER (requires ${VOAT_PQ} and ${TOX_CSV})"
fi

echo "[5/7] Topics (BERTopic)"
if [[ -f "${VOAT_PQ}" ]]; then
  ${PY} scripts/voat_topic_compare.py \
    --sim2-posts-csv "${POSTS_CSV}" \
    --outdir "${SIM_DIR}/topic_compare" \
    --min-topic-size 5 \
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
    --save-heatmap || echo "voat_topic_compare.py failed" >&2
else
  echo "Skipping topic comparison (missing ${VOAT_PQ})"
fi

echo "[6/7] Embeddings (similarity)"
if [[ -f "${VOAT_PQ}" && -f "${TOX_CSV}" ]]; then
  ${PY} scripts/voat_sim_embedding_similarity.py \
    --sim2-posts "${POSTS_CSV}" \
    --sim2-tox "${TOX_CSV}" \
    --mode both \
    --plot-tsne \
    --plot-umap \
    --tsne-perplexities 5,30,80 \
  --remove-top-k-tokens 50 || echo "voat_sim_embedding_similarity.py failed" >&2
else
  echo "Skipping embedding similarity (requires ${VOAT_PQ} and ${TOX_CSV})"
fi

echo "[7/7] Matching and smaller scripts"
if [[ -f "${VOAT_PQ}" && -f "${TOX_CSV}" ]]; then
  ${PY} scripts/sim_comments_to_voat_match.py --sim2-posts "${POSTS_CSV}" --sim2-tox "${TOX_CSV}" || echo "sim_comments_to_voat_match.py failed" >&2
  ${PY} scripts/voat_toxic_match.py --sim2-posts "${POSTS_CSV}" --sim2-tox "${TOX_CSV}" || echo "voat_toxic_match.py failed" >&2
  ${PY} scripts/export_chain_texts.py --posts-csv "${POSTS_CSV}" || echo "export_chain_texts.py failed" >&2
else
  echo "Skipping matching scripts (requires ${VOAT_PQ} and ${TOX_CSV})"
fi

echo "Wrap-up: outputs saved under ${SIM_DIR}."
echo "Done."
