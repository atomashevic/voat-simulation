#!/usr/bin/env bash
set -euo pipefail

# Runs the main analysis pipeline for a simulation directory (simulation).
# Usage:
#   bash scripts/run_sim_pipeline.sh simulation

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

SIM_ARG=${1:-}
SIM_DB_ARG=${2:-}
if [[ -z "${SIM_ARG}" ]]; then
  echo "Usage: $0 <simulation_dir_or_name> [simulation_sqlite]" >&2
  exit 1
fi

SIM_NAME=$(basename "${SIM_ARG}")
RESULTS_BASE="results"
BASIC_DIR="${RESULTS_BASE}/basic/${SIM_NAME}"
NETWORKS_DIR="${RESULTS_BASE}/networks/${SIM_NAME}"
TOXICITY_DIR="${RESULTS_BASE}/toxicity/${SIM_NAME}"
CONVERGENCE_DIR="${RESULTS_BASE}/convergence/${SIM_NAME}"
DIVERSITY_DIR="${RESULTS_BASE}/diversity/${SIM_NAME}"
SIMILARITY_DIR="${RESULTS_BASE}/similarity/${SIM_NAME}"
TOPIC_DIR="${RESULTS_BASE}/topic/${SIM_NAME}"
NER_DIR="${RESULTS_BASE}/ner/${SIM_NAME}"

mkdir -p "${BASIC_DIR}" "${NETWORKS_DIR}" "${TOXICITY_DIR}" "${CONVERGENCE_DIR}" \
  "${DIVERSITY_DIR}" "${SIMILARITY_DIR}" "${TOPIC_DIR}" "${NER_DIR}"

SOURCE_DIR=""
if [[ -d "${SIM_ARG}" && "${SIM_ARG}" != "${BASIC_DIR}" ]]; then
  SOURCE_DIR="${SIM_ARG}"
fi

if [[ -n "${SOURCE_DIR}" ]]; then
  for base_file in posts.csv users.csv news.csv; do
    if [[ -f "${SOURCE_DIR}/${base_file}" ]]; then
      cp -f "${SOURCE_DIR}/${base_file}" "${BASIC_DIR}/${base_file}"
    fi
  done
fi

SIM_DIR="${BASIC_DIR}"
POSTS_CSV="${SIM_DIR}/posts.csv"
USERS_CSV="${SIM_DIR}/users.csv"
TOX_CSV="${TOXICITY_DIR}/toxigen.csv"
VOAT_PQ="MADOC/voat-technology/voat_technology_madoc.parquet"

TOXIGEN_DB=""
if [[ -n "${SIM_DB_ARG}" ]]; then
  TOXIGEN_DB="${SIM_DB_ARG}"
else
  SQLITE_CANDIDATE=""
  if [[ -d "${SIM_DIR}" ]]; then
    SQLITE_CANDIDATE="$(find "${SIM_DIR}" -maxdepth 1 -type f -name '*.sqlite' -print -quit 2>/dev/null || true)"
  fi
  if [[ -z "${SQLITE_CANDIDATE}" && -n "${SOURCE_DIR}" ]]; then
    SQLITE_CANDIDATE="$(find "${SOURCE_DIR}" -maxdepth 1 -type f -name '*.sqlite' -print -quit 2>/dev/null || true)"
  fi
  if [[ -n "${SQLITE_CANDIDATE}" ]]; then
    TOXIGEN_DB="${SQLITE_CANDIDATE}"
  fi
fi

if [[ ! -f "${POSTS_CSV}" ]]; then
  echo "Missing posts.csv at ${POSTS_CSV}" >&2
  exit 1
fi

PY=${PYTHON:-python3}

echo "[1/11] Visualize simulation (basic outputs)"
${PY} scripts/visualize_simulation_additional.py --sim-dir "${SIM_DIR}" || echo "visualize_simulation_additional.py failed" >&2

echo "[2/11] Additional plots"
if [[ -f "${USERS_CSV}" ]]; then
  ${PY} scripts/additional_plots.py --sim-dir "${SIM_DIR}" --out-dir "${BASIC_DIR}" || echo "additional_plots.py failed" >&2
else
  echo "Skipping additional_plots.py (no users.csv in ${SIM_DIR})"
fi

echo "[3/11] Core-periphery network stats (enhanced, LCC)"
${PY} scripts/core_periphery_enhanced.py "${POSTS_CSV}" --output-dir "${NETWORKS_DIR}" || echo "core_periphery_enhanced.py failed" >&2

echo "[4/11] Comparative network analysis (Voat)"
# The current comparator script looks under 'simulation' and MADOC/voat-technology
# Guard to run only when SIM_DIR is simulation and Voat data directory exists.
if [[ "${SIM_NAME}" == "simulation" && -d "MADOC/voat-technology" ]]; then
  ${PY} scripts/comparative_network_analysis_voat.py || echo "comparative_network_analysis_voat.py failed" >&2
else
  echo "Skipping comparative_network_analysis_voat.py (requires SIM_DIR=simulation and MADOC/voat-technology)"
fi

echo "[5/11] Convergence entropy"
${PY} scripts/convergence_entropy_chains.py --posts-csv "${POSTS_CSV}" --outdir "${CONVERGENCE_DIR}" --device cpu || echo "convergence_entropy_chains.py failed" >&2

echo "[6/11] ToxiGen RoBERTa toxicity scoring"
if [[ -n "${TOXIGEN_DB}" && -f "${TOXIGEN_DB}" ]]; then
  ${PY} scripts/toxigen_roberta.py "${TOXIGEN_DB}" --output-dir "${TOXICITY_DIR}" --madoc-parquet "${VOAT_PQ}" --device auto --no-plots || echo "toxigen_roberta.py failed" >&2
else
  echo "Skipping toxigen_roberta.py (no sqlite found in ${SIM_DIR}; pass a path as the second argument)"
fi

echo "[7/11] NER (requires MADOC Voat parquet + toxigen.csv)"
if [[ -f "${VOAT_PQ}" && -f "${TOX_CSV}" ]]; then
  ${PY} scripts/voat_ner_structure.py --sim2-posts "${POSTS_CSV}" --sim2-tox "${TOX_CSV}" --mode both --plot-compare --plot-bipartite --outdir "${NER_DIR}" || echo "voat_ner_structure.py failed" >&2
else
  echo "Skipping NER (requires ${VOAT_PQ} and ${TOX_CSV})"
fi

echo "[8/11] Topics (BERTopic)"
if [[ -f "${VOAT_PQ}" ]]; then
  ${PY} scripts/voat_topic_compare.py \
    --sim2-posts-csv "${POSTS_CSV}" \
    --outdir "${TOPIC_DIR}" \
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
    --composite-alpha 0.85 \
    --save-heatmap || echo "voat_topic_compare.py failed" >&2
else
  echo "Skipping topic comparison (missing ${VOAT_PQ})"
fi

echo "[9/11] Embeddings (similarity)"
if [[ -f "${VOAT_PQ}" && -f "${TOX_CSV}" ]]; then
  ${PY} scripts/voat_sim_embedding_similarity.py \
    --sim2-posts "${POSTS_CSV}" \
    --sim2-tox "${TOX_CSV}" \
    --mode both \
    --device auto \
    --plot-tsne \
    --plot-umap \
    --tsne-perplexities 5,30,80 \
    --remove-top-k-tokens 50 \
    --out-json "${SIMILARITY_DIR}/embedding_similarity.json" \
    --plot-prefix "${SIMILARITY_DIR}/sim_voat_similarity" || echo "voat_sim_embedding_similarity.py failed" >&2
else
  echo "Skipping embedding similarity (requires ${VOAT_PQ} and ${TOX_CSV})"
fi

echo "[10/11] Matching and smaller scripts"
if [[ -f "${VOAT_PQ}" && -f "${TOX_CSV}" ]]; then
  ${PY} scripts/sim_comments_to_voat_match.py --sim2-posts "${POSTS_CSV}" --sim2-tox "${TOX_CSV}" --out "${SIMILARITY_DIR}/sim_to_voat_comment_matches.txt" || echo "sim_comments_to_voat_match.py failed" >&2
  ${PY} scripts/voat_toxic_match.py --sim2-posts "${POSTS_CSV}" --sim2-tox "${TOX_CSV}" --out "${SIMILARITY_DIR}/voat_toxic_matches.txt" || echo "voat_toxic_match.py failed" >&2
  ${PY} scripts/export_chain_texts.py --posts-csv "${POSTS_CSV}" --out-csv "${CONVERGENCE_DIR}/chain_texts.csv" || echo "export_chain_texts.py failed" >&2
else
  echo "Skipping matching scripts (requires ${VOAT_PQ} and ${TOX_CSV})"
fi

echo "[11/11] MADOC vs simulation comparison pipeline"
if [[ -d "MADOC/voat-technology" ]]; then
  ${PY} scripts/compare_madoc_to_sim.py \
    --sim-dirs "${SIM_DIR}" \
    --madoc-root "MADOC/voat-technology" \
    --out-dir "${BASIC_DIR}/madoc_compare" \
    --report-prefix "voat_vs_${SIM_NAME}" \
    --log-level INFO || echo "compare_madoc_to_sim.py failed" >&2
else
  echo "Skipping MADOC comparison (MADOC/voat-technology missing)"
fi

echo "Wrap-up: outputs saved under ${RESULTS_BASE}/{basic,networks,toxicity,convergence,diversity,similarity,topic,ner}/${SIM_NAME}."
echo "Done."
