#!/usr/bin/env bash
set -euo pipefail

MODE="show"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

if [[ "$MODE" != "show" ]]; then
    echo "Only --mode show is currently supported." >&2
    exit 1
fi

cat <<'EOF'
External reproducibility inputs referenced by this repository:

- Y Social upstream:
  https://github.com/YSocialTwin/YSocial

- Modified YClient:
  https://github.com/atomashevic/YClient-Reddit

- Simulation LLM:
  ikiru/Dolphin-Mistral-24B-Venice-Edition (Ollama)

- Toxicity model:
  tomh/toxigen_roberta

- Embedding model:
  sentence-transformers/all-MiniLM-L6-v2

- Convergence entropy encoder:
  bert-base-uncased

The raw Voat parquet windows, benchmark SQLite files, and sensitivity SQLite files
used by the paper are already retained inside this repository.
EOF
