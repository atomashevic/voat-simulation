#!/usr/bin/env bash
set -uo pipefail

# Zip key figures from a simulation folder into one archive.
#
# Usage:
#   scripts/zip_simulation_figures.sh [SIM_DIR] [OUTPUT_ZIP]
#
# Defaults:
#   SIM_DIR     = simulation3
#   OUTPUT_ZIP  = archives/<sim_basename>_figures.zip
#
# The script attempts to identify figures based on the project’s descriptions
# and typical output filenames, copying them into a staging "figures" folder
# with canonical names before zipping. Missing figures are skipped with a warning.

SIM_DIR=${1:-simulation3}
SIM_DIR=${SIM_DIR%/}

if [[ ! -d "$SIM_DIR" ]]; then
  echo "Error: simulation directory '$SIM_DIR' not found" >&2
  exit 1
fi

SIM_BASE=$(basename "$SIM_DIR")
DEFAULT_OUT="archives/${SIM_BASE}_figures.zip"
OUT_ZIP=${2:-$DEFAULT_OUT}

mkdir -p "$(dirname "$OUT_ZIP")"

# Staging directory
STAGE_DIR=$(mktemp -d)
trap 'rm -rf "$STAGE_DIR"' EXIT
FIG_DIR="$STAGE_DIR/figures"
mkdir -p "$FIG_DIR"

# Enable recursive globbing
shopt -s globstar nullglob

# Helper: pick first existing file matching any of the patterns (searched recursively under SIM_DIR)
find_first_match() {
  local pattern
  for pattern in "$@"; do
    # Iterate matched files in lexical order for determinism
    local matches=("$SIM_DIR"/**/$pattern)
    if (( ${#matches[@]} > 0 )); then
      printf '%s\n' "${matches[0]}"
      return 0
    fi
  done
  return 1
}

# Map of expected canonical filenames -> candidate glob patterns
# These reflect the figure list provided in main.tex and typical outputs in simulation folders.
declare -A TARGETS

# Distribution of Posts per User (KDE)
TARGETS[voat_post_distribution_kde.png]="posts_distribution_kde.png posts_distribution.png posts_distribution_log.png"

# Degree Distribution (log–log) for interaction network
TARGETS[voat_degree.png]="degree_distribution_loglog.png degree_distribution.png *degree*loglog*.png *degree_distribution*.png"

# Core–Periphery Structure (simulation)
TARGETS[voat_sim_core_periphery.png]="enhanced_core_periphery_visualization.png additional_lcc_core_periphery_colored_weighted_size.png core*periphery*visualization.png"

# Core–Periphery Structure (matched real Voat sample) — ONLY accept enhanced version
TARGETS[voat_real_core_periphery.png]="enhanced_core_periphery_visualization.png"

# Toxicity Distribution (KDE)
TARGETS[voat_toxicity_kde.png]="plots/*toxicity*kde*.png plots/comparative_kde.png"

# Named‑Entity Frequencies (top‑30 ORG/GPE)
TARGETS[voat_comments_ner.png]="ner/voat_comments_ner.png ner/comments_entity_counts_compare_top30.png ner/*comments*ner*.png"

# Convergence Entropy subfigures
TARGETS[lag_vs_entropy.png]="convergence_entropy/lag_vs_entropy.png"
TARGETS[entropy_dist_inter_vs_intra.png]="convergence_entropy/entropy_dist_inter_vs_intra.png"

echo "Collecting figures from '$SIM_DIR'..."

copied_count=0
missing=()

for canonical in "${!TARGETS[@]}"; do
  # Read candidates into array
  IFS=' ' read -r -a patterns <<< "${TARGETS[$canonical]}"
  # Temporarily relax -e to allow not-found cases without aborting
  set +e
  # Special-case: allow sourcing the real Voat core-periphery figure from MADOC samples
  if [[ "$canonical" == "voat_real_core_periphery.png" ]]; then
    src=""
    status=1
    for base in "MADOC/voat-technology/sample_1" "MADOC/voat-technology/sample_2" "$SIM_DIR"; do
      for pattern in "${patterns[@]}"; do
        matches=("$base"/**/$pattern)
        if (( ${#matches[@]} > 0 )); then
          src="${matches[0]}"
          status=0
          break 2
        fi
      done
    done
  else
    src=$(find_first_match "${patterns[@]}")
    status=$?
  fi
  if [[ $status -eq 0 && -n "$src" ]]; then
    dest="$FIG_DIR/$canonical"
    mkdir -p "$(dirname "$dest")"
    cp -f "$src" "$dest"
    echo "  + Added: $canonical  <-  ${src#$SIM_DIR/}"
    ((copied_count++))
  else
    echo "  ! Missing: $canonical (no match for: ${TARGETS[$canonical]})" >&2
    missing+=("$canonical")
  fi
done

if (( copied_count == 0 )); then
  echo "Error: No figures were found to archive. Aborting." >&2
  exit 2
fi

( cd "$STAGE_DIR" && zip -q -r "$(pwd)/archive.zip" figures )
mv -f "$STAGE_DIR/archive.zip" "$OUT_ZIP"

echo "\nCreated archive: $OUT_ZIP"
if (( ${#missing[@]} > 0 )); then
  echo "Note: Skipped ${#missing[@]} missing figure(s): ${missing[*]}"
fi

exit 0
