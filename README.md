# voat-simulation

Reviewer-facing code and reproducibility package for:

> "Towards Operational Validation of LLM-Agent Social Simulations: A Replicated Study of a Voat-like Technology Forum"

This repository is the public **code + reproducibility** companion. The manuscript, SI, response letter, and rendered submission figures live in the separate paper repository:

- `voat-simulation-paper`: `git@github.com:atomashevic/voat-simulation-paper.git`

## What this repo keeps

- `data/voat_windows/calibration/`
  - the 10 raw Voat parquet windows used for the reported calibration table
- `data/voat_windows/validation/`
  - the 30 raw matched non-overlapping Voat parquet windows used for the benchmark baseline
- `data/voat_windows/combined/`
  - compact combined Voat corpus parquets retained for legacy topic, toxicity, and embedding pipelines
- `data/benchmark_runs/`
  - the 30 SQLite trajectory databases for the benchmark runs
- `data/sensitivity_runs/`
  - one authoritative SQLite trajectory database per retained sensitivity run
  - only the `server` DB is kept for each run
- `data/url_catalog/`
  - the Voat URL catalog used for link seeding
- `artifacts/final/`
  - small machine-readable summaries that back the reported paper/SI numbers
- `scripts/`
  - the curated analysis and export scripts needed for the retained workflow
- `reproducibility/`
  - environment metadata, model provenance, and public reproducibility manifests

## What this repo intentionally removes

- old MADOC samples and auxiliary sample outputs not used by the paper
- duplicated client/server sensitivity DB pairs
- generated figures, plots, and exploratory outputs
- large intermediate `results/`, `simulation/`, `backup/`, and `workshop/` trees
- manuscript sources and submission assets

The goal is that a reviewer or reader can clone this repository, inspect the exact retained raw inputs and trajectory databases, and audit the reported summaries without pulling several gigabytes of unrelated history.

## Repository layout

```text
data/
  benchmark_runs/        30 benchmark SQLite trajectories
  sensitivity_runs/      one server SQLite per sensitivity run
  url_catalog/           Voat URL catalog used for link seeding
  voat_windows/
    calibration/         10 raw calibration parquets
    validation/          30 raw validation parquets
artifacts/
  final/                 frozen machine-readable summaries used by the paper/SI
reproducibility/         manifests, environments, and wrapper scripts
scripts/                 curated public analysis scripts
```

## Reproducibility modes

Two public workflows are supported.

1. `Audit mode`
   - inspect `artifacts/final/` directly
   - verify the reported benchmark, sensitivity, topic/embedding, power, and tail-shape summaries

2. `Trajectory rerun mode`
   - regenerate run-side analysis outputs from the retained benchmark and sensitivity SQLite files
   - compare regenerated summaries against the frozen artifacts

The retained raw Voat windows are provided so the empirical baseline is auditable from the exact windows used in the paper. The legacy window-analysis scripts in `scripts/` expect transient per-window workspaces and are documented in `reproducibility/README.md`.

## Key retained datasets

- Calibration windows:
  - sourced from the 10-window calibration bundle that matches the reported calibration table
- Validation windows:
  - sourced from the non-overlapping family used for the 30-window benchmark baseline
  - the public mapping is `sample_2..31 -> validation windows 1..30`
- Benchmark seeds:
  - runs `01..30` correspond to seeds `42..71`
- Sensitivity conditions:
  - `c0` baseline
  - `c1` neutral persona
  - `c2` no-politics persona
  - `c3` low temperature
  - `c4` high temperature
  - `c5` flat budget slope
  - `c6` steep budget slope
  - `c7` low comment-to-post ratio
  - `c8` high comment-to-post ratio
  - `c9` low churn
  - `c10` high churn

## Quick start

Read:

- `reproducibility/README.md`
- `reproducibility/external_inputs.yaml`
- `artifacts/final/index.json`

For a simple audit:

```bash
bash reproducibility/build_final_artifacts.sh --mode audit
```

For a dry-run of the retained pipeline:

```bash
bash reproducibility/build_final_artifacts.sh --mode rerun --dry-run
```

## Notes

- This repository is intentionally optimized for public inspection, not for storing every intermediate file ever produced during development.
- The raw manuscript figures are not duplicated here. They remain in the paper repository.
