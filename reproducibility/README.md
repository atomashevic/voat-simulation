# Reproducibility Guide

This directory documents the retained public reproducibility package.

## Scope

The cleaned public repository keeps:

- the raw Voat parquet windows used by the paper
- the benchmark and sensitivity SQLite trajectory databases
- the small machine-readable outputs backing the reported paper/SI values
- the curated scripts used to generate those outputs

It does not keep exploratory outputs, development-era plots, or manuscript assets.

## Public workflows

### 1. Audit mode

Use this when you want to inspect the reported numbers without rerunning the full analysis stack.

- benchmark tables: `artifacts/final/benchmark/`
- sensitivity summaries: `artifacts/final/sensitivity/`
- topic/embedding tables: `artifacts/final/topic_embedding/`
- power analysis: `artifacts/final/power_analysis/`
- tail-shape summaries: `artifacts/final/tail_shape/`

The artifact manifest is:

- `artifacts/final/index.json`

### 2. Trajectory rerun mode

Use this when you want to regenerate run-side outputs from the retained SQLite databases.

- benchmark run DBs: `data/benchmark_runs/`
- sensitivity run DBs: `data/sensitivity_runs/`

Wrapper entrypoint:

```bash
bash reproducibility/build_final_artifacts.sh --mode rerun --dry-run
```

The wrapper is conservative by design. It documents and launches the retained run-side pipeline, but does not attempt to rebuild every empirical-window intermediate automatically inside the repository tree.

## Empirical-window provenance

### Calibration windows

The reported calibration table is backed by the 10-window calibration bundle retained under:

- `data/voat_windows/calibration/`

This bundle matches the values reported in the manuscript’s calibration table and is therefore retained verbatim as the authoritative calibration input set.

### Validation windows

The 30-window benchmark baseline is backed by:

- `data/voat_windows/validation/`

These windows were copied from the non-overlapping family with the public mapping:

- source `sample_2..31`
- public `validation/window_01..window_30`

That mapping is recorded explicitly in:

- `data/voat_windows/validation/manifest.json`

## Environment files

The retained environment descriptors from the paper repo are copied here:

- `analysis/environment_analysis.yml`
- `analysis/requirements_analysis.txt`
- `simulation/environment_simulation.yml`
- `simulation/requirements_client.txt`
- `simulation/requirements_server.txt`
- `model_hashes.md`

These files document the originally used simulation and analysis environments. They are preserved as provenance, not as a claim that every environment is reproducible byte-for-byte on every machine today.

## External code and models

Pinned external inputs are documented in:

- `external_inputs.yaml`

This includes:

- Y Social related repositories
- model identifiers
- seed ranges
- retained condition mappings

