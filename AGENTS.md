# Repository Guidelines

## Project Structure & Modules
- `scripts/`: Python analysis and simulation utilities (e.g., network mapping, reputation, moderation, sampling). Run files with `python scripts/<file>.py`.
- `simulation1/`: Example run outputs and configuration. Important: `simulation1/config/{config.json,prompts.json,rss_feeds.json,urls.txt}`.
- `MADOC/`: Source datasets (Parquet and small samples) used by scripts. Do not commit additional large files here.

## Build, Test, and Development
- Environment: Python 3.9+ recommended. Create a venv: `python3 -m venv .venv && source .venv/bin/activate`.
- Dependencies: install as needed per script (common: pandas, numpy, networkx, matplotlib; NLP: transformers/torch; moderation: openai).
- Run examples:
  - Generate CSVs: `python scripts/generate_csvs.py`
  - Map network/centrality: `python scripts/map_network.py`
  - Reddit samples: `python scripts/reddit_samples.py`
  Configure via files in `simulation1/config/` when applicable.

### End-to-end Pipeline
- Run the main analysis pipeline for a simulation directory (outputs saved under the simulation folder):
  - `bash scripts/run_sim_pipeline.sh simulation2`
  - `bash scripts/run_sim_pipeline.sh simulation3`
- Pipeline order:
  1) Visualize simulation (`visualize_simulation2_additional.py`)
  2) Additional plots (`additional_plots.py`)
  3) Convergence entropy (`convergence_entropy_chains.py`)
  4) NER (simulation vs Voat) (`voat_ner_structure.py`)
  5) Topics (BERTopic) (`voat_topic_compare.py`)
  6) Embeddings similarity (`voat_sim_embedding_similarity.py`)
  7) Matching and text exports (`sim_comments_to_voat_match.py`, `voat_toxic_match.py`, `export_chain_texts.py`)

## Coding Style & Naming
- Follow PEP 8 with 4-space indentation.
- Use `snake_case` for modules/functions, `CapWords` for classes; prefer underscores over hyphens for new filenames.
- Keep functions small and pure when possible; add short docstrings describing inputs/outputs.
- Optional format/lint: `black -l 88 scripts/` and `ruff`/`flake8` before opening a PR.

## Testing Guidelines
- No formal suite exists yet. If adding tests, place them in `tests/` and name files `test_*.py`; use `pytest`.
- Seed randomness and use small fixtures (e.g., `MADOC/*/sample_*`) to keep tests fast and deterministic.
- Run tests: `pytest -q`.

## Commit & Pull Requests
- Commits: write imperative messages; prefer Conventional-style types (`feat:`, `fix:`, `docs:`, `refactor:`).
- PRs should include: concise description, commands run, sample output/plots (paths under `simulation1/`), and any updated configs. Link related issues.
- Avoid committing secrets or large binaries; add new data to `.gitignore` and document how to regenerate it.

## Security & Configuration
- Never hardcode API keys or tokens. Use environment variables (e.g., `export OPENAI_API_KEY=...`) for scripts like `openai-moderation.py`.
- Use relative paths (`pathlib.Path`) instead of machine-specific absolute paths.
