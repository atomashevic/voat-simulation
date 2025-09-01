# ysocial-simulations

Scripts and supporting modules to (a) sample and analyze real Reddit / Voat datasets (MADOC exports), (b) analyze and visualize simulation outputs, and (c) compare structure, content, reputation, toxicity, and topic dynamics across communities and between real and simulated corpora.

The repository is script–oriented: each file is a self‑contained step or analytical module.

---

## 1. High-Level Conceptual Workflow

1. Data acquisition
   - Voat data from MADOC dataset `voat_technology_madoc.parquet` lives in `MADOC/voat-technology/` directory.
   - Simulation YSocial database file lives under `simulation/`.

2. Real community temporal sampling & descriptive analytics
   - Get Voat samples:   `scripts/voat_samples.py` (creates `MADOC/voat-technology/sample_X/`)

3. Basic simulation database extraction
   - `scripts/sim_stats.py` → exports `results/<sim_name>/posts.csv`, `users.csv`, `news.csv` and figures.
   - `scripts/generate_csvs.py` → per-user CSVs and hierarchical thread text exports.

4. Text preprocessing & corpus preparation
   - `scripts/preprocess_simulation_data.py` → cleans simulated posts/comments into a modeling-ready CSV.
   - URL extraction: `scripts/extract_voat_urls.py`

5. Network construction & core–periphery inference
   - Baseline: `scripts/core_periphery_network.py` / `scripts/core-periphery-network-real.py`
   - Enhanced stochastic block model (SBM) ensemble: `scripts/core_periphery_enhanced*.py` (+ `scripts/core_periphery_sbm/`)
   - User attribute association tests: `scripts/core_periphery_user_attribute_tests.py`

7. Comparative structural analysis
   - Simulation vs Voat:   `scripts/comparative_network_analysis_voat.py`
   - Generic network mapping: `scripts/map_network.py`

8. Convergence / conversational entropy & alternating chains
   - Chain extraction & metrics: `scripts/convergence_entropy_chains.py`
   - Legacy entropy kernels: `scripts/entropy.py`
   - AB alternating chain export: `scripts/export_chain_texts.py`

9. Topic modeling & semantic similarity
   - Voat vs Simulation advanced comparison: `scripts/voat_topic_compare.py`
   - Basic BERTopic pipeline: `scripts/topic_compare_basic.py`
   - Simplified lightweight comparison: `scripts/simplified_topic_compare.py`
   - Embedding similarity distributions: `scripts/voat_sim_embedding_similarity.py`
   - Text cleaning helper (title/body split): `scripts/preprocess_simulation_data.py`

10. Toxicity analysis & plots
    - KDE & ECDF panel: `scripts/toxicity_kde_ecdf.py`
    - Single-panel posts vs comments: `scripts/toxicity_kde_posts_vs_comments.py`
    - Targeted toxic pair matching: `scripts/voat_toxic_match.py`, `scripts/sim_comments_to_voat_match.py`

11. Named Entity / structural Voat vs simulation diagnostics
    - `scripts/voat_ner_structure.py`

12. Integrated simulation pipeline runner
    - `scripts/run_sim_pipeline.sh` orchestrates many of the above (visuals, network, topics, embeddings, matching).

13. Panel / figure utilities & miscellany
    - `scripts/panel-figure.py`, `scripts/posts_per_user_kde_voat_vs_sim.py`, `scripts/additional_plots.py`, `scripts/visualize_simulation_additional.py`, etc.

14. Archival / experimental
    - `scripts/archive/` (legacy exploratory scripts). Avoid relying on these for production runs.

---


## 2. Detailed Usage Examples

Below are more explicit runs, assuming you are in repository root.


### 4.1 Voat Sampling

```bash
python scripts/voat_samples.py
```

### 4.2 Simulation Stats from SQLite

```bash
python scripts/sim_stats.py path/to/simulation.sqlite
# Produces: results/<db_basename>/posts.csv users.csv news.csv + figures
```

### 4.3 Generate User & Thread CSV/TXT

```bash
python scripts/generate_csvs.py results/<db_basename>/posts.csv --output simulation_extracted
```

### 4.4 Clean Simulation Text

```bash
python scripts/preprocess_simulation_data.py \
  --input simulation/posts.csv \
  --output simulation/clean_text.csv \
  --text-column tweet --min-length 25
```
### 4.5 Core–Periphery Network (Simulation CSV)

```bash
python scripts/core_periphery_network.py simulation/posts.csv simulation/core_periphery
```

(Exact CLI arguments may depend on internal argument parsing in the script—open the script if unsure.)

### 4.6 Enhanced SBM Ensemble (Voat)

```bash
python scripts/core_periphery_enhanced_voat.py MADOC/voat-technology/sample_1/voat_sample_1.parquet voat_cp_enhanced
```

### 4.7 Comparative Network (Voat vs Simulation)

```bash
python scripts/comparative_network_analysis_voat.py
# Assumes data directories exist: simulation/ and MADOC/voat-technology/
```

### 4.10 Convergence / Entropy

```bash
python scripts/convergence_entropy_chains.py --posts-csv simulation/posts.csv \
  --out-dir simulation/convergence
```

### 4.11 Topic Comparison (Voat vs Simulation)

Simplified version (when you only want a quick match):

```bash
python scripts/simplified_topic_compare.py \
  --corpus1 MADOC/voat-technology/sample_1/voat_sample_1.parquet \
  --corpus2 simulation/clean_text.csv \
  --corpus2-column full_text \
  --min-topic-size 20 \
  --nr-topics auto \
  --output-dir topic_compare_simple
```

Advanced Voat vs Simulation:

```bash
python scripts/voat_topic_compare.py \
  --sim2-posts-csv simulation/posts.csv \
  --outdir simulation/topic_compare \
  --min-topic-size 5 \
  --drop-header-rows \
  --min-doc-chars 25
```

### 4.12 Toxicity KDE

```bash
python scripts/toxicity_kde_ecdf.py --sim-dir simulation
python scripts/toxicity_kde_posts_vs_comments.py --sim-dir simulation
```

### 4.13 Toxic Matching

```bash
python scripts/voat_toxic_match.py \
  --sim2-posts simulation/posts.csv \
  --sim2-tox simulation/toxigen.csv \
  --madoc-parquet MADOC/voat-technology/sample_1/voat_sample_1.parquet \
  --topn 20
```

### 4.14 Embedding Similarity (Voat vs Simulation)

```bash
python scripts/voat_sim_embedding_similarity.py \
  --sim2-posts simulation/posts.csv \
  --sim2-tox simulation/toxigen.csv \
  --mode both \
  --plot-tsne --plot-umap
```
