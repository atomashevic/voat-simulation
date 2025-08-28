# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Environment

**Python Environment Setup:**
```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
```

**Installing Dependencies:**
```bash
# Common dependencies for most scripts
pip install pandas numpy matplotlib networkx python-louvain

# NLP and analysis dependencies  
pip install transformers torch

# Moderation API (requires API key)
pip install openai
```

**Running Scripts:**
```bash
python scripts/<script_name>.py [args]

# Examples:
python scripts/generate_csvs.py
python scripts/map_network.py simulation1/posts.csv
python scripts/reddit_samples.py
python scripts/madoc-tech.py
```

## Code Quality

**Formatting and Linting:**
```bash
black -l 88 scripts/
ruff check scripts/  # or flake8 scripts/
```

**Testing:**
```bash
pytest -q  # Run tests (if tests/ directory exists)
```

## Project Architecture

**Core Directories:**
- `scripts/` - Python analysis tools for network mapping, reputation analysis, data sampling, and moderation
- `simulation1/`, `simulation2/` - Simulation runs with configuration files and output data
- `MADOC/` - Source datasets in Parquet format organized by platform (reddit-technology, voat-technology) with sample subdirectories

**Key Configuration Files:**
- `simulation*/config/config.json` - Main simulation parameters (agents, actions, servers)
- `simulation*/config/prompts.json` - LLM prompts for agent behavior
- `simulation*/config/rss_feeds.json` - News sources for content
- `simulation*/config/urls.txt` - External URLs for sharing

**Data Flow:**
1. **Input**: Raw social media data (Parquet files in `MADOC/`)
2. **Processing**: Analysis scripts process data to extract metrics, build networks, calculate reputation
3. **Output**: CSV files, network visualizations, statistical plots, and analysis reports in simulation directories

**Key Analysis Components:**
- **Network Analysis** (`map_network.py`): Builds interaction networks, calculates centrality measures, creates k-core visualizations
- **Reputation Systems** (`reddit_reputation.py`): Implements engagement and popularity-based reputation scoring
- **Data Sampling** (`reddit_samples.py`, `voat_samples.py`): Extracts representative samples from large datasets
- **Content Moderation** (`openai-moderation.py`, `toxigen_roberta.py`): Toxicity analysis using various models

**Configuration-Driven Architecture:**
- Simulation parameters are externalized in JSON config files
- Agent behaviors, LLM endpoints, and activity patterns are configurable
- Supports different client implementations (`YClientBase`) for various social platforms

## Environment Variables

**API Keys (never commit these):**
```bash
export OPENAI_API_KEY="your_key_here"  # For moderation scripts
```

## Common Workflows

**Analyzing New Dataset:**
1. Place dataset in `MADOC/<platform-name>/`
2. Create samples: `python scripts/<platform>-samples.py`
3. Generate network: `python scripts/map_network.py path/to/posts.csv`
4. Analyze reputation: `python scripts/reddit_reputation.py`

**Running Simulation Analysis:**
1. Configure simulation in `simulation*/config/config.json`
2. Generate analysis outputs with relevant scripts
3. Network visualizations and stats will be saved in simulation directory

**Data Processing Pipeline:**
- Raw data → Sampling → Network analysis → Reputation calculation → Visualization
- All intermediate results are preserved as CSV/JSON files for reproducibility
