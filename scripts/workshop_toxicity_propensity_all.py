#!/usr/bin/env python3
"""
Generate boxplot of mean toxicity per user grouped by toxicity propensity level.
Pools data from all 30 simulation runs.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


LEVEL_ORDER = ["absolutely no", "no", "moderately"]
PASTEL_MAP = {
    "absolutely no": "#8dd3c7",
    "no": "#ffffb3",
    "moderately": "#fb8072",
}


def load_run_data(run_dir: Path) -> pd.DataFrame:
    """Load and merge posts, users, toxigen data from a single run."""
    posts_path = run_dir / "posts.csv"
    users_path = run_dir / "users.csv"
    toxigen_path = run_dir / "toxigen.csv"
    
    for path in (posts_path, users_path, toxigen_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")
    
    posts = pd.read_csv(posts_path)
    users = pd.read_csv(users_path)
    toxigen = pd.read_csv(toxigen_path)
    
    # Normalize column names
    posts.columns = posts.columns.str.lower()
    users.columns = users.columns.str.lower()
    toxigen.columns = toxigen.columns.str.lower()
    
    # Prepare for merge
    toxigen = toxigen[["id", "toxicity"]].rename(columns={"id": "post_id", "toxicity": "toxicity_score"})
    posts = posts[["id", "user_id"]].rename(columns={"id": "post_id"})
    users = users[["id", "toxicity"]].rename(columns={"id": "user_id", "toxicity": "user_level"})
    
    # Merge
    merged = toxigen.merge(posts, on="post_id", how="inner").merge(users, on="user_id", how="inner")
    merged["toxicity_score"] = pd.to_numeric(merged["toxicity_score"], errors="coerce")
    merged = merged.dropna(subset=["toxicity_score", "user_level"])
    
    return merged


def load_all_runs(results_dir: Path) -> pd.DataFrame:
    """Load and pool data from all simulation runs."""
    all_data = []
    
    for run_dir in sorted(results_dir.glob("run*")):
        if not run_dir.is_dir():
            continue
        try:
            df = load_run_data(run_dir)
            df["run"] = run_dir.name
            all_data.append(df)
            logger.debug("Loaded %d rows from %s", len(df), run_dir.name)
        except Exception as e:
            logger.warning("Failed to load %s: %s", run_dir.name, e)
    
    if not all_data:
        raise ValueError("No data loaded from any run")
    
    result = pd.concat(all_data, ignore_index=True)
    logger.info("Loaded %d total rows from %d runs", len(result), len(all_data))
    return result


def compute_user_means(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute mean toxicity per user (pooled across runs)."""
    # Group by user_id across all runs (same user_id in different runs = different users)
    grouped = (
        frame.groupby(["run", "user_id", "user_level"], dropna=False)["toxicity_score"]
        .mean()
        .reset_index()
        .rename(columns={"toxicity_score": "mean_toxicity"})
    )
    logger.info("Computed means for %d users", len(grouped))
    return grouped


def validate_levels(df: pd.DataFrame) -> List[str]:
    """Validate and return ordered toxicity levels."""
    observed = set(df["user_level"].str.lower().unique())
    expected = set(l.lower() for l in LEVEL_ORDER)
    
    if not expected.issubset(observed):
        missing = expected - observed
        logger.warning("Missing expected levels: %s", missing)
    
    # Normalize levels
    df["user_level"] = df["user_level"].str.lower()
    
    # Filter to expected levels only
    df_filtered = df[df["user_level"].isin([l.lower() for l in LEVEL_ORDER])]
    
    return LEVEL_ORDER


def plot_boxplot(df: pd.DataFrame, levels: List[str], output_path: Path, n_runs: int) -> None:
    """Render and save the boxplot of mean toxicity per user."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(10, 7))
    
    palette = [PASTEL_MAP[level] for level in levels]
    
    # Filter data to valid levels
    df_plot = df[df["user_level"].isin([l.lower() for l in levels])].copy()
    
    # Sample data to reduce clutter - show every 5th point
    df_sampled = df_plot.groupby("user_level", group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), len(x) // 5 + 1), random_state=42)
    )
    
    # Stripplot with colors matching groups, black edge circles
    sns.stripplot(
        data=df_sampled,
        x="user_level",
        y="mean_toxicity",
        hue="user_level",
        order=[l.lower() for l in levels],
        hue_order=[l.lower() for l in levels],
        palette=palette,
        jitter=0.35,
        alpha=0.4,
        size=6,
        edgecolor="black",
        linewidth=0.5,
        legend=False,
        ax=ax,
    )
    
    # Add mean markers - large diamond with black edge
    for i, level in enumerate(levels):
        subset = df_plot[df_plot["user_level"] == level.lower()]["mean_toxicity"]
        mean_val = subset.mean()
        
        # Large diamond marker for mean
        ax.scatter([i], [mean_val], s=250, c=palette[i], marker='D', 
                   edgecolor='black', linewidth=2, zorder=10)
        
        # Horizontal line at mean
        ax.hlines(mean_val, i - 0.4, i + 0.4, colors='black', linewidth=2, zorder=9)
    
    # Format x-axis labels
    ax.set_xticklabels([l.title() for l in levels])
    
    ax.set_xlabel("User Toxicity Propensity", fontsize=12)
    ax.set_ylabel("Mean Toxicity per User", fontsize=12)
    ax.set_title(f"Mean Toxicity by User Propensity Level\n(n={n_runs} simulation runs, {len(df_plot):,} users)", 
                 fontsize=13, weight="bold")
    ax.grid(False)
    
    # Add mean value labels above the diamonds
    for i, level in enumerate(levels):
        subset = df_plot[df_plot["user_level"] == level.lower()]["mean_toxicity"]
        mean_val = subset.mean()
        ax.text(i, mean_val + 0.03, f"μ = {mean_val:.3f}", 
                ha="center", fontsize=11, fontweight='bold', color="black")
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot to %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate toxicity by propensity boxplot from all simulation runs."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing run01, run02, ... subdirectories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("workshop/figures/toxicity_by_propensity_30runs.png"),
        help="Output file for the generated figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    
    # Load all runs
    frame = load_all_runs(results_dir)
    n_runs = frame["run"].nunique()
    
    # Compute user means
    user_means = compute_user_means(frame)
    
    # Validate levels
    levels = validate_levels(user_means)
    
    # Plot
    plot_boxplot(user_means, levels, output_path, n_runs)


if __name__ == "__main__":
    main()

