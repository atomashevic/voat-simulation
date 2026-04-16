#!/usr/bin/env python3
"""
Create activity evolution panel with mean + 95% CI bands across 30 simulation runs.

Metrics:
    1) Cumulative posts
    2) Cumulative comments
    3) Cumulative users (by joined_on)
    4) Cumulative active users (ever posted/commented)

Output: 2x2 panel with shaded CI bands showing variability across runs.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_run_counts(run_dir: Path) -> pd.DataFrame:
    """Load simulation CSVs and build cumulative time series indexed by round."""
    posts_path = run_dir / "posts.csv"
    users_path = run_dir / "users.csv"

    if not posts_path.exists() or not users_path.exists():
        raise FileNotFoundError(f"Missing posts.csv or users.csv in {run_dir}")

    posts = pd.read_csv(posts_path)
    users = pd.read_csv(users_path)

    # Build a unified round index
    round_cols = [
        pd.Index(posts["round"].dropna().unique(), name="round"),
        pd.Index(users["joined_on"].dropna().unique(), name="round"),
    ]
    timeline = sorted(set().union(*round_cols))
    frame = pd.DataFrame(index=pd.Index(timeline, name="round"))

    # Posts and comments cumulative counts
    root_mask = posts["comment_to"] == -1
    posts_per_round = posts.loc[root_mask].groupby("round").size().sort_index()
    comments_per_round = (
        posts.loc[posts["comment_to"] != -1]
        .groupby("round")
        .size()
        .sort_index()
    )

    frame["cumulative_posts"] = posts_per_round.cumsum().reindex(frame.index).ffill().fillna(0)
    frame["cumulative_comments"] = (
        comments_per_round.cumsum().reindex(frame.index).ffill().fillna(0)
    )

    # Users cumulative counts by join round
    users_per_round = users.groupby("joined_on").size().sort_index()
    frame["cumulative_users"] = (
        users_per_round.cumsum().reindex(frame.index).ffill().fillna(0)
    )

    # Cumulative active users based on first activity round (post or comment)
    first_activity = (
        posts.dropna(subset=["round"])
        .groupby("user_id")["round"]
        .min()
        .value_counts()
        .sort_index()
    )
    frame["cumulative_active_users"] = (
        first_activity.cumsum().reindex(frame.index).ffill().fillna(0)
    )

    return frame.reset_index()


def load_all_runs(results_dir: Path) -> Tuple[List[pd.DataFrame], List[str]]:
    """Load cumulative counts from all run directories."""
    run_dfs = []
    run_names = []

    for run_dir in sorted(results_dir.glob("run*")):
        if not run_dir.is_dir():
            continue
        try:
            df = load_run_counts(run_dir)
            run_dfs.append(df)
            run_names.append(run_dir.name)
            logger.info("Loaded %s: %d rounds", run_dir.name, len(df))
        except Exception as e:
            logger.warning("Failed to load %s: %s", run_dir.name, e)

    return run_dfs, run_names


def align_runs(run_dfs: List[pd.DataFrame], n_points: int = 100) -> Dict[str, np.ndarray]:
    """Align runs to a normalized timeline (0-100%) and stack into arrays.
    
    Each run's cumulative series is interpolated to n_points evenly spaced 
    positions along its timeline, allowing comparison across runs with 
    different round ranges.
    """
    metrics = ["cumulative_posts", "cumulative_comments", "cumulative_users", "cumulative_active_users"]
    
    # Use normalized progress (0 to 1) with n_points steps
    normalized_x = np.linspace(0, 1, n_points)
    result = {"rounds": normalized_x}  # Now represents progress fraction
    
    for metric in metrics:
        stacked = []
        for df in run_dfs:
            # Get the cumulative series for this run
            rounds = df["round"].values
            values = df[metric].values
            
            # Normalize rounds to 0-1 scale based on this run's range
            round_min, round_max = rounds.min(), rounds.max()
            if round_max > round_min:
                normalized_rounds = (rounds - round_min) / (round_max - round_min)
            else:
                normalized_rounds = np.zeros_like(rounds)
            
            # Interpolate to common normalized x positions
            interpolated = np.interp(normalized_x, normalized_rounds, values)
            stacked.append(interpolated)
        
        arr = np.array(stacked)  # shape: (n_runs, n_points)
        result[metric] = arr
        
        # Log final values for this metric
        final_mean = np.mean(arr[:, -1])
        logger.info("%s: final mean = %.1f", metric, final_mean)

    return result


def compute_percentile_bands(
    data: np.ndarray,
    lower_pct: float = 5,
    upper_pct: float = 95,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean and percentile bands showing run-to-run spread.

    data: shape (n_runs, n_rounds)
    Returns: (mean, lower_band, upper_band) each of shape (n_rounds,)
    
    This shows the actual spread of individual runs (5th-95th percentile),
    NOT the confidence interval of the mean.
    """
    means = np.mean(data, axis=0)
    lower_band = np.percentile(data, lower_pct, axis=0)
    upper_band = np.percentile(data, upper_pct, axis=0)
    
    return means, lower_band, upper_band


def plot_panel(aligned: Dict[str, np.ndarray], output_path: Path, n_runs: int) -> None:
    """Render the 2x2 panel with CI bands."""
    plt.style.use("default")
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    metrics = [
        ("cumulative_posts", "Cumulative Root Posts", "Count"),
        ("cumulative_comments", "Cumulative Comments", "Count"),
        ("cumulative_users", "All Users (joined)", "Users"),
        ("cumulative_active_users", "Active Users (posted/commented)", "Users"),
    ]

    # X-axis is normalized progress (0-100%)
    progress_pct = aligned["rounds"] * 100

    for ax, (color, meta) in zip(axes.flat, zip(colors, metrics)):
        metric, title, ylabel = meta
        data = aligned[metric]

        means, lower_band, upper_band = compute_percentile_bands(data)

        ax.fill_between(progress_pct, lower_band, upper_band, color=color, alpha=0.3, label="5-95th pct")
        ax.plot(progress_pct, means, color=color, linewidth=2.5, label="Mean")

        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Simulation Progress (%)")
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#fafafa")

        # Final value annotation (should match aggregated_30runs.json)
        final_val = means[-1] if len(means) > 0 and not np.isnan(means[-1]) else 0
        label = f"{int(round(final_val)):,}"
        ax.text(
            0.02,
            0.88,
            label,
            transform=ax.transAxes,
            ha="left",
            color=color,
            fontsize=11,
            weight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, boxstyle="round,pad=0.3"),
        )

        # Add legend
        ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    fig.suptitle(f"Simulation Growth Overview (n={n_runs} runs, mean with 5-95th percentile)", fontsize=14, weight="bold")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved panel to %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create cumulative growth panel with CIs for 30 simulation runs."
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
        default=Path("workshop/figures/activity_growth_30runs_ci.png"),
        help="Output file for the generated figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    run_dfs, run_names = load_all_runs(results_dir)
    if len(run_dfs) < 2:
        logger.error("Need at least 2 runs to compute CIs, found %d", len(run_dfs))
        return

    logger.info("Loaded %d runs", len(run_dfs))
    aligned = align_runs(run_dfs)
    plot_panel(aligned, output_path, n_runs=len(run_dfs))


if __name__ == "__main__":
    main()

