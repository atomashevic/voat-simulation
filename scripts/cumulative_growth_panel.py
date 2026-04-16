#!/usr/bin/env python3
"""
Create a pastel-styled panel of cumulative simulation metrics.

Metrics:
    1) Cumulative posts
    2) Cumulative comments
    3) Cumulative users (by joined_on)
    4) Cumulative active users (ever posted/commented)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_counts(sim_dir: Path) -> pd.DataFrame:
    """Load simulation CSVs and build cumulative time series indexed by round."""
    posts_path = sim_dir / "posts.csv"
    users_path = sim_dir / "users.csv"

    missing = [p.name for p in (posts_path, users_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files in {sim_dir}: {', '.join(missing)}")

    posts = pd.read_csv(posts_path)
    users = pd.read_csv(users_path)

    # Build a unified round index across all series
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


def plot_panel(df: pd.DataFrame, output_path: Path) -> None:
    """Render the 2x2 panel with pastel colors and no grid lines."""
    sns.set_theme(style="white")
    colors = sns.color_palette("deep", 4)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    metrics = [
        ("cumulative_posts", "Cumulative Posts", "Count"),
        ("cumulative_comments", "Cumulative Comments", "Count"),
        ("cumulative_users", "All Users (joined)", "Users"),
        ("cumulative_active_users", "Active Users (posted/commented)", "Users"),
    ]

    rounds = df["round"].values
    for ax, (color, meta) in zip(axes.flat, zip(colors, metrics)):
        metric, title, ylabel = meta
        series = df[metric].values

        ax.plot(rounds, series, color=color, linewidth=3)
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Round")
        ax.grid(False)
        ax.set_facecolor("#ffffff")
        for spine in ax.spines.values():
            spine.set_visible(False)

        final_val = series[-1]
        label = f"{int(round(final_val)):,}"
        ax.text(
            0.02,
            0.85,
            label,
            transform=ax.transAxes,
            ha="left",
            color=color,
            fontsize=10,
            weight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, boxstyle="round,pad=0.3"),
        )

    fig.suptitle("Simulation Growth Overview", fontsize=14, weight="bold")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create cumulative growth panel for a simulation run.")
    parser.add_argument(
        "--sim-dir",
        default="simulation",
        help="Path to simulation directory containing posts.csv and users.csv.",
    )
    parser.add_argument(
        "--out-file",
        default="simulation/plots/cumulative_growth_panel.png",
        help="Output file for the generated figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sim_dir = Path(args.sim_dir).expanduser().resolve()
    output_path = Path(args.out_file).expanduser().resolve()

    df = load_counts(sim_dir)
    plot_panel(df, output_path)
    print(f"Saved panel to {output_path}")


if __name__ == "__main__":
    main()
