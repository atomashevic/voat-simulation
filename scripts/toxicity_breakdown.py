"""
Generate a boxplot of mean toxicity scores per user grouped by their toxicity level.
The script loads toxigen outputs, joins them with posts and user metadata, aggregates
per user, validates that exactly three toxicity levels exist, and writes a plot.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib
matplotlib.use("Agg")  # ensure headless rendering
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def resolve_columns(df: pd.DataFrame, required: Iterable[str], context: str) -> Tuple[str, ...]:
    """Map lower-case column names back to their originals."""
    lookup = {col.lower(): col for col in df.columns}
    missing = [col for col in required if col not in lookup]
    if missing:
        raise ValueError(f"{context} missing columns: {', '.join(missing)}")
    return tuple(lookup[col] for col in required)


def load_tables(sim_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load posts, users, and toxigen tables from the simulation directory."""
    posts_path = sim_dir / "posts.csv"
    users_path = sim_dir / "users.csv"
    toxigen_path = sim_dir / "toxigen.csv"
    for path in (posts_path, users_path, toxigen_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
        logger.info("Loading %s", path)
    return pd.read_csv(posts_path), pd.read_csv(users_path), pd.read_csv(toxigen_path)


def build_dataset(sim_dir: Path) -> pd.DataFrame:
    """
    Merge toxigen outputs with posts and user metadata to assign each toxicity score
    to the originating user and their toxicity level.
    """
    posts, users, toxigen = load_tables(sim_dir)
    tox_id, tox_score = resolve_columns(toxigen, ("id", "toxicity"), "toxigen")
    post_id, post_user = resolve_columns(posts, ("id", "user_id"), "posts")
    user_id, user_level = resolve_columns(users, ("id", "toxicity"), "users")

    toxigen = toxigen[[tox_id, tox_score]].rename(columns={tox_id: "post_id", tox_score: "toxicity_score"})
    posts = posts[[post_id, post_user]].rename(columns={post_id: "post_id", post_user: "user_id"})
    users = users[[user_id, user_level]].rename(columns={user_id: "user_id", user_level: "user_level"})

    merged = toxigen.merge(posts, on="post_id", how="inner").merge(users, on="user_id", how="inner")

    frame = merged.dropna(subset=["user_id", "toxicity_score", "user_level"])
    frame["toxicity_score"] = pd.to_numeric(frame["toxicity_score"], errors="coerce")
    frame = frame.dropna(subset=["toxicity_score"])

    logger.info("Merged %d rows after dropping nulls", len(frame))
    return frame


def compute_user_means(frame: pd.DataFrame) -> pd.DataFrame:
    """Average toxicity per user and return a tidy dataframe."""
    grouped = (
        frame.groupby(["user_id", "user_level"], dropna=False)["toxicity_score"]
        .mean()
        .reset_index()
        .rename(columns={"toxicity_score": "mean_toxicity"})
    )
    logger.info("Computed means for %d users", len(grouped))
    return grouped


LEVEL_ORDER = ["absolutely no", "no", "moderately"]
PASTEL_MAP = {
    "absolutely no": "#8dd3c7",
    "no": "#ffffb3",
    "moderately": "#fb8072",
}


def validate_levels(df: pd.DataFrame) -> list:
    """Ensure there are exactly three distinct toxicity levels present."""
    observed = sorted(df["user_level"].unique())
    if set(observed) != set(LEVEL_ORDER):
        raise ValueError(f"Expected toxicity levels {LEVEL_ORDER}, found {observed}")
    logger.info("Confirmed toxicity levels: %s", LEVEL_ORDER)
    return LEVEL_ORDER


def plot_boxplot(df: pd.DataFrame, levels: list, output_path: Path) -> None:
    """Render and save the boxplot of mean toxicity per user."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_style("white")
    plt.figure(figsize=(9, 6))
    palette = [PASTEL_MAP[level] for level in levels]
    sns.boxplot(
        data=df,
        x="user_level",
        y="mean_toxicity",
        order=levels,
        showfliers=False,
        palette=palette,
        linewidth=1.2,
    )
    sns.stripplot(
        data=df,
        x="user_level",
        y="mean_toxicity",
        order=levels,
        color="black",
        jitter=0.2,
        alpha=0.3,
        size=2.5,
    )
    plt.xlabel("User Toxicity Level")
    plt.ylabel("Mean Toxicity per User")
    plt.title("Mean Toxicity by User Level")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Saved plot to %s", output_path)


def main(sim_dir: Path, output_path: Path) -> None:
    frame = build_dataset(sim_dir)
    user_means = compute_user_means(frame)
    levels = validate_levels(user_means)
    plot_boxplot(user_means, levels, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate boxplot of mean user toxicity grouped by user level.")
    parser.add_argument("--sim-dir", type=Path, required=True, help="Directory with posts.csv, users.csv, and toxigen.csv.")
    parser.add_argument("--output-path", type=Path, required=True, help="Destination for the rendered PNG plot.")
    args = parser.parse_args()
    main(args.sim_dir, args.output_path)
