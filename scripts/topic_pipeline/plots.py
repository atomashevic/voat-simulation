"""
Plotting helpers for topic modelling artefacts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def plot_topic_sizes(
    topics_info: pd.DataFrame,
    out_path: Path,
    *,
    top_k: int = 20,
    title: Optional[str] = None,
) -> None:
    """Save a horizontal bar chart for the `top_k` largest topics."""
    if topics_info.empty:
        logger.warning("No topics to plot for %s", out_path)
        return

    top = topics_info.nlargest(top_k, "size").copy()
    top = top.sort_values("size", ascending=True)

    plt.figure(figsize=(10, max(4, 0.4 * len(top))))
    plt.barh(top["label"], top["size"], color="#1f77b4")
    plt.xlabel("Thread count")
    plt.title(title or "Largest Topics")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_similarity_heatmap(
    sim_matrix: pd.DataFrame,
    out_path: Path,
    *,
    title: Optional[str] = None,
    cmap: str = "viridis",
) -> None:
    """Render a heatmap of topic similarities."""
    if sim_matrix.empty:
        logger.warning("Similarity matrix is empty; skipping heatmap %s", out_path)
        return

    plt.figure(figsize=(12, max(4, 0.4 * len(sim_matrix.index))))
    plt.imshow(sim_matrix.values, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    plt.colorbar(label="cosine similarity")
    plt.xticks(range(len(sim_matrix.columns)), sim_matrix.columns, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(sim_matrix.index)), sim_matrix.index, fontsize=8)
    plt.title(title or "Topic similarity heatmap")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
