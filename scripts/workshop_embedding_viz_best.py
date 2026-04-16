#!/usr/bin/env python3
"""
Generate UMAP/TSNE visualization for the simulation run with highest embedding similarity.

Finds the run with highest comments_mean_cosine and creates scatter plots showing
simulation vs Voat embeddings in reduced 2D space.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

URL_RE = re.compile(r"https?://\S+")
WS_RE = re.compile(r"\s+")


def clean_text(s: str) -> str:
    """Clean text for embedding."""
    s = s or ""
    s = URL_RE.sub(" ", s)
    s = s.replace("\n", " ")
    s = WS_RE.sub(" ", s)
    return s.strip()


def find_run_by_rank(results_dir: Path, rank: str = "best") -> Tuple[str, float, float]:
    """Find run by rank: 'best', 'worst', 'median', or a specific run name like 'run01'.
    
    Returns: (run_name, comments_mean_cosine, posts_mean_cosine)
    """
    all_runs = []
    
    for run_dir in sorted(results_dir.glob("run*")):
        if not run_dir.is_dir():
            continue
        
        metrics_file = run_dir / "metrics.json"
        if not metrics_file.exists():
            continue
        
        try:
            data = json.loads(metrics_file.read_text())
            emb = data.get("embedding", {})
            comments_score = emb.get("comments_mean_cosine", 0)
            posts_score = emb.get("posts_mean_cosine", 0)
            all_runs.append((run_dir.name, comments_score, posts_score))
        except Exception:
            pass
    
    if not all_runs:
        raise ValueError("No valid runs found")
    
    # Sort by comments score ascending
    all_runs.sort(key=lambda x: x[1])
    
    # Select based on rank
    if rank == "best":
        selected = all_runs[-1]
    elif rank == "worst":
        selected = all_runs[0]
    elif rank == "median":
        mid_idx = len(all_runs) // 2
        selected = all_runs[mid_idx]
    elif rank.startswith("run"):
        # Specific run requested
        matches = [r for r in all_runs if r[0] == rank]
        if not matches:
            raise ValueError(f"Run {rank} not found")
        selected = matches[0]
    else:
        raise ValueError(f"Unknown rank: {rank}")
    
    logger.info("Selected run: %s (comments=%.4f, posts=%.4f, rank=%s)", 
                selected[0], selected[1], selected[2], rank)
    return selected[0], selected[1], selected[2]


def load_sim_texts(run_dir: Path, want_comments: bool) -> pd.DataFrame:
    """Load simulation texts."""
    posts_path = run_dir / "posts.csv"
    tox_path = run_dir / "toxigen.csv"
    
    if not posts_path.exists():
        raise FileNotFoundError(f"Missing posts.csv in {run_dir}")
    
    posts = pd.read_csv(posts_path)
    posts.columns = posts.columns.str.lower()
    
    # Determine comments vs posts
    if "comment_to" in posts.columns:
        if want_comments:
            posts = posts[posts["comment_to"] != -1]
        else:
            posts = posts[posts["comment_to"] == -1]
    
    # Get text column
    text_col = "tweet" if "tweet" in posts.columns else "content"
    posts["text"] = posts[text_col].astype(str).map(clean_text)
    posts = posts[posts["text"].str.len() > 10]
    
    return posts[["id", "text"]].reset_index(drop=True)


def load_voat_texts(voat_parquet: Path, want_comments: bool, sample_size: int = 5000) -> pd.DataFrame:
    """Load Voat texts, sampling if necessary."""
    df = pd.read_parquet(voat_parquet, columns=["post_id", "content", "interaction_type"])
    
    # Filter by interaction type
    types = df["interaction_type"].astype(str).str.lower()
    if want_comments:
        df = df[types.isin(["comment", "comments"])]
    else:
        df = df[types.isin(["post", "posts"])]
    
    df = df.dropna(subset=["content"])
    df["text"] = df["content"].astype(str).map(clean_text)
    df = df[df["text"].str.len() > 10]
    
    # Sample if too large
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    return df[["post_id", "text"]].reset_index(drop=True)


def compute_embeddings(texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 128) -> np.ndarray:
    """Compute embeddings using sentence transformers."""
    from sentence_transformers import SentenceTransformer
    
    logger.info("Loading model %s...", model_name)
    model = SentenceTransformer(model_name)
    
    logger.info("Encoding %d texts...", len(texts))
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    
    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms
    
    return embeddings


def create_2d_plots(
    sim_embs: np.ndarray,
    voat_embs: np.ndarray,
    output_dir: Path,
    run_name: str,
    tsne_perplexity: int = 30,
    rank: str = "best",
    comments_cosine: float = 0.0,
    posts_cosine: float = 0.0,
) -> None:
    """Create UMAP and t-SNE scatter plots."""
    from sklearn.manifold import TSNE
    from umap import UMAP
    
    # Combine embeddings
    X = np.vstack([sim_embs, voat_embs])
    y = np.array([0] * len(sim_embs) + [1] * len(voat_embs))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # UMAP
    logger.info("Computing UMAP...")
    umap_reducer = UMAP(n_components=2, metric="cosine", random_state=42, n_neighbors=15, min_dist=0.1)
    coords_umap = umap_reducer.fit_transform(X)
    
    # Plot UMAP
    for cls, color, label in [(1, "tab:orange", "Voat"), (0, "tab:blue", "Simulation")]:
        pts = coords_umap[y == cls]
        ax1.scatter(pts[:, 0], pts[:, 1], s=10 if cls == 1 else 15, alpha=0.6, c=color, label=label)
    ax1.set_title(f"Simulation vs Voat (UMAP)", fontsize=12, weight="bold")
    ax1.legend(loc="upper right")
    ax1.set_xlabel("UMAP 1")
    ax1.set_ylabel("UMAP 2")
    
    # t-SNE
    logger.info("Computing t-SNE (perplexity=%d)...", tsne_perplexity)
    tsne_reducer = TSNE(n_components=2, perplexity=tsne_perplexity, metric="cosine", 
                         learning_rate="auto", init="pca", random_state=42)
    coords_tsne = tsne_reducer.fit_transform(X)
    
    # Plot t-SNE
    for cls, color, label in [(1, "tab:orange", "Voat"), (0, "tab:blue", "Simulation")]:
        pts = coords_tsne[y == cls]
        ax2.scatter(pts[:, 0], pts[:, 1], s=10 if cls == 1 else 15, alpha=0.6, c=color, label=label)
    ax2.set_title(f"Simulation vs Voat (t-SNE, p={tsne_perplexity})", fontsize=12, weight="bold")
    ax2.legend(loc="upper right")
    ax2.set_xlabel("t-SNE 1")
    ax2.set_ylabel("t-SNE 2")
    
    # Add text box with similarity scores
    stats_text = (
        f"Mean Cosine Similarity\n"
        f"{'─' * 22}\n"
        f"Comments: {comments_cosine:.4f}\n"
        f"Posts:    {posts_cosine:.4f}"
    )
    ax2.text(0.97, 0.03, stats_text, transform=ax2.transAxes, ha="right", va="bottom",
             fontsize=10, family="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#dddddd", alpha=0.95))
    
    # Determine suffix for output file based on rank
    rank_label = rank if rank in ["best", "worst", "median"] else "selected"
    fig.suptitle(f"Embedding Similarity: {run_name} ({rank} cosine similarity)\nComments: n_sim={len(sim_embs)}, n_voat={len(voat_embs)}", 
                 fontsize=13, weight="bold")
    fig.tight_layout()
    
    # Save combined figure
    output_path = output_dir / f"embedding_umap_tsne_{rank_label}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved combined figure to %s", output_path)
    
    # Also save individual figures
    for name, coords, ax_title in [("umap", coords_umap, "UMAP"), ("tsne", coords_tsne, f"t-SNE (p={tsne_perplexity})")]:
        fig2, ax = plt.subplots(figsize=(8, 6))
        for cls, color, label in [(1, "tab:orange", "Voat"), (0, "tab:blue", "Simulation")]:
            pts = coords[y == cls]
            ax.scatter(pts[:, 0], pts[:, 1], s=10 if cls == 1 else 15, alpha=0.6, c=color, label=label)
        ax.set_title(f"Simulation vs Voat ({ax_title})", fontsize=12, weight="bold")
        ax.legend(loc="upper right")
        # Add stats box to individual figures too
        ax.text(0.97, 0.03, stats_text, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=10, family="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#dddddd", alpha=0.95))
        fig2.tight_layout()
        fig2.savefig(output_dir / f"embedding_{name}_{rank_label}.png", dpi=300, bbox_inches="tight")
        plt.close(fig2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate UMAP/TSNE for selected run based on embedding similarity."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing run01, run02, ... subdirectories.",
    )
    parser.add_argument(
        "--voat-parquet",
        type=Path,
        default=Path("MADOC/voat-technology/voat_technology_madoc.parquet"),
        help="Full Voat parquet file for comparison.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("workshop/figures"),
        help="Output directory for figures.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=3000,
        help="Max Voat samples to include in visualization.",
    )
    parser.add_argument(
        "--tsne-perplexity",
        type=int,
        default=30,
        help="t-SNE perplexity parameter.",
    )
    parser.add_argument(
        "--rank",
        type=str,
        default="best",
        help="Which run to select: 'best', 'worst', 'median', or a specific run name like 'run01'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    voat_parquet = args.voat_parquet.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    
    # Find run by rank
    selected_run, comments_cosine, posts_cosine = find_run_by_rank(results_dir, args.rank)
    run_dir = results_dir / selected_run
    
    # Load texts (comments for comparison)
    sim_df = load_sim_texts(run_dir, want_comments=True)
    voat_df = load_voat_texts(voat_parquet, want_comments=True, sample_size=args.sample_size)
    
    logger.info("Loaded %d simulation comments, %d Voat comments", len(sim_df), len(voat_df))
    
    # Compute embeddings
    all_texts = sim_df["text"].tolist() + voat_df["text"].tolist()
    embeddings = compute_embeddings(all_texts)
    
    sim_embs = embeddings[:len(sim_df)]
    voat_embs = embeddings[len(sim_df):]
    
    # Create visualizations
    create_2d_plots(sim_embs, voat_embs, output_dir, selected_run, args.tsne_perplexity, 
                    args.rank, comments_cosine, posts_cosine)


if __name__ == "__main__":
    main()

