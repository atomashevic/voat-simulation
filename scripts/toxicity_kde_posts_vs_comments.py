"""
Single-panel toxicity comparison (KDE) for posts vs comments.

Inputs:
  - <sim-dir>/toxigen.csv with columns: toxicity, is_comment (bool) or post_type

Output:
  - <sim-dir>/toxicity_kde_posts_vs_comments.png (unless --out-file provided)

Usage:
  pyenv activate ysocial && \
  python scripts/toxicity_kde_posts_vs_comments.py --sim-dir simulation
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _gaussian_kde_manual(x: np.ndarray, grid: np.ndarray, bandwidth: float | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.zeros_like(grid, dtype=float)
    n = x.size
    std = np.std(x, ddof=1) if n > 1 else 0.0
    if bandwidth is None:
        if std == 0.0:
            bandwidth = 0.05
        else:
            bandwidth = 1.06 * std * (n ** (-1 / 5))
            bandwidth = max(bandwidth, 1e-3)
    u = (grid[:, None] - x[None, :]) / bandwidth
    kern = np.exp(-0.5 * u * u) / np.sqrt(2 * np.pi)
    dens = kern.sum(axis=1) / (n * bandwidth)
    return dens


def _tail_share(x: np.ndarray, threshold: float) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    return float((x >= threshold).mean())


def load_toxicity(sim_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    tox_path = sim_dir / "toxigen.csv"
    if not tox_path.exists():
        raise FileNotFoundError(f"toxigen.csv not found in {sim_dir}")
    df = pd.read_csv(tox_path)
    if "toxicity" not in df.columns:
        raise ValueError("toxigen.csv must contain a 'toxicity' column")
    tox = pd.to_numeric(df["toxicity"], errors="coerce").to_numpy()

    if "is_comment" in df.columns:
        is_comment = df["is_comment"].astype(bool).to_numpy()
    elif "post_type" in df.columns:
        is_comment = df["post_type"].astype(str).str.contains("comment", case=False, na=False).to_numpy()
    else:
        is_comment = np.zeros(len(df), dtype=bool)

    tox_posts = tox[~is_comment]
    tox_comments = tox[is_comment]
    return tox_posts, tox_comments


def plot_single_kde(sim_dir: Path, out_file: Path) -> None:
    tox_posts, tox_comments = load_toxicity(sim_dir)

    grid = np.linspace(0.0, 1.0, 400)
    dens_posts = _gaussian_kde_manual(tox_posts, grid)
    dens_comments = _gaussian_kde_manual(tox_comments, grid)

    thr1, thr2 = 0.25, 0.50
    posts_t1 = _tail_share(tox_posts, thr1)
    posts_t2 = _tail_share(tox_posts, thr2)
    comments_t1 = _tail_share(tox_comments, thr1)
    comments_t2 = _tail_share(tox_comments, thr2)

    # Scientific style
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        plt.style.use("seaborn-whitegrid")
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 2.8,
        "figure.dpi": 150,
    })

    fig, ax = plt.subplots(1, 1, figsize=(6.0, 4.2), constrained_layout=True)

    ax.plot(grid, dens_posts, color="#1f77b4", label="Posts KDE")
    ax.plot(grid, dens_comments, color="#ff7f0e", label="Comments KDE")
    ax.set_title("Toxicity: Posts vs Comments (KDE)")
    ax.set_xlabel("Toxicity")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1)

    # Threshold markers and annotation
    for thr in (thr1, thr2):
        ax.axvline(thr, color="#7f7f7f", linestyle=":", linewidth=1.8)

    txt = (
        f"Posts ≥{thr1:.2f}: {posts_t1*100:.1f}%  •  ≥{thr2:.2f}: {posts_t2*100:.1f}%\n"
        f"Comments ≥{thr1:.2f}: {comments_t1*100:.1f}%  •  ≥{thr2:.2f}: {comments_t2*100:.1f}%"
    )
    ax.text(
        0.98, 0.98, txt,
        transform=ax.transAxes, va="top", ha="right",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75, edgecolor="#cccccc"),
    )

    ax.legend(frameon=False, loc="upper left", fontsize=9)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Single-panel Posts vs Comments toxicity KDE plot")
    parser.add_argument("--sim-dir", type=Path, default=Path("simulation"), help="Simulation directory")
    parser.add_argument("--out-file", type=Path, default=None, help="Output file (default: <sim-dir>/toxicity_kde_posts_vs_comments.png)")
    args = parser.parse_args()

    sim_dir: Path = args.sim_dir
    out_file: Path = args.out_file or (sim_dir / "toxicity_kde_posts_vs_comments.png")
    plot_single_kde(sim_dir, out_file)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
