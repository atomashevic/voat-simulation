"""
KDE comparison of posts-per-user: Voat sample 1 vs Simulation.

Reads:
  - Simulation: <sim-dir>/posts.csv with columns: user_id, comment_to, round
    Top-level posts are detected as rows with comment_to <= 0 (or NaN).
  - Voat: <voat-dir>/voat_sample_1.parquet with columns including
    user_id and interaction_type == 'posts' for top-level posts.

Outputs:
  - <out-file> (default: simulation/voat_vs_sim_posts_per_user_kde.png)

Usage:
  pyenv activate ysocial && \
  python scripts/posts_per_user_kde_voat_vs_sim.py \
    --sim-dir simulation \
    --voat-dir MADOC/voat-technology/sample_1
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
            bandwidth = 0.1
        else:
            bandwidth = 1.06 * std * (n ** (-1 / 5))
            bandwidth = max(bandwidth, 1e-3)
    u = (grid[:, None] - x[None, :]) / bandwidth
    kern = np.exp(-0.5 * u * u) / np.sqrt(2 * np.pi)
    dens = kern.sum(axis=1) / (n * bandwidth)
    return dens


def load_sim_posts_per_user(sim_dir: Path) -> np.ndarray:
    p = sim_dir / "posts.csv"
    if not p.exists():
        raise FileNotFoundError(f"Simulation posts.csv not found at {p}")
    df = pd.read_csv(p, engine="python")
    cols = {c.lower(): c for c in df.columns}
    uid_col = cols.get("user_id")
    cto_col = cols.get("comment_to")
    if uid_col is None or cto_col is None:
        raise ValueError("posts.csv must contain 'user_id' and 'comment_to'")
    cto = pd.to_numeric(df[cto_col], errors="coerce")
    is_post = (cto <= 0) | cto.isna()
    counts = (
        df.loc[is_post, uid_col].value_counts().astype(int).to_numpy()
    )
    return counts


def load_voat_posts_per_user(voat_dir: Path) -> np.ndarray:
    p = voat_dir / "voat_sample_1.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Voat parquet not found at {p}")
    df = pd.read_parquet(p)
    # top-level posts via interaction_type == 'posts'
    mask = df["interaction_type"].astype(str).str.lower().eq("posts")
    counts = df.loc[mask, "user_id"].value_counts().astype(int).to_numpy()
    return counts


def load_sim_all_interactions_per_user(sim_dir: Path) -> np.ndarray:
    p = sim_dir / "posts.csv"
    if not p.exists():
        raise FileNotFoundError(f"Simulation posts.csv not found at {p}")
    df = pd.read_csv(p, engine="python")
    cols = {c.lower(): c for c in df.columns}
    uid_col = cols.get("user_id")
    if uid_col is None:
        raise ValueError("posts.csv must contain 'user_id'")
    counts = df[uid_col].value_counts().astype(int).to_numpy()
    return counts


def load_voat_all_interactions_per_user(voat_dir: Path) -> np.ndarray:
    p = voat_dir / "voat_sample_1.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Voat parquet not found at {p}")
    df = pd.read_parquet(p)
    counts = df["user_id"].value_counts().astype(int).to_numpy()
    return counts


def plot_kde(
    sim_counts: np.ndarray,
    voat_counts: np.ndarray,
    out_file: Path,
    sim_all_counts: np.ndarray | None = None,
    voat_all_counts: np.ndarray | None = None,
) -> None:
    # Panel 1: linear scale, fixed x-axis range 0..20 (requested)
    grid = np.linspace(0, 20, 400)

    dens_sim = _gaussian_kde_manual(sim_counts, grid)
    dens_voat = _gaussian_kde_manual(voat_counts, grid)

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

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.2), constrained_layout=True)
    ax.plot(grid, dens_sim, color="#1f77b4", label="Simulation (posts/user)")
    ax.plot(grid, dens_voat, color="#ff7f0e", label="Voat sample (posts/user)")
    ax.set_title("Posts per user: Voat vs Simulation (KDE)")
    ax.set_xlabel("Posts per user")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 20)
    ax.legend(frameon=False, fontsize=9)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)

    # Panel 2: log(x+1) version on total interactions (posts+comments) if available
    if sim_all_counts is not None and voat_all_counts is not None:
        sim_log = np.log1p(sim_all_counts)
        voat_log = np.log1p(voat_all_counts)
    else:
        sim_log = np.log1p(sim_counts)
        voat_log = np.log1p(voat_counts)
    if sim_log.size or voat_log.size:
        xmax = float(max(sim_log.max() if sim_log.size else 0.0, voat_log.max() if voat_log.size else 0.0))
    else:
        xmax = 1.0
    grid_log = np.linspace(0.0, xmax, 400)
    dens_sim_log = _gaussian_kde_manual(sim_log, grid_log)
    dens_voat_log = _gaussian_kde_manual(voat_log, grid_log)

    fig2, ax2 = plt.subplots(1, 1, figsize=(6.2, 4.2), constrained_layout=True)
    ax2.plot(grid_log, dens_sim_log, color="#1f77b4", label="Simulation")
    ax2.plot(grid_log, dens_voat_log, color="#ff7f0e", label="Voat sample")
    ax2.set_title("Posts and comments per user — log(x+1) KDE")
    ax2.set_xlabel("log(1 + interactions per user)")
    ax2.set_ylabel("Density")
    ax2.legend(frameon=False, fontsize=9)

    # Save with suffix _logx1 before extension
    out_file = Path(out_file)
    stem, suf = out_file.stem, ''.join(out_file.suffixes) or '.png'
    out_log = out_file.with_name(f"{stem}_logx1{suf}")
    fig2.savefig(out_log, bbox_inches="tight")
    plt.close(fig2)


def main():
    parser = argparse.ArgumentParser(description="KDE comparison of posts per user for Voat sample 1 vs simulation")
    parser.add_argument("--sim-dir", type=Path, default=Path("simulation"), help="Simulation directory containing posts.csv")
    parser.add_argument("--voat-dir", type=Path, default=Path("MADOC/voat-technology/sample_1"), help="Voat sample directory")
    parser.add_argument("--out-file", type=Path, default=None, help="Output image (default: <sim-dir>/voat_vs_sim_posts_per_user_kde.png)")
    args = parser.parse_args()

    sim_counts = load_sim_posts_per_user(args.sim_dir)
    voat_counts = load_voat_posts_per_user(args.voat_dir)

    # Also compute all interactions (posts + comments) per user for the log plot
    sim_all_counts = load_sim_all_interactions_per_user(args.sim_dir)
    voat_all_counts = load_voat_all_interactions_per_user(args.voat_dir)

    out_file = args.out_file or (args.sim_dir / "voat_vs_sim_posts_per_user_kde.png")
    plot_kde(sim_counts, voat_counts, out_file, sim_all_counts, voat_all_counts)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
