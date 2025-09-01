"""
Compact toxicity comparison figure for a simulation directory.

Produces a side-by-side panel:
  - Left: Overall toxicity KDE (all posts + comments)
  - Right: Posts vs. Comments comparison with KDE (solid) and ECDF (dashed)
    and annotations for tail shares at thresholds: >=0.25 and >=0.50.

Inputs (under --sim-dir, default: simulation):
  - toxigen.csv with columns: id, toxicity, post_type, is_comment

Usage:
  python scripts/toxicity_kde_ecdf.py --sim-dir simulation \
    [--out-file simulation/toxicity_kde_ecdf.png]
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
            bandwidth = 0.05  # minimal smoothing if constant
        else:
            bandwidth = 1.06 * std * (n ** (-1 / 5))
            bandwidth = max(bandwidth, 1e-3)
    u = (grid[:, None] - x[None, :]) / bandwidth
    kern = np.exp(-0.5 * u * u) / np.sqrt(2 * np.pi)
    dens = kern.sum(axis=1) / (n * bandwidth)
    return dens


def _ecdf(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.zeros_like(grid, dtype=float)
    x_sorted = np.sort(x)
    # For each grid point, fraction of x <= grid
    idx = np.searchsorted(x_sorted, grid, side="right")
    return idx / x_sorted.size


def _tail_share(x: np.ndarray, threshold: float) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    return float((x >= threshold).mean())


def load_toxicity(sim_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tox_path = sim_dir / "toxigen.csv"
    if not tox_path.exists():
        raise FileNotFoundError(f"toxigen.csv not found in {sim_dir}")
    df = pd.read_csv(tox_path)
    if "toxicity" not in df.columns:
        raise ValueError("toxigen.csv must contain a 'toxicity' column")

    # Optional boolean is_comment; if missing, infer from post_type
    is_comment = None
    if "is_comment" in df.columns:
        is_comment = df["is_comment"].astype(bool).to_numpy()
    elif "post_type" in df.columns:
        is_comment = df["post_type"].astype(str).str.contains("comment", case=False, na=False).to_numpy()
    else:
        is_comment = np.zeros(len(df), dtype=bool)

    tox = pd.to_numeric(df["toxicity"], errors="coerce").to_numpy()
    tox_posts = tox[~is_comment]
    tox_comments = tox[is_comment]
    return tox, tox_posts, tox_comments


def plot_toxicity_kde_ecdf(sim_dir: Path, out_file: Path) -> None:
    tox_all, tox_posts, tox_comments = load_toxicity(sim_dir)

    # Grid on [0, 1]
    grid = np.linspace(0.0, 1.0, 400)

    # KDEs
    dens_all = _gaussian_kde_manual(tox_all, grid)
    dens_posts = _gaussian_kde_manual(tox_posts, grid)
    dens_comments = _gaussian_kde_manual(tox_comments, grid)

    # ECDFs
    ecdf_posts = _ecdf(tox_posts, grid)
    ecdf_comments = _ecdf(tox_comments, grid)

    # Tail shares
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
        "lines.linewidth": 2.5,
        "figure.dpi": 150,
    })

    fig, (ax_all, ax_cmp) = plt.subplots(1, 2, figsize=(8.5, 4.5), constrained_layout=True)

    # Left: overall KDE
    ax_all.plot(grid, dens_all, color="#4c78a8", label="All")
    ax_all.set_title("Overall toxicity (KDE)")
    ax_all.set_xlabel("Toxicity")
    ax_all.set_ylabel("Density")
    ax_all.set_xlim(0, 1)
    ax_all.legend(frameon=False)

    # Right: posts vs comments KDE + ECDF (twin y-axis)
    ax_cmp.set_title("Posts vs Comments (KDE/ECDF)")
    ax_cmp.set_xlabel("Toxicity")
    ax_cmp.set_xlim(0, 1)

    # KDE (left axis)
    ax_cmp.plot(grid, dens_posts, color="#1f77b4", label="Posts KDE")
    ax_cmp.plot(grid, dens_comments, color="#ff7f0e", label="Comments KDE")
    ax_cmp.set_ylabel("Density")

    # ECDF (right axis)
    ax2 = ax_cmp.twinx()
    ax2.plot(grid, ecdf_posts, color="#1f77b4", linestyle="--", alpha=0.9, label="Posts ECDF")
    ax2.plot(grid, ecdf_comments, color="#ff7f0e", linestyle="--", alpha=0.9, label="Comments ECDF")
    ax2.set_ylabel("Cumulative probability")
    ax2.set_ylim(0, 1)

    # Threshold markers
    for thr in (thr1, thr2):
        ax_cmp.axvline(thr, color="#7f7f7f", linestyle=":", linewidth=1.8)
        ax_all.axvline(thr, color="#7f7f7f", linestyle=":", linewidth=1.8)

    # Legend combining both axes
    lines1, labels1 = ax_cmp.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax_cmp.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=9, loc="upper left")

    # Tail annotations (exact percentages)
    txt = (
        f"Posts: ≥{thr1:.2f}: {posts_t1*100:.1f}%  •  ≥{thr2:.2f}: {posts_t2*100:.1f}%\n"
        f"Comments: ≥{thr1:.2f}: {comments_t1*100:.1f}%  •  ≥{thr2:.2f}: {comments_t2*100:.1f}%"
    )
    ax_cmp.text(
        0.02,
        0.98,
        txt,
        transform=ax_cmp.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="#cccccc"),
    )

    fig.suptitle("Toxicity distributions", fontsize=12)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Create compact toxicity KDE/ECDF comparison plots for a simulation directory.")
    parser.add_argument("--sim-dir", type=Path, default=Path("simulation"), help="Simulation directory containing toxigen.csv")
    parser.add_argument("--out-file", type=Path, default=None, help="Output image file (default: <sim-dir>/toxicity_kde_ecdf.png)")
    args = parser.parse_args()

    sim_dir: Path = args.sim_dir
    out_file: Path = args.out_file or (sim_dir / "toxicity_kde_ecdf.png")

    plot_toxicity_kde_ecdf(sim_dir, out_file)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
