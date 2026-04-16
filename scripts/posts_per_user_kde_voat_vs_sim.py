"""
KDE comparison of posts-per-user: Voat samples vs simulation runs.

Reads:
  - Simulation runs: <sim-dir>/posts.csv with columns: user_id, comment_to, round
    Top-level posts are detected as rows with comment_to <= 0 (or NaN).
  - Voat samples: <voat-dir>/*.parquet with columns including
    user_id and interaction_type == 'posts' for top-level posts.

Outputs:
  - <out-file> (linear posts/user KDE)
  - <out-file> with `_logx1` suffix (log(1 + posts/user) KDE)

Usage:
  pyenv activate ysocial && \
  python scripts/posts_per_user_kde_voat_vs_sim.py \
    --sim-dirs-glob "results/run*" \
    --voat-dirs-glob "MADOC/voat-technology/sample_*" \
    --out-file paper/figures/voat_vs_sim_posts_per_user_kde.png
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


def _gaussian_kde_manual(x: np.ndarray, grid: np.ndarray, bandwidth: Optional[float] = None) -> np.ndarray:
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


def _list_dirs(glob_pattern: str) -> List[Path]:
    # Accept both directory globs ("results/run*") and file globs ("results/run*/posts.csv").
    paths = sorted(Path().glob(glob_pattern))
    if not paths:
        return []

    if all(p.is_file() for p in paths):
        return sorted({p.parent for p in paths})

    return [p for p in paths if p.is_dir()]


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
    candidates = sorted(voat_dir.glob("*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"Voat parquet not found in {voat_dir} (expected *.parquet)")
    if len(candidates) > 1:
        logger.warning("Multiple Voat parquet files in %s; using %s", voat_dir, candidates[0])
    df = pd.read_parquet(candidates[0])
    # top-level posts via interaction_type == 'posts'
    mask = df["interaction_type"].astype(str).str.lower().eq("posts")
    counts = df.loc[mask, "user_id"].value_counts().astype(int).to_numpy()
    return counts


def _kde_bands(
    samples: List[np.ndarray],
    grid: np.ndarray,
    *,
    log1p: bool = False,
    q_low: float = 0.05,
    q_high: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not samples:
        zeros = np.zeros_like(grid, dtype=float)
        return zeros, zeros, zeros

    densities = []
    for arr in samples:
        x = np.asarray(arr, dtype=float)
        x = x[np.isfinite(x)]
        if log1p:
            x = np.log1p(x)
        densities.append(_gaussian_kde_manual(x, grid))

    dens = np.vstack(densities)
    mean = dens.mean(axis=0)
    lo = np.quantile(dens, q_low, axis=0)
    hi = np.quantile(dens, q_high, axis=0)
    return mean, lo, hi


def plot_kde(
    sim_counts_list: List[np.ndarray],
    voat_counts_list: List[np.ndarray],
    out_file: Path,
    *,
    sim_label: str,
    voat_label: str,
) -> None:
    # Panel 1: linear scale, fixed x-axis range 0..20 (requested)
    grid = np.linspace(0, 20, 400)

    dens_sim_mean, dens_sim_lo, dens_sim_hi = _kde_bands(sim_counts_list, grid, log1p=False)
    dens_voat_mean, dens_voat_lo, dens_voat_hi = _kde_bands(voat_counts_list, grid, log1p=False)

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
    ax.plot(grid, dens_sim_mean, color="#1f77b4", label=sim_label)
    ax.fill_between(grid, dens_sim_lo, dens_sim_hi, color="#1f77b4", alpha=0.18, linewidth=0)
    ax.plot(grid, dens_voat_mean, color="#ff7f0e", label=voat_label)
    ax.fill_between(grid, dens_voat_lo, dens_voat_hi, color="#ff7f0e", alpha=0.18, linewidth=0)
    ax.set_title("Posts per User: Simulation vs Voat\nMean with 5th-95th Percentile Bands")
    ax.set_xlabel("Posts per User")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 20)
    ax.legend(frameon=False, fontsize=9)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)

    # Panel 2: log(x+1) version on posts/user
    max_sim = max((np.max(c) if len(c) else 0 for c in sim_counts_list), default=0)
    max_voat = max((np.max(c) if len(c) else 0 for c in voat_counts_list), default=0)
    xmax = float(np.log1p(max(max_sim, max_voat, 1)))
    grid_log = np.linspace(0.0, xmax, 400)
    dens_sim_log_mean, dens_sim_log_lo, dens_sim_log_hi = _kde_bands(sim_counts_list, grid_log, log1p=True)
    dens_voat_log_mean, dens_voat_log_lo, dens_voat_log_hi = _kde_bands(voat_counts_list, grid_log, log1p=True)

    fig2, ax2 = plt.subplots(1, 1, figsize=(6.2, 4.2), constrained_layout=True)
    ax2.plot(grid_log, dens_sim_log_mean, color="#1f77b4", label=sim_label)
    ax2.fill_between(grid_log, dens_sim_log_lo, dens_sim_log_hi, color="#1f77b4", alpha=0.18, linewidth=0)
    ax2.plot(grid_log, dens_voat_log_mean, color="#ff7f0e", label=voat_label)
    ax2.fill_between(grid_log, dens_voat_log_lo, dens_voat_log_hi, color="#ff7f0e", alpha=0.18, linewidth=0)
    ax2.set_title("Posts per User: Simulation vs Voat\nMean with 5th-95th Percentile Bands (Log Scale)")
    ax2.set_xlabel("Log Posts per User")
    ax2.set_ylabel("Density")
    ax2.legend(frameon=False, fontsize=9)

    # Save with suffix _logx1 before extension
    out_file = Path(out_file)
    stem, suf = out_file.stem, ''.join(out_file.suffixes) or '.png'
    out_log = out_file.with_name(f"{stem}_logx1{suf}")
    fig2.savefig(out_log, bbox_inches="tight")
    plt.close(fig2)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    parser = argparse.ArgumentParser(
        description="KDE comparison of posts per user across simulation runs vs Voat samples"
    )
    parser.add_argument("--sim-dir", type=Path, default=None, help="Single simulation directory containing posts.csv")
    parser.add_argument("--voat-dir", type=Path, default=None, help="Single Voat sample directory containing a parquet window")
    parser.add_argument("--sim-dirs-glob", type=str, default="results/run*", help="Glob for simulation run directories (default: results/run*)")
    parser.add_argument("--voat-dirs-glob", type=str, default="MADOC/voat-technology/sample_*", help="Glob for Voat sample directories")
    parser.add_argument("--max-sim", type=int, default=30, help="Max simulation runs to load (default: 30)")
    parser.add_argument("--max-voat", type=int, default=30, help="Max Voat samples to load (default: 30)")
    parser.add_argument("--out-file", type=Path, default=Path("paper/figures/voat_vs_sim_posts_per_user_kde.png"), help="Output image path for the linear-scale plot")
    args = parser.parse_args()

    if args.sim_dir is not None:
        sim_dirs = [args.sim_dir]
    else:
        sim_dirs = _list_dirs(args.sim_dirs_glob)

    if args.voat_dir is not None:
        voat_dirs = [args.voat_dir]
    else:
        voat_dirs = _list_dirs(args.voat_dirs_glob)

    if args.max_sim is not None:
        sim_dirs = sim_dirs[: args.max_sim]
    if args.max_voat is not None:
        voat_dirs = voat_dirs[: args.max_voat]

    if not sim_dirs:
        raise SystemExit(f"No simulation directories found for pattern: {args.sim_dirs_glob}")
    if not voat_dirs:
        raise SystemExit(f"No Voat sample directories found for pattern: {args.voat_dirs_glob}")

    logger.info("Loading %d simulation runs and %d Voat samples", len(sim_dirs), len(voat_dirs))

    sim_counts_list = []  # type: List[np.ndarray]
    for d in sim_dirs:
        try:
            sim_counts_list.append(load_sim_posts_per_user(d))
        except Exception as exc:
            logger.warning("Skipping simulation dir %s: %s", d, exc)

    voat_counts_list = []  # type: List[np.ndarray]
    for d in voat_dirs:
        try:
            voat_counts_list.append(load_voat_posts_per_user(d))
        except Exception as exc:
            logger.warning("Skipping Voat dir %s: %s", d, exc)

    if not sim_counts_list:
        raise SystemExit("No valid simulation runs loaded (missing posts.csv?)")
    if not voat_counts_list:
        raise SystemExit("No valid Voat samples loaded (missing parquet windows?)")

    sim_label = f"Simulation (n={len(sim_counts_list)} runs)"
    voat_label = f"Voat (n={len(voat_counts_list)} samples)"
    plot_kde(sim_counts_list, voat_counts_list, args.out_file, sim_label=sim_label, voat_label=voat_label)
    logger.info("Saved: %s (and _logx1 variant)", args.out_file)


if __name__ == "__main__":
    main()
