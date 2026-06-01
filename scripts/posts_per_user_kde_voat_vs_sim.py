"""
KDE comparison of posts-per-user: Voat samples vs simulation runs.

Reads:
  - Simulation runs: retained SQLite trajectory DBs or <sim-dir>/posts.csv
    with columns: user_id, comment_to, round. Top-level posts are detected
    as rows with comment_to <= 0 or missing.
  - Voat validation windows: retained parquet files or <voat-dir>/*.parquet
    with columns including user_id and interaction_type == 'posts'.

Outputs:
  - <out-file> (linear posts/user KDE)
  - <out-file> with `_logx1` suffix (log(1 + posts/user) KDE)

Usage:
  pyenv activate ysocial && \
  python scripts/posts_per_user_kde_voat_vs_sim.py \
    --sim-inputs-glob "data/benchmark_runs/run*.sqlite" \
    --voat-inputs-glob "data/voat_windows/validation/window_*/*.parquet" \
    --out-file paper/figures/voat_vs_sim_posts_per_user_kde.png
"""

import argparse
import logging
import sqlite3
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


VOAT_COLOR = "#1f77b4"
SIM_COLOR = "#ff7f0e"


def _list_input_paths(glob_pattern: str) -> List[Path]:
    # Accept directory globs, direct file globs, and single concrete paths.
    direct_path = Path(glob_pattern)
    if direct_path.exists():
        return [direct_path]

    paths = sorted(Path().glob(glob_pattern))
    if not paths:
        return []

    return paths


def load_sim_posts_per_user(sim_input: Path) -> np.ndarray:
    if sim_input.is_dir():
        p = sim_input / "posts.csv"
        if not p.exists():
            raise FileNotFoundError(f"Simulation posts.csv not found at {p}")
        df = pd.read_csv(p, engine="python")
    elif sim_input.suffix.lower() in {".sqlite", ".sqlite3", ".db"}:
        with sqlite3.connect(sim_input) as conn:
            df = pd.read_sql_query(
                "SELECT user_id, comment_to FROM post",
                conn,
            )
    else:
        df = pd.read_csv(sim_input, engine="python")

    cols = {c.lower(): c for c in df.columns}
    uid_col = cols.get("user_id")
    cto_col = cols.get("comment_to")
    if uid_col is None or cto_col is None:
        raise ValueError("posts.csv must contain 'user_id' and 'comment_to'")
    cto = pd.to_numeric(df[cto_col], errors="coerce")
    is_post = cto.le(0) | cto.isna()
    counts = df.loc[is_post, uid_col].value_counts().astype(int).to_numpy()
    return counts


def load_voat_posts_per_user(voat_input: Path) -> np.ndarray:
    if voat_input.is_dir():
        candidates = sorted(voat_input.glob("*.parquet"))
        if not candidates:
            raise FileNotFoundError(f"Voat parquet not found in {voat_input} (expected *.parquet)")
        if len(candidates) > 1:
            logger.warning("Multiple Voat parquet files in %s; using %s", voat_input, candidates[0])
        p = candidates[0]
    else:
        p = voat_input

    df = pd.read_parquet(p)
    # top-level posts via interaction_type == 'posts'
    if "interaction_type" in df.columns:
        mask = df["interaction_type"].astype(str).str.lower().eq("posts")
    elif "parent_id" in df.columns:
        mask = df["parent_id"].isna()
    else:
        raise ValueError(f"Cannot identify root posts in {p}")
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
    main_figure_file: Optional[Path] = None,
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
        "axes.labelsize": 18,
        "axes.linewidth": 1.8,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 19,
        "lines.linewidth": 3.0,
        "figure.dpi": 160,
        "savefig.dpi": 300,
    })

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.0), constrained_layout=True)
    _style_axis(ax)
    _plot_bands(
        ax,
        grid,
        dens_voat_mean,
        dens_voat_lo,
        dens_voat_hi,
        color=VOAT_COLOR,
        label=voat_label,
        linestyle="-",
    )
    _plot_bands(
        ax,
        grid,
        dens_sim_mean,
        dens_sim_lo,
        dens_sim_hi,
        color=SIM_COLOR,
        label=sim_label,
        linestyle="-",
    )
    ax.set_xlabel("Root Posts per User")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 20)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False)

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

    fig2, ax2 = plt.subplots(1, 1, figsize=(6.5, 5.0), constrained_layout=True)
    _style_axis(ax2)
    _plot_bands(
        ax2,
        grid_log,
        dens_voat_log_mean,
        dens_voat_log_lo,
        dens_voat_log_hi,
        color=VOAT_COLOR,
        label=voat_label,
        linestyle="-",
    )
    _plot_bands(
        ax2,
        grid_log,
        dens_sim_log_mean,
        dens_sim_log_lo,
        dens_sim_log_hi,
        color=SIM_COLOR,
        label=sim_label,
        linestyle="-",
    )
    ax2.set_xlabel("Log Root Posts per User")
    ax2.set_ylabel("Density")
    ax2.set_ylim(bottom=0)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False)

    # Save with suffix _logx1 before extension
    out_file = Path(out_file)
    stem, suf = out_file.stem, ''.join(out_file.suffixes) or '.png'
    out_log = out_file.with_name(f"{stem}_logx1{suf}")
    out_log.parent.mkdir(parents=True, exist_ok=True)
    fig2.savefig(out_log, bbox_inches="tight")
    if main_figure_file is not None:
        main_figure_file.parent.mkdir(parents=True, exist_ok=True)
        fig2.savefig(main_figure_file, bbox_inches="tight")
    plt.close(fig2)


def _plot_bands(
    ax: plt.Axes,
    grid: np.ndarray,
    mean: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    color: str,
    label: str,
    linestyle: str,
) -> None:
    ax.fill_between(grid, lower, upper, color=color, alpha=0.18, linewidth=0)
    ax.plot(grid, mean, color=color, linestyle=linestyle, label=label)


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(True, which="major", color="#bdbdbd", alpha=0.8, linewidth=0.8)
    ax.tick_params(axis="both", colors="#222222", width=1.6, length=5)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(1.8)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    parser = argparse.ArgumentParser(
        description="KDE comparison of posts per user across simulation runs vs Voat samples"
    )
    parser.add_argument("--sim-dir", type=Path, default=None, help="Single simulation SQLite, CSV, or directory containing posts.csv")
    parser.add_argument("--voat-dir", type=Path, default=None, help="Single Voat parquet or directory containing a parquet window")
    parser.add_argument(
        "--sim-inputs-glob",
        "--sim-dirs-glob",
        dest="sim_inputs_glob",
        type=str,
        default="data/benchmark_runs/run*.sqlite",
        help="Glob for simulation SQLite/CSV files or run directories",
    )
    parser.add_argument(
        "--voat-inputs-glob",
        "--voat-dirs-glob",
        dest="voat_inputs_glob",
        type=str,
        default="data/voat_windows/validation/window_*/*.parquet",
        help="Glob for Voat parquet files or window directories",
    )
    parser.add_argument("--max-sim", type=int, default=30, help="Max simulation runs to load (default: 30)")
    parser.add_argument("--max-voat", type=int, default=30, help="Max Voat samples to load (default: 30)")
    parser.add_argument("--out-file", type=Path, default=Path("paper/figures/voat_vs_sim_posts_per_user_kde.png"), help="Output image path for the linear-scale plot")
    parser.add_argument(
        "--main-figure-file",
        type=Path,
        default=Path("paper/figures_main/Figure3.png"),
        help="Output image path for the manuscript log-scale Figure 3",
    )
    args = parser.parse_args()

    if args.sim_dir is not None:
        sim_inputs = [args.sim_dir]
    else:
        sim_inputs = _list_input_paths(args.sim_inputs_glob)

    if args.voat_dir is not None:
        voat_inputs = [args.voat_dir]
    else:
        voat_inputs = _list_input_paths(args.voat_inputs_glob)

    if args.max_sim is not None:
        sim_inputs = sim_inputs[: args.max_sim]
    if args.max_voat is not None:
        voat_inputs = voat_inputs[: args.max_voat]

    if not sim_inputs:
        raise SystemExit(f"No simulation inputs found for pattern: {args.sim_inputs_glob}")
    if not voat_inputs:
        raise SystemExit(f"No Voat inputs found for pattern: {args.voat_inputs_glob}")

    logger.info("Loading %d simulation runs and %d Voat windows", len(sim_inputs), len(voat_inputs))

    sim_counts_list = []  # type: List[np.ndarray]
    for d in sim_inputs:
        try:
            sim_counts_list.append(load_sim_posts_per_user(d))
        except Exception as exc:
            logger.warning("Skipping simulation input %s: %s", d, exc)

    voat_counts_list = []  # type: List[np.ndarray]
    for d in voat_inputs:
        try:
            voat_counts_list.append(load_voat_posts_per_user(d))
        except Exception as exc:
            logger.warning("Skipping Voat input %s: %s", d, exc)

    if not sim_counts_list:
        raise SystemExit("No valid simulation runs loaded")
    if not voat_counts_list:
        raise SystemExit("No valid Voat windows loaded")

    plot_kde(
        sim_counts_list,
        voat_counts_list,
        args.out_file,
        sim_label="Simulation",
        voat_label="Voat",
        main_figure_file=args.main_figure_file,
    )
    logger.info("Saved: %s, _logx1 variant, and %s", args.out_file, args.main_figure_file)


if __name__ == "__main__":
    main()
