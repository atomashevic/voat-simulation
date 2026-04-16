#!/usr/bin/env python3
"""
Create toxicity distribution comparison: 30 simulations vs 30 Voat samples.

Shows mean KDE with percentile bands to visualize run-to-run variability.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_metrics import bootstrap_summary

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def gaussian_kde(x: np.ndarray, grid: np.ndarray, bandwidth: float | None = None) -> np.ndarray:
    """Compute Gaussian KDE manually."""
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
            # Silverman's rule of thumb
            bandwidth = 1.06 * std * (n ** (-1 / 5))
            bandwidth = max(bandwidth, 1e-3)
    
    u = (grid[:, None] - x[None, :]) / bandwidth
    kern = np.exp(-0.5 * u * u) / np.sqrt(2 * np.pi)
    dens = kern.sum(axis=1) / (n * bandwidth)
    return dens


def load_sim_toxicity_per_run(results_dir: Path) -> List[np.ndarray]:
    """Load toxicity scores from each simulation run separately."""
    all_runs = []
    
    for run_dir in sorted(results_dir.glob("run*")):
        if not run_dir.is_dir():
            continue
        
        tox_file = run_dir / "toxigen.csv"
        if tox_file.exists():
            try:
                df = pd.read_csv(tox_file)
                if "toxicity" in df.columns:
                    scores = pd.to_numeric(df["toxicity"], errors="coerce").dropna().values
                    if len(scores) > 0:
                        all_runs.append(scores)
                        logger.debug("Loaded %d scores from %s", len(scores), run_dir.name)
            except Exception as e:
                logger.warning("Failed to load %s: %s", tox_file, e)
    
    logger.info("Loaded toxicity from %d simulation runs", len(all_runs))
    return all_runs


def load_voat_toxicity_per_sample(voat_dir: Path) -> List[np.ndarray]:
    """Load toxicity scores from each Voat sample separately."""
    all_samples = []
    
    for sample_dir in sorted(voat_dir.glob("sample_*")):
        if not sample_dir.is_dir():
            continue
        
        # Try parquet file
        parquet_files = list(sample_dir.glob("*.parquet"))
        if parquet_files:
            try:
                df = pd.read_parquet(parquet_files[0])
                # Check various column names for toxicity
                tox_col = None
                for col in ["toxicity", "toxicity_toxigen", "toxigen"]:
                    if col in df.columns:
                        tox_col = col
                        break
                
                if tox_col:
                    scores = pd.to_numeric(df[tox_col], errors="coerce").dropna().values
                    if len(scores) > 0:
                        all_samples.append(scores)
                        logger.debug("Loaded %d scores from %s", len(scores), sample_dir.name)
            except Exception as e:
                logger.warning("Failed to load parquet from %s: %s", sample_dir, e)
    
    logger.info("Loaded toxicity from %d Voat samples", len(all_samples))
    return all_samples


def compute_kde_ensemble(
    runs: List[np.ndarray],
    grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute KDE for each run and return mean with percentile bands.
    
    Returns: (mean_kde, lower_5pct, upper_95pct)
    """
    kdes = []
    for scores in runs:
        kde = gaussian_kde(scores, grid)
        kdes.append(kde)
    
    kdes = np.array(kdes)  # shape: (n_runs, n_grid_points)
    
    mean_kde = np.mean(kdes, axis=0)
    lower = np.percentile(kdes, 5, axis=0)
    upper = np.percentile(kdes, 95, axis=0)
    
    return mean_kde, lower, upper


def plot_toxicity_comparison(
    sim_runs: List[np.ndarray],
    voat_samples: List[np.ndarray],
    output_path: Path,
    confidence: float = 0.99,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> None:
    """Create comparison plot with mean KDEs and percentile bands."""
    
    # Grid for KDE evaluation (toxicity is typically in [0, 1])
    grid = np.linspace(0, 1, 400)
    
    # Compute KDE ensembles
    logger.info("Computing KDE ensemble for %d simulation runs...", len(sim_runs))
    sim_mean, sim_lower, sim_upper = compute_kde_ensemble(sim_runs, grid)
    
    logger.info("Computing KDE ensemble for %d Voat samples...", len(voat_samples))
    voat_mean, voat_lower, voat_upper = compute_kde_ensemble(voat_samples, grid)
    
    # Compute overall statistics
    sim_all = np.concatenate(sim_runs)
    voat_all = np.concatenate(voat_samples)
    sim_ci = bootstrap_summary(
        [float(np.mean(run)) for run in sim_runs],
        confidence=confidence,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    voat_ci = bootstrap_summary(
        [float(np.mean(sample)) for sample in voat_samples],
        confidence=confidence,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    
    # Create figure with clean, modern styling
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Set background
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('white')
    
    # Plot Voat first (behind)
    ax.fill_between(grid, voat_lower, voat_upper, color="#ff7f0e", alpha=0.2, linewidth=0)
    ax.plot(grid, voat_mean, color="#ff7f0e", linewidth=3, label=f"Voat (n={len(voat_samples)} samples)")
    
    # Plot simulation on top
    ax.fill_between(grid, sim_lower, sim_upper, color="#1f77b4", alpha=0.2, linewidth=0)
    ax.plot(grid, sim_mean, color="#1f77b4", linewidth=3, label=f"Simulation (n={len(sim_runs)} runs)")
    
    # Add vertical reference lines for key thresholds
    for thr, label in [(0.25, "Low"), (0.5, "Medium"), (0.75, "High")]:
        ax.axvline(thr, color="#cccccc", linestyle="--", linewidth=1.5, zorder=1)
        ax.text(thr, ax.get_ylim()[1] * 0.02, label, ha="center", fontsize=9, color="#888888")
    
    # Summary statistics box
    sim_mean_val = np.mean(sim_all)
    voat_mean_val = np.mean(voat_all)
    
    stats_text = (
        f"Summary Statistics\n"
        f"{'─' * 24}\n"
        f"Simulation (n={len(sim_all):,}):\n"
        f"  Mean: {sim_mean_val:.3f}\n"
        f"  {int(confidence * 100)}% CI: [{sim_ci['ci_lower']:.3f}, {sim_ci['ci_upper']:.3f}]\n\n"
        f"Voat (n={len(voat_all):,}):\n"
        f"  Mean: {voat_mean_val:.3f}\n"
        f"  {int(confidence * 100)}% CI: [{voat_ci['ci_lower']:.3f}, {voat_ci['ci_upper']:.3f}]"
    )
    
    ax.text(0.97, 0.65, stats_text, transform=ax.transAxes, ha="right", va="top",
            fontsize=11, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#dddddd", alpha=0.95))
    
    # Style
    ax.set_xlabel("Toxicity Score", fontsize=13, fontweight='bold')
    ax.set_ylabel("Density", fontsize=13, fontweight='bold')
    ax.set_title("Toxicity Distribution: Simulation vs Voat\nMean with 5th-95th Percentile Bands", 
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    
    # Grid
    ax.grid(True, axis='y', alpha=0.3, linestyle='-', color='#cccccc')
    
    # Legend
    ax.legend(loc="upper right", fontsize=12, framealpha=0.95, 
              edgecolor='#dddddd', fancybox=True)
    
    # Tick styling
    ax.tick_params(axis='both', labelsize=11, colors='#333333')
    
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor='white')
    plt.close(fig)
    logger.info("Saved figure to %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create toxicity distribution comparison for simulations vs Voat samples."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing simulation run subdirectories.",
    )
    parser.add_argument(
        "--voat-dir",
        type=Path,
        default=Path("MADOC/voat-technology"),
        help="Directory containing Voat sample subdirectories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("workshop/figures/toxicity_pooled_30v30.png"),
        help="Output file for the generated figure.",
    )
    parser.add_argument("--confidence", type=float, default=0.99)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    voat_dir = args.voat_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    
    # Load toxicity scores per run/sample
    sim_runs = load_sim_toxicity_per_run(results_dir)
    voat_samples = load_voat_toxicity_per_sample(voat_dir)
    
    if len(sim_runs) == 0:
        logger.error("No simulation toxicity data loaded")
        return
    
    if len(voat_samples) == 0:
        logger.error("No Voat toxicity data loaded")
        return
    
    # Create comparison plot
    plot_toxicity_comparison(
        sim_runs,
        voat_samples,
        output_path,
        confidence=args.confidence,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
