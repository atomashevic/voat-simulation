#!/usr/bin/env python3
"""
Visualize convergence entropy means across 30 runs compared to a GPT-4o mini baseline.

Loads H_per_token overall mean from each run's agg_stats.json and compares
to the GPT-4o mini relative convergence entropy reported by Chaiyakul et al. (2025)
(mu = 0.2827 +/- 0.0002 bits/token).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def load_entropy_means(results_dir: Path) -> List[Dict[str, float]]:
    """Load overall H_per_token mean from each run's agg_stats.json."""
    data = []

    for run_dir in sorted(results_dir.glob("run*")):
        if not run_dir.is_dir():
            continue

        stats_file = run_dir / "agg_stats.json"
        if not stats_file.exists():
            continue

        try:
            with stats_file.open() as f:
                stats_data = json.load(f)

            h_per_token_mean = stats_data["overall"]["H_per_token"]["mean"]
            h_mean = stats_data["overall"]["H"]["mean"]
            count = stats_data["overall"]["H_per_token"]["count"]

            data.append({
                "run": run_dir.name,
                "H_per_token_mean": h_per_token_mean,
                "H_mean": h_mean,
                "count": count,
            })
        except Exception as e:
            print(f"Warning: Failed to load {stats_file}: {e}")

    return data


def create_benchmark_comparison_figure(
    data: List[Dict[str, float]],
    benchmark: float,
    output_path: Path,
) -> None:
    """Create comprehensive comparison figure."""

    df = pd.DataFrame(data)
    values = df["H_per_token_mean"].values

    # Compute statistics
    mean_val = np.mean(values)
    std_val = np.std(values, ddof=1)
    sem_val = std_val / np.sqrt(len(values))
    ci_95 = stats.t.interval(0.95, len(values) - 1, loc=mean_val, scale=sem_val)

    # One-sample t-test against benchmark
    t_stat, p_val = stats.ttest_1samp(values, benchmark)

    # Create figure with two panels - add extra height for title
    fig = plt.figure(figsize=(12, 5.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], hspace=0.3, wspace=0.35,
                         top=0.88, bottom=0.1, left=0.08, right=0.95)

    # === Panel 1: Stripplot with violin ===
    ax1 = fig.add_subplot(gs[0])

    # Violin plot
    parts = ax1.violinplot([values], positions=[0], widths=0.7,
                            showmeans=False, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor('#3498db')
        pc.set_alpha(0.3)
        pc.set_edgecolor('#2980b9')
        pc.set_linewidth(1.5)

    # Add jittered points
    np.random.seed(42)
    jitter = np.random.normal(0, 0.04, size=len(values))
    ax1.scatter(jitter, values, alpha=0.6, s=80, c='#2c3e50',
                edgecolor='white', linewidth=1, zorder=3)

    # Mean line
    ax1.hlines(mean_val, -0.4, 0.4, colors='#e74c3c', linewidth=3,
               label=f'Simulation mean: {mean_val:.4f}', zorder=4)

    # Benchmark line
    ax1.hlines(benchmark, -0.4, 0.4, colors='#f39c12', linewidth=3,
               linestyle='--', label=f'GPT-4o mini baseline: {benchmark}', zorder=4)

    # CI error bar
    ax1.errorbar([0], [mean_val], yerr=[[mean_val - ci_95[0]], [ci_95[1] - mean_val]],
                 fmt='none', ecolor='#e74c3c', elinewidth=2.5, capsize=8, capthick=2.5,
                 alpha=0.8, zorder=5)

    ax1.set_xlim(-0.5, 0.5)
    ax1.set_xticks([])
    ax1.set_ylabel('Convergence Entropy (H/T)', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution Across 30 Runs', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower center', fontsize=10, framealpha=0.95, ncol=1)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.grid(True, axis='y', alpha=0.3, linestyle='--')

    # === Panel 2: Run-by-run values ===
    ax2 = fig.add_subplot(gs[1])

    run_nums = np.arange(1, len(values) + 1)
    ax2.scatter(run_nums, values, s=60, c='#3498db', alpha=0.7,
                edgecolor='#2c3e50', linewidth=1)
    ax2.axhline(mean_val, color='#e74c3c', linewidth=2,
                label='Mean', linestyle='-', alpha=0.8)
    ax2.axhline(benchmark, color='#f39c12', linewidth=2,
                label='GPT-4o mini', linestyle='--', alpha=0.8)

    # Shade CI region
    ax2.fill_between([0, len(values) + 1], ci_95[0], ci_95[1],
                      color='#e74c3c', alpha=0.15, label='95% CI')

    ax2.set_xlabel('Run Number', fontsize=11, fontweight='bold')
    ax2.set_ylabel('H/T', fontsize=11, fontweight='bold')
    ax2.set_title('Run-by-Run Values', fontsize=13, fontweight='bold')
    ax2.set_xlim(0, len(values) + 1)
    ax2.legend(loc='best', fontsize=9, framealpha=0.95)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # Determine significance for summary
    if p_val < 0.001:
        sig_text = "p < 0.001***"
    elif p_val < 0.01:
        sig_text = f"p = {p_val:.3f}**"
    elif p_val < 0.05:
        sig_text = f"p = {p_val:.3f}*"
    else:
        sig_text = f"p = {p_val:.3f} (n.s.)"

    diff = mean_val - benchmark
    pct_diff = (diff / benchmark) * 100

    # Main title with proper spacing
    fig.suptitle(
        'Convergence Entropy: 30 Simulation Runs vs GPT-4o Mini Baseline',
        fontsize=14, fontweight='bold', y=0.97
    )

    # Subtitle
    fig.text(0.5, 0.92, 'Lower entropy indicates greater linguistic convergence',
             ha='center', fontsize=11, style='italic', color='#555555')

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"\n{'=' * 60}")
    print(f"Convergence Entropy Benchmark Comparison")
    print(f"{'=' * 60}")
    print(f"Simulation mean (n={len(values)}): {mean_val:.4f} ± {std_val:.4f}")
    print(f"95% CI: [{ci_95[0]:.4f}, {ci_95[1]:.4f}]")
    print(f"GPT-4o mini baseline (Chaiyakul et al., 2025): {benchmark}")
    print(f"Difference: {diff:+.4f} ({pct_diff:+.2f}%)")
    print(f"t({len(values)-1}) = {t_stat:.3f}, {sig_text}")
    print(f"\nFigure saved to: {output_path}")
    print(f"{'=' * 60}\n")


def main():
    results_dir = Path("results")
    output_path = Path("paper/figures/convergence_entropy_benchmark_comparison.png")
    benchmark = 0.2827  # GPT-4o mini mean relative convergence entropy (Chaiyakul et al., 2025)

    print("Loading convergence entropy data from 30 runs...")
    data = load_entropy_means(results_dir)

    if not data:
        raise ValueError("No entropy data found")

    print(f"Loaded {len(data)} runs")

    create_benchmark_comparison_figure(data, benchmark, output_path)


if __name__ == "__main__":
    main()
