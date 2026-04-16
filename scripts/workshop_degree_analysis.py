#!/usr/bin/env python3
"""
Rigorous analysis of degree distributions and hub concentration.

Compares 30 simulation runs against 30 matched Voat windows.
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
from scipy.optimize import minimize_scalar

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def build_network_from_posts(posts_csv: Path) -> nx.Graph:
    """Build undirected network from simulation posts.csv."""
    df = pd.read_csv(posts_csv)
    G = nx.Graph()
    G.add_nodes_from(df["user_id"].unique())
    
    comment_df = df[df["comment_to"] != -1]
    post_users = df.set_index("id")["user_id"].to_dict()
    
    for _, row in comment_df.iterrows():
        commenter = row["user_id"]
        parent_id = row["comment_to"]
        parent_user = post_users.get(parent_id)
        
        if parent_user and parent_user != commenter:
            if G.has_edge(commenter, parent_user):
                G[commenter][parent_user]["weight"] += 1
            else:
                G.add_edge(commenter, parent_user, weight=1)
    
    return G


def build_network_from_parquet(parquet_path: Path) -> nx.Graph:
    """Build undirected network from Voat parquet."""
    df = pd.read_parquet(parquet_path)
    G = nx.Graph()
    
    if "user_id" not in df.columns:
        return G
    
    G.add_nodes_from(df["user_id"].dropna().unique())
    
    if "parent_id" in df.columns:
        post_users = df.set_index("post_id")["user_id"].to_dict()
        
        for _, row in df.iterrows():
            if pd.notna(row.get("parent_id")) and row["parent_id"] != row["post_id"]:
                child_user = row["user_id"]
                parent_user = post_users.get(row["parent_id"])
                
                if parent_user and child_user and parent_user != child_user:
                    if G.has_edge(child_user, parent_user):
                        G[child_user][parent_user]["weight"] += 1
                    else:
                        G.add_edge(child_user, parent_user, weight=1)
    
    return G


def gini_coefficient(values: np.ndarray) -> float:
    """Calculate Gini coefficient (0=perfect equality, 1=perfect inequality)."""
    values = np.array(values, dtype=float)
    values = values[values > 0]
    if len(values) < 2:
        return 0.0
    sorted_values = np.sort(values)
    n = len(sorted_values)
    cumsum = np.cumsum(sorted_values)
    gini = (2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (n * cumsum[-1]) - (n + 1) / n
    return gini


def fit_powerlaw_mle(degrees: np.ndarray, xmin: int = None) -> Tuple[float, int, float]:
    """
    Fit power law using MLE (Clauset et al. method).
    
    Returns: (alpha, xmin, ks_statistic)
    """
    degrees = np.array([d for d in degrees if d > 0])
    if len(degrees) < 20:
        return np.nan, np.nan, np.nan
    
    # If xmin not provided, find optimal xmin
    if xmin is None:
        unique_degrees = np.unique(degrees)
        unique_degrees = unique_degrees[unique_degrees >= 1]
        
        best_xmin = 1
        best_ks = np.inf
        best_alpha = np.nan
        
        for candidate_xmin in unique_degrees[:min(20, len(unique_degrees))]:
            tail = degrees[degrees >= candidate_xmin]
            if len(tail) < 10:
                continue
            
            # MLE for alpha
            n = len(tail)
            alpha = 1 + n / np.sum(np.log(tail / (candidate_xmin - 0.5)))
            
            if alpha < 1.5 or alpha > 5:
                continue
            
            # KS statistic
            sorted_tail = np.sort(tail)
            empirical_cdf = np.arange(1, n + 1) / n
            theoretical_cdf = 1 - (candidate_xmin / sorted_tail) ** (alpha - 1)
            ks = np.max(np.abs(empirical_cdf - theoretical_cdf))
            
            if ks < best_ks:
                best_ks = ks
                best_xmin = candidate_xmin
                best_alpha = alpha
        
        return best_alpha, best_xmin, best_ks
    else:
        tail = degrees[degrees >= xmin]
        if len(tail) < 10:
            return np.nan, xmin, np.nan
        
        n = len(tail)
        alpha = 1 + n / np.sum(np.log(tail / (xmin - 0.5)))
        
        sorted_tail = np.sort(tail)
        empirical_cdf = np.arange(1, n + 1) / n
        theoretical_cdf = 1 - (xmin / sorted_tail) ** (alpha - 1)
        ks = np.max(np.abs(empirical_cdf - theoretical_cdf))
        
        return alpha, xmin, ks


def compute_degree_metrics(G: nx.Graph) -> Dict[str, float]:
    """Compute comprehensive degree distribution metrics."""
    metrics = {}
    
    degrees = np.array([d for n, d in G.degree()])
    degrees = degrees[degrees > 0]
    
    if len(degrees) < 5:
        return metrics
    
    n_nodes = len(degrees)
    n_edges = G.number_of_edges()
    
    metrics["n_nodes"] = n_nodes
    metrics["n_edges"] = n_edges
    
    # Basic stats
    metrics["mean_degree"] = np.mean(degrees)
    metrics["median_degree"] = np.median(degrees)
    metrics["max_degree"] = np.max(degrees)
    metrics["std_degree"] = np.std(degrees)
    
    # Hub concentration metrics
    metrics["max_to_mean_ratio"] = metrics["max_degree"] / metrics["mean_degree"]
    metrics["max_to_median_ratio"] = metrics["max_degree"] / metrics["median_degree"]
    
    # Gini coefficient (inequality)
    metrics["gini"] = gini_coefficient(degrees)
    
    # Top-heavy metrics
    sorted_degrees = np.sort(degrees)[::-1]
    total_degree = np.sum(degrees)  # = 2 * edges
    
    # % of total degree held by top 1%, 5%, 10% of nodes
    for pct in [1, 5, 10]:
        top_n = max(1, int(n_nodes * pct / 100))
        top_degree = np.sum(sorted_degrees[:top_n])
        metrics[f"top_{pct}pct_share"] = 100 * top_degree / total_degree
    
    # Power law fit (with proper xmin search)
    alpha, xmin, ks = fit_powerlaw_mle(degrees)
    metrics["powerlaw_alpha"] = alpha
    metrics["powerlaw_xmin"] = xmin
    metrics["powerlaw_ks"] = ks
    
    # Also fit with fixed xmin=2 for comparison
    alpha_fixed, _, ks_fixed = fit_powerlaw_mle(degrees, xmin=2)
    metrics["powerlaw_alpha_xmin2"] = alpha_fixed
    metrics["powerlaw_ks_xmin2"] = ks_fixed
    
    # Coefficient of variation
    metrics["cv"] = metrics["std_degree"] / metrics["mean_degree"] if metrics["mean_degree"] > 0 else 0
    
    # Skewness
    metrics["skewness"] = stats.skew(degrees)
    
    return metrics


def load_all_metrics(
    results_dir: Path,
    voat_dir: Path,
) -> Tuple[List[Tuple[str, Dict]], List[Tuple[str, Dict]]]:
    """Load and compute metrics for all networks."""
    
    sim_metrics = []
    voat_metrics = []
    
    for run_dir in sorted(results_dir.glob("run*")):
        if not run_dir.is_dir():
            continue
        posts_file = run_dir / "posts.csv"
        if posts_file.exists():
            try:
                G = build_network_from_posts(posts_file)
                metrics = compute_degree_metrics(G)
                if metrics:
                    sim_metrics.append((run_dir.name, metrics))
            except Exception as e:
                logger.warning("Failed: %s - %s", run_dir.name, e)
    
    logger.info("Processed %d simulation networks", len(sim_metrics))
    
    for sample_dir in sorted(voat_dir.glob("sample_*")):
        if not sample_dir.is_dir():
            continue
        parquet_files = list(sample_dir.glob("*.parquet"))
        if parquet_files:
            try:
                G = build_network_from_parquet(parquet_files[0])
                metrics = compute_degree_metrics(G)
                if metrics:
                    voat_metrics.append((sample_dir.name, metrics))
            except Exception as e:
                logger.warning("Failed: %s - %s", sample_dir.name, e)
    
    logger.info("Processed %d Voat networks", len(voat_metrics))
    
    return sim_metrics, voat_metrics


def create_comparison_plot(
    sim_metrics: List[Tuple[str, Dict]],
    voat_metrics: List[Tuple[str, Dict]],
    output_path: Path,
) -> None:
    """Create comparison plot with better metrics."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.patch.set_facecolor('white')
    
    for ax in axes.flat:
        ax.set_facecolor('#fafafa')
    
    sim_color = "#1f77b4"
    voat_color = "#ff7f0e"
    
    def get_vals(metrics_list, key):
        return [m[key] for name, m in metrics_list if key in m and not np.isnan(m.get(key, np.nan))]
    
    positions = [0, 1]
    
    # 1. Gini Coefficient
    ax = axes[0, 0]
    sim_vals = get_vals(sim_metrics, "gini")
    voat_vals = get_vals(voat_metrics, "gini")
    
    bp = ax.boxplot([sim_vals, voat_vals], positions=positions, widths=0.6, patch_artist=True)
    bp['boxes'][0].set_facecolor(sim_color)
    bp['boxes'][1].set_facecolor(voat_color)
    for box in bp['boxes']:
        box.set_alpha(0.6)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(['Simulation', 'Voat'])
    ax.set_ylabel('Gini Coefficient', fontweight='bold')
    ax.set_title('Degree Inequality (Gini)\n(Higher = More Unequal)', fontweight='bold')
    
    # 2. Max/Mean Ratio
    ax = axes[0, 1]
    sim_vals = get_vals(sim_metrics, "max_to_mean_ratio")
    voat_vals = get_vals(voat_metrics, "max_to_mean_ratio")
    
    bp = ax.boxplot([sim_vals, voat_vals], positions=positions, widths=0.6, patch_artist=True)
    bp['boxes'][0].set_facecolor(sim_color)
    bp['boxes'][1].set_facecolor(voat_color)
    for box in bp['boxes']:
        box.set_alpha(0.6)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(['Simulation', 'Voat'])
    ax.set_ylabel('Max Degree / Mean Degree', fontweight='bold')
    ax.set_title('Hub Dominance\n(Higher = Bigger Hubs)', fontweight='bold')
    
    # 3. Top 10% Share
    ax = axes[0, 2]
    sim_vals = get_vals(sim_metrics, "top_10pct_share")
    voat_vals = get_vals(voat_metrics, "top_10pct_share")
    
    bp = ax.boxplot([sim_vals, voat_vals], positions=positions, widths=0.6, patch_artist=True)
    bp['boxes'][0].set_facecolor(sim_color)
    bp['boxes'][1].set_facecolor(voat_color)
    for box in bp['boxes']:
        box.set_alpha(0.6)
    
    ax.axhline(10, color='gray', linestyle='--', linewidth=1.5, label='Equal (10%)')
    ax.set_xticks(positions)
    ax.set_xticklabels(['Simulation', 'Voat'])
    ax.set_ylabel('% of Connections', fontweight='bold')
    ax.set_title('Top 10% Node Share\n(Higher = More Concentrated)', fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    
    # 4. Top 5% Share
    ax = axes[1, 0]
    sim_vals = get_vals(sim_metrics, "top_5pct_share")
    voat_vals = get_vals(voat_metrics, "top_5pct_share")
    
    bp = ax.boxplot([sim_vals, voat_vals], positions=positions, widths=0.6, patch_artist=True)
    bp['boxes'][0].set_facecolor(sim_color)
    bp['boxes'][1].set_facecolor(voat_color)
    for box in bp['boxes']:
        box.set_alpha(0.6)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(['Simulation', 'Voat'])
    ax.set_ylabel('% of Connections', fontweight='bold')
    ax.set_title('Top 5% Node Share\n(Higher = More Concentrated)', fontweight='bold')
    
    # 5. Skewness
    ax = axes[1, 1]
    sim_vals = get_vals(sim_metrics, "skewness")
    voat_vals = get_vals(voat_metrics, "skewness")
    
    bp = ax.boxplot([sim_vals, voat_vals], positions=positions, widths=0.6, patch_artist=True)
    bp['boxes'][0].set_facecolor(sim_color)
    bp['boxes'][1].set_facecolor(voat_color)
    for box in bp['boxes']:
        box.set_alpha(0.6)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(['Simulation', 'Voat'])
    ax.set_ylabel('Skewness', fontweight='bold')
    ax.set_title('Degree Skewness\n(Higher = More Right-Skewed)', fontweight='bold')
    
    # 6. Top 1% Share
    ax = axes[1, 2]
    sim_vals = get_vals(sim_metrics, "top_1pct_share")
    voat_vals = get_vals(voat_metrics, "top_1pct_share")
    
    bp = ax.boxplot([sim_vals, voat_vals], positions=positions, widths=0.6, patch_artist=True)
    bp['boxes'][0].set_facecolor(sim_color)
    bp['boxes'][1].set_facecolor(voat_color)
    for box in bp['boxes']:
        box.set_alpha(0.6)
    
    ax.axhline(1, color='gray', linestyle='--', linewidth=1.5, label='Equal (1%)')
    ax.set_xticks(positions)
    ax.set_xticklabels(['Simulation', 'Voat'])
    ax.set_ylabel('% of Connections', fontweight='bold')
    ax.set_title('Top 1% Node Share\n(Mega-Hub Concentration)', fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    
    # Style
    for ax in axes.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cccccc')
        ax.spines['bottom'].set_color('#cccccc')
        ax.grid(True, axis='y', alpha=0.3, linestyle='-', color='#cccccc')
        ax.tick_params(labelsize=10)
    
    fig.suptitle(
        'Degree Distribution Analysis: Simulation vs Voat\n'
        'Better Metrics for Hub Concentration and Inequality',
        fontsize=14,
        fontweight='bold',
        y=1.02,
    )
    
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info("Saved plot to %s", output_path)


def print_summary_stats(sim_metrics, voat_metrics):
    """Print summary statistics table."""
    
    def get_stats(metrics_list, key):
        vals = [m[key] for name, m in metrics_list if key in m and not np.isnan(m.get(key, np.nan))]
        if not vals:
            return "N/A", "N/A"
        return f"{np.mean(vals):.3f}", f"{np.std(vals):.3f}"
    
    print("\n" + "="*70)
    print("DEGREE DISTRIBUTION ANALYSIS SUMMARY")
    print("="*70)
    print(f"{'Metric':<25} {'Sim Mean':<12} {'Sim Std':<12} {'Voat Mean':<12} {'Voat Std':<12}")
    print("-"*70)
    
    metrics = [
        ("gini", "Gini Coefficient"),
        ("max_to_mean_ratio", "Max/Mean Ratio"),
        ("top_1pct_share", "Top 1% Share (%)"),
        ("top_10pct_share", "Top 10% Share (%)"),
        ("top_5pct_share", "Top 5% Share (%)"),
        ("skewness", "Skewness"),
        ("cv", "Coef. of Variation"),
    ]
    
    for key, label in metrics:
        sim_mean, sim_std = get_stats(sim_metrics, key)
        voat_mean, voat_std = get_stats(voat_metrics, key)
        print(f"{label:<25} {sim_mean:<12} {sim_std:<12} {voat_mean:<12} {voat_std:<12}")
    
    print("="*70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Degree distribution analysis.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--voat-dir", type=Path, default=Path("MADOC/voat-technology-midlife30"))
    parser.add_argument("--output-dir", type=Path, default=Path("workshop/figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    voat_dir = args.voat_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    
    sim_metrics, voat_metrics = load_all_metrics(results_dir, voat_dir)
    
    if not sim_metrics and not voat_metrics:
        logger.error("No networks loaded")
        return
    
    # Print summary
    print_summary_stats(sim_metrics, voat_metrics)
    
    # Create plot
    create_comparison_plot(
        sim_metrics, voat_metrics,
        output_dir / "degree_distribution_analysis.png"
    )


if __name__ == "__main__":
    main()
