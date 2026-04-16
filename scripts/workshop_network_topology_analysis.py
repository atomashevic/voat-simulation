#!/usr/bin/env python3
"""
Analyze network topology properties across simulation runs and Voat samples.

Tests for:
1. Random-like networks (Erdős-Rényi comparison)
2. Small-world properties (high clustering, short paths)
3. Preferential attachment (power-law degree distribution)
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

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


def fit_power_law(degrees: List[int]) -> Tuple[float, float, float]:
    """Fit power law to degree distribution using MLE.
    
    Returns: (alpha, xmin, ks_statistic)
    """
    degrees = np.array([d for d in degrees if d > 0])
    if len(degrees) < 10:
        return np.nan, np.nan, np.nan
    
    # Use simple MLE for power law: alpha = 1 + n / sum(ln(x/xmin))
    xmin = max(1, np.percentile(degrees, 10))  # Use 10th percentile as xmin
    degrees_above = degrees[degrees >= xmin]
    
    if len(degrees_above) < 5:
        return np.nan, xmin, np.nan
    
    n = len(degrees_above)
    alpha = 1 + n / np.sum(np.log(degrees_above / xmin))
    
    # KS test against fitted power law
    # Compare empirical CDF to theoretical
    sorted_d = np.sort(degrees_above)
    empirical_cdf = np.arange(1, n + 1) / n
    theoretical_cdf = 1 - (xmin / sorted_d) ** (alpha - 1)
    ks_stat = np.max(np.abs(empirical_cdf - theoretical_cdf))
    
    return alpha, xmin, ks_stat


def compute_network_metrics(G: nx.Graph) -> Dict[str, float]:
    """Compute comprehensive network topology metrics."""
    metrics = {}
    
    n = G.number_of_nodes()
    m = G.number_of_edges()
    metrics["n_nodes"] = n
    metrics["n_edges"] = m
    
    if n == 0 or m == 0:
        return metrics
    
    # Density
    metrics["density"] = nx.density(G)
    
    # Degree statistics
    degrees = [d for n, d in G.degree()]
    metrics["mean_degree"] = np.mean(degrees)
    metrics["max_degree"] = max(degrees) if degrees else 0
    
    # Clustering coefficient
    metrics["clustering"] = nx.average_clustering(G)
    
    # Use largest connected component for path-based metrics
    if nx.is_connected(G):
        lcc = G
    else:
        lcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    
    metrics["lcc_fraction"] = len(lcc) / n if n > 0 else 0
    
    # Average shortest path length (on LCC)
    if len(lcc) > 1:
        try:
            metrics["avg_path_length"] = nx.average_shortest_path_length(lcc)
        except Exception:
            metrics["avg_path_length"] = np.nan
    else:
        metrics["avg_path_length"] = np.nan
    
    # Diameter
    if len(lcc) > 1:
        try:
            metrics["diameter"] = nx.diameter(lcc)
        except Exception:
            metrics["diameter"] = np.nan
    else:
        metrics["diameter"] = np.nan
    
    # Assortativity (degree correlation)
    try:
        metrics["assortativity"] = nx.degree_assortativity_coefficient(G)
    except Exception:
        metrics["assortativity"] = np.nan
    
    # Power law fit for preferential attachment
    alpha, xmin, ks_stat = fit_power_law(degrees)
    metrics["powerlaw_alpha"] = alpha
    metrics["powerlaw_xmin"] = xmin
    metrics["powerlaw_ks"] = ks_stat
    
    # Compare to random graph (Erdős-Rényi)
    # Expected clustering for ER: p = 2m / (n*(n-1))
    if n > 1:
        p_er = 2 * m / (n * (n - 1))
        metrics["er_expected_clustering"] = p_er
        metrics["clustering_ratio"] = metrics["clustering"] / p_er if p_er > 0 else np.nan
        
        # Expected path length for ER: ln(n) / ln(k) where k = mean degree
        if metrics["mean_degree"] > 1:
            metrics["er_expected_path"] = np.log(n) / np.log(metrics["mean_degree"])
        else:
            metrics["er_expected_path"] = np.nan
        
        if not np.isnan(metrics["avg_path_length"]) and not np.isnan(metrics["er_expected_path"]):
            metrics["path_ratio"] = metrics["avg_path_length"] / metrics["er_expected_path"]
        else:
            metrics["path_ratio"] = np.nan
    
    # Small-world coefficient: sigma = (C/C_rand) / (L/L_rand)
    # sigma > 1 suggests small-world
    if not np.isnan(metrics.get("clustering_ratio", np.nan)) and not np.isnan(metrics.get("path_ratio", np.nan)):
        if metrics["path_ratio"] > 0:
            metrics["small_world_sigma"] = metrics["clustering_ratio"] / metrics["path_ratio"]
        else:
            metrics["small_world_sigma"] = np.nan
    else:
        metrics["small_world_sigma"] = np.nan
    
    return metrics


def load_all_networks(
    results_dir: Path,
    voat_dir: Path,
) -> Tuple[List[Tuple[str, Dict]], List[Tuple[str, Dict]]]:
    """Load and compute metrics for all networks."""
    
    sim_metrics = []
    voat_metrics = []
    
    # Load simulation networks
    for run_dir in sorted(results_dir.glob("run*")):
        if not run_dir.is_dir():
            continue
        
        posts_file = run_dir / "posts.csv"
        if posts_file.exists():
            try:
                G = build_network_from_posts(posts_file)
                metrics = compute_network_metrics(G)
                sim_metrics.append((run_dir.name, metrics))
                logger.debug("Computed metrics for %s", run_dir.name)
            except Exception as e:
                logger.warning("Failed to process %s: %s", run_dir.name, e)
    
    logger.info("Processed %d simulation networks", len(sim_metrics))
    
    # Load Voat networks
    for sample_dir in sorted(voat_dir.glob("sample_*")):
        if not sample_dir.is_dir():
            continue
        
        parquet_files = list(sample_dir.glob("*.parquet"))
        if parquet_files:
            try:
                G = build_network_from_parquet(parquet_files[0])
                metrics = compute_network_metrics(G)
                voat_metrics.append((sample_dir.name, metrics))
                logger.debug("Computed metrics for %s", sample_dir.name)
            except Exception as e:
                logger.warning("Failed to process %s: %s", sample_dir.name, e)
    
    logger.info("Processed %d Voat networks", len(voat_metrics))
    
    return sim_metrics, voat_metrics


def create_topology_comparison_plot(
    sim_metrics: List[Tuple[str, Dict]],
    voat_metrics: List[Tuple[str, Dict]],
    output_path: Path,
) -> None:
    """Create comparison plot of network topology metrics."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Set background
    fig.patch.set_facecolor('white')
    for ax in axes.flat:
        ax.set_facecolor('#fafafa')
    
    # Colors
    sim_color = "#1f77b4"
    voat_color = "#ff7f0e"
    
    # Extract values
    def get_vals(metrics_list, key):
        return [m[key] for name, m in metrics_list if key in m and not np.isnan(m.get(key, np.nan))]
    
    # 1. Clustering Coefficient
    ax = axes[0, 0]
    sim_clust = get_vals(sim_metrics, "clustering")
    voat_clust = get_vals(voat_metrics, "clustering")
    
    positions = [0, 1]
    bp = ax.boxplot([sim_clust, voat_clust], positions=positions, widths=0.6, patch_artist=True)
    bp['boxes'][0].set_facecolor(sim_color)
    bp['boxes'][1].set_facecolor(voat_color)
    for box in bp['boxes']:
        box.set_alpha(0.6)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(['Simulation', 'Voat'])
    ax.set_ylabel('Clustering Coefficient', fontweight='bold')
    ax.set_title('Clustering\n(Higher = More Clustered)', fontweight='bold')
    
    # 2. Average Path Length
    ax = axes[0, 1]
    sim_path = get_vals(sim_metrics, "avg_path_length")
    voat_path = get_vals(voat_metrics, "avg_path_length")
    
    bp = ax.boxplot([sim_path, voat_path], positions=positions, widths=0.6, patch_artist=True)
    bp['boxes'][0].set_facecolor(sim_color)
    bp['boxes'][1].set_facecolor(voat_color)
    for box in bp['boxes']:
        box.set_alpha(0.6)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(['Simulation', 'Voat'])
    ax.set_ylabel('Avg. Shortest Path Length', fontweight='bold')
    ax.set_title('Path Length\n(Lower = More Connected)', fontweight='bold')
    
    # 3. Small-World Sigma
    ax = axes[0, 2]
    sim_sigma = get_vals(sim_metrics, "small_world_sigma")
    voat_sigma = get_vals(voat_metrics, "small_world_sigma")
    
    bp = ax.boxplot([sim_sigma, voat_sigma], positions=positions, widths=0.6, patch_artist=True)
    bp['boxes'][0].set_facecolor(sim_color)
    bp['boxes'][1].set_facecolor(voat_color)
    for box in bp['boxes']:
        box.set_alpha(0.6)
    
    ax.axhline(1, color='red', linestyle='--', linewidth=1.5, label='σ=1 (Random)')
    ax.set_xticks(positions)
    ax.set_xticklabels(['Simulation', 'Voat'])
    ax.set_ylabel('Small-World σ', fontweight='bold')
    ax.set_title('Small-World Coefficient\n(σ > 1 = Small-World)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    
    # 4. Power-Law Alpha (Preferential Attachment)
    ax = axes[1, 0]
    sim_alpha = get_vals(sim_metrics, "powerlaw_alpha")
    voat_alpha = get_vals(voat_metrics, "powerlaw_alpha")
    
    bp = ax.boxplot([sim_alpha, voat_alpha], positions=positions, widths=0.6, patch_artist=True)
    bp['boxes'][0].set_facecolor(sim_color)
    bp['boxes'][1].set_facecolor(voat_color)
    for box in bp['boxes']:
        box.set_alpha(0.6)
    
    ax.axhline(2, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axhline(3, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Typical α (2-3)')
    ax.set_xticks(positions)
    ax.set_xticklabels(['Simulation', 'Voat'])
    ax.set_ylabel('Power-Law α', fontweight='bold')
    ax.set_title('Degree Distribution Exponent\n(α ∈ [2,3] = Pref. Attachment)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    
    # 5. Degree Assortativity
    ax = axes[1, 1]
    sim_assort = get_vals(sim_metrics, "assortativity")
    voat_assort = get_vals(voat_metrics, "assortativity")
    
    bp = ax.boxplot([sim_assort, voat_assort], positions=positions, widths=0.6, patch_artist=True)
    bp['boxes'][0].set_facecolor(sim_color)
    bp['boxes'][1].set_facecolor(voat_color)
    for box in bp['boxes']:
        box.set_alpha(0.6)
    
    ax.axhline(0, color='gray', linestyle='--', linewidth=1.5, label='Neutral')
    ax.set_xticks(positions)
    ax.set_xticklabels(['Simulation', 'Voat'])
    ax.set_ylabel('Assortativity', fontweight='bold')
    ax.set_title('Degree Assortativity\n(< 0 = Disassortative)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    
    # 6. Clustering vs Random (C/C_rand ratio)
    ax = axes[1, 2]
    sim_cratio = get_vals(sim_metrics, "clustering_ratio")
    voat_cratio = get_vals(voat_metrics, "clustering_ratio")
    
    bp = ax.boxplot([sim_cratio, voat_cratio], positions=positions, widths=0.6, patch_artist=True)
    bp['boxes'][0].set_facecolor(sim_color)
    bp['boxes'][1].set_facecolor(voat_color)
    for box in bp['boxes']:
        box.set_alpha(0.6)
    
    ax.axhline(1, color='red', linestyle='--', linewidth=1.5, label='Random (ratio=1)')
    ax.set_xticks(positions)
    ax.set_xticklabels(['Simulation', 'Voat'])
    ax.set_ylabel('C / C_random', fontweight='bold')
    ax.set_title('Clustering vs Random\n(> 1 = More Clustered)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    
    # Style all axes
    for ax in axes.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cccccc')
        ax.spines['bottom'].set_color('#cccccc')
        ax.grid(True, axis='y', alpha=0.3, linestyle='-', color='#cccccc')
        ax.tick_params(labelsize=10)
    
    fig.suptitle('Network Topology Analysis: Simulation vs Voat\n'
                 'Testing for Random, Small-World, and Preferential Attachment Properties',
                 fontsize=14, fontweight='bold', y=1.02)
    
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info("Saved comparison plot to %s", output_path)


def create_summary_stats_plot(
    sim_metrics: List[Tuple[str, Dict]],
    voat_metrics: List[Tuple[str, Dict]],
    output_path: Path,
) -> None:
    """Create summary statistics and interpretation plot."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('white')
    
    # Calculate statistics
    def get_stats(metrics_list, key):
        vals = [m[key] for name, m in metrics_list if key in m and not np.isnan(m.get(key, np.nan))]
        if not vals:
            return np.nan, np.nan
        return np.mean(vals), np.std(vals)
    
    metrics_to_show = [
        ("clustering", "Clustering Coefficient"),
        ("avg_path_length", "Avg. Path Length"),
        ("small_world_sigma", "Small-World σ"),
        ("powerlaw_alpha", "Power-Law α"),
        ("assortativity", "Assortativity"),
        ("clustering_ratio", "C/C_random Ratio"),
        ("density", "Network Density"),
        ("mean_degree", "Mean Degree"),
    ]
    
    # Create table data
    table_data = []
    for key, label in metrics_to_show:
        sim_mean, sim_std = get_stats(sim_metrics, key)
        voat_mean, voat_std = get_stats(voat_metrics, key)
        
        # Statistical test
        sim_vals = [m[key] for name, m in sim_metrics if key in m and not np.isnan(m.get(key, np.nan))]
        voat_vals = [m[key] for name, m in voat_metrics if key in m and not np.isnan(m.get(key, np.nan))]
        
        if len(sim_vals) > 2 and len(voat_vals) > 2:
            t_stat, p_val = stats.ttest_ind(sim_vals, voat_vals)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        else:
            p_val = np.nan
            sig = ""
        
        table_data.append([
            label,
            f"{sim_mean:.3f} ± {sim_std:.3f}" if not np.isnan(sim_mean) else "N/A",
            f"{voat_mean:.3f} ± {voat_std:.3f}" if not np.isnan(voat_mean) else "N/A",
            f"{p_val:.4f}{sig}" if not np.isnan(p_val) else "N/A"
        ])
    
    # Hide axes
    ax.axis('off')
    
    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=['Metric', 'Simulation (mean ± std)', 'Voat (mean ± std)', 'p-value'],
        cellLoc='center',
        loc='center',
        colWidths=[0.25, 0.25, 0.25, 0.15]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#1f77b4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('white')
    
    # Add interpretation text
    interpretations = [
        "Interpretation Guide:",
        "• Small-World: σ > 1 indicates small-world properties (high clustering, short paths)",
        "• Preferential Attachment: Power-law α typically 2-3 for scale-free networks",  
        "• Assortativity < 0: Hubs connect to non-hubs (typical of social networks)",
        "• C/C_random >> 1: Much more clustered than random (community structure)",
        "",
        "Significance: * p<0.05, ** p<0.01, *** p<0.001"
    ]
    
    ax.text(0.5, -0.15, '\n'.join(interpretations), transform=ax.transAxes,
            fontsize=10, family='monospace', ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='#dddddd'))
    
    ax.set_title('Network Topology: Statistical Summary\n'
                 f'Simulation (n={len(sim_metrics)}) vs Voat (n={len(voat_metrics)})',
                 fontsize=14, fontweight='bold', pad=20)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info("Saved summary stats to %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze network topology properties."
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
        "--output-dir",
        type=Path,
        default=Path("workshop/figures"),
        help="Output directory for figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    voat_dir = args.voat_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    
    sim_metrics, voat_metrics = load_all_networks(results_dir, voat_dir)
    
    if not sim_metrics and not voat_metrics:
        logger.error("No networks loaded")
        return
    
    # Create comparison boxplots
    create_topology_comparison_plot(
        sim_metrics, voat_metrics,
        output_dir / "network_topology_comparison.png"
    )
    
    # Create summary statistics table
    create_summary_stats_plot(
        sim_metrics, voat_metrics,
        output_dir / "network_topology_summary.png"
    )


if __name__ == "__main__":
    main()

