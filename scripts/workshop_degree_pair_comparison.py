#!/usr/bin/env python3
"""
Compare degree distribution metrics for a specific sim-voat pair.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def build_network_from_posts(posts_csv: Path) -> nx.Graph:
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


def fit_powerlaw_mle(degrees: np.ndarray, xmin: int = 2):
    degrees = np.array([d for d in degrees if d >= xmin])
    if len(degrees) < 10:
        return np.nan
    n = len(degrees)
    alpha = 1 + n / np.sum(np.log(degrees / (xmin - 0.5)))
    return alpha


def compute_metrics(G: nx.Graph) -> Dict[str, float]:
    degrees = np.array([d for n, d in G.degree()])
    degrees = degrees[degrees > 0]
    
    n_nodes = len(degrees)
    
    metrics = {}
    metrics["n_nodes"] = n_nodes
    metrics["n_edges"] = G.number_of_edges()
    metrics["mean_degree"] = np.mean(degrees)
    metrics["max_degree"] = np.max(degrees)
    metrics["max_to_mean_ratio"] = metrics["max_degree"] / metrics["mean_degree"]
    
    # Top 1% share
    sorted_degrees = np.sort(degrees)[::-1]
    total_degree = np.sum(degrees)
    top_n = max(1, int(n_nodes * 0.01))
    top_degree = np.sum(sorted_degrees[:top_n])
    metrics["top_1pct_share"] = 100 * top_degree / total_degree
    
    # Skewness
    metrics["skewness"] = stats.skew(degrees)
    
    # Power law alpha
    metrics["powerlaw_alpha"] = fit_powerlaw_mle(degrees, xmin=2)
    
    return metrics, degrees


def create_comparison_figure(
    sim_metrics: Dict,
    voat_metrics: Dict,
    sim_degrees: np.ndarray,
    voat_degrees: np.ndarray,
    sim_name: str,
    voat_name: str,
    output_path: Path,
) -> None:
    """Create side-by-side comparison figure."""
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    
    fig.patch.set_facecolor('white')
    
    sim_color = "#1f77b4"
    voat_color = "#ff7f0e"
    
    # All 4 metrics in one row
    metrics_to_plot = [
        ("max_to_mean_ratio", "Max/Mean Ratio", "Higher = Bigger Hubs", False),
        ("top_1pct_share", "Top 1% Share (%)", "Higher = More Concentrated", False),
        ("skewness", "Skewness", "Higher = More Right-Skewed", False),
        ("powerlaw_alpha", "Power-Law α", "Lower = Heavier Tail", True),
    ]
    
    for idx, (key, label, subtitle, is_alpha) in enumerate(metrics_to_plot):
        ax = axes[idx]
        ax.set_facecolor('#fafafa')
        
        sim_val = sim_metrics[key]
        voat_val = voat_metrics[key]
        
        bars = ax.bar([0, 1], [sim_val, voat_val], color=[sim_color, voat_color], 
                      alpha=0.7, edgecolor='black', linewidth=1.5)
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f'Sim\n({sim_name})', f'Voat\n({voat_name})'], fontsize=10)
        ax.set_ylabel(label, fontweight='bold', fontsize=10)
        ax.set_title(f'{label}\n({subtitle})', fontweight='bold', fontsize=11)
        
        # Add reference lines for power-law alpha
        if is_alpha:
            ax.axhline(2, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
            ax.axhline(3, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
            ax.fill_between([-0.5, 1.5], 2, 3, color='green', alpha=0.1)
            ax.set_xlim(-0.5, 1.5)
        
        # Add value labels on bars
        fmt = f'{sim_val:.2f}' if is_alpha else f'{sim_val:.1f}'
        for bar, val in zip(bars, [sim_val, voat_val]):
            fmt_val = f'{val:.2f}' if is_alpha else f'{val:.1f}'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02*max(sim_val, voat_val),
                    fmt_val, ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cccccc')
        ax.spines['bottom'].set_color('#cccccc')
        ax.grid(True, axis='y', alpha=0.3, linestyle='-', color='#cccccc')
    
    fig.suptitle(f'Degree Distribution Metrics: {sim_name} vs {voat_name}',
                 fontsize=14, fontweight='bold', y=1.02)
    
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info("Saved figure to %s", output_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-run", type=str, default="run03")
    parser.add_argument("--voat-sample", type=str, default="sample_23")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--voat-dir", type=Path, default=Path("MADOC/voat-technology"))
    parser.add_argument("--output", type=Path, default=Path("workshop/figures/degree_pair_comparison.png"))
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load simulation network
    sim_path = args.results_dir / args.sim_run / "posts.csv"
    G_sim = build_network_from_posts(sim_path)
    sim_metrics, sim_degrees = compute_metrics(G_sim)
    
    # Load Voat network
    voat_sample_dir = args.voat_dir / args.voat_sample
    voat_parquet = list(voat_sample_dir.glob("*.parquet"))[0]
    G_voat = build_network_from_parquet(voat_parquet)
    voat_metrics, voat_degrees = compute_metrics(G_voat)
    
    logger.info("Simulation (%s): %d nodes, %d edges", args.sim_run, sim_metrics["n_nodes"], sim_metrics["n_edges"])
    logger.info("Voat (%s): %d nodes, %d edges", args.voat_sample, voat_metrics["n_nodes"], voat_metrics["n_edges"])
    
    # Create figure
    create_comparison_figure(
        sim_metrics, voat_metrics,
        sim_degrees, voat_degrees,
        args.sim_run, args.voat_sample,
        args.output
    )


if __name__ == "__main__":
    main()

