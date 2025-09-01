"""
visualize_simulation_additional.py

Create additional network visualizations for Simulation-2 results, using the
outputs produced by core-periphery-enhanced-voat.py. It rebuilds the interaction
network from posts.csv and uses enhanced_core_periphery_membership.csv (if
available) for core-periphery labels and reference metrics.

Generates four plots saved into the simulation folder:
  1) Full network: nodes scaled by degree, transparent rendering
  2) Largest component: nodes colored by weighted degree (uniform size)
  3) Largest component: nodes colored by k-core, nodes sized by weighted degree
  4) Largest component: nodes colored by core-periphery, nodes sized by weighted degree

Usage:
  python scripts/visualize_simulation_additional.py --sim-dir simulation

Notes:
  - Weighted degree is computed from normalized edge weights (sum of weights).
  - Core-periphery labels are read from enhanced_core_periphery_membership.csv
    or fallback to core_periphery_membership.csv when enhanced is missing.
  - All titles include the label "Simulated Network" as requested.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import pandas as pd


def build_graph_from_posts(posts_csv: Path) -> Tuple[nx.Graph, Dict[int, float], Dict[int, int]]:
    """Build undirected user interaction network from posts CSV.

    Expects columns: id, user_id, comment_to
    Edges connect (user_id -> parent_user_id) for replies (comment_to != -1).

    Returns:
      - G: undirected graph with attrs 'weight' (normalized) and 'raw_weight'
      - weighted_degree: sum of normalized weights per node
      - degree_dict: unweighted degree per node
    """
    df = pd.read_csv(posts_csv)

    # Ensure required columns exist
    required_cols = {"id", "user_id", "comment_to"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in posts.csv: {sorted(missing)}")

    # Prepare parent lookup (post id -> parent user id)
    parent_lookup = df[["id", "user_id"]].rename(
        columns={"user_id": "parent_user_id"}
    )

    # Join to map each comment to its parent's user_id
    df_edges = (
        df.merge(
            parent_lookup,
            left_on="comment_to",
            right_on="id",
            how="left",
            suffixes=("", "_parent"),
        )
        .loc[lambda d: d["comment_to"].notna()]
        .loc[lambda d: d["comment_to"] != -1]
    )

    # Keep clean pairs and drop self-interactions
    df_pairs = (
        df_edges[["user_id", "parent_user_id"]]
        .dropna()
        .astype({"user_id": int, "parent_user_id": int})
        .loc[lambda d: d["user_id"] != d["parent_user_id"]]
    )

    # Aggregate weights by pair
    pair_weights = (
        df_pairs.groupby(["user_id", "parent_user_id"]).size().reset_index(name="weight")
    )

    # Build undirected graph with normalized weights
    G = nx.from_pandas_edgelist(
        pair_weights, source="user_id", target="parent_user_id", edge_attr="weight"
    )
    if G.number_of_edges() > 0:
        max_w = max(G[u][v]["weight"] for u, v in G.edges())
        for u, v in G.edges():
            raw_w = G[u][v]["weight"]
            G[u][v]["raw_weight"] = raw_w
            G[u][v]["weight"] = raw_w / max_w if max_w else raw_w

    # Compute (normalized) weighted degree and unweighted degree
    weighted_degree = {
        n: sum(G[n][nbr]["weight"] for nbr in G.neighbors(n)) for n in G.nodes()
    }
    degree_dict = dict(G.degree())

    return G, weighted_degree, degree_dict


def load_cp_labels(sim_dir: Path) -> Dict[object, int]:
    """Load core-periphery labels from enhanced or base membership CSV.

    Returns mapping user_id -> cp_label (0 core, 1 periphery). Keys are stored
    in a type-flexible way: original value is preserved; if the user_id is a
    numeric string, both int and str keys are added to maximize matching across
    datasets.
    """
    enhanced = sim_dir / "enhanced_core_periphery_membership.csv"
    base = sim_dir / "core_periphery_membership.csv"

    cp_path = enhanced if enhanced.exists() else base
    if not cp_path.exists():
        return {}

    df = pd.read_csv(cp_path)
    # Normalize column names defensively
    cols = {c.lower(): c for c in df.columns}
    uid_col = cols.get("user_id", "user_id")
    cp_col = cols.get("cp_label", "cp_label")

    labels: Dict[object, int] = {}
    for _, row in df.iterrows():
        uid = row[uid_col]
        try:
            cpv = int(row[cp_col])
        except Exception:
            continue
        # Preserve original key
        labels[uid] = cpv
        # If uid looks numeric, also store int form
        try:
            uid_int = int(uid)
        except Exception:
            uid_int = None
        if uid_int is not None:
            labels[uid_int] = cpv
    return labels


def compute_lcc(G: nx.Graph) -> nx.Graph:
    """Return a copy of the largest connected component subgraph."""
    if G.number_of_nodes() == 0:
        return G.copy()
    if nx.is_connected(G):
        return G.copy()
    largest_cc = max(nx.connected_components(G), key=len)
    return G.subgraph(largest_cc).copy()


def scale_sizes(values: Dict[int, float], nodelist, min_size=10, max_size=300) -> np.ndarray:
    """Min-max scale values to node sizes for the given node order."""
    arr = np.array([values.get(n, 0.0) for n in nodelist], dtype=float)
    vmin, vmax = float(np.min(arr)), float(np.max(arr))
    if vmax <= vmin + 1e-12:
        return np.full_like(arr, (min_size + max_size) / 2.0)
    norm = (arr - vmin) / (vmax - vmin)
    return min_size + norm * (max_size - min_size)


def draw_full_network_degree(
    G: nx.Graph,
    degree: Dict[int, int],
    out_path: Path,
    seed: int = 42,
):
    """Plot the entire network with nodes scaled by (unweighted) degree."""
    if G.number_of_nodes() == 0:
        return

    pos = nx.spring_layout(G, seed=seed, weight="weight")

    plt.figure(figsize=(18, 14))
    # Edges with some transparency to reduce clutter (more visible)
    edge_widths = [0.3 + 1.2 * G[u][v].get("weight", 0.0) for u, v in G.edges()]
    nx.draw_networkx_edges(
        G,
        pos,
        alpha=0.35,
        edge_color="#7f7f7f",
        width=edge_widths,
    )

    nodelist = list(G.nodes())
    sizes = scale_sizes(degree, nodelist, min_size=10, max_size=200)
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=nodelist,
        node_size=sizes,
        node_color="#377eb8",
        alpha=0.6,
        linewidths=0.0,
    )

    plt.title("Simulated Network — Full Network (nodes sized by degree)", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def draw_lcc_weighted_degree_colored(
    G_lcc: nx.Graph,
    weighted_degree: Dict[int, float],
    out_path: Path,
    seed: int = 42,
):
    """Largest component: uniform size, nodes colored by weighted degree."""
    if G_lcc.number_of_nodes() == 0:
        return

    pos = nx.spring_layout(G_lcc, seed=seed, weight="weight")
    vals = np.array([weighted_degree.get(n, 0.0) for n in G_lcc.nodes()], dtype=float)
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    if not (vmax > vmin):
        vmax = vmin + 1e-6
    cmap = cm.viridis

    plt.figure(figsize=(18, 14))
    edge_widths = [0.6 + 2.0 * G_lcc[u][v].get("weight", 0.0) for u, v in G_lcc.edges()]
    nx.draw_networkx_edges(
        G_lcc, pos, alpha=0.50, edge_color="#6f6f6f", width=edge_widths
    )
    nodes = nx.draw_networkx_nodes(
        G_lcc,
        pos,
        node_size=24,
        node_color=vals,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.0,
    )
    cbar = plt.colorbar(nodes, shrink=0.8)
    cbar.set_label("Weighted degree (normalized)")
    plt.title("Simulated Network — Largest Component: Weighted Degree (color)", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def draw_lcc_kcore_colored_weighted_sized(
    G_lcc: nx.Graph,
    weighted_degree: Dict[int, float],
    out_path: Path,
    seed: int = 42,
):
    """Largest component: nodes colored by k-core, sized by weighted degree."""
    if G_lcc.number_of_nodes() == 0:
        return

    pos = nx.spring_layout(G_lcc, seed=seed, weight="weight")
    kcore = nx.core_number(G_lcc)
    k_vals = np.array([kcore.get(n, 0) for n in G_lcc.nodes()], dtype=float)
    kmin, kmax = float(np.min(k_vals)), float(np.max(k_vals))
    if not (kmax > kmin):
        kmax = kmin + 1e-6
    cmap = cm.plasma

    nodelist = list(G_lcc.nodes())
    sizes = scale_sizes(weighted_degree, nodelist, min_size=14, max_size=260)

    plt.figure(figsize=(18, 14))
    edge_widths = [0.6 + 2.0 * G_lcc[u][v].get("weight", 0.0) for u, v in G_lcc.edges()]
    nx.draw_networkx_edges(
        G_lcc, pos, alpha=0.50, edge_color="#6f6f6f", width=edge_widths
    )
    nodes = nx.draw_networkx_nodes(
        G_lcc,
        pos,
        nodelist=nodelist,
        node_size=sizes,
        node_color=k_vals,
        cmap=cmap,
        vmin=kmin,
        vmax=kmax,
        linewidths=0.0,
    )
    cbar = plt.colorbar(nodes, shrink=0.8)
    cbar.set_label("k-core index")
    plt.title(
        "Simulated Network — Largest Component: k-Core (color), Weighted Degree (size)",
        fontsize=14,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def draw_lcc_core_periphery_colored_weighted_sized(
    G_lcc: nx.Graph,
    cp_labels: Dict[int, int],
    weighted_degree: Dict[int, float],
    out_path: Path,
    seed: int = 42,
):
    """Largest component: nodes colored by core-periphery, sized by weighted degree.

    cp_labels: mapping user_id -> label (0 core, 1 periphery). Any missing or
    unknown labels are treated as periphery, so the visualization only uses two
    classes (Core and Periphery).
    """
    if G_lcc.number_of_nodes() == 0:
        return

    pos = nx.spring_layout(G_lcc, seed=seed, weight="weight")

    # Map labels to colors (two classes only). Unknown/missing -> Periphery (1).
    color_map = {0: "#e41a1c", 1: "#377eb8"}
    nodelist = list(G_lcc.nodes())
    colors = []
    for n in nodelist:
        lab = int(cp_labels.get(n, 1))  # default to periphery
        lab = 0 if lab == 0 else 1      # coerce any non-core to periphery
        colors.append(color_map[lab])
    sizes = scale_sizes(weighted_degree, nodelist, min_size=14, max_size=260)

    plt.figure(figsize=(18, 14))
    edge_widths = [0.6 + 2.0 * G_lcc[u][v].get("weight", 0.0) for u, v in G_lcc.edges()]
    nx.draw_networkx_edges(
        G_lcc, pos, alpha=0.50, edge_color="#6f6f6f", width=edge_widths
    )
    nx.draw_networkx_nodes(
        G_lcc,
        pos,
        nodelist=nodelist,
        node_size=sizes,
        node_color=colors,
        linewidths=0.0,
        alpha=0.8,
    )

    # Legend handles
    from matplotlib.lines import Line2D

    legends = [
        Line2D([0], [0], marker="o", color="w", label="Core", markerfacecolor=color_map[0], markersize=8),
        Line2D([0], [0], marker="o", color="w", label="Periphery", markerfacecolor=color_map[1], markersize=8),
    ]
    plt.legend(handles=legends, loc="lower right")
    plt.title(
        "Simulated Network — Largest Component: Core–Periphery (color), Weighted Degree (size)",
        fontsize=14,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_simulation_plots(sim_dir: Path) -> None:
    """Generate the 4 additional network plots for a simulation directory.

    Expects `posts.csv` and core-periphery membership CSVs under `sim_dir`.
    """
    posts_csv = sim_dir / "posts.csv"
    if not posts_csv.exists():
        raise FileNotFoundError(f"posts.csv not found at {posts_csv}")

    G, weighted_degree, degree = build_graph_from_posts(posts_csv)
    G_lcc = compute_lcc(G)
    cp_labels = load_cp_labels(sim_dir)

    full_out = sim_dir / "additional_full_network_degree_scaled.png"
    lcc_wdeg_out = sim_dir / "additional_lcc_weighted_degree_colored.png"
    lcc_kcore_out = sim_dir / "additional_lcc_kcore_colored_weighted_size.png"
    lcc_cp_out = sim_dir / "additional_lcc_core_periphery_colored_weighted_size.png"

    draw_full_network_degree(G, degree, full_out)
    draw_lcc_weighted_degree_colored(G_lcc, weighted_degree, lcc_wdeg_out)
    draw_lcc_kcore_colored_weighted_sized(G_lcc, weighted_degree, lcc_kcore_out)
    draw_lcc_core_periphery_colored_weighted_sized(G_lcc, cp_labels, weighted_degree, lcc_cp_out)


def build_graph_from_voat_parquet(parquet_path: Path) -> Tuple[nx.Graph, Dict[int, float], Dict[int, int]]:
    """Build undirected interaction graph from a Voat sample parquet file.

    Expects columns: user_id, parent_user_id, parent_id.
    Edges connect user_id -> parent_user_id for rows with non-null parent_user_id.
    """
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    if "parent_user_id" not in df.columns or "user_id" not in df.columns:
        raise ValueError(
            f"Expected columns 'user_id' and 'parent_user_id' in {parquet_path}"
        )
    df = df.dropna(subset=["parent_user_id"])  # keep rows with valid parents

    # Aggregate interaction weights by user pair
    df_pairs = (
        df[["user_id", "parent_user_id"]]
        .astype({"user_id": str, "parent_user_id": str})
        .loc[lambda d: d["user_id"] != d["parent_user_id"]]
        .groupby(["user_id", "parent_user_id"])  # type: ignore[arg-type]
        .size()
        .reset_index(name="weight")
    )

    G = nx.from_pandas_edgelist(
        df_pairs, source="user_id", target="parent_user_id", edge_attr="weight"
    )
    # Normalize edge weights
    if G.number_of_edges() > 0:
        max_w = max(G[u][v]["weight"] for u, v in G.edges())
        for u, v in G.edges():
            raw_w = G[u][v]["weight"]
            G[u][v]["raw_weight"] = raw_w
            G[u][v]["weight"] = raw_w / max_w if max_w else raw_w

    weighted_degree = {n: sum(G[n][nbr]["weight"] for nbr in G.neighbors(n)) for n in G.nodes()}
    degree_dict = dict(G.degree())
    return G, weighted_degree, degree_dict


def generate_voat_plots(voat_dir: Path, parquet_name: str = "voat_sample_1.parquet") -> None:
    """Generate the 4 additional network plots for a Voat sample directory.

    Uses enhanced core-periphery membership if available.
    """
    parquet_path = voat_dir / parquet_name
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found at {parquet_path}")

    G, weighted_degree, degree = build_graph_from_voat_parquet(parquet_path)
    G_lcc = compute_lcc(G)
    cp_labels = load_cp_labels(voat_dir)

    full_out = voat_dir / "additional_full_network_degree_scaled.png"
    lcc_wdeg_out = voat_dir / "additional_lcc_weighted_degree_colored.png"
    lcc_kcore_out = voat_dir / "additional_lcc_kcore_colored_weighted_size.png"
    lcc_cp_out = voat_dir / "additional_lcc_core_periphery_colored_weighted_size.png"

    draw_full_network_degree(G, degree, full_out)
    draw_lcc_weighted_degree_colored(G_lcc, weighted_degree, lcc_wdeg_out)
    draw_lcc_kcore_colored_weighted_sized(G_lcc, weighted_degree, lcc_kcore_out)
    draw_lcc_core_periphery_colored_weighted_sized(G_lcc, cp_labels, weighted_degree, lcc_cp_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Additional network visualizations for Simulation and Voat sample"
    )
    parser.add_argument(
        "--sim-dir",
        type=str,
        default="simulation",
        help="Path to simulation results directory containing posts.csv",
    )
    parser.add_argument(
        "--voat-dir",
        type=str,
        default=None,
        help="Optional path to MADOC Voat sample directory (e.g., MADOC/voat-technology/sample_1)",
    )
    parser.add_argument(
        "--voat-parquet-name",
        type=str,
        default="voat_sample_1.parquet",
        help="Parquet filename inside --voat-dir (default: voat_sample_1.parquet)",
    )
    args = parser.parse_args()

    # Always run for simulation dir
    generate_simulation_plots(Path(args.sim_dir))

    # Optionally run for Voat sample
    if args.voat_dir:
        generate_voat_plots(Path(args.voat_dir), parquet_name=args.voat_parquet_name)
