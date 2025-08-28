"""
map-network.py - Network Analysis Script for Reddit Simulation
Builds a network representation of user interactions from posts.csv data
where nodes are users and edges represent comments between users.
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import community as community_louvain  # python-louvain package
import matplotlib.cm as cm

def print_header(simulation_name):
    """Print script header with simulation metadata"""
    print(f"\n{'-'*60}")
    print(f"Network Analysis for Simulation: {simulation_name}")
    print(f"{'-'*60}\n")

def load_posts_data(csv_path):
    """Load posts data from CSV file"""
    print("[1/6] Loading posts data...")
    df = pd.read_csv(csv_path)
    print(f"  - Loaded {len(df)} posts")
    return df

def build_network(df):
    """Build an undirected graph where nodes are users and edges represent comments"""
    print("[2/6] Building network...")
    
    # Create undirected graph
    G = nx.Graph()
    
    # Add all unique user IDs as nodes
    user_ids = df['user_id'].unique()
    G.add_nodes_from(user_ids)
    print(f"  - Added {len(user_ids)} nodes (users)")
    
    # Filter to only comments (where comment_to != -1)
    comment_df = df[df['comment_to'] != -1]
    
    # For each comment, create an edge between the commenter and the author of the parent post
    edge_count = 0
    for _, comment in comment_df.iterrows():
        commenter_id = comment['user_id']
        comment_to_id = comment['comment_to']
        
        # Find the user_id of the post being commented on
        if comment_to_id in df['id'].values:
            parent_post = df[df['id'] == comment_to_id]
            if not parent_post.empty:
                parent_author_id = parent_post.iloc[0]['user_id']
                
                # Skip self-loops (when a user comments on their own post)
                if commenter_id == parent_author_id:
                    continue
                
                # Add edge (commenter -- parent author)
                if G.has_edge(commenter_id, parent_author_id):
                    # Increment weight if edge already exists
                    G[commenter_id][parent_author_id]['weight'] += 1
                else:
                    # Create new edge with weight 1
                    G.add_edge(commenter_id, parent_author_id, weight=1)
                edge_count += 1
    
    print(f"  - Created {edge_count} edges (comments)")
    
    # Normalize edge weights to [0, 1]
    print("  - Normalizing edge weights...")
    max_weight = max([G[u][v]['weight'] for u, v in G.edges()]) if G.edges() else 1
    for u, v in G.edges():
        G[u][v]['weight'] = G[u][v]['weight'] / max_weight
        G[u][v]['raw_weight'] = G[u][v]['weight'] * max_weight  # Keep raw weight for reference
    
    return G

def calculate_network_statistics(G):
    """Calculate and return basic network statistics"""
    print("[3/6] Calculating network statistics...")
    
    # Calculate weighted degree for each node
    weighted_degree = {node: sum(G[node][neighbor]['weight'] for neighbor in G.neighbors(node)) 
                      for node in G.nodes()}
    
    stats = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'is_directed': nx.is_directed(G),
        'is_connected': nx.is_connected(G),
        'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes(),
        'avg_weighted_degree': sum(weighted_degree.values()) / G.number_of_nodes(),
        'avg_clustering': nx.average_clustering(G, weight='weight'),
        'num_components': nx.number_connected_components(G),
        'density': nx.density(G),
    }
    
    # Get largest component size
    components = list(nx.connected_components(G))
    largest_component = max(components, key=len)
    stats['largest_component_size'] = len(largest_component)
    stats['largest_component_ratio'] = len(largest_component) / G.number_of_nodes()
    
    try:
        # These can be computationally expensive for large networks
        if G.number_of_nodes() < 10000:  # Only calculate for reasonably sized networks
            stats['avg_shortest_path_length'] = nx.average_shortest_path_length(G, weight='weight')
            stats['diameter'] = nx.diameter(G)
    except (nx.NetworkXError, nx.NetworkXNoPath):
        # Network might be disconnected
        stats['avg_shortest_path_length'] = "N/A (disconnected graph)"
        stats['diameter'] = "N/A (disconnected graph)"
    
    return stats, weighted_degree

def calculate_centrality_measures(G, weighted_degree, output_dir):
    """Calculate various centrality measures and export to CSV"""
    print("[4/6] Calculating centrality measures...")
    
    # Calculate centrality measures (with progress indicators)
    print("  - Calculating degree centrality...")
    degree_centrality = nx.degree_centrality(G)
    
    print("  - Calculating weighted degree...")
    # Normalize weighted degree to [0,1] for comparison with other centrality measures
    max_weighted_degree = max(weighted_degree.values()) if weighted_degree else 1
    normalized_weighted_degree = {node: val/max_weighted_degree for node, val in weighted_degree.items()}
    
    # For large networks, betweenness and eigenvector centrality can be very time-consuming
    # So we'll only calculate these for the most connected nodes
    
    # Get largest connected component for other centrality measures
    largest_cc = max(nx.connected_components(G), key=len)
    G_sub = G.subgraph(largest_cc).copy()
    
    # Further limit to top nodes by degree if still too large
    if len(G_sub) > 1000:
        degrees = dict(G_sub.degree())
        sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        top_nodes = [n for n, _ in sorted_nodes[:1000]]
        G_sub = G_sub.subgraph(top_nodes).copy()
        print(f"  - Limited centrality calculations to top 1000 nodes by degree")
    
    print(f"  - Calculating betweenness centrality for subgraph with {len(G_sub)} nodes...")
    betweenness_centrality = nx.betweenness_centrality(G_sub, k=min(500, len(G_sub)), weight='weight')
    
    print("  - Calculating eigenvector centrality...")
    try:
        eigenvector_centrality = nx.eigenvector_centrality(G_sub, max_iter=1000, weight='weight')
    except nx.PowerIterationFailedConvergence:
        eigenvector_centrality = {node: 0 for node in G_sub.nodes()}
        print("    Warning: Eigenvector centrality failed to converge")
    
    print("  - Calculating closeness centrality...")
    try:
        closeness_centrality = nx.closeness_centrality(G_sub, distance='weight')
    except:
        closeness_centrality = {node: 0 for node in G_sub.nodes()}
        print("    Warning: Closeness centrality calculation failed")
    
    # Prepare DataFrame
    centrality_df = pd.DataFrame({
        'user_id': list(G.nodes()),
        'degree': [G.degree(n) for n in G.nodes()],
        'weighted_degree': [weighted_degree.get(n, 0) for n in G.nodes()],
        'degree_centrality': [degree_centrality.get(n, 0) for n in G.nodes()],
        'weighted_degree_centrality': [normalized_weighted_degree.get(n, 0) for n in G.nodes()],
        'betweenness_centrality': [betweenness_centrality.get(n, 0) for n in G.nodes()],
        'eigenvector_centrality': [eigenvector_centrality.get(n, 0) for n in G.nodes()],
        'closeness_centrality': [closeness_centrality.get(n, 0) for n in G.nodes()]
    })
    
    # Sort by weighted degree (descending)
    centrality_df.sort_values('weighted_degree', ascending=False, inplace=True)
    
    # Export to CSV
    csv_path = os.path.join(output_dir, 'centrality_measures.csv')
    centrality_df.to_csv(csv_path, index=False)
    
    return centrality_df, csv_path

def visualize_network(G, weighted_degree, output_dir, simulation_name):
    """Create and save network visualizations"""
    print("[5/6] Creating network visualizations...")
    
    # For visualization, we'll use the largest connected component
    # to make the graph more interpretable
    print("  - Extracting largest connected component for visualization...")
    largest_cc = max(nx.connected_components(G), key=len)
    G_vis = G.subgraph(largest_cc).copy()
    
    # If still too large, limit to top nodes by weighted degree
    if len(G_vis) > 500:
        # Get weighted degree for nodes in G_vis
        vis_weighted_degree = {n: weighted_degree[n] for n in G_vis.nodes() if n in weighted_degree}
        sorted_nodes = sorted(vis_weighted_degree.items(), key=lambda x: x[1], reverse=True)
        top_nodes = [n for n, _ in sorted_nodes[:500]]
        G_vis = G_vis.subgraph(top_nodes).copy()
        print(f"  - Limited visualization to top 500 nodes by weighted degree")
    
    print(f"  - Visualizing network with {G_vis.number_of_nodes()} nodes and {G_vis.number_of_edges()} edges")
    
    # Detect communities
    print("  - Detecting communities...")
    partition = community_louvain.best_partition(G_vis)
    
    # Set up figure with a decent size
    plt.figure(figsize=(20, 20))
    
    # Compute layout (this can take time for large networks)
    print("  - Computing network layout...")
    pos = nx.spring_layout(G_vis, seed=42, weight='weight')
    
    # Number of communities
    communities = set(partition.values())
    print(f"  - Detected {len(communities)} communities")
    
    # Draw nodes, colored by community
    for comm in communities:
        node_list = [node for node in partition.keys() if partition[node] == comm]
        nx.draw_networkx_nodes(G_vis, pos, 
                              nodelist=node_list,
                              node_size=40,
                              node_color=str(comm/len(communities)),
                              cmap=plt.cm.tab20)
    
    # Scale edge widths based on normalized weight
    edge_weights = [G_vis[u][v]['weight'] * 3 for u, v in G_vis.edges()]
    
    # Draw edges with alpha transparency
    nx.draw_networkx_edges(G_vis, pos, 
                          width=edge_weights,
                          alpha=0.3,
                          edge_color='lightgray')
    
    # Add labels for top nodes by weighted degree (limited number for readability)
    vis_weighted_degree = {n: weighted_degree[n] for n in G_vis.nodes() if n in weighted_degree}
    top_nodes = sorted(vis_weighted_degree.items(), key=lambda x: x[1], reverse=True)[:20]
    labels = {node: str(node) for node, _ in top_nodes}
    nx.draw_networkx_labels(G_vis, pos, labels=labels, font_size=10, font_color='black')
    
    # Add title and save
    plt.title(f"Network for {simulation_name} - Largest Component\n{G_vis.number_of_nodes()} nodes, {G_vis.number_of_edges()} edges", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    
    # Save the visualization
    plt.savefig(os.path.join(output_dir, 'network_visualization.png'), dpi=300)
    plt.close()
    
    # Create a visualization with all nodes
    visualize_full_network(G, weighted_degree, output_dir, simulation_name)
    
    # Create a simplified visualization for very large networks
    if G.number_of_nodes() > 1000:
        visualize_simplified_network(G, weighted_degree, output_dir, simulation_name)
    
    return os.path.join(output_dir, 'network_visualization.png')

def visualize_full_network(G, weighted_degree, output_dir, simulation_name):
    """Create a visualization of the full network with node size based on weighted degree"""
    print("  - Creating full network visualization...")
    
    # Set up figure
    plt.figure(figsize=(24, 24))
    
    # Compute k-core decomposition
    print("  - Computing k-core decomposition...")
    core_numbers = nx.core_number(G)
    max_core = max(core_numbers.values())
    print(f"  - Network has cores from 1 to {max_core}")
    
    # Create a colormap for cores
    cmap = plt.cm.viridis
    
    # Compute layout for the largest connected component first with increased repulsion
    largest_cc = max(nx.connected_components(G), key=len)
    G_largest = G.subgraph(largest_cc).copy()
    
    # Use spring layout with weight parameter and increased repulsion (k parameter)
    # Higher k means more repulsion between nodes
    print("  - Computing network layout for full visualization with increased repulsion...")
    k_value = 1.5 / np.sqrt(len(G_largest))  # Default is 1/sqrt(n)
    pos = nx.spring_layout(G_largest, seed=42, weight='weight', k=k_value, iterations=100)
    
    # Add positions for disconnected components
    for component in nx.connected_components(G):
        if component != largest_cc:
            # Position smaller components around the periphery
            component_pos = nx.spring_layout(G.subgraph(component), seed=42, weight='weight')
            # Adjust positions to place around the main component
            offset_x = np.random.uniform(-2, 2)
            offset_y = np.random.uniform(-2, 2)
            for node in component:
                if node in component_pos:
                    pos[node] = (component_pos[node][0] + offset_x, component_pos[node][1] + offset_y)
    
    # Draw edges with colors based on the minimum core number of their endpoints
    print("  - Drawing edges with colors based on k-core...")
    
    # Group edges by core number for batch drawing
    edges_by_core = {}
    for u, v in G.edges():
        min_core = min(core_numbers[u], core_numbers[v])
        if min_core not in edges_by_core:
            edges_by_core[min_core] = []
        edges_by_core[min_core].append((u, v))
    
    # Draw edges in order of increasing core number (lower cores first, higher cores on top)
    for core in sorted(edges_by_core.keys()):
        # Normalize core number to [0, 1] for color mapping
        color = cmap(core / max_core)
        
        # Get edge weights for this core
        edge_weights = [G[u][v]['weight'] * (1 + core/max_core) for u, v in edges_by_core[core]]
        
        # Draw edges with alpha transparency
        nx.draw_networkx_edges(G, pos, 
                              edgelist=edges_by_core[core],
                              width=edge_weights,
                              alpha=0.2 + 0.4 * (core / max_core),  # Higher cores are more opaque
                              edge_color=[color] * len(edges_by_core[core]))
    
    # Draw nodes with size proportional to weighted degree and color by core number
    print("  - Drawing nodes colored by k-core...")
    
    # Group nodes by core number for batch drawing
    nodes_by_core = {}
    for node, core in core_numbers.items():
        if core not in nodes_by_core:
            nodes_by_core[core] = []
        nodes_by_core[core].append(node)
    
    # Scale node sizes based on weighted degree (min size 5, max size 300)
    max_weighted_degree = max(weighted_degree.values()) if weighted_degree else 1
    
    # Draw nodes in order of increasing core number (higher cores on top)
    for core in sorted(nodes_by_core.keys()):
        # Get node sizes for this core
        node_sizes = [5 + (weighted_degree.get(n, 0) / max_weighted_degree) * 295 for n in nodes_by_core[core]]
        
        # Normalize core number to [0, 1] for color mapping
        color = cmap(core / max_core)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                              nodelist=nodes_by_core[core],
                              node_size=node_sizes,
                              node_color=[color] * len(nodes_by_core[core]),
                              alpha=0.6 + 0.4 * (core / max_core))  # Higher cores are more opaque
    
    # Add a colorbar to show the core number mapping
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, max_core))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), orientation='vertical', pad=0.01, fraction=0.02)
    cbar.set_label('K-Core Number', fontsize=12)
    
    # Add title and save
    plt.title(f"Full Network for {simulation_name}\n{G.number_of_nodes()} nodes, {G.number_of_edges()} edges\nColored by k-core decomposition, node size by weighted degree", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    
    # Save the visualization
    plt.savefig(os.path.join(output_dir, 'network_visualization_full.png'), dpi=300)
    plt.close()
    
    # Create a visualization focusing only on the highest cores
    if max_core > 5:
        visualize_high_cores(G, weighted_degree, core_numbers, max_core, pos, output_dir, simulation_name)

def visualize_high_cores(G, weighted_degree, core_numbers, max_core, pos, output_dir, simulation_name):
    """Create a visualization focusing only on the highest k-cores"""
    print("  - Creating high k-cores visualization...")
    
    # Set a threshold for "high" cores (e.g., top 30% of cores)
    high_core_threshold = max(1, int(max_core * 0.7))
    print(f"  - Focusing on cores {high_core_threshold} to {max_core}")
    
    # Filter nodes by core number
    high_core_nodes = [node for node, core in core_numbers.items() if core >= high_core_threshold]
    G_high_cores = G.subgraph(high_core_nodes).copy()
    
    # Set up figure
    plt.figure(figsize=(20, 20))
    
    # Use the same positions as the full visualization for consistency
    high_core_pos = {node: pos[node] for node in G_high_cores.nodes() if node in pos}
    
    # Create a colormap for cores
    cmap = plt.cm.viridis
    
    # Draw edges with colors based on the minimum core number of their endpoints
    edges_by_core = {}
    for u, v in G_high_cores.edges():
        min_core = min(core_numbers[u], core_numbers[v])
        if min_core not in edges_by_core:
            edges_by_core[min_core] = []
        edges_by_core[min_core].append((u, v))
    
    # Draw edges in order of increasing core number
    for core in sorted(edges_by_core.keys()):
        # Normalize core number to [0, 1] for color mapping
        color = cmap((core - high_core_threshold + 1) / (max_core - high_core_threshold + 1))
        
        # Get edge weights for this core
        edge_weights = [G_high_cores[u][v]['weight'] * 2 for u, v in edges_by_core[core]]
        
        # Draw edges with alpha transparency
        nx.draw_networkx_edges(G_high_cores, high_core_pos, 
                              edgelist=edges_by_core[core],
                              width=edge_weights,
                              alpha=0.3 + 0.5 * ((core - high_core_threshold) / (max_core - high_core_threshold)),
                              edge_color=[color] * len(edges_by_core[core]))
    
    # Draw nodes with size proportional to weighted degree and color by core number
    nodes_by_core = {}
    for node in G_high_cores.nodes():
        core = core_numbers[node]
        if core not in nodes_by_core:
            nodes_by_core[core] = []
        nodes_by_core[core].append(node)
    
    # Scale node sizes based on weighted degree
    max_weighted_degree = max(weighted_degree.values()) if weighted_degree else 1
    
    # Draw nodes in order of increasing core number
    for core in sorted(nodes_by_core.keys()):
        # Get node sizes for this core
        node_sizes = [10 + (weighted_degree.get(n, 0) / max_weighted_degree) * 290 for n in nodes_by_core[core]]
        
        # Normalize core number to [0, 1] for color mapping
        color = cmap((core - high_core_threshold + 1) / (max_core - high_core_threshold + 1))
        
        # Draw nodes
        nx.draw_networkx_nodes(G_high_cores, high_core_pos, 
                              nodelist=nodes_by_core[core],
                              node_size=node_sizes,
                              node_color=[color] * len(nodes_by_core[core]),
                              alpha=0.7)
    
    # Add labels for top nodes by weighted degree
    #high_core_weighted_degree = {n: weighted_degree[n] for n in G_high_cores.nodes() if n in weighted_degree}
    #top_nodes = sorted(high_core_weighted_degree.items(), key=lambda x: x[1], reverse=True)[:40]
    #labels = {node: str(node) for node, _ in top_nodes}
    #inx.draw_networkx_labels(G_high_cores, high_core_pos, labels=labels, font_size=8, font_color='black', font_weight='bold')
    
    # Add a colorbar to show the core number mapping
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(high_core_threshold, max_core))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), orientation='vertical', pad=0.01, fraction=0.02)
    cbar.set_label(f'K-Core Number ({high_core_threshold}-{max_core})', fontsize=12)
    
    # Add title and save
    #plt.title(f"High K-Cores Network for {simulation_name}\n{G_high_cores.number_of_nodes()} nodes, {G_high_cores.number_of_edges()} edges\nShowing cores {high_core_threshold}-{max_core}", fontsize=16)
    plt.title("")
    plt.axis('off')
    plt.tight_layout()
    
    # Save the visualization
    plt.savefig(os.path.join(output_dir, 'network_visualization_high_cores.png'), dpi=300)
    plt.close()

def visualize_simplified_network(G, weighted_degree, output_dir, simulation_name):
    """Create a simplified visualization for very large networks"""
    print("  - Creating simplified network visualization...")
    
    # Take top 100 nodes by weighted degree
    sorted_nodes = sorted(weighted_degree.items(), key=lambda x: x[1], reverse=True)
    top_nodes = [n for n, _ in sorted_nodes[:100]]
    G_simple = G.subgraph(top_nodes).copy()
    
    # Set up figure
    plt.figure(figsize=(16, 16))
    
    # Compute layout
    pos = nx.spring_layout(G_simple, seed=42, weight='weight')
    
    # Draw nodes with size proportional to weighted degree
    node_size = [5 + weighted_degree[n] * 50 for n in G_simple.nodes()]
    nx.draw_networkx_nodes(G_simple, pos, 
                          node_size=node_size,
                          node_color='skyblue',
                          alpha=0.8)
    
    # Draw edges with width proportional to weight
    edge_weights = [G_simple[u][v]['weight'] * 3 for u, v in G_simple.edges()]
    nx.draw_networkx_edges(G_simple, pos, 
                          width=edge_weights,
                          alpha=0.4,
                          edge_color='gray')
    
    # Add labels for nodes
    labels = {node: str(node) for node in G_simple.nodes()}
    nx.draw_networkx_labels(G_simple, pos, labels=labels, font_size=8)
    
    # Add title and save
    plt.title(f"Top 100 Users by Weighted Degree - {simulation_name}", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    
    # Save the visualization
    plt.savefig(os.path.join(output_dir, 'network_visualization_top100.png'), dpi=300)
    plt.close()

def plot_degree_distribution(G, weighted_degree, output_dir, simulation_name):
    """Create plots of degree distribution"""
    print("[6/6] Creating degree distribution plots...")
    
    # Get degree values
    degrees = [d for _, d in G.degree()]
    weighted_degrees = list(weighted_degree.values())
    
    # Plot 1: Degree distribution
    plt.figure(figsize=(12, 8))
    degree_counts = pd.Series(degrees).value_counts().sort_index()
    plt.bar(degree_counts.index, degree_counts.values, alpha=0.7, color='steelblue')
    plt.title(f"Degree Distribution - {simulation_name}\nNumber of users with X connections", fontsize=14)
    plt.xlabel("Degree (Number of Connections)", fontsize=12)
    plt.ylabel("Number of Users", fontsize=12)
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'degree_distribution.png'), dpi=300)
    plt.close()
    
    # Plot 2: Log-Log degree distribution
    plt.figure(figsize=(12, 8))
    plt.loglog(degree_counts.index, degree_counts.values, 'o', alpha=0.7, color='red')
    plt.title(f"Log-Log Degree Distribution - {simulation_name}", fontsize=14)
    plt.xlabel("Degree (Log Scale)", fontsize=12)
    plt.ylabel("Number of Users (Log Scale)", fontsize=12)
    plt.grid(alpha=0.3, which='both')
    plt.savefig(os.path.join(output_dir, 'degree_distribution_loglog.png'), dpi=300)
    plt.close()
    
    # Plot 3: Weighted degree distribution
    plt.figure(figsize=(12, 8))
    # Create bins for weighted degree
    bins = np.linspace(0, max(weighted_degrees), 30)
    plt.hist(weighted_degrees, bins=bins, alpha=0.7, color='forestgreen')
    plt.title(f"Weighted Degree Distribution - {simulation_name}", fontsize=14)
    plt.xlabel("Weighted Degree", fontsize=12)
    plt.ylabel("Number of Users", fontsize=12)
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'weighted_degree_distribution.png'), dpi=300)
    plt.close()
    
    # Plot 4: Cumulative degree distribution
    plt.figure(figsize=(12, 8))
    sorted_degrees = sorted(degrees)
    y = np.arange(1, len(sorted_degrees) + 1) / len(sorted_degrees)
    plt.plot(sorted(degrees), y, marker='.', linestyle='none', alpha=0.5, color='purple')
    plt.title(f"Cumulative Degree Distribution - {simulation_name}", fontsize=14)
    plt.xlabel("Degree", fontsize=12)
    plt.ylabel("Cumulative Proportion of Users", fontsize=12)
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'degree_cumulative_distribution.png'), dpi=300)
    plt.close()
    
    return os.path.join(output_dir, 'degree_distribution.png')

def create_degree_and_kcore_panel(G, weighted_degree, output_dir, simulation_name):
    """Create a two-panel image: log-log degree distribution (left) and high k-core network (right)"""
    print("  - Creating two-panel degree distribution and k-core visualization...")
    
    # Set up figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Left panel: Log-log degree distribution
    degrees = [d for _, d in G.degree()]
    degree_counts = pd.Series(degrees).value_counts().sort_index()
    
    ax1.loglog(degree_counts.index, degree_counts.values, 'o', alpha=0.8, color='purple', markersize=6)
    ax1.set_title("Log-Log Degree Distribution", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Degree (Log Scale)", fontsize=12)
    ax1.set_ylabel("Number of Users (Log Scale)", fontsize=12)
    ax1.grid(alpha=0.3, which='both')
    
    # Right panel: High k-core network
    core_numbers = nx.core_number(G)
    max_core = max(core_numbers.values()) if core_numbers else 1
    
    # Extract high k-core subgraph (top 30% of cores, minimum k=2)
    min_core_threshold = max(2, int(0.7 * max_core))
    high_core_nodes = [node for node, core in core_numbers.items() if core >= min_core_threshold]
    
    if high_core_nodes:
        G_core = G.subgraph(high_core_nodes).copy()
        
        # Limit to largest connected component if too large
        if G_core.number_of_nodes() > 300:
            largest_component = max(nx.connected_components(G_core), key=len)
            G_core = G_core.subgraph(largest_component).copy()
        
        # Compute layout with increased repulsion for better edge visibility
        k_value = 1.5 / np.sqrt(len(G_core))
        pos = nx.spring_layout(G_core, seed=42, weight='weight', k=k_value, iterations=100)
        
        # Create a colormap for cores
        cmap = plt.cm.viridis
        
        # Draw edges with colors based on the minimum core number of their endpoints
        edges_by_core = {}
        for u, v in G_core.edges():
            min_core = min(core_numbers[u], core_numbers[v])
            if min_core not in edges_by_core:
                edges_by_core[min_core] = []
            edges_by_core[min_core].append((u, v))
        
        # Draw edges in order of increasing core number
        for core in sorted(edges_by_core.keys()):
            # Normalize core number to [0, 1] for color mapping (match high k-cores visualization)
            color = cmap((core - min_core_threshold) / (max_core - min_core_threshold))
            
            # Get edge weights for this core
            edge_weights = [G_core[u][v]['weight'] * 2 for u, v in edges_by_core[core]]
            
            # Draw edges with alpha transparency
            nx.draw_networkx_edges(G_core, pos, 
                                  edgelist=edges_by_core[core],
                                  width=edge_weights,
                                  alpha=0.3 + 0.5 * ((core - min_core_threshold) / (max_core - min_core_threshold)),
                                  edge_color=[color] * len(edges_by_core[core]),
                                  ax=ax2)
        
        # Draw nodes with size proportional to weighted degree and color by core number
        nodes_by_core = {}
        for node in G_core.nodes():
            core = core_numbers[node]
            if core not in nodes_by_core:
                nodes_by_core[core] = []
            nodes_by_core[core].append(node)
        
        # Scale node sizes based on weighted degree
        max_weighted_degree = max(weighted_degree.values()) if weighted_degree else 1
        
        # Draw nodes in order of increasing core number
        for core in sorted(nodes_by_core.keys()):
            # Get node sizes for this core
            node_sizes = [10 + (weighted_degree.get(n, 0) / max_weighted_degree) * 180 for n in nodes_by_core[core]]
            
            # Normalize core number to [0, 1] for color mapping (match high k-cores visualization)
            color = cmap((core - min_core_threshold) / (max_core - min_core_threshold))
            
            # Draw nodes
            nx.draw_networkx_nodes(G_core, pos, 
                                  nodelist=nodes_by_core[core],
                                  node_size=node_sizes,
                                  node_color=[color] * len(nodes_by_core[core]),
                                  alpha=0.7,
                                  ax=ax2)
        
        # Add colorbar for k-core values (consistent with other network plots)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min_core_threshold, max_core))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax2, orientation='vertical', pad=0.01, fraction=0.02)
        cbar.set_label('K-Core Number', fontsize=12)
        
        ax2.set_title(f"Top K-Core Network (k≥{min_core_threshold})\n"
                     f"{G_core.number_of_nodes()} nodes, {G_core.number_of_edges()} edges", 
                     fontsize=14, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, "No high k-core nodes found\n(all nodes have k-core < 2)", 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title("Top K-Core Network", fontsize=14, fontweight='bold')
    
    ax2.set_aspect('equal')
    ax2.axis('off')
    
    # Adjust layout and save
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'degree_kcore_panel.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  - Two-panel visualization saved to: {output_path}")
    return output_path

def main(csv_path):
    """Main network analysis workflow"""
    try:
        # Extract simulation name from the csv file path
        csv_path = Path(csv_path)
        sim_name = csv_path.parent.name
        
        # Clean up simulation name for display
        display_name = sim_name.replace('-', ' ').title()
        
        # Set up output directory
        output_dir = csv_path.parent
        
        # Redirect stdout to both console and file
        network_stats_path = os.path.join(output_dir, 'network_stats.txt')
        original_stdout = sys.stdout
        with open(network_stats_path, 'w') as f:
            class TeeOutput:
                def write(self, text):
                    original_stdout.write(text)
                    f.write(text)
                def flush(self):
                    original_stdout.flush()
                    f.flush()
            
            sys.stdout = TeeOutput()
            
            # Print header
            print_header(display_name)
            print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Input data: {csv_path}")
            
            # Load data
            df = load_posts_data(csv_path)
            
            # Build network
            G = build_network(df)
            
            # Calculate network statistics
            stats, weighted_degree = calculate_network_statistics(G)
            
            # Calculate and export centrality measures
            centrality_df, centrality_csv_path = calculate_centrality_measures(G, weighted_degree, output_dir)
            
            # Create visualizations
            viz_path = visualize_network(G, weighted_degree, output_dir, display_name)
            
            # Create degree distribution plots
            dist_path = plot_degree_distribution(G, weighted_degree, output_dir, display_name)
            
            # Create two-panel image of degree distribution and high k-core network
            panel_path = create_degree_and_kcore_panel(G, weighted_degree, output_dir, display_name)
            
            # Print results
            print("\nNetwork Analysis Complete!")
            print("\nNetwork Statistics:")
            for k, v in stats.items():
                print(f"- {k.replace('_', ' ').title()}: {v}")
            
            # Print top users by different centrality measures
            print("\nTop 10 Users by Weighted Degree:")
            for idx, row in centrality_df.nlargest(10, 'weighted_degree').iterrows():
                print(f"- User {row['user_id']}: {row['weighted_degree']:.4f}")
            
            print("\nTop 10 Users by Degree Centrality:")
            for idx, row in centrality_df.nlargest(10, 'degree_centrality').iterrows():
                print(f"- User {row['user_id']}: {row['degree_centrality']:.4f}")
            
            print("\nTop 10 Users by Betweenness Centrality:")
            for idx, row in centrality_df.nlargest(10, 'betweenness_centrality').iterrows():
                print(f"- User {row['user_id']}: {row['betweenness_centrality']:.4f}")
                
            print(f"\nCentrality measures exported to: {centrality_csv_path}")
            print(f"Network visualization saved to: {viz_path}")
            print(f"Degree distribution plots saved to: {dist_path} (and others)")
            print(f"Network statistics saved to: {network_stats_path}")
            print(f"Two-panel visualization saved to: {panel_path}")
            
            # Restore original stdout
            sys.stdout = original_stdout
            
    except Exception as e:
        if 'sys.stdout' in locals() and sys.stdout != original_stdout:
            sys.stdout = original_stdout
        print(f"\nError during network analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Map network from Reddit simulation data')
    parser.add_argument('csv_path', type=str, help='Path to posts.csv file (output from sim-stats.py)')
    args = parser.parse_args()
    
    main(args.csv_path)
