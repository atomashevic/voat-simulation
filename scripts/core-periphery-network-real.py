"""
core-periphery-network.py - Network Analysis Script with Core-Periphery Detection
Builds a network representation of user interactions from posts.csv data
and performs core-periphery detection using stochastic block model.
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

# Add the parent directory to sys.path to enable imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import core-periphery modules
from core_periphery_sbm import core_periphery as cp
from core_periphery_sbm import model_fit as mf

def print_header(simulation_name):
    """Print script header with simulation metadata"""
    print(f"\n{'-'*60}")
    print(f"Network Analysis with Core-Periphery Detection for Simulation: {simulation_name}")
    print(f"{'-'*60}\n")

def load_posts_data(parquet_path):
    """Load posts data from parquet file"""
    print("[1/5] Loading posts data...")
    df = pd.read_parquet(parquet_path)
    df = df.dropna(subset = ['parent_user_id'])
    print(f"  - Loaded {len(df)} posts")
    return df

def build_network(df):
    """Build an undirected graph where nodes are users and edges represent comments"""
    print("[2/5] Building network...")
    
    # Create undirected graph
    G = nx.Graph()
    
    # Add all unique user IDs as nodes
    user_ids = df['user_id'].unique()
    G.add_nodes_from(user_ids)
    print(f"  - Added {len(user_ids)} nodes (users)")
    
    # Filter to only comments (where parent_user_id != -1)
    #comment_df = df[df['parent_user_id'] != -1]
    df = df[['user_id','parent_user_id','parent_id']]
    
    df = df.groupby(['user_id','parent_user_id']).size().reset_index(name='weight').sort_values(by='weight',ascending = False)
    
    G = nx.from_pandas_edgelist(
    df,
    source='user_id',
    target='parent_user_id',
    edge_attr='weight'  # Set Edge Attribute to Weight Column
    )
    G.remove_edges_from(nx.selfloop_edges(G))
    
    edge_count = G.number_of_edges()
    
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
    print("[3/5] Calculating network statistics...")
    
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
    print(stats['num_components'],stats['density'])
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

def run_sbm(G):
    """Run stochastic block model core-periphery detection"""
    print("[4/5] Running core-periphery detection using stochastic block model...")
    
    # Initialize and run the hub-spoke core-periphery model
    print("  - Initializing HubSpokeCorePeriphery model...")
    hubspoke = cp.HubSpokeCorePeriphery(n_gibbs=100, n_mcmc=10*len(G))
    
    print("  - Inferring core-periphery structure...")
    hubspoke.infer(G)
    
    print("  - Extracting core-periphery labels...")
    labels = hubspoke.get_labels(last_n_samples=50, prob=False, return_dict=True)
    
    # Model selection
    print("  - Performing model selection...")
    inf_labels_hs = hubspoke.get_labels(last_n_samples=50, prob=False, return_dict=False)
    mdl_hubspoke = mf.mdl_hubspoke(G, inf_labels_hs, n_samples=100000)
    
    print(f"  - Minimum description length (MDL): {mdl_hubspoke:.4f}")
    
    # Count nodes in core (0) and periphery (1)
    core_count = sum(1 for label in labels.values() if label == 0)
    periphery_count = sum(1 for label in labels.values() if label == 1)
    print(f"  - Core size: {core_count} nodes")
    print(f"  - Periphery size: {periphery_count} nodes")
    
    return labels, mdl_hubspoke

def save_core_periphery_membership(G, cp_labels, weighted_degree, output_dir):
    """Save core-periphery membership to CSV file"""
    print("  - Saving core-periphery membership to CSV...")
    
    # Prepare DataFrame with core-periphery membership
    cp_df = pd.DataFrame({
        'user_id': list(G.nodes()),
        'degree': [G.degree(n) for n in G.nodes()],
        'weighted_degree': [weighted_degree.get(n, 0) for n in G.nodes()],
        'cp_label': [cp_labels.get(n, -1) for n in G.nodes()],
        'is_core': [cp_labels.get(n, -1) == 0 for n in G.nodes()]
    })
    
    # Sort by weighted degree (descending)
    cp_df.sort_values('weighted_degree', ascending=False, inplace=True)
    
    # Export to CSV
    csv_path = os.path.join(output_dir, 'core_periphery_membership.csv')
    cp_df.to_csv(csv_path, index=False)
    
    return cp_df, csv_path

def visualize_core_periphery(G, cp_labels, weighted_degree, output_dir, simulation_name):
    """Create visualization of the network with core-periphery structure"""
    print("  - Visualizing core-periphery structure...")
    
    # For visualization, we'll use the largest connected component
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
    
    # Set up figure with a decent size
    plt.figure(figsize=(20, 20))
    
    # Compute layout (this can take time for large networks)
    print("  - Computing network layout...")
    pos = nx.spring_layout(G_vis, seed=42, weight='weight')
    
    # Create node lists for core and periphery
    core_nodes = [n for n in G_vis.nodes() if cp_labels.get(n, -1) == 0]
    periphery_nodes = [n for n in G_vis.nodes() if cp_labels.get(n, -1) == 1]
    print(f"  - Core: {len(core_nodes)} nodes, Periphery: {len(periphery_nodes)} nodes")
    
    # Scale node sizes based on weighted degree
    max_weighted_degree = max(weighted_degree.values()) if weighted_degree else 1
    if max_weighted_degree==0:
        print ('OPA')
        print(sum(weighted_degree.values()))
        
    core_node_sizes = [20 + (weighted_degree.get(n, 0) / max_weighted_degree) * 280 for n in core_nodes]
    periphery_node_sizes = [5 + (weighted_degree.get(n, 0) / max_weighted_degree) * 95 for n in periphery_nodes]
    
    # Draw edges with alpha transparency
    edge_weights = [G_vis[u][v]['weight'] * 3 for u, v in G_vis.edges()]
    nx.draw_networkx_edges(G_vis, pos, 
                           width=edge_weights,
                           alpha=0.3,
                           edge_color='lightgray')
    
    # Draw core nodes
    nx.draw_networkx_nodes(G_vis, pos, 
                           nodelist=core_nodes,
                           node_size=core_node_sizes,
                           node_color='#e41a1c',  # Red
                           label='Core')
    
    # Draw periphery nodes
    nx.draw_networkx_nodes(G_vis, pos, 
                           nodelist=periphery_nodes,
                           node_size=periphery_node_sizes,
                           node_color='#377eb8',  # Blue
                           label='Periphery')
    
    # Add labels for top nodes by weighted degree
    vis_weighted_degree = {n: weighted_degree.get(n, 0) for n in G_vis.nodes()}
    top_nodes = sorted(vis_weighted_degree.items(), key=lambda x: x[1], reverse=True)[:20]
    labels = {node: str(node) for node, _ in top_nodes}
    nx.draw_networkx_labels(G_vis, pos, labels=labels, font_size=10, font_color='black')
    
    # Add title, legend and save
    plt.title(f"Core-Periphery Structure for {simulation_name}\n{G_vis.number_of_nodes()} nodes, {G_vis.number_of_edges()} edges", fontsize=16)
    plt.legend(scatterpoints=1, loc='lower right', fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    
    # Save the visualization
    plt.savefig(os.path.join(output_dir, 'core_periphery_visualization.png'), dpi=300)
    plt.close()
    
    return os.path.join(output_dir, 'core_periphery_visualization.png')

def visualize_network(G, weighted_degree, cp_labels, output_dir, simulation_name):
    """Create and save network visualizations"""
    print("[5/5] Creating network visualizations...")
    
    # Create core-periphery specific visualization
    cp_viz_path = visualize_core_periphery(G, cp_labels, weighted_degree, output_dir, simulation_name)
    
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
    
    # Create a visualization of the full network
    visualize_full_network(G, weighted_degree, cp_labels, output_dir, simulation_name)
    
    return os.path.join(output_dir, 'network_visualization.png')

def visualize_full_network(G, weighted_degree, cp_labels, output_dir, simulation_name):
    """Create a visualization of the full network with node size based on weighted degree
    and coloring based on core-periphery membership"""
    print("  - Creating full network visualization with core-periphery coloring...")
    
    # Set up figure
    plt.figure(figsize=(24, 24))
    
    # Create separate lists for core and periphery nodes
    core_nodes = [n for n in G.nodes() if cp_labels.get(n, -1) == 0]
    periphery_nodes = [n for n in G.nodes() if cp_labels.get(n, -1) == 1]
    print(f"  - Core: {len(core_nodes)} nodes, Periphery: {len(periphery_nodes)} nodes")
    
    # Compute k-core decomposition for layout purposes (not for coloring)
    print("  - Computing k-core decomposition for layout...")
    core_numbers = nx.core_number(G)
    max_core = max(core_numbers.values())
    print(f"  - Network has cores from 1 to {max_core}")
    
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
    
    # Draw edges with alpha transparency
    print("  - Drawing edges...")
    edge_weights = [G[u][v]['weight'] * 2 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, 
                          width=edge_weights,
                          alpha=0.2,
                          edge_color='gray')
    
    # Scale node sizes based on weighted degree
    max_weighted_degree = max(weighted_degree.values()) if weighted_degree else 1
    if max_weighted_degree==0:
        print ('OPA')
        print(sum(weighted_degree.values()))
    # Draw core nodes
    print("  - Drawing core nodes...")
    core_node_sizes = [10 + (weighted_degree.get(n, 0) / max_weighted_degree) * 290 for n in core_nodes]
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=core_nodes,
                          node_size=core_node_sizes,
                          node_color='#e41a1c',  # Red
                          alpha=0.7,
                          label='Core')
    
    # Draw periphery nodes
    print("  - Drawing periphery nodes...")
    periphery_node_sizes = [5 + (weighted_degree.get(n, 0) / max_weighted_degree) * 95 for n in periphery_nodes]
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=periphery_nodes,
                          node_size=periphery_node_sizes,
                          node_color='#377eb8',  # Blue
                          alpha=0.5,
                          label='Periphery')
    
    # Add title, legend and save
    plt.title(f"Full Network with Core-Periphery Structure for {simulation_name}\n{G.number_of_nodes()} nodes, {G.number_of_edges()} edges\nRed = Core, Blue = Periphery, Size by Weighted Degree", fontsize=16)
    plt.legend(scatterpoints=1, loc='lower right', fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    
    # Save the visualization
    plt.savefig(os.path.join(output_dir, 'core-periphery-full-network.png'), dpi=300)
    plt.close()
    
    # Also save a duplicate with the standard name for backward compatibility
    plt.figure(figsize=(24, 24))
    plt.text(0.5, 0.5, "See core-periphery-full-network.png", 
            horizontalalignment='center', verticalalignment='center', 
            fontsize=20, transform=plt.gca().transAxes)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'network_visualization_full.png'), dpi=100)
    plt.close()

def plot_degree_distribution(G, weighted_degree, cp_labels, output_dir, simulation_name):
    """Create plots of degree distribution with core-periphery information"""
    print("  - Creating degree distribution plots...")
    
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
    plt.loglog(degree_counts.index, degree_counts.values, 'o-', alpha=0.7, color='steelblue')
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
    
    # Plot 4: Degree distribution separated by core/periphery
    plt.figure(figsize=(12, 8))
    
    # Get degrees for core and periphery
    core_degrees = [G.degree(n) for n in G.nodes() if cp_labels.get(n, -1) == 0]
    periphery_degrees = [G.degree(n) for n in G.nodes() if cp_labels.get(n, -1) == 1]
    
    # Create bins
    max_degree = max(degrees) if degrees else 1
    bins = np.linspace(0, max_degree, 30)
    
    # Plot histograms
    plt.hist(core_degrees, bins=bins, alpha=0.6, color='#e41a1c', label='Core')
    plt.hist(periphery_degrees, bins=bins, alpha=0.6, color='#377eb8', label='Periphery')
    
    plt.title(f"Degree Distribution by Core-Periphery - {simulation_name}", fontsize=14)
    plt.xlabel("Degree", fontsize=12)
    plt.ylabel("Number of Users", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'degree_distribution_cp.png'), dpi=300)
    plt.close()
    
    return os.path.join(output_dir, 'degree_distribution.png')

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
        network_stats_path = os.path.join(output_dir, 'network_stats_cp.txt')
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
            
            # Run core-periphery detection
            cp_labels, mdl_value = run_sbm(G)
            
            # Save core-periphery membership
            cp_df, cp_csv_path = save_core_periphery_membership(G, cp_labels, weighted_degree, output_dir)
            
            # Create visualizations with core-periphery information
            viz_path = visualize_network(G, weighted_degree, cp_labels, output_dir, display_name)
            
            # Create degree distribution plots
            dist_path = plot_degree_distribution(G, weighted_degree, cp_labels, output_dir, display_name)
            
            # Print results
            print("\nNetwork Analysis with Core-Periphery Detection Complete!")
            print("\nNetwork Statistics:")
            for k, v in stats.items():
                print(f"- {k.replace('_', ' ').title()}: {v}")
            
            # Print core-periphery results
            print("\nCore-Periphery Detection Results:")
            print(f"- Minimum Description Length (MDL): {mdl_value:.4f}")
            core_count = sum(1 for label in cp_labels.values() if label == 0)
            periphery_count = sum(1 for label in cp_labels.values() if label == 1)
            print(f"- Core Size: {core_count} nodes ({(core_count/G.number_of_nodes())*100:.2f}% of network)")
            print(f"- Periphery Size: {periphery_count} nodes ({(periphery_count/G.number_of_nodes())*100:.2f}% of network)")
            
            # Print top core nodes by weighted degree
            core_nodes = [n for n, label in cp_labels.items() if label == 0]
            core_weighted_degrees = {n: weighted_degree.get(n, 0) for n in core_nodes}
            top_core_nodes = sorted(core_weighted_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
            
            print("\nTop 10 Core Nodes by Weighted Degree:")
            for node, wd in top_core_nodes:
                print(f"- User {node}: {wd:.4f}")
            
            print(f"\nCore-periphery membership exported to: {cp_csv_path}")
            print(f"Core-periphery visualization saved to: {os.path.join(output_dir, 'core_periphery_visualization.png')}")
            print(f"Network visualization saved to: {viz_path}")
            print(f"Full network visualization saved to: {os.path.join(output_dir, 'core-periphery-full-network.png')}")
            print(f"Degree distribution plots saved to: {dist_path} (and others)")
            print(f"Network statistics saved to: {network_stats_path}")
            
            # Restore original stdout
            sys.stdout = original_stdout
            
    except Exception as e:
        if 'sys.stdout' in locals() and sys.stdout != original_stdout:
            sys.stdout = original_stdout
        print(f"\nError during network analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Network analysis with core-periphery detection from Reddit simulation data')
    parser.add_argument('csv_path', type=str, help='Path to posts.csv file (output from sim-stats.py)')
    args = parser.parse_args()
    
    main(args.csv_path) 