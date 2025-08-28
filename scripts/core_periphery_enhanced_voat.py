"""
core-periphery-enhanced-voat.py - Enhanced Network Analysis with Multi-Sample Core-Periphery Detection for Voat Data
Builds a network representation of user interactions from Voat parquet data and performs comprehensive core-periphery detection
using multiple sampling strategies to find the best core-periphery partition.
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
import community as community_louvain
import matplotlib.cm as cm
from collections import defaultdict, Counter

# Add the parent directory to sys.path to enable imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import core-periphery modules
from core_periphery_sbm import core_periphery as cp
from core_periphery_sbm import model_fit as mf

def print_header(simulation_name):
    """Print script header with simulation metadata"""
    print(f"\n{'-'*80}")
    print(f"Enhanced Core-Periphery Analysis for Voat Data: {simulation_name}")
    print(f"Multi-Sample Analysis with Quality Assessment")
    print(f"{'-'*80}\n")

def load_posts_data(parquet_path):
    """Load posts data from parquet file"""
    print("[1/7] Loading posts data...")
    df = pd.read_parquet(parquet_path)
    df = df.dropna(subset=['parent_user_id'])
    print(f"  - Loaded {len(df)} posts with valid parent relationships")
    return df

def build_network(df):
    """Build an undirected graph where nodes are users and edges represent interactions"""
    print("[2/7] Building network...")
    
    # Filter to required columns and create edge list efficiently
    df_edges = df[['user_id', 'parent_user_id', 'parent_id']]
    
    # Group by user interactions and count weights
    df_grouped = df_edges.groupby(['user_id', 'parent_user_id']).size().reset_index(name='weight').sort_values(by='weight', ascending=False)
    
    print(f"  - Created {len(df_grouped)} unique user interaction pairs")
    
    # Create network from edge list
    G = nx.from_pandas_edgelist(
        df_grouped,
        source='user_id',
        target='parent_user_id',
        edge_attr='weight'
    )
    
    # Remove self-loops
    G.remove_edges_from(nx.selfloop_edges(G))
    
    print(f"  - Network has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Normalize edge weights to [0, 1]
    print("  - Normalizing edge weights...")
    max_weight = max([G[u][v]['weight'] for u, v in G.edges()]) if G.edges() else 1
    for u, v in G.edges():
        G[u][v]['raw_weight'] = G[u][v]['weight']  # Keep raw weight for reference
        G[u][v]['weight'] = G[u][v]['weight'] / max_weight
    
    return G

def calculate_network_statistics(G):
    """Calculate and return basic network statistics"""
    print("[3/7] Calculating network statistics...")
    
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
        if G.number_of_nodes() < 10000:
            stats['avg_shortest_path_length'] = nx.average_shortest_path_length(G, weight='weight')
            stats['diameter'] = nx.diameter(G)
    except (nx.NetworkXError, nx.NetworkXNoPath):
        stats['avg_shortest_path_length'] = "N/A (disconnected graph)"
        stats['diameter'] = "N/A (disconnected graph)"
    
    return stats, weighted_degree

def calculate_partition_quality(G, labels):
    """Calculate quality metrics for a core-periphery partition"""
    core_nodes = [n for n, label in labels.items() if label == 0]
    periphery_nodes = [n for n, label in labels.items() if label == 1]
    
    if len(core_nodes) == 0:
        return {
            'core_density': 0.0,
            'periphery_density': 0.0,
            'core_periphery_density': 0.0,
            'modularity': 0.0,
            'assortativity': 0.0,
            'core_avg_degree': 0.0,
            'periphery_avg_degree': 0.0,
            'valid_partition': False
        }
    
    # Create subgraphs
    G_core = G.subgraph(core_nodes)
    G_periphery = G.subgraph(periphery_nodes)
    
    # Calculate densities
    core_density = nx.density(G_core) if len(core_nodes) > 1 else 0.0
    periphery_density = nx.density(G_periphery) if len(periphery_nodes) > 1 else 0.0
    
    # Core-periphery connections
    cp_edges = sum(1 for u, v in G.edges() if 
                   (u in core_nodes and v in periphery_nodes) or 
                   (u in periphery_nodes and v in core_nodes))
    max_cp_edges = len(core_nodes) * len(periphery_nodes)
    core_periphery_density = cp_edges / max_cp_edges if max_cp_edges > 0 else 0.0
    
    # Modularity with respect to core-periphery partition
    communities = [labels[n] for n in G.nodes()]
    try:
        modularity = nx.algorithms.community.modularity(G, [core_nodes, periphery_nodes], weight='weight')
    except:
        modularity = 0.0
    
    # Assortativity
    try:
        assortativity = nx.attribute_assortativity_coefficient(G, 'cp_label')
    except:
        # Add cp_label attribute temporarily
        for n in G.nodes():
            G.nodes[n]['cp_label'] = labels.get(n, -1)
        try:
            assortativity = nx.attribute_assortativity_coefficient(G, 'cp_label')
        except:
            assortativity = 0.0
    
    # Average degrees
    core_avg_degree = np.mean([G.degree(n) for n in core_nodes]) if core_nodes else 0.0
    periphery_avg_degree = np.mean([G.degree(n) for n in periphery_nodes]) if periphery_nodes else 0.0
    
    return {
        'core_density': core_density,
        'periphery_density': periphery_density,
        'core_periphery_density': core_periphery_density,
        'modularity': modularity,
        'assortativity': assortativity,
        'core_avg_degree': core_avg_degree,
        'periphery_avg_degree': periphery_avg_degree,
        'valid_partition': True
    }

def run_enhanced_sbm(G, n_runs=5):
    """Run multiple SBM iterations and analyze all partitions on largest connected component"""
    print("[4/7] Running enhanced core-periphery detection...")
    
    # Focus on largest connected component for core-periphery detection
    if not nx.is_connected(G):
        print("  - Network is disconnected, focusing on largest connected component...")
        components = list(nx.connected_components(G))
        largest_component = max(components, key=len)
        G_main = G.subgraph(largest_component).copy()
        print(f"  - Largest component: {len(largest_component)} nodes ({len(largest_component)/len(G)*100:.1f}% of network)")
        print(f"  - Component has {G_main.number_of_edges()} edges, density: {nx.density(G_main):.4f}")
    else:
        G_main = G
        print("  - Network is connected, analyzing full network...")
    
    print(f"  - Performing {n_runs} independent runs for robustness...")
    
    all_results = []
    valid_partitions = []
    
    for run_idx in range(n_runs):
        print(f"  - Run {run_idx + 1}/{n_runs}...")
        
        # Initialize with different random seeds
        hubspoke = cp.HubSpokeCorePeriphery(n_gibbs=100, n_mcmc=10*len(G_main))
        
        # Run inference on the main component
        hubspoke.infer(G_main)
        
        # Get multiple samples from the posterior
        print(f"    - Extracting samples from posterior...")
        samples = []
        sample_ranges = [(0, 25), (25, 50), (50, 75), (75, 100)]
        
        for start, end in sample_ranges:
            try:
                labels = hubspoke.get_labels(last_n_samples=end-start, prob=False, return_dict=True)
                if labels:
                    samples.append(labels)
            except:
                continue
        
        # Also get the consensus labels
        try:
            consensus_labels = hubspoke.get_labels(last_n_samples=50, prob=False, return_dict=True)
            if consensus_labels:
                samples.append(consensus_labels)
        except:
            pass
        
        # Analyze each sample
        for sample_idx, labels in enumerate(samples):
            core_count = sum(1 for label in labels.values() if label == 0)
            
            if core_count > 0:  # Only analyze partitions with non-empty cores
                # Calculate MDL on the component we analyzed
                try:
                    labels_list = hubspoke.get_labels(last_n_samples=50, prob=False, return_dict=False)
                    mdl_score = mf.mdl_hubspoke(G_main, labels_list, n_samples=10000)
                except:
                    mdl_score = float('inf')
                
                # Map labels back to full network (nodes not in main component get label -1)
                full_labels = {}
                for node in G.nodes():
                    if node in labels:
                        full_labels[node] = labels[node]
                    else:
                        full_labels[node] = -1  # Not in analyzed component
                
                # Calculate quality metrics on the main component only
                quality = calculate_partition_quality(G_main, labels)
                
                result = {
                    'run': run_idx,
                    'sample': sample_idx,
                    'labels': labels,  # Component labels
                    'full_labels': full_labels,  # Full network labels
                    'core_size': core_count,
                    'periphery_size': len(labels) - core_count,
                    'component_size': len(G_main.nodes()),
                    'mdl_score': mdl_score,
                    'quality': quality
                }
                
                all_results.append(result)
                if quality['valid_partition']:
                    valid_partitions.append(result)
    
    print(f"  - Found {len(valid_partitions)} valid partitions with non-empty cores out of {len(all_results)} total partitions")
    
    # Return analysis info
    analysis_info = {
        'analyzed_component_size': len(G_main.nodes()),
        'analyzed_component_edges': G_main.number_of_edges(),
        'analyzed_component_density': nx.density(G_main),
        'is_full_network': nx.is_connected(G)
    }
    
    return all_results, valid_partitions, analysis_info

def analyze_partition_ensemble(valid_partitions):
    """Analyze the ensemble of valid partitions"""
    if not valid_partitions:
        return None, None
    
    print("[5/7] Analyzing partition ensemble...")
    
    # Sort by different criteria
    by_mdl = sorted(valid_partitions, key=lambda x: x['mdl_score'])
    by_core_density = sorted(valid_partitions, key=lambda x: x['quality']['core_density'], reverse=True)
    by_modularity = sorted(valid_partitions, key=lambda x: x['quality']['modularity'], reverse=True)
    
    # Create composite score
    for partition in valid_partitions:
        q = partition['quality']
        # Composite score balancing multiple factors
        partition['composite_score'] = (
            q['core_density'] * 0.3 +
            q['core_periphery_density'] * 0.3 +
            q['modularity'] * 0.2 +
            (1.0 / (1.0 + partition['mdl_score'] / 1000)) * 0.2  # Normalized MDL contribution
        )
    
    by_composite = sorted(valid_partitions, key=lambda x: x['composite_score'], reverse=True)
    
    # Select best partition (using composite score)
    best_partition = by_composite[0] if by_composite else None
    
    # Print analysis
    print(f"  - Best partition by MDL: Core={by_mdl[0]['core_size']}, MDL={by_mdl[0]['mdl_score']:.2f}")
    print(f"  - Best partition by core density: Core={by_core_density[0]['core_size']}, Density={by_core_density[0]['quality']['core_density']:.3f}")
    print(f"  - Best partition by composite score: Core={best_partition['core_size']}, Score={best_partition['composite_score']:.3f}")
    
    # Analyze core size distribution
    core_sizes = [p['core_size'] for p in valid_partitions]
    print(f"  - Core size range: {min(core_sizes)} - {max(core_sizes)} nodes")
    print(f"  - Average core size: {np.mean(core_sizes):.1f} nodes")
    
    return best_partition, {
        'by_mdl': by_mdl[0],
        'by_core_density': by_core_density[0],
        'by_modularity': by_modularity[0],
        'by_composite': best_partition,
        'core_size_stats': {
            'min': min(core_sizes),
            'max': max(core_sizes),
            'mean': np.mean(core_sizes),
            'std': np.std(core_sizes)
        }
    }

def save_enhanced_results(G, best_partition, analysis_summary, weighted_degree, output_dir):
    """Save enhanced analysis results"""
    print("[6/7] Saving enhanced analysis results...")
    
    if best_partition is None:
        print("  - No valid partitions found, saving empty results...")
        # Create empty results
        empty_df = pd.DataFrame({
            'user_id': list(G.nodes()),
            'degree': [G.degree(n) for n in G.nodes()],
            'weighted_degree': [weighted_degree.get(n, 0) for n in G.nodes()],
            'cp_label': [-1] * len(G.nodes()),
            'is_core': [False] * len(G.nodes())
        })
        empty_df.to_csv(os.path.join(output_dir, 'enhanced_core_periphery_membership.csv'), index=False)
        return empty_df, None
    
    # Save best partition
    labels = best_partition['labels']
    cp_df = pd.DataFrame({
        'user_id': list(G.nodes()),
        'degree': [G.degree(n) for n in G.nodes()],
        'weighted_degree': [weighted_degree.get(n, 0) for n in G.nodes()],
        'cp_label': [labels.get(n, -1) for n in G.nodes()],
        'is_core': [labels.get(n, -1) == 0 for n in G.nodes()]
    })
    
    cp_df.sort_values('weighted_degree', ascending=False, inplace=True)
    csv_path = os.path.join(output_dir, 'enhanced_core_periphery_membership.csv')
    cp_df.to_csv(csv_path, index=False)
    
    # Save detailed analysis
    analysis_path = os.path.join(output_dir, 'partition_analysis.txt')
    with open(analysis_path, 'w') as f:
        f.write("Enhanced Core-Periphery Analysis Results\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Best Partition Quality Metrics:\n")
        quality = best_partition['quality']
        f.write(f"- Core Density: {quality['core_density']:.4f}\n")
        f.write(f"- Periphery Density: {quality['periphery_density']:.4f}\n")
        f.write(f"- Core-Periphery Density: {quality['core_periphery_density']:.4f}\n")
        f.write(f"- Modularity: {quality['modularity']:.4f}\n")
        f.write(f"- Assortativity: {quality['assortativity']:.4f}\n")
        f.write(f"- Core Avg Degree: {quality['core_avg_degree']:.2f}\n")
        f.write(f"- Periphery Avg Degree: {quality['periphery_avg_degree']:.2f}\n")
        f.write(f"- MDL Score: {best_partition['mdl_score']:.2f}\n")
        f.write(f"- Composite Score: {best_partition['composite_score']:.4f}\n\n")
        
        f.write("Alternative Best Partitions:\n")
        f.write(f"- Best by MDL: Core={analysis_summary['by_mdl']['core_size']}, MDL={analysis_summary['by_mdl']['mdl_score']:.2f}\n")
        f.write(f"- Best by Core Density: Core={analysis_summary['by_core_density']['core_size']}, Density={analysis_summary['by_core_density']['quality']['core_density']:.3f}\n")
        f.write(f"- Best by Modularity: Core={analysis_summary['by_modularity']['core_size']}, Modularity={analysis_summary['by_modularity']['quality']['modularity']:.3f}\n\n")
        
        f.write("Core Size Statistics:\n")
        stats = analysis_summary['core_size_stats']
        f.write(f"- Range: {stats['min']} - {stats['max']} nodes\n")
        f.write(f"- Mean: {stats['mean']:.1f} ± {stats['std']:.1f} nodes\n")
    
    return cp_df, csv_path

def create_enhanced_visualizations(G, best_partition, weighted_degree, output_dir, simulation_name):
    """Create enhanced visualizations with quality information"""
    print("[7/7] Creating enhanced visualizations...")
    
    if best_partition is None:
        print("  - No valid partitions to visualize")
        return None
    
    labels = best_partition['labels']
    quality = best_partition['quality']
    
    # Create quality-focused visualization
    plt.figure(figsize=(20, 16))
    
    # Use largest component for visualization
    largest_cc = max(nx.connected_components(G), key=len)
    G_vis = G.subgraph(largest_cc).copy()
    
    if len(G_vis) > 500:
        vis_weighted_degree = {n: weighted_degree[n] for n in G_vis.nodes() if n in weighted_degree}
        sorted_nodes = sorted(vis_weighted_degree.items(), key=lambda x: x[1], reverse=True)
        top_nodes = [n for n, _ in sorted_nodes[:500]]
        G_vis = G_vis.subgraph(top_nodes).copy()
    
    pos = nx.spring_layout(G_vis, seed=42, weight='weight')
    
    # Separate core and periphery nodes
    core_nodes = [n for n in G_vis.nodes() if labels.get(n, -1) == 0]
    periphery_nodes = [n for n in G_vis.nodes() if labels.get(n, -1) == 1]
    
    # Scale node sizes
    max_weighted_degree = max(weighted_degree.values()) if weighted_degree else 1
    core_node_sizes = [20 + (weighted_degree.get(n, 0) / max_weighted_degree) * 280 for n in core_nodes]
    periphery_node_sizes = [5 + (weighted_degree.get(n, 0) / max_weighted_degree) * 95 for n in periphery_nodes]
    
    # Draw edges
    edge_weights = [G_vis[u][v]['weight'] * 3 for u, v in G_vis.edges()]
    nx.draw_networkx_edges(G_vis, pos, width=edge_weights, alpha=0.3, edge_color='lightgray')
    
    # Draw nodes
    nx.draw_networkx_nodes(G_vis, pos, nodelist=core_nodes, node_size=core_node_sizes, 
                          node_color='#e41a1c', alpha=0.8, label='Core')
    nx.draw_networkx_nodes(G_vis, pos, nodelist=periphery_nodes, node_size=periphery_node_sizes, 
                          node_color='#377eb8', alpha=0.6, label='Periphery')
    
    # Add labels for top nodes
    vis_weighted_degree = {n: weighted_degree.get(n, 0) for n in G_vis.nodes()}
    top_nodes = sorted(vis_weighted_degree.items(), key=lambda x: x[1], reverse=True)[:15]
    node_labels = {node: str(node) for node, _ in top_nodes}
    nx.draw_networkx_labels(G_vis, pos, labels=node_labels, font_size=8, font_color='black')
    
    # Add comprehensive title with quality metrics
    title = (f"Enhanced Core-Periphery Analysis: {simulation_name}\n"
             f"Core: {best_partition['core_size']} nodes, Periphery: {best_partition['periphery_size']} nodes\n"
             f"Core Density: {quality['core_density']:.3f}, C-P Density: {quality['core_periphery_density']:.3f}, "
             f"Modularity: {quality['modularity']:.3f}\n"
             f"MDL: {best_partition['mdl_score']:.1f}, Composite Score: {best_partition['composite_score']:.3f}")
    
    plt.title(title, fontsize=14, pad=20)
    plt.legend(scatterpoints=1, loc='lower right', fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    
    viz_path = os.path.join(output_dir, 'enhanced_core_periphery_visualization.png')
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return viz_path

def main(parquet_path, n_runs=5):
    """Enhanced network analysis workflow for Voat parquet data"""
    try:
        # Extract dataset name from the parquet file path
        parquet_path = Path(parquet_path)
        dataset_name = parquet_path.stem
        display_name = dataset_name.replace('_', ' ').title()
        output_dir = parquet_path.parent
        
        # Set up logging
        log_path = os.path.join(output_dir, 'enhanced_network_analysis.txt')
        original_stdout = sys.stdout
        
        with open(log_path, 'w') as f:
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
            print(f"Input data: {parquet_path}")
            print(f"Number of independent runs: {n_runs}")
            
            # Load and process data
            df = load_posts_data(parquet_path)
            G = build_network(df)
            stats, weighted_degree = calculate_network_statistics(G)
            
            # Run enhanced analysis
            all_results, valid_partitions, analysis_info = run_enhanced_sbm(G, n_runs)
            best_partition, analysis_summary = analyze_partition_ensemble(valid_partitions)
            
            # Save results
            cp_df, csv_path = save_enhanced_results(G, best_partition, analysis_summary, weighted_degree, output_dir)
            
            # Create visualizations
            viz_path = create_enhanced_visualizations(G, best_partition, weighted_degree, output_dir, display_name)
            
            # Print final results
            print("\nEnhanced Core-Periphery Analysis Complete!")
            print("\nNetwork Statistics:")
            for k, v in stats.items():
                print(f"- {k.replace('_', ' ').title()}: {v}")
            
            print(f"\nCore-Periphery Analysis Scope:")
            if analysis_info['is_full_network']:
                print(f"- Analyzed full connected network")
            else:
                print(f"- Analyzed largest connected component only")
                print(f"- Component size: {analysis_info['analyzed_component_size']} nodes ({analysis_info['analyzed_component_size']/stats['num_nodes']*100:.1f}% of network)")
                print(f"- Component edges: {analysis_info['analyzed_component_edges']}")
                print(f"- Component density: {analysis_info['analyzed_component_density']:.4f}")
            
            if best_partition:
                print(f"\nBest Core-Periphery Partition:")
                component_size = best_partition['component_size']
                print(f"- Core Size: {best_partition['core_size']} nodes ({(best_partition['core_size']/component_size)*100:.2f}% of analyzed component)")
                print(f"- Periphery Size: {best_partition['periphery_size']} nodes ({(best_partition['periphery_size']/component_size)*100:.2f}% of analyzed component)")
                print(f"- MDL Score: {best_partition['mdl_score']:.2f}")
                print(f"- Composite Quality Score: {best_partition['composite_score']:.4f}")
                
                quality = best_partition['quality']
                print(f"\nPartition Quality Metrics:")
                print(f"- Core Density: {quality['core_density']:.4f}")
                print(f"- Core-Periphery Density: {quality['core_periphery_density']:.4f}")
                print(f"- Modularity: {quality['modularity']:.4f}")
                print(f"- Core Avg Degree: {quality['core_avg_degree']:.2f}")
                
                # Show top core nodes
                core_nodes = [n for n, label in best_partition['labels'].items() if label == 0]
                core_weighted_degrees = {n: weighted_degree.get(n, 0) for n in core_nodes}
                top_core = sorted(core_weighted_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
                
                print(f"\nTop Core Nodes by Weighted Degree:")
                for node, wd in top_core:
                    print(f"- User {node}: {wd:.4f}")
            else:
                print(f"\nNo valid core-periphery partitions found across {n_runs} runs")
                print("This suggests the network may not have a clear core-periphery structure")
            
            print(f"\nResults saved to:")
            print(f"- Enhanced membership: {os.path.join(output_dir, 'enhanced_core_periphery_membership.csv')}")
            print(f"- Detailed analysis: {os.path.join(output_dir, 'partition_analysis.txt')}")
            if viz_path:
                print(f"- Enhanced visualization: {viz_path}")
            print(f"- Complete log: {log_path}")
            
            # Restore stdout
            sys.stdout = original_stdout
            
    except Exception as e:
        if 'sys.stdout' in locals() and sys.stdout != original_stdout:
            sys.stdout = original_stdout
        print(f"\nError during enhanced analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enhanced core-periphery analysis with multi-sample evaluation for Voat parquet data')
    parser.add_argument('parquet_path', type=str, help='Path to .parquet file containing Voat data')
    parser.add_argument('--runs', type=int, default=5, help='Number of independent runs (default: 5)')
    args = parser.parse_args()
    
    main(args.parquet_path, args.runs)