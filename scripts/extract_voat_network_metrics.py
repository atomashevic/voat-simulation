#!/usr/bin/env python3
"""
Extract network metrics with confidence intervals from MADOC Voat samples.
Computes: density, core density, periphery density, avg degree, avg weighted degree, 
clustering, LCC %, core % in comparison to LCC, core-periphery density
"""

import os
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_network_metrics(file_path: str) -> Dict:
    """Extract network metrics from enhanced_network_analysis.txt file."""
    metrics = {}
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Extract sample ID from path
        sample_match = re.search(r'sample_(\d+)', file_path)
        metrics['sample_id'] = int(sample_match.group(1)) if sample_match else None
        
        # Network Statistics
        metrics['num_nodes'] = extract_value(content, r'Num Nodes:\s*(\d+)')
        metrics['num_edges'] = extract_value(content, r'Num Edges:\s*(\d+)')
        metrics['avg_degree'] = extract_value(content, r'Avg Degree:\s*([\d.]+)')
        metrics['avg_weighted_degree'] = extract_value(content, r'Avg Weighted Degree:\s*([\d.]+)')
        metrics['avg_clustering'] = extract_value(content, r'Avg Clustering:\s*([\d.]+)')
        metrics['density'] = extract_value(content, r'Density:\s*([\d.]+)')
        metrics['largest_component_size'] = extract_value(content, r'Largest Component Size:\s*(\d+)')
        metrics['largest_component_ratio'] = extract_value(content, r'Largest Component Ratio:\s*([\d.]+)')
        
        # Component-specific metrics
        metrics['component_size'] = extract_value(content, r'Component size:\s*(\d+) nodes')
        metrics['component_density'] = extract_value(content, r'Component density:\s*([\d.]+)')
        
        # Core-Periphery metrics
        metrics['core_size'] = extract_value(content, r'Core Size:\s*(\d+) nodes')
        metrics['periphery_size'] = extract_value(content, r'Periphery Size:\s*(\d+) nodes')
        metrics['core_density'] = extract_value(content, r'Core Density:\s*([\d.]+)')
        metrics['core_periphery_density'] = extract_value(content, r'Core-Periphery Density:\s*([\d.]+)')
        metrics['core_avg_degree'] = extract_value(content, r'Core Avg Degree:\s*([\d.]+)')
        
        # Calculate derived metrics
        if metrics['num_nodes'] and metrics['largest_component_size']:
            metrics['lcc_percentage'] = (metrics['largest_component_size'] / metrics['num_nodes']) * 100
        
        if metrics['component_size'] and metrics['core_size']:
            metrics['core_percentage_of_lcc'] = (metrics['core_size'] / metrics['component_size']) * 100
            
        # Calculate periphery density (approximation)
        if metrics['core_size'] and metrics['periphery_size'] and metrics['component_size']:
            # Approximate periphery density based on component and core densities
            core_edges = metrics['core_density'] * (metrics['core_size'] * (metrics['core_size'] - 1) / 2)
            total_component_edges = metrics['component_density'] * (metrics['component_size'] * (metrics['component_size'] - 1) / 2)
            cp_edges = metrics['core_periphery_density'] * (metrics['core_size'] * metrics['periphery_size'])
            
            periphery_edges = total_component_edges - core_edges - cp_edges
            if metrics['periphery_size'] > 1:
                metrics['periphery_density'] = periphery_edges / (metrics['periphery_size'] * (metrics['periphery_size'] - 1) / 2)
            else:
                metrics['periphery_density'] = 0.0
        
    except Exception as e:
        logger.error(f"Error extracting metrics from {file_path}: {e}")
        
    return metrics

def extract_value(content: str, pattern: str) -> float:
    """Extract a numeric value using regex pattern."""
    match = re.search(pattern, content)
    return float(match.group(1)) if match else None

def get_t_critical(df: int, confidence: float = 0.99) -> float:
    """Get t-critical value for given degrees of freedom and confidence level."""
    # Pre-calculated t-values for common df and 99% confidence
    t_values_99 = {
        1: 63.657, 2: 9.925, 3: 5.841, 4: 4.604, 5: 4.032,
        6: 3.707, 7: 3.499, 8: 3.355, 9: 3.250, 10: 3.169,
        11: 3.106, 12: 3.055, 13: 3.012, 14: 2.977, 15: 2.947,
        16: 2.921, 17: 2.898, 18: 2.878, 19: 2.861, 20: 2.845,
        21: 2.831, 22: 2.819, 23: 2.807, 24: 2.797, 25: 2.787,
        26: 2.779, 27: 2.771, 28: 2.763, 29: 2.756, 30: 2.750
    }
    
    if df <= 30:
        return t_values_99.get(df, 3.250)  # Default to conservative value
    else:
        # For df > 30, use normal approximation (z-score for 99% CI)
        return 2.576

def calculate_confidence_interval(values: List[float], confidence: float = 0.99) -> Tuple[float, float, float]:
    """Calculate confidence interval for a list of values using t-distribution."""
    if not values or all(v is None for v in values):
        return None, None, None
    
    # Filter out None values
    clean_values = [v for v in values if v is not None]
    
    if len(clean_values) == 0:
        return None, None, None
    
    mean = np.mean(clean_values)
    std = np.std(clean_values, ddof=1)
    
    # Calculate confidence interval using t-distribution
    n = len(clean_values)
    if n < 2:
        return mean, mean, mean
    
    df = n - 1  # degrees of freedom
    t_critical = get_t_critical(df, confidence)
    margin_error = t_critical * (std / np.sqrt(n))
    
    return mean, mean - margin_error, mean + margin_error

def analyze_all_samples(base_path: str) -> pd.DataFrame:
    """Analyze all Voat samples and compute confidence intervals."""
    sample_files = []
    
    # Find all enhanced_network_analysis.txt files
    for sample_file in Path(base_path).glob("sample_*/enhanced_network_analysis.txt"):
        sample_files.append(str(sample_file))
    
    logger.info(f"Found {len(sample_files)} samples to analyze")
    
    # Extract metrics from all samples
    all_metrics = []
    for file_path in sorted(sample_files):
        logger.info(f"Processing {file_path}")
        metrics = extract_network_metrics(file_path)
        all_metrics.append(metrics)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_metrics)
    
    # Calculate confidence intervals for each metric
    results = []
    metrics_to_analyze = [
        'density', 'core_density', 'periphery_density', 'avg_degree', 
        'avg_weighted_degree', 'avg_clustering', 'lcc_percentage', 
        'core_percentage_of_lcc', 'core_periphery_density'
    ]
    
    for metric in metrics_to_analyze:
        if metric in df.columns:
            values = df[metric].tolist()
            mean, ci_lower, ci_upper = calculate_confidence_interval(values)
            
            results.append({
                'metric': metric,
                'mean': mean,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'std': np.std([v for v in values if v is not None]) if any(v is not None for v in values) else None,
                'n_samples': len([v for v in values if v is not None])
            })
    
    results_df = pd.DataFrame(results)
    
    # Also save raw data for reference
    df.to_csv(base_path + '/voat_network_metrics_raw.csv', index=False)
    
    return results_df

def main():
    """Main function to run the analysis."""
    base_path = "/home/socio/ysocial-simulations/MADOC/voat-technology"
    
    logger.info("Starting network metrics analysis for MADOC Voat samples with 99% confidence intervals")
    
    try:
        results = analyze_all_samples(base_path)
        
        # Save results
        output_file = base_path + '/voat_network_metrics_confidence_intervals_99.csv'
        results.to_csv(output_file, index=False)
        
        # Also save as JSON for easy reading
        json_file = base_path + '/voat_network_metrics_confidence_intervals_99.json'
        results.to_json(json_file, orient='records', indent=2)
        
        logger.info(f"Results saved to {output_file}")
        logger.info(f"JSON results saved to {json_file}")
        
        # Print summary
        print("\n" + "="*80)
        print("NETWORK METRICS WITH 99% CONFIDENCE INTERVALS - MADOC VOAT SAMPLES")
        print("="*80)
        
        for _, row in results.iterrows():
            if row['mean'] is not None:
                print(f"\n{row['metric'].replace('_', ' ').title()}:")
                print(f"  Mean: {row['mean']:.4f}")
                print(f"  99% CI: [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]")
                print(f"  Std Dev: {row['std']:.4f}")
                print(f"  N: {row['n_samples']}")
        
        print("\n" + "="*80)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()