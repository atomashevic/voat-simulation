"""
toxygen-madoc.py - Content Moderation Analysis for Reddit Data
Uses ToxiGen RoBERTa model to analyze posts from a parquet file containing Reddit data.
"""

import pandas as pd
import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import random
from tqdm import tqdm

def load_data(parquet_path, sample_size=None):
    """Load data from parquet file and sample one month of data"""
    print("[1/3] Loading parquet file...")
    df = pd.read_parquet(parquet_path, engine='pyarrow')
    
    # Convert UNIX timestamp to datetime
    df['datetime'] = pd.to_datetime(df['publish_date'], unit='s')
    
    # Get min and max dates
    min_date = df['datetime'].min()
    max_date = df['datetime'].max()
    print(f"Data spans from {min_date} to {max_date}")
    
    # Get all available months in the dataset
    df['year_month'] = df['datetime'].dt.to_period('M')
    available_months = df['year_month'].unique()
    
    # Randomly select one month
    selected_month = random.choice(available_months)
    print(f"Randomly selected month: {selected_month}")
    
    # Filter data for the selected month
    month_data = df[df['year_month'] == selected_month].copy()
    print(f"Data for {selected_month}: {len(month_data)} records")
    
    # Take a random sample if specified
    if sample_size and len(month_data) > sample_size:
        month_data = month_data.sample(sample_size, random_state=42)
        print(f"Sampled {sample_size} records from {selected_month}")
    
    return month_data

def analyze_toxicity(df):
    """Analyze posts using ToxiGen RoBERTa model"""
    print("[2/3] Analyzing toxicity...")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("tomh/toxigen_roberta")
    model = AutoModelForSequenceClassification.from_pretrained("tomh/toxigen_roberta")
    
    # Process in batches
    batch_size = 200
    toxic_scores = []
    
    # Use tqdm for progress bar
    for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
        batch = df['content'].iloc[i:i+batch_size].fillna("").tolist()
        
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=128  # Limit sequence length
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            toxic_scores.extend(probs[:, 1].tolist())
    
    # Add toxicity scores to dataframe
    df['toxicity'] = toxic_scores
    
    return df

def print_summary(df, toxic_threshold=0.5):
    """Print summary of toxicity analysis"""
    print("[3/3] Generating summary...")
    
    def analyze_category(data, category_name):
        print(f"\n=== {category_name} ===")
        print(f"Total items: {len(data)}")
        
        if len(data) == 0:
            print("No items in this category")
            return
            
        toxic_items = data[data['toxicity'] > toxic_threshold]
        mean_toxicity = data['toxicity'].mean()
        print(f"Mean toxicity: {mean_toxicity:.3f}")
        print(f"Items with toxicity > {toxic_threshold}: {len(toxic_items)} ({len(toxic_items)/len(data)*100:.1f}%)")
        
        if len(toxic_items) > 0:
            print("\nMost toxic examples:")
            for _, item in toxic_items.nlargest(3, 'toxicity').iterrows():
                print(f"\nToxic score: {item['toxicity']:.3f}")
                print(f"Content: {item['content'][:200]}...")
        return toxic_items.nlargest(3, 'toxicity') if len(toxic_items) > 0 else None
    
    # Analyze each category
    examples = {}
    for interaction_type in df['interaction_type'].unique():
        category_data = df[df['interaction_type'] == interaction_type].copy()
        toxic_examples = analyze_category(category_data, interaction_type)
        if toxic_examples is not None:
            examples[interaction_type] = toxic_examples
    return examples

def plot_toxicity_distributions(toxicity_df, output_dir):
    """Create visualizations for toxicity distributions"""
    print("\nGenerating toxicity distribution visualizations...")
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set style for all plots
    plt.style.use('ggplot')
    
    # ---- Overall Toxicity Distribution Plots ----
    
    # Version 1: Histogram with KDE
    plt.figure(figsize=(10, 6))
    sns.histplot(toxicity_df['toxicity'], kde=True, bins=30, color='darkblue')
    plt.title('Distribution of Toxicity Scores (Overall)', fontsize=15)
    plt.xlabel('Toxicity Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.axvline(x=0.5, color='red', linestyle='--', label='Toxic Threshold (0.5)')
    plt.axvline(x=0.25, color='orange', linestyle='--', label='Mild Threshold (0.25)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'toxicity_histogram.png'), dpi=300)
    plt.close()
    
    # Version 2: Violin Plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(y=toxicity_df['toxicity'], color='darkblue')
    plt.title('Violin Plot of Toxicity Scores (Overall)', fontsize=15)
    plt.ylabel('Toxicity Score', fontsize=12)
    plt.axhline(y=0.5, color='red', linestyle='--', label='Toxic Threshold (0.5)')
    plt.axhline(y=0.25, color='orange', linestyle='--', label='Mild Threshold (0.25)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'toxicity_violin.png'), dpi=300)
    plt.close()
    
    # Version 3: ECDF (Empirical Cumulative Distribution Function)
    plt.figure(figsize=(10, 6))
    sns.ecdfplot(data=toxicity_df, x='toxicity', color='darkblue', linewidth=2)
    plt.title('Cumulative Distribution of Toxicity Scores (Overall)', fontsize=15)
    plt.xlabel('Toxicity Score', fontsize=12)
    plt.ylabel('Cumulative Proportion', fontsize=12)
    plt.axvline(x=0.5, color='red', linestyle='--', label='Toxic Threshold (0.5)')
    plt.axvline(x=0.25, color='orange', linestyle='--', label='Mild Threshold (0.25)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'toxicity_ecdf.png'), dpi=300)
    plt.close()
    
    # ---- Comparative Toxicity Distribution Plots ----
    
    # Filter data
    posts = toxicity_df[toxicity_df['interaction_type'] == 'POST']
    comments = toxicity_df[toxicity_df['interaction_type'] == 'COMMENT']
    
    # Version 1: Side-by-side histograms
    plt.figure(figsize=(12, 6))
    
    # Create a custom colormap
    colors = ['#1f77b4', '#ff7f0e']  # Blue for posts, orange for comments
    
    plt.hist([posts['toxicity'], comments['toxicity']], 
             bins=30, 
             alpha=0.7, 
             label=['Posts', 'Comments'],
             color=colors)
    
    plt.title('Comparative Toxicity Distribution: Posts vs Comments', fontsize=15)
    plt.xlabel('Toxicity Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.axvline(x=0.5, color='red', linestyle='--', label='Toxic Threshold (0.5)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'comparative_histogram.png'), dpi=300)
    plt.close()
    
    # Version 2: Kernel Density Estimation (KDE) plot
    plt.figure(figsize=(12, 6))
    sns.kdeplot(data=posts, x='toxicity', fill=True, common_norm=False, alpha=0.5, color='#1f77b4', label='Posts')
    sns.kdeplot(data=comments, x='toxicity', fill=True, common_norm=False, alpha=0.5, color='#ff7f0e', label='Comments')
    plt.title('Density of Toxicity Scores: Posts vs Comments', fontsize=15)
    plt.xlabel('Toxicity Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.axvline(x=0.5, color='red', linestyle='--', label='Toxic Threshold (0.5)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'comparative_kde.png'), dpi=300)
    plt.close()
    
    # Version 3: Box plot comparison
    plt.figure(figsize=(10, 6))
    comparison_df = pd.DataFrame({
        'Content Type': ['Posts'] * len(posts) + ['Comments'] * len(comments),
        'Toxicity': list(posts['toxicity']) + list(comments['toxicity'])
    })
    
    sns.boxplot(x='Content Type', y='Toxicity', data=comparison_df, palette=['#1f77b4', '#ff7f0e'])
    plt.title('Toxicity Score Distribution: Posts vs Comments', fontsize=15)
    plt.xlabel('Content Type', fontsize=12)
    plt.ylabel('Toxicity Score', fontsize=12)
    plt.axhline(y=0.5, color='red', linestyle='--', label='Toxic Threshold (0.5)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'comparative_boxplot.png'), dpi=300)
    plt.close()
    
    print(f"Visualizations saved to: {plots_dir}")

def main(parquet_path, sample_size=None):
    """Main analysis workflow"""
    try:
        # Load and sample data
        df = load_data(parquet_path, sample_size)
        
        # Run toxicity analysis
        toxicity_df = analyze_toxicity(df)
        
        # Construct output file path
        output_dir = "results/madoc_toxigen"
        output_csv = f"{output_dir}/toxigen_results.csv"
        summary_file = f"{output_dir}/toxigen_summary.txt"

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save results
        toxicity_df[['post_id', 'toxicity', 'interaction_type']].to_csv(output_csv, index=False)
        print(f"\nToxicity results saved to: {output_csv}")
        
        # Calculate overall statistics
        mean_toxicity = toxicity_df['toxicity'].mean()
        items_above_50 = (toxicity_df['toxicity'] > 0.5).sum()
        items_above_25 = (toxicity_df['toxicity'] > 0.25).sum()
        total_items = len(toxicity_df)
        
        # Get toxic examples for each category
        print("\nToxicity Summary:")
        toxic_examples = print_summary(toxicity_df)
        
        # Generate visualizations
        plot_toxicity_distributions(toxicity_df, output_dir)
        
        # Save summary to file
        with open(summary_file, 'w') as f:
            f.write("Toxicity Analysis Summary\n")
            f.write("========================\n\n")
            f.write(f"Total items analyzed: {total_items}\n")
            f.write(f"Mean toxicity score: {mean_toxicity:.4f}\n")
            f.write(f"Items above 0.5 threshold: {items_above_50}\n")
            f.write(f"Items above 0.25 threshold: {items_above_25}\n")
            f.write(f"Percentage above 0.5: {(items_above_50/total_items)*100:.2f}%\n")
            f.write(f"Percentage above 0.25: {(items_above_25/total_items)*100:.2f}%\n\n")
            
            # Add breakdown by interaction type
            f.write("Breakdown by Interaction Type\n")
            f.write("============================\n")
            for interaction_type in ['POST', 'COMMENT']:
                category_data = toxicity_df[toxicity_df['interaction_type'] == interaction_type]
                if len(category_data) > 0:
                    mean_cat_toxicity = category_data['toxicity'].mean()
                    cat_above_50 = (category_data['toxicity'] > 0.5).sum()
                    f.write(f"\n{interaction_type}:\n")
                    f.write(f"Count: {len(category_data)}\n")
                    f.write(f"Mean toxicity: {mean_cat_toxicity:.4f}\n")
                    f.write(f"Items above 0.5: {cat_above_50} ({(cat_above_50/len(category_data))*100:.2f}%)\n")
                    
                    # Add toxic examples for this category
                    if interaction_type in toxic_examples:
                        f.write("\nMost toxic examples:\n")
                        for _, item in toxic_examples[interaction_type].iterrows():
                            f.write(f"\nToxic score: {item['toxicity']:.3f}\n")
                            f.write(f"Content: {item['content'][:200]}...\n")
                    f.write("\n")
            
            # Add information about visualizations
            f.write("\nVisualizations\n")
            f.write("==============\n")
            f.write(f"Visualization files have been saved to: {output_dir}/plots/\n")
            f.write("The following visualizations were generated:\n")
            f.write("1. Overall toxicity distribution (histogram, violin plot, ECDF)\n")
            f.write("2. Comparative toxicity distribution between posts and comments\n")
            f.write("   (histogram, KDE plot, boxplot)\n")
        
        print(f"\nAnalysis complete! Summary saved to: {summary_file}")
        print(f"Visualizations saved to: {output_dir}/plots/")
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze toxicity in Reddit data using ToxiGen RoBERTa")
    parser.add_argument("--parquet_path", default="madoc-calibration/technology.parquet", 
                        help="Path to parquet file (default: madoc-calibration/technology.parquet)")
    parser.add_argument("--sample_size", type=int, default=10000,
                        help="Number of samples to analyze (default: 10000)")
    args = parser.parse_args()
    main(args.parquet_path, args.sample_size)
