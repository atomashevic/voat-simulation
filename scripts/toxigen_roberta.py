"""
toxigen-roberta.py - Content Moderation Analysis
Uses ToxiGen RoBERTa model to analyze posts from an SQLite database.
"""

import pandas as pd
import sqlite3
import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, Tuple


def resolve_device(device: str) -> torch.device:
    device = (device or "auto").strip().lower()
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device in {"cuda", "gpu"}:
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device, but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if device == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("Requested MPS device, but torch.backends.mps.is_available() is False.")
        return torch.device("mps")
    if device == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device: {device}")


def load_tweets(db_path):
    """Load tweets from SQLite database"""
    print("[1/3] Connecting to database...")
    conn = sqlite3.connect(db_path)

    print("[2/3] Loading tweet data...")
    df = pd.read_sql_query("SELECT * FROM post", conn)

    # Add post type column
    df['post_type'] = 'regular_post'  # Default type
    df.loc[(~df['news_id'].isna()) & (df['comment_to'] == -1), 'post_type'] = 'news_share'
    df.loc[df['comment_to'] != -1, 'post_type'] = 'comment'

    # Add is_comment column for easier filtering
    df['is_comment'] = df['comment_to'] != -1

    return conn, df

def analyze_toxicity(df, device: torch.device):
    """Analyze posts using ToxiGen RoBERTa model"""
    print("[3/3] Analyzing toxicity...")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("tomh/toxigen_roberta")
    model = AutoModelForSequenceClassification.from_pretrained("tomh/toxigen_roberta")
    model.eval()
    model.to(device)

    # Process in larger batches to utilize more RAM
    batch_size = 200  # Increased from 50 to 200
    toxic_scores = []

    for i in range(0, len(df), batch_size):
        batch = df['tweet'].iloc[i:i+batch_size].tolist()

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=128  # Limit sequence length
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs)  # type: ignore[arg-type]
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            toxic_scores.extend(probs[:, 1].tolist())

        print(f"Processed {min(i+batch_size, len(df))}/{len(df)} posts...")

    return pd.DataFrame({
        'id': df['id'],
        'toxicity': toxic_scores,
        'post_type': df['post_type'],
        'is_comment': df['is_comment']
    })

def print_summary(df, toxic_threshold=0.5):
    """Print summary of toxicity analysis"""

    def analyze_category(data, category_name):
        print(f"\n=== {category_name} ===")
        print(f"Total posts: {len(data)}")

        if len(data) == 0:
            print("No posts in this category")
            return

        toxic_posts = data[data['toxicity'] > toxic_threshold]
        mean_toxicity = data['toxicity'].mean()
        print(f"Mean toxicity: {mean_toxicity:.3f}")
        print(f"Posts with toxicity > {toxic_threshold}: {len(toxic_posts)} ({len(toxic_posts)/len(data)*100:.1f}%)")

        if len(toxic_posts) > 0:
            print("\nMost toxic examples:")
            for _, post in toxic_posts.nlargest(3, 'toxicity').iterrows():
                print(f"\nToxic score: {post['toxicity']:.3f}")
                print(f"Text: {post['text'][:200]}...")
        return toxic_posts.nlargest(3, 'toxicity') if len(toxic_posts) > 0 else None

    # Analyze each category
    examples = {}
    for post_type in ['regular_post', 'news_share', 'comment']:
        category_data = df[df['post_type'] == post_type].copy()  # Create a copy to avoid SettingWithCopyWarning
        toxic_examples = analyze_category(category_data, post_type.replace('_', ' ').title())
        if toxic_examples is not None:
            examples[post_type] = toxic_examples
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
    plt.style.use('default')  # Reset to default style
    sns.histplot(toxicity_df['toxicity'], kde=True, bins=30, color='darkblue')
    plt.title('Simulated Toxicity Distribution', fontsize=15)
    plt.xlabel('Toxicity Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.axvline(x=0.5, color='red', linestyle='--', label='Toxic Threshold (0.5)')
    plt.axvline(x=0.25, color='orange', linestyle='--', label='Mild Threshold (0.25)')
    plt.legend()
    # Add mean toxicity text below the legend
    mean_toxicity = np.mean(toxicity_df['toxicity'])
    plt.text(0.7, plt.gca().get_ylim()[1] * 0.7, f'Mean Toxicity: {mean_toxicity:.4f}',
            fontsize=14,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    # Remove all grid lines for minimalistic look
    plt.grid(False)
    plt.gca().set_axisbelow(True)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
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
    comments = toxicity_df[toxicity_df['is_comment'] == True]
    posts = toxicity_df[toxicity_df['is_comment'] == False]

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

def load_madoc_toxicity(parquet_path: str) -> Tuple[Optional[float], int]:
    """Compute mean ToxiGen score from a MADOC parquet export."""
    if not os.path.isfile(parquet_path):
        print(f"MADOC parquet not found: {parquet_path}; skipping comparison.")
        return None, 0

    try:
        parquet_df = pd.read_parquet(parquet_path, columns=["toxicity_toxigen"])
    except Exception as exc:  # pragma: no cover - best effort comparison
        print(f"Failed to read MADOC parquet {parquet_path}: {exc}")
        return None, 0

    parquet_df = parquet_df.dropna(subset=["toxicity_toxigen"])
    if parquet_df.empty:
        print(f"No toxigen scores found in MADOC data at {parquet_path}.")
        return None, 0

    return float(parquet_df["toxicity_toxigen"].mean()), len(parquet_df)

def main(argv=None) -> int:
    """Main analysis workflow"""
    parser = argparse.ArgumentParser(description="Analyze toxicity in posts using ToxiGen RoBERTa")
    parser.add_argument("db_path", help="Path to SQLite database")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Destination directory for toxigen outputs (default: results/<db_basename>)",
    )
    parser.add_argument(
        "--madoc-parquet",
        type=str,
        default="MADOC/voat-technology/voat_technology_madoc.parquet",
        help="MADOC parquet file containing toxigen scores for comparison",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device: auto|cuda|cpu|mps (default: auto).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating PNG plots (useful for large batch runs).",
    )
    args = parser.parse_args(argv)

    db_name = os.path.splitext(os.path.basename(args.db_path))[0]
    output_dir = args.output_dir or os.path.join("results", db_name)
    output_csv = os.path.join(output_dir, "toxigen.csv")
    summary_file = os.path.join(output_dir, "toxigen_summary.txt")

    conn = None
    try:
        device = resolve_device(args.device)
        print(f"Using device: {device}")
        if device.type == "cuda":
            try:
                print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            except Exception:
                pass

        # Load tweets
        conn, df = load_tweets(args.db_path)

        # Run toxicity analysis
        toxicity_df = analyze_toxicity(df, device=device)

        # Add tweet text and other columns to toxicity_df
        toxicity_df["text"] = df["tweet"]

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save results (without the text column)
        toxicity_df[["id", "toxicity", "post_type", "is_comment"]].to_csv(output_csv, index=False)
        print(f"\nToxicity results saved to: {output_csv}")

        # Calculate overall statistics
        mean_toxicity = toxicity_df["toxicity"].mean()
        posts_above_50 = (toxicity_df["toxicity"] > 0.5).sum()
        posts_above_25 = (toxicity_df["toxicity"] > 0.25).sum()
        total_posts = len(toxicity_df)

        # Get toxic examples for each category
        print("\nToxicity Summary:")
        toxic_examples = print_summary(toxicity_df)

        # Generate visualizations
        if args.no_plots:
            print("Skipping toxicity distribution visualizations (--no-plots).")
        else:
            plot_toxicity_distributions(toxicity_df, output_dir)

        # Compare to MADOC scores if available
        madoc_mean, madoc_count = load_madoc_toxicity(args.madoc_parquet)
        if madoc_mean is not None:
            print(f"\nMADOC mean ToxiGen score ({madoc_count} rows): {madoc_mean:.4f}")
            print(f"Simulation vs MADOC difference: {(mean_toxicity - madoc_mean):.4f}")
        else:
            print(f"\nMADOC comparison skipped (path: {args.madoc_parquet}).")

        # Save summary to file
        with open(summary_file, "w") as f:
            f.write("Toxicity Analysis Summary\n")
            f.write("========================\n\n")
            f.write(f"Total posts analyzed: {total_posts}\n")
            f.write(f"Mean toxicity score: {mean_toxicity:.4f}\n")
            f.write(f"Posts above 0.5 threshold: {posts_above_50}\n")
            f.write(f"Posts above 0.25 threshold: {posts_above_25}\n")
            if total_posts > 0:
                f.write(f"Percentage above 0.5: {(posts_above_50/total_posts)*100:.2f}%\n")
                f.write(f"Percentage above 0.25: {(posts_above_25/total_posts)*100:.2f}%\n\n")
            else:
                f.write("Percentage above 0.5: n/a (no posts)\n")
                f.write("Percentage above 0.25: n/a (no posts)\n\n")

            if madoc_mean is not None:
                f.write(f"MADOC toxigen mean (n={madoc_count}): {madoc_mean:.4f}\n")
                f.write(f"Simulation - MADOC difference: {(mean_toxicity - madoc_mean):.4f}\n\n")
            else:
                f.write(f"MADOC comparison data unavailable: {args.madoc_parquet}\n\n")

            # Add breakdown by post type
            f.write("Breakdown by Post Type\n")
            f.write("=====================\n")
            for post_type in ["regular_post", "news_share", "comment"]:
                category_data = toxicity_df[toxicity_df["post_type"] == post_type]
                if len(category_data) > 0:
                    mean_cat_toxicity = category_data["toxicity"].mean()
                    cat_above_50 = (category_data["toxicity"] > 0.5).sum()
                    f.write(f"\n{post_type.replace('_', ' ').title()}:\n")
                    f.write(f"Count: {len(category_data)}\n")
                    f.write(f"Mean toxicity: {mean_cat_toxicity:.4f}\n")
                    f.write(
                        f"Posts above 0.5: {cat_above_50} ({(cat_above_50/len(category_data))*100:.2f}%)\n"
                    )

                    # Add toxic examples for this category
                    if post_type in toxic_examples:
                        f.write("\nMost toxic examples:\n")
                        for _, post in toxic_examples[post_type].iterrows():
                            f.write(f"\nToxic score: {post['toxicity']:.3f}\n")
                            f.write(f"Text: {post['text'][:200]}...\n")
                    f.write("\n")

            # Add information about visualizations
            f.write("\nVisualizations\n")
            f.write("==============\n")
            f.write(f"Visualization files have been saved to: {output_dir}/plots/\n")
            f.write("The following visualizations were generated:\n")
            f.write("1. Overall toxicity distribution (histogram, violin plot, ECDF)\n")
            f.write("2. Comparative toxicity distribution between posts and comments\n")
            f.write("   (histogram, KDE plot, boxplot)\n")

        # Close database connection
        conn.close()

        print(f"\nAnalysis complete! Summary saved to: {summary_file}")
        if not args.no_plots:
            print(f"Visualizations saved to: {output_dir}/plots/")

        return 0

    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        if conn is not None:
            conn.close()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
