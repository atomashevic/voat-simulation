#!/usr/bin/env python3
"""
Simplified topic comparison between two corpora using BERTopic.
Back-to-basics approach with sensible defaults.
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity


def load_data(path: Path, text_column: str = None, max_docs: int = None) -> Tuple[List[str], pd.DataFrame]:
    """Load data from CSV or Parquet file."""
    if path.suffix.lower() == '.csv':
        df = pd.read_csv(path)
    elif path.suffix.lower() == '.parquet':
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    
    # Auto-detect text column if not specified
    if text_column is None:
        # Try common column names in order of preference
        for col in ['content', 'text', 'body', 'tweet', 'message', 'selftext']:
            if col in df.columns:
                text_column = col
                print(f"Auto-detected text column: {text_column}")
                break
        
        if text_column is None:
            # Fall back to first string column
            str_cols = [col for col in df.columns if df[col].dtype == 'object']
            if str_cols:
                text_column = str_cols[0]
                print(f"Using first string column: {text_column}")
            else:
                raise ValueError("No suitable text column found")
    
    # Extract and clean text
    texts = df[text_column].fillna('').astype(str).tolist()
    
    # Basic cleaning
    cleaned_texts = []
    for text in texts:
        # Fix missing spaces after periods (common in simulated text)
        text = re.sub(r'\.([A-Z])', r'. \1', text)
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        # Remove username mentions at start of replies
        text = re.sub(r'^[.@]\S+\s+', '', text)
        # Remove excessive whitespace
        text = ' '.join(text.split())
        # Skip very short texts
        if len(text) >= 50:  # Minimum 50 characters
            cleaned_texts.append(text)
    
    if max_docs and len(cleaned_texts) > max_docs:
        import random
        random.seed(42)
        cleaned_texts = random.sample(cleaned_texts, max_docs)
    
    print(f"Loaded {len(cleaned_texts)} documents from {path.name}")
    return cleaned_texts, df


def train_bertopic_model(texts: List[str], 
                        min_topic_size: int = 15,
                        nr_topics: str = "auto") -> tuple:
    """Train a BERTopic model with sensible defaults."""
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from hdbscan import HDBSCAN
    from umap import UMAP
    
    # Embedding model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # UMAP for dimensionality reduction
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )
    
    # HDBSCAN for clustering
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )
    
    # Vectorizer with reasonable constraints
    vectorizer_model = CountVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        min_df=10,  # Words must appear in at least 10 documents
        max_df=0.5  # Words can't appear in more than 50% of documents
    )
    
    # Create and train model
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        nr_topics=nr_topics,
        min_topic_size=min_topic_size,
        calculate_probabilities=False,
        verbose=True
    )
    
    topics, probs = topic_model.fit_transform(texts)
    
    return topic_model, topics


def extract_topic_info(model, top_n_words: int = 10) -> pd.DataFrame:
    """Extract clean topic information."""
    info = model.get_topic_info()
    
    # Filter out outlier topic (-1)
    info = info[info['Topic'] != -1].copy()
    
    # Extract clean top words for each topic
    topic_words = []
    for topic_id in info['Topic']:
        words_scores = model.get_topic(topic_id)
        if words_scores:
            # Get just the words (not scores)
            words = [w for w, _ in words_scores[:top_n_words]]
            # Filter out very short words and numbers
            words = [w for w in words if len(w) > 2 and not w.isdigit()]
            topic_words.append(' '.join(words[:top_n_words]))
        else:
            topic_words.append('')
    
    info['TopWords'] = topic_words
    
    # Create cleaner labels
    info['Label'] = info.apply(
        lambda row: f"Topic {row['Topic']}: {' '.join(row['TopWords'].split()[:3])}", 
        axis=1
    )
    
    return info


def compute_topic_similarity(model1, model2, topics1_info, topics2_info) -> np.ndarray:
    """Compute similarity between topics from two models."""
    from sentence_transformers import SentenceTransformer
    
    # Use topic embeddings if available, otherwise create from top words
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Get embeddings for model 1 topics
    if hasattr(model1, 'topic_embeddings_'):
        emb1 = model1.topic_embeddings_
        # Filter to non-outlier topics
        valid_topics1 = topics1_info['Topic'].values
        emb1 = emb1[valid_topics1]
    else:
        texts1 = topics1_info['TopWords'].tolist()
        emb1 = embedding_model.encode(texts1, show_progress_bar=False)
    
    # Get embeddings for model 2 topics  
    if hasattr(model2, 'topic_embeddings_'):
        emb2 = model2.topic_embeddings_
        # Filter to non-outlier topics
        valid_topics2 = topics2_info['Topic'].values
        emb2 = emb2[valid_topics2]
    else:
        texts2 = topics2_info['TopWords'].tolist()
        emb2 = embedding_model.encode(texts2, show_progress_bar=False)
    
    # Compute cosine similarity
    if len(emb1) > 0 and len(emb2) > 0:
        return cosine_similarity(emb1, emb2)
    else:
        return np.array([[]])


def find_best_matches(sim_matrix: np.ndarray, threshold: float = 0.5) -> List[dict]:
    """Find best topic matches using Hungarian algorithm."""
    if sim_matrix.size == 0:
        return []
    
    # Use Hungarian algorithm for optimal matching
    cost_matrix = 1 - sim_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    matches = []
    for r, c in zip(row_ind, col_ind):
        sim = sim_matrix[r, c]
        if sim >= threshold:
            matches.append({
                'corpus1_topic': int(r),
                'corpus2_topic': int(c),
                'similarity': float(sim)
            })
    
    return matches


def main():
    parser = argparse.ArgumentParser(description="Simplified topic comparison")
    parser.add_argument("--corpus1", type=Path, required=True, help="Path to first corpus")
    parser.add_argument("--corpus2", type=Path, required=True, help="Path to second corpus")
    parser.add_argument("--corpus1-column", type=str, help="Text column for corpus1")
    parser.add_argument("--corpus2-column", type=str, help="Text column for corpus2")
    parser.add_argument("--min-topic-size", type=int, default=20, help="Minimum topic size")
    parser.add_argument("--nr-topics", type=str, default="auto", help="Number of topics or 'auto'")
    parser.add_argument("--similarity-threshold", type=float, default=0.5, help="Minimum similarity for matching")
    parser.add_argument("--max-docs", type=int, help="Maximum documents to use per corpus")
    parser.add_argument("--output-dir", type=Path, default=Path("topic_comparison_results"))
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("SIMPLIFIED TOPIC COMPARISON")
    print("=" * 80)
    
    # Load data
    print("\n1. Loading data...")
    texts1, df1 = load_data(args.corpus1, args.corpus1_column, args.max_docs)
    texts2, df2 = load_data(args.corpus2, args.corpus2_column, args.max_docs)
    
    # Train models
    print("\n2. Training topic models...")
    print(f"   Using min_topic_size={args.min_topic_size}")
    
    print(f"\n   Training model for {args.corpus1.name}...")
    model1, topics1 = train_bertopic_model(texts1, args.min_topic_size, args.nr_topics)
    
    print(f"\n   Training model for {args.corpus2.name}...")
    model2, topics2 = train_bertopic_model(texts2, args.min_topic_size, args.nr_topics)
    
    # Extract topic info
    print("\n3. Extracting topic information...")
    topics1_info = extract_topic_info(model1)
    topics2_info = extract_topic_info(model2)
    
    print(f"   Found {len(topics1_info)} topics in corpus1")
    print(f"   Found {len(topics2_info)} topics in corpus2")
    
    # Save topic info
    topics1_info.to_csv(args.output_dir / "corpus1_topics.csv", index=False)
    topics2_info.to_csv(args.output_dir / "corpus2_topics.csv", index=False)
    
    # Compute similarity
    print("\n4. Computing topic similarity...")
    sim_matrix = compute_topic_similarity(model1, model2, topics1_info, topics2_info)
    
    # Find matches
    print(f"\n5. Finding matches (threshold={args.similarity_threshold})...")
    matches = find_best_matches(sim_matrix, args.similarity_threshold)
    
    print(f"   Found {len(matches)} matching topic pairs")
    
    # Create detailed matches table
    if matches:
        matches_df = []
        for match in matches:
            r = match['corpus1_topic']
            c = match['corpus2_topic']
            
            matches_df.append({
                'corpus1_topic': r,
                'corpus1_label': topics1_info.iloc[r]['Label'],
                'corpus1_words': topics1_info.iloc[r]['TopWords'],
                'corpus1_size': topics1_info.iloc[r]['Count'],
                'corpus2_topic': c,
                'corpus2_label': topics2_info.iloc[c]['Label'],
                'corpus2_words': topics2_info.iloc[c]['TopWords'],
                'corpus2_size': topics2_info.iloc[c]['Count'],
                'similarity': match['similarity']
            })
        
        matches_df = pd.DataFrame(matches_df)
        matches_df = matches_df.sort_values('similarity', ascending=False)
        matches_df.to_csv(args.output_dir / "topic_matches.csv", index=False)
        
        print("\n6. Top matching topics:")
        print("-" * 80)
        for _, row in matches_df.head(10).iterrows():
            print(f"Similarity: {row['similarity']:.3f}")
            print(f"  Corpus1: {row['corpus1_label']}")
            print(f"  Corpus2: {row['corpus2_label']}")
            print()
    
    # Save similarity matrix
    if sim_matrix.size > 0:
        np.save(args.output_dir / "similarity_matrix.npy", sim_matrix)
    
    # Summary
    summary = {
        'corpus1_path': str(args.corpus1),
        'corpus2_path': str(args.corpus2),
        'corpus1_docs': len(texts1),
        'corpus2_docs': len(texts2),
        'corpus1_topics': len(topics1_info),
        'corpus2_topics': len(topics2_info),
        'matches': len(matches),
        'avg_similarity': np.mean([m['similarity'] for m in matches]) if matches else 0,
        'parameters': {
            'min_topic_size': args.min_topic_size,
            'nr_topics': args.nr_topics,
            'similarity_threshold': args.similarity_threshold
        }
    }
    
    with open(args.output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n7. Results saved to {args.output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
