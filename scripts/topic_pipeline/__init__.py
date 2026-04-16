"""
Utilities for the unified topic modelling pipeline.

This package bundles helpers that load thread-level corpora from simulation
outputs and MADOC Voat data, preprocess the text, fit BERTopic models, and
score cross-dataset topic similarity. The public entry point for running the
pipeline is `scripts/run_topic_similarity_pipeline.py`.
"""

from __future__ import annotations

__all__ = [
    "data_io",
    "modeling",
    "plots",
    "preprocessing",
    "similarity",
]
