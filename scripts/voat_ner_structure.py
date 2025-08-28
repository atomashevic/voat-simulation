#!/usr/bin/env python3
"""
Named-entity structure comparisons between simulation2 and Voat MADOC parquet data.

Outputs per subset (posts, comments):
- Doc–Entity bipartite graph (GEXF/GraphML or CSV edgelist)
- Distributional comparisons (entity label counts, top entities, Jaccard overlap, Jensen–Shannon distance)

Run (pyenv):
  PYENV_VERSION=ysocial python scripts/voat_ner_structure.py \
    --madoc-parquet MADOC/voat-technology/voat_technology_madoc.parquet \
    --mode both --graph-format gexf

Notes:
- Uses spaCy for NER (default: en_core_web_sm). You can override with --spacy-model.
- If your spaCy model is not installed, install it in your environment or pass a local model path.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import collections
import json
import math
import re
import sys

import numpy as np
import pandas as pd


# -------------------------
# Text normalization utils
# -------------------------
URL_RE = re.compile(r"https?://\S+")
WS_RE = re.compile(r"\s+")


def clean_text(s: str) -> str:
    s = s or ""
    s = URL_RE.sub(" ", s)
    s = s.replace("\n", " ")
    s = WS_RE.sub(" ", s)
    return s.strip()


# -------------------------
# Data loading (consistent with other scripts)
# -------------------------
def load_sim2_subset(posts_csv: Path, tox_csv: Path, want_comments: bool) -> pd.DataFrame:
    posts = pd.read_csv(posts_csv)
    tox = pd.read_csv(tox_csv)
    if not {"id", "tweet"}.issubset(posts.columns):
        raise ValueError("simulation2 posts.csv must contain columns: id,tweet")
    if not {"id", "is_comment"}.issubset(tox.columns):
        raise ValueError("simulation2 toxigen.csv must contain columns: id,is_comment")
    df = posts.merge(tox[["id", "is_comment"]], on="id", how="left")
    df = df[df["is_comment"] == bool(want_comments)].copy()
    df["text"] = df["tweet"].astype(str).map(clean_text)
    df = df[df["text"].str.len() > 0]
    return df[["id", "text"]].reset_index(drop=True)


def load_voat_subset(parquet_path: Path, want_comments: bool) -> pd.DataFrame:
    if not parquet_path.exists():
        raise FileNotFoundError(f"MADOC parquet not found: {parquet_path}")
    df = pd.read_parquet(parquet_path, columns=["post_id", "content", "interaction_type"])
    types = df["interaction_type"].astype(str).str.lower()
    if want_comments:
        mask = types.isin(["comment", "comments"])
    else:
        mask = types.isin(["post", "posts"])
    df = df[mask].dropna(subset=["content"]).copy()
    df["text"] = df["content"].astype(str).map(clean_text)
    df = df[df["text"].str.len() > 0]
    return df[["post_id", "text"]].reset_index(drop=True)


# -------------------------
# NER extraction via spaCy
# -------------------------
def load_spacy(model_name: str, max_length: int) -> "spacy.language.Language":  # type: ignore[name-defined]
    try:
        import spacy

        nlp = spacy.load(model_name, disable=["tagger", "lemmatizer", "textcat"])  # keep ner
        nlp.max_length = max_length
        if "ner" not in nlp.pipe_names:
            raise RuntimeError(f"spaCy model '{model_name}' does not include NER pipe.")
        return nlp
    except Exception as e:
        raise RuntimeError(
            "Failed to load spaCy model. Install it (e.g., python -m spacy download en_core_web_sm) or provide --spacy-model path."
        ) from e


def extract_entities(
    nlp,
    texts: Sequence[str],
    entity_types: Optional[Sequence[str]] = None,
    lowercase: bool = True,
    min_len: int = 2,
    batch_size: int = 64,
) -> List[List[Tuple[str, str]]]:
    """Return per-text list of (entity_text, entity_label)."""
    results: List[List[Tuple[str, str]]] = []
    keep = set(t.upper() for t in (entity_types or []))
    for doc in nlp.pipe(texts, batch_size=batch_size):
        items: List[Tuple[str, str]] = []
        for ent in getattr(doc, "ents", []) or []:
            label = ent.label_.upper()
            if keep and label not in keep:
                continue
            text = ent.text.strip()
            if lowercase:
                text = text.lower()
            if len(text) < min_len:
                continue
            items.append((text, label))
        results.append(items)
    return results


# -------------------------
# Graph construction and stats
# -------------------------
def normalize_entity_text(text: str, do_normalize: bool = False) -> str:
    """Normalize entity surface forms to collapse common variants.

    - Lowercase and trim whitespace
    - Optional heuristic normalization for common platforms and domains
    """
    t = (text or "").strip().lower()
    if not do_normalize:
        return t
    # Strip surrounding punctuation
    t = t.strip("\t\n\r \"'`.,;:!?()[]{}")
    # Collapse whitespace
    t = re.sub(r"\s+", " ", t)
    # Simple alias map
    alias = {
        "yt": "youtube",
        "you tube": "youtube",
        "u tube": "youtube",
        "fb": "facebook",
        "insta": "instagram",
        "ig": "instagram",
        "gh": "github",
        "x.com": "twitter",
        "x": "twitter",
        "twitter.com": "twitter",
        "youtube.com": "youtube",
        "youtu.be": "youtube",
        "facebook.com": "facebook",
        "instagram.com": "instagram",
        "reddit.com": "reddit",
        "github.com": "github",
        "google.com": "google",
        "stackoverflow.com": "stackoverflow",
        "stack overflow": "stackoverflow",
    }
    if t in alias:
        return alias[t]
    # Strip leading www.
    if t.startswith("www."):
        t = t[4:]
    # Collapse common domain suffixes if the token looks like a bare domain
    for suf in (".com", ".org", ".net", ".io"):
        if t.endswith(suf):
            base = t[: -len(suf)]
            if base and " " not in base:  # avoid stripping from phrases
                t = base
                break
    # Final light cleanup
    t = t.replace("\u200b", "")
    return t


def build_bipartite(
    doc_ids: Sequence[str],
    ents_per_doc: Sequence[Sequence[Tuple[str, str]]],
    dataset_tag: str,
    collapse_by_text: bool = False,
    normalize_variants: bool = False,
) -> "networkx.Graph":  # type: ignore[name-defined]
    import networkx as nx

    G = nx.Graph()
    # Add nodes
    for did in doc_ids:
        G.add_node(f"doc::{dataset_tag}::{did}", bipartite=0, kind="doc", dataset=dataset_tag)

    # Entity nodes and edges
    for did, ents in zip(doc_ids, ents_per_doc):
        doc_node = f"doc::{dataset_tag}::{did}"
        if collapse_by_text:
            # Aggregate mentions per normalized text (across labels)
            agg: Dict[str, Dict[str, int]] = {}
            for (ent_text, ent_label) in ents:
                tnorm = normalize_entity_text(ent_text, do_normalize=normalize_variants)
                if not tnorm:
                    continue
                if tnorm not in agg:
                    agg[tnorm] = {}
                agg[tnorm][ent_label] = agg[tnorm].get(ent_label, 0) + 1
            for ent_text, label_counts in agg.items():
                ent_node = f"ent::{ent_text}"
                if not G.has_node(ent_node):
                    G.add_node(
                        ent_node,
                        bipartite=1,
                        kind="entity",
                        label=",".join(sorted(label_counts.keys())),
                        text=ent_text,
                        labels=list(sorted(label_counts.keys())),
                    )
                if G.has_edge(doc_node, ent_node):
                    G[doc_node][ent_node]["weight"] += int(sum(label_counts.values()))
                else:
                    G.add_edge(doc_node, ent_node, weight=int(sum(label_counts.values())))
        else:
            counts = collections.Counter(ents)
            for (ent_text, ent_label), w in counts.items():
                tnorm = normalize_entity_text(ent_text, do_normalize=normalize_variants)
                if not tnorm:
                    continue
                ent_node = f"ent::{tnorm}::{ent_label}"
                if not G.has_node(ent_node):
                    G.add_node(ent_node, bipartite=1, kind="entity", label=ent_label, text=tnorm)
                if G.has_edge(doc_node, ent_node):
                    G[doc_node][ent_node]["weight"] += int(w)
                else:
                    G.add_edge(doc_node, ent_node, weight=int(w))
    return G


def entity_counts(
    ents_per_doc: Sequence[Sequence[Tuple[str, str]]],
    collapse_by_text: bool = False,
    normalize_variants: bool = False,
) -> Tuple[collections.Counter, collections.Counter]:
    """Return (entity_text counts, entity_label counts).

    - If collapse_by_text: counts are aggregated by normalized text across labels.
    - Label counts are returned as counts per label; text normalization has no effect.
    """
    text_counts: collections.Counter = collections.Counter()
    label_counts: collections.Counter = collections.Counter()
    for ents in ents_per_doc:
        for text, label in ents:
            tnorm = normalize_entity_text(text, do_normalize=normalize_variants)
            if tnorm:
                text_counts[tnorm] += 1
            label_counts[label] += 1
    # If not collapsing, this is already by normalized text; collapsing across labels is implicit here
    # because we keyed by normalized text only.
    return text_counts, label_counts


def jensen_shannon(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Jensen–Shannon distance between two discrete distributions (keys union)."""
    keys = set(p) | set(q)
    if not keys:
        return 0.0
    p_vec = np.array([p.get(k, 0.0) for k in keys], dtype=float)
    q_vec = np.array([q.get(k, 0.0) for k in keys], dtype=float)
    p_vec = p_vec / (p_vec.sum() or 1.0)
    q_vec = q_vec / (q_vec.sum() or 1.0)
    m = 0.5 * (p_vec + q_vec)

    def _kl(a, b):
        a = np.where(a == 0, 1e-12, a)
        b = np.where(b == 0, 1e-12, b)
        return float(np.sum(a * np.log(a / b)))

    jsd = 0.5 * _kl(p_vec, m) + 0.5 * _kl(q_vec, m)
    return float(math.sqrt(max(jsd, 0.0)))


def to_dist(c: collections.Counter) -> Dict[str, float]:
    total = sum(c.values())
    return {k: v / total for k, v in c.items()} if total else {}


def write_graph(G, out_path: Path, fmt: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = fmt.lower()
    if fmt == "gexf":
        import networkx as nx

        nx.write_gexf(G, out_path)
    elif fmt == "graphml":
        import networkx as nx

        nx.write_graphml(G, out_path)
    elif fmt == "csv":
        # Write edgelist CSV with weights
        import networkx as nx

        rows = []
        for u, v, d in G.edges(data=True):
            rows.append({"source": u, "target": v, "weight": int(d.get("weight", 1))})
        pd.DataFrame(rows).to_csv(out_path, index=False)
    else:
        raise ValueError(f"Unsupported graph format: {fmt}")


def plot_bipartite(G, out_path: Path, title: str, max_docs: int = 50, max_entities: int = 30) -> None:
    """Plot a sampled doc–entity bipartite graph.

    - Samples up to max_docs document nodes and selects connected top entities.
    - Places docs on x=0 and entities on x=1 for readability.
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
        import numpy as np

        # Separate nodes by type
        docs = [n for n, d in G.nodes(data=True) if d.get("kind") == "doc"]
        ents = [n for n, d in G.nodes(data=True) if d.get("kind") == "entity"]
        if not docs or not ents:
            return

        # Sample docs uniformly
        if len(docs) > max_docs:
            rng = np.random.default_rng(42)
            docs = list(rng.choice(docs, size=max_docs, replace=False))

        # Determine top entities by degree among those connected to sampled docs
        ent_scores = {}
        for d in docs:
            for _, e, w in G.edges(d, data=True):
                ent_scores[e] = ent_scores.get(e, 0) + int(w.get("weight", 1))
        top_ents = sorted(ent_scores.items(), key=lambda x: -x[1])
        ents_sel = [e for e, _ in top_ents[:max_entities]]
        nodes_keep = set(docs) | set(ents_sel)
        H = G.subgraph(nodes_keep).copy()

        # Layout: docs on left, entities on right
        y_docs = np.linspace(0, 1, num=len(docs), endpoint=True)
        y_ents = np.linspace(0, 1, num=len(ents_sel), endpoint=True)
        pos = {}
        for i, d in enumerate(docs):
            pos[d] = (0.0, float(y_docs[i]))
        for i, e in enumerate(ents_sel):
            pos[e] = (1.0, float(y_ents[i]))

        plt.figure(figsize=(10, 8))
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        nx.draw_networkx_edges(H, pos, alpha=0.3)
        nx.draw_networkx_nodes(H, pos, nodelist=docs, node_color="#1f77b4", node_size=50, label="Docs")
        nx.draw_networkx_nodes(H, pos, nodelist=ents_sel, node_color="#ff7f0e", node_size=80, label="Entities")

        # Entity labels (use entity text attribute)
        ent_labels = {e: H.nodes[e].get("text", "") for e in ents_sel}
        nx.draw_networkx_labels(H, pos, labels=ent_labels, font_size=7, verticalalignment="center", horizontalalignment="left", bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=0.2))
        plt.axis("off")
        plt.title(title)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Warning: failed to plot bipartite graph to {out_path}: {e}", file=sys.stderr)


def plot_entity_compare(
    sim_counts: collections.Counter,
    voat_counts: collections.Counter,
    out_path: Path,
    title: str,
    top_k: int = 30,
) -> None:
    """Side-by-side bar chart comparing top-K entities in Simulation vs Voat using relative frequencies (percent).

    Bars show percentage of total entity mentions in each corpus, making sizes comparable.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        # Build dataframes of raw counts
        sim_df = pd.DataFrame(sorted(sim_counts.items(), key=lambda x: -x[1])[:top_k], columns=["entity", "sim"])
        voat_df = pd.DataFrame(sorted(voat_counts.items(), key=lambda x: -x[1])[:top_k], columns=["entity", "voat"])
        merged = pd.merge(sim_df, voat_df, on="entity", how="outer").fillna(0)
        # Determine ordering by combined raw counts, then convert to percentages
        merged["total_raw"] = merged["sim"] + merged["voat"]
        merged = merged.sort_values("total_raw", ascending=False).head(top_k)

        # Convert to relative frequencies (%) per corpus
        tot_sim = float(sum(sim_counts.values()) or 1.0)
        tot_voat = float(sum(voat_counts.values()) or 1.0)
        merged["sim_pct"] = merged["sim"].astype(float) / tot_sim * 100.0
        merged["voat_pct"] = merged["voat"].astype(float) / tot_voat * 100.0

        x = np.arange(len(merged))
        width = 0.45
        plt.figure(figsize=(min(14, 0.4 * len(merged) + 6), 6))
        plt.bar(x - width / 2, merged["sim_pct"], width, label="Simulation", color="#1f77b4")
        plt.bar(x + width / 2, merged["voat_pct"], width, label="Voat", color="#ff7f0e")
        plt.xticks(x, merged["entity"], rotation=60, ha="right")
        plt.ylabel("Percentage of entity mentions (%)")
        plt.title(title)
        plt.legend()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Warning: failed to plot entity comparison {out_path}: {e}", file=sys.stderr)


def build_entity_cooccurrence(
    ents_per_doc: Sequence[Sequence[Tuple[str, str]]],
    collapse_by_text: bool = False,
    normalize_variants: bool = False,
) -> "networkx.Graph":  # type: ignore[name-defined]
    """Build entity–entity co-occurrence graph: connect entities that appear in the same document.

    Edge weights count the number of documents in which a pair co-occurs (unique per doc).
    """
    import networkx as nx
    from itertools import combinations

    G = nx.Graph()
    for ents in ents_per_doc:
        # Unique entities per doc to avoid multiple counts in same doc
        if collapse_by_text:
            texts = set()
            labels_map: Dict[str, set] = {}
            for (text, label) in ents:
                tnorm = normalize_entity_text(text, do_normalize=normalize_variants)
                if not tnorm:
                    continue
                texts.add(tnorm)
                labels_map.setdefault(tnorm, set()).add(label)
            uniq = sorted(texts)
            for t in uniq:
                node = f"ent::{t}"
                if not G.has_node(node):
                    G.add_node(node, kind="entity", text=t, labels=sorted(labels_map.get(t, set())))
            for (t1, t2) in combinations(uniq, 2):
                n1 = f"ent::{t1}"
                n2 = f"ent::{t2}"
                if G.has_edge(n1, n2):
                    G[n1][n2]["weight"] += 1
                else:
                    G.add_edge(n1, n2, weight=1)
        else:
            uniq = list(set((normalize_entity_text(t, do_normalize=normalize_variants), l) for (t, l) in ents))
            for (text, label) in uniq:
                node = f"ent::{text}::{label}"
                if not G.has_node(node):
                    G.add_node(node, kind="entity", label=label, text=text)
            for (e1_text, e1_label), (e2_text, e2_label) in combinations(uniq, 2):
                n1 = f"ent::{e1_text}::{e1_label}"
                n2 = f"ent::{e2_text}::{e2_label}"
                if G.has_edge(n1, n2):
                    G[n1][n2]["weight"] += 1
                else:
                    G.add_edge(n1, n2, weight=1)
    return G


def plot_cooccurrence(G, out_path: Path, title: str, max_nodes: int = 200) -> None:
    """Plot entity co-occurrence graph focusing on readability.

    - Uses only the largest connected component (LCC).
    - Draws all nodes the same (small) size; labels only high-degree nodes.
    - Limits to `max_nodes` by keeping top-degree nodes within the LCC.
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
        import numpy as np

        if G.number_of_nodes() == 0:
            return

        # Largest connected component only
        if nx.number_connected_components(G) > 1:
            lcc_nodes = max(nx.connected_components(G), key=len)
            H = G.subgraph(lcc_nodes).copy()
        else:
            H = G.copy()

        # Downsample by top-degree nodes if too large
        deg = dict(H.degree())
        nodes_sorted = sorted(deg.items(), key=lambda x: -x[1])
        if H.number_of_nodes() > max_nodes:
            keep_nodes = [n for n, _ in nodes_sorted[:max_nodes]]
            H = H.subgraph(keep_nodes).copy()
            # Ensure still connected
            if hasattr(nx, "number_connected_components") and nx.number_connected_components(H) > 1:
                lcc_nodes = max(nx.connected_components(H), key=len)
                H = H.subgraph(lcc_nodes).copy()

        # Layout (force-directed)
        pos = nx.spring_layout(H, seed=42, iterations=200)

        # Styling: small equal-size nodes; light edges
        node_size = 30
        edge_alpha = 0.12
        edge_color = "#aaaaaa"
        node_color = "#1f77b4"

        plt.figure(figsize=(10, 8))
        nx.draw_networkx_edges(H, pos, alpha=edge_alpha, edge_color=edge_color, width=0.5)
        nx.draw_networkx_nodes(H, pos, node_color=node_color, node_size=node_size)

        # Labels for high-degree nodes only (top 25 or all if fewer)
        deg_H = dict(H.degree())
        top_nodes = sorted(deg_H.items(), key=lambda x: -x[1])[: min(25, len(deg_H))]
        label_map = {n: H.nodes[n].get("text", "") for n, _ in top_nodes}
        nx.draw_networkx_labels(
            H,
            pos,
            labels=label_map,
            font_size=7,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=0.2),
        )
        plt.axis("off")
        plt.title(title)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Warning: failed to plot co-occurrence graph to {out_path}: {e}", file=sys.stderr)


def run_subset(
    subset_label: str,
    sim_df: pd.DataFrame,
    voat_df: pd.DataFrame,
    nlp,
    entity_types: Optional[List[str]],
    lowercase: bool,
    min_len: int,
    max_items: Optional[int],
    outdir: Path,
    graph_fmt: str,
    do_plot_bipartite: bool,
    do_plot_compare: bool,
    topk_entities_plot: int,
    bipartite_max_docs: int,
    bipartite_max_entities: int,
    collapse_by_text: bool,
    normalize_variants: bool,
) -> Tuple[Dict[str, object], List[List[Tuple[str, str]]], List[List[Tuple[str, str]]]]:
    # Optional sampling to control graph size
    if max_items and len(sim_df) > max_items:
        sim_df = sim_df.sample(n=max_items, random_state=42)
    if max_items and len(voat_df) > max_items:
        voat_df = voat_df.sample(n=max_items, random_state=42)

    # Extract entities
    sim_ents = extract_entities(
        nlp,
        sim_df["text"].astype(str).tolist(),
        entity_types=entity_types,
        lowercase=lowercase,
        min_len=min_len,
    )
    voat_ents = extract_entities(
        nlp,
        voat_df["text"].astype(str).tolist(),
        entity_types=entity_types,
        lowercase=lowercase,
        min_len=min_len,
    )

    # Graphs
    import networkx as nx  # noqa: F401 (used by writer)

    sim_ids = [f"sim_{i}_{row['id']}" for i, row in sim_df.reset_index().iterrows()]
    voat_ids = [f"voat_{i}_{row['post_id']}" for i, row in voat_df.reset_index().iterrows()]
    G_sim = build_bipartite(
        sim_ids,
        sim_ents,
        dataset_tag="simulation",
        collapse_by_text=collapse_by_text,
        normalize_variants=normalize_variants,
    )
    G_voat = build_bipartite(
        voat_ids,
        voat_ents,
        dataset_tag="voat",
        collapse_by_text=collapse_by_text,
        normalize_variants=normalize_variants,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    write_graph(G_sim, outdir / f"{subset_label}_simulation_bipartite.{graph_fmt}", graph_fmt)
    write_graph(G_voat, outdir / f"{subset_label}_voat_bipartite.{graph_fmt}", graph_fmt)

    # Plots
    if do_plot_bipartite:
        plot_bipartite(
            G_sim,
            outdir / f"{subset_label}_simulation_bipartite.png",
            title=f"Simulation {subset_label}: Doc–Entity Bipartite",
            max_docs=bipartite_max_docs,
            max_entities=bipartite_max_entities,
        )
        plot_bipartite(
            G_voat,
            outdir / f"{subset_label}_voat_bipartite.png",
            title=f"Voat {subset_label}: Doc–Entity Bipartite",
            max_docs=bipartite_max_docs,
            max_entities=bipartite_max_entities,
        )

    # Distributions
    sim_text_counts, sim_label_counts = entity_counts(
        sim_ents, collapse_by_text=collapse_by_text, normalize_variants=normalize_variants
    )
    voat_text_counts, voat_label_counts = entity_counts(
        voat_ents, collapse_by_text=collapse_by_text, normalize_variants=normalize_variants
    )

    # Convert to DataFrames
    def _to_df(cnt: collections.Counter, kcol: str, vcol: str) -> pd.DataFrame:
        return pd.DataFrame(sorted(cnt.items(), key=lambda x: (-x[1], x[0])), columns=[kcol, vcol])

    df_sim_text = _to_df(sim_text_counts, "entity", "count")
    df_voat_text = _to_df(voat_text_counts, "entity", "count")
    df_sim_label = _to_df(sim_label_counts, "label", "count")
    df_voat_label = _to_df(voat_label_counts, "label", "count")

    df_sim_text.to_csv(outdir / f"{subset_label}_simulation_entity_counts.csv", index=False)
    df_voat_text.to_csv(outdir / f"{subset_label}_voat_entity_counts.csv", index=False)
    df_sim_label.to_csv(outdir / f"{subset_label}_simulation_label_counts.csv", index=False)
    df_voat_label.to_csv(outdir / f"{subset_label}_voat_label_counts.csv", index=False)

    # Comparative plots
    if do_plot_compare:
        plot_entity_compare(
            sim_text_counts,
            voat_text_counts,
            out_path=outdir / f"{subset_label}_entity_counts_compare_top{topk_entities_plot}.png",
            title=f"Top entities in Simulation vs Voat ({subset_label})",
            top_k=topk_entities_plot,
        )

    # Overlaps and divergences
    # Label distribution JS distance
    js_label = jensen_shannon(to_dist(sim_label_counts), to_dist(voat_label_counts))

    # Entity set overlap (use top-K for stability)
    K = 200
    sim_top = set(df_sim_text.head(K)["entity"].tolist())
    voat_top = set(df_voat_text.head(K)["entity"].tolist())
    jacc_top = float(len(sim_top & voat_top)) / float(len(sim_top | voat_top) or 1)

    summary = {
        "subset": subset_label,
        "n_simulation_docs": int(len(sim_df)),
        "n_voat_docs": int(len(voat_df)),
        "n_simulation_entities": int(sum(sim_text_counts.values())),
        "n_voat_entities": int(sum(voat_text_counts.values())),
        "label_js_distance": js_label,
        "top_entity_jaccard@200": jacc_top,
        "top_simulation_entities": df_sim_text.head(25).to_dict(orient="records"),
        "top_voat_entities": df_voat_text.head(25).to_dict(orient="records"),
        "label_counts_simulation": df_sim_label.to_dict(orient="records"),
        "label_counts_voat": df_voat_label.to_dict(orient="records"),
    }

    (outdir / f"{subset_label}_summary.json").write_text(json.dumps(summary, indent=2))
    return summary, sim_ents, voat_ents


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="NER bipartite graphs and distributional comparisons: simulation2 vs Voat parquet")
    p.add_argument("--sim2-posts", type=Path, default=Path("simulation2/posts.csv"))
    p.add_argument("--sim2-tox", type=Path, default=Path("simulation2/toxigen.csv"))
    p.add_argument("--madoc-parquet", type=Path, default=Path("MADOC/voat-technology/voat_technology_madoc.parquet"))
    p.add_argument("--mode", type=str, choices=["both", "posts", "comments"], default="both")
    p.add_argument("--spacy-model", type=str, default="en_core_web_sm")
    p.add_argument("--spacy-max-length", type=int, default=2_000_000)
    p.add_argument("--entity-types", type=str, default="PERSON,ORG,GPE,LOC,NORP,PRODUCT,EVENT,WORK_OF_ART,LAW,LANGUAGE")
    p.add_argument("--no-lowercase", action="store_true", help="Do not lowercase entity surface forms")
    p.add_argument("--min-entity-len", type=int, default=2)
    p.add_argument("--max-items", type=int, default=5000, help="Max docs per set for graphs; 0 = no cap")
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--graph-format", type=str, choices=["gexf", "graphml", "csv"], default="gexf")
    # Plotting options
    p.add_argument("--plot-bipartite", action="store_true", help="Render sampled doc–entity bipartite PNGs")
    p.add_argument("--plot-compare", action="store_true", help="Render bar chart comparing top entities in Simulation vs Voat")
    p.add_argument("--topk-entities-plot", type=int, default=30)
    p.add_argument("--bipartite-max-docs", type=int, default=50)
    p.add_argument("--bipartite-max-entities", type=int, default=30)
    # Collapsing/normalization
    p.add_argument("--collapse-by-text", action="store_true", help="Collapse entities by normalized surface (ignore label differences)")
    p.add_argument("--normalize-variants", action="store_true", help="Normalize common surface variants (e.g., youtube.com -> youtube, fb -> facebook)")
    # Co-occurrence networks
    p.add_argument("--build-cooc", action="store_true", help="Build unified entity co-occurrence networks (Simulation and Voat)")
    p.add_argument("--plot-cooc", action="store_true", help="Render co-occurrence PNGs")
    p.add_argument("--cooc-max-nodes", type=int, default=200)
    args = p.parse_args(argv)

    # Determine default outdir if not provided: place under simulation posts folder
    if args.outdir is None:
        try:
            default_outdir = args.sim2_posts.parent / "ner"
        except Exception:
            default_outdir = Path(".") / "ner"
        args.outdir = default_outdir

    nlp = load_spacy(args.spacy_model, max_length=args.spacy_max_length)
    ent_types = [s.strip().upper() for s in args.entity_types.split(",") if s.strip()]
    lowercase = not args.no_lowercase
    max_items = None if (args.max_items is None or args.max_items <= 0) else int(args.max_items)

    results: List[Dict[str, object]] = []
    sim_ents_all: List[List[Tuple[str, str]]] = []
    voat_ents_all: List[List[Tuple[str, str]]] = []
    if args.mode in {"both", "comments"}:
        sim_comments = load_sim2_subset(args.sim2_posts, args.sim2_tox, want_comments=True)
        voat_comments = load_voat_subset(args.madoc_parquet, want_comments=True)
        if sim_comments.empty or voat_comments.empty:
            print("Warning: comments subset empty; skipping.", file=sys.stderr)
        else:
            res, sim_ents, voat_ents = run_subset(
                subset_label="comments",
                sim_df=sim_comments,
                voat_df=voat_comments,
                nlp=nlp,
                entity_types=ent_types,
                lowercase=lowercase,
                min_len=args.min_entity_len,
                max_items=max_items,
                outdir=args.outdir,
                graph_fmt=args.graph_format,
                do_plot_bipartite=args.plot_bipartite,
                do_plot_compare=args.plot_compare,
                topk_entities_plot=args.topk_entities_plot,
                bipartite_max_docs=args.bipartite_max_docs,
                bipartite_max_entities=args.bipartite_max_entities,
                collapse_by_text=args.collapse_by_text,
                normalize_variants=args.normalize_variants,
            )
            results.append(res)
            sim_ents_all.extend(sim_ents)
            voat_ents_all.extend(voat_ents)

    if args.mode in {"both", "posts"}:
        sim_posts = load_sim2_subset(args.sim2_posts, args.sim2_tox, want_comments=False)
        voat_posts = load_voat_subset(args.madoc_parquet, want_comments=False)
        if sim_posts.empty or voat_posts.empty:
            print("Warning: posts subset empty; skipping.", file=sys.stderr)
        else:
            res, sim_ents, voat_ents = run_subset(
                subset_label="posts",
                sim_df=sim_posts,
                voat_df=voat_posts,
                nlp=nlp,
                entity_types=ent_types,
                lowercase=lowercase,
                min_len=args.min_entity_len,
                max_items=max_items,
                outdir=args.outdir,
                graph_fmt=args.graph_format,
                do_plot_bipartite=args.plot_bipartite,
                do_plot_compare=args.plot_compare,
                topk_entities_plot=args.topk_entities_plot,
                bipartite_max_docs=args.bipartite_max_docs,
                bipartite_max_entities=args.bipartite_max_entities,
                collapse_by_text=args.collapse_by_text,
                normalize_variants=args.normalize_variants,
            )
            results.append(res)
            sim_ents_all.extend(sim_ents)
            voat_ents_all.extend(voat_ents)

    # Build unified co-occurrence networks across posts+comments per dataset
    if args.build_cooc and (sim_ents_all or voat_ents_all):
        args.outdir.mkdir(parents=True, exist_ok=True)
        if sim_ents_all:
            Gs = build_entity_cooccurrence(
                sim_ents_all,
                collapse_by_text=args.collapse_by_text,
                normalize_variants=args.normalize_variants,
            )
            write_graph(Gs, args.outdir / f"unified_simulation_entity_cooc.{args.graph_format}", args.graph_format)
            if args.plot_cooc:
                plot_cooccurrence(Gs, args.outdir / "unified_simulation_entity_cooc.png", title="Simulation (posts+comments): Entity Co-occurrence", max_nodes=args.cooc_max_nodes)
        if voat_ents_all:
            Gv = build_entity_cooccurrence(
                voat_ents_all,
                collapse_by_text=args.collapse_by_text,
                normalize_variants=args.normalize_variants,
            )
            write_graph(Gv, args.outdir / f"unified_voat_entity_cooc.{args.graph_format}", args.graph_format)
            if args.plot_cooc:
                plot_cooccurrence(Gv, args.outdir / "unified_voat_entity_cooc.png", title="Voat (posts+comments): Entity Co-occurrence", max_nodes=args.cooc_max_nodes)

    # Write index summary
    if results:
        index = {
            "spacy_model": args.spacy_model,
            "madoc_parquet": str(args.madoc_parquet),
            "mode": args.mode,
            "entity_types": ent_types,
            "results": results,
        }
        args.outdir.mkdir(parents=True, exist_ok=True)
        (args.outdir / "index.json").write_text(json.dumps(index, indent=2))
        print(f"Wrote NER outputs to {args.outdir}")
        return 0
    else:
        print("No results produced.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
