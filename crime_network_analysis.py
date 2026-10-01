"""
Graph-Based Crime Network Analysis (scaled version)

Runs a full graph analytics pipeline on a REAL crime network edge list and writes
all measured numbers to results.json so the resume can quote them exactly.

Input: a CSV edge list with two columns (source,target). Optional --bipartite flag
if the file links people to crime events (e.g. the St. Louis "Rosenfeld crime"
network); the script then projects it onto a person-person graph.

Usage:
    python crime_network_analysis.py --edges edges.csv --out results
    python crime_network_analysis.py --edges edges.csv --bipartite --out results
"""
import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms import bipartite
from networkx.algorithms.community import (
    asyn_lpa_communities,
    louvain_communities,
    modularity,
)


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_graph(path, is_bipartite, project="denser"):
    # header=None + comment="#": handles Netzschleuder-style "# source,target" headers
    # (skipped as a comment) as well as plain "source,target" headers (dropped below).
    df = pd.read_csv(path, comment="#", header=None, sep=None, engine="python")
    df = df.iloc[:, :2]
    df.columns = ["src", "dst"]
    df = df.astype(str)
    if df.iloc[0].str.lower().tolist() in (["source", "target"], ["src", "dst"], ["from", "to"]):
        df = df.iloc[1:]
    g = nx.from_pandas_edgelist(df, "src", "dst")
    g.remove_edges_from(nx.selfloop_edges(g))
    info = {"raw_nodes": g.number_of_nodes(), "raw_edges": g.number_of_edges()}

    if is_bipartite:
        # Two-mode (person - crime event) -> one-mode (person - person)
        color = bipartite.color(g)  # works on disconnected graphs
        left = {n for n, c in color.items() if c == 0}
        right = {n for n, c in color.items() if c == 1}
        # Project onto whichever side yields the denser person-person graph
        # (people are linked through shared crime events).
        p_left = bipartite.projected_graph(g, left)
        p_right = bipartite.projected_graph(g, right)
        if project == "larger":      # e.g. people (829) outnumber crime events (551)
            proj = p_left if len(left) >= len(right) else p_right
        elif project == "smaller":
            proj = p_left if len(left) < len(right) else p_right
        else:                        # "denser" (default)
            proj = p_left if p_left.number_of_edges() >= p_right.number_of_edges() else p_right
        info["projection_rule"] = project
        info["projected_from_bipartite"] = True
        info["bipartite_sides"] = [len(left), len(right)]
        g = proj
    return g, info


# ----------------------------------------------------------------------------
# Analysis pieces
# ----------------------------------------------------------------------------
def basic_stats(g):
    comps = list(nx.connected_components(g))
    giant = g.subgraph(max(comps, key=len))
    stats = {
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "density": nx.density(g),
        "avg_degree": 2 * g.number_of_edges() / g.number_of_nodes(),
        "connected_components": len(comps),
        "giant_component_nodes": giant.number_of_nodes(),
        "giant_component_share": giant.number_of_nodes() / g.number_of_nodes(),
        "avg_clustering": nx.average_clustering(g),
        "triangles_total": sum(nx.triangles(g).values()) // 3,
        "transitivity": nx.transitivity(g),
    }
    if giant.number_of_nodes() <= 5000:
        stats["giant_diameter"] = nx.diameter(giant)
        stats["giant_avg_path_length"] = nx.average_shortest_path_length(giant)
    return stats


def centralities(g, k_sample=None):
    t0 = time.time()
    pr = nx.pagerank(g, alpha=0.85)
    t_pr = time.time() - t0
    t0 = time.time()
    # Exact betweenness is O(VE); sample for big graphs.
    if k_sample and g.number_of_nodes() > k_sample:
        bc = nx.betweenness_centrality(g, k=k_sample, seed=42)
    else:
        bc = nx.betweenness_centrality(g)
    t_bc = time.time() - t0
    deg = dict(g.degree())
    core = nx.core_number(g)  # true k-core decomposition
    return pr, bc, deg, core, {"pagerank_s": t_pr, "betweenness_s": t_bc}


def communities(g):
    out = {}
    t0 = time.time()
    lou = louvain_communities(g, seed=42)
    out["louvain"] = {
        "n_communities": len(lou),
        "modularity": modularity(g, lou),
        "largest": max(len(c) for c in lou),
        "seconds": time.time() - t0,
    }
    t0 = time.time()
    lpa = list(asyn_lpa_communities(g, seed=42))
    out["label_propagation"] = {
        "n_communities": len(lpa),
        "modularity": modularity(g, lpa),
        "largest": max(len(c) for c in lpa),
        "seconds": time.time() - t0,
    }
    return out, lou


def triad_census(g):
    # Directed census needs a DiGraph; for undirected data count open vs closed triads.
    deg = dict(g.degree())
    connected_triples = sum(d * (d - 1) // 2 for d in deg.values())
    closed = sum(nx.triangles(g).values()) // 3
    return {
        "connected_triples": connected_triples,
        "closed_triangles": closed,
        "open_triads": connected_triples - 3 * closed,
    }


def topk_overlap(a, b, k):
    ta = {n for n, _ in sorted(a.items(), key=lambda x: -x[1])[:k]}
    tb = {n for n, _ in sorted(b.items(), key=lambda x: -x[1])[:k]}
    return len(ta & tb) / k


def robustness(g, pr, bc, deg, frac=0.10, seed=42):
    """Network disruption: remove the top-X% of actors by each metric (and random)
    and measure how much of the giant component survives."""
    n = g.number_of_nodes()
    k = max(1, int(frac * n))
    rng = random.Random(seed)

    def giant_share(h):
        if h.number_of_nodes() == 0:
            return 0.0
        return len(max(nx.connected_components(h), key=len)) / n

    results = {"baseline_giant_share": giant_share(g), "removed_nodes": k}
    for name, score in [("pagerank", pr), ("betweenness", bc), ("degree", deg)]:
        top = [x for x, _ in sorted(score.items(), key=lambda x: -x[1])[:k]]
        h = g.copy()
        h.remove_nodes_from(top)
        results[f"after_removing_top_{name}"] = giant_share(h)
    rand_runs = []
    for _ in range(20):
        h = g.copy()
        h.remove_nodes_from(rng.sample(list(g.nodes()), k))
        rand_runs.append(giant_share(h))
    results["after_removing_random_mean"] = float(np.mean(rand_runs))
    return results


def stability(g, pr, k=20, drop=0.10, trials=10, seed=42):
    """How stable is the top-k PageRank list if 10% of edges are missing?
    (Criminal network data is incomplete by nature.)"""
    rng = random.Random(seed)
    edges = list(g.edges())
    overlaps = []
    for _ in range(trials):
        keep = rng.sample(edges, int((1 - drop) * len(edges)))
        h = nx.Graph()
        h.add_nodes_from(g.nodes())
        h.add_edges_from(keep)
        overlaps.append(topk_overlap(pr, nx.pagerank(h), k))
    return {"top_k": k, "edges_dropped": drop, "mean_overlap": float(np.mean(overlaps)),
            "min_overlap": float(np.min(overlaps))}


def draw(g, lou, path):
    giant = g.subgraph(max(nx.connected_components(g), key=len))
    if giant.number_of_nodes() > 1500:
        return
    label = {}
    for i, c in enumerate(lou):
        for n in c:
            label[n] = i
    colors = [label.get(n, 0) for n in giant.nodes()]
    plt.figure(figsize=(11, 8))
    pos = nx.spring_layout(giant, seed=42, k=0.15)
    nx.draw_networkx_nodes(giant, pos, node_size=18, node_color=colors, cmap=plt.cm.tab20)
    nx.draw_networkx_edges(giant, pos, alpha=0.2, width=0.5)
    plt.title("Crime network - Louvain communities (giant component)")
    plt.axis("off")
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", required=True)
    ap.add_argument("--bipartite", action="store_true")
    ap.add_argument("--project", choices=["denser", "larger", "smaller"], default="denser",
                    help="which side of a bipartite graph to keep (use 'larger' if people "
                         "outnumber events, 'smaller' if events outnumber people)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--bc-sample", type=int, default=2000,
                    help="sample size for approximate betweenness on big graphs")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)

    g, info = load_graph(args.edges, args.bipartite, args.project)
    res = {"input": info, "graph": basic_stats(g)}

    pr, bc, deg, core, timings = centralities(g, args.bc_sample)
    res["timings"] = timings
    res["max_core_number"] = max(core.values())
    res["nodes_in_max_core"] = sum(1 for v in core.values() if v == max(core.values()))
    res["top10_pagerank"] = sorted(pr.items(), key=lambda x: -x[1])[:10]
    res["top10_betweenness"] = sorted(bc.items(), key=lambda x: -x[1])[:10]
    res["pagerank_vs_betweenness_top20_overlap"] = topk_overlap(pr, bc, 20)

    comm, lou = communities(g)
    res["communities"] = comm
    res["triads"] = triad_census(g)
    res["robustness_10pct_removal"] = robustness(g, pr, bc, deg)
    res["pagerank_stability_10pct_missing_edges"] = stability(g, pr)

    draw(g, lou, out / "communities.png")
    (out / "results.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
