"""
Link prediction on the crime person-network (missing-ties problem).

Police data is incomplete by nature, so some real associations are never recorded.
This script hides a share of the known ties and tests whether a model can recover them.

Protocol (per random seed):
  1. Hold out 20% of ties as TEST positives -> observed graph G_obs.
  2. From G_obs hold out 20% of its ties as TRAIN positives -> feature graph G_feat.
  3. Train features are computed on G_feat, test features on G_obs, so no model ever
     sees a tie it is asked to predict (no leakage).
  4. Negatives = an equal number of random non-adjacent pairs.
  5. Compare a heuristic baseline (Adamic-Adar) with Logistic Regression,
     Random Forest and Gradient Boosting. Report mean +/- std over seeds.

Also reports a HARD subset: test pairs with no common neighbour in G_obs, where the
simple neighbourhood heuristics have no signal.

Usage:
    python link_prediction.py --edges data/edges.csv --bipartite --project larger --out results
"""
import argparse
import json
import random
from pathlib import Path

import networkx as nx
import numpy as np
from networkx.algorithms.community import louvain_communities
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from crime_network_analysis import load_graph

FEATURES = ["common_nbrs", "jaccard", "adamic_adar", "resource_alloc",
            "pref_attach", "same_community", "core_min", "core_max"]


def sample_negatives(g, n, rng, forbidden=None):
    nodes = list(g.nodes())
    out, seen = [], set()
    while len(out) < n:
        u, v = rng.sample(nodes, 2)
        key = (min(u, v), max(u, v))
        if g.has_edge(u, v) or key in seen or (forbidden and key in forbidden):
            continue
        seen.add(key)
        out.append(key)
    return out


def features(g, pairs):
    comm = {}
    for i, c in enumerate(louvain_communities(g, seed=0)):
        for node in c:
            comm[node] = i
    core = nx.core_number(g)
    deg = dict(g.degree())
    rows = []
    for u, v in pairs:
        nu, nv = set(g[u]), set(g[v])
        cn = nu & nv
        union = nu | nv
        aa = sum(1.0 / np.log(deg[w]) for w in cn if deg[w] > 1)
        ra = sum(1.0 / deg[w] for w in cn if deg[w] > 0)
        rows.append([
            len(cn),
            len(cn) / len(union) if union else 0.0,
            aa,
            ra,
            deg[u] * deg[v],
            float(comm[u] == comm[v]),
            min(core[u], core[v]),
            max(core[u], core[v]),
        ])
    return np.array(rows, dtype=float)


def precision_at_k(y, score, k):
    idx = np.argsort(-score)[:k]
    return float(np.mean(y[idx]))


def run_seed(g, seed):
    rng = random.Random(seed)
    edges = [tuple(sorted(e)) for e in g.edges()]
    rng.shuffle(edges)
    n_test = int(0.2 * len(edges))
    test_pos, obs_edges = edges[:n_test], edges[n_test:]

    g_obs = nx.Graph()
    g_obs.add_nodes_from(g.nodes())
    g_obs.add_edges_from(obs_edges)

    n_tr = int(0.2 * len(obs_edges))
    train_pos, feat_edges = obs_edges[:n_tr], obs_edges[n_tr:]
    g_feat = nx.Graph()
    g_feat.add_nodes_from(g.nodes())
    g_feat.add_edges_from(feat_edges)

    all_true = {tuple(sorted(e)) for e in g.edges()}
    train_neg = sample_negatives(g_feat, len(train_pos), rng, forbidden=all_true)
    test_neg = sample_negatives(g_obs, len(test_pos), rng, forbidden=all_true)

    Xtr = features(g_feat, train_pos + train_neg)
    ytr = np.array([1] * len(train_pos) + [0] * len(train_neg))
    Xte = features(g_obs, test_pos + test_neg)
    yte = np.array([1] * len(test_pos) + [0] * len(test_neg))

    hard = Xte[:, FEATURES.index("common_nbrs")] == 0

    models = {
        "Adamic-Adar (heuristic baseline)": None,
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=seed),
    }
    res = {}
    for name, model in models.items():
        if model is None:
            score = Xte[:, FEATURES.index("adamic_adar")]
        else:
            model.fit(Xtr, ytr)
            score = model.predict_proba(Xte)[:, 1]
        r = {
            "roc_auc": roc_auc_score(yte, score),
            "avg_precision": average_precision_score(yte, score),
            "precision_at_100": precision_at_k(yte, score, 100),
        }
        if hard.sum() > 20 and len(set(yte[hard])) == 2:
            r["roc_auc_hard"] = roc_auc_score(yte[hard], score[hard])
        res[name] = r
        if name == "Random Forest":
            res["_rf_importance"] = dict(zip(FEATURES, model.feature_importances_.tolist()))
    res["_hard_pairs"] = int(hard.sum())
    res["_test_pairs"] = int(len(yte))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", required=True)
    ap.add_argument("--bipartite", action="store_true")
    ap.add_argument("--project", default="larger")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    g, info = load_graph(args.edges, args.bipartite, args.project)
    runs = [run_seed(g, s) for s in range(args.seeds)]

    summary = {"graph": {"nodes": g.number_of_nodes(), "edges": g.number_of_edges()},
               "seeds": args.seeds, "models": {}}
    for name in [k for k in runs[0] if not k.startswith("_")]:
        summary["models"][name] = {}
        for metric in runs[0][name]:
            vals = [r[name][metric] for r in runs if metric in r[name]]
            summary["models"][name][metric] = {"mean": float(np.mean(vals)),
                                               "std": float(np.std(vals)),
                                               "n_runs": len(vals)}
    imp = np.mean([[r["_rf_importance"][f] for f in FEATURES] for r in runs], axis=0)
    summary["random_forest_feature_importance"] = dict(zip(FEATURES, map(float, imp)))
    summary["avg_hard_pairs_per_run"] = float(np.mean([r["_hard_pairs"] for r in runs]))
    summary["test_pairs_per_run"] = runs[0]["_test_pairs"]

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    (out / "link_prediction_results.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
