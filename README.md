# Graph-Based Crime Network Analysis

Graph analytics on a real criminal-association network: key-actor detection,
community structure, network-disruption simulation, and robustness to missing data.
Extends an earlier PySpark/GraphFrames prototype (Big Data Analytics course project)
from a 7-node simulated graph to real police-record data.

## Data
**St. Louis Homicide Project crime network ("Rosenfeld crime", 1991)** - associations
among suspects, victims and witnesses, derived from police records. Two-mode:
829 people and 551 crime events, 1,476 person-event links.

- Source: Netzschleuder catalogue (`crime`), originally from KONECT (`moreno_crime`)
- Original study: S. Decker, C. W. Kohfeld, R. Rosenfeld, J. Sprague,
  *St. Louis Homicide Project: Local Responses to a National Problem*,
  University of Missouri-St. Louis (1991)
- Check the dataset's license/terms on the source pages before redistributing `data/`.

The two-mode data is projected onto people: two people are linked if they were involved
in the same crime event (829 nodes, 2,253 ties).

## Run
```bash
pip install networkx pandas numpy scipy matplotlib scikit-learn
python crime_network_analysis.py --edges data/edges.csv --bipartite --project larger --out results
```
`--project larger` keeps the larger side of the bipartite graph (the 829 people).
All numbers below come from `results/results.json`.

## Results
| Measure | Value |
|---|---|
| Nodes / edges (person network) | 829 / 2,253 |
| Connected components | 20 (largest group: 754 people, 91%) |
| Diameter / avg path length (largest group) | 16 / 6.8 |
| Louvain communities | 45 (modularity 0.88) |
| Label propagation communities | 148 (modularity 0.77) |
| Top-20 overlap, PageRank vs betweenness | 40% |

**Network disruption** - share of the network still in one connected group after removing
the top 10% of actors (82 people):

| Removed by | Largest group remaining |
|---|---|
| (none) | 91% |
| PageRank | 5% |
| Betweenness | 6% |
| Degree | 6% |
| Random (mean of 20 runs) | 71% |

**Robustness** - with 10% of ties removed at random, 90% of the top-20 PageRank list is
unchanged (min 85% over 10 trials).

## Link prediction (recovering missing ties)
`link_prediction.py` hides 20% of ties and tests whether a model can recover them
(features computed only on the observed graph, no leakage; 5 random splits, mean +/- std).
Results are in `results/link_prediction_results.json`.

| Model | ROC-AUC (all pairs) | ROC-AUC (pairs with no shared contacts) |
|---|---|---|
| Adamic-Adar heuristic (baseline) | 0.934 | 0.500 |
| Logistic Regression | 0.981 | 0.871 |
| Random Forest | 0.971 | 0.797 |
| Gradient Boosting | 0.975 | 0.829 |

About 56% of held-out ties join people with no shared contacts, where neighbourhood
heuristics have no signal. The simplest model (Logistic Regression) performed best on this
small dataset. Easy ties are inflated by the clique structure created by projecting
two-mode data.

Run: `pip install scikit-learn` then
`python link_prediction.py --edges data/edges.csv --bipartite --project larger --out results`

## Notes and limitations
- Small, historical, snowball-sampled data (from five initial homicides): results describe
  this dataset, not crime networks in general.
- Projecting two-mode data creates cliques (everyone at one crime is linked), which inflates
  clustering (0.72) and the k-core (17). Interpret those two with care.
- Betweenness is exact at this size; for larger graphs the script samples.

## Optional: scalability benchmark
`spark_scale_test.py` times PageRank, label propagation, triangle count and connected
components with PySpark/GraphFrames on a large public edge list (benchmark only, not crime data).

## Files
- `crime_network_analysis.py` - analysis pipeline
- `link_prediction.py` - missing-tie prediction experiment
- `spark_scale_test.py` - Spark benchmark
- `data/` - edge list, node metadata, dataset properties
- `results/` - `results.json`, `communities.png`
