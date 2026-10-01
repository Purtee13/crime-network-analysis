"""
PySpark + GraphFrames scalability benchmark.

Run this in Google Colab (or any machine with Spark) on a LARGE public edge list
(e.g. a SNAP graph) to show the pipeline scales. This is a SCALABILITY benchmark,
not crime data: say so on your resume/README.

Colab setup (first cell):
    !pip install -q pyspark==3.5.1 graphframes-py
    # then start Spark with the matching GraphFrames package:
    #   .config("spark.jars.packages", "io.graphframes:graphframes-spark3_2.12:0.9.3")
    # (check the GraphFrames docs for the package that matches your Spark version)

Usage:
    python spark_scale_test.py --edges big_edges.txt --sep " "
"""
import argparse
import json
import time

from pyspark.sql import SparkSession
from graphframes import GraphFrame


def timed(label, fn, log):
    t0 = time.time()
    out = fn()
    log[label] = round(time.time() - t0, 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", required=True)
    ap.add_argument("--sep", default="\t")
    ap.add_argument("--out", default="spark_results.json")
    args = ap.parse_args()

    spark = (
        SparkSession.builder.appName("Crime-Graph-Scale-Test")
        .config("spark.jars.packages", "io.graphframes:graphframes-spark3_2.12:0.9.3")
        .config("spark.sql.shuffle.partitions", "64")
        .getOrCreate()
    )
    spark.sparkContext.setCheckpointDir("/tmp/graphframes_ckpt")

    edges = (
        spark.read.option("comment", "#").option("sep", args.sep)
        .csv(args.edges).toDF("src", "dst").dropna().distinct()
    )
    verts = edges.select("src").union(edges.select("dst")).distinct().withColumnRenamed("src", "id")
    g = GraphFrame(verts, edges)

    log = {"nodes": verts.count(), "edges": edges.count()}
    timed("degrees_s", lambda: g.degrees.count(), log)
    timed("pagerank_s", lambda: g.pageRank(resetProbability=0.15, maxIter=10).vertices.count(), log)
    timed("label_propagation_s", lambda: g.labelPropagation(maxIter=5).count(), log)
    timed("triangle_count_s", lambda: g.triangleCount().count(), log)
    timed("connected_components_s", lambda: g.connectedComponents().count(), log)

    print(json.dumps(log, indent=2))
    with open(args.out, "w") as f:
        json.dump(log, f, indent=2)


if __name__ == "__main__":
    main()
