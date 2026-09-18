from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import osmnx as ox
import pandas as pd

from score_gwanghwamun_edges import compute_vibe_scores, fetch_vibe_pois


GRAPH_PATH = Path("osmnx_outputs/gwanghwamun_walk_2km_scored.graphml")
SUMMARY_PATH = Path("osmnx_outputs/gwanghwamun_vibe_score_summary.csv")


def main() -> None:
    """Add/refresh vibe_score on the scored graph without touching the other axes."""
    sys.stdout.reconfigure(encoding="utf-8")
    graph_wgs84 = ox.load_graphml(GRAPH_PATH)
    graph = ox.project_graph(graph_wgs84)
    pois = fetch_vibe_pois(graph.graph["crs"])
    print(f"Vibe POIs in bbox: {len(pois):,}")
    scores = compute_vibe_scores(graph, pois)
    for (u, v, key), values in scores.items():
        graph_wgs84[u][v][key].update(values)
    ox.save_graphml(graph_wgs84, GRAPH_PATH)

    frame = pd.DataFrame.from_dict(scores, orient="index")
    frame["highway"] = [str(graph_wgs84[u][v][k].get("highway")) for u, v, k in frame.index]
    quantiles = frame[["vibe_poi_count", "vibe_score"]].quantile([0.5, 0.75, 0.9, 0.95, 0.99, 1.0]).T
    quantiles["edges_gt0_pct"] = (frame[["vibe_poi_count", "vibe_score"]] > 0).mean() * 100
    quantiles.to_csv(SUMMARY_PATH, encoding="utf-8-sig")
    print(quantiles.round(3).to_string())
    print(f"living_street edges: {(frame['highway'] == 'living_street').sum():,}")
    print(f"Saved {GRAPH_PATH} (+vibe_poi_count, vibe_score) and {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
