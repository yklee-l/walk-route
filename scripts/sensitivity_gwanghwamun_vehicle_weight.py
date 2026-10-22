from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

from score_gwanghwamun_edges import highway_values, vehicle_exposure_score


GRAPH_PATH = Path("osmnx_outputs/gwanghwamun_walk_2km_scored.graphml")
OD_PATH = Path("osmnx_outputs/gwanghwamun_route_overlap_100_pairs.csv")
DETAIL_PATH = Path("osmnx_outputs/gwanghwamun_weight_sensitivity_100_pairs.csv")
SUMMARY_PATH = Path("osmnx_outputs/gwanghwamun_weight_sensitivity_summary.csv")
WEIGHTS = [0.5, 1, 2, 4, 8]


def number(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def selected_edges(graph: nx.MultiDiGraph, nodes: list[object], cost_key: str) -> list[tuple[object, object, object]]:
    result = []
    for u, v in zip(nodes, nodes[1:]):
        key, _ = min(graph[u][v].items(), key=lambda item: number(item[1].get(cost_key)))
        result.append((u, v, key))
    return result


def route_length(graph: nx.MultiDiGraph, edges: list[tuple[object, object, object]]) -> float:
    return sum(number(graph[u][v][key].get("length")) for u, v, key in edges)


def main() -> None:
    graph = ox.load_graphml(GRAPH_PATH)
    # Update the score attribute in-place without re-querying tree/park data.
    for _, _, _, data in graph.edges(keys=True, data=True):
        data["vehicle_exposure_score"] = vehicle_exposure_score(highway_values(data.get("highway")))
    ox.save_graphml(graph, GRAPH_PATH)

    od_pairs = pd.read_csv(OD_PATH)[["pair_id", "origin", "destination"]]
    details: list[dict[str, object]] = []
    for _, pair in od_pairs.iterrows():
        origin, destination = pair["origin"], pair["destination"]
        shortest_nodes = nx.shortest_path(graph, origin, destination, weight="length")
        shortest_edges = selected_edges(graph, shortest_nodes, "length")
        shortest_length = route_length(graph, shortest_edges)
        shortest_set = set(shortest_edges)
        for weight in WEIGHTS:
            cost_key = f"exposure_cost_w_{str(weight).replace('.', '_')}"
            for _, _, _, data in graph.edges(keys=True, data=True):
                data[cost_key] = number(data.get("length")) * (1 + weight * number(data.get("vehicle_exposure_score")) / 4)
            weighted_nodes = nx.shortest_path(graph, origin, destination, weight=cost_key)
            weighted_edges = selected_edges(graph, weighted_nodes, cost_key)
            weighted_set = set(weighted_edges)
            union = shortest_set | weighted_set
            details.append({
                "pair_id": int(pair["pair_id"]), "origin": origin, "destination": destination, "w": weight,
                "shortest_path_nodes": json.dumps(shortest_nodes, default=int),
                "exposure_weighted_path_nodes": json.dumps(weighted_nodes, default=int),
                "shortest_length_m": shortest_length,
                "exposure_weighted_length_m": route_length(graph, weighted_edges),
                "distance_ratio": route_length(graph, weighted_edges) / shortest_length,
                "shared_edge_count": len(shortest_set & weighted_set),
                "edge_overlap_jaccard_pct": len(shortest_set & weighted_set) / len(union) * 100 if union else 100.0,
            })

    detail_df = pd.DataFrame(details)
    summary = detail_df.groupby("w", as_index=False).agg(
        distance_ratio_median=("distance_ratio", "median"),
        edge_overlap_jaccard_pct_median=("edge_overlap_jaccard_pct", "median"),
    )
    detail_df.to_csv(DETAIL_PATH, index=False, encoding="utf-8-sig")
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    print(summary.to_csv(index=False))
    print(f"Saved {DETAIL_PATH}")
    print(f"Saved {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
