from __future__ import annotations

import json
import random
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd


INPUT = Path("osmnx_outputs/gwanghwamun_walk_2km_scored.graphml")
OUTPUT = Path("osmnx_outputs/gwanghwamun_route_overlap_100_pairs.csv")
PAIR_COUNT = 100
SEED = 20260915
EXPOSURE_PRIORITY = 1_000_000


def number(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def chosen_edges(graph: nx.MultiDiGraph, nodes: list[object], weight: str) -> list[tuple[object, object, object]]:
    result = []
    for u, v in zip(nodes, nodes[1:]):
        key, _ = min(graph[u][v].items(), key=lambda item: number(item[1].get(weight)))
        result.append((u, v, key))
    return result


def route_value(graph: nx.MultiDiGraph, edges: list[tuple[object, object, object]], attribute: str) -> float:
    return sum(number(graph[u][v][key].get(attribute)) for u, v, key in edges)


def main() -> None:
    graph = ox.load_graphml(INPUT)
    for _, _, _, data in graph.edges(keys=True, data=True):
        exposure = number(data.get("vehicle_exposure_score"))
        length = number(data.get("length"))
        # Lexicographic objective: minimize exposure first, then length.
        data["exposure_priority_cost"] = exposure * EXPOSURE_PRIORITY + length

    rng = random.Random(SEED)
    nodes = list(graph.nodes)
    rows: list[dict[str, object]] = []
    attempts = 0
    while len(rows) < PAIR_COUNT:
        attempts += 1
        if attempts > PAIR_COUNT * 50:
            raise RuntimeError("Could not find enough reachable origin-destination pairs.")
        origin, destination = rng.sample(nodes, 2)
        try:
            shortest_nodes = nx.shortest_path(graph, origin, destination, weight="length")
            exposure_nodes = nx.shortest_path(graph, origin, destination, weight="exposure_priority_cost")
        except nx.NetworkXNoPath:
            continue
        shortest_edges = chosen_edges(graph, shortest_nodes, "length")
        exposure_edges = chosen_edges(graph, exposure_nodes, "exposure_priority_cost")
        shortest_set, exposure_set = set(shortest_edges), set(exposure_edges)
        shared = shortest_set & exposure_set
        union = shortest_set | exposure_set
        rows.append({
            "pair_id": len(rows) + 1,
            "origin": origin,
            "destination": destination,
            "shortest_path_nodes": json.dumps(shortest_nodes),
            "exposure_priority_path_nodes": json.dumps(exposure_nodes),
            "shortest_path_edges": json.dumps(shortest_edges),
            "exposure_priority_path_edges": json.dumps(exposure_edges),
            "shortest_length_m": route_value(graph, shortest_edges, "length"),
            "exposure_priority_length_m": route_value(graph, exposure_edges, "length"),
            "shortest_exposure_sum": route_value(graph, shortest_edges, "vehicle_exposure_score"),
            "exposure_priority_exposure_sum": route_value(graph, exposure_edges, "vehicle_exposure_score"),
            "shortest_edge_count": len(shortest_edges),
            "exposure_priority_edge_count": len(exposure_edges),
            "shared_edge_count": len(shared),
            "edge_overlap_jaccard_pct": len(shared) / len(union) * 100 if union else 100.0,
        })

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    overlap = result["edge_overlap_jaccard_pct"]
    histogram = pd.cut(
        overlap, bins=[-0.001, 10, 25, 50, 75, 90, 100],
        labels=["0-10%", "10-25%", "25-50%", "50-75%", "75-90%", "90-100%"],
        include_lowest=True,
    ).value_counts(sort=False)
    print("OVERLAP_SUMMARY")
    print({
        "min": overlap.min(), "max": overlap.max(), "mean": overlap.mean(), "median": overlap.median(),
        "p10": overlap.quantile(0.10), "p25": overlap.quantile(0.25), "p75": overlap.quantile(0.75), "p90": overlap.quantile(0.90),
    })
    print("OVERLAP_HISTOGRAM")
    print(histogram.to_dict())
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
