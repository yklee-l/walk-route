from __future__ import annotations

import ast
import math
from pathlib import Path

import osmnx as ox
import pandas as pd


INPUT = Path("osmnx_outputs/gwanghwamun_walk_2km.graphml")
OUTPUT = Path("osmnx_outputs/gwanghwamun_walk_2km_scored.graphml")
CENTER = (37.5720, 126.9769)
DIST_METERS = 2_000
TREE_RADIUS_M = 20


def highway_values(value: object) -> set[str]:
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value}
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return {value}
        if isinstance(parsed, (list, tuple, set)):
            return {str(item) for item in parsed}
    return {str(value)} if value is not None else set()


def vehicle_exposure_score(highways: set[str]) -> int:
    # Lower score wins where a multi-value highway tag appears.
    scores = {"footway": 0, "path": 0, "residential": 1, "tertiary": 2, "secondary": 3, "primary": 4}
    matched = [scores[value] for value in highways if value in scores]
    return min(matched) if matched else 0


def main() -> None:
    graph_wgs84 = ox.load_graphml(INPUT)
    graph = ox.project_graph(graph_wgs84)
    metric_crs = graph.graph["crs"]
    print(f"Projected CRS: {metric_crs}")

    # Identical 2 km bbox and tags to the existing regional download, allowing
    # OSMnx to reuse its local Overpass response cache where available.
    bbox = ox.utils_geo.bbox_from_point(CENTER, dist=DIST_METERS, project_utm=False)
    features = ox.features_from_bbox(
        bbox,
        tags={"natural": ["tree", "tree_row", "water"], "leisure": "park", "waterway": True},
    ).to_crs(metric_crs)
    trees = features.loc[features.get("natural", pd.Series(index=features.index, dtype=object)) == "tree"].copy()
    parks = features.loc[features.get("leisure", pd.Series(index=features.index, dtype=object)) == "park"].copy()
    tree_points = trees.geometry.representative_point()
    tree_index = tree_points.sindex
    park_index = parks.sindex
    print(f"Trees: {len(trees)}, parks: {len(parks)}")

    edges = ox.graph_to_gdfs(graph, nodes=False, fill_edge_geometry=True)
    centers = edges.geometry.interpolate(0.5, normalized=True)
    attrs: dict[tuple[object, object, object], dict[str, float | int]] = {}

    for position, ((u, v, key), edge) in enumerate(edges.iterrows(), start=1):
        center = centers.loc[(u, v, key)]
        highways = highway_values(edge.get("highway"))
        if len(tree_points):
            candidate_positions = tree_index.query(center.buffer(TREE_RADIUS_M), predicate="intersects")
            tree_count = int(sum(tree_points.iloc[i].distance(center) <= TREE_RADIUS_M for i in candidate_positions))
        else:
            tree_count = 0
        if len(parks):
            _, distances = park_index.nearest(center, return_all=False, return_distance=True)
            park_distance = float(distances[0]) if len(distances) else math.nan
        else:
            park_distance = math.nan
        attrs[(u, v, key)] = {
            "vehicle_exposure_score": vehicle_exposure_score(highways),
            "pedestrian_only_score": int(bool(highways & {"footway", "path", "pedestrian"})),
            "steps_score": int("steps" in highways),
            "tree_count_20m": tree_count,
            "park_distance_m": park_distance,
        }
        if position % 5_000 == 0:
            print(f"Scored {position:,}/{len(edges):,} edges", flush=True)

    for (u, v, key), values in attrs.items():
        graph[u][v][key].update(values)
    scored_wgs84 = ox.project_graph(graph, to_crs=graph_wgs84.graph["crs"])
    ox.save_graphml(scored_wgs84, OUTPUT)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
