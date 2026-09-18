from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd

from score_gwanghwamun_edges import CENTER, DIST_METERS, TREE_RADIUS_M


TREE_CSV = Path("data/서울시 가로수 위치 정보(경도, 위도).csv")
GRAPH_PATH = Path("osmnx_outputs/gwanghwamun_walk_2km_scored.graphml")
SUMMARY_PATH = Path("osmnx_outputs/gwanghwamun_tree_count_seoul_summary.csv")


def load_trees(crs: object) -> gpd.GeoDataFrame:
    trees = pd.read_csv(TREE_CSV, encoding="cp949").dropna(subset=["좌표(경도)", "좌표(위도)"])
    # Same 2 km bbox the graph was downloaded with, so boundary edges keep their trees.
    bbox = ox.utils_geo.bbox_from_point(CENTER, dist=DIST_METERS, project_utm=False)
    left, bottom, right, top = bbox
    inside = trees["좌표(경도)"].between(left, right) & trees["좌표(위도)"].between(bottom, top)
    trees = trees.loc[inside]
    return gpd.GeoDataFrame(trees, geometry=gpd.points_from_xy(trees["좌표(경도)"], trees["좌표(위도)"]), crs="EPSG:4326").to_crs(crs)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    graph_wgs84 = ox.load_graphml(GRAPH_PATH)
    graph = ox.project_graph(graph_wgs84)
    trees = load_trees(graph.graph["crs"])
    print(f"Street trees inside the 2 km bbox: {len(trees):,}")
    print(trees["수종"].value_counts().head(8).to_string())

    edges = ox.graph_to_gdfs(graph, nodes=False, fill_edge_geometry=True)
    # Trees within TREE_RADIUS_M of the whole edge line, not just its midpoint: street
    # trees run along the road, and a midpoint test misses most of them on long edges.
    edge_positions, _ = trees.sindex.query(edges.geometry.buffer(TREE_RADIUS_M), predicate="intersects")
    counts = np.bincount(edge_positions, minlength=len(edges))

    old = pd.to_numeric(edges.get("tree_count_20m"), errors="coerce").fillna(0).astype(int).to_numpy()
    for (u, v, key), old_count, new_count in zip(edges.index, old, counts):
        data = graph_wgs84[u][v][key]
        data["tree_count_20m_osm"] = int(old_count)
        data["tree_count_20m"] = int(new_count)
    ox.save_graphml(graph_wgs84, GRAPH_PATH)

    per_100m = counts / edges["length"].to_numpy() * 100
    summary = pd.DataFrame({
        "metric": ["tree_count_20m_osm", "tree_count_20m_seoul", "trees_per_100m_seoul"],
        "edges_gt0_pct": [(old > 0).mean() * 100, (counts > 0).mean() * 100, (per_100m > 0).mean() * 100],
        **{f"p{q}": [np.percentile(old, q), np.percentile(counts, q), np.percentile(per_100m, q)] for q in (50, 75, 90, 95, 99)},
        "max": [old.max(), counts.max(), per_100m.max()],
    })
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    print(summary.round(2).to_string(index=False))
    print(f"Overwrote tree_count_20m in {GRAPH_PATH} (previous values kept as tree_count_20m_osm)")
    print(f"Saved {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
