from __future__ import annotations

import math
import sys
from pathlib import Path

import osmnx as ox
import pandas as pd


DIST_METERS = 2_000
LOCATIONS = {
    "gwanghwamun": (37.5720, 126.9769),
    "seongsu_seoul_forest": (37.5445, 127.0557),
    "yongsan_hannam": (37.5340, 126.9940),
    "apgujeong": (37.5270, 127.0280),
    "yeouido": (37.5216, 126.9241),
    "yangjae_dogok": (37.4870, 127.0430),
}
HIGHWAY_CLASSES = ["residential", "primary", "secondary", "tertiary", "footway", "path", "steps"]
OUT_DIR = Path("osmnx_outputs")

# The public default endpoint can become saturated during large feature queries.
ox.settings.overpass_url = "https://overpass-api.de/api"
ox.settings.overpass_rate_limit = True
ox.settings.requests_timeout = 180


def filled(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple, set)):
        return bool(value)
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return bool(str(value).strip())


def has_highway(value: object, highway_class: str) -> bool:
    if isinstance(value, (list, tuple, set)):
        return highway_class in value
    return value == highway_class


def count_tagged(features: pd.DataFrame, tag: str, expected: str | None = None) -> int:
    if tag not in features.columns:
        return 0
    column = features[tag]
    if expected is None:
        return int(column.map(filled).sum())
    return int((column == expected).sum())


def collect_row(name: str, point: tuple[float, float]) -> dict[str, object]:
    graphml_path = OUT_DIR / f"{name}_walk_2km.graphml"
    if graphml_path.exists():
        print(f"Loading cached {name} graph...", flush=True)
        graph = ox.load_graphml(graphml_path)
    else:
        print(f"Downloading {name} graph...", flush=True)
        graph = ox.graph_from_point(point, dist=DIST_METERS, dist_type="bbox", network_type="walk")
        ox.save_graphml(graph, graphml_path)

    edge_data = list(graph.edges(data=True))
    edge_total = len(edge_data)
    row: dict[str, object] = {
        "region": name,
        "latitude": point[0],
        "longitude": point[1],
        "distance_m": DIST_METERS,
        "distance_type": "bbox",
        "network_type": "walk",
        "nodes": graph.number_of_nodes(),
        "edges": edge_total,
    }
    for highway_class in HIGHWAY_CLASSES:
        count = sum(has_highway(data.get("highway"), highway_class) for _, _, data in edge_data)
        row[f"highway_{highway_class}_edge_pct"] = count / edge_total * 100 if edge_total else 0.0
    row["lit_edge_pct"] = sum(filled(data.get("lit")) for _, _, data in edge_data) / edge_total * 100 if edge_total else 0.0
    row["sidewalk_edge_pct"] = sum(filled(data.get("sidewalk")) for _, _, data in edge_data) / edge_total * 100 if edge_total else 0.0

    # Use the same 2 km bounding box as graph_from_point for non-network OSM features.
    bbox = ox.utils_geo.bbox_from_point(point, dist=DIST_METERS, project_utm=False)
    print(f"Downloading {name} non-network features...", flush=True)
    green_blue = ox.features_from_bbox(
        bbox,
        tags={"natural": ["tree", "tree_row", "water"], "leisure": "park", "waterway": True},
    )
    row["natural_tree_count"] = count_tagged(green_blue, "natural", "tree")
    row["natural_tree_row_count"] = count_tagged(green_blue, "natural", "tree_row")
    row["leisure_park_count"] = count_tagged(green_blue, "leisure", "park")
    row["natural_water_or_waterway_count"] = int(
        ((green_blue.get("natural") == "water") | green_blue.get("waterway", pd.Series(False, index=green_blue.index)).map(filled)).sum()
    )

    buildings = ox.features_from_bbox(bbox, tags={"building": True})
    building_total = len(buildings)
    building_with_height = (
        buildings.get("building:levels", pd.Series(index=buildings.index, dtype=object)).map(filled)
        | buildings.get("height", pd.Series(index=buildings.index, dtype=object)).map(filled)
    )
    row["buildings_total"] = building_total
    row["buildings_with_levels_or_height_pct"] = float(building_with_height.mean() * 100) if building_total else 0.0
    return row


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    csv_path = OUT_DIR / "walk_network_comparison.csv"
    rows = pd.read_csv(csv_path).to_dict("records") if csv_path.exists() else []
    existing = {row["region"] for row in rows}
    selected = sys.argv[1:] or list(LOCATIONS)
    unknown = set(selected) - set(LOCATIONS)
    if unknown:
        raise SystemExit(f"Unknown region(s): {', '.join(sorted(unknown))}")
    for name in selected:
        if name in existing:
            print(f"Skipping {name}; it is already in {csv_path}", flush=True)
            continue
        rows.append(collect_row(name, LOCATIONS[name]))
        pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    result = pd.DataFrame(rows)
    print("\nComparison")
    print(result.to_csv(index=False))
    print(f"Saved CSV: {OUT_DIR / 'walk_network_comparison.csv'}")


if __name__ == "__main__":
    main()
