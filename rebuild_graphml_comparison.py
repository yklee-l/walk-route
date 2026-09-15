from __future__ import annotations

from pathlib import Path

import osmnx as ox
import pandas as pd


OUT_DIR = Path("osmnx_outputs")
CSV_PATH = OUT_DIR / "walk_network_comparison.csv"
DIST_METERS = 2_000
LOCATIONS = {
    "gwanghwamun": (37.5720, 126.9769),
    "seongsu_seoul_forest": (37.5445, 127.0557),
    "yongsan_hannam": (37.5340, 126.9940),
}
HIGHWAYS = ["residential", "primary", "secondary", "tertiary", "footway", "path", "steps"]
SPATIAL_COLUMNS = [
    "natural_tree_count", "natural_tree_row_count", "leisure_park_count",
    "natural_water_or_waterway_count", "buildings_total", "buildings_with_levels_or_height_pct",
]


def filled(value: object) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return bool(str(value).strip())


def highway_has(value: object, wanted: str) -> bool:
    return wanted in value if isinstance(value, (list, tuple, set)) else value == wanted


def spatial_stats(point: tuple[float, float]) -> dict[str, float | int]:
    bbox = ox.utils_geo.bbox_from_point(point, dist=DIST_METERS, project_utm=False)
    features = ox.features_from_bbox(
        bbox,
        tags={"natural": ["tree", "tree_row", "water"], "leisure": "park", "waterway": True},
    )
    buildings = ox.features_from_bbox(bbox, tags={"building": True})
    natural = features.get("natural", pd.Series(index=features.index, dtype=object))
    leisure = features.get("leisure", pd.Series(index=features.index, dtype=object))
    waterway = features.get("waterway", pd.Series(index=features.index, dtype=object))
    levels = buildings.get("building:levels", pd.Series(index=buildings.index, dtype=object))
    height = buildings.get("height", pd.Series(index=buildings.index, dtype=object))
    return {
        "natural_tree_count": int((natural == "tree").sum()),
        "natural_tree_row_count": int((natural == "tree_row").sum()),
        "leisure_park_count": int((leisure == "park").sum()),
        "natural_water_or_waterway_count": int(((natural == "water") | waterway.map(filled)).sum()),
        "buildings_total": len(buildings),
        "buildings_with_levels_or_height_pct": float((levels.map(filled) | height.map(filled)).mean() * 100) if len(buildings) else 0.0,
    }


def region_for(path: Path) -> str:
    return path.name.removesuffix("_walk_2km_scored.graphml").removesuffix("_walk_2km.graphml")


def main() -> None:
    prior = pd.read_csv(CSV_PATH).set_index("region").to_dict("index") if CSV_PATH.exists() else {}
    rows = []
    for graphml in sorted(OUT_DIR.glob("*.graphml")):
        region = region_for(graphml)
        graph = ox.load_graphml(graphml)
        edges = list(graph.edges(data=True))
        row: dict[str, object] = {
            "source_graphml": graphml.name, "region": region,
            "latitude": LOCATIONS[region][0], "longitude": LOCATIONS[region][1],
            "distance_m": DIST_METERS, "distance_type": "bbox", "network_type": "walk",
            "nodes": graph.number_of_nodes(), "edges": len(edges),
        }
        for highway in HIGHWAYS:
            row[f"highway_{highway}_edge_pct"] = sum(highway_has(data.get("highway"), highway) for _, _, data in edges) / len(edges) * 100
        row["lit_edge_pct"] = sum(filled(data.get("lit")) for _, _, data in edges) / len(edges) * 100
        row["sidewalk_edge_pct"] = sum(filled(data.get("sidewalk")) for _, _, data in edges) / len(edges) * 100
        if region in prior:
            row.update({column: prior[region][column] for column in SPATIAL_COLUMNS})
        else:
            print(f"Downloading spatial features for {region}...", flush=True)
            row.update(spatial_stats(LOCATIONS[region]))
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print(result.to_csv(index=False))
    print(f"Saved {CSV_PATH}")


if __name__ == "__main__":
    main()
