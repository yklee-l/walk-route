from __future__ import annotations

import json
from pathlib import Path

import folium
import osmnx as ox
import pandas as pd
from shapely.geometry import LineString

from score_gwanghwamun_edges import highway_values, vehicle_exposure_score


GRAPH_PATH = Path("osmnx_outputs/gwanghwamun_walk_2km_scored.graphml")
DETAIL_PATH = Path("osmnx_outputs/gwanghwamun_weight_sensitivity_100_pairs.csv")
OUTPUT = Path("osmnx_outputs/gwanghwamun_w0_w4_median_distance_ratio.html")
W = 4.0


def number(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def edge_cost(data: dict[str, object]) -> float:
    score = number(data.get("vehicle_exposure_score"))
    return number(data.get("length")) * (1 + W * score / 4)


def route_coordinates(graph, nodes: list[int], weighted: bool) -> list[tuple[float, float]]:
    coordinates: list[tuple[float, float]] = []
    for u, v in zip(nodes, nodes[1:]):
        key, data = min(
            graph[u][v].items(),
            key=lambda item: edge_cost(item[1]) if weighted else number(item[1].get("length")),
        )
        geometry = data.get("geometry")
        if geometry is None:
            geometry = LineString([(graph.nodes[u]["x"], graph.nodes[u]["y"]), (graph.nodes[v]["x"], graph.nodes[v]["y"])])
        segment = list(geometry.coords)
        source = (graph.nodes[u]["x"], graph.nodes[u]["y"])
        if (segment[0][0] - source[0]) ** 2 + (segment[0][1] - source[1]) ** 2 > (segment[-1][0] - source[0]) ** 2 + (segment[-1][1] - source[1]) ** 2:
            segment.reverse()
        if coordinates and coordinates[-1] == segment[0]:
            segment = segment[1:]
        coordinates.extend(segment)
    return [(lat, lon) for lon, lat in coordinates]


def main() -> None:
    details = pd.read_csv(DETAIL_PATH)
    subset = details.loc[details["w"] == W].copy()
    median_ratio = subset["distance_ratio"].median()
    selected = subset.loc[(subset["distance_ratio"] - median_ratio).abs().idxmin()]
    graph = ox.load_graphml(GRAPH_PATH)
    # Ensure the map uses the same updated mapping that produced the sensitivity table.
    for _, _, _, data in graph.edges(keys=True, data=True):
        data["vehicle_exposure_score"] = vehicle_exposure_score(highway_values(data.get("highway")))

    shortest_nodes = [int(node) for node in json.loads(selected["shortest_path_nodes"])]
    weighted_nodes = [int(node) for node in json.loads(selected["exposure_weighted_path_nodes"])]
    shortest_coords = route_coordinates(graph, shortest_nodes, weighted=False)
    weighted_coords = route_coordinates(graph, weighted_nodes, weighted=True)
    origin = shortest_coords[0]
    destination = shortest_coords[-1]
    route_map = folium.Map(
        location=[(origin[0] + destination[0]) / 2, (origin[1] + destination[1]) / 2],
        zoom_start=15,
        control_scale=True,
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr="CartoDB",
    )

    folium.PolyLine(shortest_coords, color="#2563eb", weight=5, opacity=0.85, tooltip="w=0: shortest-distance route").add_to(route_map)
    folium.PolyLine(weighted_coords, color="#f97316", weight=5, opacity=0.85, tooltip="w=4: vehicle-exposure-weighted route").add_to(route_map)
    folium.Marker(origin, tooltip="Origin", icon=folium.Icon(color="green", icon="play")).add_to(route_map)
    folium.Marker(destination, tooltip="Destination", icon=folium.Icon(color="red", icon="flag")).add_to(route_map)
    route_map.fit_bounds([origin, destination] + shortest_coords + weighted_coords)
    legend = f"""
    <div style='position: fixed; bottom: 28px; left: 28px; z-index: 1000; background: white; padding: 10px 12px; border: 1px solid #777; border-radius: 4px;'>
      <b>Selected OD pair #{int(selected['pair_id'])}</b><br>
      Median-target distance ratio: {selected['distance_ratio']:.3f}x<br>
      <span style='color:#2563eb'>&#9632;</span> w=0 shortest distance<br>
      <span style='color:#f97316'>&#9632;</span> w=4 exposure-weighted
    </div>"""
    route_map.get_root().html.add_child(folium.Element(legend))
    route_map.save(OUTPUT)
    print({
        "pair_id": int(selected["pair_id"]), "median_distance_ratio": median_ratio,
        "selected_distance_ratio": selected["distance_ratio"], "shortest_length_m": selected["shortest_length_m"],
        "w4_length_m": selected["exposure_weighted_length_m"], "output": str(OUTPUT),
    })


if __name__ == "__main__":
    main()
