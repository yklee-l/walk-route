from __future__ import annotations

import ast
from pathlib import Path

import folium
import networkx as nx
import osmnx as ox
import pandas as pd
from shapely.geometry import LineString

from score_gwanghwamun_edges import highway_values, vehicle_exposure_score


GRAPH_PATH = Path("osmnx_outputs/gwanghwamun_walk_2km_scored.graphml")
MAP_PATH = Path("osmnx_outputs/demo_gyeongbokgung_myeongdong.html")
TABLE_PATH = Path("osmnx_outputs/demo_gyeongbokgung_myeongdong_highway_composition.csv")
ORIGIN = (37.5796, 126.9770)
DESTINATION = (37.5636, 126.9850)
ROUTES = {0: "#2563eb", 1: "#16a34a", 4: "#f97316"}


def number(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def cost(data: dict[str, object], w: float) -> float:
    return number(data.get("length")) * (1 + w * number(data.get("vehicle_exposure_score")) / 4)


def route_edges(graph: nx.MultiDiGraph, nodes: list[object], w: float) -> list[tuple[object, object, object]]:
    edges = []
    for u, v in zip(nodes, nodes[1:]):
        key, _ = min(graph[u][v].items(), key=lambda item: cost(item[1], w))
        edges.append((u, v, key))
    return edges


def highway_label(value: object) -> str:
    values = highway_values(value)
    return "|".join(sorted(values)) if values else "missing"


def coordinates(graph: nx.MultiDiGraph, edges: list[tuple[object, object, object]]) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for u, v, key in edges:
        data = graph[u][v][key]
        geometry = data.get("geometry")
        if geometry is None:
            geometry = LineString([(graph.nodes[u]["x"], graph.nodes[u]["y"]), (graph.nodes[v]["x"], graph.nodes[v]["y"])])
        segment = list(geometry.coords)
        source = (graph.nodes[u]["x"], graph.nodes[u]["y"])
        if (segment[0][0] - source[0]) ** 2 + (segment[0][1] - source[1]) ** 2 > (segment[-1][0] - source[0]) ** 2 + (segment[-1][1] - source[1]) ** 2:
            segment.reverse()
        if result and result[-1] == segment[0]:
            segment = segment[1:]
        result.extend(segment)
    return [(lat, lon) for lon, lat in result]


def nearest_node(graph: nx.MultiDiGraph, point: tuple[float, float]) -> object:
    lat, lon = point
    return min(
        graph.nodes,
        key=lambda node: (number(graph.nodes[node]["y"]) - lat) ** 2 + (number(graph.nodes[node]["x"]) - lon) ** 2,
    )


def main() -> None:
    graph = ox.load_graphml(GRAPH_PATH)
    for _, _, _, data in graph.edges(keys=True, data=True):
        data["vehicle_exposure_score"] = vehicle_exposure_score(highway_values(data.get("highway")))
    origin_node = nearest_node(graph, ORIGIN)
    destination_node = nearest_node(graph, DESTINATION)

    route_data: dict[int, tuple[list[tuple[object, object, object]], list[tuple[float, float]], float, float]] = {}
    composition_rows = []
    for w in ROUTES:
        nodes = nx.shortest_path(graph, origin_node, destination_node, weight=lambda u, v, data: min(cost(edge, w) for edge in data.values()))
        edges = route_edges(graph, nodes, w)
        length = sum(number(graph[u][v][key].get("length")) for u, v, key in edges)
        # Exposure-weighted length, E = sum(length * score / 4), in meters.
        exposure_weighted_sum = sum(number(graph[u][v][key].get("length")) * number(graph[u][v][key].get("vehicle_exposure_score")) / 4 for u, v, key in edges)
        route_data[w] = (edges, coordinates(graph, edges), length, exposure_weighted_sum)
        for u, v, key in edges:
            data = graph[u][v][key]
            composition_rows.append({"w": w, "highway": highway_label(data.get("highway")), "edge_length_m": number(data.get("length"))})

    composition = pd.DataFrame(composition_rows).groupby(["w", "highway"], as_index=False).agg(edge_count=("edge_length_m", "size"), total_length_m=("edge_length_m", "sum"))
    totals = composition.groupby("w")["total_length_m"].transform("sum")
    composition["length_pct"] = composition["total_length_m"] / totals * 100
    composition = composition.sort_values(["w", "total_length_m"], ascending=[True, False])
    composition.to_csv(TABLE_PATH, index=False, encoding="utf-8-sig")

    route_map = folium.Map(
        location=[(ORIGIN[0] + DESTINATION[0]) / 2, (ORIGIN[1] + DESTINATION[1]) / 2],
        zoom_start=15,
        control_scale=True,
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr="CartoDB",
    )
    legend_lines = []
    for w, color in ROUTES.items():
        _, coords, length, exposure_sum = route_data[w]
        folium.PolyLine(coords, color=color, weight=5, opacity=0.85, tooltip=f"w={w}").add_to(route_map)
        legend_lines.append(f"<span style='color:{color}'>&#9632;</span> w={w}: {length:,.1f}m, E={exposure_sum:,.1f}m")
    folium.Marker(ORIGIN, tooltip="Gyeongbokgung", icon=folium.Icon(color="green", icon="play")).add_to(route_map)
    folium.Marker(DESTINATION, tooltip="Myeongdong", icon=folium.Icon(color="red", icon="flag")).add_to(route_map)
    route_map.get_root().html.add_child(folium.Element("<div style='position: fixed; bottom: 28px; left: 28px; z-index: 1000; background: white; padding: 10px 12px; border: 1px solid #777; border-radius: 4px;'><b>Gyeongbokgung → Myeongdong</b><br>" + "<br>".join(legend_lines) + "<br><small>E = sum(length × exposure score / 4)</small></div>"))
    route_map.save(MAP_PATH)

    selected = composition.loc[(composition["w"] == 4) & composition["highway"].apply(lambda value: bool(set(value.split("|")) & {"service", "unclassified", "track"}))]
    special_length = selected["total_length_m"].sum()
    w4_length = route_data[4][2]
    print("ROUTES")
    for w, (_, _, length, exposure_sum) in route_data.items():
        print({"w": w, "length_m": length, "exposure_weighted_sum_m": exposure_sum})
    print("COMPOSITION")
    print(composition.to_csv(index=False))
    print({"w4_service_unclassified_track_length_m": special_length, "w4_service_unclassified_track_pct": special_length / w4_length * 100})
    print(f"Saved {MAP_PATH}")
    print(f"Saved {TABLE_PATH}")


if __name__ == "__main__":
    main()
