from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import osmnx as ox
from shapely.geometry import LineString

from score_gwanghwamun_edges import highway_values, vehicle_exposure_score


GRAPH_PATH = Path("osmnx_outputs/gwanghwamun_walk_2km_scored.graphml")
COST_KEY = "_route_cost"
# Normalization caps for the penalty axes (both map to [0, 1]).
# tree_count_20m counts Seoul street trees within 20 m of the edge line (rescore_trees_seoul.py),
# so it scales with edge length; the axis uses density instead. 15 trees / 100 m is roughly one
# fully planted side at the usual 6-8 m spacing (graph p75 = 14, p90 = 34 per 100 m).
TREE_PER_100M_CAP = 15.0
PARK_CAP_M = 300.0  # park_distance_m p90 ~ 275 m

Edge = tuple[object, object, object]


def number(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_graph(path: Path = GRAPH_PATH) -> nx.MultiDiGraph:
    graph = ox.load_graphml(path)
    # Same in-place refresh the other scripts do, so every consumer sees one mapping.
    for _, _, _, data in graph.edges(keys=True, data=True):
        data["length"] = number(data.get("length"))
        data["vehicle_exposure_score"] = vehicle_exposure_score(highway_values(data.get("highway")))
        data["vehicle_exposure_osm"] = data["vehicle_exposure_score"]  # baseline; negative feedback bumps the score
        data["tree_count_20m"] = number(data.get("tree_count_20m"))
        data["steps_score"] = int(number(data.get("steps_score")))
        data["park_distance_m"] = number(data.get("park_distance_m"), default=PARK_CAP_M)
        data["vibe_score"] = min(max(number(data.get("vibe_score")), 0.0), 1.0)
        data["vibe_score_osm"] = data["vibe_score"]  # baseline; reviews (server.py) move vibe_score away from it
        data["review_count"] = 0
        data["review_score_sum"] = 0.0  # sum of 0..10 scores, so score_avg = review_score_sum / review_count
        # Negative feedback (server.py): avoid_penalty = avoid_score (from the average review score) + the
        # avoid_by_tag contributions for the tags a user chose to avoid. It multiplies the base cost for every
        # profile; only the pure shortest ignores it. The server assembles avoid_penalty per request.
        data["avoid_penalty"] = 0.0
        data["avoid_by_tag"] = {}
        data["avoid_score"] = 0.0
        data["vibe_nudge"] = 0.0  # small additive vibe adjustments from tags (e.g. 어두움), on top of the review average
        data["vehicle_bump"] = 0.0
        data["negative_count"] = 0
    return graph


def without_steps(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Copy of the graph with stair edges removed, sharing the remaining edges' data dicts with `graph`.

    A hard exclusion rather than a weight (design doc §05 decision 4), used for presets like strollers.
    Sharing the dicts means review/feedback updates on `graph` apply here too; a plain copy would not,
    and an nx.subgraph_view is ~8x slower to route on. Build once and reuse.
    """
    stripped = nx.MultiDiGraph(**graph.graph)
    stripped.add_nodes_from(graph.nodes(data=True))
    for u, v, k, data in graph.edges(keys=True, data=True):
        if data.get("steps_score"):
            continue
        stripped.add_edge(u, v, key=k)
        stripped._succ[u][v][k] = data  # same dict object in both adjacency tables (networkx keeps them in sync)
        stripped._pred[v][u][k] = data
    return stripped


def edge_base_cost(data: dict[str, object], avoid: bool = True) -> float:
    # length * (1 + avoid_penalty): reported segments cost more for every profile, even all-zero weights.
    return data["length"] * (1 + number(data.get("avoid_penalty"))) if avoid else data["length"]


# Each axis is a penalty in [0, 1] (0 = best) times edge length, i.e. "penalty meters".
def edge_exposure(data: dict[str, object]) -> float:
    # Vehicle exposure: length * score / 4.
    return data["length"] * number(data.get("vehicle_exposure_score")) / 4


def edge_tree_penalty(data: dict[str, object]) -> float:
    # Lack of trees: 0 at TREE_PER_100M_CAP or denser, 1 with no trees within 20 m.
    length = data["length"]
    if length <= 0:
        return 0.0
    density = number(data.get("tree_count_20m")) / length * 100
    return length * (1 - min(density, TREE_PER_100M_CAP) / TREE_PER_100M_CAP)


def edge_park_penalty(data: dict[str, object]) -> float:
    # Distance from the nearest park, saturating at PARK_CAP_M.
    return data["length"] * min(number(data.get("park_distance_m"), default=PARK_CAP_M), PARK_CAP_M) / PARK_CAP_M


def edge_vibe_penalty(data: dict[str, object]) -> float:
    # Lack of vibe: vibe_score is 0..1 (higher is nicer), so the penalty is 1 - score.
    return data["length"] * (1 - number(data.get("vibe_score")))


def edge_cost(data: dict[str, object], w_vehicle: float, w_tree: float = 0.0, w_park: float = 0.0, w_vibe: float = 0.0, avoid: bool = True) -> float:
    # length * (1 + avoid) + length * (w_vehicle * v + w_tree * t + w_park * p + w_vibe * b); all terms >= 0 so Dijkstra is safe.
    return (
        edge_base_cost(data, avoid)
        + w_vehicle * edge_exposure(data)
        + w_tree * edge_tree_penalty(data)
        + w_park * edge_park_penalty(data)
        + w_vibe * edge_vibe_penalty(data)
    )


def apply_cost(graph: nx.MultiDiGraph, w_vehicle: float, w_tree: float = 0.0, w_park: float = 0.0, w_vibe: float = 0.0, avoid: bool = True) -> None:
    for _, _, _, data in graph.edges(keys=True, data=True):
        data[COST_KEY] = edge_cost(data, w_vehicle, w_tree, w_park, w_vibe, avoid)


class CostTable:
    """Per-edge (base, penalty) for one profile, so re-costing the graph for another lambda is one multiply-add per edge.

    The candidate search runs Dijkstra 5-10 times per request with the same profile and different
    lambdas; evaluating the five axis functions per edge each time dominated request latency.
    """

    def __init__(self, graph: nx.MultiDiGraph, profile: "Weights", avoid: bool = True) -> None:
        self.profile, self.avoid = profile, avoid
        self.rows: list[tuple[dict[str, object], float, float]] = []
        for _, _, _, data in graph.edges(keys=True, data=True):
            penalty = (
                profile.w_vehicle * edge_exposure(data)
                + profile.w_tree * edge_tree_penalty(data)
                + profile.w_park * edge_park_penalty(data)
                + profile.w_vibe * edge_vibe_penalty(data)
            )
            self.rows.append((data, edge_base_cost(data, avoid), penalty))

    def apply(self, factor: float) -> None:
        """Write COST_KEY = base + factor * penalty on every edge (equivalent to apply_cost with profile.scaled(factor))."""
        for data, base, penalty in self.rows:
            data[COST_KEY] = base + factor * penalty


@dataclass(frozen=True)
class Weights:
    w_vehicle: float = 0.0
    w_tree: float = 0.0
    w_park: float = 0.0
    w_vibe: float = 0.0

    def scaled(self, factor: float) -> "Weights":
        return Weights(self.w_vehicle * factor, self.w_tree * factor, self.w_park * factor, self.w_vibe * factor)


@dataclass(frozen=True)
class Route:
    weights: Weights
    nodes: list[object]
    edges: list[Edge]
    length_m: float
    base_cost_m: float  # length_m plus avoid penalties; the lambda-independent part of the cost
    exposure_m: float  # vehicle-exposure penalty meters
    tree_m: float  # tree-lack penalty meters
    park_m: float  # park-distance penalty meters
    vibe_m: float  # lack-of-vibe penalty meters
    tree_count: int  # street trees within 20 m of the route's edges (double counts shared trees)

    @property
    def w(self) -> float:
        return self.weights.w_vehicle

    def penalty_m(self, profile: Weights) -> float:
        """Weighted penalty meters under `profile`; cost = length_m + penalty_m(weights)."""
        return profile.w_vehicle * self.exposure_m + profile.w_tree * self.tree_m + profile.w_park * self.park_m + profile.w_vibe * self.vibe_m

    @property
    def edge_set(self) -> frozenset[Edge]:
        return frozenset(self.edges)


def shortest_route(graph: nx.MultiDiGraph, origin: object, destination: object, weights: Weights | float, avoid: bool = True, table: CostTable | None = None, factor: float = 1.0) -> Route:
    """Min-cost route under `weights`; avoid=False ignores reported-segment penalties (the true shortest path).

    With `table`, costs come from table.apply(factor) (weights must equal table.profile.scaled(factor)).
    """
    if not isinstance(weights, Weights):
        weights = Weights(w_vehicle=float(weights))
    if table is not None:
        table.apply(factor)
    else:
        apply_cost(graph, weights.w_vehicle, weights.w_tree, weights.w_park, weights.w_vibe, avoid)
    nodes = nx.shortest_path(graph, origin, destination, weight=COST_KEY)
    edges: list[Edge] = []
    for u, v in zip(nodes, nodes[1:]):
        key = min(graph[u][v], key=lambda k: graph[u][v][k][COST_KEY])
        edges.append((u, v, key))
    datas = [graph[u][v][key] for u, v, key in edges]
    return Route(
        weights=weights, nodes=nodes, edges=edges,
        length_m=sum(d["length"] for d in datas),
        base_cost_m=sum(edge_base_cost(d) for d in datas),
        exposure_m=sum(edge_exposure(d) for d in datas),
        tree_m=sum(edge_tree_penalty(d) for d in datas),
        park_m=sum(edge_park_penalty(d) for d in datas),
        vibe_m=sum(edge_vibe_penalty(d) for d in datas),
        tree_count=int(sum(number(d.get("tree_count_20m")) for d in datas)),
    )


def edge_name(data: dict[str, object]) -> str | None:
    value = data.get("name")
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if isinstance(value, str) and value.startswith("["):
        value = next(iter(highway_values(value)), None)
    return str(value) if value else None


def route_segments(graph: nx.MultiDiGraph, route: Route) -> list[dict[str, object]]:
    """Consecutive edges grouped by street name, with their own geometry so a client can point at one."""
    segments: list[dict[str, object]] = []
    for u, v, k in route.edges:
        data = graph[u][v][k]
        name, highway = edge_name(data), "|".join(sorted(highway_values(data.get("highway")))) or None
        if segments and segments[-1]["name"] == name and segments[-1]["highway"] == highway:
            segments[-1]["edges"].append([u, v, k])
        else:
            segments.append({"name": name, "highway": highway, "edges": [[u, v, k]]})
    for segment in segments:
        edges = [tuple(e) for e in segment["edges"]]
        segment["length_m"] = round(sum(graph[a][b][c]["length"] for a, b, c in edges), 1)
        segment["negative_count"] = int(max(number(graph[a][b][c].get("negative_count")) for a, b, c in edges))
        # A review covers every edge of the stretch it was left on, so max() approximates "reviews here".
        counts = [number(graph[a][b][c].get("review_count")) for a, b, c in edges]
        total = sum(counts)
        segment["review_count"] = int(max(counts))
        segment["score_avg"] = round(sum(number(graph[a][b][c].get("review_score_sum")) for a, b, c in edges) / total, 1) if total else None
        segment["edge_id"] = ",".join(f"{a}-{b}-{c}" for a, b, c in edges)
        # Tags with an avoid penalty on any edge here (from bad-tag reviews / negative feedback), for the card copy.
        segment["report_tags"] = sorted({tag for a, b, c in edges for tag, value in (graph[a][b][c].get("avoid_by_tag") or {}).items() if value > 0})
        segment["coordinates"] = [[round(lon, 6), round(lat, 6)] for lon, lat in route_coordinates(graph, edges)]
    return segments


def bearing(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Compass bearing in degrees (0 = north, clockwise) from (lon, lat) a to b."""
    lon1, lat1, lon2, lat2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return math.degrees(math.atan2(x, y)) % 360


def turn_by_turn(graph: nx.MultiDiGraph, route: Route, segments: list[dict[str, object]] | None = None) -> list[dict[str, object]]:
    """Walking directions: one step per street segment with the turn taken to enter it.

    The turn angle is the heading change from the end of the previous segment to the start of
    this one (right turns positive). Unnamed short footways between roads are labelled as
    crossings (the graph has no crossing tag, so this is a heuristic), stairs by highway=steps.
    """
    segments = segments if segments is not None else route_segments(graph, route)
    steps: list[dict[str, object]] = []
    prev_out: float | None = None
    for i, seg in enumerate(segments):
        coords = seg["coordinates"]
        points = [coords[0]] + [c for j, c in enumerate(coords[1:], 1) if c != coords[j - 1]]  # drop repeated points
        heading_in = bearing(points[0], points[1]) if len(points) > 1 else (prev_out or 0.0)
        heading_out = bearing(points[-2], points[-1]) if len(points) > 1 else heading_in
        turn = 0.0 if prev_out is None else ((heading_in - prev_out + 180) % 360) - 180  # -180..180, right = +
        highways = set(str(seg["highway"] or "").split("|"))
        unnamed_footway = seg["name"] is None and highways <= {"footway", "path", "pedestrian"}
        neighbours_named = (i > 0 and segments[i - 1]["name"]) or (i + 1 < len(segments) and segments[i + 1]["name"])
        if i == 0:
            kind, arrow, text = "start", "●", "출발"
        elif "steps" in highways:
            kind, arrow, text = "steps", "⇅", "계단"
        elif unnamed_footway and seg["length_m"] <= 45 and neighbours_named:
            kind, arrow, text = "crossing", "↗" if turn > 25 else "↖" if turn < -25 else "↑", "횡단보도 건너기"
        elif abs(turn) < 25:
            kind, arrow, text = "straight", "↑", "직진"
        elif abs(turn) < 70:
            kind, arrow, text = ("slight_right", "↗", "오른쪽으로 살짝") if turn > 0 else ("slight_left", "↖", "왼쪽으로 살짝")
        elif abs(turn) <= 150:
            kind, arrow, text = ("right", "→", "우회전") if turn > 0 else ("left", "←", "좌회전")
        else:
            kind, arrow, text = "uturn", "↩", "유턴"
        steps.append({
            "kind": kind, "arrow": arrow, "instruction": text, "name": seg["name"], "highway": seg["highway"],
            "length_m": seg["length_m"], "turn_deg": round(turn), "heading_deg": round(heading_in), "edge_id": seg["edge_id"],
        })
        prev_out = heading_out
    if steps:
        steps.append({"kind": "arrive", "arrow": "🏁", "instruction": "도착", "name": None, "highway": None, "length_m": 0.0, "turn_deg": 0, "heading_deg": round(prev_out or 0), "edge_id": ""})
    return steps


def route_feature(graph: nx.MultiDiGraph, route: Route, **properties: object) -> dict[str, object]:
    """GeoJSON Feature (LineString, WGS84 lon/lat) with route stats plus any extra properties."""
    length = route.length_m or 1.0
    stats = {
        "length_m": round(route.length_m, 1),
        "exposure_m": round(route.exposure_m, 1),
        "tree_m": round(route.tree_m, 1),
        "park_m": round(route.park_m, 1),
        "vibe_m": round(route.vibe_m, 1),
        "vehicle_score_avg": round(route.exposure_m / length * 4, 2),  # 0 (footway) .. 4 (primary)
        "tree_count": route.tree_count,
        "trees_per_100m": round(route.tree_count / length * 100, 1),
        "park_distance_avg_m": round(route.park_m / length * PARK_CAP_M, 0),  # capped at PARK_CAP_M
        "vibe_score_avg": round(1 - route.vibe_m / length, 3),  # 0..1, length-weighted
        "vibe_score_osm_avg": round(sum(graph[u][v][k]["length"] * number(graph[u][v][k].get("vibe_score_osm")) for u, v, k in route.edges) / length, 3),
        "reviewed_length_pct": round(sum(graph[u][v][k]["length"] for u, v, k in route.edges if number(graph[u][v][k].get("review_count")) > 0) / length * 100, 1),
        "review_count": int(max((number(graph[u][v][k].get("review_count")) for u, v, k in route.edges), default=0)),
        "review_score_avg": (lambda n: round(sum(number(graph[u][v][k].get("review_score_sum")) for u, v, k in route.edges) / n, 1) if n else None)(sum(number(graph[u][v][k].get("review_count")) for u, v, k in route.edges)),
        "steps_count": int(sum(1 for u, v, k in route.edges if graph[u][v][k].get("steps_score"))),
        "reported_length_pct": round(sum(graph[u][v][k]["length"] for u, v, k in route.edges if number(graph[u][v][k].get("negative_count")) > 0) / length * 100, 1),
        "edges": [[u, v, k] for u, v, k in route.edges],  # lets a client send this exact route back (reviews)
        "segments": (segments := route_segments(graph, route)),
        "turn_by_turn": turn_by_turn(graph, route, segments),
        "weights": {"w_vehicle": route.weights.w_vehicle, "w_tree": route.weights.w_tree, "w_park": route.weights.w_park, "w_vibe": route.weights.w_vibe},
        "node_count": len(route.nodes),
    }
    return {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": [[round(lon, 6), round(lat, 6)] for lon, lat in route_coordinates(graph, route.edges)]},
        "properties": {**stats, **properties},
    }


def route_coordinates(graph: nx.MultiDiGraph, edges: list[Edge]) -> list[tuple[float, float]]:
    """(lon, lat) vertices along the route, following edge geometry where present."""
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
    return result


def nearest_node(graph: nx.MultiDiGraph, latlon: tuple[float, float]) -> object:
    # Brute force like the demo script; ox.nearest_nodes needs scikit-learn on unprojected graphs.
    lat, lon = latlon
    return min(graph.nodes, key=lambda n: (number(graph.nodes[n]["y"]) - lat) ** 2 + (number(graph.nodes[n]["x"]) - lon) ** 2)
