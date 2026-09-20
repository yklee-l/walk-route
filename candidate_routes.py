from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import folium
import networkx as nx

from routing import CostTable, Route, Weights, load_graph, nearest_node, route_coordinates, route_feature, shortest_route
from waypoints import (
    DEFAULT_REVISIT_PENALTY, MATRIX_BETA, WAYPOINT_BETAS, PenalizedTable, WaypointError,
    concat_routes, dijkstra_route, plan, revisit_ratio, walk_legs,
)


BETAS = (1.0, 1.15, 1.3)  # shortest / balanced / calm; see validate_candidate_routes.py for why 1.0 is in
LABELS = {1.0: "최단", 1.15: "균형", 1.3: "쾌적"}
PROFILE = Weights(w_vehicle=1.0)  # direction of the penalty vector; the search scales it by lambda
W_MAX = 256.0  # at lambda=256 a residential edge costs 65x its length: effectively the min-penalty route
MAX_STEPS = 30
WALK_M_PER_MIN = 75  # 4.5 km/h, same as the page
ORIGIN = (37.5796, 126.9770)
DESTINATION = (37.5636, 126.9850)
MAP_PATH = Path("osmnx_outputs/candidates_gyeongbokgung_myeongdong.html")
COLORS = ("#16a34a", "#f97316", "#dc2626")


@dataclass(frozen=True)
class Candidate:
    beta: float
    route: Route
    budget_m: float
    w_switch: float  # smallest w at which the route flips to one that breaks the budget (W_MAX if never)
    budget_binding: bool  # False when even w=W_MAX stays within the budget
    duplicate_of: float | None  # beta of an earlier candidate with the same edges
    # Waypoint routing only (None otherwise): see waypoints.py
    revisit_ratio: float | None = None  # walked length on edges passed more than once / total length
    detour_m: float | None = None  # extra length vs. the direct origin->destination shortest path
    visit_order: list[int] | None = None  # input indices of the waypoints in visiting order
    waypoint_nodes: list[object] | None = None  # snapped waypoint nodes in visiting order


class RouteCache:
    """One Dijkstra per distinct lambda for a fixed OD pair; shared across the beta searches.

    Cost is length + lambda * penalty_m(profile), so a single scalar is searched no
    matter how many axes the profile mixes.
    """

    def __init__(self, graph: nx.MultiDiGraph, origin: object, destination: object, profile: Weights = PROFILE) -> None:
        self.graph, self.origin, self.destination, self.profile = graph, origin, destination, profile
        self.routes: dict[float, Route] = {}
        self.table = CostTable(graph, profile)  # per-edge base/penalty computed once per request

    def __call__(self, w: float) -> Route:
        if w not in self.routes:
            self.routes[w] = shortest_route(self.graph, self.origin, self.destination, self.profile.scaled(w), table=self.table, factor=w)
        return self.routes[w]

    def pure_shortest(self) -> Route:
        """True shortest path: no weights and no avoid penalties. Used as the beta=1.0 candidate and budget baseline."""
        if "pure" not in self.routes:
            self.routes["pure"] = shortest_route(self.graph, self.origin, self.destination, Weights(), avoid=False)
        return self.routes["pure"]

    def penalty(self, route: Route) -> float:
        return route.penalty_m(self.profile)


def route_within_budget(cache: RouteCache, budget_m: float, w_max: float, max_steps: int) -> tuple[Route, float]:
    """Route at the largest w whose length stays within budget_m.

    Cost is base + w * penalty (base = length unless avoid penalties are set), so
    route length is non-decreasing in w and the feasible w form a prefix [0, w*].
    Instead of blind bisection on w we bracket w*
    with a feasible and an infeasible route and probe the w where their costs tie:
    if Dijkstra returns one of the two, no other route is optimal in between and the
    feasible one is the answer; otherwise the new route tightens the bracket.
    Returns (route, w_switch), where w_switch is the tie point with the next route.
    """
    feasible = cache(0.0)
    if feasible.length_m > budget_m:
        return feasible, 0.0  # avoid penalties alone already exceed the budget: best effort
    infeasible = cache(w_max)
    if infeasible.length_m <= budget_m:
        return infeasible, w_max
    w = w_max
    for _ in range(max_steps):
        denominator = cache.penalty(feasible) - cache.penalty(infeasible)
        if denominator <= 0:
            break
        w = (infeasible.base_cost_m - feasible.base_cost_m) / denominator
        route = cache(w)
        if route.edge_set in (feasible.edge_set, infeasible.edge_set):
            break
        if route.length_m <= budget_m:
            feasible = route
        else:
            infeasible = route
    return feasible, w


def candidate_routes(
    graph: nx.MultiDiGraph,
    origin: object,
    destination: object,
    betas: tuple[float, ...] = BETAS,
    profile: Weights = PROFILE,
    w_max: float = W_MAX,
    max_steps: int = MAX_STEPS,
    waypoints: list[tuple[float, float]] | None = None,
    revisit_penalty: float = DEFAULT_REVISIT_PENALTY,
    keep_order: bool = False,
) -> tuple[Route, list[Candidate]]:
    if waypoints:
        return candidate_routes_with_waypoints(graph, origin, destination, list(waypoints), profile, w_max, max_steps, revisit_penalty, keep_order)
    cache = RouteCache(graph, origin, destination, profile)
    shortest = cache.pure_shortest()
    candidates: list[Candidate] = []
    for beta in sorted(betas):
        budget = beta * shortest.length_m
        if beta <= 1.0:
            route, w_switch = shortest, 0.0  # beta 1.0 is the true shortest path, reports and weights ignored
        else:
            route, w_switch = route_within_budget(cache, budget, w_max, max_steps)
        duplicate = next((c.beta for c in candidates if c.route.edge_set == route.edge_set), None)
        candidates.append(Candidate(beta=beta, route=route, budget_m=budget, w_switch=w_switch, budget_binding=w_switch < w_max, duplicate_of=duplicate))
    return shortest, candidates


def candidate_routes_with_waypoints(
    graph: nx.MultiDiGraph,
    origin: object,
    destination: object,
    waypoints: list[tuple[float, float]],
    profile: Weights = PROFILE,
    w_max: float = W_MAX,
    max_steps: int = MAX_STEPS,
    revisit_penalty: float = DEFAULT_REVISIT_PENALTY,
    keep_order: bool = False,
) -> tuple[Route, list[Candidate]]:
    """Origin -> waypoints (exact best order, or the given order with keep_order) -> destination, as two candidates (beta 1.0 and 1.15).

    Legs are computed one after another; edges walked by earlier legs are multiplied by
    `revisit_penalty` (both directions, all parallel keys) before the next leg runs, and the
    graph is restored afterwards (waypoints.walk_legs). The visit order is fixed once, from a
    pairwise matrix that ignores the revisit penalty (approximation), and shared by every beta.
    """
    # Lambda of the direct beta=1.15 route: the matrix / order are evaluated at this preference strength.
    direct_cache = RouteCache(graph, origin, destination, profile)
    direct_pure = direct_cache.pure_shortest()
    direct_route, _ = route_within_budget(direct_cache, MATRIX_BETA * direct_pure.length_m, w_max, max_steps)
    direct_lambda = next((f for f, r in direct_cache.routes.items() if r is direct_route and f != "pure"), 0.0)
    wp = plan(graph, origin, destination, waypoints, profile, float(direct_lambda), keep_order)
    seq = wp.nodes

    # beta 1.0: pure shortest per leg (no weights, no avoid penalties) - no lambda search, one Dijkstra per leg.
    table0 = PenalizedTable(graph, Weights(), avoid=False, multiplier=revisit_penalty)
    table0.apply(0.0)
    legs0 = walk_legs(graph, table0, seq, lambda a, b: dijkstra_route(graph, a, b, Weights()))
    shortest = concat_routes(graph, legs0, Weights())

    # beta 1.15: per-leg lambda search (route_within_budget, unchanged) over one shared penalised table.
    table = PenalizedTable(graph, profile, avoid=True, multiplier=revisit_penalty)
    leg_cache = RouteCache(graph, origin, destination, profile)
    leg_cache.table = table  # the search re-costs the graph through this table, so the bumps survive every apply()
    switches: list[float] = []

    def leg(a: object, b: object) -> Route:
        leg_cache.origin, leg_cache.destination, leg_cache.routes = a, b, {}
        budget = WAYPOINT_BETAS[1] * leg_cache.pure_shortest().length_m  # each leg gets its own 15 % budget
        route, w = route_within_budget(leg_cache, budget, w_max, max_steps)
        switches.append(w)
        return route

    legs1 = walk_legs(graph, table, seq, leg)
    balanced = concat_routes(graph, legs1, profile)

    extras = {"visit_order": wp.visit_order, "waypoint_nodes": seq[1:-1]}
    candidates = [
        Candidate(beta=WAYPOINT_BETAS[0], route=shortest, budget_m=shortest.length_m, w_switch=0.0, budget_binding=True, duplicate_of=None,
                  revisit_ratio=revisit_ratio(graph, shortest), detour_m=round(shortest.length_m - wp.direct.length_m, 1), **extras),
        Candidate(beta=WAYPOINT_BETAS[1], route=balanced, budget_m=WAYPOINT_BETAS[1] * shortest.length_m, w_switch=min(switches, default=w_max),
                  budget_binding=any(w < w_max for w in switches), duplicate_of=WAYPOINT_BETAS[0] if balanced.edge_set == shortest.edge_set else None,
                  revisit_ratio=revisit_ratio(graph, balanced), detour_m=round(balanced.length_m - wp.direct.length_m, 1), **extras),
    ]
    return shortest, candidates


def summary_rows(shortest: Route, candidates: list[Candidate]) -> list[dict[str, object]]:
    def reduction(value: float, base: float) -> float:
        return (1 - value / base) * 100 if base else 0.0

    def row(beta: object, route: Route, w_switch: float | None, binding: bool, duplicate: float | None) -> dict[str, object]:
        return {
            "beta": beta, "weights": route.weights, "w_switch": w_switch, "length_m": route.length_m,
            "distance_ratio": route.length_m / shortest.length_m,
            "exposure_m": route.exposure_m, "tree_m": route.tree_m, "park_m": route.park_m, "vibe_m": route.vibe_m,
            "exposure_reduction_pct": reduction(route.exposure_m, shortest.exposure_m),
            "tree_reduction_pct": reduction(route.tree_m, shortest.tree_m),
            "park_reduction_pct": reduction(route.park_m, shortest.park_m),
            "vibe_reduction_pct": reduction(route.vibe_m, shortest.vibe_m),
            "budget_binding": binding, "duplicate_of": duplicate,
        }

    return [row("shortest", shortest, None, True, None)] + [row(c.beta, c.route, c.w_switch, c.budget_binding, c.duplicate_of) for c in candidates]


def candidates_geojson(graph: nx.MultiDiGraph, shortest: Route, candidates: list[Candidate]) -> dict[str, object]:
    """FeatureCollection with one LineString per candidate, in beta order."""
    def reduction(value: float, base: float) -> float:
        return round((1 - value / base) * 100, 1) if base else 0.0

    features = [
        route_feature(
            graph, c.route,
            beta=c.beta, label=LABELS.get(c.beta, f"β={c.beta}"), rank=rank,
            distance_ratio=round(c.route.length_m / shortest.length_m, 4),
            exposure_reduction_pct=reduction(c.route.exposure_m, shortest.exposure_m),
            tree_reduction_pct=reduction(c.route.tree_m, shortest.tree_m),
            park_reduction_pct=reduction(c.route.park_m, shortest.park_m),
            vibe_reduction_pct=reduction(c.route.vibe_m, shortest.vibe_m),
            w_switch=round(c.w_switch, 4), budget_binding=c.budget_binding, duplicate_of=c.duplicate_of,
            **({"revisit_ratio": c.revisit_ratio, "detour_m": c.detour_m, "detour_min": round(c.detour_m / WALK_M_PER_MIN, 1), "visit_order": c.visit_order} if c.revisit_ratio is not None else {}),
        )
        for rank, c in enumerate(candidates)
    ]
    collection: dict[str, object] = {"type": "FeatureCollection", "features": features, "shortest_length_m": round(shortest.length_m, 1)}
    if candidates and candidates[0].visit_order is not None:
        collection["visit_order"] = candidates[0].visit_order
        collection["waypoints"] = [[graph.nodes[n]["x"], graph.nodes[n]["y"]] for n in candidates[0].waypoint_nodes or []]
    return collection


def save_map(graph: nx.MultiDiGraph, shortest: Route, candidates: list[Candidate], origin: tuple[float, float], destination: tuple[float, float], path: Path) -> None:
    route_map = folium.Map(
        location=[(origin[0] + destination[0]) / 2, (origin[1] + destination[1]) / 2],
        zoom_start=15, control_scale=True,
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", attr="CartoDB",
    )
    legend = []
    layers = [("shortest", "#2563eb", shortest)] + [(f"beta={c.beta}", color, c.route) for c, color in zip(candidates, COLORS)]
    for label, color, route in layers:
        latlon = [(lat, lon) for lon, lat in route_coordinates(graph, route.edges)]
        folium.PolyLine(latlon, color=color, weight=5, opacity=0.8, tooltip=label).add_to(route_map)
        legend.append(f"<span style='color:{color}'>&#9632;</span> {label}: {route.length_m:,.0f}m ({route.length_m / shortest.length_m:.3f}), E={route.exposure_m:,.0f}m, T={route.tree_m:,.0f}m, P={route.park_m:,.0f}m, V={route.vibe_m:,.0f}m")
    folium.Marker(origin, icon=folium.Icon(color="green", icon="play")).add_to(route_map)
    folium.Marker(destination, icon=folium.Icon(color="red", icon="flag")).add_to(route_map)
    route_map.get_root().html.add_child(folium.Element("<div style='position: fixed; bottom: 28px; left: 28px; z-index: 1000; background: white; padding: 10px 12px; border: 1px solid #777; border-radius: 4px;'>" + "<br>".join(legend) + "</div>"))
    route_map.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detour-budget candidate routes via bisection on the exposure weight w.")
    parser.add_argument("--origin", type=float, nargs=2, default=ORIGIN, metavar=("LAT", "LON"))
    parser.add_argument("--destination", type=float, nargs=2, default=DESTINATION, metavar=("LAT", "LON"))
    parser.add_argument("--betas", type=float, nargs="+", default=BETAS)
    parser.add_argument("--w-vehicle", type=float, default=PROFILE.w_vehicle)
    parser.add_argument("--w-tree", type=float, default=PROFILE.w_tree)
    parser.add_argument("--w-park", type=float, default=PROFILE.w_park)
    parser.add_argument("--w-vibe", type=float, default=PROFILE.w_vibe)
    parser.add_argument("--map", type=Path, default=MAP_PATH)
    parser.add_argument("--geojson", type=Path, default=None, help="also write the candidates as a GeoJSON FeatureCollection")
    args = parser.parse_args()

    graph = load_graph()
    origin, destination = tuple(args.origin), tuple(args.destination)
    profile = Weights(args.w_vehicle, args.w_tree, args.w_park, args.w_vibe)
    shortest, candidates = candidate_routes(graph, nearest_node(graph, origin), nearest_node(graph, destination), tuple(args.betas), profile)
    for row in summary_rows(shortest, candidates):
        print(row)
    save_map(graph, shortest, candidates, origin, destination, args.map)
    print(f"Saved {args.map}")
    if args.geojson:
        args.geojson.write_text(json.dumps(candidates_geojson(graph, shortest, candidates), ensure_ascii=False), encoding="utf-8")
        print(f"Saved {args.geojson}")


if __name__ == "__main__":
    main()
