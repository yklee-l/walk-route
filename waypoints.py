"""Waypoint routing layered on top of routing.py / candidate_routes.py.

Nothing here changes how costs are defined: edge costs are still written into COST_KEY
("_route_cost") and Dijkstra runs with weight=COST_KEY. This module only decides the
visit order, runs the legs one after another, and raises the cost of already-walked
edges between legs so the walk does not double back on itself.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import networkx as nx

from routing import (
    COST_KEY, CostTable, Edge, Route, Weights,
    edge_base_cost, edge_exposure, edge_park_penalty, edge_tree_penalty, edge_vibe_penalty,
    nearest_node, number, shortest_route,
)


MAX_WAYPOINTS = 6
SNAP_MAX_M = 300.0  # a waypoint further than this from any graph node is outside the covered area
WAYPOINT_BETAS = (1.0, 1.15)  # with waypoints the lambda search cost multiplies by the number of legs, so two candidates
MATRIX_BETA = 1.15  # the pairwise matrix / visit order is computed at the lambda of this beta on the direct route
DEFAULT_REVISIT_PENALTY = 3.0


class WaypointError(ValueError):
    """Bad waypoint input (out of range, unreachable, too many)."""


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Metres between two (lat, lon) points."""
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6_371_000 * math.asin(math.sqrt(h))


# ---------------------------------------------------------------------------
# Revisit penalty on top of CostTable
# ---------------------------------------------------------------------------
class PenalizedTable(CostTable):
    """CostTable plus a multiplier on edges that earlier legs already used.

    CostTable.apply() rewrites COST_KEY on every edge for a new lambda, which would erase any
    penalty written into the graph. This subclass re-applies the multiplier to the bumped edges
    after every apply(), so the lambda search of candidate_routes.route_within_budget keeps
    seeing the penalty without that code knowing about it.

    bump() raises every parallel key of (u, v, *) and the reverse (v, u, *): for a pedestrian
    they are the same alley, and leaving out the reverse direction halves the effect.
    Each edge is multiplied once, never compounded, so a dead-end waypoint (where doubling
    back is unavoidable) costs at most `multiplier` times its length instead of forcing an
    unbounded detour.
    """

    def __init__(self, graph: nx.MultiDiGraph, profile: Weights, avoid: bool = True, multiplier: float = DEFAULT_REVISIT_PENALTY) -> None:
        super().__init__(graph, profile, avoid)
        self.graph = graph
        self.multiplier = multiplier
        self.factor = 0.0
        # CostTable iterates graph.edges(keys=True, data=True) in graph order; the same iteration gives the edge ids.
        self.row_of: dict[Edge, tuple[dict[str, object], float, float]] = {
            (u, v, k): row for (u, v, k), row in zip(graph.edges(keys=True), self.rows)
        }
        self.bumped: set[Edge] = set()

    def apply(self, factor: float) -> None:
        super().apply(factor)
        self.factor = factor
        for edge in self.bumped:  # re-apply after the full rewrite; only the touched edges
            self.row_of[edge][0][COST_KEY] *= self.multiplier

    def bump(self, edges: list[Edge]) -> None:
        """Raise COST_KEY on the given edges, all their parallel keys, and the reverse direction."""
        if self.multiplier == 1.0:
            return
        for u, v, _ in edges:
            for a, b in ((u, v), (v, u)):
                if not self.graph.has_edge(a, b):
                    continue
                for k in self.graph[a][b]:
                    edge = (a, b, k)
                    if edge in self.bumped:
                        continue
                    self.bumped.add(edge)
                    self.row_of[edge][0][COST_KEY] *= self.multiplier

    def restore(self) -> None:
        """Put the plain (un-bumped) cost for the current factor back on every bumped edge."""
        for edge in self.bumped:
            data, base, penalty = self.row_of[edge]
            data[COST_KEY] = base + self.factor * penalty
        self.bumped.clear()


# ---------------------------------------------------------------------------
# Route helpers that do NOT rewrite costs (the caller has already applied them)
# ---------------------------------------------------------------------------
def route_from_edges(graph: nx.MultiDiGraph, nodes: list[object], edges: list[Edge], weights: Weights) -> Route:
    datas = [graph[u][v][k] for u, v, k in edges]
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


def dijkstra_route(graph: nx.MultiDiGraph, origin: object, destination: object, weights: Weights) -> Route:
    """Shortest path on whatever is currently in COST_KEY (costs must already be applied)."""
    nodes = nx.shortest_path(graph, origin, destination, weight=COST_KEY)
    edges = [(u, v, min(graph[u][v], key=lambda k: graph[u][v][k][COST_KEY])) for u, v in zip(nodes, nodes[1:])]
    return route_from_edges(graph, nodes, edges, weights)


def concat_routes(graph: nx.MultiDiGraph, legs: list[Route], weights: Weights) -> Route:
    nodes = list(legs[0].nodes) + [n for leg in legs[1:] for n in leg.nodes[1:]]
    edges = [e for leg in legs for e in leg.edges]
    return route_from_edges(graph, nodes, edges, weights)


def revisit_ratio(graph: nx.MultiDiGraph, route: Route) -> float:
    """Share of the walked length on edges (either direction) that the route passes more than once."""
    if not route.edges:
        return 0.0
    undirected = lambda u, v, k: (min(u, v), max(u, v), k)  # noqa: E731
    counts: dict[tuple, int] = {}
    for u, v, k in route.edges:
        counts[undirected(u, v, k)] = counts.get(undirected(u, v, k), 0) + 1
    repeated = sum(graph[u][v][k]["length"] for u, v, k in route.edges if counts[undirected(u, v, k)] >= 2)
    return round(repeated / route.length_m, 4) if route.length_m else 0.0


# ---------------------------------------------------------------------------
# Snapping, distance matrix, visit order
# ---------------------------------------------------------------------------
def snap_waypoints(graph: nx.MultiDiGraph, waypoints: list[tuple[float, float]], origin: object, destination: object) -> tuple[list[object], list[int]]:
    """Waypoint (lat, lon) -> graph nodes. Drops duplicates of origin/destination/each other; keeps input indices."""
    if len(waypoints) > MAX_WAYPOINTS:
        raise WaypointError(f"경유지는 최대 {MAX_WAYPOINTS}개까지 지정할 수 있습니다.")
    nodes: list[object] = []
    kept: list[int] = []
    seen = {origin, destination}
    for i, latlon in enumerate(waypoints):
        lat, lon = float(latlon[0]), float(latlon[1])
        node = nearest_node(graph, (lat, lon))
        gap = haversine_m((lat, lon), (number(graph.nodes[node]["y"]), number(graph.nodes[node]["x"])))
        if gap > SNAP_MAX_M:
            raise WaypointError(f"경유지 {i + 1}이(가) 서비스 범위(광화문 2 km) 밖입니다 (가장 가까운 도로까지 {gap:,.0f} m).")
        if node in seen:
            continue  # same node as the origin/destination or an earlier waypoint: nothing to visit
        seen.add(node)
        nodes.append(node)
        kept.append(i)
    return nodes, kept


def cost_matrix(graph: nx.MultiDiGraph, points: list[object]) -> list[list[float]]:
    """Pairwise costs on the currently applied COST_KEY, one one-to-all Dijkstra per point (k + 2 calls)."""
    matrix: list[list[float]] = []
    for source in points:
        dist = nx.single_source_dijkstra_path_length(graph, source, weight=COST_KEY)
        matrix.append([dist.get(target, math.inf) for target in points])
    return matrix


def visit_order(matrix: list[list[float]]) -> list[int]:
    """Exact fixed-endpoint path TSP over the inner points (index 0 = origin, last = destination).

    k <= 6 inner points -> at most 720 permutations, so brute force is exact and cheap.
    Returns the inner indices (1..k) in visiting order.
    """
    inner = list(range(1, len(matrix) - 1))
    end = len(matrix) - 1
    best, best_cost = list(inner), math.inf
    for perm in itertools.permutations(inner):
        seq = [0, *perm, end]
        cost = sum(matrix[a][b] for a, b in zip(seq, seq[1:]))
        if cost < best_cost:
            best, best_cost = list(perm), cost
    if not math.isfinite(best_cost):
        raise WaypointError("경유지 중 도보로 연결되지 않는 지점이 있습니다.")
    return best


# ---------------------------------------------------------------------------
# Main entry: legs in order, revisit penalty between legs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WaypointPlan:
    nodes: list[object]  # origin, waypoints in visit order, destination
    visit_order: list[int]  # input indices of the waypoints in visit order (dropped duplicates omitted)
    direct: Route  # origin -> destination with no waypoints, pure shortest (for detour_m)


def plan(graph: nx.MultiDiGraph, origin: object, destination: object, waypoints: list[tuple[float, float]], profile: Weights, direct_lambda: float, keep_order: bool = False) -> WaypointPlan:
    """Snap, build the matrix at `direct_lambda`, solve the order. Steps 2-3 ignore the revisit penalty (approximation).

    keep_order=True skips the TSP and visits the waypoints in the given order (the page's drag order).
    """
    wp_nodes, kept = snap_waypoints(graph, waypoints, origin, destination)
    direct = shortest_route(graph, origin, destination, Weights(), avoid=False)  # pure shortest, reused for detour_m
    if not wp_nodes:
        return WaypointPlan(nodes=[origin, destination], visit_order=[], direct=direct)
    if keep_order:
        return WaypointPlan(nodes=[origin, *wp_nodes, destination], visit_order=list(kept), direct=direct)
    # Matrix and order use the plain (un-penalised) cost at the lambda the direct beta=1.15 route was found at.
    # This is an approximation: the revisit penalty applied while walking the legs is not known yet.
    table = CostTable(graph, profile)
    table.apply(direct_lambda)
    points = [origin, *wp_nodes, destination]
    order = visit_order(cost_matrix(graph, points))
    return WaypointPlan(nodes=[origin, *(wp_nodes[i - 1] for i in order), destination], visit_order=[kept[i - 1] for i in order], direct=direct)


def walk_legs(graph: nx.MultiDiGraph, table: PenalizedTable, seq: list[object], leg_fn) -> list[Route]:
    """Run leg_fn(a, b) for each consecutive pair, bumping used edges in between; always restores the graph."""
    legs: list[Route] = []
    try:
        for a, b in zip(seq, seq[1:]):
            leg = leg_fn(a, b)
            legs.append(leg)
            table.bump(leg.edges)  # everything walked so far costs more for the next leg
    finally:
        table.restore()  # the graph is shared between requests: never leave raised costs behind
    return legs
