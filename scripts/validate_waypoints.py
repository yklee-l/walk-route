"""Waypoint routing check: revisit ratio with the penalty off (1.0) vs on (3.0), plus timing.

Real Gwanghwamun-area coordinates; origin 경복궁, destination 명동. Run after any change to
waypoints.py / candidate_routes.py. Prints one block per waypoint count and the 4-waypoint
response time (the routing part only, no HTTP).
"""
from __future__ import annotations

import sys
import time

from candidate_routes import candidate_routes
from routing import COST_KEY, Weights, load_graph, nearest_node
from waypoints import PenalizedTable

ORIGIN = (37.5796, 126.9770)  # 경복궁
DESTINATION = (37.5636, 126.9850)  # 명동
WAYPOINTS = [
    (37.5704, 126.9769),  # 광화문광장
    (37.5744, 126.9847),  # 인사동
    (37.5680, 126.9997),  # 청계천 (삼일교 부근)
    (37.5788, 126.9829),  # 북촌 입구
]
PROFILE = Weights(w_vehicle=3, w_vibe=1)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    graph = load_graph()
    origin, destination = nearest_node(graph, ORIGIN), nearest_node(graph, DESTINATION)
    timings: dict[int, float] = {}
    for k in (2, 3, 4):
        wps = WAYPOINTS[:k]
        print(f"\n== {k} waypoints ==")
        for penalty in (1.0, 3.0):
            t = time.time()
            shortest, cands = candidate_routes(graph, origin, destination, profile=PROFILE, waypoints=wps, revisit_penalty=penalty)
            dt = time.time() - t
            timings[k] = max(timings.get(k, 0.0), dt)
            for c in cands:
                print(f"  penalty {penalty:.1f}  beta {c.beta:<5} {c.route.length_m:7.0f} m  revisit {c.revisit_ratio:.3f}  detour +{c.detour_m:.0f} m  order {c.visit_order}" + ("  (dup)" if c.duplicate_of else ""))
            print(f"  penalty {penalty:.1f}  time {dt:.2f}s")
    # restore() must put every bumped edge back exactly: snapshot -> bump a route's edges -> restore -> compare
    table = PenalizedTable(graph, PROFILE, avoid=True, multiplier=3.0)
    table.apply(1.0)
    before = {e: graph[e[0]][e[1]][e[2]][COST_KEY] for e in graph.edges(keys=True)}
    table.bump(shortest.edges)
    changed = sum(1 for e, v in before.items() if graph[e[0]][e[1]][e[2]][COST_KEY] != v)
    table.restore()
    dirty = sum(1 for e, v in before.items() if abs(graph[e[0]][e[1]][e[2]][COST_KEY] - v) > 1e-9)
    print(f"\nbump touched {changed} edge keys (both directions, all parallel keys) for {len(shortest.edges)} route edges; leftover after restore: {dirty} (expected 0)")
    print(f"4-waypoint worst time: {timings[4]:.2f}s" + ("  <-- over the 2 s target" if timings[4] > 2 else ""))


if __name__ == "__main__":
    main()
