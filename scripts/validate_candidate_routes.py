from __future__ import annotations

from pathlib import Path

import pandas as pd

from candidate_routes import BETAS, candidate_routes, summary_rows
from routing import load_graph


OD_PATH = Path("osmnx_outputs/gwanghwamun_route_overlap_100_pairs.csv")
DETAIL_PATH = Path("osmnx_outputs/gwanghwamun_candidates_100_pairs.csv")
SUMMARY_PATH = Path("osmnx_outputs/gwanghwamun_candidates_summary.csv")


def main() -> None:
    graph = load_graph()
    od_pairs = pd.read_csv(OD_PATH)[["pair_id", "origin", "destination"]]
    rows: list[dict[str, object]] = []
    for _, pair in od_pairs.iterrows():
        shortest, candidates = candidate_routes(graph, pair["origin"], pair["destination"], BETAS)
        distinct = len({c.route.edge_set for c in candidates})
        for row in summary_rows(shortest, candidates)[1:]:
            rows.append({"pair_id": int(pair["pair_id"]), "distinct_candidates": distinct, **row})
    detail = pd.DataFrame(rows)
    detail["is_duplicate"] = detail["duplicate_of"].notna()
    detail["hits_budget"] = detail["distance_ratio"] >= detail["beta"] - 0.05  # within 5%p of the budget line
    summary = detail.groupby("beta", as_index=False).agg(
        distance_ratio_median=("distance_ratio", "median"),
        distance_ratio_p90=("distance_ratio", lambda s: s.quantile(0.9)),
        exposure_reduction_pct_median=("exposure_reduction_pct", "median"),
        budget_binding_pct=("budget_binding", "mean"),
        duplicate_pct=("is_duplicate", "mean"),
        hits_budget_pct=("hits_budget", "mean"),
    )
    for column in ("budget_binding_pct", "duplicate_pct", "hits_budget_pct"):
        summary[column] *= 100
    distinct_counts = detail.drop_duplicates("pair_id")["distinct_candidates"].value_counts().sort_index()
    detail.drop(columns="is_duplicate").to_csv(DETAIL_PATH, index=False, encoding="utf-8-sig")
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print("distinct candidates per OD pair (count of pairs):")
    print(distinct_counts.to_string())
    print(f"Saved {DETAIL_PATH}")
    print(f"Saved {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
