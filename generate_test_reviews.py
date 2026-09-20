"""Synthetic segment reviews for demos and for testing the ranking (GET /api/ranking).

Picks 20-30 street stretches inside the 광화문 2 km graph, splits them into "good" (인사동길, 청계천 side
streets, quiet footways and alleys with a high vibe score) and "bad" (primary/secondary roads with heavy
traffic), and writes 3-7 reviews per stretch from user1..user10 into osmnx_outputs/reviews.json, in the
same shape server.py stores them. Every stretch is a run of edges on one street (same name + highway),
which is what the page's segment reviews and the ranking's grouping expect.

    python generate_test_reviews.py            # replace reviews.json (the old file is kept as a .bak)
    python generate_test_reviews.py --append   # keep the existing reviews and add the synthetic ones
    python generate_test_reviews.py --seed 7 --count 25

Restart server.py afterwards: it replays reviews.json at startup only.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import networkx as nx

from routing import Edge, edge_name, load_graph, number
from score_gwanghwamun_edges import highway_values


REVIEWS_PATH = Path("osmnx_outputs/reviews.json")
PROFILES_PATH = Path("osmnx_outputs/profiles.json")
USERS = [f"user{i}" for i in range(1, 11)]
# Each dummy user gets a fixed taste (the page's presets: purpose -> sub, from web/index.html PURPOSES); their
# reviews carry it in profile.purpose / subs, so the review list's "취향 일치순" and the admin stats have data.
PURPOSE_SUBS = {"date": ["quiet", "hot", "photo", "park"], "solo": ["quiet", "lively", "food"], "family": ["stroller", "park", "sights"], "tour": ["photo", "sights", "food"], "fast": []}
SUB_WEIGHTS = {  # [w_vehicle, w_vibe, w_park] as the page stores them (w_tree stays 0)
    ("date", "quiet"): [3, 2, 0], ("date", "hot"): [1, 4, 0], ("date", "photo"): [1.5, 3, 0], ("date", "park"): [3, 1, 2],
    ("solo", "quiet"): [4, 0.5, 0], ("solo", "lively"): [0.5, 3, 0], ("solo", "food"): [1, 3.5, 0],
    ("family", "stroller"): [3, 1, 0], ("family", "park"): [3, 1, 2], ("family", "sights"): [1.5, 3, 0],
    ("tour", "photo"): [1, 3.5, 0], ("tour", "sights"): [1, 2.5, 0], ("tour", "food"): [0.5, 4, 0], ("fast", None): [0, 0, 0],
}
AVOID_CHOICES = ["공사중", "냄새", "더러움", "어두움", "좁고 답답함", "무서움"]
PURPOSES = list(PURPOSE_SUBS)
DAYS_BACK = 90  # timestamps uniform over the last 90 days, so the server's 30-day half-life decay is visible (1.0 .. 0.125)
RUN_EDGES = (3, 8)  # edges per stretch
REVIEWS_PER_STRETCH = (3, 7)  # >= 3 so every stretch can enter the ranking

GOOD_NAMES = ("인사동", "청계천", "삼청", "북촌", "돈화문", "서순라", "율곡로1길", "관훈", "공평")
BAD_HIGHWAYS = {"primary", "secondary", "primary_link", "secondary_link", "trunk"}
GOOD_TAGS = ["분위기 좋아요", "한산해요", "인생샷 좋아요", "맛집 많아요", "여러명 걷기 좋아요", "포장 좋아요", "자전거 도로 있어요"]
BAD_TAGS = ["차 많음", "무서움", "어두움", "사람 붐빔", "좁고 답답함"]
GOOD_COMMENTS = [
    "조용하고 걷기 좋아요", "가로수 그늘이 있어서 시원해요", "카페 골목이라 분위기가 좋네요", "사진 찍기 좋은 곳이 많아요",
    "차가 거의 안 다녀서 편해요", "밤에도 산책하기 좋았어요", "골목 구경하는 재미가 있어요", "천천히 걷기 딱 좋은 길",
    "데이트 코스로 추천해요", "맛집이 많아서 들르기 좋아요", None, None,
]
BAD_COMMENTS = [
    "차가 너무 많아서 시끄러워요", "밤에는 어두워서 무서웠어요", "매연이 심하고 인도가 좁아요", "횡단보도 기다리는 시간이 길어요",
    "큰길이라 산책 느낌이 안 나요", "사람이 너무 많아서 걷기 힘들어요", "공사 소음이 심해요", None, None,
]


def street_runs(graph: nx.MultiDiGraph) -> dict[tuple[str | None, str], list[Edge]]:
    """Edges grouped by street (name + highway), one direction per stretch (u < v keeps a single copy)."""
    streets: dict[tuple[str | None, str], list[Edge]] = defaultdict(list)
    seen: set[tuple[object, object, object]] = set()
    for u, v, k, data in graph.edges(keys=True, data=True):
        key = (min(u, v), max(u, v), k)
        if key in seen:
            continue
        seen.add(key)
        streets[(edge_name(data), "|".join(sorted(highway_values(data.get("highway")))))].append((u, v, k))
    return streets


def contiguous_run(graph: nx.MultiDiGraph, edges: list[Edge], rng: random.Random, size: int, tries: int = 5) -> list[Edge]:
    """A connected chain of up to `size` edges from one street: the longest of `tries` chains grown from random edges."""
    by_node: dict[object, list[Edge]] = defaultdict(list)
    for e in edges:
        by_node[e[0]].append(e)
        by_node[e[1]].append(e)
    best: list[Edge] = []
    for _ in range(tries):
        run = grow_run(by_node, rng.choice(edges), rng, size)
        if len(run) > len(best):
            best = run
        if len(best) >= size:
            break
    return best


def grow_run(by_node: dict[object, list[Edge]], start: Edge, rng: random.Random, size: int) -> list[Edge]:
    run, used, ends = [start], {start}, [start[0], start[1]]
    while len(run) < size:
        rng.shuffle(ends)
        grown = False
        for node in ends:
            nxt = [e for e in by_node[node] if e not in used]
            if nxt:
                e = rng.choice(nxt)
                run.append(e)
                used.add(e)
                ends[ends.index(node)] = e[1] if e[0] == node else e[0]
                grown = True
                break
        if not grown:
            break
    return run


def classify(graph: nx.MultiDiGraph, name: str | None, highway: str, edges: list[Edge]) -> str | None:
    """'good' / 'bad' / None (skip) for a street, from its name, road class and the scored edge attributes."""
    datas = [graph[u][v][k] for u, v, k in edges]
    length = sum(d["length"] for d in datas) or 1.0
    vehicle = sum(number(d.get("vehicle_exposure_score")) * d["length"] for d in datas) / length
    vibe = sum(number(d.get("vibe_score")) * d["length"] for d in datas) / length
    park = sum(number(d.get("park_distance_m"), 300.0) * d["length"] for d in datas) / length
    highways = set(highway.split("|"))
    if "steps" in highways:
        return None
    if highways & BAD_HIGHWAYS and vehicle >= 3:
        return "bad"
    if name and any(kw in name for kw in GOOD_NAMES) and vehicle <= 2:
        return "good"
    if highways <= {"footway", "pedestrian", "residential", "living_street", "path"} and (vibe >= 0.5 or park <= 60) and vehicle <= 1:
        return "good"
    return None


def pick_stretches(graph: nx.MultiDiGraph, rng: random.Random, count: int) -> list[tuple[str, str | None, str, list[Edge]]]:
    """(kind, name, highway, edges) for `count` stretches, about 60 % good / 40 % bad, named streets preferred."""
    streets = street_runs(graph)
    pools: dict[str, list[tuple[str | None, str, list[Edge]]]] = {"good": [], "bad": []}
    for (name, highway), edges in streets.items():
        if len(edges) < RUN_EDGES[0]:
            continue
        kind = classify(graph, name, highway, edges)
        if kind:
            pools[kind].append((name, highway, edges))
    # Named 인사동/청계천-style streets first so the demo shows recognisable places, then the rest at random.
    for kind in pools:
        rng.shuffle(pools[kind])
        pools[kind].sort(key=lambda s: not (s[0] and any(kw in s[0] for kw in GOOD_NAMES)) if kind == "good" else s[0] is None)
    n_good = round(count * 0.6)
    chosen: list[tuple[str, str | None, str, list[Edge]]] = []
    for kind, n in (("good", n_good), ("bad", count - n_good)):
        for name, highway, edges in pools[kind][:n]:
            chosen.append((kind, name, highway, contiguous_run(graph, edges, rng, rng.randint(*RUN_EDGES))))
    rng.shuffle(chosen)
    return chosen


def make_profiles(rng: random.Random) -> dict[str, dict[str, object]]:
    """user1..user10 -> profile in server.ProfileStore's shape (what POST /api/profile/{nickname} stores)."""
    profiles: dict[str, dict[str, object]] = {}
    for i, user in enumerate(USERS):
        purpose = PURPOSES[i % len(PURPOSES)] if i < len(PURPOSES) else rng.choice(PURPOSES)  # every purpose at least once
        sub = rng.choice(PURPOSE_SUBS[purpose]) if PURPOSE_SUBS[purpose] else None
        w = SUB_WEIGHTS[(purpose, sub)]
        profiles[user] = {
            "mode": "buttons", "purpose": purpose, "subs": [sub] if sub else [],
            "avoid": rng.sample(AVOID_CHOICES, rng.choice([0, 1, 1, 2])),
            "weights": {"w_vehicle": w[0], "w_tree": 0, "w_park": w[2], "w_vibe": w[1]},
            "avoid_steps": purpose == "family" and sub == "stroller",
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() - rng.uniform(0, 90) * 86400)),
        }
    return profiles


def make_review(graph: nx.MultiDiGraph, kind: str, edges: list[Edge], rng: random.Random, now: float, profile: dict[str, object] | None = None) -> dict[str, object]:
    good = kind == "good"
    score = float(rng.randint(7, 10)) if good else float(rng.randint(1, 3))
    tags_pool = GOOD_TAGS if good else BAD_TAGS
    tags = rng.sample(tags_pool, rng.choice([0, 1, 1, 2, 2, 3]))
    if not good and tags and rng.random() < 0.7:  # the brief: 무서움/어두움/차 많음 dominate the bad ones
        tags[0] = rng.choice(["차 많음", "무서움", "어두움"])
    tags = list(dict.fromkeys(tags))
    days = rng.uniform(0, DAYS_BACK)
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now - days * 86400 - rng.randint(0, 3600 * 12)))
    purpose = str(profile["purpose"]) if profile else rng.choice(PURPOSES)
    subs = list(profile.get("subs", [])) if profile else []
    first, last = graph.nodes[edges[0][0]], graph.nodes[edges[-1][1]]
    return {
        "ts": ts, "user_id": rng.choice(USERS), "kind": "segment",
        "edge_id": ",".join(f"{u}-{v}-{k}" for u, v, k in edges),
        "score": score, "good_tags": tags if good else [], "bad_tags": [] if good else tags,
        # A review is left while looking at a route; approximate that route's ends with points around the stretch.
        "origin": [round(number(first["y"]) + rng.uniform(-0.004, 0.004), 6), round(number(first["x"]) + rng.uniform(-0.004, 0.004), 6)],
        "destination": [round(number(last["y"]) + rng.uniform(-0.004, 0.004), 6), round(number(last["x"]) + rng.uniform(-0.004, 0.004), 6)],
        "preset_tags": [purpose] + [f"{purpose}/{sb}" for sb in subs], "profile": {"purpose": purpose, "subs": subs},
        "photos": [], "comment": rng.choice(GOOD_COMMENTS if good else BAD_COMMENTS),
        "synthetic": True,
    }


def danger_reviews(graph: nx.MultiDiGraph, rng: random.Random, stretches: list[tuple[str, str | None, str, list[Edge]]], count: int, profiles: dict[str, dict[str, object]], now: float) -> list[dict[str, object]]:
    """`count` bad stretches get 4-6 fresh (<= 7 days old) 무서움/어두움 reviews each, so their decayed sum clears the
    danger threshold (3) and the ⚠️ 위험 구간 toggle has something to show."""
    bad = [st for st in stretches if st[0] == "bad"]
    rng.shuffle(bad)
    out: list[dict[str, object]] = []
    for _, _, _, edges in bad[:count]:
        for user in rng.sample(USERS, rng.randint(4, 6)):
            review = make_review(graph, "bad", edges, rng, now, profiles.get(user))
            review["user_id"] = user
            review["bad_tags"] = list(dict.fromkeys([rng.choice(["무서움", "어두움"]), *[t for t in review["bad_tags"] if t not in ("무서움", "어두움")][:1]]))
            review["score"] = float(rng.randint(1, 3))
            review["comment"] = rng.choice(["밤에 너무 어두워서 무서웠어요", "가로등이 없어서 혼자 걷기 무서워요", "인적이 드물고 어두워요", "저녁에는 피하는 게 좋겠어요", None])
            days = rng.uniform(0, 7)
            review["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now - days * 86400 - rng.randint(0, 3600 * 12)))
            out.append(review)
    return out


def generate(graph: nx.MultiDiGraph, rng: random.Random, count: int, profiles: dict[str, dict[str, object]] | None = None, danger: int = 0) -> tuple[list[dict[str, object]], list[tuple[str, str | None, str, list[Edge]]]]:
    """(reviews, stretches): `count` stretches x 3-7 reviews each, reviews sorted by timestamp.

    `profiles` (user -> profile, see make_profiles) gives each review its author's purpose / subs.
    Also used by the admin page (POST /api/admin/data/generate).
    """
    profiles = profiles if profiles is not None else make_profiles(rng)
    stretches = pick_stretches(graph, rng, count)
    now = time.time()
    reviews: list[dict[str, object]] = []
    for kind, _, _, edges in stretches:
        # Each user reviews a stretch at most once, like the page allows.
        for user in rng.sample(USERS, rng.randint(*REVIEWS_PER_STRETCH)):
            review = make_review(graph, kind, edges, rng, now, profiles.get(user))
            review["user_id"] = user
            reviews.append(review)
    if danger:
        reviews.extend(danger_reviews(graph, rng, stretches, danger, profiles, now))
    reviews.sort(key=lambda r: str(r["ts"]))
    return reviews, stretches


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Korean street names on a cp949 console
    parser = argparse.ArgumentParser(description="Generate synthetic segment reviews into reviews.json.")
    parser.add_argument("--count", type=int, default=None, help="number of stretches (default: random 20-30)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--append", action="store_true", help="keep the existing reviews.json entries")
    parser.add_argument("--out", type=Path, default=REVIEWS_PATH)
    parser.add_argument("--profiles", type=Path, default=PROFILES_PATH, help="profiles.json to add user1..user10 to (other nicknames are kept)")
    parser.add_argument("--no-profiles", action="store_true", help="do not touch profiles.json")
    parser.add_argument("--danger", type=int, default=0, help="also give this many bad stretches 4-6 fresh 무서움/어두움 reports (danger zones)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    count = args.count or rng.randint(20, 30)
    graph = load_graph()
    profiles = make_profiles(rng)
    reviews, stretches = generate(graph, rng, count, profiles, args.danger)
    if not args.no_profiles:
        existing_profiles = json.loads(args.profiles.read_text(encoding="utf-8")) if args.profiles.exists() else {}
        args.profiles.parent.mkdir(parents=True, exist_ok=True)
        args.profiles.write_text(json.dumps({**existing_profiles, **profiles}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"Profiles for {', '.join(USERS)} written to {args.profiles} ({len(existing_profiles)} other nickname(s) kept)")

    existing: dict[str, object] = {}
    if args.out.exists():
        existing = json.loads(args.out.read_text(encoding="utf-8"))
        if not args.append:
            backup = args.out.with_name(f"{args.out.stem}.{time.strftime('%Y%m%d-%H%M%S')}.bak")
            args.out.rename(backup)
            print(f"Existing {args.out} moved to {backup}")
    payload = {**existing, "reviews": (list(existing.get("reviews", [])) if args.append else []) + reviews} if args.append else {"reviews": reviews, "negatives": []}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    now = time.time()
    ages = [(now - time.mktime(time.strptime(str(r["ts"]), "%Y-%m-%dT%H:%M:%S"))) / 86400 for r in reviews]
    buckets = [sum(1 for a in ages if lo <= a < hi) for lo, hi in ((0, 7), (7, 30), (30, 60), (60, 91))]
    kinds = [k for k, *_ in stretches]
    print(f"Wrote {len(reviews)} reviews on {len(stretches)} stretches ({kinds.count('good')} good / {kinds.count('bad')} bad) to {args.out}")
    print(f"  age: {buckets[0]} within 7 days, {buckets[1]} in 7-30 days (decay > 0.5), {buckets[2]} in 30-60 days (0.5-0.25), {buckets[3]} in 60-90 days (< 0.25)")
    for kind, name, highway, edges in stretches:
        length = sum(graph[u][v][k]["length"] for u, v, k in edges)
        n = sum(1 for r in reviews if r["edge_id"] == ",".join(f"{u}-{v}-{k}" for u, v, k in edges))
        print(f"  {'GOOD' if kind == 'good' else 'BAD '} {name or '(이름 없음)'} [{highway}] {len(edges)} edges, {length:,.0f} m, {n} reviews")
    print("Restart server.py to replay the new reviews.")


if __name__ == "__main__":
    main()
