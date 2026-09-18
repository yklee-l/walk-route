from __future__ import annotations

import argparse
import base64
import hashlib
import re
import json
import os
import sys
import threading
import time
from collections import Counter
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import anthropic
import networkx as nx

from candidate_routes import BETAS, PROFILE, candidate_routes, candidates_geojson
from routing import Weights, load_graph, nearest_node, without_steps
from waypoints import DEFAULT_REVISIT_PENALTY, MAX_WAYPOINTS, WaypointError


WEB_DIR = Path(__file__).parent / "web"
REVIEWS_PATH = Path("osmnx_outputs/reviews.json")
PROFILES_PATH = Path("osmnx_outputs/profiles.json")  # {nickname: {mode, purpose, subs, avoid, ai, weights, avoid_steps, updated}}
PHOTOS_DIR = Path("osmnx_outputs/photos")  # photos/{edge_dir}/{ts}_{n}.jpg; served at /photos/...
PHOTO_MAX, PHOTO_MAX_BYTES = 3, 5 * 1024 * 1024
PHOTO_TYPES = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
ENV_PATH = Path(__file__).parent / ".env"
PORT = 8000

# ---- natural-language -> weights (POST /api/interpret) ----
INTERPRET_MODEL = "claude-opus-5"
INTERPRET_SYSTEM = """당신은 서울 광화문 일대 산책 경로 추천 서비스의 취향 해석기입니다.
사용자가 원하는 길을 한국어 자연어로 말하면, 경로 탐색 가중치 JSON으로 바꿉니다.

축 (각 0~4, 0 = 무시, 4 = 최우선):
- w_vehicle: 차량 통행 회피 (조용한, 한산한, 차 없는, 골목, 안전 → 높게)
- w_vibe: 분위기·감성 (카페, 데이트, 감성, 예쁜, 핫플, 사진, 맛집, 활기 → 높게)
- w_park: 공원·녹지·수변 근접 (공원, 자연, 강, 냇가, 물가 → 높게)
- w_tree: 가로수 (나무 그늘, 가로수길 → 높게)
- avoid_steps: 계단 제외 여부 (유모차, 휠체어, 무릎, 캐리어 → true)
- avoid_tags: 사용자 신고 기반으로 피할 태그. 다음 중에서만 고르세요: 공사중, 냄새, 더러움, 어두움, 좁고 답답함, 무서움, 차 많음, 사람 붐빔, 포장 상태
- summary: 해석 결과를 한 문장(30자 이내)으로, 예: "조용한 골목과 카페 위주"
- origin_name / destination_name: 문장에 출발지·목적지 지명이 있으면 그 지명만 그대로 (예: "광화문에서 경복궁까지" → "광화문", "경복궁").
  언급이 없으면 빈 문자열. "A에서 B까지/로/으로" 꼴이면 A가 출발지, B가 목적지이고, "B까지"만 있으면 목적지만 채웁니다.

"빠르게", "최단", "급해요"처럼 시간이 우선이면 모든 가중치를 0으로 두세요.
언급되지 않은 축은 0으로 둡니다. 과장하지 말고 문장에 드러난 강도만큼만 반영하세요."""
INTERPRET_SCHEMA = {
    "type": "object",
    "properties": {
        # Structured outputs reject minimum/maximum on numbers; the prompt states the 0..4 range and we clamp after.
        "w_vehicle": {"type": "number"},
        "w_vibe": {"type": "number"},
        "w_park": {"type": "number"},
        "w_tree": {"type": "number"},
        "avoid_steps": {"type": "boolean"},
        "avoid_tags": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
        "origin_name": {"type": "string"},
        "destination_name": {"type": "string"},
    },
    "required": ["w_vehicle", "w_vibe", "w_park", "w_tree", "avoid_steps", "avoid_tags", "summary", "origin_name", "destination_name"],
    "additionalProperties": False,
}


EXPLAIN_SYSTEM = """당신은 서울 광화문 일대 산책 경로 추천 서비스입니다. 사용자의 요청과 후보 경로들의 수치를 보고,
각 추천 경로를 왜 추천하는지 한국어로 설명합니다.

규칙:
- 전문적이되 친근한 톤, 존댓말, 이모지 금지. 경로당 1~2문장, 70자 이내.
- 사용자의 요청 문장과 연결해서 설명하세요 (예: "조용한 길을 원하셔서 …").
- 주어진 수치만 사용하고 지어내지 마세요. 최단 대비 추가 소요 시간(extra_min)은 꼭 언급하세요 (0이면 "소요 시간 차이는 거의 없습니다").
- 신고 구간(report_tags)이나 낮은 평가가 있으면 그것을 우회했다는 점을 우선 설명하세요.
- 경로 이름(label)은 그대로 쓰세요."""
EXPLAIN_SCHEMA = {
    "type": "object",
    "properties": {
        "explanations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"rank": {"type": "integer"}, "text": {"type": "string"}},
                "required": ["rank", "text"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["explanations"],
    "additionalProperties": False,
}


def explain_routes(text: str, summary: str, candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    """Ask Claude for a one-line rationale per recommended route (not the shortest)."""
    client = anthropic.Anthropic()
    prompt = json.dumps({"user_request": text, "interpreted_as": summary, "candidates": candidates}, ensure_ascii=False)
    response = client.messages.create(
        model=INTERPRET_MODEL,
        max_tokens=1024,
        system=EXPLAIN_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": EXPLAIN_SCHEMA}},
    )
    if response.stop_reason == "refusal":
        raise ValueError("설명을 생성할 수 없습니다.")
    data = json.loads(next(block.text for block in response.content if block.type == "text"))
    wanted = {int(c["rank"]) for c in candidates}
    return [{"rank": int(e["rank"]), "text": str(e["text"])[:200]} for e in data["explanations"] if int(e["rank"]) in wanted]


def ai_token(password: str) -> str:
    """Separate token for the Claude endpoints (AI_PASSWORD); sent back as X-AI-Token for the tab session only."""
    return hashlib.sha256(f"walkroute-ai:{password}".encode("utf-8")).hexdigest()


def load_env(path: Path = ENV_PATH) -> None:
    """Minimal .env loader (KEY=value lines); never overrides variables already set in the environment."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and value and key not in os.environ:
            os.environ[key] = value


def interpret_request(text: str) -> dict[str, object]:
    """Ask Claude to turn a free-text wish into routing weights (structured output, validated by the schema)."""
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY (loaded from .env at startup)
    response = client.messages.create(
        model=INTERPRET_MODEL,
        max_tokens=1024,
        system=INTERPRET_SYSTEM,
        messages=[{"role": "user", "content": text}],
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": INTERPRET_SCHEMA}},
    )
    if response.stop_reason == "refusal":
        raise ValueError("요청을 해석할 수 없습니다.")
    data = json.loads(next(block.text for block in response.content if block.type == "text"))
    clamp = lambda v: round(min(max(float(v), 0.0), 4.0) * 2) / 2  # noqa: E731  (0..4 in 0.5 steps, like the presets)
    return {
        "w_vehicle": clamp(data["w_vehicle"]), "w_vibe": clamp(data["w_vibe"]), "w_park": clamp(data["w_park"]), "w_tree": clamp(data["w_tree"]),
        "avoid_steps": bool(data["avoid_steps"]),
        "avoid_tags": [t for t in data["avoid_tags"] if t in NEGATIVE_REASONS],
        "summary": str(data["summary"])[:60],
        "origin_name": str(data.get("origin_name", "")).strip()[:60] or None,
        "destination_name": str(data.get("destination_name", "")).strip()[:60] or None,
        "model": response.model,
        "usage": {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens},
    }

Edge = tuple[int, int, int]

# A segment review is a 0..10 score plus tags. The score moves the edge's vibe_score away from
# the OSM baseline by weighted average: vibe = (PRIOR_WEIGHT * osm + sum(score / 10)) / (PRIOR_WEIGHT + n).
# The prior counts as PRIOR_WEIGHT reviews, so one 10/10 on a 0.2 street lands at (0.4 + 1.0) / 3 = 0.47.
PRIOR_WEIGHT = 2.0
# Per-user weighting of reviews (applied per request, see ReviewStore.personalize):
#   own reviews count OWN_WEIGHT times; a stranger's review counts STRANGER_MIN alone and grows to 1.0
#   once STRANGER_FULL_AT others have reviewed the same edge. 0/10 scores are dropped as outliers when
#   at least OUTLIER_MIN_OTHERS non-extreme reviews exist and the extreme sits >= OUTLIER_GAP from their median.
OWN_WEIGHT = 3.0
STRANGER_MIN, STRANGER_FULL_AT = 0.5, 5
OUTLIER_MIN_OTHERS, OUTLIER_GAP = 3, 4.0
GOOD_TAGS = ["여러명 걷기 좋아요", "자전거 도로 있어요", "인생샷 좋아요", "분위기 좋아요", "한산해요", "맛집 많아요", "포장 좋아요"]
# The edge's average review score also sets an avoid penalty (avoid_score): first threshold the average
# reaches, top-down. It is recomputed from the average, so it relaxes again as better reviews arrive.
SCORE_AVOID = [(8, 0.0), (5, 0.3), (2, 0.8), (0, 1.5)]
# Bad tags, as effects on the reported edges (cumulative, so repeated reports stack up to the caps):
#   avoid   -> avoid_by_tag[tag]; counted in the base cost multiplier 1 + avoid_penalty only when the user asked
#              to avoid that tag (avoid_tags request param; absent = every tag). The pure shortest ignores it.
#   vehicle -> vehicle_exposure_score (0..4, capped at 4): felt through the w_vehicle axis only
#   vibe    -> vibe_nudge, added to vibe_score after the review average (negative = worse): w_vibe axis only
NEGATIVE_REASONS: dict[str, dict[str, float]] = {
    # strong
    "공사중": {"avoid": 1.5},
    "냄새": {"avoid": 1.5},
    "더러움": {"avoid": 1.5},
    "무서움": {"avoid": 1.5},
    # mild: nudge one axis
    "좁고 답답함": {"avoid": 0.3},
    "차 많음": {"vehicle": 0.5},
    "어두움": {"vibe": -0.15, "avoid": 0.8},
    # minimal
    "사람 붐빔": {"avoid": 0.1},
    "포장 상태": {"avoid": 0.1},
}
BAD_TAGS = list(NEGATIVE_REASONS)
AVOID_CAP = 5.0
VEHICLE_CAP = 4.0


def score_avoid(average: float | None) -> float:
    if average is None:
        return 0.0
    return next(avoid for threshold, avoid in SCORE_AVOID if average >= threshold)


def parse_list(params: dict[str, object], key: str) -> list[str] | None:
    """Comma-separated or JSON-list param -> list; None when the param is absent."""
    if key not in params:
        return None
    value = params[key]
    items = value if isinstance(value, list) else str(value).split(",")
    return [str(t).strip() for t in items if str(t).strip()]


def photo_dir_name(edge_id: str) -> str:
    """Folder name for an edge_id ('u-v-k' or 'u-v-k,u-v-k,...'): commas -> '_', long lists shortened with a hash."""
    name = re.sub(r"[^0-9A-Za-z_-]", "_", edge_id.replace(",", "_"))
    return name if len(name) <= 80 else f"{name[:60]}_{hashlib.sha1(edge_id.encode()).hexdigest()[:10]}"


def save_photos(edge_id: str, photos: list[dict[str, object]]) -> list[str]:
    """Decode up to PHOTO_MAX base64 images (<= PHOTO_MAX_BYTES each) into PHOTOS_DIR/{edge_dir}/; returns URL paths."""
    if len(photos) > PHOTO_MAX:
        raise ValueError(f"사진은 최대 {PHOTO_MAX}장까지 올릴 수 있습니다.")
    folder = PHOTOS_DIR / photo_dir_name(edge_id)
    saved: list[str] = []
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for i, photo in enumerate(photos):
        mime = str(photo.get("type", "")).lower()
        if mime not in PHOTO_TYPES:
            raise ValueError("사진은 JPEG, PNG, WebP만 올릴 수 있습니다.")
        data = str(photo.get("data", ""))
        data = data.split(",", 1)[1] if data.startswith("data:") else data  # accept data URLs or bare base64
        raw = base64.b64decode(data, validate=False)
        if len(raw) > PHOTO_MAX_BYTES:
            raise ValueError("사진 한 장은 5MB 이하여야 합니다.")
        folder.mkdir(parents=True, exist_ok=True)
        name = f"{stamp}_{i + 1}_{hashlib.sha1(raw).hexdigest()[:8]}{PHOTO_TYPES[mime]}"
        (folder / name).write_bytes(raw)
        saved.append(f"/photos/{folder.name}/{name}")
    return saved


def parse_waypoints(params: dict[str, object]) -> list[tuple[float, float]]:
    """waypoints='lat,lon;lat,lon' (query) or [[lat, lon], ...] (JSON) -> list of (lat, lon); [] when absent."""
    value = params.get("waypoints")
    if value in (None, ""):
        return []
    items = value if isinstance(value, list) else [part for part in str(value).split(";") if part.strip()]
    points: list[tuple[float, float]] = []
    for item in items:
        pair = item if isinstance(item, (list, tuple)) else str(item).split(",")
        if len(pair) != 2:
            raise ValueError("waypoints must be 'lat,lon;lat,lon' pairs")
        points.append((float(pair[0]), float(pair[1])))
    if len(points) > MAX_WAYPOINTS:
        raise ValueError(f"waypoints: at most {MAX_WAYPOINTS}")
    return points


def parse_revisit_penalty(params: dict[str, object]) -> float:
    value = float(params.get("revisit_penalty", DEFAULT_REVISIT_PENALTY) or DEFAULT_REVISIT_PENALTY)
    if value < 1.0:
        raise ValueError("revisit_penalty must be >= 1")
    return value


def parse_user(params: dict[str, object]) -> str | None:
    """Nickname the client identifies as (user_id); free text, trimmed, up to 30 chars."""
    value = str(params.get("user_id", "") or "").strip()
    return value[:30] or None


def parse_tags(params: dict[str, object]) -> list[str]:
    """Preset tags the client chose (e.g. ['date', 'date/quiet', 'date/photo']); echoed back and stored, not yet interpreted."""
    return parse_list(params, "tags") or []


def parse_avoid_tags(params: dict[str, object]) -> set[str] | None:
    """Tags whose reports should be avoided; None (param absent) means all of them. Unknown tags are dropped."""
    tags = parse_list(params, "avoid_tags")
    return None if tags is None else {t for t in tags if t in NEGATIVE_REASONS}


def parse_edge_id(value: object) -> list[Edge]:
    """'u-v-k', 'u-v-k,u-v-k' or a list of those -> [(u, v, k), ...]."""
    ids = value if isinstance(value, list) else str(value).split(",")
    edges = []
    for edge_id in ids:
        parts = tuple(int(part) for part in str(edge_id).strip().split("-"))
        if len(parts) != 3:
            raise ValueError(edge_id)
        edges.append(parts)
    return edges


class ProfileStore:
    """Per-nickname preferences (what the page stores as its default), kept in profiles.json."""

    KEEP = ("mode", "purpose", "subs", "avoid", "ai", "weights", "avoid_steps")

    def __init__(self, path: Path = PROFILES_PATH) -> None:
        self.path = path
        self.profiles: dict[str, dict[str, object]] = {}
        if path.exists():
            try:
                self.profiles = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self.profiles = {}

    def get(self, nickname: str) -> dict[str, object] | None:
        return self.profiles.get(nickname)

    def put(self, nickname: str, data: dict[str, object]) -> dict[str, object]:
        profile = {k: data[k] for k in self.KEEP if k in data}
        profile["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.profiles[nickname] = profile
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.profiles, ensure_ascii=False, indent=1), encoding="utf-8")
        return profile


class ReviewStore:
    """reviews.json = {"reviews": [...], "negatives": [...]}, replayed onto the graph at startup."""

    def __init__(self, graph: nx.MultiDiGraph, path: Path = REVIEWS_PATH) -> None:
        self.graph, self.path = graph, path
        self.reviews: list[dict[str, object]] = []
        self.negatives: list[dict[str, object]] = []
        self.edge_reviews: dict[Edge, list[tuple[float, str | None]]] = {}  # per edge: (score, user_id) in arrival order
        self.user_id: str | None = None  # whose point of view the graph is currently recomputed for

    # ---- persistence ----
    def load(self) -> int:
        if not self.path.exists():
            return 0
        data = json.loads(self.path.read_text(encoding="utf-8"))
        for review in data.get("reviews", []):
            self.apply_review(float(review["score"]), list(review.get("good_tags", [])), list(review.get("bad_tags", [])), parse_edge_id(review["edge_id"]), review.get("user_id"))
            self.reviews.append(review)
        for negative in data.get("negatives", []):
            self.apply_negative(NEGATIVE_REASONS.get(str(negative["reason"]), {"avoid": 1.0}), parse_edge_id(negative["edge_id"]), str(negative["reason"]))
            self.negatives.append(negative)
        return len(self.reviews) + len(self.negatives)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"reviews": self.reviews, "negatives": self.negatives}, ensure_ascii=False, indent=1), encoding="utf-8")

    def reset(self) -> Path | None:
        if not self.path.exists():
            return None
        archived = self.path.with_name(f"{self.path.stem}.{time.strftime('%Y%m%d-%H%M%S')}.bak")
        self.path.rename(archived)
        return archived

    # ---- reviews ----
    def add_review(self, edges: list[Edge], score: float, good_tags: list[str], bad_tags: list[str], meta: dict[str, object], user_id: str | None = None) -> list[Edge]:
        RouteHandler.route_cache.clear()  # edge scores changed: every cached route may be stale
        updated = self.apply_review(score, good_tags, bad_tags, edges, user_id)
        self.reviews.append({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "user_id": user_id, "edge_id": ",".join(f"{u}-{v}-{k}" for u, v, k in updated),
            "score": score, "good_tags": good_tags, "bad_tags": bad_tags, **meta,
        })
        self.save()
        return updated

    def apply_review(self, score: float, good_tags: list[str], bad_tags: list[str], edges: list[Edge], user_id: str | None = None) -> list[Edge]:
        touched = self.both_directions(edges)
        for edge in touched:
            data = self.graph[edge[0]][edge[1]][edge[2]]
            self.edge_reviews.setdefault(edge, []).append((score, user_id))
            data["review_count"] = int(data.get("review_count", 0)) + 1  # raw count/sum, for display
            data["review_score_sum"] = float(data.get("review_score_sum", 0.0)) + score
            for tag in bad_tags:
                self.apply_effects(NEGATIVE_REASONS[tag], data, tag)
            self.recompute(edge, self.user_id)
        return sorted(e for e in touched if e in set(edges))

    def weighted_reviews(self, edge: Edge, user_id: str | None = None) -> list[tuple[float, float]]:
        """(score, weight) per review on this edge from `user_id`'s point of view, outliers removed."""
        reviews = self.edge_reviews.get(edge, [])
        if not reviews:
            return []
        others = [sc for sc, uid in reviews if uid is None or uid != user_id]
        middle = sorted(sc for sc in others if 0 < sc < 10)
        keep: list[tuple[float, float]] = []
        for score, uid in reviews:
            own = user_id is not None and uid == user_id
            if not own and score in (0.0, 10.0) and len(middle) >= OUTLIER_MIN_OTHERS:
                median = middle[len(middle) // 2]
                if abs(score - median) >= OUTLIER_GAP:
                    continue  # a sudden 0 or 10 against a settled consensus
            n_others = len(others)
            weight = OWN_WEIGHT if own else STRANGER_MIN + (1.0 - STRANGER_MIN) * min(1.0, max(0, n_others - 1) / (STRANGER_FULL_AT - 1))
            keep.append((score, weight))
        return keep

    def recompute(self, edge: Edge, user_id: str | None = None) -> None:
        """Derive vibe_score and avoid_score from the edge's components, weighting reviews for `user_id`."""
        data = self.graph[edge[0]][edge[1]][edge[2]]
        weighted = self.weighted_reviews(edge, user_id)
        total = sum(w for _, w in weighted)
        average_vibe = (PRIOR_WEIGHT * data["vibe_score_osm"] + sum(sc / 10 * w for sc, w in weighted)) / (PRIOR_WEIGHT + total)
        data["vibe_score"] = min(max(average_vibe + float(data.get("vibe_nudge", 0.0)), 0.0), 1.0)
        data["avoid_score"] = score_avoid(sum(sc * w for sc, w in weighted) / total if total else None)

    def personalize(self, user_id: str | None) -> None:
        """Recompute every reviewed edge for this user (own reviews x3 etc.); cheap, only reviewed edges are touched."""
        if user_id == self.user_id:
            return
        self.user_id = user_id
        for edge in self.edge_reviews:
            self.recompute(edge, user_id)

    def apply_avoid_tags(self, avoid_tags: set[str] | None) -> None:
        """Set every edge's avoid_penalty for this request: score part + the chosen tags' parts (None = all tags)."""
        for _, _, _, data in self.graph.edges(keys=True, data=True):
            by_tag = data.get("avoid_by_tag") or {}
            tagged = sum(v for t, v in by_tag.items() if avoid_tags is None or t in avoid_tags)
            data["avoid_penalty"] = min(float(data.get("avoid_score", 0.0)) + tagged, AVOID_CAP)

    def stats(self, edges: list[Edge], user_id: str | None = None) -> dict[str, object]:
        """Reviews touching any of the edges (a review counts once even if it covers several of them)."""
        existing = [e for e in edges if self.graph.has_edge(*e)]
        wanted = set(existing) | {(v, u, k) for u, v, k in existing}
        hits = [r for r in self.reviews if wanted & set(parse_edge_id(r["edge_id"]))]
        reports = [n for n in self.negatives if wanted & set(parse_edge_id(n["edge_id"]))]
        good: Counter[str] = Counter()
        bad: Counter[str] = Counter()
        for review in hits:
            good.update(review.get("good_tags", []))
            bad.update(review.get("bad_tags", []))
        bad.update(str(n["reason"]) for n in reports)
        return {
            "edge_id": ",".join(f"{u}-{v}-{k}" for u, v, k in existing), "edge_count": len(existing),
            "review_count": len(hits), "score_avg": round(sum(float(r["score"]) for r in hits) / len(hits), 2) if hits else None,
            "own_count": sum(1 for r in hits if user_id and r.get("user_id") == user_id),
            "negative_count": len(reports),
            "good_tags": dict(good.most_common()), "bad_tags": dict(bad.most_common()),
            "photos": [{"url": url, "user_id": r.get("user_id"), "score": r.get("score"), "ts": r.get("ts")} for r in hits for url in r.get("photos", [])],
            "vibe_score_avg": round(sum(self.graph[u][v][k]["vibe_score"] for u, v, k in existing) / len(existing), 3) if existing else None,
        }

    # ---- tag-only negative feedback (POST /api/feedback/negative) ----
    def add_negative(self, reason: str, edges: list[Edge], meta: dict[str, object]) -> list[Edge]:
        RouteHandler.route_cache.clear()
        updated = self.apply_negative(NEGATIVE_REASONS[reason], edges, reason)
        self.negatives.append({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "edge_id": ",".join(f"{u}-{v}-{k}" for u, v, k in updated), "reason": reason, **meta})
        self.save()
        return updated

    def apply_negative(self, effects: dict[str, float], edges: list[Edge], tag: str = "") -> list[Edge]:
        touched = self.both_directions(edges)
        for edge in touched:
            data = self.graph[edge[0]][edge[1]][edge[2]]
            data["negative_count"] = int(data.get("negative_count", 0)) + 1
            self.apply_effects(effects, data, tag)
            self.recompute(edge, self.user_id)
        return sorted(e for e in touched if e in set(edges))

    @staticmethod
    def apply_effects(effects: dict[str, float], data: dict[str, object], tag: str = "") -> None:
        if "avoid" in effects:
            by_tag = data.setdefault("avoid_by_tag", {})
            by_tag[tag] = min(by_tag.get(tag, 0.0) + effects["avoid"], AVOID_CAP)
        if "vehicle" in effects:
            data["vehicle_bump"] = float(data.get("vehicle_bump", 0.0)) + effects["vehicle"]
            data["vehicle_exposure_score"] = min(data["vehicle_exposure_osm"] + data["vehicle_bump"], VEHICLE_CAP)
        if "vibe" in effects:
            data["vibe_nudge"] = max(float(data.get("vibe_nudge", 0.0)) + effects["vibe"], -1.0)

    def both_directions(self, edges: list[Edge]) -> set[Edge]:
        touched: set[Edge] = set()
        for u, v, k in edges:
            for edge in ((u, v, k), (v, u, k)):
                if self.graph.has_edge(*edge):
                    touched.add(edge)
        return touched


class RouteHandler(SimpleHTTPRequestHandler):
    """Static files from web/, GET /api/route, GET /api/reviews/{edge_id}, POST /api/review, POST /api/feedback/negative."""

    graph: nx.MultiDiGraph
    graph_no_steps: nx.MultiDiGraph  # same edge data as graph, stair edges removed
    reviews: ReviewStore
    profiles: ProfileStore
    api_enabled: bool = True  # --no-api turns the Claude endpoints off (demo switch)
    ai_token_value: str | None = None  # None when AI_PASSWORD is empty: AI endpoints need no extra unlock
    route_cache: dict[tuple, dict[str, object]] = {}  # request key -> GeoJSON; dropped whenever a review/report changes the graph
    ROUTE_CACHE_MAX = 256
    lock = threading.Lock()  # shortest_route rewrites edge costs in place, so one search at a time

    def do_GET(self) -> None:
        url = urlparse(self.path)
        query = {k: v[0] for k, v in parse_qs(url.query, keep_blank_values=True).items()}
        if url.path == "/api/status":
            return self.send_json({"api_enabled": self.api_enabled and bool(os.environ.get("ANTHROPIC_API_KEY")), "api_flag": self.api_enabled, "has_key": bool(os.environ.get("ANTHROPIC_API_KEY")), "ai_locked": self.ai_token_value is not None})
        if url.path == "/api/ai-auth":
            if self.ai_token_value is None:
                return self.send_json({"ok": True, "token": ""})
            if query.get("token") == self.ai_token_value or ("password" in query and ai_token(query["password"]) == self.ai_token_value):
                return self.send_json({"ok": True, "token": self.ai_token_value})
            return self.send_json({"ok": False, "error": "AI 비밀번호가 맞지 않습니다."}, HTTPStatus.UNAUTHORIZED)
        if url.path == "/api/route":
            try:
                od = self.parse_params(query)
                waypoints, revisit_penalty = parse_waypoints(query), parse_revisit_penalty(query)
            except ValueError as error:
                return self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            with self.lock:
                try:
                    self.send_json(self.route(*od, tags=parse_tags(query), avoid_tags=parse_avoid_tags(query), user_id=parse_user(query), waypoints=waypoints, revisit_penalty=revisit_penalty))
                except nx.NetworkXNoPath:
                    self.send_json({"error": "no walkable path between these points"}, HTTPStatus.UNPROCESSABLE_ENTITY)
                except WaypointError as error:
                    self.send_json({"error": str(error), "waypoints": waypoints}, HTTPStatus.BAD_REQUEST)
        elif url.path.startswith("/api/reviews/"):
            try:
                edges = parse_edge_id(unquote(url.path[len("/api/reviews/"):]))
            except ValueError:
                return self.send_json({"error": "edge_id must be u-v-k (comma-separated for several)"}, HTTPStatus.BAD_REQUEST)
            with self.lock:
                self.send_json(self.reviews.stats(edges, parse_user(query)))
        elif url.path.startswith("/api/profile/"):
            nickname = unquote(url.path[len("/api/profile/"):]).strip()[:30]
            if not nickname:
                return self.send_json({"error": "nickname is required"}, HTTPStatus.BAD_REQUEST)
            with self.lock:
                profile = self.profiles.get(nickname)
            if profile is None:
                return self.send_json({"nickname": nickname, "profile": None}, HTTPStatus.NOT_FOUND)
            self.send_json({"nickname": nickname, "profile": profile})
        elif url.path == "/api/user":
            user_id = parse_user(query)
            with self.lock:
                mine = [r for r in self.reviews.reviews if user_id and r.get("user_id") == user_id]
                reports = [n for n in self.reviews.negatives if user_id and n.get("user_id") == user_id]
            return self.send_json({"user_id": user_id, "review_count": len(mine), "negative_count": len(reports), "photo_count": sum(len(r.get("photos", [])) for r in mine), "last_review_ts": mine[-1]["ts"] if mine else None})
        elif url.path.startswith("/api/photos/"):
            # Photo index for an edge/segment (for the gallery now, for AI analysis later): same matching as /api/reviews.
            try:
                edges = parse_edge_id(unquote(url.path[len("/api/photos/"):]))
            except ValueError:
                return self.send_json({"error": "edge_id must be u-v-k (comma-separated for several)"}, HTTPStatus.BAD_REQUEST)
            with self.lock:
                stats = self.reviews.stats(edges, parse_user(query))
            self.send_json({"edge_id": stats["edge_id"], "photos": stats["photos"], "count": len(stats["photos"])})
        elif url.path.startswith("/photos/"):
            self.send_photo(unquote(url.path[len("/photos/"):]))
        else:
            super().do_GET()

    def send_photo(self, relative: str) -> None:
        """Serve one stored photo; the path must stay inside PHOTOS_DIR."""
        root = PHOTOS_DIR.resolve()
        target = (root / relative).resolve()
        if root not in target.parents or not target.is_file():
            return self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        mime = {".jpg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}.get(target.suffix.lower(), "application/octet-stream")
        body = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=86400")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        if route == "/api/review":
            return self.review()
        if route == "/api/feedback/negative":
            return self.negative_feedback()
        if route.startswith("/api/profile/"):
            nickname = unquote(route[len("/api/profile/"):]).strip()[:30]
            try:
                body = self.read_body()
            except (TypeError, ValueError, json.JSONDecodeError):
                body = None
            if not nickname or not isinstance(body, dict) or body.get("mode") not in ("buttons", "ai"):
                return self.send_json({"error": "expected /api/profile/{nickname} with {mode: buttons|ai, purpose?, subs?, avoid?, ai?, weights?, avoid_steps?}"}, HTTPStatus.BAD_REQUEST)
            with self.lock:
                profile = self.profiles.put(nickname, body)
            return self.send_json({"nickname": nickname, "profile": profile})
        if route == "/api/interpret":
            return self.interpret()
        if route == "/api/explain":
            return self.explain()
        self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def claude_unavailable(self) -> str | None:
        if not self.api_enabled:
            return "AI 기능이 꺼져 있습니다 (--no-api)."
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return "ANTHROPIC_API_KEY가 설정되지 않았습니다. .env 파일에 키를 넣고 서버를 재시작하세요."
        if self.ai_token_value is not None and self.headers.get("X-AI-Token") != self.ai_token_value:
            return "AI 비밀번호로 먼저 AI 기능을 켜 주세요."
        return None

    def call_claude(self, fn, *args):
        """Run a Claude helper and map SDK errors to (payload, status); the API's own message goes in `detail` and the log."""
        def failed(message: str, status: HTTPStatus, error: Exception) -> tuple[dict[str, object], HTTPStatus]:
            detail = getattr(error, "message", None) or str(error)
            request_id = getattr(error, "request_id", None) or getattr(getattr(error, "response", None), "headers", {}).get("request-id")
            print(f"[claude] {fn.__name__} failed: {type(error).__name__}: {detail}" + (f" (request_id={request_id})" if request_id else ""), flush=True)
            return {"error": message, "detail": detail, "request_id": request_id, "kind": type(error).__name__}, status
        try:
            return fn(*args), HTTPStatus.OK
        except anthropic.AuthenticationError as error:
            return failed("ANTHROPIC_API_KEY가 올바르지 않습니다.", HTTPStatus.UNAUTHORIZED, error)
        except anthropic.RateLimitError as error:
            return failed("요청이 많아 잠시 후 다시 시도해 주세요.", HTTPStatus.TOO_MANY_REQUESTS, error)
        except anthropic.APIStatusError as error:
            return failed(f"Claude API 오류 ({error.status_code})", HTTPStatus.BAD_GATEWAY, error)
        except anthropic.APIConnectionError as error:
            return failed("Claude API에 연결할 수 없습니다.", HTTPStatus.BAD_GATEWAY, error)
        except (ValueError, KeyError, TypeError, StopIteration) as error:
            return failed(str(error) or "응답을 읽을 수 없습니다.", HTTPStatus.BAD_GATEWAY, error)

    def explain(self) -> None:
        """POST /api/explain {text, summary, candidates:[{rank,label,...}]} -> {explanations:[{rank,text}]}."""
        try:
            body = self.read_body()
            text, summary = str(body.get("text", "")).strip()[:500], str(body.get("summary", ""))[:100]
            candidates = [c for c in body.get("candidates", []) if isinstance(c, dict) and "rank" in c][:4]
        except (TypeError, ValueError, json.JSONDecodeError):
            return self.send_json({"error": "expected {text, summary, candidates: [...]}"}, HTTPStatus.BAD_REQUEST)
        if not candidates:
            return self.send_json({"explanations": []})
        if (reason := self.claude_unavailable()):
            return self.send_json({"error": reason, "api_enabled": False}, HTTPStatus.SERVICE_UNAVAILABLE)
        result, status = self.call_claude(explain_routes, text, summary, candidates)
        self.send_json({"explanations": result} if status == HTTPStatus.OK else result, status)

    def interpret(self) -> None:
        """POST /api/interpret {text} -> weights JSON via Claude. No graph access, so it runs outside the routing lock."""
        try:
            text = str(self.read_body().get("text", "")).strip()
        except (TypeError, ValueError, json.JSONDecodeError):
            text = ""
        if not text:
            return self.send_json({"error": "text is required"}, HTTPStatus.BAD_REQUEST)
        if (reason := self.claude_unavailable()):
            return self.send_json({"error": reason, "api_enabled": False}, HTTPStatus.SERVICE_UNAVAILABLE)
        result, status = self.call_claude(interpret_request, text[:500])
        self.send_json({"text": text, **result} if status == HTTPStatus.OK else result, status)

    def read_body(self) -> dict[str, object]:
        return json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")

    def review(self) -> None:
        try:
            body = self.read_body()
            edges = parse_edge_id(body["edge_id"])
            score = float(body["score"])
            good_tags = [str(t) for t in body.get("good_tags", [])]
            bad_tags = [str(t) for t in body.get("bad_tags", [])]
            if not 0 <= score <= 10 or any(t not in GOOD_TAGS for t in good_tags) or any(t not in BAD_TAGS for t in bad_tags):
                raise ValueError
            od = self.parse_params(body) if "olat" in body else None
            photos_in = [ph for ph in body.get("photos", []) if isinstance(ph, dict)]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return self.send_json({"error": "expected {edge_id: 'u-v-k[,u-v-k]' | [..], score: 0..10, good_tags: [..], bad_tags: [..], photos?: [{type, data(base64)}]} (+ optional olat, olon, dlat, dlon, w_* to get routes back)", "good_tags": GOOD_TAGS, "bad_tags": BAD_TAGS}, HTTPStatus.BAD_REQUEST)
        if not edges:
            return self.send_json({"error": "edge_id is empty"}, HTTPStatus.BAD_REQUEST)
        user_id = parse_user(body)
        try:
            photo_urls = save_photos(",".join(f"{u}-{v}-{k}" for u, v, k in edges), photos_in) if photos_in else []
        except (ValueError, OSError) as error:
            return self.send_json({"error": str(error) or "사진을 저장할 수 없습니다."}, HTTPStatus.BAD_REQUEST)
        with self.lock:
            self.reviews.personalize(user_id)
            before = self.reviews.stats(edges, user_id)
            updated = self.reviews.add_review(edges, score, good_tags, bad_tags, {"origin": od[0] if od else None, "destination": od[1] if od else None, "preset_tags": parse_tags(body), "photos": photo_urls}, user_id)
            after = self.reviews.stats(updated, user_id)
            routes = self.routes_or_none(od, parse_tags(body), parse_avoid_tags(body), user_id)
        self.send_json({
            "applied": {
                "score": score, "good_tags": good_tags, "bad_tags": bad_tags, "edges_updated": len(updated),
                "review_count": after["review_count"], "score_avg": after["score_avg"],
                "vibe_before": before["vibe_score_avg"], "vibe_after": after["vibe_score_avg"],
                "avoid_penalty_max": round(max((float(self.graph[u][v][k].get("avoid_score", 0.0)) + sum((self.graph[u][v][k].get("avoid_by_tag") or {}).values()) for u, v, k in updated), default=0.0), 2),
                "reviews_total": len(self.reviews.reviews),
                "length_m": round(sum(self.graph[u][v][k]["length"] for u, v, k in updated), 1),
                "photos": photo_urls,
            },
            "stats": after,
            "routes": routes,
        })

    def negative_feedback(self) -> None:
        try:
            body = self.read_body()
            reason = str(body["reason"])
            if reason not in NEGATIVE_REASONS:
                raise ValueError
            edges = parse_edge_id(body["edge_id"]) if "edge_id" in body else [tuple(int(x) for x in e) for e in body["edges"]]  # type: ignore[misc]
            if any(len(e) != 3 for e in edges):
                raise ValueError
            od = self.parse_params(body) if "olat" in body else None
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return self.send_json({"error": f"expected {{reason: {'|'.join(NEGATIVE_REASONS)}, edge_id: 'u-v-k[,..]' | [..]}} (+ optional olat, olon, dlat, dlon, w_*)"}, HTTPStatus.BAD_REQUEST)
        if not edges:
            return self.send_json({"error": "edge_id is empty"}, HTTPStatus.BAD_REQUEST)
        with self.lock:
            updated = self.reviews.add_negative(reason, edges, {"origin": od[0] if od else None, "destination": od[1] if od else None, "preset_tags": parse_tags(body), "user_id": parse_user(body)})
            routes = self.routes_or_none(od, parse_tags(body), parse_avoid_tags(body), parse_user(body))
        datas = [self.graph[u][v][k] for u, v, k in updated]
        self.send_json({
            "applied": {
                "reason": reason, "effects": NEGATIVE_REASONS[reason], "edges_updated": len(updated), "negatives_total": len(self.reviews.negatives),
                "length_m": round(sum(d["length"] for d in datas), 1),
                "avoid_penalty_max": round(max((float(d.get("avoid_penalty", 0.0)) for d in datas), default=0.0), 2),
                "vehicle_score_max": round(max((float(d["vehicle_exposure_score"]) for d in datas), default=0.0), 2),
                "vibe_score_avg": round(sum(d["vibe_score"] * d["length"] for d in datas) / (sum(d["length"] for d in datas) or 1.0), 3),
            },
            "routes": routes,
        })

    def routes_or_none(self, od: tuple[tuple[float, float], tuple[float, float], Weights, bool] | None, tags: list[str] | None = None, avoid_tags: set[str] | None = None, user_id: str | None = None) -> dict[str, object] | None:
        if od is None:
            return None
        try:
            return self.route(*od, tags=tags, avoid_tags=avoid_tags, user_id=user_id)
        except nx.NetworkXNoPath:
            return None

    def parse_params(self, params: dict[str, object]) -> tuple[tuple[float, float], tuple[float, float], Weights, bool]:
        """(origin, destination, weights, avoid_steps) from query or JSON params."""
        try:
            origin = (float(params["olat"]), float(params["olon"]))
            destination = (float(params["dlat"]), float(params["dlon"]))
            profile = Weights(**{name: float(params.get(name, getattr(PROFILE, name))) for name in ("w_vehicle", "w_tree", "w_park", "w_vibe")})
            avoid_steps = str(params.get("avoid_steps", "0")).lower() in ("1", "true", "yes")
        except (KeyError, TypeError, ValueError):
            raise ValueError("olat, olon, dlat, dlon (and optional w_vehicle, w_tree, w_park, w_vibe, avoid_steps) must be numbers") from None
        if min(profile.__dict__.values()) < 0:
            raise ValueError("weights must be >= 0")
        return origin, destination, profile, avoid_steps

    def route(self, origin: tuple[float, float], destination: tuple[float, float], profile: Weights, avoid_steps: bool = False, tags: list[str] | None = None, avoid_tags: set[str] | None = None, user_id: str | None = None, waypoints: list[tuple[float, float]] | None = None, revisit_penalty: float = DEFAULT_REVISIT_PENALTY) -> dict[str, object]:
        origin_node, destination_node = nearest_node(self.graph, origin), nearest_node(self.graph, destination)
        waypoints = waypoints or []
        # Cache key covers everything the result depends on; waypoints (rounded to ~1 cm) and the penalty included,
        # otherwise a request with different waypoints would be served another request's routes.
        key = (
            origin_node, destination_node, profile, avoid_steps, None if avoid_tags is None else frozenset(avoid_tags), user_id,
            tuple((round(lat, 7), round(lon, 7)) for lat, lon in waypoints), revisit_penalty if waypoints else None,
        )
        cached = self.route_cache.get(key)
        if cached is not None:
            return {**cached, "tags": tags or [], "cached": True}
        self.reviews.personalize(user_id)  # own reviews weigh more, outliers dropped - per requester
        self.reviews.apply_avoid_tags(avoid_tags)  # graph_no_steps shares the edge dicts, so this covers both
        graph, steps_fallback = self.graph, False
        if avoid_steps:
            graph = self.graph_no_steps
            try:
                shortest, candidates = candidate_routes(graph, origin_node, destination_node, BETAS, profile, waypoints=waypoints, revisit_penalty=revisit_penalty)
            except nx.NetworkXNoPath:
                graph, steps_fallback = self.graph, True  # only reachable via stairs: fall back and say so
        if graph is self.graph:
            shortest, candidates = candidate_routes(graph, origin_node, destination_node, BETAS, profile, waypoints=waypoints, revisit_penalty=revisit_penalty)
        collection = candidates_geojson(graph, shortest, candidates)
        if waypoints:
            collection["waypoints_input"] = [[lat, lon] for lat, lon in waypoints]
            collection["revisit_penalty"] = revisit_penalty
        collection["tags"] = tags or []
        collection["user_id"] = user_id
        collection["avoid_tags"] = sorted(avoid_tags) if avoid_tags is not None else list(NEGATIVE_REASONS)
        if len(self.route_cache) >= self.ROUTE_CACHE_MAX:
            self.route_cache.pop(next(iter(self.route_cache)))  # FIFO
        self.route_cache[key] = collection
        collection["avoid_steps"] = avoid_steps and not steps_fallback
        collection["steps_fallback"] = steps_fallback
        collection["origin"] = [self.graph.nodes[origin_node]["x"], self.graph.nodes[origin_node]["y"]]
        collection["destination"] = [self.graph.nodes[destination_node]["x"], self.graph.nodes[destination_node]["y"]]
        collection["reviews_total"] = len(self.reviews.reviews)
        collection["negatives_total"] = len(self.reviews.negatives)
        collection["good_tags"], collection["bad_tags"] = GOOD_TAGS, BAD_TAGS
        return collection

    def send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        if args and "/api/" in str(args[0]):
            super().log_message(format, *args)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Walk-route API + static frontend.")
    parser.add_argument("port", type=int, nargs="?", default=PORT)
    parser.add_argument("--reset", action="store_true", help="archive reviews.json so vibe_score and penalties start from the OSM baseline")
    parser.add_argument("--no-api", action="store_true", help="disable the Claude endpoints (/api/interpret, /api/explain) for demos")
    args = parser.parse_args()

    load_env()
    RouteHandler.api_enabled = not args.no_api
    ai_password = os.environ.get("AI_PASSWORD", "").strip()
    RouteHandler.ai_token_value = ai_token(ai_password) if ai_password else None
    print("Claude endpoints: " + ("disabled (--no-api)" if args.no_api else "ready" if os.environ.get("ANTHROPIC_API_KEY") else "disabled (ANTHROPIC_API_KEY not set in .env)"))
    print("AI password: " + ("required to unlock AI" if ai_password else "not set (AI unlock is free)"))
    print("Loading graph...", flush=True)
    RouteHandler.graph = load_graph()
    RouteHandler.graph_no_steps = without_steps(RouteHandler.graph)
    print(f"Stair edges excluded for avoid_steps: {RouteHandler.graph.number_of_edges() - RouteHandler.graph_no_steps.number_of_edges():,}")
    RouteHandler.reviews = ReviewStore(RouteHandler.graph)
    RouteHandler.profiles = ProfileStore()
    print(f"Profiles: {len(RouteHandler.profiles.profiles)} nickname(s) in {PROFILES_PATH}")
    if args.reset:
        archived = RouteHandler.reviews.reset()
        print("Reset: vibe_score back to OSM baseline" + (f" (reviews archived to {archived})" if archived else " (no reviews to archive)"))
    else:
        print(f"Replayed {RouteHandler.reviews.load()} review/feedback entries from {REVIEWS_PATH}")
    handler = partial(RouteHandler, directory=str(WEB_DIR))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"Serving on http://127.0.0.1:{args.port}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
