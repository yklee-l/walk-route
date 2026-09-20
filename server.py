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
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlencode, urlparse

import anthropic
import networkx as nx

from candidate_routes import BETAS, PROFILE, candidate_routes, candidates_geojson
from routing import DANGER_THRESHOLD, Weights, edge_name, is_danger, load_graph, nearest_node, number, route_coordinates, without_steps
from score_gwanghwamun_edges import highway_values
from waypoints import DEFAULT_REVISIT_PENALTY, MAX_WAYPOINTS, WaypointError, haversine_m


WEB_DIR = Path(__file__).parent / "web"
# Everything the server writes lives under DATA_DIR. On a host with an ephemeral filesystem (Railway, Render, ...)
# point it at a mounted volume, otherwise reviews and photos are lost on the next deploy.
DATA_DIR = Path(os.environ.get("DATA_DIR") or "osmnx_outputs")
REVIEWS_PATH = DATA_DIR / "reviews.json"
PROFILES_PATH = DATA_DIR / "profiles.json"  # {nickname: {mode, purpose, subs, avoid, ai, weights, avoid_steps, updated}}
VISITS_PATH = DATA_DIR / "visits.json"  # {date: {page, routes, users: [...]}} for the admin stats
PHOTOS_DIR = DATA_DIR / "photos"  # photos/{edge_dir}/{ts}_{n}.jpg; served at /photos/...
PHOTO_MAX, PHOTO_MAX_BYTES = 3, 5 * 1024 * 1024
PHOTO_TYPES = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
ENV_PATH = Path(__file__).parent / ".env"
PORT = int(os.environ.get("PORT") or 8000)  # PaaS hosts hand the port over in $PORT
HOST = os.environ.get("HOST") or "127.0.0.1"  # 0.0.0.0 in a container, loopback locally

# ---- Kakao Local (GET /api/poi): places along a route ----
KAKAO_URL = "https://dapi.kakao.com/v2/local/search/category.json"
KAKAO_CATEGORIES = {"cafe": "CE7", "restaurant": "FD6", "tourist": "AT4"}  # our name -> Kakao category_group_code
POI_ANCHORS = 6  # the route polyline is split into this many equal stretches; each stretch's midpoint is searched
POI_RADIUS_M = 150
POI_PER_ANCHOR = 2  # closest places kept per anchor per category (1-2 in the brief)
POI_CACHE_TTL = 3600  # seconds a (route, categories) POI result is reused
POI_ROUTES_MAX = 256  # route geometries remembered for route_id lookups

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
- waypoints: 중간에 들르는 곳 지명 목록 (예: "경복궁에서 인사동 들르고 명동 가고 싶어" → origin "경복궁", waypoints ["인사동"], destination "명동").
  "들르고/거쳐서/찍고/경유해서/지나서" 같은 표현이 단서입니다. 언급 순서대로, 최대 6개, 없으면 빈 배열. 출발지·목적지는 넣지 않습니다.

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
        "waypoints": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["w_vehicle", "w_vibe", "w_park", "w_tree", "avoid_steps", "avoid_tags", "summary", "origin_name", "destination_name", "waypoints"],
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


SUMMARY_SYSTEM = """당신은 산책길 한 구간(거리)의 리뷰를 한 문장으로 요약합니다. 주어진 리뷰 문장과 태그 집계(⚑는 부정 태그)를 보고,
이 구간이 어떤 평을 받는지 한국어 한 문장(40자 이내, 존댓말, 이모지 없이)으로 씁니다. 예: "조용하고 가로수 많아 산책하기 좋다는 평이 많아요"
근거 없는 내용은 지어내지 마세요. 부정적인 평이 많으면 그것도 그대로 반영합니다."""
SUMMARY_SCHEMA = {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"], "additionalProperties": False}


ANALYZE_SYSTEM = """당신은 서울 광화문 일대 산책 경로 서비스의 리뷰 분석기입니다. 사용자가 길의 한 구간(거리)에 남긴
리뷰 문장을 읽고 JSON으로 정리합니다. 문장에 실제로 드러난 내용만 쓰고, 없는 내용은 0 / 빈 배열로 둡니다.

- axes: 문장이 다음 축에 대해 말하면 -1.0 ~ 1.0 강도 (0 = 언급 없음)
  vehicle: 차가 많고 시끄럽다(+) ↔ 차가 없고 조용하다(-)
  vibe: 분위기·감성이 좋다(+) ↔ 분위기가 나쁘다·삭막하다(-)
  park: 공원·녹지·물가가 가깝다(+) ↔ 녹지가 없다(-)
  tree: 가로수·나무 그늘이 많다(+) ↔ 그늘이 없다(-)
- good_tags: 문장이 뒷받침하는 것만, 다음 중에서: {good}
- bad_tags: 문장이 뒷받침하는 것만, 다음 중에서: {bad}
- keywords: 위 축이나 태그로는 표현되지 않는, 길의 특징이 될 만한 새 키워드 (명사 1~3개, 예: 고양이, 벚꽃, 야경, 벽화). 없으면 빈 배열.
  구체적인 장소 이름이나 일반적인 단어(길, 산책, 좋다)는 넣지 않습니다.
- summary: 15자 이내 한국어 요약"""
ANALYZE_SCHEMA = {
    "type": "object",
    "properties": {
        "axes": {
            "type": "object",
            "properties": {name: {"type": "number"} for name in ("vehicle", "vibe", "park", "tree")},
            "required": ["vehicle", "vibe", "park", "tree"], "additionalProperties": False,
        },
        "good_tags": {"type": "array", "items": {"type": "string"}},
        "bad_tags": {"type": "array", "items": {"type": "string"}},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["axes", "good_tags", "bad_tags", "keywords", "summary"],
    "additionalProperties": False,
}
# How strongly one review comment moves an edge (per unit of axis strength, cumulative, capped in apply_axes):
#   vehicle -> vehicle_bump (0..4 exposure score), vibe -> vibe_nudge (0..1 vibe), tree -> tree_nudge (trees / 100 m),
#   park -> park_nudge (metres; a positive park signal brings the park closer).
AXIS_STEPS = {"vehicle": 0.5, "vibe": 0.1, "tree": 3.0, "park": -60.0}
AXIS_CAPS = {"vehicle": 4.0, "vibe": 1.0, "tree": 15.0, "park": 300.0}


def analyze_comment(comment: str, score: float) -> dict[str, object]:
    """Ask Claude what a review comment says about the axes, which tags it implies, and any new keyword."""
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=INTERPRET_MODEL, max_tokens=512,
        system=ANALYZE_SYSTEM.format(good=", ".join(GOOD_TAGS), bad=", ".join(BAD_TAGS)),
        messages=[{"role": "user", "content": json.dumps({"comment": comment, "score_0_to_10": score}, ensure_ascii=False)}],
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": ANALYZE_SCHEMA}},
    )
    if response.stop_reason == "refusal":
        raise ValueError("리뷰를 분석할 수 없습니다.")
    return clean_analysis(json.loads(next(b.text for b in response.content if b.type == "text")))


def clean_analysis(data: dict[str, object]) -> dict[str, object]:
    """Clamp / whitelist an ANALYZE_SCHEMA result (from text or photos) into what the store applies."""
    clamp = lambda v: round(min(max(float(v or 0), -1.0), 1.0), 2)  # noqa: E731
    keywords: list[str] = []
    for word in data.get("keywords", []) or []:
        word = str(word).strip()[:20]
        if word and word not in keywords and word not in GOOD_TAGS and word not in BAD_TAGS:
            keywords.append(word)
    axes = data.get("axes") or {}
    return {
        "axes": {name: clamp(axes.get(name, 0)) for name in AXIS_STEPS},
        "good_tags": [t for t in dict.fromkeys(str(t) for t in data.get("good_tags", []) or []) if t in GOOD_TAGS],
        "bad_tags": [t for t in dict.fromkeys(str(t) for t in data.get("bad_tags", []) or []) if t in BAD_TAGS],
        "keywords": keywords[:3],
        "summary": str(data.get("summary", ""))[:40],
    }


PHOTO_SYSTEM = """당신은 서울 광화문 일대 산책 경로 서비스의 리뷰 사진 분석기입니다. 사용자가 길의 한 구간에서 찍은 사진(1~3장)을 보고
그 길의 특징을 JSON으로 정리합니다. 사진에 실제로 보이는 것만 쓰고, 보이지 않는 내용은 0 / 빈 배열로 둡니다.

가장 중요한 규칙 — 사진에서 본 것을 두 종류로 나누세요.
(A) 구조적 특성: 며칠·몇 달이 지나도 그 길에 남아 있는 것. 예: 포장 상태, 보도 폭, 가로수·그늘, 경사·계단, 벤치, 가로등,
    화단·녹지, 벽화, 상점가 여부, 자전거 도로. → axes / good_tags / bad_tags / keywords 에 넣습니다.
(B) 일시적 상황: 지금 이 순간에만 해당하는 것. 예: 지나가는 차량의 수, 사람이 많고 적음, 날씨·눈비, 노점, 공사 차량,
    행사, 계절 장식, 낮/밤 밝기. → transient_notes 에만 적고, 절대로 good_tags / bad_tags / keywords 에는 넣지 마세요.
헷갈리면 "내일 같은 자리에서 찍어도 똑같이 보일까?"를 기준으로 판단하세요. 아니라면 (B)입니다.

- axes: 구조적 특성만으로 판단한 정도를 -1.0 ~ 1.0 (0 = 판단 불가)
  vibe: 예쁘고 분위기 있다(+) ↔ 삭막하다(-)
  park: 공원·잔디·물가가 보인다(+) ↔ 녹지가 전혀 없다(-)
  tree: 가로수·나무 그늘이 많다(+) ↔ 나무가 없다(-)
  ※ 차량 통행량은 일시적이므로 여기서 판단하지 않습니다. 대신 '차도가 넓고 보도가 좁다' 같은 구조는 vibe 에 반영하세요.
- good_tags: 구조적 특성만, 다음 중에서: {good}
- bad_tags: 구조적 특성만, 다음 중에서: {bad}
  ※ '차 많음', '사람 붐빔'은 일시적 상황이므로 사진 분석에서는 절대 고르지 마세요.
- transient_notes: (B)에 해당하는 것을 짧은 말로 0~3개 (예: "차량 정체", "관광객 많음", "비 오는 날"). 없으면 빈 배열.
- transient_vibe: 그 일시적 상황이 사진 속 길의 인상에 준 영향 -1.0 ~ 1.0 (없으면 0)
- keywords: 위 축이나 태그로 표현되지 않는 구조적 특징 (명사 1~3개, 예: 벚꽃길, 벽화, 한옥담장, 계단). 없으면 빈 배열.
  사람·차량·날씨 같은 일시적인 것, 사람 얼굴·차량 번호·상호명 같은 개인정보, 특정 장소명은 넣지 않습니다.
- summary: 사진 속 길을 15자 이내로 (구조적 특징 위주)"""
PHOTO_SCHEMA = {
    "type": "object",
    "properties": {
        "axes": {"type": "object", "properties": {n: {"type": "number"} for n in ("vibe", "park", "tree")}, "required": ["vibe", "park", "tree"], "additionalProperties": False},
        "good_tags": {"type": "array", "items": {"type": "string"}},
        "bad_tags": {"type": "array", "items": {"type": "string"}},
        "transient_notes": {"type": "array", "items": {"type": "string"}},
        "transient_vibe": {"type": "number"},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["axes", "good_tags", "bad_tags", "transient_notes", "transient_vibe", "keywords", "summary"],
    "additionalProperties": False,
}
# ---- photo trust: a review photo should show the stretch it is filed under ----
PLACE_SYSTEM = """당신은 산책 경로 서비스의 리뷰 사진 검증기입니다. 앞의 사진들은 같은 길 구간에 이미 등록된 참고 사진이고,
마지막 사진은 방금 올라온 사진입니다. 마지막 사진이 참고 사진들과 같은 장소(같은 거리·구간)에서 찍힌 것인지 판단하세요.

- 같은 장소라도 계절·시간대·날씨·촬영 방향·사람과 차량은 다를 수 있습니다. 그런 차이만으로 다른 장소라고 하지 마세요.
- 건물·간판 배치, 도로 폭과 형태, 가로수 줄, 담장·공원 같은 고정 지형지물이 전혀 맞지 않을 때만 다른 장소입니다.
- 실내 사진, 음식 사진, 사람만 크게 찍힌 사진, 캡처 이미지처럼 길이 보이지 않으면 same_place=false, reason에 그 이유를 적으세요.
- 판단이 애매하면 same_place=true, confidence를 낮게 두세요. 의심은 확실할 때만 합니다.

- same_place: 같은 장소로 보이면 true
- confidence: 0.0 ~ 1.0 (판단의 확신 정도)
- reason: 30자 이내 한국어 근거"""
PLACE_SCHEMA = {
    "type": "object",
    "properties": {"same_place": {"type": "boolean"}, "confidence": {"type": "number"}, "reason": {"type": "string"}},
    "required": ["same_place", "confidence", "reason"], "additionalProperties": False,
}
PLACE_REFS_MAX = 2  # reference photos shown to the model
PLACE_SUSPICIOUS_CONFIDENCE = 0.6  # below this the "different place" verdict is treated as unsure, not suspicious
# Per-nickname photo trust (profiles.json): 1.0 = full weight. A suspicious photo costs PENALTY, each consistent one
# recovers RECOVER, so a wrong flag fades after a handful of good reviews. Below MIN_APPLY a photo's findings are
# recorded but never applied to the graph.
PHOTO_TRUST_DEFAULT, PHOTO_TRUST_PENALTY, PHOTO_TRUST_RECOVER, PHOTO_TRUST_MIN_APPLY = 1.0, 0.25, 0.05, 0.4


def image_block(mime: str, data: str) -> dict[str, object]:
    """Anthropic image content block from a base64 payload (with or without the data: prefix)."""
    if mime not in PHOTO_TYPES:
        raise ValueError("사진은 JPEG, PNG, WebP만 분석할 수 있습니다.")
    return {"type": "image", "source": {"type": "base64", "media_type": mime, "data": data.split(",", 1)[1] if data.startswith("data:") else data}}


def verify_photo_place(photos: list[dict[str, object]], references: list[tuple[str, str]]) -> dict[str, object]:
    """Does the new photo show the same place as the stretch's existing photos? references = [(mime, base64), ...]."""
    content: list[dict[str, object]] = [{"type": "text", "text": "참고 사진 (이 구간에 이미 등록된 사진):"}]
    content += [image_block(mime, data) for mime, data in references[:PLACE_REFS_MAX]]
    content.append({"type": "text", "text": "방금 올라온 사진:"})
    content.append(image_block(str(photos[0].get("type", "")).lower(), str(photos[0].get("data", ""))))
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=INTERPRET_MODEL, max_tokens=256, system=PLACE_SYSTEM,
        messages=[{"role": "user", "content": content}],
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": PLACE_SCHEMA}},
    )
    if response.stop_reason == "refusal":
        raise ValueError("사진을 확인할 수 없습니다.")
    data = json.loads(next(b.text for b in response.content if b.type == "text"))
    same, confidence = bool(data.get("same_place", True)), min(max(float(data.get("confidence", 0) or 0), 0.0), 1.0)
    return {
        "checked": True, "same_place": same, "confidence": round(confidence, 2), "reason": str(data.get("reason", ""))[:60],
        "suspicious": (not same) and confidence >= PLACE_SUSPICIOUS_CONFIDENCE,
    }


# Tags that exist in the vocabulary but describe a moment, not the street: a photo must never turn them into tags.
PHOTO_TRANSIENT_TAGS = {"차 많음", "사람 붐빔"}
# A photo is one moment, so what it shows about traffic / crowds only nudges the vibe axis, at this share.
PHOTO_TRANSIENT_VIBE_WEIGHT = 0.3


def clean_photo_analysis(data: dict[str, object]) -> dict[str, object]:
    """Split the vision result: structural findings become axes / tags / keywords, transient ones only touch vibe."""
    axes = data.get("axes") or {}
    clamp = lambda v: min(max(float(v or 0), -1.0), 1.0)  # noqa: E731
    transient_vibe = clamp(data.get("transient_vibe", 0))
    structural = {
        "axes": {"vehicle": 0, "vibe": axes.get("vibe", 0), "park": axes.get("park", 0), "tree": axes.get("tree", 0)},
        # Transient tags are dropped even if the model picked one anyway.
        "good_tags": [t for t in data.get("good_tags", []) or [] if t not in PHOTO_TRANSIENT_TAGS],
        "bad_tags": [t for t in data.get("bad_tags", []) or [] if t not in PHOTO_TRANSIENT_TAGS],
        "keywords": data.get("keywords", []),
        "summary": data.get("summary", ""),
    }
    result = clean_analysis(structural)
    # The moment's impression is a reference for the vibe axis only - never a tag, never the vehicle axis.
    result["axes"]["vibe"] = round(min(max(result["axes"]["vibe"] + PHOTO_TRANSIENT_VIBE_WEIGHT * transient_vibe, -1.0), 1.0), 2)
    result["transient_notes"] = [str(n).strip()[:20] for n in (data.get("transient_notes") or []) if str(n).strip()][:3]
    result["transient_vibe"] = round(transient_vibe, 2)
    place = data.get("place_check") if isinstance(data.get("place_check"), dict) else {}
    result["place_check"] = {
        "checked": bool(place.get("checked")), "same_place": place.get("same_place"),
        "confidence": round(min(max(float(place.get("confidence", 0) or 0), 0.0), 1.0), 2), "reason": str(place.get("reason", ""))[:60],
        "suspicious": bool(place.get("suspicious")),
    }
    result["suspicious"] = bool(place.get("suspicious"))
    # How the photo finding ended up accepted or not - "미반영"은 사용자가 거절한 경우만이 아니므로 경로를 남깁니다.
    #   user_yes / user_no       : 폼의 버튼을 눌러 결정
    #   confirm_yes / confirm_no : 결정하지 않고 제출해서 확인 창으로 결정
    #   no_findings              : 사진에서 반영할 것이 없어 물어보지 않음
    #   auto                     : 사용자 확인 없이 서버가 분석 (기록만)
    #   unknown                  : 경로를 기록하기 전에 저장된 예전 리뷰
    decision = str(data.get("decision", "") or "")
    result["decision"] = decision if decision in ("user_yes", "user_no", "confirm_yes", "confirm_no", "no_findings", "auto") else "unknown"
    return result


def analyze_photos(photos: list[dict[str, object]]) -> dict[str, object]:
    """Claude vision over the attached review photos: same axes / tags / keywords shape as the comment analysis."""
    content: list[dict[str, object]] = []
    for photo in photos[:PHOTO_MAX]:
        data = str(photo.get("data", ""))
        if len(data) > PHOTO_MAX_BYTES * 4 // 3 + 40:
            raise ValueError("사진 한 장은 5MB 이하여야 합니다.")
        content.append(image_block(str(photo.get("type", "")).lower(), data))
    if not content:
        raise ValueError("분석할 사진이 없습니다.")
    content.append({"type": "text", "text": "이 사진들은 같은 길 구간에서 찍은 것입니다. 규칙대로 JSON으로 정리하세요."})
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=INTERPRET_MODEL, max_tokens=512,
        system=PHOTO_SYSTEM.format(good=", ".join(t for t in GOOD_TAGS if t not in PHOTO_TRANSIENT_TAGS), bad=", ".join(t for t in BAD_TAGS if t not in PHOTO_TRANSIENT_TAGS)),
        messages=[{"role": "user", "content": content}],
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": PHOTO_SCHEMA}},
    )
    if response.stop_reason == "refusal":
        raise ValueError("사진을 분석할 수 없습니다.")
    result = clean_photo_analysis(json.loads(next(b.text for b in response.content if b.type == "text")))
    result["photo_count"] = len(content) - 1
    return result


SYNONYM_SYSTEM = """당신은 산책 경로 서비스의 '새 축 후보' 키워드를 정리합니다. 리뷰 글에서 새로 뽑힌 키워드가
기존 후보 키워드 중 어느 것과 사실상 같은 특징(같은 축으로 묶어도 되는 의미)을 뜻하는지 판단합니다.
예: 젊은층 ≈ 트렌디, 조용함 ≈ 한산함, 벚꽃 ≈ 꽃구경, 야경 ≈ 조명. 반대로 고양이 ≠ 강아지, 벽화 ≠ 조형물처럼 대상이 다르면 다른 키워드입니다.
새 키워드마다 same_as에 기존 키워드 하나를 넣거나, 겹치는 것이 없으면 null로 두세요. 애매하면 null입니다."""
SYNONYM_SCHEMA = {
    "type": "object",
    "properties": {"matches": {"type": "array", "items": {"type": "object", "properties": {"keyword": {"type": "string"}, "same_as": {"type": ["string", "null"]}}, "required": ["keyword", "same_as"], "additionalProperties": False}}},
    "required": ["matches"], "additionalProperties": False,
}


def match_synonyms(new_words: list[str], existing: list[str]) -> dict[str, str]:
    """new keyword -> existing candidate it duplicates (Claude's judgement); words with no match are left out."""
    if not new_words or not existing:
        return {}
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=INTERPRET_MODEL, max_tokens=512, system=SYNONYM_SYSTEM,
        messages=[{"role": "user", "content": json.dumps({"new_keywords": new_words, "existing_keywords": existing}, ensure_ascii=False)}],
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SYNONYM_SCHEMA}},
    )
    if response.stop_reason == "refusal":
        return {}
    data = json.loads(next(b.text for b in response.content if b.type == "text"))
    allowed = set(existing)
    return {str(m["keyword"]): str(m["same_as"]) for m in data.get("matches", []) if m.get("same_as") in allowed and str(m["keyword"]) in new_words}


def summarize_reviews(comments: list[str], tags: dict[str, int]) -> str:
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=INTERPRET_MODEL, max_tokens=256, system=SUMMARY_SYSTEM,
        messages=[{"role": "user", "content": json.dumps({"comments": comments[:30], "tags": tags}, ensure_ascii=False)}],
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SUMMARY_SCHEMA}},
    )
    if response.stop_reason == "refusal":
        raise ValueError("요약할 수 없습니다.")
    return str(json.loads(next(b.text for b in response.content if b.type == "text"))["summary"])[:80]


def ai_token(password: str) -> str:
    """Separate token for the Claude endpoints (AI_PASSWORD); sent back as X-AI-Token for the tab session only."""
    return hashlib.sha256(f"walkroute-ai:{password}".encode("utf-8")).hexdigest()


def admin_token(password: str) -> str:
    """Token for /admin and /api/admin/* (ADMIN_PASSWORD in .env); sent as X-Admin-Token."""
    return hashlib.sha256(f"walkroute-admin:{password}".encode("utf-8")).hexdigest()


def entry_id(entry: dict[str, object]) -> str:
    """Stable id for a stored review / report (older entries have none): hash of its timestamp, user and edges."""
    return hashlib.sha1(f"{entry.get('ts')}|{entry.get('user_id')}|{entry.get('edge_id')}|{entry.get('score')}|{entry.get('reason')}".encode("utf-8")).hexdigest()[:10]


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
        "waypoints": [str(w).strip()[:60] for w in data.get("waypoints", []) if str(w).strip()][:MAX_WAYPOINTS],
        "model": response.model,
        "usage": {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens},
    }

Edge = tuple[int, int, int]

# A segment review is a 0..10 score plus tags. The score moves the edge's vibe_score away from
# the OSM baseline by weighted average: vibe = (PRIOR_WEIGHT * osm + sum(score / 10)) / (PRIOR_WEIGHT + n).
# The prior counts as PRIOR_WEIGHT reviews, so one 10/10 on a 0.2 street lands at (0.4 + 1.0) / 3 = 0.47.
PRIOR_WEIGHT = 2.0
# Time decay: every stored entry's influence (review weight in the vibe/avoid averages, tag effects, AI axis
# nudges, danger reports) is multiplied by 0.5 ** (age_days / DECAY_HALF_LIFE_DAYS): a fresh review counts fully,
# a 30-day-old one half, a 60-day-old one a quarter. Applied when the store is replayed (startup, rebuild) and
# refreshed once a day (ReviewStore.refresh_decay). Displayed counts / averages on segment badges stay raw.
DECAY_HALF_LIFE_DAYS = 30.0
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
# 1) danger: this many 무서움/어두움 reports on an edge force the maximum avoid penalty for everyone
DANGER_TAGS = {"무서움", "어두움"}  # DANGER_THRESHOLD (3) and is_danger() come from routing.py
# 3) whole-route star ratings (1..5) enter the vibe average as 0..10 scores at a fraction of a segment review's weight
ROUTE_REVIEW_WEIGHT = 0.3
# 5) ranking (GET /api/ranking): segments (the stretch a reviewer rated) with at least RANK_MIN_REVIEWS segment
#    reviews in the period, scored by a review-count-weighted average: (n * avg + m * prior) / (n + m) with
#    m = RANK_PRIOR_WEIGHT and prior = the period's mean score, so a 10/10 with 3 reviews ranks below a 9/10 with 20.
RANK_MIN_REVIEWS = 3
RANK_PRIOR_WEIGHT = 3.0
RANK_SUMMARY_TTL = 6 * 3600  # seconds the Claude summary of a segment's comments is cached
RANK_SUMMARY_RETRY = 600  # seconds before a failed Claude summary is tried again (the tag fallback is served meanwhile)
# Tag -> phrase for the fallback summary ("... 평이 많아요"), used while the Claude summary is pending or unavailable.
TAG_PHRASES = {
    "분위기 좋아요": "분위기 좋다는", "한산해요": "한산하다는", "인생샷 좋아요": "사진 찍기 좋다는", "맛집 많아요": "맛집이 많다는",
    "여러명 걷기 좋아요": "여럿이 걷기 좋다는", "포장 좋아요": "길이 잘 포장돼 있다는", "자전거 도로 있어요": "자전거 도로가 있다는",
    "차 많음": "차가 많다는", "무서움": "무섭다는", "어두움": "어둡다는", "사람 붐빔": "사람이 붐빈다는", "좁고 답답함": "좁고 답답하다는",
    "공사중": "공사 중이라는", "냄새": "냄새가 난다는", "더러움": "지저분하다는", "포장 상태": "포장 상태가 나쁘다는",
}


def fallback_summary(good: Counter[str], bad: Counter[str], score_avg: float | None) -> str:
    """Tag-based one-liner: the two most common tags of the side the average score leans to."""
    side = good if (score_avg is None or score_avg >= 5.5) or not bad else bad
    if not side and (good or bad):
        side = good or bad
    phrases = [TAG_PHRASES.get(tag, f"'{tag}'") for tag, _ in side.most_common(2)]
    if len(phrases) == 2:
        return f"{phrases[0]} 평과 {phrases[1]} 평이 많아요"
    if phrases:
        return f"{phrases[0]} 평이 많아요"
    if score_avg is None:
        return "리뷰가 더 모이면 요약이 표시돼요"
    return "만족스럽다는 평이 많아요" if score_avg >= 8 else "아쉽다는 평이 많아요" if score_avg <= 3 else "평이 갈리는 구간이에요"
RANK_HIGHWAY_NAMES = {"footway": "보행로", "path": "산책로", "pedestrian": "보행자 거리", "steps": "계단", "residential": "주택가 골목", "service": "이면도로", "living_street": "생활도로", "primary": "대로", "secondary": "큰길", "tertiary": "큰길", "unclassified": "골목"}
PURPOSE_NAMES = {"date": "데이트", "solo": "혼자 산책", "family": "가족 나들이", "tour": "관광", "fast": "빠르게 이동", "ai": "AI 취향"}


def parse_ts(ts: object) -> float | None:
    try:
        return time.mktime(time.strptime(str(ts), "%Y-%m-%dT%H:%M:%S"))
    except (TypeError, ValueError):
        return None


def decay(ts: object, now: float | None = None) -> float:
    """0.5 ** (age_days / DECAY_HALF_LIFE_DAYS) for an entry timestamp; 1.0 when the timestamp is missing / in the future."""
    stamp = parse_ts(ts)
    if stamp is None:
        return 1.0
    age_days = max(0.0, ((now if now is not None else time.time()) - stamp) / 86400)
    return 0.5 ** (age_days / DECAY_HALF_LIFE_DAYS)


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


def mask_nickname(name: object) -> str | None:
    """Public form of a nickname: first character + one '*' per remaining character ("이영규" -> "이**", "Andy" -> "A***")."""
    if not name:
        return None
    text = str(name)
    return text[0] + "*" * max(len(text) - 1, 1)


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

    TRUST_FIELDS = ("photo_trust_score", "photo_checks", "photo_flags", "photo_trust_ts")

    def photo_trust(self, nickname: str | None) -> float:
        if not nickname:
            return PHOTO_TRUST_DEFAULT
        return float((self.profiles.get(nickname) or {}).get("photo_trust_score", PHOTO_TRUST_DEFAULT))

    def bump_photo_trust(self, nickname: str | None, suspicious: bool) -> float:
        """Move a nickname's photo trust after one checked photo; returns the new score (1.0 when there is no nickname)."""
        if not nickname:
            return PHOTO_TRUST_DEFAULT
        profile = self.profiles.setdefault(nickname, {"updated": time.strftime("%Y-%m-%dT%H:%M:%S")})
        score = float(profile.get("photo_trust_score", PHOTO_TRUST_DEFAULT))
        score = min(max(score + (-PHOTO_TRUST_PENALTY if suspicious else PHOTO_TRUST_RECOVER), 0.0), 1.0)
        profile["photo_trust_score"] = round(score, 3)
        profile["photo_checks"] = int(profile.get("photo_checks", 0)) + 1
        profile["photo_flags"] = int(profile.get("photo_flags", 0)) + (1 if suspicious else 0)
        profile["photo_trust_ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.profiles, ensure_ascii=False, indent=1), encoding="utf-8")
        return profile["photo_trust_score"]

    def put(self, nickname: str, data: dict[str, object]) -> dict[str, object]:
        existing = self.profiles.get(nickname) or {}
        profile = {k: data[k] for k in self.KEEP if k in data}
        profile.update({k: existing[k] for k in self.TRUST_FIELDS if k in existing})  # the page never sends these
        profile["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.profiles[nickname] = profile
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.profiles, ensure_ascii=False, indent=1), encoding="utf-8")
        return profile


# Edge fields that reviews / reports / admin overrides write; ReviewStore.rebuild() resets them to these baselines.
REVIEW_FIELDS = {"review_count": 0, "review_score_sum": 0.0, "avoid_by_tag": dict, "avoid_score": 0.0, "vibe_nudge": 0.0, "vehicle_bump": 0.0,
                 "negative_count": 0, "danger_count": 0, "danger_manual": 0, "tree_nudge": 0.0, "park_nudge": 0.0, "avoid_penalty": 0.0}


class ReviewStore:
    """reviews.json = {"reviews", "negatives", "new_axis_candidates", "danger_manual"}, replayed onto the graph at startup."""

    def __init__(self, graph: nx.MultiDiGraph, path: Path = REVIEWS_PATH) -> None:
        self.graph, self.path = graph, path
        self.reviews: list[dict[str, object]] = []
        self.negatives: list[dict[str, object]] = []
        self.candidates: dict[str, dict[str, object]] = {}  # keyword -> {keyword, count, edges, examples, status, first_ts, last_ts}
        self.danger_manual: dict[str, dict[str, object]] = {}  # edge_id ("u-v-k") -> {forced: bool, ts, note}
        self.edge_reviews: dict[Edge, list[tuple[float, str | None, str, float]]] = {}  # per edge: (score, user_id, kind, decay) in arrival order; kind = segment | route
        self.user_id: str | None = None  # whose point of view the graph is currently recomputed for
        self.decay_day: str = time.strftime("%Y-%m-%d")  # the day the decay weights were last computed for

    # ---- persistence ----
    def load(self) -> int:
        if not self.path.exists():
            return 0
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.reviews = [dict(r, id=r.get("id") or entry_id(r)) for r in data.get("reviews", [])]
        self.negatives = [dict(n, id=n.get("id") or entry_id(n)) for n in data.get("negatives", [])]
        self.candidates = {str(c["keyword"]): c for c in data.get("new_axis_candidates", []) if c.get("keyword")}
        self.danger_manual = {str(k): v for k, v in data.get("danger_manual", {}).items()}
        self.replay()
        return len(self.reviews) + len(self.negatives)

    def replay(self) -> None:
        """Apply every stored entry to the (clean) graph, in stored order, each weighted by its time decay."""
        now = time.time()
        self.decay_day = time.strftime("%Y-%m-%d")
        for review in self.reviews:
            edges = parse_edge_id(review["edge_id"])
            weight = decay(review.get("ts"), now)
            self.apply_review(float(review["score"]), list(review.get("good_tags", [])), list(review.get("bad_tags", [])), edges, review.get("user_id"), kind=review.get("kind", "segment"), weight=weight)
            if isinstance(review.get("ai"), dict) and review["ai"].get("axes"):
                self.apply_axes(edges, review["ai"]["axes"], weight)
            photo_ai = review.get("ai_photo")
            if isinstance(photo_ai, dict) and photo_ai.get("accepted") and photo_ai.get("axes"):
                trust = float(photo_ai.get("trust", PHOTO_TRUST_DEFAULT))
                if trust >= PHOTO_TRUST_MIN_APPLY:
                    self.apply_axes(edges, photo_ai["axes"], weight * trust)
        for negative in self.negatives:
            self.apply_negative(NEGATIVE_REASONS.get(str(negative["reason"]), {"avoid": 1.0}), parse_edge_id(negative["edge_id"]), str(negative["reason"]), weight=decay(negative.get("ts"), now))
        for edge_id, override in self.danger_manual.items():
            for u, v, k in self.both_directions(parse_edge_id(edge_id)):
                self.graph[u][v][k]["danger_manual"] = 1 if override.get("forced") else -1

    def refresh_decay(self) -> bool:
        """Once a day, replay the store so every entry's decay weight matches its current age."""
        if self.decay_day == time.strftime("%Y-%m-%d"):
            return False
        self.rebuild()
        return True

    def rebuild(self) -> None:
        """Reset every review-derived edge field and replay the store (after a delete, a bulk reload or the daily decay refresh)."""
        for _, _, _, data in self.graph.edges(keys=True, data=True):
            for field, default in REVIEW_FIELDS.items():
                data[field] = default() if callable(default) else default
            data["vehicle_exposure_score"] = data["vehicle_exposure_osm"]
            data["vibe_score"] = data["vibe_score_osm"]
        self.edge_reviews = {}
        self.user_id = None
        self.replay()
        RouteHandler.route_cache.clear()
        RouteHandler.ranking_cache.clear()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"reviews": self.reviews, "negatives": self.negatives, "new_axis_candidates": list(self.candidates.values()), "danger_manual": self.danger_manual}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    def reset(self) -> Path | None:
        if not self.path.exists():
            return None
        archived = self.path.with_name(f"{self.path.stem}.{time.strftime('%Y%m%d-%H%M%S')}.bak")
        self.path.rename(archived)
        return archived

    def clear(self) -> Path | None:
        """Admin reset: archive reviews.json (like --reset) and put the graph back to the OSM baseline."""
        archived = self.reset()
        self.reviews, self.negatives, self.candidates, self.danger_manual = [], [], {}, {}
        self.rebuild()
        self.save()  # an empty store, so the next load / admin read finds a file
        return archived

    def delete(self, entry: str) -> dict[str, object] | None:
        """Remove one review or report by id and replay the rest; returns the removed entry."""
        for bucket in (self.reviews, self.negatives):
            hit = next((e for e in bucket if e.get("id") == entry), None)
            if hit is not None:
                bucket.remove(hit)
                self.rebuild()
                self.save()
                return hit
        return None

    # ---- reviews ----
    def add_review(self, edges: list[Edge], score: float, good_tags: list[str], bad_tags: list[str], meta: dict[str, object], user_id: str | None = None, kind: str = "segment", ai: dict[str, object] | None = None) -> list[Edge]:
        RouteHandler.route_cache.clear()  # edge scores changed: every cached route may be stale
        RouteHandler.ranking_cache.clear()
        updated = self.apply_review(score, good_tags, bad_tags, edges, user_id, kind)
        review = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "user_id": user_id, "kind": kind, "edge_id": ",".join(f"{u}-{v}-{k}" for u, v, k in updated),
            "score": score, "good_tags": good_tags, "bad_tags": bad_tags, **meta,
        }
        if ai:
            review["ai"] = ai
            self.apply_axes(updated, ai.get("axes") or {})
            self.note_keywords(ai.get("keywords") or [], review["edge_id"], str(meta.get("comment") or ""), review["ts"])
        photo_ai = meta.get("ai_photo")
        if isinstance(photo_ai, dict) and photo_ai.get("accepted"):  # photo findings count only once the user said yes
            trust = float(photo_ai.get("trust", PHOTO_TRUST_DEFAULT))
            if trust >= PHOTO_TRUST_MIN_APPLY:  # a low-trust nickname's photo is recorded but changes nothing
                self.apply_axes(updated, photo_ai.get("axes") or {}, trust)
                self.note_keywords(photo_ai.get("keywords") or [], review["edge_id"], f"[사진] {photo_ai.get('summary') or ''}".strip(), review["ts"])
        review["id"] = entry_id(review)
        self.reviews.append(review)
        self.save()
        return updated

    def apply_axes(self, edges: list[Edge], axes: dict[str, object], weight: float = 1.0) -> None:
        """Move an edge along the axes a review comment talked about (both directions, cumulative, capped, decayed by `weight`)."""
        for u, v, k in self.both_directions(edges):
            data = self.graph[u][v][k]
            for axis, step in AXIS_STEPS.items():
                strength = min(max(float(axes.get(axis, 0) or 0), -1.0), 1.0) * weight
                if not strength:
                    continue
                cap = AXIS_CAPS[axis]
                if axis == "vehicle":
                    data["vehicle_bump"] = min(max(float(data.get("vehicle_bump", 0.0)) + step * strength, -cap), cap)
                    data["vehicle_exposure_score"] = min(max(data["vehicle_exposure_osm"] + data["vehicle_bump"], 0.0), VEHICLE_CAP)
                elif axis == "vibe":
                    data["vibe_nudge"] = min(max(float(data.get("vibe_nudge", 0.0)) + step * strength, -cap), cap)
                else:  # tree / park: read by routing.edge_tree_penalty / edge_park_penalty
                    data[f"{axis}_nudge"] = min(max(float(data.get(f"{axis}_nudge", 0.0)) + step * strength, -cap), cap)
            self.recompute((u, v, k), self.user_id)

    def canonical_keyword(self, word: str) -> str | None:
        """The candidate a word already belongs to: its own entry, or one that lists it as an alias."""
        if word in self.candidates:
            return word
        return next((k for k, c in self.candidates.items() if word in c.get("aliases", [])), None)

    def note_keywords(self, keywords: list[str], edge_id: str, comment: str, ts: str) -> None:
        """Keep count of keywords Claude found that no axis or tag covers (new_axis_candidates in reviews.json).

        A word that already is a candidate or one of its aliases is counted there. Genuinely new words are checked
        against the existing candidates with Claude (match_synonyms): 젊은층 -> 트렌디 etc. join the existing entry as an
        alias instead of starting a duplicate. The check is best effort - if it fails the word is registered on its own.
        """
        targets: dict[str, str] = {}
        unknown = [w for w in dict.fromkeys(keywords) if self.canonical_keyword(w) is None]
        if unknown and RouteHandler.api_enabled and os.environ.get("ANTHROPIC_API_KEY"):
            try:
                targets = match_synonyms(unknown, list(self.candidates))
            except Exception as error:  # noqa: BLE001 - never block a review on this
                print(f"[claude] match_synonyms failed: {type(error).__name__}: {error}", flush=True)
        for word in keywords:
            key = self.canonical_keyword(word) or targets.get(word)
            if key is None:
                key = word
            entry = self.candidates.setdefault(key, {"keyword": key, "count": 0, "edges": [], "examples": [], "aliases": [], "status": "pending", "first_ts": ts, "last_ts": ts})
            if word != key and word not in entry.setdefault("aliases", []):
                entry["aliases"].append(word)
            entry["count"] = int(entry["count"]) + 1
            entry["last_ts"] = ts
            if edge_id not in entry["edges"]:
                entry["edges"].append(edge_id)
            if comment and comment not in entry["examples"]:
                entry["examples"] = (entry["examples"] + [comment])[-5:]

    def keyword_stats(self, words: list[str]) -> dict[str, object]:
        """count / edges / examples / first_ts / last_ts for `words`, recomputed from the stored reviews' ai.keywords.

        Each review keeps the raw keywords Claude extracted, so a merged candidate can be taken apart again exactly.
        """
        wanted = set(words)
        count, edges, examples, stamps = 0, [], [], []
        for r in self.reviews:
            kws = (r.get("ai") or {}).get("keywords") or []
            hits = [w for w in kws if w in wanted]
            if not hits:
                continue
            count += len(hits)
            if r["edge_id"] not in edges:
                edges.append(r["edge_id"])
            if r.get("comment") and r["comment"] not in examples:
                examples.append(r["comment"])
            stamps.append(str(r.get("ts", "")))
        return {"count": count, "edges": edges, "examples": examples[-5:], "first_ts": min(stamps) if stamps else None, "last_ts": max(stamps) if stamps else None}

    def split_candidate(self, alias: str) -> tuple[dict[str, object], dict[str, object]] | None:
        """Undo a merge: `alias` leaves its parent candidate and becomes its own pending entry (admin).

        Both entries are recomputed from the reviews (keyword_stats), so counts stay exact whichever order merges
        and splits happened in.
        """
        parent_key = next((k for k, c in self.candidates.items() if alias in c.get("aliases", [])), None)
        if parent_key is None:
            return None
        parent = self.candidates[parent_key]
        parent["aliases"] = [a for a in parent.get("aliases", []) if a != alias]
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        own = self.keyword_stats([alias])
        self.candidates[alias] = {"keyword": alias, **own, "first_ts": own["first_ts"] or now, "last_ts": own["last_ts"] or now, "aliases": [], "status": "pending", "split_from": parent_key, "split_ts": now}
        rest = self.keyword_stats([parent_key, *parent["aliases"]])
        parent.update({"count": max(rest["count"], 0), "edges": rest["edges"], "examples": rest["examples"] or parent.get("examples", [])[:5]})
        if rest["first_ts"]:
            parent["first_ts"], parent["last_ts"] = rest["first_ts"], rest["last_ts"]
        self.save()
        return self.candidates[alias], parent

    def merge_candidates(self, source: str, target: str) -> dict[str, object] | None:
        """Fold candidate `source` into `target` (admin): counts add up, edges / examples / aliases union, source removed."""
        if source == target or source not in self.candidates or target not in self.candidates:
            return None
        src, dst = self.candidates.pop(source), self.candidates[target]
        dst["count"] = int(dst["count"]) + int(src.get("count", 0))
        dst["edges"] = list(dict.fromkeys(list(dst.get("edges", [])) + list(src.get("edges", []))))
        dst["examples"] = list(dict.fromkeys(list(dst.get("examples", [])) + list(src.get("examples", []))))[-5:]
        dst["aliases"] = list(dict.fromkeys(list(dst.get("aliases", [])) + [source] + list(src.get("aliases", []))))
        dst["first_ts"] = min(str(dst.get("first_ts", "")), str(src.get("first_ts", ""))) or dst.get("first_ts")
        dst["last_ts"] = max(str(dst.get("last_ts", "")), str(src.get("last_ts", ""))) or dst.get("last_ts")
        self.save()
        return dst

    def set_candidate(self, keyword: str, status: str) -> dict[str, object] | None:
        entry = self.candidates.get(keyword)
        if entry is None:
            return None
        entry["status"] = status
        entry["decided_ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.save()
        return entry

    # ---- danger overrides (admin) ----
    def set_danger(self, edges: list[Edge], forced: bool | None, note: str = "") -> list[Edge]:
        """forced=True marks the edges dangerous, False clears them despite reports, None removes the override."""
        touched = sorted(self.both_directions(edges))
        edge_id = ",".join(f"{u}-{v}-{k}" for u, v, k in sorted(e for e in touched if e in set(edges)) or touched)
        # One override per edge: drop earlier entries that mention any of these edges.
        for key in [k for k in self.danger_manual if set(parse_edge_id(k)) & set(touched)]:
            del self.danger_manual[key]
        for u, v, k in touched:
            self.graph[u][v][k]["danger_manual"] = 0 if forced is None else 1 if forced else -1
        if forced is not None:
            self.danger_manual[edge_id] = {"forced": forced, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "note": note[:100]}
        RouteHandler.route_cache.clear()
        self.save()
        return touched

    def apply_review(self, score: float, good_tags: list[str], bad_tags: list[str], edges: list[Edge], user_id: str | None = None, kind: str = "segment", weight: float = 1.0) -> list[Edge]:
        """`weight` is the entry's time decay (1.0 for a new review); it scales the review's vote and its tag effects."""
        touched = self.both_directions(edges)
        for edge in touched:
            data = self.graph[edge[0]][edge[1]][edge[2]]
            self.edge_reviews.setdefault(edge, []).append((score, user_id, kind, weight))
            if kind == "segment":  # raw count/sum shown on segment badges: segment reviews only, not decayed
                data["review_count"] = int(data.get("review_count", 0)) + 1
                data["review_score_sum"] = float(data.get("review_score_sum", 0.0)) + score
            for tag in bad_tags:
                self.apply_effects(NEGATIVE_REASONS[tag], data, tag, weight)
            self.recompute(edge, self.user_id)
        return sorted(e for e in touched if e in set(edges))

    def weighted_reviews(self, edge: Edge, user_id: str | None = None) -> list[tuple[float, float]]:
        """(score, weight) per review on this edge from `user_id`'s point of view, outliers removed."""
        reviews = self.edge_reviews.get(edge, [])
        if not reviews:
            return []
        others = [sc for sc, uid, _, _ in reviews if uid is None or uid != user_id]
        middle = sorted(sc for sc in others if 0 < sc < 10)
        keep: list[tuple[float, float]] = []
        for score, uid, kind, age_weight in reviews:
            own = user_id is not None and uid == user_id
            if not own and score in (0.0, 10.0) and len(middle) >= OUTLIER_MIN_OTHERS:
                median = middle[len(middle) // 2]
                if abs(score - median) >= OUTLIER_GAP:
                    continue  # a sudden 0 or 10 against a settled consensus
            n_others = len(others)
            weight = OWN_WEIGHT if own else STRANGER_MIN + (1.0 - STRANGER_MIN) * min(1.0, max(0, n_others - 1) / (STRANGER_FULL_AT - 1))
            if kind == "route":
                weight *= ROUTE_REVIEW_WEIGHT  # a whole-route star says less about one segment than a segment review
            keep.append((score, weight * age_weight))  # time decay: old reviews fade out of the average
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
            data["avoid_penalty"] = AVOID_CAP if is_danger(data) else min(float(data.get("avoid_score", 0.0)) + tagged, AVOID_CAP)

    def stats(self, edges: list[Edge], user_id: str | None = None, full: bool = False) -> dict[str, object]:
        """Reviews touching any of the edges (a review counts once even if it covers several of them).

        full=True adds `reviews`: every segment review as the page's review list shows it (tags, AI-added tags,
        comment, photos, reviewer purpose/subs for the taste-match sort, 도움돼요 count).
        """
        existing = [e for e in edges if self.graph.has_edge(*e)]
        wanted = set(existing) | {(v, u, k) for u, v, k in existing}
        hits = [r for r in self.reviews if wanted & set(parse_edge_id(r["edge_id"]))]
        reports = [n for n in self.negatives if wanted & set(parse_edge_id(n["edge_id"]))]
        route_hits = [r for r in hits if r.get("kind") == "route"]
        hits = [r for r in hits if r.get("kind", "segment") == "segment"]
        good: Counter[str] = Counter()
        bad: Counter[str] = Counter()
        for review in hits:
            good.update(review.get("good_tags", []))
            bad.update(review.get("bad_tags", []))
        bad.update(str(n["reason"]) for n in reports)
        by_purpose: dict[str, list[float]] = {}
        for review in hits:
            purpose = (review.get("profile") or {}).get("purpose")
            if purpose:
                by_purpose.setdefault(purpose, []).append(float(review["score"]))
        return {
            "edge_id": ",".join(f"{u}-{v}-{k}" for u, v, k in existing), "edge_count": len(existing),
            "review_count": len(hits), "score_avg": round(sum(float(r["score"]) for r in hits) / len(hits), 2) if hits else None,
            "own_count": sum(1 for r in hits if user_id and r.get("user_id") == user_id),
            "route_rating_count": len(route_hits), "route_rating_avg": round(sum(float(r.get("route_score", 0)) for r in route_hits) / len(route_hits), 2) if route_hits else None,
            "by_purpose": {k: {"name": PURPOSE_NAMES.get(k, k), "avg": round(sum(v) / len(v), 1), "count": len(v)} for k, v in sorted(by_purpose.items(), key=lambda kv: -len(kv[1]))},
            "comments": [{"user_id": mask_nickname(r.get("user_id")), "score": r.get("score"), "comment": r["comment"], "ts": r.get("ts"), "purpose": (r.get("profile") or {}).get("purpose")} for r in reversed(hits) if r.get("comment")][:8],
            "danger": any(is_danger(self.graph[u][v][k]) for u, v, k in existing),
            "negative_count": len(reports),
            "good_tags": dict(good.most_common()), "bad_tags": dict(bad.most_common()),
            "photos": [{"url": url, "user_id": r.get("user_id") if user_id and r.get("user_id") == user_id else mask_nickname(r.get("user_id")), "score": r.get("score"), "ts": r.get("ts")} for r in hits for url in r.get("photos", [])],
            "vibe_score_avg": round(sum(self.graph[u][v][k]["vibe_score"] for u, v, k in existing) / len(existing), 3) if existing else None,
            **({"reviews": [self.review_card(r, user_id) for r in reversed(hits)][:200]} if full else {}),
        }

    @staticmethod
    def review_card(r: dict[str, object], user_id: str | None = None) -> dict[str, object]:
        ai = r.get("ai") if isinstance(r.get("ai"), dict) else {}
        auto_good, auto_bad = set(ai.get("auto_good_tags", [])), set(ai.get("auto_bad_tags", []))
        helpful = [str(u) for u in r.get("helpful", [])]
        profile = r.get("profile") or {}
        mine = bool(user_id and r.get("user_id") == user_id)
        return {
            # Other people's nicknames are masked ("이영규" -> "이**"); the requester sees their own in full.
            "id": r.get("id"), "ts": r.get("ts"), "user_id": r.get("user_id") if mine else mask_nickname(r.get("user_id")), "score": r.get("score"),
            "good_tags": [t for t in r.get("good_tags", []) if t not in auto_good], "bad_tags": [t for t in r.get("bad_tags", []) if t not in auto_bad],
            "ai_good_tags": sorted(auto_good), "ai_bad_tags": sorted(auto_bad), "ai_summary": ai.get("summary") or None,
            "comment": r.get("comment"), "photos": list(r.get("photos", [])),
            "purpose": profile.get("purpose"), "subs": list(profile.get("subs", [])),
            "helpful_count": len(helpful), "helped": bool(user_id and user_id in helpful), "mine": mine,
        }

    def toggle_helpful(self, review_id: str, user_id: str) -> dict[str, object] | None:
        """도움돼요 on a review (one per nickname, toggles); returns the updated card."""
        review = next((r for r in self.reviews if r.get("id") == review_id), None)
        if review is None:
            return None
        helpful = [str(u) for u in review.get("helpful", [])]
        if user_id in helpful:
            helpful.remove(user_id)
        else:
            helpful.append(user_id)
        review["helpful"] = helpful
        self.save()
        return self.review_card(review, user_id)

    # ---- tag-only negative feedback (POST /api/feedback/negative) ----
    def add_negative(self, reason: str, edges: list[Edge], meta: dict[str, object]) -> list[Edge]:
        RouteHandler.route_cache.clear()
        updated = self.apply_negative(NEGATIVE_REASONS[reason], edges, reason)
        entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "edge_id": ",".join(f"{u}-{v}-{k}" for u, v, k in updated), "reason": reason, **meta}
        entry["id"] = entry_id(entry)
        self.negatives.append(entry)
        self.save()
        return updated

    def apply_negative(self, effects: dict[str, float], edges: list[Edge], tag: str = "", weight: float = 1.0) -> list[Edge]:
        touched = self.both_directions(edges)
        for edge in touched:
            data = self.graph[edge[0]][edge[1]][edge[2]]
            data["negative_count"] = int(data.get("negative_count", 0)) + 1
            self.apply_effects(effects, data, tag, weight)
            self.recompute(edge, self.user_id)
        return sorted(e for e in touched if e in set(edges))

    @staticmethod
    def apply_effects(effects: dict[str, float], data: dict[str, object], tag: str = "", weight: float = 1.0) -> None:
        """Cumulative tag effects on an edge, scaled by the entry's time decay `weight`."""
        if tag in DANGER_TAGS:
            data["danger_count"] = float(data.get("danger_count", 0.0)) + weight  # >= DANGER_THRESHOLD -> avoid_penalty forced to AVOID_CAP
        if "avoid" in effects:
            by_tag = data.setdefault("avoid_by_tag", {})
            by_tag[tag] = min(by_tag.get(tag, 0.0) + effects["avoid"] * weight, AVOID_CAP)
        if "vehicle" in effects:
            data["vehicle_bump"] = float(data.get("vehicle_bump", 0.0)) + effects["vehicle"] * weight
            data["vehicle_exposure_score"] = min(max(data["vehicle_exposure_osm"] + data["vehicle_bump"], 0.0), VEHICLE_CAP)
        if "vibe" in effects:
            data["vibe_nudge"] = max(float(data.get("vibe_nudge", 0.0)) + effects["vibe"] * weight, -1.0)

    def both_directions(self, edges: list[Edge]) -> set[Edge]:
        touched: set[Edge] = set()
        for u, v, k in edges:
            for edge in ((u, v, k), (v, u, k)):
                if self.graph.has_edge(*edge):
                    touched.add(edge)
        return touched


def route_anchors(coords: list[tuple[float, float]], count: int = POI_ANCHORS) -> list[tuple[float, float]]:
    """Midpoints of `count` equal-length stretches of a (lon, lat) polyline, as (lat, lon)."""
    if not coords:
        return []
    if len(coords) == 1:
        return [(coords[0][1], coords[0][0])]
    steps = [haversine_m((a[1], a[0]), (b[1], b[0])) for a, b in zip(coords, coords[1:])]
    total = sum(steps) or 1.0
    targets = [(i + 0.5) / count * total for i in range(count)]
    anchors: list[tuple[float, float]] = []
    walked, j = 0.0, 0
    for target in targets:
        while j < len(steps) - 1 and walked + steps[j] < target:
            walked += steps[j]
            j += 1
        frac = min(max((target - walked) / steps[j], 0.0), 1.0) if steps[j] else 0.0
        (x1, y1), (x2, y2) = coords[j], coords[j + 1]
        anchors.append((y1 + (y2 - y1) * frac, x1 + (x2 - x1) * frac))
    return anchors


def kakao_category_search(key: str, code: str, lat: float, lon: float, radius_m: float, size: int = 5) -> list[dict[str, object]]:
    """One Kakao Local category search around a point, closest first; raises urllib errors for the caller to map."""
    query = urlencode({"category_group_code": code, "x": f"{lon:.6f}", "y": f"{lat:.6f}", "radius": int(radius_m), "sort": "distance", "size": size})
    request = urllib.request.Request(f"{KAKAO_URL}?{query}", headers={"Authorization": f"KakaoAK {key}", "KA": "sdk/1.0 os/python origin/walk-route"})
    with urllib.request.urlopen(request, timeout=6) as response:
        return json.load(response).get("documents", [])


def route_pois(key: str, coords: list[tuple[float, float]], categories: list[str]) -> dict[str, object]:
    """POIs along a route: POI_ANCHORS anchors x categories, POI_PER_ANCHOR closest each, deduplicated by place id."""
    anchors = route_anchors(coords)
    jobs = [(i, anchor, cat) for i, anchor in enumerate(anchors) for cat in categories]
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda job: kakao_category_search(key, KAKAO_CATEGORIES[job[2]], job[1][0], job[1][1], POI_RADIUS_M), jobs))
    pois: dict[str, dict[str, object]] = {}
    for (i, anchor, cat), docs in zip(jobs, results):
        for doc in docs[:POI_PER_ANCHOR]:
            pid = str(doc.get("id"))
            if pid in pois:
                continue  # already found from a neighbouring anchor
            pois[pid] = {
                "id": pid, "name": doc.get("place_name"), "category": cat, "category_name": doc.get("category_name"),
                "place_url": doc.get("place_url"), "x": float(doc.get("x", 0)), "y": float(doc.get("y", 0)),  # x = lon, y = lat (Kakao)
                "address": doc.get("road_address_name") or doc.get("address_name"), "phone": doc.get("phone") or None,
                "distance_m": int(float(doc.get("distance") or 0)), "anchor": i,
            }
    return {"pois": list(pois.values()), "anchors": [[round(lat, 6), round(lon, 6)] for lat, lon in anchors], "radius_m": POI_RADIUS_M, "categories": categories}


def axis_guide(keyword: str) -> list[str]:
    """What to change to turn an approved keyword into a routing axis (shown on the admin page)."""
    key = re.sub(r"[^a-z0-9]", "", keyword.lower()) or "new"
    return [
        f"1. 데이터: 엣지마다 '{keyword}' 신호가 필요합니다. 리뷰 키워드 카운트(new_axis_candidates.edges)나 외부 데이터로 edge 속성 '{key}_score'(0~1)를 만들고 rescore_*.py처럼 graphml에 저장하세요.",
        f"2. routing.py: load_graph()에서 data['{key}_score']를 정규화해 읽고, edge_{key}_penalty(data) = length * (1 - score)를 추가하세요. Weights에 w_{key}: float = 0.0 필드를, edge_cost / apply_cost / CostTable.__init__ / Route.penalty_m / route_feature 통계에 w_{key} 항을 더하세요.",
        f"3. server.py: parse_params()의 가중치 이름 목록에 'w_{key}'를, INTERPRET_SYSTEM 축 설명과 INTERPRET_SCHEMA에 w_{key}를, AXIS_STEPS / AXIS_CAPS와 ANALYZE_SYSTEM에 '{key}' 축을 추가하세요.",
        f"4. web/index.html: state.weights에 w_{key}: 0을, PURPOSES 프리셋의 w 배열과 weightParams()에 새 축을, 카드 지표(metric)에 표시를 추가하세요.",
        "5. validate_candidate_routes.py로 가중치 민감도를 다시 확인하고 커밋하세요.",
    ]


class VisitLog:
    """Daily page views, route searches and distinct nicknames (visits.json), for the admin stats tab."""

    def __init__(self, path: Path = VISITS_PATH) -> None:
        self.path = path
        self.days: dict[str, dict[str, object]] = {}
        if path.exists():
            try:
                self.days = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self.days = {}
        self.dirty = False

    def hit(self, kind: str, user_id: str | None = None) -> None:
        day = self.days.setdefault(time.strftime("%Y-%m-%d"), {"page": 0, "routes": 0, "users": []})
        day[kind] = int(day.get(kind, 0)) + 1
        if user_id and user_id not in day["users"]:
            day["users"].append(user_id)
        self.dirty = True

    def flush(self) -> None:
        if not self.dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.days, ensure_ascii=False), encoding="utf-8")
        self.dirty = False

    def rows(self, days: int = 30) -> list[dict[str, object]]:
        return [{"date": d, "page": v.get("page", 0), "routes": v.get("routes", 0), "users": len(v.get("users", []))} for d, v in sorted(self.days.items())[-days:]]


class RouteHandler(SimpleHTTPRequestHandler):
    """Static files from web/, GET /api/route, GET /api/reviews/{edge_id}, POST /api/review, POST /api/feedback/negative."""

    graph: nx.MultiDiGraph
    graph_no_steps: nx.MultiDiGraph  # same edge data as graph, stair edges removed
    reviews: ReviewStore
    profiles: ProfileStore
    visits: VisitLog
    api_enabled: bool = True  # --no-api turns the Claude endpoints off (demo switch)
    ai_token_value: str | None = None  # None when AI_PASSWORD is empty: AI endpoints need no extra unlock
    admin_token_value: str | None = None  # None when ADMIN_PASSWORD is empty: /admin is disabled
    route_cache: dict[tuple, dict[str, object]] = {}  # request key -> GeoJSON; dropped whenever a review/report changes the graph
    poi_routes: dict[str, list[tuple[float, float]]] = {}  # route_id -> (lon, lat) polyline, for GET /api/poi?route_id=
    poi_cache: dict[tuple[str, tuple[str, ...]], tuple[float, dict[str, object]]] = {}  # (route_id, categories) -> (timestamp, result)
    ranking_cache: dict[tuple, tuple[float, str, bool]] = {}  # (segment edge_id, period) -> (timestamp, summary, from Claude?)
    summary_jobs: set[tuple[str, str]] = set()  # (edge_id, period) summaries a worker is generating right now
    summary_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="summary")  # at most two Claude calls in flight
    ROUTE_CACHE_MAX = 256
    POI_ROUTES_MAX = POI_ROUTES_MAX
    lock = threading.Lock()  # shortest_route rewrites edge costs in place, so one search at a time

    def do_GET(self) -> None:
        url = urlparse(self.path)
        query = {k: v[0] for k, v in parse_qs(url.query, keep_blank_values=True).items()}
        if url.path == "/api/status":
            return self.send_json({"api_enabled": self.api_enabled and bool(os.environ.get("ANTHROPIC_API_KEY")), "api_flag": self.api_enabled, "has_key": bool(os.environ.get("ANTHROPIC_API_KEY")), "ai_locked": self.ai_token_value is not None, "kakao": bool(os.environ.get("KAKAO_API_KEY")),
                                   "kakao_js_key": os.environ.get("KAKAO_JS_KEY", "").strip() or None,  # JavaScript 키 (카카오맵 SDK용; 도메인 등록된 공개 키)
                                   "vworld_key": os.environ.get("VWORLD_API_KEY", "").strip() or None})  # VWorld 타일 키 (페이지가 타일 URL에 넣음; 도메인 제한 공개 키)
        if url.path == "/api/ai-auth":
            if self.ai_token_value is None:
                return self.send_json({"ok": True, "token": ""})
            if query.get("token") == self.ai_token_value or ("password" in query and ai_token(query["password"]) == self.ai_token_value):
                return self.send_json({"ok": True, "token": self.ai_token_value})
            return self.send_json({"ok": False, "error": "AI 비밀번호가 맞지 않습니다."}, HTTPStatus.UNAUTHORIZED)
        if url.path.startswith("/api/admin/"):
            return self.admin_get(url.path[len("/api/admin/"):], query)
        if url.path == "/api/route":
            with self.lock:
                self.visits.hit("routes", parse_user(query))
                self.visits.flush()
            try:
                od = self.parse_params(query)
                waypoints, revisit_penalty = parse_waypoints(query), parse_revisit_penalty(query)
                keep_order = str(query.get("keep_order", "0")).lower() in ("1", "true", "yes")
            except ValueError as error:
                return self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            with self.lock:
                try:
                    self.send_json(self.route(*od, tags=parse_tags(query), avoid_tags=parse_avoid_tags(query), user_id=parse_user(query), waypoints=waypoints, revisit_penalty=revisit_penalty, keep_order=keep_order))
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
                self.send_json(self.reviews.stats(edges, parse_user(query), full=str(query.get("full", "0")).lower() in ("1", "true", "yes")))
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
        elif url.path == "/api/user/reviews":
            # My reviews page: every entry I left, newest first, split into 긍정 (good tags or score >= 6) and
            # 신고 (bad tags, score <= 5, or a tag-only report). An entry with both good and bad tags shows in both.
            user_id = parse_user(query)
            if not user_id:
                return self.send_json({"error": "user_id is required"}, HTTPStatus.BAD_REQUEST)
            with self.lock:
                entries: list[dict[str, object]] = []
                for r in self.reviews.reviews:
                    if r.get("user_id") != user_id:
                        continue
                    card = ReviewStore.review_card(r, user_id)
                    score, kind = float(r.get("score", 0)), r.get("kind", "segment")
                    good, bad = list(r.get("good_tags", [])), list(r.get("bad_tags", []))
                    entries.append({
                        **card, "type": "route" if kind == "route" else "review", "route_score": r.get("route_score"),
                        "positive": kind == "route" and int(r.get("route_score") or 0) >= 3 or (kind != "route" and (bool(good) or score >= 6)),
                        "report": kind != "route" and (bool(bad) or score <= 5),
                        **self.segment_summary_for(r["edge_id"]),
                    })
                for n in self.reviews.negatives:
                    if n.get("user_id") != user_id:
                        continue
                    entries.append({
                        "id": n.get("id"), "ts": n.get("ts"), "user_id": user_id, "type": "report", "score": None, "reason": n.get("reason"),
                        "good_tags": [], "bad_tags": [str(n.get("reason"))], "ai_good_tags": [], "ai_bad_tags": [], "comment": None, "photos": [],
                        "positive": False, "report": True, "mine": True, "helpful_count": 0, "helped": False,
                        **self.segment_summary_for(n["edge_id"]),
                    })
                entries.sort(key=lambda e: str(e.get("ts", "")), reverse=True)
            return self.send_json({"user_id": user_id, "entries": entries, "positive_count": sum(1 for e in entries if e["positive"]), "report_count": sum(1 for e in entries if e["report"])})
        elif url.path == "/api/ranking":
            try:
                period = str(query.get("period", "week"))
                if period not in ("week", "month"):
                    raise ValueError
                lat, lon = float(query["lat"]), float(query["lon"])
                radius = float(query.get("radius", 2000))
            except (KeyError, TypeError, ValueError):
                return self.send_json({"error": "expected period=week|month, lat, lon (and optional radius)"}, HTTPStatus.BAD_REQUEST)
            with self.lock:
                ranking = self.ranking(period, (lat, lon), radius)
            self.send_json({"period": period, "center": [lat, lon], "radius_m": radius, "min_reviews": RANK_MIN_REVIEWS, "segments": ranking})
        elif url.path == "/api/poi":
            self.poi(query)
        elif url.path == "/api/segments/search":
            q = str(query.get("q", "")).strip()
            if len(q) < 1:
                return self.send_json({"error": "q is required"}, HTTPStatus.BAD_REQUEST)
            with self.lock:
                self.send_json({"q": q, "segments": self.search_segments(q, parse_user(query))})
        elif url.path == "/api/danger-zones":
            try:
                lat, lon = float(query["lat"]), float(query["lon"])
                radius = float(query.get("radius", 2000))
            except (KeyError, TypeError, ValueError):
                return self.send_json({"error": "expected lat, lon (and optional radius)"}, HTTPStatus.BAD_REQUEST)
            with self.lock:
                zones = self.danger_zones((lat, lon), radius)
            self.send_json({"center": [lat, lon], "radius_m": radius, "threshold": DANGER_THRESHOLD, "tags": sorted(DANGER_TAGS), "zones": zones})
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
        elif url.path == "/ranking":
            self.path = "/index.html"  # the page opens its ranking view from the URL
            super().do_GET()
        elif url.path == "/review":
            self.path = "/index.html"  # the page's phone review form lives at /review?edge_id=&route_id= (history API)
            super().do_GET()
        elif url.path == "/admin":
            self.path = "/admin.html"
            super().do_GET()
        else:
            if url.path in ("/", "/index.html"):
                with self.lock:
                    self.visits.hit("page")
                    self.visits.flush()
            super().do_GET()

    def ranking(self, period: str, center: tuple[float, float], radius_m: float) -> list[dict[str, object]]:
        """Popular segments: segment reviews in the period, grouped by the stretch they were left on, near `center`.

        A review covers the edges of one page segment (a run of edges with the same street name). Reviews whose
        edge sets overlap are merged into one entry, so the same street rated from two different routes counts once.
        """
        since = time.time() - (7 if period == "week" else 30) * 86400
        undirected = lambda u, v, k: (min(u, v), max(u, v), k)  # noqa: E731 - both directions are the same stretch on foot
        recent: list[tuple[frozenset[Edge], dict[str, object]]] = []
        for r in self.reviews.reviews:
            if r.get("kind", "segment") != "segment":
                continue  # whole-route stars say nothing about one stretch
            try:
                ts = time.mktime(time.strptime(str(r["ts"]), "%Y-%m-%dT%H:%M:%S"))
                edges = parse_edge_id(r["edge_id"])
            except ValueError:
                continue
            if ts < since:
                continue
            # A page segment is one street (same name + highway). Reviews spanning several streets are whole-route
            # ratings from before `kind` existed (or any multi-street stretch) and say nothing about one stretch.
            streets = {self.street_key(e) for e in edges if self.graph.has_edge(*e) or self.graph.has_edge(e[1], e[0], e[2])}
            if len(streets) != 1:
                continue
            recent.append((frozenset(undirected(*e) for e in edges), r))
        # Merge overlapping stretches, largest first, so a short crossing rated inside a long street joins the street.
        groups: list[tuple[set[Edge], list[dict[str, object]]]] = []
        for key, r in sorted(recent, key=lambda kr: -len(kr[0])):
            hit = next((g for g in groups if g[0] & key), None)
            if hit is None:
                groups.append((set(key), [r]))
            else:
                hit[0].update(key)
                hit[1].append(r)
        eligible = [(k, rs) for k, rs in groups if len(rs) >= RANK_MIN_REVIEWS]
        prior = sum(float(r["score"]) for _, rs in eligible for r in rs) / max(1, sum(len(rs) for _, rs in eligible))
        rows: list[dict[str, object]] = []
        for key, reviews in eligible:
            # Directed edges that exist in the graph (one per stretch), in the order of the longest review's listing.
            longest = max(reviews, key=lambda r: len(parse_edge_id(r["edge_id"])))
            ordered = [e if self.graph.has_edge(*e) else (e[1], e[0], e[2]) for e in parse_edge_id(longest["edge_id"])]
            ordered = [e for e in ordered if self.graph.has_edge(*e)]
            extra = [next((d for d in ((u, v, k), (v, u, k)) if self.graph.has_edge(*d)), None) for u, v, k in key - {undirected(*e) for e in ordered}]
            edges = ordered + [e for e in extra if e is not None]
            if not edges:
                continue
            coords = route_coordinates(self.graph, ordered or edges)
            mid = coords[len(coords) // 2]  # (lon, lat)
            if haversine_m(center, (mid[1], mid[0])) > radius_m:
                continue
            datas = [self.graph[u][v][k] for u, v, k in edges]
            names = Counter(n for n in (edge_name(d) for d in datas) if n)
            highways = Counter(h for d in datas for h in highway_values(d.get("highway")))
            highway = highways.most_common(1)[0][0] if highways else None
            n, avg = len(reviews), sum(float(r["score"]) for r in reviews) / len(reviews)
            good: Counter[str] = Counter()
            bad: Counter[str] = Counter()
            for r in reviews:
                good.update(r.get("good_tags", []))
                bad.update(r.get("bad_tags", []))
            edge_id = ",".join(f"{u}-{v}-{k}" for u, v, k in edges)
            rows.append({
                "edge_id": edge_id, "edge_count": len(edges),
                "name": names.most_common(1)[0][0] if names else RANK_HIGHWAY_NAMES.get(highway or "", "이름 없는 길"),
                "highway": highway, "highway_name": RANK_HIGHWAY_NAMES.get(highway or "", highway),
                "length_m": round(sum(d["length"] for d in datas), 1),
                "score_avg": round(avg, 2), "stars": round(avg / 2, 2), "review_count": n,
                "score": round((n * avg + RANK_PRIOR_WEIGHT * prior) / (n + RANK_PRIOR_WEIGHT), 3),
                "good_tags": dict(good.most_common()), "bad_tags": dict(bad.most_common()),
                **self.segment_summary(edge_id, period, [str(r["comment"]) for r in reviews if r.get("comment")], good, bad, avg),
                "photos": [url for r in reviews for url in r.get("photos", [])][:3],
                "midpoint": [round(mid[1], 6), round(mid[0], 6)],  # [lat, lon]: what the page adds as a waypoint
                "coordinates": [[round(lon, 6), round(lat, 6)] for lon, lat in coords],
            })
        rows.sort(key=lambda r: (-r["score"], -r["review_count"], -r["length_m"]))
        for i, row in enumerate(rows, 1):
            row["rank"] = i
        return rows[:10]

    def danger_zones(self, center: tuple[float, float], radius_m: float) -> list[dict[str, object]]:
        """Stretches with >= DANGER_THRESHOLD 무서움/어두움 reports (reviews' bad tags + tag-only reports) near `center`.

        Flagged edges are grouped by street and adjacency (edges sharing a node on the same street form one
        zone). Each zone carries one polyline per edge so the page can draw them exactly, plus a midpoint for the marker.
        """
        flagged: dict[Edge, dict[str, object]] = {}
        for u, v, k, data in self.graph.edges(keys=True, data=True):
            if is_danger(data) and (v, u, k) not in flagged:
                flagged[(u, v, k)] = data
        # Union-find over shared nodes, per street.
        parent: dict[Edge, Edge] = {e: e for e in flagged}

        def find(e: Edge) -> Edge:
            while parent[e] != e:
                parent[e] = parent[parent[e]]
                e = parent[e]
            return e

        by_node: dict[tuple[object, tuple[str | None, str]], Edge] = {}
        for e in flagged:
            street = self.street_key(e)
            for node in (e[0], e[1]):
                other = by_node.setdefault((node, street), e)
                if other is not e:
                    parent[find(e)] = find(other)
        groups: dict[Edge, list[Edge]] = {}
        for e in flagged:
            groups.setdefault(find(e), []).append(e)
        zones: list[dict[str, object]] = []
        for edges in groups.values():
            undirected = {(min(u, v), max(u, v), k) for u, v, k in edges}
            lines = [[[round(lon, 6), round(lat, 6)] for lon, lat in route_coordinates(self.graph, [e])] for e in edges]
            longest = max(lines, key=len)
            mid = longest[len(longest) // 2]  # [lon, lat]
            if haversine_m(center, (mid[1], mid[0])) > radius_m:
                continue
            # Per-tag counts from the stored entries (danger_count on the edge is their sum, both tags together).
            tags: Counter[str] = Counter()
            reporters: set[str] = set()
            for r in self.reviews.reviews:
                if r.get("kind", "segment") == "segment" and undirected & {(min(u, v), max(u, v), k) for u, v, k in parse_edge_id(r["edge_id"])}:
                    hits = [t for t in r.get("bad_tags", []) if t in DANGER_TAGS]
                    tags.update(hits)
                    if hits and r.get("user_id"):
                        reporters.add(str(r["user_id"]))
            for n in self.reviews.negatives:
                if str(n["reason"]) in DANGER_TAGS and undirected & {(min(u, v), max(u, v), k) for u, v, k in parse_edge_id(n["edge_id"])}:
                    tags[str(n["reason"])] += 1
                    if n.get("user_id"):
                        reporters.add(str(n["user_id"]))
            datas = [self.graph[u][v][k] for u, v, k in edges]
            name, highway = self.street_key(edges[0])
            zones.append({
                "edge_id": ",".join(f"{u}-{v}-{k}" for u, v, k in edges), "edge_count": len(edges),
                "name": name or RANK_HIGHWAY_NAMES.get(highway, "이름 없는 길"), "highway": highway or None, "highway_name": RANK_HIGHWAY_NAMES.get(highway, highway or None),
                "length_m": round(sum(d["length"] for d in datas), 1),
                "danger_count": round(max(float(d.get("danger_count", 0)) for d in datas), 2),  # decayed sum of 무서움/어두움 reports
                "manual": next((int(d.get("danger_manual", 0)) for d in datas if int(d.get("danger_manual", 0))), 0),  # 1 = admin-forced
                "tags": {tag: tags.get(tag, 0) for tag in sorted(DANGER_TAGS)}, "reporter_count": len(reporters),
                "midpoint": [mid[1], mid[0]], "lines": lines,
            })
        zones.sort(key=lambda z: (-z["danger_count"], -z["length_m"]))
        return zones

    SEGMENT_CHUNK_M = 300.0  # long streets are cut into stretches of about this length for reviewing
    SEGMENT_SEARCH_MAX = 30

    def search_segments(self, q: str, user_id: str | None = None) -> list[dict[str, object]]:
        """Street-name search over the graph: matching streets, split into connected stretches (<= SEGMENT_CHUNK_M).

        Each result has the same shape the page's route segments use (name, highway, edge_id, length_m, coordinates,
        review_count, score_avg), so it plugs straight into the review popup / page.
        """
        needle = q.lower().replace(" ", "")
        by_street: dict[tuple[str, str], list[Edge]] = {}
        seen: set[Edge] = set()
        for u, v, k, data in self.graph.edges(keys=True, data=True):
            name = edge_name(data)
            if not name or needle not in name.lower().replace(" ", ""):
                continue
            key = (min(u, v), max(u, v), k)
            if key in seen:
                continue
            seen.add(key)
            by_street.setdefault((name, "|".join(sorted(highway_values(data.get("highway"))))), []).append((u, v, k))
        results: list[dict[str, object]] = []
        for (name, highway), edges in by_street.items():
            for chain in self.edge_chains(edges):
                for chunk in self.chunk_chain(chain):
                    datas = [self.graph[u][v][k] for u, v, k in chunk]
                    coords = route_coordinates(self.graph, chunk)
                    if len(coords) < 2:
                        continue
                    counts = [number(d.get("review_count")) for d in datas]
                    total = sum(counts)
                    mid = coords[len(coords) // 2]
                    results.append({
                        "name": name, "highway": highway or None, "highway_name": RANK_HIGHWAY_NAMES.get(highway, highway or None),
                        "edge_id": ",".join(f"{u}-{v}-{k}" for u, v, k in chunk), "edges": [[u, v, k] for u, v, k in chunk],
                        "length_m": round(sum(d["length"] for d in datas), 1),
                        "review_count": int(max(counts)) if counts else 0,
                        "score_avg": round(sum(number(d.get("review_score_sum")) for d in datas) / total, 1) if total else None,
                        "negative_count": int(max(number(d.get("negative_count")) for d in datas)) if datas else 0,
                        "danger": any(is_danger(d) for d in datas),
                        "midpoint": [round(mid[1], 6), round(mid[0], 6)],
                        "coordinates": [[round(lon, 6), round(lat, 6)] for lon, lat in coords],
                    })
        # exact / prefix matches first, then longer stretches
        norm = lambda n: n.lower().replace(" ", "")  # noqa: E731
        results.sort(key=lambda r: (norm(r["name"]) != needle, not norm(r["name"]).startswith(needle), r["name"], -r["length_m"]))
        return results[: self.SEGMENT_SEARCH_MAX]

    def edge_chains(self, edges: list[Edge]) -> list[list[Edge]]:
        """Split a street's edges into connected chains, each ordered end to end (branches are appended in walk order)."""
        by_node: dict[object, list[Edge]] = {}
        for e in edges:
            by_node.setdefault(e[0], []).append(e)
            by_node.setdefault(e[1], []).append(e)
        left = set(edges)
        chains: list[list[Edge]] = []
        while left:
            # start from a chain end (a node touching one remaining edge) when there is one
            start = next((e for e in left if len([x for x in by_node[e[0]] if x in left]) == 1 or len([x for x in by_node[e[1]] if x in left]) == 1), next(iter(left)))
            chain = [start]; left.discard(start)
            # extend forward from the end node, then backward from the start node
            for direction in (1, -1):
                node = chain[-1][1] if direction == 1 else chain[0][0]
                while True:
                    nxt = next((x for x in by_node.get(node, []) if x in left), None)
                    if nxt is None:
                        break
                    left.discard(nxt)
                    if direction == 1:
                        if nxt[0] != node:
                            nxt = (nxt[1], nxt[0], nxt[2]) if self.graph.has_edge(nxt[1], nxt[0], nxt[2]) else nxt
                        chain.append(nxt); node = nxt[1]
                    else:
                        if nxt[1] != node:
                            nxt = (nxt[1], nxt[0], nxt[2]) if self.graph.has_edge(nxt[1], nxt[0], nxt[2]) else nxt
                        chain.insert(0, nxt); node = nxt[0]
            chains.append(chain)
        return chains

    def chunk_chain(self, chain: list[Edge]) -> list[list[Edge]]:
        chunks: list[list[Edge]] = [[]]
        length = 0.0
        for e in chain:
            step = self.graph[e[0]][e[1]][e[2]]["length"]
            if chunks[-1] and length + step > self.SEGMENT_CHUNK_M:
                chunks.append([]); length = 0.0
            chunks[-1].append(e); length += step
        return [c for c in chunks if c]

    def street_key(self, edge: Edge) -> tuple[str | None, str]:
        """(name, highway) of an edge in either direction: the same grouping routing.route_segments uses."""
        u, v, k = edge
        data = self.graph[u][v][k] if self.graph.has_edge(u, v, k) else self.graph[v][u][k]
        return edge_name(data), "|".join(sorted(highway_values(data.get("highway"))))

    def segment_summary(self, key: str, period: str, comments: list[str], good: Counter[str], bad: Counter[str], score_avg: float | None) -> dict[str, object]:
        """{summary, summary_ai, summary_pending}: the cached Claude summary if there is one, else the tag fallback.

        A missing summary is generated in the background (summary_pool) so /api/ranking returns at once; the page
        polls POST /api/ranking/summaries until summary_pending clears. Cached for RANK_SUMMARY_TTL.
        """
        cached = self.ranking_cache.get((key, period))
        if cached and time.time() - cached[0] < RANK_SUMMARY_TTL:
            return {"summary": cached[1], "summary_ai": cached[2], "summary_pending": False}
        fallback = fallback_summary(good, bad, score_avg)
        if not (comments and self.api_enabled and os.environ.get("ANTHROPIC_API_KEY")):
            return {"summary": fallback, "summary_ai": False, "summary_pending": False}
        if (key, period) not in self.summary_jobs:
            self.summary_jobs.add((key, period))
            tags = {**good, **{f"⚑{k}": v for k, v in bad.items()}}
            self.summary_pool.submit(self.summary_worker, key, period, list(comments), tags, fallback)
        return {"summary": fallback, "summary_ai": False, "summary_pending": True}

    @classmethod
    def summary_worker(cls, key: str, period: str, comments: list[str], tags: dict[str, int], fallback: str) -> None:
        """Background Claude call; on failure the fallback is cached briefly (RANK_SUMMARY_RETRY) so it is not retried in a loop."""
        try:
            summary, stamp, ai = summarize_reviews(comments, tags), time.time(), True
        except Exception as error:  # noqa: BLE001 - ranking must not fail because of the summary
            print(f"[claude] summarize_reviews failed: {type(error).__name__}: {error}", flush=True)
            summary, stamp, ai = fallback, time.time() - RANK_SUMMARY_TTL + RANK_SUMMARY_RETRY, False
        with cls.lock:
            cls.ranking_cache[(key, period)] = (stamp, summary, ai)
            cls.summary_jobs.discard((key, period))

    def ranking_summaries(self, period: str, edge_ids: list[str]) -> dict[str, dict[str, object]]:
        """Current state of the requested summaries (for the page's polling), without triggering new jobs."""
        now = time.time()
        out: dict[str, dict[str, object]] = {}
        for edge_id in edge_ids:
            cached = self.ranking_cache.get((edge_id, period))
            if cached and now - cached[0] < RANK_SUMMARY_TTL:
                out[edge_id] = {"summary": cached[1], "summary_ai": cached[2], "summary_pending": False}
            else:
                out[edge_id] = {"summary": None, "summary_ai": False, "summary_pending": (edge_id, period) in self.summary_jobs}
        return out

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
        if route == "/api/ranking/summaries":
            try:
                body = self.read_body()
                period = str(body.get("period", "week"))
                ids = [str(x) for x in body.get("edge_ids", [])][:20]
                if period not in ("week", "month"):
                    raise ValueError
            except (TypeError, ValueError, json.JSONDecodeError):
                return self.send_json({"error": "expected {period: week|month, edge_ids: [...]}"}, HTTPStatus.BAD_REQUEST)
            with self.lock:
                return self.send_json({"period": period, "summaries": self.ranking_summaries(period, ids)})
        if route.startswith("/api/review/") and route.endswith("/helpful"):
            try:
                user_id = parse_user(self.read_body())
            except (TypeError, ValueError, json.JSONDecodeError):
                user_id = None
            if not user_id:
                return self.send_json({"error": "닉네임으로 로그인한 뒤 누를 수 있어요."}, HTTPStatus.BAD_REQUEST)
            with self.lock:
                card = self.reviews.toggle_helpful(route[len("/api/review/"):-len("/helpful")], user_id)
            if card is None:
                return self.send_json({"error": "그런 리뷰가 없습니다."}, HTTPStatus.NOT_FOUND)
            return self.send_json({"ok": True, "review": card})
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
        if route.startswith("/api/admin/"):
            return self.admin_post(route[len("/api/admin/"):])
        if route == "/api/analyze-photo":
            return self.analyze_photo()
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

    def reference_photos(self, edge_id: str, exclude_user: str | None = None) -> list[tuple[str, str]]:
        """(mime, base64) of photos already filed on this stretch, newest first - the baseline for the place check.

        Photos from the same nickname are skipped: the point is to compare against what others have seen there.
        """
        try:
            edges = parse_edge_id(edge_id)
        except ValueError:
            return []
        wanted = {(min(u, v), max(u, v), k) for u, v, k in edges}
        refs: list[tuple[str, str]] = []
        for review in reversed(self.reviews.reviews):
            if not review.get("photos") or (exclude_user and review.get("user_id") == exclude_user):
                continue
            try:
                theirs = {(min(u, v), max(u, v), k) for u, v, k in parse_edge_id(review["edge_id"])}
            except ValueError:
                continue
            if not (wanted & theirs):
                continue
            for url in review["photos"]:
                path = (PHOTOS_DIR / str(url)[len("/photos/"):]).resolve() if str(url).startswith("/photos/") else None
                if path is None or not path.is_file() or PHOTOS_DIR.resolve() not in path.parents:
                    continue
                mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}.get(path.suffix.lower())
                if not mime:
                    continue
                refs.append((mime, base64.b64encode(path.read_bytes()).decode()))
                if len(refs) >= PLACE_REFS_MAX:
                    return refs
        return refs

    def analyze_photo(self) -> None:
        """POST /api/analyze-photo {photos: [{type, data(base64)}]} -> axes / tags / keywords Claude sees in the photos.

        The page shows the result and asks the user before anything is applied (the answer travels with POST /api/review).
        """
        edge_id, user_id = "", None
        try:
            body = self.read_body()
            photos = [ph for ph in body.get("photos", []) if isinstance(ph, dict)][:PHOTO_MAX]
            edge_id, user_id = str(body.get("edge_id", "")), parse_user(body)
        except (TypeError, ValueError, json.JSONDecodeError):
            photos = []
        sizes = [len(str(ph.get("data", ""))) * 3 // 4 // 1024 for ph in photos]
        print(f"[photo] analyze request: {len(photos)} photo(s), ~{sizes} KB", flush=True)
        if not photos:
            print("[photo] rejected: no photos in body", flush=True)
            return self.send_json({"error": "photos: [{type, data}] is required"}, HTTPStatus.BAD_REQUEST)
        if (reason := self.claude_unavailable()):
            print(f"[photo] rejected: {reason}", flush=True)
            return self.send_json({"error": reason, "api_enabled": False}, HTTPStatus.SERVICE_UNAVAILABLE)
        started = time.time()
        result, status = self.call_claude(analyze_photos, photos)
        if status != HTTPStatus.OK:
            print(f"[photo] analysis failed ({status}): {result.get('error')} {result.get('detail', '')[:120]}", flush=True)
            return self.send_json(result, status)
        print(f"[photo] analysis in {time.time() - started:.1f}s: axes={result['axes']} good={result['good_tags']} bad={result['bad_tags']} keywords={result['keywords']} transient={result['transient_notes']}({result['transient_vibe']}) summary={result['summary']!r}", flush=True)
        # Place check against photos others already filed on this stretch; the first photo of a stretch is trusted.
        place = {"checked": False, "same_place": None, "confidence": 0.0, "reason": "", "suspicious": False}
        if edge_id:
            with self.lock:
                refs = self.reference_photos(edge_id, exclude_user=user_id)
            if refs:
                checked, st2 = self.call_claude(verify_photo_place, photos, refs)
                place = checked if st2 == HTTPStatus.OK else {**place, "reason": "확인하지 못했습니다"}
                print(f"[photo] place check ({len(refs)} ref): same_place={place.get('same_place')} conf={place.get('confidence')} suspicious={place.get('suspicious')} reason={place.get('reason')!r}", flush=True)
            else:
                print("[photo] place check: no reference photo on this stretch yet - trusted", flush=True)
        with self.lock:
            trust = self.profiles.photo_trust(user_id)
        result["place_check"] = place
        result["suspicious"] = bool(place.get("suspicious"))
        result["photo_trust"] = round(trust, 3)
        self.send_json(result, status)

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
            comment = str(body.get("comment", "") or "").strip()[:200]
            route_score = body.get("route_score")
            route_score = int(route_score) if route_score not in (None, "") else None
            if route_score is not None and not 1 <= route_score <= 5:
                raise ValueError
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return self.send_json({"error": "expected {edge_id: 'u-v-k[,u-v-k]' | [..], score: 0..10, good_tags: [..], bad_tags: [..], photos?: [{type, data(base64)}]} (+ optional olat, olon, dlat, dlon, w_* to get routes back)", "good_tags": GOOD_TAGS, "bad_tags": BAD_TAGS}, HTTPStatus.BAD_REQUEST)
        if not edges:
            return self.send_json({"error": "edge_id is empty"}, HTTPStatus.BAD_REQUEST)
        user_id = parse_user(body)
        try:
            photo_urls = save_photos(",".join(f"{u}-{v}-{k}" for u, v, k in edges), photos_in) if photos_in else []
        except (ValueError, OSError) as error:
            return self.send_json({"error": str(error) or "사진을 저장할 수 없습니다."}, HTTPStatus.BAD_REQUEST)
        # Comment analysis (Claude, outside the lock): axis signals, tags the text implies, new keywords.
        # Best effort - a failed call never blocks the review. Not gated by the AI password: it runs server-side.
        ai: dict[str, object] | None = None
        if comment and route_score is None and self.api_enabled and os.environ.get("ANTHROPIC_API_KEY"):
            result, status = self.call_claude(analyze_comment, comment, score)
            if status == HTTPStatus.OK:
                ai = result
                good_tags = list(dict.fromkeys(good_tags + [t for t in ai["good_tags"] if t not in good_tags]))
                bad_tags = list(dict.fromkeys(bad_tags + [t for t in ai["bad_tags"] if t not in bad_tags]))
                ai["auto_good_tags"] = [t for t in ai["good_tags"] if t not in body.get("good_tags", [])]
                ai["auto_bad_tags"] = [t for t in ai["bad_tags"] if t not in body.get("bad_tags", [])]
        # Photo analysis came from POST /api/analyze-photo and was confirmed (or declined) by the user in the form.
        # Re-validated here; accepted tags were already merged into good/bad_tags by the page.
        photo_ai = body.get("photo_ai") if isinstance(body.get("photo_ai"), dict) else None
        if photo_ai is not None:
            photo_ai = {**clean_photo_analysis(photo_ai), "accepted": bool(photo_ai.get("accepted")), "photo_count": len(photo_urls)}
            print(f"[photo] review carries photo_ai: accepted={photo_ai['accepted']} decision={photo_ai['decision']} suspicious={photo_ai['suspicious']} good={photo_ai['good_tags']} bad={photo_ai['bad_tags']} keywords={photo_ai['keywords']}", flush=True)
        elif photos_in and self.api_enabled and os.environ.get("ANTHROPIC_API_KEY"):
            # The page normally analyses photos first (POST /api/analyze-photo) and asks the user. If that did not
            # happen, analyse here so the finding is at least recorded; accepted=False means it changes no scores.
            print(f"[photo] review has {len(photos_in)} photo(s) and no photo_ai - analysing server-side", flush=True)
            result, status = self.call_claude(analyze_photos, photos_in)
            if status == HTTPStatus.OK:
                photo_ai = {**result, "accepted": False, "auto": True, "decision": "auto", "photo_count": len(photo_urls), "place_check": {"checked": False, "same_place": None, "confidence": 0.0, "reason": "", "suspicious": False}}
                print(f"[photo] server-side analysis: good={photo_ai['good_tags']} bad={photo_ai['bad_tags']} keywords={photo_ai['keywords']} (recorded, not applied)", flush=True)
        elif photos_in:
            print(f"[photo] review has {len(photos_in)} photo(s) but no photo_ai and Claude is unavailable", flush=True)
        # Photo trust: a checked photo moves the nickname's score (suspicious costs more than a consistent one gains),
        # and the score at write time is stored on the review so replays apply the same weight.
        if photo_ai is not None and photos_in:
            with self.lock:
                if photo_ai["place_check"]["checked"]:
                    trust = self.profiles.bump_photo_trust(user_id, photo_ai["suspicious"])
                    print(f"[photo] trust for {user_id or '(익명)'}: {'-' if photo_ai['suspicious'] else '+'} -> {trust}", flush=True)
                else:
                    trust = self.profiles.photo_trust(user_id)
            photo_ai["trust"] = round(trust, 3)
        with self.lock:
            self.reviews.personalize(user_id)
            before = self.reviews.stats(edges, user_id)
            tags = parse_tags(body)
            profile = {"purpose": tags[0].split("/")[0] if tags else None, "subs": [t.split("/", 1)[1] for t in tags if "/" in t]}
            meta = {"origin": od[0] if od else None, "destination": od[1] if od else None, "preset_tags": tags, "profile": profile, "photos": photo_urls, "comment": comment or None, **({"ai_photo": photo_ai} if photo_ai else {})}
            if route_score is not None:  # whole-route star: stored separately, 1..5 -> 0..10 for the vibe average
                meta.update({"route_score": route_score, "origin_name": str(body.get("origin_name", ""))[:60] or None, "destination_name": str(body.get("destination_name", ""))[:60] or None})
                origin_node, destination_node = (nearest_node(self.graph, od[0]), nearest_node(self.graph, od[1])) if od else (None, None)
                meta["route_key"] = f"{origin_node}->{destination_node}" if od else None
                updated = self.reviews.add_review(edges, float(route_score * 2), [], [], meta, user_id, kind="route")
            else:
                updated = self.reviews.add_review(edges, score, good_tags, bad_tags, meta, user_id, ai=ai)
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
                "ai": ai,  # axes / auto tags / keywords Claude read from the comment (None without a comment or AI)
                "ai_photo": photo_ai,  # what Claude saw in the photos and whether the user accepted it
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

    def route(self, origin: tuple[float, float], destination: tuple[float, float], profile: Weights, avoid_steps: bool = False, tags: list[str] | None = None, avoid_tags: set[str] | None = None, user_id: str | None = None, waypoints: list[tuple[float, float]] | None = None, revisit_penalty: float = DEFAULT_REVISIT_PENALTY, keep_order: bool = False) -> dict[str, object]:
        origin_node, destination_node = nearest_node(self.graph, origin), nearest_node(self.graph, destination)
        waypoints = waypoints or []
        # Cache key covers everything the result depends on; waypoints (rounded to ~1 cm) and the penalty included,
        # otherwise a request with different waypoints would be served another request's routes.
        key = (
            origin_node, destination_node, profile, avoid_steps, None if avoid_tags is None else frozenset(avoid_tags), user_id,
            tuple((round(lat, 7), round(lon, 7)) for lat, lon in waypoints), revisit_penalty if waypoints else None, keep_order if waypoints else None,
        )
        if self.reviews.refresh_decay():  # a new day: every entry aged, replay with the new weights
            self.route_cache.clear()
        cached = self.route_cache.get(key)
        if cached is not None:
            return {**cached, "tags": tags or [], "cached": True}
        self.reviews.personalize(user_id)  # own reviews weigh more, outliers dropped - per requester
        self.reviews.apply_avoid_tags(avoid_tags)  # graph_no_steps shares the edge dicts, so this covers both
        graph, steps_fallback = self.graph, False
        if avoid_steps:
            graph = self.graph_no_steps
            try:
                shortest, candidates = candidate_routes(graph, origin_node, destination_node, BETAS, profile, waypoints=waypoints, revisit_penalty=revisit_penalty, keep_order=keep_order)
            except nx.NetworkXNoPath:
                graph, steps_fallback = self.graph, True  # only reachable via stairs: fall back and say so
        if graph is self.graph:
            shortest, candidates = candidate_routes(graph, origin_node, destination_node, BETAS, profile, waypoints=waypoints, revisit_penalty=revisit_penalty, keep_order=keep_order)
        collection = candidates_geojson(graph, shortest, candidates)
        for feature in collection["features"]:  # route_id lets the page ask for POIs along this exact line (GET /api/poi)
            route_id = hashlib.sha1(",".join(f"{u}-{v}-{k}" for u, v, k in feature["properties"]["edges"]).encode()).hexdigest()[:12]
            feature["properties"]["route_id"] = route_id
            if route_id not in self.poi_routes and len(self.poi_routes) >= self.POI_ROUTES_MAX:
                self.poi_routes.pop(next(iter(self.poi_routes)))
            self.poi_routes[route_id] = [tuple(c) for c in feature["geometry"]["coordinates"]]
        if waypoints:
            collection["waypoints_input"] = [[lat, lon] for lat, lon in waypoints]
            collection["revisit_penalty"] = revisit_penalty
            collection["keep_order"] = keep_order
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

    def poi(self, query: dict[str, str]) -> None:
        """GET /api/poi?route_id=&categories=cafe,restaurant,tourist -> Kakao places within POI_RADIUS_M of 6 route anchors.

        `coords=lon,lat;lon,lat;...` can replace route_id (e.g. after a server restart forgot the route).
        """
        key = os.environ.get("KAKAO_API_KEY", "").strip()
        if not key:
            return self.send_json({"error": "KAKAO_API_KEY가 설정되지 않았습니다. .env에 키를 넣고 서버를 재시작하세요.", "enabled": False}, HTTPStatus.SERVICE_UNAVAILABLE)
        categories = [c for c in (parse_list(query, "categories") or list(KAKAO_CATEGORIES)) if c in KAKAO_CATEGORIES]
        if not categories:
            return self.send_json({"error": f"categories must be some of {', '.join(KAKAO_CATEGORIES)}"}, HTTPStatus.BAD_REQUEST)
        route_id = str(query.get("route_id", "")).strip()
        coords = self.poi_routes.get(route_id) if route_id else None
        if coords is None and query.get("coords"):
            try:
                coords = [(float(x), float(y)) for x, y in (pair.split(",") for pair in str(query["coords"]).split(";") if pair.strip())]
            except ValueError:
                return self.send_json({"error": "coords must be lon,lat;lon,lat;..."}, HTTPStatus.BAD_REQUEST)
            route_id = route_id or hashlib.sha1(str(query["coords"]).encode()).hexdigest()[:12]
        if not coords:
            return self.send_json({"error": "route_id를 모릅니다. 경로를 다시 검색해 주세요.", "unknown_route": True}, HTTPStatus.NOT_FOUND)
        cache_key = (route_id, tuple(sorted(categories)))
        cached = self.poi_cache.get(cache_key)
        if cached and time.time() - cached[0] < POI_CACHE_TTL:
            return self.send_json({**cached[1], "route_id": route_id, "cached": True})
        try:
            result = route_pois(key, coords, categories)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:200]
            print(f"[kakao] {error.code}: {detail}", flush=True)
            message = ("카카오 앱에서 '카카오맵' 서비스가 꺼져 있습니다. developers.kakao.com › 앱 설정 › 카카오맵에서 활성화하세요." if "OPEN_MAP_AND_LOCAL" in detail
                       else "카카오 API 키가 올바르지 않습니다.") if error.code in (401, 403) else "카카오 API 오류"
            return self.send_json({"error": f"{message} ({error.code})", "detail": detail}, HTTPStatus.BAD_GATEWAY)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            print(f"[kakao] connection failed: {error}", flush=True)
            return self.send_json({"error": "카카오 API에 연결할 수 없습니다."}, HTTPStatus.BAD_GATEWAY)
        if len(self.poi_cache) >= 256:
            self.poi_cache.pop(next(iter(self.poi_cache)))
        self.poi_cache[cache_key] = (time.time(), result)
        self.send_json({**result, "route_id": route_id, "cached": False})

    # ---- admin (/admin, /api/admin/*): ADMIN_PASSWORD in .env, token in X-Admin-Token ----
    def admin_denied(self) -> tuple[dict[str, object], HTTPStatus] | None:
        """None when this request may use the admin API: the page's X-Admin-Token, or Bearer <ADMIN_PASSWORD> for scripts."""
        if self.admin_token_value is None:
            return {"error": "관리자 기능이 꺼져 있습니다. .env에 ADMIN_PASSWORD를 넣고 서버를 재시작하세요."}, HTTPStatus.FORBIDDEN
        if self.headers.get("X-Admin-Token") == self.admin_token_value:
            return None
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Bearer ") and admin_token(auth[len("Bearer "):].strip()) == self.admin_token_value:
            return None
        return {"error": "관리자 로그인이 필요합니다."}, HTTPStatus.UNAUTHORIZED

    def admin_get(self, action: str, query: dict[str, str]) -> None:
        if action == "status":
            return self.send_json({"enabled": self.admin_token_value is not None, "logged_in": self.admin_token_value is not None and self.headers.get("X-Admin-Token") == self.admin_token_value, "api_enabled": self.api_enabled and bool(os.environ.get("ANTHROPIC_API_KEY"))})
        if (denied := self.admin_denied()):
            return self.send_json(*denied)
        with self.lock:
            if action == "reviews":
                limit = min(int(query.get("limit", 200) or 200), 1000)
                entries = [{**r, "type": "review"} for r in self.reviews.reviews] + [{**n, "type": "report"} for n in self.reviews.negatives]
                entries.sort(key=lambda e: str(e.get("ts", "")), reverse=True)
                now = time.time()
                for e in entries:
                    e["segment"] = self.segment_label(e["edge_id"])
                    stamp = parse_ts(e.get("ts"))
                    e["age_days"] = round((now - stamp) / 86400, 1) if stamp else None
                    e["decay"] = round(decay(e.get("ts"), now), 3)
                    # The author's photo trust as it stands now (the review keeps the score it was written with).
                    profile = self.profiles.get(str(e.get("user_id") or "")) or {}
                    e["user_photo_trust"] = profile.get("photo_trust_score")
                    e["user_photo_checks"] = profile.get("photo_checks")
                    e["user_photo_flags"] = profile.get("photo_flags")
                return self.send_json({
                    "entries": entries[:limit], "total": len(entries),
                    "photo_trust": {"default": PHOTO_TRUST_DEFAULT, "penalty": PHOTO_TRUST_PENALTY, "recover": PHOTO_TRUST_RECOVER, "min_apply": PHOTO_TRUST_MIN_APPLY},
                })
            if action == "candidates":
                rows = sorted(self.reviews.candidates.values(), key=lambda c: (c.get("status") != "pending", -int(c.get("count", 0))))
                return self.send_json({"candidates": rows, "axes": list(AXIS_STEPS)})
            if action == "danger":
                zones = self.danger_zones((37.5720, 126.9769), 1e9)  # whole graph
                overrides = [{"edge_id": k, **v, "segment": self.segment_label(k)} for k, v in self.reviews.danger_manual.items()]
                return self.send_json({"zones": zones, "overrides": overrides, "threshold": DANGER_THRESHOLD})
            if action == "stats":
                return self.send_json(self.admin_stats())
        self.send_json({"error": "unknown admin action"}, HTTPStatus.NOT_FOUND)

    def admin_post(self, action: str) -> None:
        try:
            body = self.read_body()
        except (TypeError, ValueError, json.JSONDecodeError):
            body = {}
        if action == "login":
            if self.admin_token_value is None:
                return self.send_json({"error": "관리자 기능이 꺼져 있습니다. .env에 ADMIN_PASSWORD를 넣고 서버를 재시작하세요."}, HTTPStatus.FORBIDDEN)
            if admin_token(str(body.get("password", ""))) != self.admin_token_value:
                return self.send_json({"error": "비밀번호가 맞지 않습니다."}, HTTPStatus.UNAUTHORIZED)
            return self.send_json({"ok": True, "token": self.admin_token_value})
        if (denied := self.admin_denied()):
            return self.send_json(*denied)
        with self.lock:
            if action.startswith("review/") and action.endswith("/delete"):
                removed = self.reviews.delete(action[len("review/"):-len("/delete")])
                if removed is None:
                    return self.send_json({"error": "그런 리뷰가 없습니다."}, HTTPStatus.NOT_FOUND)
                return self.send_json({"ok": True, "removed": removed, "reviews_total": len(self.reviews.reviews), "negatives_total": len(self.reviews.negatives)})
            if action == "candidates/split":
                result = self.reviews.split_candidate(str(body.get("keyword", "")))
                if result is None:
                    return self.send_json({"error": "그 단어는 어느 후보의 별칭도 아닙니다."}, HTTPStatus.BAD_REQUEST)
                return self.send_json({"ok": True, "candidate": result[0], "parent": result[1]})
            if action == "candidates/merge":
                source, target = str(body.get("source", "")), str(body.get("target", ""))
                merged = self.reviews.merge_candidates(source, target)
                if merged is None:
                    return self.send_json({"error": "합칠 두 키워드를 확인하세요 (같은 키워드이거나 없는 키워드)."}, HTTPStatus.BAD_REQUEST)
                return self.send_json({"ok": True, "candidate": merged})
            if action.startswith("candidates/"):
                keyword, status = unquote(action[len("candidates/"):]), str(body.get("status", ""))
                if status not in ("approved", "rejected", "pending"):
                    return self.send_json({"error": "status must be approved | rejected | pending"}, HTTPStatus.BAD_REQUEST)
                entry = self.reviews.set_candidate(keyword, status)
                if entry is None:
                    return self.send_json({"error": "그런 키워드가 없습니다."}, HTTPStatus.NOT_FOUND)
                return self.send_json({"ok": True, "candidate": entry, "guide": axis_guide(keyword) if status == "approved" else None})
            if action == "danger":
                try:
                    edges = parse_edge_id(body["edge_id"])
                    mode = str(body.get("mode", ""))
                    if mode not in ("force", "clear", "auto") or not edges:
                        raise ValueError
                    missing = [e for e in edges if not self.graph.has_edge(*e) and not self.graph.has_edge(e[1], e[0], e[2])]
                    if missing:
                        return self.send_json({"error": f"그래프에 없는 엣지: {', '.join('-'.join(map(str, e)) for e in missing)}"}, HTTPStatus.BAD_REQUEST)
                except (KeyError, TypeError, ValueError):
                    return self.send_json({"error": "expected {edge_id: 'u-v-k[,..]', mode: force | clear | auto, note?}"}, HTTPStatus.BAD_REQUEST)
                touched = self.reviews.set_danger(edges, {"force": True, "clear": False, "auto": None}[mode], str(body.get("note", "")))
                return self.send_json({"ok": True, "mode": mode, "edges_updated": len(touched), "danger": any(is_danger(self.graph[u][v][k]) for u, v, k in touched)})
            if action == "data/generate":
                from generate_test_reviews import generate  # imported lazily: the script also has CLI-only code
                import random
                count = min(max(int(body.get("count", 25) or 25), 1), 60)
                reviews, _ = generate(self.graph, random.Random(body.get("seed")), count)
                for r in reviews:
                    r["id"] = entry_id(r)
                if not body.get("append", True):
                    self.reviews.reviews, self.reviews.negatives = [], []
                self.reviews.reviews.extend(reviews)
                self.reviews.rebuild()
                self.reviews.save()
                return self.send_json({"ok": True, "generated": len(reviews), "reviews_total": len(self.reviews.reviews)})
            if action in ("import-data", "data/import"):
                return self.send_json(*self.import_data(body))
            if action == "data/reset":
                archived = self.reviews.clear()
                return self.send_json({"ok": True, "archived": str(archived) if archived else None})
        self.send_json({"error": "unknown admin action"}, HTTPStatus.NOT_FOUND)

    def segment_summary_for(self, edge_id: str) -> dict[str, object]:
        """Segment name / length / geometry for one stored edge_id (my-reviews cards and their "지도에서 보기")."""
        try:
            edges = [e for e in parse_edge_id(edge_id) if self.graph.has_edge(*e)]
        except ValueError:
            edges = []
        if not edges:
            return {"edge_id": edge_id, "segment_name": "알 수 없는 구간", "length_m": 0, "coordinates": []}
        names = Counter(n for n in (edge_name(self.graph[u][v][k]) for u, v, k in edges) if n)
        highway = Counter(h for u, v, k in edges for h in highway_values(self.graph[u][v][k].get("highway"))).most_common(1)
        return {
            "edge_id": edge_id,
            "segment_name": names.most_common(1)[0][0] if names else RANK_HIGHWAY_NAMES.get(highway[0][0] if highway else "", "이름 없는 길"),
            "length_m": round(sum(self.graph[u][v][k]["length"] for u, v, k in edges), 1),
            "coordinates": [[round(lon, 6), round(lat, 6)] for lon, lat in route_coordinates(self.graph, edges)],
        }

    def import_data(self, body: dict[str, object]) -> tuple[dict[str, object], HTTPStatus]:
        """POST /api/admin/import-data: load reviews / profiles / visits (and optionally photo files) from a local copy.

        body: {"reviews": [...] | {"reviews": [...], "negatives": [...], ...},
               "profiles": {nickname: {...}}, "visits": {date: {...}} | [{"date": ..., ...}],
               "photos": {"<dir>/<file>.jpg": "<base64>"},        # optional, so imported reviews keep their images
               "mode": "replace" | "merge"}                        # reviews only; default replace
        Files under DATA_DIR are written and the running server is refreshed, so no restart is needed.
        """
        mode = str(body.get("mode", "replace"))
        if mode not in ("replace", "merge"):
            return {"error": "mode must be replace or merge"}, HTTPStatus.BAD_REQUEST
        result: dict[str, object] = {"ok": True, "mode": mode}
        store = self.reviews
        payload = body.get("reviews")
        if payload is not None:
            data = payload if isinstance(payload, dict) else {"reviews": payload}
            if not isinstance(data.get("reviews", []), list):
                return {"error": "reviews must be a list, or the whole reviews.json object"}, HTTPStatus.BAD_REQUEST
            incoming = [dict(r, id=r.get("id") or entry_id(r)) for r in data.get("reviews", [])]
            negatives = [dict(n, id=n.get("id") or entry_id(n)) for n in data.get("negatives", [])]
            if mode == "merge":
                known = {r.get("id") for r in store.reviews}
                incoming = store.reviews + [r for r in incoming if r["id"] not in known]
                known_n = {n.get("id") for n in store.negatives}
                negatives = store.negatives + [n for n in negatives if n["id"] not in known_n]
                store.candidates = {**store.candidates, **{str(c["keyword"]): c for c in data.get("new_axis_candidates", []) if c.get("keyword")}}
                store.danger_manual = {**store.danger_manual, **{str(k): v for k, v in (data.get("danger_manual") or {}).items()}}
            else:
                store.candidates = {str(c["keyword"]): c for c in data.get("new_axis_candidates", []) if c.get("keyword")}
                store.danger_manual = {str(k): v for k, v in (data.get("danger_manual") or {}).items()}
            store.reviews, store.negatives = incoming, negatives
            store.rebuild()  # replay onto a clean graph so scores match the imported history
            store.save()
            result["reviews"] = len(store.reviews)
            result["negatives"] = len(store.negatives)
            result["candidates"] = len(store.candidates)
        profiles = body.get("profiles")
        if isinstance(profiles, dict):
            self.profiles.profiles = {**self.profiles.profiles, **profiles} if mode == "merge" else dict(profiles)
            self.profiles.path.parent.mkdir(parents=True, exist_ok=True)
            self.profiles.path.write_text(json.dumps(self.profiles.profiles, ensure_ascii=False, indent=1), encoding="utf-8")
            result["profiles"] = len(self.profiles.profiles)
        visits = body.get("visits")
        if visits is not None:
            days = visits if isinstance(visits, dict) else {str(v.get("date")): {k: v[k] for k in ("page", "routes", "users") if k in v} for v in visits if isinstance(v, dict) and v.get("date")}
            self.visits.days = {**self.visits.days, **days} if mode == "merge" else dict(days)
            self.visits.dirty = True
            self.visits.flush()
            result["visits"] = len(self.visits.days)
        photos = body.get("photos")
        if isinstance(photos, dict):
            saved = 0
            for relative, blob in photos.items():
                target = (PHOTOS_DIR / str(relative)).resolve()
                if PHOTOS_DIR.resolve() not in target.parents or target.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                    continue  # never write outside the photo directory
                target.parent.mkdir(parents=True, exist_ok=True)
                raw = str(blob)
                target.write_bytes(base64.b64decode(raw.split(",", 1)[1] if raw.startswith("data:") else raw, validate=False))
                saved += 1
            result["photos"] = saved
        RouteHandler.route_cache.clear()
        RouteHandler.ranking_cache.clear()
        print(f"[import] {result}", flush=True)
        return result, HTTPStatus.OK

    def segment_label(self, edge_id: str) -> str:
        """'세종대로 · 120 m' for an edge_id, for the admin tables."""
        try:
            edges = [e for e in parse_edge_id(edge_id) if self.graph.has_edge(*e)]
        except ValueError:
            return edge_id
        if not edges:
            return edge_id[:40]
        names = Counter(n for n in (edge_name(self.graph[u][v][k]) for u, v, k in edges) if n)
        length = sum(self.graph[u][v][k]["length"] for u, v, k in edges)
        return f"{names.most_common(1)[0][0] if names else '이름 없는 길'} · {length:,.0f} m · {len(edges)}엣지"

    def admin_stats(self) -> dict[str, object]:
        reviews = [r for r in self.reviews.reviews if r.get("kind", "segment") == "segment"]
        by_user: Counter[str] = Counter(str(r.get("user_id") or "(익명)") for r in self.reviews.reviews)
        by_user.update(str(n.get("user_id") or "(익명)") for n in self.reviews.negatives)
        by_purpose: dict[str, list[float]] = {}
        for r in reviews:
            purpose = (r.get("profile") or {}).get("purpose") or "(없음)"
            by_purpose.setdefault(purpose, []).append(float(r["score"]))
        tags: Counter[str] = Counter(t for r in reviews for t in r.get("good_tags", []))
        bad: Counter[str] = Counter(t for r in reviews for t in r.get("bad_tags", []))
        bad.update(str(n["reason"]) for n in self.reviews.negatives)
        return {
            "decay_half_life_days": DECAY_HALF_LIFE_DAYS,
            "totals": {"reviews": len(self.reviews.reviews), "segment_reviews": len(reviews), "route_ratings": len(self.reviews.reviews) - len(reviews), "reports": len(self.reviews.negatives),
                       "users": len(by_user), "candidates_pending": sum(1 for c in self.reviews.candidates.values() if c.get("status") == "pending"),
                       "danger_zones": len(self.danger_zones((37.5720, 126.9769), 1e9)), "ai_analyzed": sum(1 for r in reviews if r.get("ai"))},
            "by_user": [{"user_id": u, "count": n, **({k: v for k, v in ((f, (self.profiles.get(u) or {}).get(f)) for f in ("photo_trust_score", "photo_checks", "photo_flags")) if v is not None})} for u, n in by_user.most_common(50)],
            "photo_trust": {"default": PHOTO_TRUST_DEFAULT, "penalty": PHOTO_TRUST_PENALTY, "recover": PHOTO_TRUST_RECOVER, "min_apply": PHOTO_TRUST_MIN_APPLY},
            "by_purpose": [{"purpose": k, "name": PURPOSE_NAMES.get(k, k), "avg": round(sum(v) / len(v), 2), "count": len(v)} for k, v in sorted(by_purpose.items(), key=lambda kv: -len(kv[1]))],
            "good_tags": dict(tags.most_common()), "bad_tags": dict(bad.most_common()),
            "visits": self.visits.rows(30),
        }

    def send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self) -> None:
        # Static pages change on every edit; without this the browser can keep serving an old index.html from cache.
        if self.path.split("?")[0].endswith((".html", ".js", "/")) or self.path in ("/ranking", "/review", "/admin"):
            self.send_header("Cache-Control", "no-cache, must-revalidate")
        super().end_headers()

    def log_message(self, format: str, *args: object) -> None:
        if args and "/api/" in str(args[0]):
            super().log_message(format, *args)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Walk-route API + static frontend.")
    parser.add_argument("port", type=int, nargs="?", default=PORT)
    parser.add_argument("--host", default=HOST, help="bind address (0.0.0.0 to accept outside connections)")
    parser.add_argument("--reset", action="store_true", help="archive reviews.json so vibe_score and penalties start from the OSM baseline")
    parser.add_argument("--no-api", action="store_true", help="disable the Claude endpoints (/api/interpret, /api/explain) for demos")
    args = parser.parse_args()

    load_env()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RouteHandler.api_enabled = not args.no_api
    ai_password = os.environ.get("AI_PASSWORD", "").strip()
    RouteHandler.ai_token_value = ai_token(ai_password) if ai_password else None
    admin_password = os.environ.get("ADMIN_PASSWORD", "").strip()
    RouteHandler.admin_token_value = admin_token(admin_password) if admin_password else None
    print("Admin page (/admin): " + ("enabled" if admin_password else "disabled (ADMIN_PASSWORD not set in .env)"))
    print("Kakao POI (/api/poi): " + ("ready" if os.environ.get("KAKAO_API_KEY") else "disabled (KAKAO_API_KEY not set in .env)"))
    print("Map tiles: " + ("VWorld (VWORLD_API_KEY set)" if os.environ.get("VWORLD_API_KEY", "").strip() else "Carto Voyager (VWORLD_API_KEY not set in .env)"))
    print("Claude endpoints: " + ("disabled (--no-api)" if args.no_api else "ready" if os.environ.get("ANTHROPIC_API_KEY") else "disabled (ANTHROPIC_API_KEY not set in .env)"))
    print("AI password: " + ("required to unlock AI" if ai_password else "not set (AI unlock is free)"))
    print("Loading graph...", flush=True)
    RouteHandler.graph = load_graph()
    RouteHandler.graph_no_steps = without_steps(RouteHandler.graph)
    print(f"Stair edges excluded for avoid_steps: {RouteHandler.graph.number_of_edges() - RouteHandler.graph_no_steps.number_of_edges():,}")
    RouteHandler.reviews = ReviewStore(RouteHandler.graph)
    RouteHandler.profiles = ProfileStore()
    RouteHandler.visits = VisitLog()
    print(f"Profiles: {len(RouteHandler.profiles.profiles)} nickname(s) in {PROFILES_PATH}")
    if args.reset:
        archived = RouteHandler.reviews.reset()
        print("Reset: vibe_score back to OSM baseline" + (f" (reviews archived to {archived})" if archived else " (no reviews to archive)"))
    else:
        print(f"Replayed {RouteHandler.reviews.load()} review/feedback entries from {REVIEWS_PATH}")
    handler = partial(RouteHandler, directory=str(WEB_DIR))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Data directory: {DATA_DIR.resolve()}")
    print(f"Serving on http://{args.host}:{args.port}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
