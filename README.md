# 에움길 (walk-route)

**A walking-route recommender for central Seoul that trades a small, bounded detour for a calmer, greener, more pleasant walk, with an LLM that turns a plain-language request into routing preferences.**

Built by a team of four at the Wanted AI Championship 2026 hackathon (Top 20).

## Key features

- **Three routes per request: Shortest / Balanced / Pleasant.** Each one is the most pleasant route within a fixed detour budget (1.0×, 1.15× and 1.3× the shortest distance).
- **Four preference axes**: avoid car traffic, street trees, closeness to parks, and street "vibe" (density of cafés, bakeries, bookshops, galleries). They are computed per street segment from OpenStreetMap and Seoul street-tree data.
- **Natural-language input (Claude).** A request such as "a quiet route with cafés, no stairs, from Gyeongbokgung to Myeongdong" becomes axis weights, avoid-options, and origin/destination/waypoint names. Claude also writes a one-line reason for each recommended route.
- **Community feedback changes routing.** Segment reviews (score, tags, text, photos) adjust that segment's scores, with LLM analysis of the text and photos. Repeatedly reported unsafe segments are avoided for everyone, and reviews lose weight over time.
- **Waypoints (up to 6)** with an exact visiting order and a penalty for walking the same street twice, plus turn-by-turn directions and nearby places along the route.

Coverage: a 2 km radius around Gwanghwamun, Seoul.

## How it works

```
 OpenStreetMap (osmnx)          Seoul street-tree CSV
 walk network + POIs/parks      (tree locations)
            │                          │
            ▼                          ▼
 score_gwanghwamun_edges.py, scripts/rescore_*.py      (offline, once)
 per-edge attributes: vehicle exposure, tree count, park distance,
 vibe score, stairs
            │
            ▼
 osmnx_outputs/gwanghwamun_walk_2km_scored.graphml     (loaded by the server)
            │
            ▼
 per request (server.py)
  1. Preferences: UI preset, or free text → Claude → axis weights wᵢ
  2. Edge cost = length × (1 + review/report penalty)
               + λ · Σ wᵢ · penaltyᵢ        (penalty meters per axis)
  3. Candidates (candidate_routes.py): for each detour budget β,
     search λ for the lowest-penalty route with length ≤ β × shortest
     (Dijkstra; waypoints.py adds visit ordering + revisit penalty)
  4. Response: GeoJSON routes, per-axis stats, segments, turn-by-turn;
     optional Claude explanation and nearby places
            ▲
            │ reviews / reports / photos (Claude analyses text and photos)
            └── adjust per-edge scores and penalties
```

The segment scores are rule-based, computed from map data; no machine-learned scoring model is trained. The LLM is used for input interpretation, explanations, and analysis of reviews and photos. Every Claude feature is optional, and routing works without an API key.

## Project structure

```
.
├── server.py                 # HTTP server: REST API, static frontend, review store, Claude calls, admin API
├── routing.py                # graph loading, edge cost model, Dijkstra, GeoJSON + turn-by-turn output
├── candidate_routes.py       # detour-budget candidate search (λ search per β)
├── waypoints.py              # waypoint snapping, exact visit order (≤ 6), revisit penalty
├── score_gwanghwamun_edges.py# builds per-edge scores from OSM (also imported by the server)
├── generate_test_reviews.py  # synthetic reviews for demos (also used by the admin page)
├── web/
│   ├── index.html            # single-page map app (Leaflet)
│   ├── admin.html            # admin dashboard
│   └── about.html            # service introduction page
├── scripts/                  # offline data preparation, analysis and validation
│   ├── download_walk_networks.py, rebuild_graphml_comparison.py   # fetch / compare OSM walk networks
│   ├── rescore_trees_seoul.py, rescore_vibe_osm.py                # re-score tree and vibe axes
│   ├── compare_gwanghwamun_routes.py, sensitivity_gwanghwamun_vehicle_weight.py
│   ├── map_median_w4_route_pair.py, demo_gyeongbokgung_myeongdong.py   # analysis maps
│   └── validate_candidate_routes.py, validate_waypoints.py        # checks for the routing logic
├── osmnx_outputs/            # scored graph used by the server + analysis outputs (CSV / HTML maps)
├── requirements-server.txt   # runtime dependencies (server only)
├── requirements.txt          # dependencies for the scripts
├── railway.toml              # Railway deployment config
└── DEPLOY.md                 # deployment guide (Korean)
```

## Running locally

Tested with Python 3.12.

```bash
pip install -r requirements-server.txt
python server.py              # → http://127.0.0.1:8000
```

The scored graph (`osmnx_outputs/gwanghwamun_walk_2km_scored.graphml`) is included in the repository, so no data download is needed to run the server.

Optional features read keys from a `.env` file in the repository root, or from environment variables:

| Variable | Enables |
|---|---|
| `ANTHROPIC_API_KEY` | natural-language input, route explanations, review/photo analysis |
| `KAKAO_API_KEY` | nearby places along a route |
| `VWORLD_API_KEY` | Korean map tiles (Carto tiles are used otherwise) |
| `ADMIN_PASSWORD` | the `/admin` page |

`python server.py --no-api` turns the Claude features off.

**Scripts.** Install `requirements.txt` and run the scripts as modules from the repository root, because they import the root modules:

```bash
pip install -r requirements.txt
python -m scripts.validate_candidate_routes
```

The street-tree script expects the Seoul street-tree CSV under `data/`, which is not included in the repository.

**Deployment.** The service is deployed on Railway. See [DEPLOY.md](DEPLOY.md).

## Team and my role

This is a four-person team project from the Wanted AI Championship 2026 hackathon.

**My role:** [내가 채울 부분]

---

### 한국어 요약

에움길은 광화문 반경 2km 안에서 쓰는 도보 산책 경로 추천 서비스입니다. OpenStreetMap과 서울시 가로수 데이터로 길마다 차량 노출, 가로수, 공원 거리, 분위기 점수를 미리 계산해 둡니다. 요청이 들어오면 최단 거리의 1.0/1.15/1.3배 안에서 사용자 취향에 가장 잘 맞는 경로를 하나씩, 모두 세 개 추천합니다. 사용자가 자연어로 원하는 길을 말하면 Claude가 이를 경로 탐색 가중치로 바꾸고, 추천 이유도 설명합니다. 사용자가 남긴 리뷰, 신고, 사진은 해당 구간의 점수에 반영되어 이후 경로 추천이 달라집니다.
