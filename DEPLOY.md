# Railway 배포 가이드 (에움길 / walk-route)

로컬에서 검증한 설정입니다. `railway.toml`, `requirements-server.txt`가 저장소에 있고,
서버는 `HOST` / `PORT` / `DATA_DIR` 환경변수를 읽도록 수정돼 있습니다.

---

## 0. 배포 전 체크리스트

| 항목 | 상태 | 비고 |
|---|---|---|
| `railway.toml` | ✅ 생성됨 | 빌드·시작 명령, 헬스체크, 볼륨 경로 |
| `requirements-server.txt` | ✅ 생성됨 | 런타임 전용(osmnx·networkx·shapely·anthropic) |
| `0.0.0.0` 바인딩 · `$PORT` | ✅ 반영됨 | `python server.py --host 0.0.0.0` |
| 그래프 파일 | ⚠️ **커밋 필요** | `osmnx_outputs/gwanghwamun_walk_2km_scored.graphml` (15.8 MB) |
| 데이터 영속성 | ⚠️ **볼륨 필요** | 없으면 재배포마다 리뷰·사진 초기화 |
| API 키 | ⚠️ **변수 등록 필요** | `.env`는 커밋되지 않음 |

---

## 1. 그래프 파일 커밋 (필수)

서버는 시작할 때 `osmnx_outputs/gwanghwamun_walk_2km_scored.graphml`을 읽습니다.
원래 `.gitignore`의 `*.graphml`에 걸려 있어서, 이 파일만 예외로 풀어 두었습니다.

```bash
git add osmnx_outputs/gwanghwamun_walk_2km_scored.graphml   # 15.8 MB
git add railway.toml requirements-server.txt DEPLOY.md .gitignore server.py routing.py
git commit -m "Railway 배포 설정"
git push
```

> 이 파일이 없으면 배포는 성공해도 시작 직후 `FileNotFoundError`로 죽습니다.
> 저장소에 큰 파일을 넣기 싫다면 Git LFS를 쓰거나, 빌드 단계에서 외부 스토리지에서 받아
> `WALKROUTE_GRAPH` 환경변수로 경로를 지정하세요.

## 2. Railway 프로젝트 생성

**웹 콘솔**
1. https://railway.app → New Project → **Deploy from GitHub repo** → 이 저장소 선택
2. `railway.toml`을 자동으로 읽어 Nixpacks로 빌드합니다 (첫 빌드 3~5분, geopandas·pyproj 휠 설치)

**CLI (선택)**
```bash
npm i -g @railway/cli
railway login
railway link          # 기존 프로젝트에 연결, 또는 railway init
railway up            # 현재 폴더를 그대로 업로드해 배포
```

## 3. 볼륨 연결 (리뷰·사진 보존)

Railway 컨테이너의 파일시스템은 재배포하면 초기화됩니다. 리뷰·프로필·사진을 유지하려면:

1. 서비스 → **Variables 옆 Volumes** → New Volume
2. Mount path: **`/data`** (railway.toml의 `DATA_DIR`과 같아야 함)
3. 크기는 1 GB면 충분합니다 (사진 위주)

볼륨을 안 붙이면 `/data`가 컨테이너 임시 디렉터리가 되어 배포마다 리뷰가 사라집니다.

## 4. 환경변수 등록

서비스 → **Variables** 에 추가 (`.env`는 저장소에 없으므로 여기서 넣어야 합니다):

| 변수 | 필수 | 용도 |
|---|---|---|
| `ANTHROPIC_API_KEY` | 권장 | 취향 해석·추천 이유·리뷰/사진 AI 분석. 없으면 해당 기능만 꺼짐 |
| `ADMIN_PASSWORD` | 권장 | `/admin` 로그인. 없으면 관리자 페이지 비활성 |
| `KAKAO_API_KEY` | 선택 | 주변 장소(POI). 카카오 개발자 콘솔의 **REST API 키** |
| `VWORLD_API_KEY` | 선택 | 한글 지도 타일. **VWorld 콘솔에 Railway 도메인 등록 필요** |
| `AI_PASSWORD` | 선택 | AI 기능 잠금(데모용) |
| `HOST`, `DATA_DIR`, `PORT` | 자동 | railway.toml / Railway가 설정 |

## 5. 배포 확인

```bash
railway domain                      # 도메인 발급 (또는 콘솔 Settings → Networking → Generate Domain)
curl https://<도메인>/api/status     # {"api_enabled":true,...,"vworld_key":...}
```

- `/` 서비스 화면, `/admin` 관리자, `/ranking` 랭킹
- 첫 요청은 그래프 로딩(20~40초) 때문에 느립니다 → `healthcheckTimeout = 300`으로 잡아 두었습니다
- 로그: `railway logs` — `Loading graph...` → `Serving on http://0.0.0.0:<PORT>` 가 보이면 정상

## 6. 배포 후 할 일

- **VWorld**: 콘솔에서 인증키의 서비스 URL에 `https://<도메인>` 추가 (안 하면 타일이 안 보이고 Carto로 폴백)
- **더미 데이터**(데모용): 볼륨이 비어 있으므로
  ```bash
  railway run python generate_test_reviews.py --count 25 --danger 6
  ```
  또는 `/admin → 데이터 → 더미 생성` (서버 재시작 없이 반영)
- **관리자 비밀번호**를 충분히 길게 바꾸세요 (외부에 공개되는 주소입니다)

---

## 주의 / 알려진 제약

- **단일 인스턴스 전용** (`numReplicas = 1`): 그래프는 메모리에, 리뷰는 파일에 있어 복제본끼리 상태가 갈립니다.
- **메모리**: 그래프 로드 후 300~500 MB를 씁니다. Railway 기본 컨테이너에서 동작하지만, OOM이 나면 플랜의 메모리를 올리세요.
- **콜드 스타트**: 슬립 후 첫 요청은 그래프 로딩 때문에 30초 안팎 걸립니다.
- **빌드 시간**: osmnx가 geopandas/pyproj를 끌어와 첫 빌드가 3~5분입니다. 이후 캐시로 빨라집니다.
- **사진 용량**: 리뷰 사진은 볼륨에 쌓입니다(장당 최대 5 MB, 리뷰당 3장).
