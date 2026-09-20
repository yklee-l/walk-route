# 에움길 디자인 수정 가이드 (디자인 팀원용)

**대상 파일: `web/index.html` 하나만 수정합니다.**
이 파일은 위에서부터 `[디자인 가이드 주석] → [CSS: <style> 블록] → [HTML 마크업] → [JavaScript: <script> 블록]` 순서입니다.
디자인 작업은 **`<style>` 블록 안**에서만 하면 되고, 파일 안에 경계 주석이 있습니다:

```
  /* ============ 디자인 수정 가능 영역 (여기부터 style 블록 끝까지) ============ */   ← 여기부터
  ...
  /* ============ 디자인 수정 가능 영역 끝 ============ */                              ← 여기까지
  ...
<!-- ============ 여기부터 파일 끝까지 수정 금지 (JavaScript) ============ -->        ← 이 아래는 손대지 않기
```

---

## 1. 수정 가능한 것

### CSS 변수 (`:root`) — 여기만 바꿔도 전체 테마가 바뀝니다
```css
:root {
  --bg --panel --text --muted --faint      /* 배경·패널·글자색 */
  --border --track                         /* 테두리·트랙(칩 배경) */
  --accent --accent-ink                    /* 강조색과 그 위 글자색 */
  --good --bad --origin --dest             /* 긍정/부정, 출발(초록)/도착(빨강) */
  --radius --shadow                        /* 둥글기, 그림자 */
  --sheet-w --half-h                       /* 데스크톱 패널 폭, 폰 '반' 시트 높이 (주의 참고) */
}
```

### 레이아웃·스타일
- 폰트: `font-family / font-size / font-weight / line-height`
- 여백: `margin / padding / gap`
- 배경, 테두리, 그림자, 둥글기
- **경로 카드** `.card` (compact) / `.card.expanded .xp` (펼친 부분) / `.card .brief span` (특징 칩)
- **랭킹 카드** `.rkcard`, **리뷰 카드** `.rc`, **리뷰 팝업** `.rv`
- **바텀 시트** `#sheet`, `.grab`(핸들), `#odsummary`(출발→도착 요약 칩)
- **상단 바** `#topbar`, `#ask`(AI 입력창), `#menubtn`(≡), `#avatar`
- **오버레이** `#onboard .ob`(온보딩), `#intro .slide`(소개 슬라이드), `#ranking .rk`, `#rvpage / #myrv / #rvform`(전체화면 페이지), `#sidemenu .panel`(햄버거 메뉴), `#acctmenu`
- **마커** `.marker-pin .wp-pin .danger-pin .poi-pin .me-dot`, `#toast`
- 버튼: `.primary .ghost .pick .open .poi .help .segbtn` 등

### 애니메이션·전환
- `@keyframes rise / stepIn / introPop / introPulse / pendingPulse / blink`
- `transition` 값 (단, 시트 `#sheet`는 `transform`만 전환 — 아래 주의 참고)

### 아이콘·이모지
- **HTML 안의 이모지·텍스트**(버튼 라벨, 소개 슬라이드 이모지 등)는 바꿔도 됩니다.
- 단, 같은 문구가 **JS에도 중복**된 것이 많습니다(예: 취향 프리셋 이름, 태그 이모지, 마커 HTML). JS 쪽은 건드리지 말고 **"이 문구를 ○○로 바꿔 달라"고 개발자에게 요청**하세요.
- 이모지는 폰트/OS마다 다르게 보입니다. SVG 아이콘으로 통일하려면 개발자와 상의(마커는 JS에서 크기를 계산함).

---

## 2. 절대 건드리면 안 되는 것

| 대상 | 이유 |
|---|---|
| **JavaScript 함수 전체** (`<script>` ~ `</script>`) | 경로 탐색·지도·리뷰·랭킹 로직. 한 글자만 틀려도 페이지 전체가 멈춥니다 |
| **API 호출 코드** (`fetch(`, `api(`, `/api/...`) | 서버와 약속된 형식. 바꾸면 데이터가 안 옵니다 |
| **이벤트 핸들러** (`addEventListener`, `onclick`, `map.on(`) | 버튼·지도 탭 동작 |
| **HTML `id="…"` 속성** | JS가 `document.getElementById`로 찾습니다. 이름을 바꾸면 그 기능이 죽습니다 |
| **HTML `class` 이름** | JS가 상태를 켜고 끄는 클래스: `.show .on .open .peek .half .active .selected .focused .dimmed .hot .busy .loading .warn .off .manual .drag .dragging .over .sel .hover .back .blink .expanded .armed .ready, body.detail` → **스타일은 바꿔도 되지만 이름은 유지** |
| **HTML 구조** (요소 추가·삭제·순서·태그 종류) | JS가 `querySelector`로 구조를 전제합니다 |
| **Leaflet 지도 관련** (`L.map`, `L.polyline`, `L.marker`, `.leaflet-*` 클래스) | 지도 라이브러리 내부. `.leaflet-*`는 줌 버튼 위치 정도만 조정 |
| **JS 상수** `COLORS`(경로 색 3개), `LINE`(선 굵기), `REPORT_STYLE`, `DANGER_STYLE`, `scoreColor()` | 경로 선 색·굵기·점수 색은 **CSS가 아니라 JS**에 있습니다. 바꾸고 싶은 값을 개발자에게 전달 |
| `web/admin.html`, `server.py`, `routing.py`, `candidate_routes.py`, `waypoints.py`, `osmnx_outputs/*.json` | 서버·데이터 파일 |

### 특히 조심할 함정
1. **`hidden` 속성으로 숨기는 요소**(`#wpmode #wpfoot #wpmodehint #gps #routes #introprev …`)에 `display: flex/block`을 새로 주면 `hidden`이 무시되어 항상 보입니다. 필요하면 `#id[hidden] { display: none }`을 같이 두세요.
2. **`<style>` 블록 안에 `</style>` 글자를 쓰지 마세요** (주석이라도). 브라우저가 거기서 스타일을 끝내 버려 CSS가 화면에 텍스트로 찍힙니다. (실제로 한 번 발생했음)
3. **반응형 기준**: CSS `@media (max-width: 760px)` / `(min-width: 761px)`, 햄버거 메뉴는 `768px`. JS의 `isPhone()`(`innerWidth <= 760`)과 짝이므로 브레이크포인트를 옮기려면 개발자와 함께.
4. **`--sheet-w`(420px)**는 JS의 지도 여백(460)과, **`--half-h`(42vh)**는 JS의 지도 하단 여백 계산과 연동됩니다. 크게 바꾸면 경로가 시트 뒤에 가려집니다.
5. **`#sheet`는 `transform`만 애니메이션**하세요. `max-height`/`height`에 transition을 주면 폰에서 뚝뚝 끊깁니다.
6. **z-index 순서 유지**: `#intro(3100) > #lock(3000) > #aimodal(2500) > #sidemenu(1600) > #rvform(1550) > #rvpage/#myrv/#onboard(1500) > #ranking(1450) > #toast(1300) > #dtop(1200) > #dpanel(1100) > #sheet(950) > #topbar(900) > 지도`
7. **마커 크기**(`.marker-pin` 등)를 바꾸면 JS의 `iconSize / iconAnchor`(핀 끝 좌표)와 어긋나 위치가 틀어집니다.
8. 폰에서는 성능을 위해 `backdrop-filter`(블러)와 큰 그림자를 꺼 두었습니다(`@media (max-width: 760px)` 블록). 다시 켜면 지도 위에서 느려집니다.

---

## 3. Claude Code 사용 시 주의사항

프롬프트에 **항상** 아래 두 문장을 넣으세요:

> "web/index.html의 **CSS 스타일만 수정해줘**. **JavaScript, HTML 구조, id/class 이름, server.py는 건드리지 마.**"

- 요청 예시
  - ✅ "경로 카드(.card)의 배경을 연한 회색으로, 모서리를 더 둥글게 바꿔줘. CSS만."
  - ✅ ":root의 --accent를 #0ea5e9로 바꾸고, 버튼 hover 색도 맞춰줘. JS는 건드리지 마."
  - ❌ "카드에 공유 버튼 추가해줘" → HTML/JS 변경이 필요한 요청. 개발자에게.
  - ❌ "경로 선 색을 바꿔줘" → JS 상수(`COLORS`). 개발자에게 값 전달.
- Claude가 결과를 보여주면 **diff에 `<script>` 안쪽이나 `id="…"` 변경이 섞였는지** 확인하세요. 섞였으면 되돌리게 하세요.
- 수정 후 **반드시 브라우저에서 기능 동작 확인** (5번 참고). 겉모습만 보지 말고 버튼을 눌러 보세요.
- 작업 전 `git status`로 깨끗한지 확인하고, 작업 단위마다 커밋해 두면 되돌리기 쉽습니다.

---

## 4. 파일 구조

```
walk-route/
├── web/
│   ├── index.html        ← 서비스 화면. 이것만 수정 (그중 <style> 블록만)
│   └── admin.html        ← 관리자 페이지. 건드리지 말 것
├── server.py             ← 서버. 절대 건드리지 말 것
├── routing.py            ← 경로 계산. 절대 건드리지 말 것
├── candidate_routes.py   ← 후보 경로. 절대 건드리지 말 것
├── waypoints.py          ← 경유지. 절대 건드리지 말 것
├── generate_test_reviews.py, validate_*.py, rescore_*.py …  ← 개발용 스크립트. 건드리지 말 것
├── osmnx_outputs/        ← 지도 데이터·리뷰 데이터(JSON)·이 문서. 데이터 파일 건드리지 말 것
├── data/                 ← 원천 데이터
└── .env                  ← API 키. 열어보지도, 공유하지도 말 것
```

---

## 5. 확인 방법

1. 터미널에서 프로젝트 폴더로 이동 후 서버 실행
   ```
   python server.py        (Windows)
   python3 server.py       (Mac/Linux)
   ```
   `Serving on http://127.0.0.1:8000` 이 뜨면 준비 완료.
2. 브라우저에서 **http://localhost:8000** 접속
   - CSS를 고친 뒤에는 **Ctrl+F5 (Mac: Cmd+Shift+R)** 강력 새로고침. 서버 재시작은 필요 없음.
   - 모바일 확인: 브라우저 개발자 도구(F12) → 기기 모드(폰 아이콘) → 폭 390px 정도.
3. 아래 기능이 **모두 동작하는지** 눌러 봅니다.
   - [ ] 닉네임 입력 → 온보딩(취향 선택) → 지도 표시
   - [ ] **경로 탐색**: 출발/도착 입력 → "경로 탐색" → 경로 3개 + 카드 3장. 카드 탭 → 펼쳐짐/접힘, 지도 강조
   - [ ] 카드 "자세히 보기" → 턴바이턴 상세, ← 뒤로
   - [ ] **리뷰**: 지도 구간 탭(폰: 전체화면 페이지 / PC: 팝업) → 점수·태그·글 → 제출 → 토스트
   - [ ] "리뷰 N개 보기 →" 목록, 정렬 탭 3개
   - [ ] **랭킹**: 🏆 인기 경로(폰은 ≡ 메뉴) → 목록 → "지도에서 보기" / "경유지로 추가해 탐색"
   - [ ] ⚠️ 위험 구간 토글, 📍 주변 둘러보기, ≡ 메뉴, 계정 메뉴(내 리뷰, 위치 사용)
   - [ ] 폰: 시트 핸들 탭 → 접힘/반/전체 순환, 지도 탭 → 시트 내려감
4. 무언가 안 되면: **콘솔(F12 → Console)에 빨간 에러**가 있는지 확인하고, 마지막 수정을 되돌린 뒤(`git checkout web/index.html` 또는 Claude에게 "방금 수정 되돌려줘") 다시 시도하세요.
