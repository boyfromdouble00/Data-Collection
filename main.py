 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/main.py b/main.py
index fcff1ab3474c31e0610ea323967e42c81b9864d5..a67240db1c3c01fe4e0de300eeaa72ae207609bc 100644
--- a/main.py
+++ b/main.py
@@ -1,67 +1,202 @@
-import pandas as pd
+from __future__ import annotations
+
+import json
+from dataclasses import dataclass
 from datetime import datetime, timedelta
-from pykrx import stock
+from pathlib import Path
+from typing import Any, Callable, Iterable, Optional
 
-# 오늘 날짜 가져오기
-today = datetime.now()
-formatted_today = today.strftime('%Y%m%d')
+import pandas as pd
+import requests
+from pykrx import stock
 
-# 1년 전 날짜 계산
-one_year_ago = today - timedelta(days=365)
-formatted_one_year_ago = one_year_ago.strftime('%Y%m%d')
 
-# df에 ohlcv 저장
-df = stock.get_market_ohlcv_by_date(formatted_one_year_ago, formatted_today, "005930")
+DATA_DIR = Path("data")
 
-# 결과 값을 계산하여 새로운 데이터프레임 생성
-result_df = df['등락률'].apply(lambda x: 1 if x >= 0 else 0)
 
-# 일자별 Div/BPS/PER/EPS 조회
-mjm = stock.get_market_fundamental(formatted_one_year_ago, formatted_today, "005930")
+@dataclass(frozen=True)
+class APIEndpoint:
+    """구성된 외부 API 엔드포인트 정보를 표현한다."""
 
-# 최종 데이터프레임 생성
-final_df = pd.DataFrame({
-    'result': result_df,
-    '거래량': df['거래량'],
-    'PER': mjm['PER'],
-    'DIV': mjm['DIV']
-})
+    name: str
+    url: str
+    params: Optional[dict[str, Any]] = None
+    description: str = ""
+    transform: Optional[Callable[[requests.Response], Any]] = None
 
-final_df.index = final_df.index.strftime('%Y%m%d')
 
-# 365일 동안 반복하여 순매수 데이터 가져오기
-for i in range(365):
-    # 날짜 계산
-    date_for_data = today - timedelta(days=i)
-    date_for_data_str = date_for_data.strftime('%Y%m%d')
-    date_for_data1 = date_for_data - timedelta(days=1)
-    date_for_data1_str = date_for_data1.strftime('%Y%m%d')
+def fetch_endpoint(endpoint: APIEndpoint) -> Any:
+    """지정된 엔드포인트에서 데이터를 요청하고 파싱한다."""
 
     try:
-        # 데이터 가져오기
-        gattcha = stock.get_market_net_purchases_of_equities(date_for_data1_str, date_for_data_str, "KOSPI", "개인")
+        response = requests.get(endpoint.url, params=endpoint.params, timeout=15)
+        response.raise_for_status()
+    except requests.RequestException as exc:
+        print(f"[ERROR] {endpoint.name}: 요청 실패 - {exc}")
+        return {"error": str(exc)}
+
+    if endpoint.transform is not None:
+        return endpoint.transform(response)
+
+    content_type = response.headers.get("Content-Type", "")
+    if "application/json" in content_type:
+        try:
+            return response.json()
+        except ValueError:
+            print(f"[WARN] {endpoint.name}: JSON 디코딩 실패, 원본 텍스트를 반환합니다.")
+            return response.text
+
+    return response.text
+
+
+def save_json(data: Any, path: Path) -> None:
+    """결과 데이터를 JSON 형식으로 저장한다."""
+
+    with path.open("w", encoding="utf-8") as file:
+        json.dump(data, file, ensure_ascii=False, indent=2)
+
+
+def ensure_directory(base: Path) -> Path:
+    """타임스탬프가 포함된 하위 디렉터리를 생성한다."""
+
+    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
+    directory = base / timestamp
+    directory.mkdir(parents=True, exist_ok=True)
+    return directory
+
+
+def collect_external_apis(endpoints: Iterable[APIEndpoint], output_dir: Path) -> None:
+    """사전에 정의된 다양한 API 데이터를 수집한다."""
+
+    for endpoint in endpoints:
+        print(f"[INFO] Fetching {endpoint.name} ({endpoint.url})")
+        data = fetch_endpoint(endpoint)
+        save_json(data, output_dir / f"{endpoint.name}.json")
+
 
-        # gattcha가 비어 있는지 확인하고 비어 있으면 무시
-        if gattcha.empty:
-            continue  # 데이터가 없으면 다음 반복으로 넘어감
+def collect_korean_stock_data(
+    ticker: str, days: int, market: str, investor: str, output_dir: Path
+) -> None:
+    """국내 주식 데이터를 수집하여 CSV 파일로 저장한다."""
 
-        # gattcha 구조 확인
-        print(f"{date_for_data_str}에 대한 gattcha 데이터:")
+    today = datetime.now()
+    start_date = today - timedelta(days=days)
 
-        # '티커' 열이 있을 경우 필터링
-        if '티커' in gattcha.columns:
-            gattcha['티커'] = gattcha['티커'].astype(str)  # 티커를 문자열로 변환
-            gattcha_filtered = gattcha[gattcha['티커'] == '005930']  # 티커를 문자열로 사용
+    formatted_today = today.strftime("%Y%m%d")
+    formatted_start = start_date.strftime("%Y%m%d")
+
+    print(
+        f"[INFO] Collecting KRX data for ticker={ticker}, period={formatted_start}~{formatted_today}"
+    )
+
+    ohlcv = stock.get_market_ohlcv_by_date(formatted_start, formatted_today, ticker)
+    result_series = ohlcv["등락률"].apply(lambda value: 1 if value >= 0 else 0)
+    fundamentals = stock.get_market_fundamental(formatted_start, formatted_today, ticker)
+
+    combined = pd.DataFrame(
+        {
+            "result": result_series,
+            "거래량": ohlcv["거래량"],
+            "PER": fundamentals["PER"],
+            "DIV": fundamentals["DIV"],
+        }
+    )
+
+    combined.index = combined.index.strftime("%Y%m%d")
+
+    for offset in range(days):
+        current_date = today - timedelta(days=offset)
+        current_str = current_date.strftime("%Y%m%d")
+        previous_str = (current_date - timedelta(days=1)).strftime("%Y%m%d")
+
+        try:
+            purchases = stock.get_market_net_purchases_of_equities(
+                previous_str, current_str, market, investor
+            )
+        except Exception as exc:  # noqa: BLE001
+            print(f"[WARN] {current_str} 순매수 데이터 조회 실패: {exc}")
+            continue
+
+        if purchases.empty:
+            continue
+
+        if "티커" in purchases.columns:
+            purchases = purchases.copy()
+            purchases["티커"] = purchases["티커"].astype(str)
+            filtered = purchases[purchases["티커"] == ticker]
         else:
-            # 인덱스를 문자열로 변환하여 필터링
-            gattcha_filtered = gattcha[gattcha.index.astype(str) == '005930']
+            filtered = purchases[purchases.index.astype(str) == ticker]
+
+        if filtered.empty:
+            continue
+
+        filtered = filtered.copy()
+        filtered["날짜"] = current_str
+        filtered.set_index("날짜", inplace=True)
+
+        combined = combined.combine_first(filtered)
+
+    csv_path = output_dir / f"krx_{ticker}.csv"
+    combined.sort_index().to_csv(csv_path, encoding="utf-8-sig")
+    print(f"[INFO] Saved KRX dataset to {csv_path}")
+
+
+def main() -> None:
+    output_dir = ensure_directory(DATA_DIR)
+
+    endpoints = [
+        APIEndpoint(
+            name="public_apis",
+            url="https://api.publicapis.org/entries",
+            description="Public API 리스트",
+        ),
+        APIEndpoint(
+            name="exchange_rates_usd",
+            url="https://open.er-api.com/v6/latest/USD",
+            description="미국 달러 기준 환율",
+        ),
+        APIEndpoint(
+            name="bitcoin_price",
+            url="https://api.coindesk.com/v1/bpi/currentprice.json",
+            description="비트코인 시세",
+        ),
+        APIEndpoint(
+            name="spacex_latest_launch",
+            url="https://api.spacexdata.com/v4/launches/latest",
+            description="SpaceX 최신 발사 정보",
+        ),
+        APIEndpoint(
+            name="restcountries",
+            url="https://restcountries.com/v3.1/all",
+            description="전 세계 국가 정보",
+        ),
+        APIEndpoint(
+            name="datausa_population",
+            url="https://datausa.io/api/data",
+            params={"drilldowns": "Nation", "measures": "Population"},
+            description="미국 인구 통계",
+        ),
+        APIEndpoint(
+            name="world_time",
+            url="https://worldtimeapi.org/api/timezone/Etc/UTC",
+            description="세계 표준시",
+        ),
+        APIEndpoint(
+            name="github_events",
+            url="https://api.github.com/events",
+            description="GitHub 공개 이벤트",
+        ),
+    ]
+
+    collect_external_apis(endpoints, output_dir)
+
+    collect_korean_stock_data(
+        ticker="005930", days=365, market="KOSPI", investor="개인", output_dir=output_dir
+    )
+
+    print(f"[INFO] 모든 데이터가 {output_dir} 디렉터리에 저장되었습니다.")
 
-        # 날짜 열 추가 및 인덱스로 설정
-        gattcha_filtered['날짜'] = date_for_data_str  # 날짜 열 추가
-        gattcha_filtered.set_index('날짜', inplace=True)  # 날짜를 인덱스로 설정
 
-        # 기존 데이터에 추가된 gattcha_filtered 병합
-        final_df = final_df.combine_first(gattcha_filtered)
+if __name__ == "__main__":
+    main()
 
-    except Exception as e:
-        print(f"데이터 가져오기 오류: {e}")
 
EOF
)
