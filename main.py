# main.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import pandas as pd
import requests
from pykrx import stock

DATA_DIR = Path("data")


# ---------------------------
# External API definitions
# ---------------------------
@dataclass(frozen=True)
class APIEndpoint:
    """구성된 외부 API 엔드포인트 정보를 표현한다."""
    name: str
    url: str
    params: Optional[dict[str, Any]] = None
    description: str = ""
    transform: Optional[Callable[[requests.Response], Any]] = None


def fetch_endpoint(endpoint: APIEndpoint) -> Any:
    """지정된 엔드포인트에서 데이터를 요청하고 파싱한다."""
    try:
        response = requests.get(endpoint.url, params=endpoint.params, timeout=15)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"[ERROR] {endpoint.name}: 요청 실패 - {exc}")
        return {"error": str(exc)}

    if endpoint.transform is not None:
        return endpoint.transform(response)

    content_type = response.headers.get("Content-Type", "")
    if "application/json" in content_type:
        try:
            return response.json()
        except ValueError:
            print(f"[WARN] {endpoint.name}: JSON 디코딩 실패, 원본 텍스트를 반환합니다.")
            return response.text

    return response.text


def save_json(data: Any, path: Path) -> None:
    """결과 데이터를 JSON 형식으로 저장한다."""
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def ensure_directory(base: Path) -> Path:
    """타임스탬프가 포함된 하위 디렉터리를 생성한다 (UTC)."""
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    directory = base / timestamp
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def collect_external_apis(endpoints: Iterable[APIEndpoint], output_dir: Path) -> None:
    """사전에 정의된 다양한 API 데이터를 수집한다."""
    for endpoint in endpoints:
        print(f"[INFO] Fetching {endpoint.name} ({endpoint.url})")
        data = fetch_endpoint(endpoint)
        save_json(data, output_dir / f"{endpoint.name}.json")


# ---------------------------
# KRX (pykrx) data pipeline
# ---------------------------
def collect_korean_stock_data(
    ticker: str,
    days: int,
    market: str,
    investor: str,
    output_dir: Path,
) -> None:
    """국내 주식 데이터를 수집하여 CSV 파일로 저장한다."""
    today = datetime.now()
    start_date = today - timedelta(days=days)

    formatted_today = today.strftime("%Y%m%d")
    formatted_start = start_date.strftime("%Y%m%d")

    print(
        f"[INFO] Collecting KRX data for ticker={ticker}, "
        f"period={formatted_start}~{formatted_today}"
    )

    # 기본 시계열 (OHLCV, Fundamental)
    ohlcv = stock.get_market_ohlcv_by_date(formatted_start, formatted_today, ticker)
    # 상승/보합=1, 하락=0
    result_series = ohlcv["등락률"].apply(lambda value: 1 if value >= 0 else 0)
    fundamentals = stock.get_market_fundamental(formatted_start, formatted_today, ticker)

    combined = pd.DataFrame(
        {
            "result": result_series,
            "거래량": ohlcv["거래량"],
            "PER": fundamentals["PER"],
            "DIV": fundamentals["DIV"],
        }
    )
    # 날짜를 문자열(YYYYMMDD)로
    combined.index = combined.index.strftime("%Y%m%d")

    # 일자별 개인 순매수 데이터 결합
    for offset in range(days):
        current_date = today - timedelta(days=offset)
        current_str = current_date.strftime("%Y%m%d")
        previous_str = (current_date - timedelta(days=1)).strftime("%Y%m%d")

        try:
            purchases = stock.get_market_net_purchases_of_equities(
                previous_str, current_str, market, investor
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] {current_str} 순매수 데이터 조회 실패: {exc}")
            continue

        if purchases is None or getattr(purchases, "empty", True):
            continue

        # 티커 필터링 (컬럼/인덱스 양쪽 케이스 처리)
        if "티커" in purchases.columns:
            df_filtered = purchases[purchases["티커"].astype(str) == ticker]
        else:
            df_filtered = purchases[purchases.index.astype(str) == ticker]

        if df_filtered.empty:
            continue

        df_filtered = df_filtered.copy()
        df_filtered["날짜"] = current_str
        df_filtered.set_index("날짜", inplace=True)

        # 기존 프레임에 순매수 관련 컬럼(원본 명칭 유지) 주입
        combined = combined.combine_first(df_filtered)

    csv_path = output_dir / f"krx_{ticker}.csv"
    combined.sort_index().to_csv(csv_path, encoding="utf-8-sig")
    print(f"[INFO] Saved KRX dataset to {csv_path}")


# ---------------------------
# Entrypoint
# ---------------------------
def main() -> None:
    output_dir = ensure_directory(DATA_DIR)

    # (원본 JSON 스냅샷 아카이브) 외부 공개 API들
    endpoints = [
        APIEndpoint(
            name="public_apis",
            url="https://api.publicapis.org/entries",
            description="Public API 리스트",
        ),
        APIEndpoint(
            name="exchange_rates_usd",
            url="https://open.er-api.com/v6/latest/USD",
            description="미국 달러 기준 환율",
        ),
        APIEndpoint(
            name="bitcoin_price",
            url="https://api.coindesk.com/v1/bpi/currentprice.json",
            description="비트코인 시세",
        ),
        APIEndpoint(
            name="spacex_latest_launch",
            url="https://api.spacexdata.com/v4/launches/latest",
            description="SpaceX 최신 발사 정보",
        ),
        APIEndpoint(
            name="restcountries",
            url="https://restcountries.com/v3.1/all",
            description="전 세계 국가 정보",
        ),
        APIEndpoint(
            name="datausa_population",
            url="https://datausa.io/api/data",
            params={"drilldowns": "Nation", "measures": "Population"},
            description="미국 인구 통계",
        ),
        APIEndpoint(
            name="world_time",
            url="https://worldtimeapi.org/api/timezone/Etc/UTC",
            description="세계 표준시",
        ),
        APIEndpoint(
            name="github_events",
            url="https://api.github.com/events",
            description="GitHub 공개 이벤트",
        ),
    ]

    collect_external_apis(endpoints, output_dir)

    # ★ 5년치: days=1825 로 변경 ★
    collect_korean_stock_data(
        ticker="005930",
        days=1825,                  # 약 5년
        market="KOSPI",
        investor="개인",
        output_dir=output_dir,
    )

    print(f"[INFO] 모든 데이터가 {output_dir} 디렉터리에 저장되었습니다.")


if __name__ == "__main__":
    main()
