%%writefile main.py
# main.py (매크로 지표 확장: KOSPI/KOSDAQ/SP500/NASDAQ/USD-KRW/WTI/Gold/VIX/DollarIndex)
from __future__ import annotations

import importlib
import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

pd: Any | None = None
requests_module: Any | None = None
stock: Any | None = None
yf: Any | None = None  # yfinance

DATA_DIR = Path("data")


@dataclass(frozen=True)
class APIEndpoint:
    name: str
    url: str
    params: Optional[dict[str, Any]] = None
    description: str = ""
    transform: Optional[Callable[[Any], Any]] = None


def ensure_packages() -> None:
    """Colab/로컬 어디서든 필요한 패키지를 보장"""
    global pd, requests_module, stock, yf

    required = {
        "pandas": "pandas",
        "requests": "requests",
        "pykrx": "pykrx",
        "yfinance": "yfinance",
    }

    for module_name, package_name in required.items():
        if importlib.util.find_spec(module_name) is None:
            print(f"[INFO] Installing missing package: {package_name}")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            except subprocess.CalledProcessError as exc:
                print(f"[WARN] Failed to install {package_name}: {exc}")

    pd = importlib.import_module("pandas")
    requests_module = importlib.import_module("requests")
    pykrx_module = importlib.import_module("pykrx")
    stock = getattr(pykrx_module, "stock")
    try:
        yf = importlib.import_module("yfinance")
    except ModuleNotFoundError:
        yf = None


def fetch_endpoint(endpoint: APIEndpoint) -> Any:
    assert requests_module is not None
    try:
        response = requests_module.get(endpoint.url, params=endpoint.params, timeout=15)
        response.raise_for_status()
    except requests_module.RequestException as exc:
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
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def ensure_directory(base: Path) -> Path:
    # UTC 타임스탬프 하위 폴더 생성
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    d = base / ts
    d.mkdir(parents=True, exist_ok=True)
    return d


def collect_external_apis(endpoints: Iterable[APIEndpoint], output_dir: Path) -> None:
    for ep in endpoints:
        print(f"[INFO] Fetching {ep.name} ({ep.url})")
        data = fetch_endpoint(ep)
        save_json(data, output_dir / f"{ep.name}.json")


def _to_str_date_index(df: Any, idx_name: str = "날짜") -> Any:
    """DatetimeIndex → YYYYMMDD 문자열 인덱스로 정규화"""
    if df is None or getattr(df, "empty", True):
        return df
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.Index([d.strftime("%Y%m%d") for d in df.index], name=idx_name)
    elif "날짜" in getattr(df, "columns", []):
        df = df.copy()
        df.set_index("날짜", inplace=True)
    return df


def _join_outer(base: Any, other: Any) -> Any:
    if base is None or getattr(base, "empty", True):
        return other
    if other is None or getattr(other, "empty", True):
        return base
    return base.join(other, how="outer")


def _download_close_series(sym: str, start: str, end: str) -> Any:
    """yfinance에서 Close만 내려받아 YYYYMMDD 인덱스로 반환"""
    df = yf.download(sym, start=start, end=end, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return None
    df = df[["Close"]]
    df = _to_str_date_index(df)
    return df


def _add_macro_features_with_yfinance(combined: Any, start_date: datetime, end_date: datetime) -> Any:
    """yfinance로 지수/환율/원자재 등 매크로 피처를 추가"""
    if yf is None:
        print("[INFO] yfinance 미설치: 매크로 피처 스킵")
        return combined

    s = start_date.strftime("%Y-%m-%d")
    e = end_date.strftime("%Y-%m-%d")

    # 여러 심볼 후보를 순차 시도하는 사전 (첫 성공을 채택)
    symbol_candidates: dict[str, list[str]] = {
        "macro_KOSPI":  ["^KS11"],
        "macro_KOSDAQ": ["^KQ11", "^KOSDAQ"],  # 가끔 제공 안 될 수 있어 후보 넣음
        "macro_SP500":  ["^GSPC"],
        "macro_NASDAQ": ["^IXIC"],
        "macro_USDKRW": ["KRW=X"],             # 1 USD당 KRW
        "macro_WTI":    ["CL=F"],
        "macro_GOLD":   ["GC=F"],
        "macro_VIX":    ["^VIX"],
        # 달러인덱스: 선물/현물 심볼이 지역에 따라 다름 → 순차 시도
        "macro_DOLLAR_INDEX": ["DX-Y.NYB", "^DXY"],
    }

    out = combined.copy() if combined is not None else pd.DataFrame()

    for col_name, candidates in symbol_candidates.items():
        success = False
        for sym in candidates:
            try:
                print(f"[INFO] yfinance {col_name} 후보 {sym} 다운로드")
                df = _download_close_series(sym, s, e)
                if df is None or df.empty:
                    continue
                df = df.rename(columns={"Close": col_name})
                out = _join_outer(out, df)
                success = True
                break
            except Exception as exc:
                print(f"[WARN] {col_name} ({sym}) 실패: {exc}")
        if not success:
            print(f"[WARN] {col_name} 모든 후보 실패 → 스킵")
    return out


def collect_korean_stock_data(
    ticker: str,
    days: int,
    market: str,
    investor: str,
    output_dir: Path,
) -> None:
    """pykrx + yfinance + (매크로지표) 를 합쳐 CSV 저장"""
    assert pd is not None and stock is not None

    today = datetime.now()
    start_date = today - timedelta(days=days)
    f_today = today.strftime("%Y%m%d")
    f_start = start_date.strftime("%Y%m%d")

    print(f"[INFO] Collecting KRX data for ticker={ticker}, period={f_start}~{f_today}")

    combined = pd.DataFrame()

    # OHLCV
    try:
        ohlcv = stock.get_market_ohlcv_by_date(f_start, f_today, ticker)
        if not ohlcv.empty:
            ohlcv.index = ohlcv.index.strftime("%Y%m%d")
            combined = ohlcv.copy()
            combined["result"] = combined["등락률"].apply(lambda v: 1 if v >= 0 else 0)
    except Exception as exc:
        print(f"[WARN] OHLCV 조회 실패: {exc}")

    # Fundamentals
    try:
        fundamentals = stock.get_market_fundamental(f_start, f_today, ticker)
        if not fundamentals.empty:
            fundamentals.index = fundamentals.index.strftime("%Y%m%d")
            combined = combined.join(fundamentals, how="outer") if not combined.empty else fundamentals
    except Exception as exc:
        print(f"[WARN] 펀더멘털 조회 실패: {exc}")

    # 시가총액/외국인보유/공매도
    extra = [
        ("시가총액", lambda: stock.get_market_cap(f_start, f_today, ticker)),
        ("외국인보유", lambda: stock.get_exhaustion_rates_of_foreign_investment(f_start, f_today, ticker)),
        ("공매도",     lambda: stock.get_shorting_status_by_date(f_start, f_today, ticker)),
    ]
    for label, loader in extra:
        try:
            df = loader()
        except Exception as exc:
            print(f"[WARN] {label} 조회 실패: {exc}")
            continue
        if df is None or getattr(df, "empty", True):
            continue
        df = df.copy()
        if isinstance(df.index, pd.DatetimeIndex):
            df.index = df.index.strftime("%Y%m%d")
        elif "날짜" in df.columns:
            df.set_index("날짜", inplace=True)
        df = df.add_prefix(f"{label}_")
        combined = combined.join(df, how="outer") if not combined.empty else df

    # 투자자별 순매수 (느릴 수 있음)
    cats = [investor, "개인", "외국인", "기관합계", "금융투자", "보험", "투신", "사모", "은행", "기타금융", "연기금 등", "국가", "기타법인"]
    cats = list(dict.fromkeys([c for c in cats if c]))
    for offset in range(days):
        cur = today - timedelta(days=offset)
        cur_s = cur.strftime("%Y%m%d")
        prev_s = (cur - timedelta(days=1)).strftime("%Y%m%d")
        for c in cats:
            try:
                p = stock.get_market_net_purchases_of_equities(prev_s, cur_s, market, c)
            except Exception as exc:
                print(f"[WARN] {cur_s} {c} 순매수 조회 실패: {exc}")
                continue
            if p is None or getattr(p, "empty", True):
                continue
            if "티커" in p.columns:
                row = p[p["티커"].astype(str) == ticker]
            else:
                row = p[p.index.astype(str) == ticker]
            if row.empty:
                continue
            row = row.copy()
            if "티커" in row.columns:
                row.drop(columns=["티커"], inplace=True)
            row.rename(columns={col: f"{c}_{col}" for col in row.columns}, inplace=True)
            row["날짜"] = cur_s
            row.set_index("날짜", inplace=True)
            combined = combined.combine_first(row)

    # 보조 소스: yfinance (개별 종목)
    if yf is not None:
        try:
            sym = f"{ticker}.KS"  # 삼성전자
            yfd = yf.download(sym, start=start_date.strftime("%Y-%m-%d"), end=today.strftime("%Y-%m-%d"), auto_adjust=False, progress=False)
            if not yfd.empty:
                yfd.index = pd.Index([dt.strftime("%Y%m%d") for dt in yfd.index], name="날짜")
                yfd = yfd.add_prefix("yfi_")
                combined = combined.join(yfd, how="outer") if not combined.empty else yfd
        except Exception as exc:
            print(f"[WARN] yfinance 종목 조회 실패: {exc}")

    # 🔥 매크로 피처(지수/환율/원자재/VIX/달러인덱스) 추가
    combined = _add_macro_features_with_yfinance(combined, start_date, today)

    # 저장
    csv_path = output_dir / f"krx_{ticker}.csv"
    combined.sort_index().to_csv(csv_path, encoding="utf-8-sig")
    print(f"[INFO] Saved KRX dataset to {csv_path}")
    print(f"[INFO] Rows={len(combined):,}, Cols={len(combined.columns):,}")


def collect_naver_finance_data(ticker: str, output_dir: Path) -> None:
    """네이버 금융 비공식 API에서 최근 가격 리스트"""
    assert pd is not None and requests_module is not None
    url = f"https://api.stock.naver.com/domestic/stock/{ticker}/price"
    params = {"pageSize": "200"}
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests_module.get(url, params=params, headers=headers, timeout=15)
        r.raise_for_status()
    except requests_module.RequestException as exc:
        print(f"[WARN] 네이버 금융 조회 실패: {exc}")
        return

    try:
        payload = r.json()
    except ValueError:
        print("[WARN] 네이버 응답이 JSON 형식이 아님.")
        return

    records = payload.get("data") or payload.get("datas") or payload.get("prices") or payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(records, list) or not records:
        print("[WARN] 네이버에서 변환 가능한 데이터 없음.")
        return

    frame = pd.DataFrame(records)
    if frame.empty:
        print("[WARN] 네이버 데이터가 비어있음.")
        return

    if "localTradedAt" in frame.columns:
        frame.set_index("localTradedAt", inplace=True)
    elif "tradeDate" in frame.columns:
        frame.set_index("tradeDate", inplace=True)
    frame.sort_index(inplace=True)

    save_json(frame.to_dict(orient="index"), output_dir / f"naver_{ticker}.json")
    print(f"[INFO] Saved Naver Finance dataset for {ticker}")


def main() -> None:
    ensure_packages()

    # Google Drive 마운트(가능하면)
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        print("[INFO] Google Drive mounted successfully.")
        base_dir = Path("/content/drive/MyDrive/krx_data")
    except Exception as e:
        print(f"[WARN] Google Drive mount failed: {e}")
        print("[INFO] 로컬 data 폴더에 저장합니다.")
        base_dir = Path("data")

    output_dir = ensure_directory(base_dir)

    endpoints = [
        APIEndpoint("public_apis", "https://api.publicapis.org/entries", description="Public API 리스트"),
        APIEndpoint("exchange_rates_usd", "https://open.er-api.com/v6/latest/USD", description="USD 기준 환율"),
        APIEndpoint("ipinfo", "https://ipinfo.io/json", description="IP 정보"),
        APIEndpoint("world_time", "https://worldtimeapi.org/api/timezone/Etc/UTC", description="세계 표준시"),
        APIEndpoint("github_events", "https://api.github.com/events", description="GitHub 공개 이벤트"),
    ]

    # 외부 API는 실패해도 계속 진행
    collect_external_apis(endpoints, output_dir)

    collect_korean_stock_data(
        ticker="005930",
        days=1825,         # 약 5년
        market="KOSPI",
        investor="개인",
        output_dir=output_dir,
    )

    collect_naver_finance_data("005930", output_dir)

    print(f"[INFO] 모든 데이터가 {output_dir} 디렉터리에 저장되었습니다.")
    print(f"[INFO] Google Drive 경로: {output_dir.resolve()}")
    print("[INFO] 예시 파일:")
    print(f" - {output_dir / 'krx_005930.csv'}")


if __name__ == "__main__":
    main()
