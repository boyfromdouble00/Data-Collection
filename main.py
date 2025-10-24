%%writefile main.py
# main.py — 1000영업일: 네이버 OHLCV + yfinance(환율/지수/원자재/VIX/DXY) + 네이버뉴스(개수/감성) → 드라이브 저장
from __future__ import annotations

import os, re, json, math, time, sys, subprocess, importlib, importlib.util
from io import StringIO
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional
from math import ceil

pd: Any | None = None
requests: Any | None = None
bs4: Any | None = None
yf: Any | None = None
transformers: Any | None = None
torch: Any | None = None

# ===== 사용자 파라미터 (환경변수로 변경 가능) =====
CODE = os.getenv("CODE", "005930")            # 네이버 종목코드 (삼성전자)
TARGET_DAYS = int(os.getenv("DAYS", "1000"))  # 목표 영업일 수
SAVE_DIR = Path(os.getenv("SAVE_DIR", "/content/drive/MyDrive/krx_data"))
ADD_TECH = os.getenv("TECH", "1") == "1"      # 간단 기술지표 생성
ADD_NEWS = os.getenv("NEWS", "1") == "1"      # 네이버 뉴스(개수/감성) 추가
# 뉴스 감성 모델(멀티링구얼 별점 1~5 → -1~+1로 변환)
SENTIMENT_MODEL = os.getenv("SENT_MODEL", "nlptown/bert-base-multilingual-uncased-sentiment")

UA = {"User-Agent": "Mozilla/5.0"}

def ensure_packages() -> None:
    global pd, requests, bs4, yf, transformers, torch
    need = {
        "pandas": "pandas",
        "requests": "requests",
        "beautifulsoup4": "beautifulsoup4",
        "yfinance": "yfinance",
        "transformers": "transformers",
        "torch": "torch",
    }
    for mod, pkg in need.items():
        if importlib.util.find_spec(mod) is None:
            print(f"[INFO] Installing {pkg} ...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            except subprocess.CalledProcessError as e:
                print(f"[WARN] Failed to install {pkg}: {e}")

    pd = importlib.import_module("pandas")
    requests = importlib.import_module("requests")
    bs4 = importlib.import_module("bs4")
    yf = importlib.import_module("yfinance")
    try:
        transformers = importlib.import_module("transformers")
        torch = importlib.import_module("torch")
    except ModuleNotFoundError:
        transformers, torch = None, None


# ---------- 공통 유틸 ----------
def _clean_numeric(x: Any) -> Optional[float]:
    if pd.isna(x): return None
    if isinstance(x, (int, float)): return float(x)
    s = re.sub(r"[^\d\.-]", "", str(x))
    if s in ("", "-", ".", "-.", ".-"): return None
    try: return float(s)
    except ValueError: return None

def _to_str_index(df: Any) -> Any:
    if df is None or getattr(df, "empty", True): return df
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.copy(); df.index = df.index.strftime("%Y%m%d")
    return df

def _join_left(base: Any, other: Any) -> Any:
    if base is None or getattr(base, "empty", True): return other
    if other is None or getattr(other, "empty", True): return base
    return base.join(other, how="left")


# ---------- 1) 네이버 일별 시세(OHLCV) ----------
def fetch_naver_ohlcv(code: str, target_days: int) -> pd.DataFrame:
    """네이버 일별시세 페이지를 필요한 행 수가 채워질 때까지 크롤링"""
    base = "https://finance.naver.com/item/sise_day.nhn"
    frames = []
    # 1페이지 ≈ 10영업일 → 여유분 포함해 페이지 수 추정
    max_pages = ceil(target_days / 10) + 20
    for p in range(1, max_pages + 1):
        url = f"{base}?code={code}&page={p}"
        html = requests.get(url, headers=UA, timeout=15).text
        # FutureWarning 해결: StringIO로 감싸서 파싱
        tables = pd.read_html(StringIO(html))
        if not tables:
            continue
        df = tables[0].dropna(how="all").dropna()
        if df.empty:
            continue
        df.columns = ["날짜","종가","전일비","시가","고가","저가","거래량"]
        frames.append(df)
        # 충분히 모이면 중단
        if sum(len(f) for f in frames) >= target_days + 50:
            break
        time.sleep(0.15)  # 네이버 예의상 살짝 쉬기

    if not frames:
        return pd.DataFrame()

    df_all = pd.concat(frames, ignore_index=True)
    df_all["날짜"] = pd.to_datetime(df_all["날짜"], format="%Y.%m.%d", errors="coerce")
    df_all = df_all.dropna(subset=["날짜"]).sort_values("날짜")
    for c in ["종가","전일비","시가","고가","저가","거래량"]:
        df_all[c] = df_all[c].map(_clean_numeric)
    df_all.index = df_all["날짜"].dt.strftime("%Y%m%d")
    df_all.drop(columns=["날짜"], inplace=True)

    # 최신 target_days만 슬라이스
    if len(df_all) > target_days:
        df_all = df_all.iloc[-target_days:].copy()

    # 등락률/라벨
    df_all["등락률"] = df_all["종가"].pct_change() * 100
    df_all["result"] = (df_all["등락률"].fillna(0) >= 0).astype(int)

    order = ["종가","시가","고가","저가","거래량","전일비","등락률","result"]
    return df_all[[c for c in order if c in df_all.columns]]


# ---------- 2) yfinance: 환율/지수/원자재/VIX/DXY ----------
def yf_close(sym: str, s: str, e: str) -> pd.DataFrame | None:
    # 멀티레벨 방지/평탄화 옵션
    data = yf.download(
        sym, start=s, end=e, auto_adjust=False, progress=False,
        group_by="column", threads=False
    )
    if data is None or data.empty:
        return None

    # tz 제거
    try:
        data.index = data.index.tz_localize(None)
    except Exception:
        pass

    # Close만 단일 컬럼으로 강제
    if isinstance(data.columns, pd.MultiIndex):
        try:
            close = data.xs("Close", level=0, axis=1)
            if isinstance(close, pd.DataFrame) and close.shape[1] >= 1:
                close = close.iloc[:, 0]
        except Exception:
            try:
                close = data[("Close", sym)]
            except Exception:
                return None
    else:
        close = data.get("Close", None)
        if close is None:
            return None

    ser = close.to_frame(name="Close")
    if isinstance(ser.index, pd.DatetimeIndex):
        ser.index = ser.index.strftime("%Y%m%d")
    return ser

def add_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    s = pd.to_datetime(df.index.min(), format="%Y%m%d").strftime("%Y-%m-%d")
    e = pd.to_datetime(df.index.max(), format="%Y%m%d").strftime("%Y-%m-%d")

    # 원심볼 실패 대비 ETF 대체 심볼 추가
    candidates = {
        "macro_KOSPI":  ["^KS11", "EWY"],        # KOSPI / 한국 ETF
        "macro_KOSDAQ": ["^KQ11"],               # 대체 마땅치 않음
        "macro_SP500":  ["^GSPC", "SPY"],        # S&P500 / ETF
        "macro_NASDAQ": ["^IXIC", "QQQ"],        # NASDAQ / ETF
        "macro_USDKRW": ["KRW=X"],               # 환율
        "macro_WTI":    ["CL=F", "USO"],         # 원유 선물 / ETF
        "macro_GOLD":   ["GC=F", "GLD"],         # 금 선물 / ETF
        "macro_VIX":    ["^VIX", "VIXY"],        # VIX / ETF
        "macro_DXY":    ["DX-Y.NYB", "^DXY", "UUP"],  # 달러인덱스 / ETF
    }

    out = df.copy()
    for col, syms in candidates.items():
        ok = False
        for sym in syms:
            try:
                print(f"[INFO] macro {col} ← {sym}")
                ser = yf_close(sym, s, e)
                if ser is None or ser.empty: continue
                ser = ser.rename(columns={"Close": col})
                out = _join_left(out, ser)
                ok = True; break
            except Exception as ex:
                print(f"[WARN] {col} ({sym}) 실패: {ex}")
        if not ok:
            print(f"[WARN] {col} 모든 후보 실패 → 스킵")
    return out


# ---------- 3) 뉴스: 네이버 뉴스 크롤링 (개수 + 감성) ----------
def build_sentiment_pipeline():
    if transformers is None or torch is None:
        print("[INFO] transformers/torch 없음 → 뉴스 감성 스킵")
        return None
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    try:
        tok = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
        mdl = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL)
        return pipeline("sentiment-analysis", model=mdl, tokenizer=tok, truncation=True)
    except Exception as e:
        print(f"[WARN] 감성모델 로드 실패: {e}")
        return None

def _score_from_label(label: str) -> float:
    # '1 star'~'5 stars' → -1.0 ~ +1.0
    m = re.search(r"(\d)", label)
    if not m: return 0.0
    stars = int(m.group(1))
    return (stars - 3) / 2.0  # 1→-1, 3→0, 5→+1

def crawl_naver_news_titles(query: str, start_dt: datetime, end_dt: datetime, max_pages: int = 80) -> pd.DataFrame:
    from bs4 import BeautifulSoup
    rows = []
    for page in range(1, max_pages + 1):
        start_idx = 1 + (page - 1) * 10
        url = f"https://search.naver.com/search.naver?where=news&query={query}&sm=tab_opt&sort=1&start={start_idx}"
        html = requests.get(url, headers=UA, timeout=15).text
        soup = BeautifulSoup(html, "html.parser")
        items = soup.select("div.news_area")
        if not items:
            break
        for it in items:
            title_el = it.select_one("a.news_tit")
            if not title_el: continue
            title = title_el.get("title") or title_el.text.strip()
            # 날짜 추정(어제/몇시간전/절대날짜 등)
            date_el = it.select_one("span.info")
            date_txt = date_el.text.strip() if date_el else ""
            pub_dt = _parse_news_datetime(date_txt)
            if pub_dt is None:
                continue
            if not (start_dt <= pub_dt <= end_dt):
                continue
            rows.append({"date": pub_dt.date(), "title": title})
        time.sleep(0.15)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df

def _parse_news_datetime(s: str) -> Optional[datetime]:
    s = s.strip()
    now = datetime.now()
    m = re.match(r"(\d{4})\.(\d{1,2})\.(\d{1,2})", s)
    if m:
        y, mo, d = map(int, m.groups())
        return datetime(y, mo, d)
    if "어제" in s:
        return now - timedelta(days=1)
    m = re.match(r"(\d+)\s*일\s*전", s)
    if m:
        return now - timedelta(days=int(m.group(1)))
    m = re.match(r"(\d+)\s*시간\s*전", s)
    if m:
        return now - timedelta(hours=int(m.group(1)))
    m = re.match(r"(\d+)\s*분\s*전", s)
    if m:
        return now - timedelta(minutes=int(m.group(1)))
    return None

def add_news_features(df: pd.DataFrame, query: str = "삼성전자") -> pd.DataFrame:
    if df.empty:
        return df
    start = pd.to_datetime(df.index.min(), format="%Y%m%d")
    end = pd.to_datetime(df.index.max(), format="%Y%m%d")
    news = crawl_naver_news_titles(query, start, end, max_pages=80)
    if news.empty:
        print("[WARN] 뉴스 검색 결과 없음 → news_count/news_sentiment 생략")
        return df

    pipe = build_sentiment_pipeline() if ADD_NEWS else None
    if pipe is None:
        print("[INFO] 감성모델 없음 → news_count만 추가")
        daily = news.groupby(news["date"].dt.strftime("%Y%m%d")).size().rename("news_count")
        out = df.join(daily, how="left")
        out["news_count"] = out["news_count"].fillna(0).astype(int)
        return out

    scores = []
    batch, batch_dates = [], []
    for _, row in news.iterrows():
        batch.append(str(row["title"])[:256])
        batch_dates.append(row["date"].strftime("%Y%m%d"))
        if len(batch) == 32:
            res = pipe(batch, truncation=True)
            scores += [_score_from_label(r["label"]) for r in res]
            batch, batch_dates = [], []
    if batch:
        res = pipe(batch, truncation=True)
        scores += [_score_from_label(r["label"]) for r in res]

    news = news.iloc[:len(scores)].copy()
    news["score"] = scores
    agg = news.groupby(news["date"].dt.strftime("%Y%m%d")).agg(
        news_count=("title","count"),
        news_sentiment=("score","mean"),
    )
    out = df.join(agg, how="left")
    out["news_count"] = out["news_count"].fillna(0).astype(int)
    out["news_sentiment"] = out["news_sentiment"].fillna(0.0)
    return out


# ---------- 4) 기술지표 ----------
def add_tech(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "종가" not in df.columns:
        return df
    out = df.copy()
    out["ret_1d"] = out["종가"].pct_change()
    out["logret_1d"] = out["종가"].apply(lambda x: math.log(x) if isinstance(x,(int,float)) and x>0 else None)
    out["logret_1d"] = out["logret_1d"] - out["logret_1d"].shift(1)
    for w in (5, 20, 60, 120):
        out[f"ma{w}"] = out["종가"].rolling(w).mean()
        out[f"vol_{w}"] = out["ret_1d"].rolling(w).std()
    return out


# ---------- 메인 ----------
def main() -> None:
    ensure_packages()

    # 구글 드라이브 마운트 시도
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        print("[INFO] Google Drive mounted.")
    except Exception as e:
        print(f"[WARN] Drive mount failed: {e}")
        global SAVE_DIR
        SAVE_DIR = Path("/content/krx_data")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outdir = SAVE_DIR / ts
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Naver OHLCV: code={CODE}, target_days={TARGET_DAYS}")
    df = fetch_naver_ohlcv(CODE, TARGET_DAYS)
    if df.empty:
        print("[ERROR] 네이버 OHLCV 수집 실패."); return

    # 매크로 지표 붙이기
    df = add_macro_features(df)

    # 뉴스(개수/감성) 붙이기
    if ADD_NEWS:
        df = add_news_features(df, query="삼성전자")
    else:
        print("[INFO] NEWS=0 → 뉴스 피처 생략")

    # 기술지표
    if ADD_TECH:
        df = add_tech(df)

    # 저장
    csv_path = outdir / f"{CODE}_1000d_merged.csv"
    df.to_csv(csv_path, encoding="utf-8-sig")
    with (outdir / "preview.json").open("w", encoding="utf-8") as f:
        json.dump(df.tail(5).reset_index().to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    print(f"[INFO] Saved CSV: {csv_path.resolve()}")
    print(f"[INFO] Rows={len(df):,}, Cols={len(df.columns):,}")
    print(f"[INFO] Folder: {outdir.resolve()}")

if __name__ == "__main__":
    main()
