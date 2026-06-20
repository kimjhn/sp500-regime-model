"""대시보드 데이터 자동 갱신 스크립트.

FinanceDataReader(SPY) + FRED ALFRED(매크로) → 동일 스키마 CSV 갱신.
HMM 라벨은 기존 CSV에서 복사 보존(새 날짜는 NaN; 대시보드는 라벨 안 읽음).

CI 환경 변수: FRED_API_KEY (GitHub Secret으로 주입).
실행 시간 약 30~60초. 변경된 행이 있을 때만 호출자(워크플로우)가 커밋.
"""
from __future__ import annotations
import os
import sys
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from fredapi import Fred
import FinanceDataReader as fdr

# === 설정 ===
START_DATE = "1999-01-01"
END_DATE = datetime.datetime.today().strftime("%Y-%m-%d")
OUTPUT_PATH = Path(__file__).resolve().parent / "Deep" / "data" / "sp500_regime_dataset_final.csv"


# === 1. SPY → Track1 피처 + close ===================================================
def load_sp500(start: str, end: str) -> pd.DataFrame:
    raw = fdr.DataReader("SPY", start, end)
    px = raw["Adj Close"]
    df = pd.DataFrame(index=raw.index)

    df["log_return"] = np.log(px / px.shift(1))
    df["vol_5d"] = df["log_return"].rolling(5).std()
    df["vol_20d"] = df["log_return"].rolling(20).std()
    df["vol_60d"] = df["log_return"].rolling(60).std()
    df["log_vol_5d"] = np.log(df["vol_5d"])
    df["log_vol_20d"] = np.log(df["vol_20d"])
    df["log_vol_60d"] = np.log(df["vol_60d"])
    df["vol_ratio_5_20"] = df["vol_5d"] / df["vol_20d"]
    df["vol_ratio_5_60"] = df["vol_5d"] / df["vol_60d"]

    for w in [20, 60, 200]:
        df[f"sma_gap_{w}d"] = px / px.rolling(w).mean() - 1

    delta = px.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    rs = gain.ewm(alpha=1/14, adjust=False).mean() / loss.ewm(alpha=1/14, adjust=False).mean()
    df["rsi_14"] = 100 - (100 / (1 + rs))

    ema12 = px.ewm(span=12, adjust=False).mean()
    ema26 = px.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    df["macd_pct"] = macd / px
    df["macd_hist"] = df["macd_pct"] - macd.ewm(span=9, adjust=False).mean() / px

    df["close"] = px
    return df.dropna()


# === 2. FRED 매크로 → Track2 피처 (first_release + 발표지연 offset) ====================
def _safe_fred(fred: Fred, ticker: str) -> pd.Series:
    """수정 전 초기 공시치(ALFRED) 우선, 실패 시 표준 시리즈로 폴백."""
    try:
        return fred.get_series_first_release(ticker)
    except Exception:
        return fred.get_series(ticker)


def _safe_reindex(data, target_index: pd.DatetimeIndex):
    """합집합 인덱스에서 ffill 후 평일 추출 — 주말 발표 NaN 방지."""
    if isinstance(data, pd.Series):
        data = pd.to_numeric(data, errors="coerce")
    else:
        data = data.apply(pd.to_numeric, errors="coerce")
    combined = data.index.union(target_index)
    return data.reindex(combined).ffill().loc[target_index]


def load_macro(fred: Fred, start: str, end: str) -> pd.DataFrame:
    bdays = pd.bdate_range(start=start, end=end)
    out = {}

    m2 = pd.to_numeric(_safe_fred(fred, "M2SL"), errors="coerce")
    m2_yoy = m2.pct_change(12) * 100
    m2_yoy.index += pd.DateOffset(days=25)
    out["M2_YOY"] = _safe_reindex(m2_yoy, bdays)

    nfci = fred.get_series("NFCI")
    nfci.index += pd.DateOffset(days=5)
    out["NFCI"] = _safe_reindex(nfci, bdays)

    ys = fred.get_series("T10Y3M")
    out["YIELD_SPREAD"] = _safe_reindex(ys, bdays)

    cpi = pd.to_numeric(_safe_fred(fred, "CPIAUCSL"), errors="coerce")
    cpi_yoy = cpi.pct_change(12) * 100
    cpi_yoy.index += pd.DateOffset(days=15)
    out["CPI_YOY"] = _safe_reindex(cpi_yoy, bdays)

    de = _safe_fred(fred, "NCBCMDPMVCE")
    de.index += pd.DateOffset(days=75)
    out["DEBT_EQUITY"] = _safe_reindex(de, bdays)

    dep = _safe_fred(fred, "DPSACBW027SBOG")
    loans = _safe_fred(fred, "TOTLL")
    ltdr = pd.DataFrame({"deposit": dep, "loans": loans})
    ltdr.index += pd.DateOffset(days=9)
    ltdr = _safe_reindex(ltdr, bdays)
    out["LOAN_DEPOSIT"] = ltdr["loans"] / ltdr["deposit"]

    return pd.concat(out, axis=1)


# === 3. 라벨 보존 (기존 CSV에서 복사) ================================================
def preserve_old_labels(old_csv: Path) -> pd.DataFrame | None:
    """기존 CSV에 HMM 라벨이 있으면 dict로 반환. 없으면 None."""
    if not old_csv.exists():
        return None
    old = pd.read_csv(old_csv, index_col=0, parse_dates=True)
    label_cols = [c for c in old.columns if c.startswith("regime_")]
    if not label_cols:
        return None
    return old[label_cols].copy()


# === 4. 메인 ==========================================================================
def main():
    fred_key = os.environ.get("FRED_API_KEY")
    if not fred_key:
        sys.exit("ERROR: FRED_API_KEY 환경 변수가 설정되지 않았습니다.")
    fred = Fred(api_key=fred_key)

    print(f"[1/4] SPY 다운로드 {START_DATE} ~ {END_DATE}")
    stock_df = load_sp500(START_DATE, END_DATE)
    print(f"      {stock_df.index.min().date()} ~ {stock_df.index.max().date()}, {len(stock_df)}행")

    print(f"[2/4] FRED 매크로 다운로드 (first_release + 발표지연 offset)")
    macro_df = load_macro(fred, START_DATE, END_DATE)
    print(f"      {len(macro_df)}행")

    print(f"[3/4] 병합 + 비정상 피처 차분")
    common = stock_df.index.intersection(macro_df.index)
    total = pd.concat([stock_df.loc[common], macro_df.loc[common]], axis=1).dropna()
    total["YIELD_SPREAD_diff"] = total["YIELD_SPREAD"].diff()
    total["DEBT_EQUITY_diff"] = total["DEBT_EQUITY"].diff()
    total = total.dropna()
    print(f"      병합 후 {len(total)}행, 마지막 {total.index[-1].date()}")

    print(f"[4/4] 기존 HMM 라벨 보존 + CSV 저장")
    old_labels = preserve_old_labels(OUTPUT_PATH)
    if old_labels is not None:
        total = total.join(old_labels, how="left")
        n_labeled = total["regime_label"].notna().sum() if "regime_label" in total else 0
        print(f"      기존 라벨 보존: {n_labeled}/{len(total)}행")
    else:
        print(f"      기존 라벨 없음(또는 첫 실행) — 라벨 컬럼 생략")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    total.to_csv(OUTPUT_PATH)
    print(f"\n저장 완료: {OUTPUT_PATH}")
    print(f"   기간 {total.index.min().date()} ~ {total.index.max().date()}")
    print(f"   행수 {len(total)}, 컬럼 {len(total.columns)}")


if __name__ == "__main__":
    main()
