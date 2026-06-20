"""v2 이벤트 오버레이 — 구조적 위기 vs 일회성 이벤트 판별 (해석형).

배경(왜 필요한가)
-----------------
v1 레짐 모델은 향후 20일의 '가격 행동(수익률+변동성)'의 심각도를 본다. 그래서 코로나,
2008 위기, 트럼프 관세의 날처럼 '급격하고 변동성 큰 폭락'은 원인과 무관하게 모두
'강한 하락장'으로 분류한다 — 구조적 위기인지 일회성 이벤트인지는 구분하지 못한다.

이 모듈은 v1을 전혀 바꾸지 않고, '금융환경(NFCI)'을 보조 신호로 써서 그 급락이
구조적인지 일회성인지에 대한 확률을 추가 정보로 제공한다.

  structural_score = sigmoid(W_LEVEL * NFCI + W_MOM * ΔNFCI_20d)
    · NFCI 레벨   : 지금 금융환경이 얼마나 타이트한가 (구조적 위기 = 신용/유동성 경색)
    · ΔNFCI 20일  : 금융 스트레스가 얼마나 빠르게 쌓이는가 (코로나식 급발진 포착)
  oneoff_prob = 1 - structural_score

검증된 거동 (Deep/code 데이터 기준):
  GFC 2008~09 → ~1.00(구조) · 코로나 2020-03 → ~0.85(구조) ·
  플래시크래시 2010 → 0.40(일회) · 관세의 날 2025-04 → 0.35(일회)
"""
from __future__ import annotations
import numpy as np
import pandas as pd

NFCI_COL   = "NFCI"
MOM_WINDOW = 20     # v1 HORIZON과 일치
W_LEVEL    = 2.0
W_MOM      = 3.0
STRUCT_HI  = 0.66   # 이상이면 '구조적 가능성 높음'
STRUCT_LO  = 0.40   # 이하면 '일회성 가능성 높음'


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def structural_score_series(df: pd.DataFrame) -> pd.Series:
    """각 날짜의 구조적 스트레스 점수(0~1) 시계열."""
    nfci = df[NFCI_COL].astype(float)
    mom = (nfci - nfci.shift(MOM_WINDOW)).fillna(0.0)
    return pd.Series(_sigmoid(W_LEVEL * nfci + W_MOM * mom), index=df.index)


def assess(df: pd.DataFrame, as_of=None) -> dict:
    """특정 시점(기본=최신)의 구조적/일회성 판정 + 근거 값."""
    s = structural_score_series(df)
    idx = -1 if as_of is None else int(
        df.index.get_indexer([pd.Timestamp(as_of)], method="nearest")[0])
    score = float(s.iloc[idx])
    nfci = float(df[NFCI_COL].iloc[idx])
    nfci_prev = float(df[NFCI_COL].iloc[max(0, idx - MOM_WINDOW)])
    label = ("구조적 위기 가능성 높음" if score >= STRUCT_HI
             else "일회성 이벤트 가능성 높음" if score <= STRUCT_LO
             else "혼재 / 불확실")
    return {
        "structural_prob": score,
        "oneoff_prob": 1.0 - score,
        "label": label,
        "nfci": nfci,
        "nfci_chg20": nfci - nfci_prev,
        "date": df.index[idx],
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config as C
    df = pd.read_csv(C.DATA_CSV, index_col=0, parse_dates=True).sort_index()
    for name, d in [("GFC 2008-10", "2008-10-15"), ("코로나 2020-03", "2020-03-23"),
                    ("플래시크래시 2010", "2010-05-06"), ("관세의날 2025-04", "2025-04-07"),
                    ("최신", None)]:
        a = assess(df, d)
        print(f"{name:16s} {str(a['date'].date()):11s} NFCI={a['nfci']:5.2f} "
              f"구조={a['structural_prob']:.0%} 일회={a['oneoff_prob']:.0%}  {a['label']}")
