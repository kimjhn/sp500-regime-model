"""Serving inference for the regime model + investor personalization.

Loads the seed-ensemble artifacts saved by the notebook (Section 8), builds feature
windows with the SAVED scalers (transform-only -- never refit), and returns averaged
regime probabilities + a recommended exposure.

Personalization is post-model: the model emits market-state probabilities (identical
for everyone); an investor's risk profile only changes the EXPOSURE LADDER applied
afterwards (``recommended_exposure = probs @ ladder``). So nothing here retrains the
model -- one model serves all profiles.

Live data refresh: the latest feature row comes from ``sp500_regime_dataset_final.csv``,
which the STEP 1 notebook regenerates with point-in-time integrity (ALFRED first_release
+ publication-lag + ffill). Re-run STEP 1 to advance the as-of date, then serve here.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import config as C


# --------------------------------------------------------------------------- #
# Investor personalization (post-model)                                       #
# --------------------------------------------------------------------------- #

# Base shapes by aggressiveness (Bear, Weak Bear, Neutral, Weak Bull, Bull).
PRESET_LADDERS = {
    "conservative": [0.00, 0.20, 0.40, 0.60, 0.80],   # 자본 보존
    "balanced":     [0.00, 0.40, 0.80, 1.00, 1.00],   # 시장 추종 + 방어
    "growth":       [0.00, 0.50, 0.90, 1.10, 1.30],   # 상승 적극 (레버리지 필요)
}
LEVERAGE_CAP = {"A": 1.0, "B": 1.3, "C": 2.0}   # Q1: 상한
SHORT_FLOOR = {"A": 0.0, "B": -0.5, "C": -1.0}  # Q2: 하한


def preset_from_score(score: int) -> str:
    return "conservative" if score <= 2 else "balanced" if score <= 5 else "growth"


def profile_to_ladder(answers: dict) -> np.ndarray:
    """Map questionnaire answers -> length-5 exposure ladder (Bear..Bull).

    answers keys (see investor_profile_questionnaire.md):
      score        : int 0..8  (Q3/Q4/Q7/Q8 공격성 합) -> 프리셋 선택
      q1_leverage  : 'A'|'B'|'C' (레버리지) -> 상한(Bull) 캡
      q2_short     : 'A'|'B'|'C' (공매도)   -> 하한(Bear) 플로어
    """
    ladder = np.array(PRESET_LADDERS[preset_from_score(int(answers.get("score", 4)))],
                      dtype=float)

    # Q1: 레버리지 미허용이면 100% 초과를 잘라내고, 허용이면 강세장 상한을 올린다.
    cap = LEVERAGE_CAP.get(answers.get("q1_leverage", "A"), 1.0)
    ladder = np.minimum(ladder, cap)
    if cap > 1.0:
        ladder[4] = cap

    # Q2: 공매도 허용이면 하락 구간을 음수로 (방어 -> 수익 추구).
    floor = SHORT_FLOOR.get(answers.get("q2_short", "A"), 0.0)
    if floor < 0:
        ladder[0] = floor
        ladder[1] = min(ladder[1], floor / 2)
    return ladder


def default_ladder() -> np.ndarray:
    return np.array([C.EXPOSURE_LADDER[i] for i in range(C.N_REGIMES)], dtype=float)


# --------------------------------------------------------------------------- #
# Ensemble loading + inference                                                #
# --------------------------------------------------------------------------- #

@dataclass
class Ensemble:
    models: list          # list[RegimePredictor] in eval mode
    temperatures: list    # per-seed temperature
    scaler_t1: object
    scaler_t2: object
    meta: dict


def load_ensemble(artifact_dir=None) -> Ensemble:
    """Load the saved model(s). Supports both the seed-ensemble format
    (ensemble_seed{i}.pt + per-seed temperatures) and the single-model format
    (best_model.pt + scalar temperature) saved by train.save_artifacts."""
    import torch
    import joblib
    from model import RegimePredictor

    d = Path(artifact_dir or C.ARTIFACT_DIR)
    meta = json.loads((d / "config.json").read_text(encoding="utf-8"))
    scaler_t1 = joblib.load(d / "scaler_t1.pkl")
    scaler_t2 = joblib.load(d / "scaler_t2.pkl")

    mk = meta.get("model_kwargs", {})
    kw = dict(gru_hidden=mk.get("gru_hidden", 64), gru_layers=mk.get("gru_layers", 2),
              fusion_dim=mk.get("fusion_dim", 32),
              dropout=mk.get("dropout", meta.get("hparams", {}).get("dropout", 0.3)))

    def _load(fname):
        m = RegimePredictor(**kw)
        m.load_state_dict(torch.load(d / fname, map_location="cpu"))
        m.eval()
        return m

    models, temps = [], []
    if "ensemble_seeds" in meta:                      # 시드 앙상블 형식
        for sd in meta["ensemble_seeds"]:
            models.append(_load(f"ensemble_seed{sd}.pt"))
            temps.append(float(meta["temperatures"][str(sd)]))
    else:                                             # 단일 모델 형식 (best_model.pt)
        models.append(_load("best_model.pt"))
        temps.append(float(meta.get("temperature", 1.0)))
    return Ensemble(models, temps, scaler_t1, scaler_t2, meta)


def predict_proba(ens: Ensemble, X1, X2) -> np.ndarray:
    """X1: (seq,nT1) or (B,seq,nT1); X2: (nT2,) or (B,nT2). Returns (B,n_regimes)
    averaged over seeds (each seed's softmax is temperature-scaled first)."""
    import torch
    X1 = np.asarray(X1, np.float32); X2 = np.asarray(X2, np.float32)
    if X1.ndim == 2: X1 = X1[None]
    if X2.ndim == 1: X2 = X2[None]
    t1, t2 = torch.from_numpy(X1), torch.from_numpy(X2)
    acc = []
    with torch.no_grad():
        for m, T in zip(ens.models, ens.temperatures):
            acc.append(torch.softmax(m(t1, t2) / T, dim=1).numpy())
    return np.mean(acc, axis=0)


# --------------------------------------------------------------------------- #
# Feature windows (from the maintained, point-in-time CSV)                    #
# --------------------------------------------------------------------------- #

def load_features(path=None, meta=None) -> pd.DataFrame:
    """Load the dataset keeping the most recent rows (which have no future label yet).

    Unlike dataset.load_dataframe (which drops rows missing the forward label), serving
    must keep today's row, so we only require the FEATURE + close columns to be present.
    """
    feats = (meta["track1_feats"] + meta["track2_feats"]) if meta else (C.TRACK1_FEATS + C.TRACK2_FEATS)
    df = pd.read_csv(path or C.DATA_CSV, index_col=0, parse_dates=True).sort_index()
    return df.dropna(subset=feats + [C.CLOSE_COL]).copy()


def _scaled_arrays(ens: Ensemble, df: pd.DataFrame):
    arr1 = ens.scaler_t1.transform(df[ens.meta["track1_feats"]].values)
    arr2 = ens.scaler_t2.transform(df[ens.meta["track2_feats"]].values)
    return arr1, arr2


def latest_window(ens: Ensemble, df: pd.DataFrame):
    """Most recent (X1 (seq,nT1), X2 (nT2,), as_of_date)."""
    seq = ens.meta["seq_len"]
    arr1, arr2 = _scaled_arrays(ens, df)
    return arr1[-seq:], arr2[-1], df.index[-1]


def predict_series(ens: Ensemble, df: pd.DataFrame, start=None):
    """Ensemble probs for every row with a full lookback window.
    Returns (dates, close, probs (N,5), preds (N,)). `start` clips to a date."""
    seq = ens.meta["seq_len"]
    arr1, arr2 = _scaled_arrays(ens, df)
    idx = np.arange(seq - 1, len(df))
    X1 = np.stack([arr1[i - seq + 1: i + 1] for i in idx]).astype(np.float32)
    X2 = arr2[idx].astype(np.float32)
    probs = predict_proba(ens, X1, X2)
    dates, close = df.index[idx], df[C.CLOSE_COL].values[idx]
    if start is not None:
        m = dates >= pd.Timestamp(start)
        dates, close, probs = dates[m], close[m], probs[m]
    return dates, close, probs, probs.argmax(1)


# --------------------------------------------------------------------------- #
# Recommendation                                                              #
# --------------------------------------------------------------------------- #

def recommend(probs, answers: dict | None = None, ladder=None) -> dict:
    """probs: (5,) or (1,5). Returns regime call + risk-off prob + recommended exposure."""
    p = np.asarray(probs, float).reshape(-1)
    if ladder is None:
        ladder = profile_to_ladder(answers) if answers else default_ladder()
    ladder = np.asarray(ladder, float)
    top = int(p.argmax())
    return {
        "probs": p.tolist(),
        "regime": top,
        "regime_name": C.REGIME_NAMES[top],
        "risk_off_prob": float(p[list(C.RISK_OFF_REGIMES)].sum()),
        "recommended_exposure": float(p @ ladder),
        "ladder": ladder.tolist(),
    }


def predict_latest(answers: dict | None = None, artifact_dir=None, csv_path=None) -> dict:
    """End-to-end: load ensemble + latest feature row -> recommendation dict."""
    ens = load_ensemble(artifact_dir)
    df = load_features(path=csv_path, meta=ens.meta)
    X1, X2, as_of = latest_window(ens, df)
    out = recommend(predict_proba(ens, X1, X2)[0], answers=answers)
    out["as_of"] = str(pd.Timestamp(as_of).date())
    return out


if __name__ == "__main__":
    # personalization sanity check (no model needed)
    for label, ans in [("보수형", {"score": 1}), ("균형형", {"score": 4}),
                       ("성장형(레버리지)", {"score": 7, "q1_leverage": "C"}),
                       ("롱숏형", {"score": 4, "q2_short": "C"})]:
        print(f"{label:16s} ladder = {np.round(profile_to_ladder(ans), 2)}")
