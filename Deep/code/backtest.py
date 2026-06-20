"""Financial validation: regime -> exposure backtests with realistic execution,
defensive-first performance metrics, deflated Sharpe (overfitting guard), and
sub-period stress analysis.

Execution model: the regime forecast made after the close of day t sets the target
equity weight w_t held into day t+1 (next-day execution -> no look-ahead). Transaction
cost is charged on |w_t - w_{t-1}|. Cash earns 0%.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis

import config as C

EULER = 0.5772156649015329


def daily_asset_returns(close: np.ndarray) -> np.ndarray:
    r = np.zeros_like(close, dtype=float)
    r[1:] = close[1:] / close[:-1] - 1.0
    return r


def _ladder_array(ladder=None) -> np.ndarray:
    """Length-N_REGIMES exposure vector; defaults to the config ladder.
    `ladder` may be a dict {regime: weight} or a sequence indexed by regime."""
    if ladder is None:
        ladder = C.EXPOSURE_LADDER
    if isinstance(ladder, dict):
        return np.array([ladder[i] for i in range(C.N_REGIMES)], dtype=float)
    return np.asarray(ladder, dtype=float)


def exposure_from_regime(preds: np.ndarray, ladder=None) -> np.ndarray:
    lad = _ladder_array(ladder)
    return lad[np.asarray(preds, dtype=int)]


def exposure_from_probs(probs: np.ndarray, ladder=None) -> np.ndarray:
    return probs @ _ladder_array(ladder)  # expected exposure under predicted distribution


def strategy_pnl(weights: np.ndarray, asset_r: np.ndarray,
                 cost=C.COST_PER_TURNOVER) -> np.ndarray:
    w_prev = np.concatenate([[0.0], weights[:-1]])     # weight held into day i
    w_prev2 = np.concatenate([[0.0, 0.0], weights[:-2]])
    turnover = np.abs(w_prev - w_prev2)
    pnl = w_prev * asset_r - cost * turnover
    pnl[0] = 0.0
    return pnl


def perf_metrics(pnl: np.ndarray, weights: np.ndarray | None = None,
                 periods=C.TRADING_DAYS) -> dict:
    pnl = np.asarray(pnl, float)
    eq = np.cumprod(1 + pnl)
    years = len(pnl) / periods
    sd = pnl.std(ddof=1)
    downside = pnl[pnl < 0].std(ddof=1) if (pnl < 0).any() else np.nan
    dd = eq / np.maximum.accumulate(eq) - 1.0
    mdd = dd.min()
    cagr = eq[-1] ** (1 / years) - 1
    out = {
        "CAGR": cagr,
        "Sharpe": pnl.mean() / sd * np.sqrt(periods) if sd > 0 else np.nan,
        "Sortino": pnl.mean() / downside * np.sqrt(periods) if downside and downside > 0 else np.nan,
        "vol": sd * np.sqrt(periods),
        "MDD": mdd,
        "Calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "cum_return": eq[-1] - 1,
    }
    if weights is not None:
        turn = np.abs(np.diff(np.concatenate([[0.0], weights])))
        out["pct_invested"] = float((weights > 0).mean())
        out["turnover_per_yr"] = float(turn.sum() / years)
    return out


def probabilistic_sharpe_ratio(pnl: np.ndarray, sr_benchmark_ann=0.0,
                               periods=C.TRADING_DAYS) -> float:
    """P(true Sharpe > benchmark), correcting for skew/kurtosis and sample size."""
    pnl = np.asarray(pnl, float)
    sr = pnl.mean() / pnl.std(ddof=1)                 # per-period
    sr0 = sr_benchmark_ann / np.sqrt(periods)
    n, g3, g4 = len(pnl), skew(pnl), kurtosis(pnl, fisher=False)
    denom = np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    return float(norm.cdf((sr - sr0) * np.sqrt(n - 1) / denom))


def deflated_sharpe_ratio(pnl: np.ndarray, n_trials: int, sr_trials_std,
                          periods=C.TRADING_DAYS) -> float:
    """DSR = PSR against the Sharpe expected from the best of `n_trials` configs.

    sr_trials_std: std of the (annualised) Sharpe ratios across the configs tried.
    """
    sr0_std = sr_trials_std / np.sqrt(periods)
    z1 = norm.ppf(1 - 1.0 / n_trials)
    z2 = norm.ppf(1 - 1.0 / (n_trials * np.e))
    sr0_expected_max = sr0_std * ((1 - EULER) * z1 + EULER * z2)  # per-period
    return probabilistic_sharpe_ratio(pnl, sr0_expected_max * np.sqrt(periods), periods)


def run_backtest(dates, close, preds, probs, y_true, ladder=None) -> pd.DataFrame:
    """Return a tidy DataFrame of daily pnl/weights for every strategy & benchmark.
    `ladder` (dict or sequence) personalizes the exposure mapping; None = config default."""
    asset_r = daily_asset_returns(np.asarray(close, float))
    strategies = {
        "Defensive (long/cash)": exposure_from_regime(preds, ladder),
        "Prob-weighted": exposure_from_probs(probs, ladder),
        "Buy&Hold": np.ones(len(preds)),
        "Oracle": exposure_from_regime(y_true, ladder),
    }
    cols = {"close": close, "asset_r": asset_r}
    for name, w in strategies.items():
        cols[f"w::{name}"] = w
        cols[f"pnl::{name}"] = strategy_pnl(w, asset_r)
    return pd.DataFrame(cols, index=pd.DatetimeIndex(dates))


def summarize(bt: pd.DataFrame, n_trials=1, sr_trials_std=0.0) -> pd.DataFrame:
    rows = {}
    for name in [c[5:] for c in bt.columns if c.startswith("pnl::")]:
        pnl = bt[f"pnl::{name}"].values
        w = bt[f"w::{name}"].values
        m = perf_metrics(pnl, w)
        m["PSR(>0)"] = probabilistic_sharpe_ratio(pnl)
        if n_trials > 1 and name in ("Defensive (long/cash)", "Prob-weighted"):
            m["DSR"] = deflated_sharpe_ratio(pnl, n_trials, sr_trials_std)
        rows[name] = m
    return pd.DataFrame(rows).T


def subperiod_analysis(bt: pd.DataFrame, periods: dict, strategy="Defensive (long/cash)"):
    """Metrics for a strategy and Buy&Hold over named date ranges."""
    out = {}
    for label, (lo, hi) in periods.items():
        seg = bt.loc[lo:hi]
        if len(seg) < 5:
            continue
        out[label] = {
            f"{strategy} MDD": perf_metrics(seg[f"pnl::{strategy}"].values)["MDD"],
            "Buy&Hold MDD": perf_metrics(seg["pnl::Buy&Hold"].values)["MDD"],
            f"{strategy} ret": (1 + seg[f"pnl::{strategy}"]).prod() - 1,
            "Buy&Hold ret": (1 + seg["pnl::Buy&Hold"]).prod() - 1,
        }
    return pd.DataFrame(out).T


SUBPERIODS = {
    "COVID crash 2020": ("2020-02-01", "2020-04-30"),
    "2022 bear": ("2022-01-01", "2022-10-31"),
    "2023-24 bull": ("2023-01-01", "2024-12-31"),
}
