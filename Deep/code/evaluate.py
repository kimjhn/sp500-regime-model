"""Classification, calibration, and prediction-stability metrics.

Beyond accuracy, the deployment story rests on (a) catching risk-off regimes
(Bear/Weak Bear recall + risk-off AUROC vs the HMM's 0.88), (b) trustworthy
probabilities (ECE/Brier, before vs after temperature scaling), and (c) stable,
non-flip-flopping predictions that detect regime changes with low lag.
"""
from __future__ import annotations
import numpy as np
import torch
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, balanced_accuracy_score, roc_auc_score)

import config as C


def collect_predictions(model, loader, device, temperature=1.0):
    """Return (probs (N,5), preds (N,), y (N,)) with temperature applied."""
    model.eval()
    logits, ys = [], []
    with torch.no_grad():
        for x1, x2, y, _ in loader:
            logits.append(model(x1.to(device), x2.to(device)).cpu())
            ys.append(y)
    logits = torch.cat(logits) / temperature
    probs = torch.softmax(logits, dim=1).numpy()
    return probs, probs.argmax(1), torch.cat(ys).numpy()


def classification_metrics(y, preds, verbose=True):
    names = [C.REGIME_NAMES[i] for i in range(C.N_REGIMES)]
    rep = classification_report(y, preds, target_names=names,
                                labels=range(C.N_REGIMES), digits=3, zero_division=0)
    cm = confusion_matrix(y, preds, labels=range(C.N_REGIMES))
    out = {"macro_f1": f1_score(y, preds, average="macro", labels=range(C.N_REGIMES),
                               zero_division=0),
           "balanced_acc": balanced_accuracy_score(y, preds),
           "confusion_matrix": cm}
    if verbose:
        print(rep)
        print("confusion matrix (rows=true, cols=pred):\n", cm)
        print(f"macro-F1={out['macro_f1']:.3f}  balanced-acc={out['balanced_acc']:.3f}")
    return out


def risk_off_auroc(y, probs):
    """AUROC for detecting risk-off (Bear+Weak Bear) one-vs-rest, plus macro OvR."""
    p_risk = probs[:, list(C.RISK_OFF_REGIMES)].sum(1)
    y_risk = np.isin(y, C.RISK_OFF_REGIMES).astype(int)
    out = {"risk_off_auroc": roc_auc_score(y_risk, p_risk)}
    try:
        out["macro_ovr_auroc"] = roc_auc_score(
            y, probs, multi_class="ovr", average="macro", labels=range(C.N_REGIMES))
    except ValueError:
        out["macro_ovr_auroc"] = float("nan")
    return out


def calibration_metrics(y, probs, n_bins=10):
    conf = probs.max(1)
    correct = (probs.argmax(1) == y).astype(float)
    onehot = np.eye(C.N_REGIMES)[y]
    ece, edges = 0.0, np.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += abs(correct[m].mean() - conf[m].mean()) * m.mean()
    brier = ((probs - onehot) ** 2).sum(1).mean()
    return {"ECE": float(ece), "Brier": float(brier)}


def reliability_curve(y, probs, n_bins=10):
    conf = probs.max(1)
    correct = (probs.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            xs.append(conf[m].mean()); ys.append(correct[m].mean())
    return np.array(xs), np.array(ys)


def prediction_stability(preds, trading_days=C.TRADING_DAYS):
    switches = int((np.diff(preds) != 0).sum())
    runs = np.diff(np.flatnonzero(np.concatenate(([1], np.diff(preds) != 0, [1]))))
    return {"n_switches": switches,
            "switches_per_year": switches / (len(preds) / trading_days),
            "avg_run_length": float(runs.mean())}


def transition_lag(y_true, preds, window=C.HORIZON):
    """Mean delay (days) before the prediction reflects a true regime change.
    Censored at `window`; reports the fraction not detected within the window."""
    changes = np.flatnonzero(np.diff(y_true) != 0) + 1
    lags, missed = [], 0
    for t in changes:
        new = y_true[t]
        future = preds[t: min(t + window + 1, len(preds))]
        hit = np.flatnonzero(future == new)
        if hit.size:
            lags.append(int(hit[0]))
        else:
            missed += 1; lags.append(window)
    return {"mean_transition_lag": float(np.mean(lags)) if lags else float("nan"),
            "missed_frac": missed / len(changes) if len(changes) else float("nan"),
            "n_transitions": int(len(changes))}


def full_report(y, probs, preds, label=""):
    print(f"\n===== classification report: {label} =====")
    out = classification_metrics(y, preds)
    out.update(risk_off_auroc(y, probs))
    out.update(calibration_metrics(y, probs))
    out.update(prediction_stability(preds))
    out.update(transition_lag(y, preds))
    print(f"risk-off AUROC={out['risk_off_auroc']:.3f}  macro-OvR AUROC={out['macro_ovr_auroc']:.3f}")
    print(f"ECE={out['ECE']:.3f}  Brier={out['Brier']:.3f}  "
          f"switches/yr={out['switches_per_year']:.1f}  "
          f"transition_lag={out['mean_transition_lag']:.1f}d")
    return out
