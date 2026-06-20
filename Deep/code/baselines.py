"""Non-deep baselines. The two-track model must beat these on the metrics that
matter (risk-off recall, backtest Calmar) to justify its complexity; if it does
not, that is an honest finding worth reporting.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression

import config as C
from dataset import Data


def _onehot(preds):
    return np.eye(C.N_REGIMES)[preds]


def majority(data: Data):
    cls = int(np.bincount(data.train.y, minlength=C.N_REGIMES).argmax())
    preds = np.full(len(data.test.y), cls)
    return preds, _onehot(preds)


def lagged_label(data: Data, horizon=C.HORIZON):
    """Persistence: predict the most recent *knowable* regime (the label whose
    forward window already fully realised, i.e. from `horizon` days ago)."""
    pos = data.df.index.get_indexer(data.test.dates)
    labels = data.df[C.LABEL_COL].values
    preds = labels[pos - horizon].astype(int)
    return preds, _onehot(preds)


def _flatten(split):
    return np.concatenate([split.X1.reshape(len(split.X1), -1), split.X2], axis=1)


def logistic(data: Data):
    Xtr, Xte = _flatten(data.train), _flatten(data.test)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, data.train.y)
    return clf.predict(Xte), clf.predict_proba(Xte)


def xgboost(data: Data):
    from xgboost import XGBClassifier
    Xtr, Xte = _flatten(data.train), _flatten(data.test)
    sample_w = data.class_weights[data.train.y]
    clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        objective="multi:softprob", num_class=C.N_REGIMES,
                        eval_metric="mlogloss", random_state=C.SEED)
    clf.fit(Xtr, data.train.y, sample_weight=sample_w)
    return clf.predict(Xte), clf.predict_proba(Xte)


ALL = {"majority": majority, "lagged": lagged_label, "logistic": logistic, "xgboost": xgboost}
