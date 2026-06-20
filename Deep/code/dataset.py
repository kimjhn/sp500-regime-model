"""Data pipeline: load, chronological split with embargo, leakage-safe scaling,
and 60-day sequence construction.

Leakage model
-------------
The label y[t] is the HMM regime of the FORWARD window (t+1 .. t+HORIZON), so a
training sample whose forward window reaches into the validation period would leak
future information across the split boundary. We therefore drop ("embargo") the
last HORIZON decision rows of each split. Feature lookback that reaches back across
a boundary is fine -- past observations are always available at inference time.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight

import config as C


@dataclass
class Split:
    X1: np.ndarray      # (n, seq_len, n_track1)
    X2: np.ndarray      # (n, n_track2)
    y: np.ndarray       # (n,) int regime label
    y_soft: np.ndarray  # (n, 5) HMM posterior (soft target)
    dates: pd.DatetimeIndex  # (n,) decision date of each sample
    close: np.ndarray   # (n,) close on the decision date


@dataclass
class Data:
    train: Split
    val: Split
    test: Split
    scaler_t1: RobustScaler
    scaler_t2: RobustScaler
    class_weights: np.ndarray  # (5,)
    df: pd.DataFrame           # full cleaned frame (for backtest close lookup)


def load_dataframe(path=None) -> pd.DataFrame:
    path = path or C.DATA_CSV
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    used = C.TRACK1_FEATS + C.TRACK2_FEATS + [C.LABEL_COL, C.CLOSE_COL] + C.SOFT_LABEL_COLS
    before = len(df)
    df = df.dropna(subset=used).copy()
    dropped = before - len(df)
    if dropped:
        print(f"[dataset] dropped {dropped} row(s) with NaN in used columns")
    df[C.LABEL_COL] = df[C.LABEL_COL].astype(int)
    return df


def _decision_positions(dates: pd.DatetimeIndex, seq_len: int, horizon: int,
                        train_end, val_start, test_start, test_end=None):
    """Return embargoed decision-row positions per split (pure index logic)."""
    n = len(dates)
    sd = lambda x, side: int(np.searchsorted(dates.values, np.datetime64(x), side))
    train_end_pos = sd(train_end, "right") - 1
    val_start_pos = sd(val_start, "left")
    test_start_pos = sd(test_start, "left")
    test_end_pos = (sd(test_end, "right") - 1) if test_end else n - 1

    all_i = np.arange(seq_len - 1, n)  # need seq_len rows of lookback (inclusive of i)

    train = all_i[(all_i <= train_end_pos) & (all_i + horizon < val_start_pos)]
    val = all_i[(all_i >= val_start_pos) & (all_i + horizon < test_start_pos)]
    test = all_i[(all_i >= test_start_pos) & (all_i <= test_end_pos) & (all_i + horizon <= n - 1)]

    # leakage guard: no forward label window may cross a split boundary
    # (empty val/test is allowed -- e.g. deploy retrain puts everything in train)
    assert len(train) == 0 or train.max() + horizon < val_start_pos, "train label leaks into val"
    assert len(val) == 0 or val.max() + horizon < test_start_pos, "val label leaks into test"
    return {"train": train, "val": val, "test": test,
            "train_end_pos": train_end_pos, "val_start_pos": val_start_pos,
            "test_start_pos": test_start_pos}


def _build(arr1, arr2, labels, soft, close, dates, positions, seq_len) -> Split:
    if len(positions) == 0:
        return Split(
            np.empty((0, seq_len, arr1.shape[1]), np.float32),
            np.empty((0, arr2.shape[1]), np.float32),
            np.empty(0, np.int64),
            np.empty((0, soft.shape[1]), np.float32),
            pd.DatetimeIndex([]),
            np.empty(0, np.float32),
        )
    X1 = np.stack([arr1[i - seq_len + 1: i + 1] for i in positions]).astype(np.float32)
    X2 = arr2[positions].astype(np.float32)
    y = labels[positions].astype(np.int64)
    y_soft = soft[positions].astype(np.float32)
    return Split(X1, X2, y, y_soft, dates[positions], close[positions].astype(np.float32))


def prepare_data(path=None, seq_len=C.SEQ_LEN, horizon=C.HORIZON,
                 train_end=C.TRAIN_END, val_start=C.VAL_START,
                 test_start=C.TEST_START, test_end=None) -> Data:
    df = load_dataframe(path)
    dates = df.index
    pos = _decision_positions(dates, seq_len, horizon,
                              train_end, val_start, test_start, test_end)

    # fit scalers on TRAIN-PERIOD ROWS ONLY (positions up to train_end_pos)
    train_rows = slice(0, pos["train_end_pos"] + 1)
    scaler_t1 = RobustScaler().fit(df[C.TRACK1_FEATS].values[train_rows])
    scaler_t2 = RobustScaler().fit(df[C.TRACK2_FEATS].values[train_rows])

    arr1 = scaler_t1.transform(df[C.TRACK1_FEATS].values)
    arr2 = scaler_t2.transform(df[C.TRACK2_FEATS].values)
    labels = df[C.LABEL_COL].values
    soft = df[C.SOFT_LABEL_COLS].values
    close = df[C.CLOSE_COL].values

    splits = {k: _build(arr1, arr2, labels, soft, close, dates, pos[k], seq_len)
              for k in ("train", "val", "test")}

    cw = compute_class_weight("balanced", classes=np.arange(C.N_REGIMES), y=splits["train"].y)
    return Data(splits["train"], splits["val"], splits["test"],
                scaler_t1, scaler_t2, cw.astype(np.float32), df)


def make_dataloaders(data: Data, batch_size=64):
    """Build torch DataLoaders (torch imported lazily so the rest of the
    pipeline can be verified without a torch install)."""
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    def ds(s: Split):
        return TensorDataset(torch.from_numpy(s.X1), torch.from_numpy(s.X2),
                             torch.from_numpy(s.y), torch.from_numpy(s.y_soft))

    g = torch.Generator().manual_seed(C.SEED)
    return (DataLoader(ds(data.train), batch_size=batch_size, shuffle=True, generator=g),
            DataLoader(ds(data.val), batch_size=batch_size, shuffle=False),
            DataLoader(ds(data.test), batch_size=batch_size, shuffle=False))


if __name__ == "__main__":
    d = prepare_data()
    for name, s in [("train", d.train), ("val", d.val), ("test", d.test)]:
        print(f"{name:5s} n={len(s.y):5d}  X1={s.X1.shape}  X2={s.X2.shape}  "
              f"dates=[{s.dates.min().date()} .. {s.dates.max().date()}]")
    print("class_weights:", np.round(d.class_weights, 3))
    print("train label dist:", np.bincount(d.train.y, minlength=5))
