"""Training loop, class-weighted loss (+ optional soft-label distillation),
LR scheduling, early stopping, temperature scaling, and artifact saving.

Defaults reproduce the proposal exactly (pure class-weighted CrossEntropy).
Setting ``soft_weight > 0`` adds the optional HMM-posterior distillation term
(a label-smoothing variant) that improves probability calibration without
changing the architecture or output.
"""
from __future__ import annotations
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

import config as C
from dataset import Data, make_dataloaders
from model import RegimePredictor


def set_seed(seed=C.SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _collect_logits(model, loader, device):
    model.eval()
    logits, ys = [], []
    with torch.no_grad():
        for x1, x2, y, _ in loader:
            logits.append(model(x1.to(device), x2.to(device)).cpu())
            ys.append(y)
    return torch.cat(logits), torch.cat(ys)


def train_model(data: Data, *, gru_hidden=64, fusion_dim=32, dropout=0.3,
                lr=1e-3, weight_decay=1e-4, soft_weight=0.0, batch_size=64,
                max_epochs=200, es_patience=20, grad_clip=1.0, seed=C.SEED,
                device=None, verbose=True):
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _ = make_dataloaders(data, batch_size)

    model = RegimePredictor(gru_hidden=gru_hidden, fusion_dim=fusion_dim,
                            dropout=dropout).to(device)
    class_w = torch.tensor(data.class_weights, device=device)
    ce = nn.CrossEntropyLoss(weight=class_w)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)

    best_f1, best_state, wait, history = -1.0, None, 0, []
    for epoch in range(max_epochs):
        model.train()
        tr_loss = 0.0
        for x1, x2, y, y_soft in train_loader:
            x1, x2, y, y_soft = x1.to(device), x2.to(device), y.to(device), y_soft.to(device)
            opt.zero_grad()
            logits = model(x1, x2)
            loss = ce(logits, y)
            if soft_weight > 0:  # optional HMM-posterior distillation
                loss = loss + soft_weight * F.kl_div(
                    F.log_softmax(logits, dim=1), y_soft, reduction="batchmean")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            tr_loss += loss.item() * len(y)
        tr_loss /= len(train_loader.dataset)

        val_logits, val_y = _collect_logits(model, val_loader, device)
        val_loss = ce(val_logits.to(device), val_y.to(device)).item()
        val_pred = val_logits.argmax(1).numpy()
        val_f1 = f1_score(val_y.numpy(), val_pred, average="macro")
        sched.step(val_loss)
        history.append({"epoch": epoch, "train_loss": tr_loss,
                        "val_loss": val_loss, "val_macro_f1": val_f1})

        if val_f1 > best_f1:
            best_f1, best_state, wait = val_f1, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
        if verbose and (epoch % 5 == 0 or wait == 0):
            print(f"ep{epoch:3d} train={tr_loss:.4f} val={val_loss:.4f} "
                  f"val_f1={val_f1:.4f}{'  *' if wait == 0 else ''}")
        if wait >= es_patience:
            if verbose:
                print(f"early stop @ {epoch} (best val_macro_f1={best_f1:.4f})")
            break

    model.load_state_dict(best_state)
    temperature = fit_temperature(model, val_loader, device)
    return model, history, temperature, device


def fit_temperature(model, val_loader, device) -> float:
    """Post-hoc temperature scaling: minimise val NLL over scalar T>0."""
    logits, y = _collect_logits(model, val_loader, device)
    log_T = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=100)
    nll = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = nll(logits / log_T.exp(), y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(log_T.exp().item())


def save_artifacts(model, data: Data, temperature, hp: dict, tag="best"):
    import joblib
    C.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), C.ARTIFACT_DIR / f"{tag}_model.pt")
    joblib.dump(data.scaler_t1, C.ARTIFACT_DIR / "scaler_t1.pkl")
    joblib.dump(data.scaler_t2, C.ARTIFACT_DIR / "scaler_t2.pkl")
    meta = {"track1_feats": C.TRACK1_FEATS, "track2_feats": C.TRACK2_FEATS,
            "seq_len": C.SEQ_LEN, "horizon": C.HORIZON, "n_regimes": C.N_REGIMES,
            "temperature": temperature, "class_weights": data.class_weights.tolist(),
            "splits": {"train_end": C.TRAIN_END, "val": [C.VAL_START, C.VAL_END],
                       "test_start": C.TEST_START}, "hparams": hp}
    (C.ARTIFACT_DIR / "config.json").write_text(json.dumps(meta, indent=2))
    print(f"[train] artifacts saved to {C.ARTIFACT_DIR}")


if __name__ == "__main__":
    from dataset import prepare_data
    d = prepare_data()
    hp = dict(gru_hidden=64, fusion_dim=32, dropout=0.3, lr=1e-3,
              weight_decay=1e-4, soft_weight=0.3)
    model, hist, temp, dev = train_model(d, max_epochs=3, es_patience=20, **hp)
    print("temperature:", round(temp, 3), "device:", dev)
