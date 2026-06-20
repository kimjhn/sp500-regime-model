"""배포용 2단계 전체 재학습 스크립트.

평가 분할(1999-2018 / 2019-2021 / 2022+)은 노트북 STEP2에 기록되어 있고 변경하지 않는다.
이 스크립트는 평가 완료 후 "100% 데이터로 최종 live-inference 모델"을 만드는 배포 절차다.

  Phase A (에폭·온도 탐색):
    train = 1999~2021,  val = 2022~2026 (5개 레짐 전부 포함 → 조기종료 신호가 건강함)
    → seed별 '최적 에폭 수'와 temperature를 얻는다.

  Phase B (전체 재학습):
    train = 1999~2026 전체 데이터를, Phase A에서 찾은 에폭 수만큼 고정 학습(조기종료 없음).
    → 최신 데이터까지 최종 가중치에 반영. temperature는 Phase A 값 재사용.

아키텍처·HPs는 STEP2에서 확정된 값 그대로 사용한다.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import numpy as np
import torch
import torch.nn as nn
import joblib

import config as C
from dataset import prepare_data, make_dataloaders
from model import RegimePredictor
from train import train_model, set_seed

# Phase A: 다양한 검증셋 (2022 긴축장 + 2023~25 회복/강세 = 5개 레짐 전부)
PHASEA_TRAIN_END = "2021-12-31"
PHASEA_VAL_START = "2022-01-01"
FUTURE           = "2030-01-01"   # 사실상 "데이터 끝까지"

HP = dict(
    gru_hidden=64, gru_layers=2, fusion_dim=32,
    dropout=0.4, lr=3e-4, weight_decay=1e-4, batch_size=64,
)
SEEDS       = [0, 1, 2]
MAX_EPOCHS  = 300
ES_PATIENCE = 25
MIN_EPOCHS  = 5    # Phase B 최소 에폭 (epoch 1 같은 극단적 저학습 방지용 안전장치)


def _hp_for_train_model():
    """train_model()은 gru_layers를 받지 않는다(모델 기본값 2 사용)."""
    return {k: v for k, v in HP.items() if k != "gru_layers"}


def best_epoch_count(history):
    """history에서 val_macro_f1이 최대인 에폭 → 학습 횟수(=index+1)로 변환."""
    best_idx = max(history, key=lambda h: h["val_macro_f1"])["epoch"]
    return max(MIN_EPOCHS, best_idx + 1)


def train_fixed_epochs(data, n_epochs, seed, device):
    """전체 데이터로 고정 에폭 학습 (검증/조기종료 없음). Phase A와 동일한 손실/옵티마이저."""
    set_seed(seed)
    loader, _, _ = make_dataloaders(data, HP["batch_size"])
    model = RegimePredictor(gru_hidden=HP["gru_hidden"], fusion_dim=HP["fusion_dim"],
                            dropout=HP["dropout"]).to(device)
    ce = nn.CrossEntropyLoss(weight=torch.tensor(data.class_weights, device=device))
    opt = torch.optim.Adam(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    for ep in range(n_epochs):
        model.train()
        tot = 0.0
        for x1, x2, y, _ in loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            opt.zero_grad()
            loss = ce(model(x1, x2), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(y)
        print(f"      ep{ep:3d} train_loss={tot/len(loader.dataset):.4f}")
    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print(f"배포용 2단계 전체 재학습  (device={device})")
    print("=" * 60)

    # ----- Phase A: 에폭·온도 탐색 -----
    dataA = prepare_data(train_end=PHASEA_TRAIN_END,
                         val_start=PHASEA_VAL_START, test_start=FUTURE)
    print(f"\n[Phase A] train {len(dataA.train.y)} "
          f"[{dataA.train.dates.min().date()}~{dataA.train.dates.max().date()}]  "
          f"val {len(dataA.val.y)} [{dataA.val.dates.min().date()}~{dataA.val.dates.max().date()}]")
    print(f"  val 레짐 분포: {np.bincount(dataA.val.y, minlength=5)}  (5개 전부 있어야 건강)")

    best_epochs, temps = [], []
    for sd in SEEDS:
        print(f"\n  ── Phase A seed {sd} (에폭 탐색) ──")
        _, hist, temp, _ = train_model(
            dataA, max_epochs=MAX_EPOCHS, es_patience=ES_PATIENCE,
            seed=sd, device=device, verbose=True, **_hp_for_train_model())
        be = best_epoch_count(hist)
        best_epochs.append(be)
        temps.append(temp)
        print(f"    → 최적 에폭 {be}회, temperature {temp:.4f}")

    # ----- Phase B: 전체 데이터로 고정 에폭 재학습 -----
    dataB = prepare_data(train_end=FUTURE, val_start=FUTURE, test_start=FUTURE)
    print(f"\n[Phase B] 전체 train {len(dataB.train.y)} "
          f"[{dataB.train.dates.min().date()}~{dataB.train.dates.max().date()}]  (검증셋 없음)")

    C.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    for sd, be in zip(SEEDS, best_epochs):
        print(f"\n  ── Phase B seed {sd}: 전체 데이터 {be} epochs ──")
        model = train_fixed_epochs(dataB, be, sd, device)
        state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        torch.save(state, C.ARTIFACT_DIR / f"ensemble_seed{sd}.pt")

    # 스케일러는 Phase B(전체 데이터)로 fit된 것을 저장 → 저장 모델과 자기일관
    joblib.dump(dataB.scaler_t1, C.ARTIFACT_DIR / "scaler_t1.pkl")
    joblib.dump(dataB.scaler_t2, C.ARTIFACT_DIR / "scaler_t2.pkl")

    meta = {
        "track1_feats": C.TRACK1_FEATS,
        "track2_feats": C.TRACK2_FEATS,
        "seq_len":      C.SEQ_LEN,
        "horizon":      C.HORIZON,
        "n_regimes":    C.N_REGIMES,
        "model_kwargs": {
            "gru_hidden": HP["gru_hidden"], "gru_layers": HP["gru_layers"],
            "fusion_dim": HP["fusion_dim"], "dropout": HP["dropout"],
        },
        "ensemble_seeds": SEEDS,
        "temperatures":   {str(s): float(t) for s, t in zip(SEEDS, temps)},
        "best_epochs":    {str(s): int(e) for s, e in zip(SEEDS, best_epochs)},
        "class_weights":  dataB.class_weights.tolist(),
        "splits": {
            "phaseA_train_end": PHASEA_TRAIN_END,
            "phaseA_val_start": PHASEA_VAL_START,
            "phaseB": "all-data fixed-epoch retrain (1999~2026)",
        },
        "hparams": HP,
        "deploy_note": (
            "2-phase deploy retrain. Phase A: epoch count + temperature found on "
            "1999-2021 train / 2022-2026 val (all 5 regimes). Phase B: retrained on "
            "ALL labeled data for those epochs; temperature reused from Phase A. "
            "Evaluation metrics (2022+ OOS) are in STEP2 notebook."
        ),
    }
    (C.ARTIFACT_DIR / "config.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 60)
    print("2단계 재학습 완료!")
    print(f"  best_epochs : { {str(s): e for s, e in zip(SEEDS, best_epochs)} }")
    print(f"  temperatures: { {str(s): round(t,4) for s, t in zip(SEEDS, temps)} }")
    print(f"  아티팩트    : {C.ARTIFACT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
