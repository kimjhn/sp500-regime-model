# S&P500 시장 레짐 예측 모델

거시경제(매크로) 지표와 주가 기술 지표를 융합한 **Two-Track 딥러닝**(GRU + MLP + Attention Late-Fusion)으로 S&P500의 **향후 약 20거래일 시장 국면을 확률 예측**하고, 투자자 성향에 맞춘 권장 주식 비중을 **Streamlit 대시보드**로 제공합니다.

> **검증된 가치:** 최대 수익이 아니라 **"덜 잃는 것"(하락 방어)** — 최대낙폭 약 -10% vs Buy&Hold -34% (2019~).

---

## 🌐 라이브 데모

(Streamlit Cloud 배포 후 URL 추가 예정)

---

## ✨ 핵심 특징

- **HMM 레짐 라벨링** — 미래 20일 가격 행동을 5개 레짐(강한하락/약한하락/중립/약한상승/강한상승)으로 요약
- **Two-Track 딥러닝** — GRU(과거 60일 주가) + MLP(매크로 6개) + Attention Late-Fusion → Softmax(5)
- **시드 앙상블(3개) + Temperature Scaling** — 단일 시드 폴드 붕괴 차단 + 확률 보정
- **확률가중 백테스트** — 매일 `exposure = probs[t] @ ladder`로 연속 변동
- **투자성향 개인화** — 설문(보수/균형/성장) 또는 슬라이더 직접 조절
- **v2 NFCI 이벤트 오버레이** — 구조적 위기 vs 일회성 충격을 보조 지표로 판별
- **미래참조편향 차단** — FRED ALFRED `first_release` + 지표별 발표지연 offset

---

## 🚀 빠른 시작 — 대시보드 실행 (5분)

### 1) 환경 준비

**Python 3.10 권장** (스케일러 pickle 호환성)

```bash
# 가상환경 생성
python -m venv venv

# 활성화 — Windows
venv\Scripts\activate
# 활성화 — macOS/Linux
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

또는 conda:
```bash
conda create -n regime python=3.10
conda activate regime
pip install -r requirements.txt
```

### 2) 실행

```bash
streamlit run streamlit_app.py
```

브라우저에서 자동으로 http://localhost:8501 열림.

### 3) 사용

- 사이드바에서 투자성향 설정 (설문 자동 / 국면별 직접 조절)
- 메인 화면: 현재 레짐 확률, 권장 비중, 가격+예측 국면 오버레이, 백테스트
- 하락장(빨강·주황) 구간에 마우스 호버 → 구조적 확률 궤적

---

## 📂 프로젝트 구조

```
streamlit_app.py                       # 메인 대시보드
requirements.txt                       # 런타임 의존성
README.md                              # 본 문서

Deep/
├── code/
│   ├── config.py                      # 상수 (피처 목록·분할일·시드 등)
│   ├── dataset.py                     # 데이터 로딩 + 누수 차단 분할
│   ├── model.py                       # RegimePredictor (GRU+MLP+Attention)
│   ├── train.py                       # 학습 + Temperature scaling
│   ├── evaluate.py                    # 평가 메트릭
│   ├── backtest.py                    # 확률가중 백테스트
│   ├── serve.py                       # 라이브 추론 + 개인화
│   ├── event_overlay.py               # v2 NFCI 보조 지표
│   ├── retrain_deploy.py              # 2단계 배포 재학습 스크립트
│   ├── STEP2_DeepLearning.ipynb       # STEP 2 학습 노트북
│   └── artifacts/                     # 학습된 산출물
│       ├── ensemble_seed{0,1,2}.pt    # 3-시드 앙상블 가중치
│       ├── scaler_t1.pkl              # Track 1 (주가) 스케일러
│       ├── scaler_t2.pkl              # Track 2 (매크로) 스케일러
│       └── config.json                # 모델 메타 (피처·온도·에폭 등)
└── data/
    └── sp500_regime_dataset_final.csv # 학습 데이터셋 (HMM 출력)

HMM_Regime_Detection/
├── code/
│   └── HMM_Regime_Detection_v2 (1).ipynb  # STEP 1 노트북 (HMM 레짐 라벨 생성)
├── output/
│   ├── regime_final.png               # 5색 레짐 시각화
│   └── sp500_regime_dataset_final.csv # 사본 (Deep/data와 동일)
└── sp500_regime_dataset_final.csv     # 사본
```

---

## 🔁 처음부터 재현하기 (전체 파이프라인)

### STEP 1 — HMM 레짐 라벨 생성

**필요:** FRED API 키 (https://fred.stlouisfed.org/docs/api/api_key.html — 무료)

1. STEP1 노트북 열기:
   `HMM_Regime_Detection/code/HMM_Regime_Detection_v2 (1).ipynb`
2. 첫 셀에 본인 FRED API 키 입력
3. 모든 셀 순차 실행
4. **산출물:** `Deep/data/sp500_regime_dataset_final.csv` 생성

> ⚠️ 매크로 데이터는 시점마다 *개정(revision)* 될 수 있으므로, 재실행 시점에 따라 데이터셋이 약간 달라질 수 있습니다 (이는 시계열의 본질적 특성이며 ALFRED `first_release` 사용으로 *결정 시점의* 데이터는 보존됩니다).

### STEP 2 — 딥러닝 모델 학습

**권장:** Google Colab GPU (모델 학습은 CPU에선 매우 느림)

1. Colab에 `Deep/code/STEP2_DeepLearning.ipynb` 업로드
2. Drive에 본 레포 마운트 또는 `Deep/code/*.py` + `Deep/data/*.csv` 함께 업로드
3. 노트북 셀 순차 실행 — 학습·검증 진행
4. **산출물:** 평가 메트릭(macro-F1 ≈ 0.63, risk-off AUROC ≈ 0.98)

### STEP 3 — 배포용 100% 데이터 2단계 재학습

학습 모델을 *최신 데이터까지 100% 반영*하여 라이브 서빙 가중치 생성:

```bash
# Colab에서 실행 권장 (CPU도 가능하지만 느림)
python Deep/code/retrain_deploy.py
```

내부 동작:
- **Phase A:** train 1999~2021, val 2022~2026 → 시드별 최적 에폭·온도 탐색
- **Phase B:** 1999~ 전체를 Phase A 에폭만큼 고정 학습 (검증 없음)
- 시드 [0,1,2] 3개 학습 → 앙상블 구성

**산출물:** `Deep/code/artifacts/ensemble_seed{0,1,2}.pt`, `scaler_t1.pkl`, `scaler_t2.pkl`, `config.json`

### STEP 4 — 대시보드 실행

위 **"빠른 시작"** 절차로 실행.

---

## 📊 데이터 출처

| 종류 | 출처 | 라이선스 |
|------|------|---------|
| 주가 (SPY) | FinanceDataReader (Yahoo Finance) | 무료 사용 |
| 매크로 (M2, NFCI, YIELD_SPREAD, CPI, DEBT_EQUITY, LOAN_DEPOSIT) | FRED ALFRED API | 무료 (API 키 필요) |

**미래참조편향 차단 방법:**
- ALFRED `first_release` (수정 전 초기 공시치) 우선 사용
- 지표별 발표지연 offset (M2 +25일, CPI +15일, NFCI +5일 등)
- `safe_reindex` (합집합 → ffill → 평일 추출, 주말 발표 NaN 방지)

---

## 🧪 재현성 보장 요소

- **고정 시드:** `SEED=42` (config.py), 앙상블 시드 `[0,1,2]` (retrain_deploy.py)
- **스케일러 버전 핀:** `scikit-learn==1.6.1` — 저장 pickle 호환 보장
- **저장 산출물:** 학습 다시 안 해도 동일 추론 재현 가능
- **누수 차단 강제:** dataset.py에서 HMM forward 윈도우와 split 경계 사이 embargo 강제(assertion)
- **2단계 배포 절차 문서화:** retrain_deploy.py 주석 + 본 README

---

## ⚠️ 한계 / 알려진 제약

- **대시보드 백테스트는 in-sample** (배포 모델이 1999~ 전체 학습됨). 엄밀한 OOS 평가는 `STEP2_DeepLearning.ipynb`의 2022+ 테스트셋 결과 참조.
- v1 모델은 *가격 행동의 심각도*만 분류 — 구조적/일회성 원인 구분은 **v2 NFCI 오버레이**가 보조.
- NFCI는 주간·발표지연이 있어 *단발 이벤트*(예: 관세 발표일)엔 반박자 느림.
- 모델은 향후 ~20일 지평이므로 일중·일간 단기 예측엔 부적합.
- GPU 비결정성 때문에 STEP 2 재학습 시 *비트 단위 일치*는 보장되지 않음 (결과 유사성은 보장).

---

## 🧠 아키텍처 한눈에

```
[미래 t+1~t+20 주가] ──► HMM(5) ──► regime_label[t]      ← 딥러닝 정답 Y
                                          ▲
[과거 ~t 주가 60일] ──► GRU ───┐         │
                               ├─► Attention Late-Fusion ─► Softmax(5)
[t 시점 매크로 6개] ──► MLP ───┘                          (확률 분포)
                                                              │
                                            +─────────────────┘
                                            ▼
                              probs[t] @ ladder = 권장 비중

[v2 보조 — v1 변경 없음]
NFCI + ΔNFCI_20d ──► sigmoid ──► structural_score (구조적 vs 일회성 확률)
```

핵심 설계 원칙:
- **HMM 입력(미래) ≠ DL 입력(과거)** — 완전 분리로 데이터 누수 차단
- **모델은 1개, 사다리만 사용자별** — 개인화는 post-model (재학습 불필요)

---

## 📜 라이선스

(추가 예정 — 학술 과제 제출용)

---

## 📝 참고

- 본 프로젝트는 학술 과제로 작성되었습니다.
- 투자 의사결정의 참고용이며, 손익 책임은 사용자에게 있습니다.
