"""S&P500 시장 레짐 대시보드 (STEP 3 배포).

모델은 향후 약 20거래일의 시장 국면 확률을 출력한다(모든 투자자 공통). 투자 성향은 모델
이후 단계에서 '익스포저 사다리'만 바꿔 권장 비중을 개인화한다(recommended = probs @ ladder).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "Deep" / "code"))
import config as C
import backtest as B
import serve as S
import event_overlay as EO

st.set_page_config(page_title="S&P500 시장 레짐 대시보드", layout="wide",
                   initial_sidebar_state="expanded")

REGIME_KR = {0: "강한 하락장", 1: "약한 하락장", 2: "중립", 3: "약한 상승장", 4: "강한 상승장"}
COLOR = {0: "#d64545", 1: "#e89b4f", 2: "#94a3b8", 3: "#5b9bd5", 4: "#3aa978"}
ORDER = [REGIME_KR[i] for i in range(C.N_REGIMES)]
REGIME_SCALE = alt.Scale(domain=ORDER, range=[COLOR[i] for i in range(C.N_REGIMES)])
STRAT_NAME = "레짐 내비게이터"
STRAT_SCALE = alt.Scale(domain=[STRAT_NAME, "Buy&Hold"], range=["#2a9d8f", "#9aa3ad"])
BACKTEST_START = "2019-01-01"   # 2020 코로나 폭락을 포함해 방어 효과가 보이도록

SCORE_OPTS = {"A (보수)": 0, "B (중립)": 1, "C (공격)": 2}
LEV_OPTS = {"안 씀 (≤100%)": "A", "약간 (~130%)": "B", "적극 (~200%)": "C"}
SHORT_OPTS = {"안 함 (현금만)": "A", "일부 (~-50%)": "B", "적극 (~-100%)": "C"}


# --------------------------------------------------------------------------- #
# 캐시 로더                                                                    #
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="모델 로딩...")
def get_ensemble():
    return S.load_ensemble()


@st.cache_data(show_spinner="데이터 로딩...")
def get_features():
    return S.load_features()


@st.cache_data(show_spinner="과거 구간 추론...")
def get_series(start):
    ens = get_ensemble()
    return S.predict_series(ens, get_features(), start=start)


# --------------------------------------------------------------------------- #
# 사이드바: 투자 성향 진단 (+ 결과 사다리)                                     #
# --------------------------------------------------------------------------- #
def sidebar_profile():
    """사이드바에서 익스포저 사다리(길이 5)를 만들어 반환."""
    st.sidebar.header("투자 성향 / 비중 설정")
    mode = st.sidebar.radio(
        "사다리 정하는 방식", ["설문으로 자동", "국면별 직접 조절"], index=0,
        help="설문: A/B/C에 답하면 자동 계산 · 직접: 국면마다 원하는 주식 비중을 슬라이더로 지정")

    if mode == "국면별 직접 조절":
        st.sidebar.caption("각 국면일 때 주식에 둘 비중을 직접 정하세요. "
                           "음수 = 공매도, 100% 초과 = 레버리지, 나머지는 현금/안전자산.")
        ladder = np.array([
            st.sidebar.slider(REGIME_KR[i], min_value=-1.0, max_value=2.0,
                              value=float(C.EXPOSURE_LADDER[i]), step=0.05, format="%.2f")
            for i in range(C.N_REGIMES)], dtype=float)
    else:
        st.sidebar.caption("성향은 '국면→비중' 사다리만 바꿉니다. 시장 국면 확률 자체는 누구에게나 같습니다.")
        q3 = st.sidebar.selectbox("연 최대 감내 손실(MDD)", list(SCORE_OPTS), index=1,
                                  help="A: -10% 이내 / B: -20% / C: -35%+")
        q4 = st.sidebar.selectbox("투자 1순위 목적", list(SCORE_OPTS), index=1,
                                  help="A: 원금 보존 / B: 시장 추종 / C: 최대 성장")
        q7 = st.sidebar.selectbox("투자 기간", list(SCORE_OPTS), index=1,
                                  help="A: 3년 미만 / B: 3~10년 / C: 10년+")
        q8 = st.sidebar.selectbox("-20% 평가손이 나면?", list(SCORE_OPTS), index=1,
                                  help="A: 매도 / B: 버팀 / C: 추가매수")
        score = sum(SCORE_OPTS[x] for x in (q3, q4, q7, q8))
        q1 = st.sidebar.selectbox("레버리지 사용", list(LEV_OPTS), index=0)
        q2 = st.sidebar.selectbox("하락장 공매도", list(SHORT_OPTS), index=0)
        ladder = S.profile_to_ladder(
            {"score": score, "q1_leverage": LEV_OPTS[q1], "q2_short": SHORT_OPTS[q2]})
        st.sidebar.markdown(f"**진단 결과: `{S.preset_from_score(score)}`** (공격성 {score}/8)")

    st.sidebar.markdown("**적용 사다리 (국면 → 주식 비중)**")
    st.sidebar.dataframe(
        pd.DataFrame({"국면": ORDER, "비중": [f"{w:.0%}" for w in ladder]}).set_index("국면"),
        use_container_width=True)
    return ladder


# --------------------------------------------------------------------------- #
# Altair 차트                                                                  #
# --------------------------------------------------------------------------- #
def prob_chart(probs):
    d = pd.DataFrame({"국면": ORDER, "확률": probs})
    base = alt.Chart(d).encode(
        x=alt.X("국면:N", sort=ORDER, axis=alt.Axis(labelAngle=0, title=None)))
    bars = base.mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
        y=alt.Y("확률:Q", axis=alt.Axis(format="%", title=None), scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("국면:N", scale=REGIME_SCALE, legend=None),
        tooltip=[alt.Tooltip("국면:N"), alt.Tooltip("확률:Q", format=".1%")])
    text = base.mark_text(dy=-8, fontSize=13, fontWeight="bold").encode(
        y="확률:Q", text=alt.Text("확률:Q", format=".0%"))
    return (bars + text).properties(height=240)


def _regime_blocks(dates, preds, struct=None):
    """연속된 같은 국면 구간을 블록으로. struct(구조적점수 배열)가 있으면 하락장 블록마다
    '구조적 확률 궤적'(초기/최고/최근)을 *별도 컬럼*으로 붙이고 is_down으로 구분한다.
    툴팁에서 한 줄로 뭉치지 않고 여러 행으로 예쁘게 표시하기 위함."""
    dts = pd.to_datetime(dates); p = np.asarray(preds); rows = []; s = 0
    for i in range(1, len(p) + 1):
        if i == len(p) or p[i] != p[s]:
            end = dts[i] if i < len(p) else dts[-1]
            reg = int(p[s])
            row = {"start": dts[s], "end": end, "국면": REGIME_KR[reg],
                   "기간": f"{dts[s].date()} ~ {end.date()}"}
            if struct is not None:
                seg = np.asarray(struct[s:i], float)
                seg = seg[~np.isnan(seg)]
                down = (reg in C.RISK_OFF_REGIMES) and len(seg) > 0
                row["is_down"] = bool(down)
                if down:
                    row["구조초기"] = f"{float(seg[0]):.0%}"
                    row["구조최고"] = f"{float(seg.max()):.0%}"
                    row["구조최근"] = f"{float(seg[-1]):.0%}"
            rows.append(row); s = i
    return pd.DataFrame(rows)


def price_regime_chart(dates, close, preds, struct=None):
    line_df = pd.DataFrame({"날짜": pd.to_datetime(dates), "종가": close})
    blocks = _regime_blocks(dates, preds, struct)
    legend = alt.Legend(title="예측 국면 (배경색)", orient="bottom", direction="horizontal")
    color_enc = alt.Color("국면:N", scale=REGIME_SCALE, legend=legend)
    base_tip = [alt.Tooltip("국면:N", title="예측 국면"), alt.Tooltip("기간:N", title="기간")]
    line = (alt.Chart(line_df).mark_line(strokeWidth=2).encode(
        x=alt.X("날짜:T", title=None),
        y=alt.Y("종가:Q", title="S&P500 (SPY)", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("날짜:T", title="날짜"), alt.Tooltip("종가:Q", format=".0f", title="종가")]))

    def _rect(data, tip):
        return alt.Chart(data).mark_rect(opacity=0.42).encode(
            x="start:T", x2="end:T", color=color_enc, tooltip=tip)

    if "is_down" in blocks.columns:
        flat, down = blocks[~blocks["is_down"]], blocks[blocks["is_down"]]
        layers = []
        if len(flat):
            layers.append(_rect(flat, base_tip))
        if len(down):
            layers.append(_rect(down, base_tip + [
                alt.Tooltip("구조초기:N", title="구조적 · 초기"),
                alt.Tooltip("구조최고:N", title="구조적 · 최고"),
                alt.Tooltip("구조최근:N", title="구조적 · 최근")]))
        layers.append(line)
        return alt.layer(*layers).properties(height=360)

    return (_rect(blocks, base_tip) + line).properties(height=360)


def backtest_frames(dates, close, probs, ladder):
    ar = B.daily_asset_returns(close)
    w = B.exposure_from_probs(probs, ladder)
    pnl_pw, pnl_bh = B.strategy_pnl(w, ar), B.strategy_pnl(np.ones(len(close)), ar)
    eq_pw, eq_bh = np.cumprod(1 + pnl_pw), np.cumprod(1 + pnl_bh)
    dd_pw = eq_pw / np.maximum.accumulate(eq_pw) - 1
    dd_bh = eq_bh / np.maximum.accumulate(eq_bh) - 1
    shallow_frac = float((dd_pw >= dd_bh - 1e-12).mean())  # 내 전략이 더 얕게(또는 같게) 빠진 날 비율
    dts = pd.to_datetime(dates)
    eq = pd.DataFrame({"날짜": dts, STRAT_NAME: eq_pw, "Buy&Hold": eq_bh}) \
        .melt("날짜", var_name="전략", value_name="자산배수")
    dd = pd.DataFrame({"날짜": dts, STRAT_NAME: dd_pw, "Buy&Hold": dd_bh}) \
        .melt("날짜", var_name="전략", value_name="낙폭")
    return (eq, dd, B.perf_metrics(pnl_pw, w),
            B.perf_metrics(pnl_bh, np.ones(len(close))), shallow_frac, float(w.mean()))


def equity_chart(eq):
    return alt.Chart(eq).mark_line(strokeWidth=2).encode(
        x=alt.X("날짜:T", title=None),
        y=alt.Y("자산배수:Q", title="자산 (시작=1)", scale=alt.Scale(zero=False)),
        color=alt.Color("전략:N", scale=STRAT_SCALE, legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("날짜:T"), "전략:N", alt.Tooltip("자산배수:Q", format=".2f")]
    ).properties(height=300)


def drawdown_chart(dd):
    return alt.Chart(dd).mark_line(strokeWidth=1.8).encode(
        x=alt.X("날짜:T", title=None),
        y=alt.Y("낙폭:Q", title="낙폭(고점 대비)", axis=alt.Axis(format="%")),
        color=alt.Color("전략:N", scale=STRAT_SCALE, legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("날짜:T"), "전략:N", alt.Tooltip("낙폭:Q", format=".1%")]
    ).properties(height=210)


# --------------------------------------------------------------------------- #
st.title("S&P500 시장 레짐 예측 대시보드")
st.caption("향후 약 20거래일(한 달) 시장 국면을 확률로 추론하고, 투자 성향에 맞춘 권장 주식 비중을 제시합니다.")

try:
    ens = get_ensemble()
except FileNotFoundError:
    st.error("학습된 모델이 없습니다. 노트북의 앙상블 셀을 Colab에서 실행해 "
             "`Deep/code/artifacts/`를 만든 뒤 다시 실행하세요.")
    st.stop()

ladder = sidebar_profile()

df = get_features()
X1, X2, as_of = S.latest_window(ens, df)
probs_now = S.predict_proba(ens, X1, X2)[0]
top = int(probs_now.argmax())
risk_off = float(probs_now[list(C.RISK_OFF_REGIMES)].sum())
exposure = float(probs_now @ ladder)

st.subheader(f"향후 약 20거래일 전망  ·  기준일 {pd.Timestamp(as_of).date()}")
st.caption("데이터는 매일 한국시간 08:00에 자동 갱신됩니다 (FRED + SPY → GitHub Actions → Streamlit Cloud 재배포).")
st.markdown(
    f"<div style='text-align:center;padding:22px 12px;border-radius:18px;"
    f"background:{COLOR[top]}1f;border:1px solid {COLOR[top]}66;margin-bottom:10px;'>"
    f"<div style='font-size:15px;color:#888;letter-spacing:1px;'>가장 가능성 높은 국면</div>"
    f"<div style='font-size:40px;font-weight:800;color:{COLOR[top]};line-height:1.25;'>{REGIME_KR[top]}</div>"
    f"<div style='font-size:64px;font-weight:800;color:{COLOR[top]};line-height:1.05;'>{probs_now[top]:.0%}</div>"
    f"<div style='font-size:13px;color:#999;'>향후 약 20거래일(한 달) 동안 이 국면일 확률</div>"
    f"</div>", unsafe_allow_html=True)
c2, c3 = st.columns(2)
c2.metric("하락장·약한 하락장이 될 확률", f"{risk_off:.0%}",
          help="‘강한 하락장’ + ‘약한 하락장’ 확률의 합. 높을수록 위험.")
c3.metric("권장 주식 비중", f"{exposure:.0%}",
          help="각 국면 확률 × 내 사다리. 나머지는 현금/안전자산.")
st.altair_chart(prob_chart(probs_now), use_container_width=True)

# v2 이벤트 오버레이 — 현재 국면에 맞춰 해석 (하락 신호일 때만 '구조적 vs 일회성')
ev = EO.assess(df)
is_downturn = (top in C.RISK_OFF_REGIMES) or (risk_off >= 0.30)
nfci_desc = "양호(완화적)" if ev["nfci"] < 0 else "긴축(스트레스)"
st.markdown("##### 이벤트 성격 분석 · v2 보조지표")

if is_downturn:
    st.caption("현재 **하락 신호**가 있어, 이 하락이 **구조적 위기**인지 **일회성 충격**인지 "
               "금융환경(NFCI)으로 보조 추정합니다 — v1 국면 예측은 바꾸지 않습니다.")
    e1, e2 = st.columns(2)
    e1.metric("일회성 이벤트 가능성", f"{ev['oneoff_prob']:.0%}",
              help="금융환경(신용·유동성)은 잠잠한데 가격만 급락 → 일시적 충격일 가능성. 변동성이 곧 진정될 수 있음.")
    e2.metric("구조적 위기 가능성", f"{ev['structural_prob']:.0%}",
              help="신용·유동성 등 금융환경이 동반 악화 → 하락이 오래갈 가능성.")
    st.caption(f"근거 · 금융환경지수 NFCI = **{ev['nfci']:+.2f}** ({nfci_desc}, 0 = 역사적 평균), "
               f"최근 20일 변화 {ev['nfci_chg20']:+.2f} → **{ev['label']}**")
    if ev["oneoff_prob"] >= 0.60:
        st.info("모델은 하락 위험을 보지만 금융환경은 잠잠합니다 → **일시적 충격일 가능성**이 큽니다. "
                "변동성이 곧 진정될 수 있으니 과도한 비중 축소는 신중히 판단하세요.")
    elif ev["structural_prob"] >= 0.66:
        st.warning("하락 위험 + 금융환경 동반 악화 → **구조적 위기 가능성**이 큽니다. 방어적 대응이 타당합니다.")
else:
    # 상승·중립 국면: '구조적/일회성 충격' 판정은 무의미 → NFCI를 금융 배경 모니터로 표시
    st.caption(f"현재는 **{REGIME_KR[top]}** 예측으로 하락 신호가 없어, ‘구조적/일회성’ 판정은 해당되지 않습니다. "
               "대신 향후 하락의 씨앗이 될 **금융환경(NFCI)**을 모니터링합니다.")
    st.metric("금융환경지수 NFCI", f"{ev['nfci']:+.2f}",
              help="시카고연준 금융환경지수. 0=역사적 평균, 음수=완화(양호), 양수=긴축(스트레스).")
    if ev["structural_prob"] >= 0.66:
        st.warning(f"금융환경({nfci_desc})이 빠르게 악화 중입니다(20일 변화 {ev['nfci_chg20']:+.2f}). "
                   "하락 전환 시 구조적일 수 있어 **사전 경고**로 참고하세요.")
    else:
        st.caption(f"금융환경 {nfci_desc} · 최근 20일 변화 {ev['nfci_chg20']:+.2f} — "
                   "현재 금융 스트레스 징후가 없어 상승 국면을 지지하는 배경입니다.")

st.divider()
dates, close, probs, preds = get_series(BACKTEST_START)
struct = EO.structural_score_series(df).reindex(pd.to_datetime(dates)).values

st.subheader("가격 + 예측 국면 (2019년~, 2020 코로나 폭락 포함)")
st.caption("하락장(빨강·주황) 구간에 마우스를 올리면 그 하락기의 **구조적 확률 궤적**(초기→최고→최근)이 "
           "표시됩니다. 한 숫자가 아니라 하락기 동안 성격이 어떻게 변했는지를 보여줍니다 "
           "— 100%에 가까울수록 신용·유동성을 동반한 구조적 위기입니다.")
st.altair_chart(price_regime_chart(dates, close, preds, struct), use_container_width=True)

st.subheader(f"백테스트: {STRAT_NAME}(개인화) vs 그냥 계속 보유(Buy&Hold)")
eq, dd, m_pw, m_bh, shallow, avg_w = backtest_frames(dates, close, probs, ladder)

# 실제 지표 차이로 설명을 유동적으로 — 사다리가 모두 100%면 Buy&Hold와 동일해진다.
d_cagr = m_pw["CAGR"] - m_bh["CAGR"]
d_mdd = m_pw["MDD"] - m_bh["MDD"]        # >0 이면 내 전략의 낙폭이 더 얕음(덜 잃음)
d_calmar = m_pw["Calmar"] - m_bh["Calmar"]
identical = abs(d_cagr) < 1e-4 and abs(d_mdd) < 1e-4 and abs(m_pw["Sharpe"] - m_bh["Sharpe"]) < 0.01
defends = d_mdd > 1e-4                    # 낙폭을 유의미하게 줄였는가

st.altair_chart(equity_chart(eq), use_container_width=True)
st.markdown("**낙폭 비교** — 아래로 내려갈수록 고점 대비 손실이 큼(0에 가까울수록 좋음)")
st.altair_chart(drawdown_chart(dd), use_container_width=True)
if identical:
    st.caption(f"현재 사다리는 모든 국면 비중이 100%에 가까워 권장 비중이 매일 ~100%입니다 → "
               f"두 곡선이 **사실상 겹칩니다**(평균 비중 {avg_w:.0%}).")
elif defends:
    st.caption(f"이 구간에서 {STRAT_NAME}의 낙폭이 Buy&Hold보다 **얕거나 같았던 날이 {shallow:.0%}**이고, "
               f"최대 낙폭은 **{m_pw['MDD']:.1%} vs {m_bh['MDD']:.1%}** 로 더 작습니다(평균 비중 {avg_w:.0%}). "
               "강세장에서 시장을 덜 따라가 가끔 더 깊어 보이는 구간이 있어도, 깊은 골(폭락장)에서는 덜 빠집니다.")
else:
    st.caption(f"현재 사다리(평균 비중 {avg_w:.0%})는 시장보다 공격적이어서 최대 낙폭이 **{m_pw['MDD']:.1%} vs "
               f"{m_bh['MDD']:.1%}** 로 오히려 더 큽니다. 하락 국면 비중을 낮추면 방어가 살아납니다.")

m = st.columns(4)
m[0].metric("CAGR (연수익률)", f"{m_pw['CAGR']:.1%}", f"{m_pw['CAGR']-m_bh['CAGR']:+.1%}",
            help="연평균 복리 수익률. S&P500 장기 평균 ≈ 7~10%. 높을수록 좋음.")
m[1].metric("MDD (최대 낙폭)", f"{m_pw['MDD']:.1%}", f"{m_pw['MDD']-m_bh['MDD']:+.1%}",
            help="고점 대비 최대 하락폭. 0에 가까울수록 좋음. 주식 장기보유는 보통 -30~-55%; -20% 이내면 방어 양호.")
m[2].metric("Calmar", f"{m_pw['Calmar']:.2f}", f"{m_pw['Calmar']-m_bh['Calmar']:+.2f}",
            help="연수익 ÷ 최대낙폭. 0.5 이상 양호, 1.0 이상 우수.")
m[3].metric("Sharpe", f"{m_pw['Sharpe']:.2f}", f"{m_pw['Sharpe']-m_bh['Sharpe']:+.2f}",
            help="변동성 대비 초과수익. 1.0 이상 좋음, 2.0 이상 매우 우수, 0 미만은 손실.")
st.caption(f"델타(초록/빨강) = {STRAT_NAME} − Buy&Hold")

if identical:
    st.info(f"현재 사다리는 모든 국면이 100%에 가까워 **{STRAT_NAME}가 Buy&Hold와 사실상 동일**합니다 "
            "(곡선·지표가 겹침). 방어 효과를 보려면 사이드바에서 **하락 국면(강한/약한 하락장)의 비중을 "
            "100%보다 낮춰** 보세요.")
elif defends and d_calmar > 0:
    st.success(f"위험 대비 수익(Calmar {m_pw['Calmar']:.2f} vs {m_bh['Calmar']:.2f})과 "
               f"최대 낙폭({m_pw['MDD']:.1%} vs {m_bh['MDD']:.1%})에서 **{STRAT_NAME}가 앞섭니다.** "
               f"총수익(CAGR {m_pw['CAGR']:.1%} vs {m_bh['CAGR']:.1%})은 "
               f"{'앞서지만' if d_cagr > 0 else '뒤지지만'} 이 모델의 목적은 '덜 잃는 것'입니다.")
elif defends:
    st.info(f"{STRAT_NAME}는 **최대 낙폭이 더 작습니다**({m_pw['MDD']:.1%} vs {m_bh['MDD']:.1%}). "
            f"다만 위험 대비 수익(Calmar {m_pw['Calmar']:.2f} vs {m_bh['Calmar']:.2f})은 앞서지 못했습니다 "
            "— 강세장이 길어 '방어의 대가(낮아진 수익)'가 컸기 때문입니다.")
else:
    st.warning(f"현재 사다리(평균 비중 {avg_w:.0%})는 시장보다 공격적이어서 **최대 낙폭이 더 큽니다** "
               f"({m_pw['MDD']:.1%} vs {m_bh['MDD']:.1%}). 총수익(CAGR {m_pw['CAGR']:.1%} vs "
               f"{m_bh['CAGR']:.1%})은 클 수 있지만 레버리지·고비중은 하락도 키웁니다. 하락 국면 비중을 "
               "낮추면 방어가 살아납니다.")

with st.expander("지표 읽는 법 (좋고 나쁨 기준)"):
    st.markdown(
        "- **CAGR (연수익률)**: 연평균 복리 수익률. S&P500 장기 평균 ≈ **7~10%**. 높을수록 좋음.\n"
        "- **MDD (최대 낙폭)**: 고점 대비 최대 하락. 0에 가까울수록 좋음. 주식 장기보유는 보통 "
        "-30~-55%까지 빠짐 → **-20% 이내면 방어 양호**.\n"
        "- **Calmar = 연수익 ÷ |최대낙폭|**: 떨어질 위험 대비 수익. **0.5 이상 양호, 1.0 이상 우수.**\n"
        "- **Sharpe = 초과수익 ÷ 변동성**: 출렁임 대비 수익. **1.0 이상 좋음, 2.0 이상 매우 우수, "
        "0 미만이면 손실.**")

with st.expander("Buy&Hold가 총수익에서 앞서면 의미가 없나요?"):
    if avg_w < 0.99:
        invest_line = (f"- 이 전략은 **항상 100% 투자하지 않습니다**(현재 사다리 평균 약 {avg_w:.0%}). 그래서 "
                       "**상승장이 오래가면 총수익은 일부 포기**합니다 — 이것이 '방어의 대가(보험료)'입니다.")
    else:
        invest_line = (f"- 현재 사다리는 평균 약 {avg_w:.0%}로 **시장 이상으로 투자**합니다 → 상승을 키우지만 "
                       "하락도 함께 키웁니다.")
    if identical:
        defend_line = "- 현재 사다리에선 낙폭이 Buy&Hold와 **동일**합니다(두 곡선이 겹침)."
        verdict_line = ("- 방어 효과를 보려면 하락 국면 비중을 낮춰야 하며, 그때 **위험 대비 수익"
                        "(Calmar·Sharpe)**과 최대 낙폭에서 차이가 납니다.")
    elif defends:
        defend_line = (f"- 대신 **폭락장에서 손실을 줄입니다.** 2020 코로나를 포함한 2019년~ 구간에서 "
                       f"{STRAT_NAME}의 최대 낙폭은 약 {m_pw['MDD']:.0%}로, Buy&Hold(약 {m_bh['MDD']:.0%})보다 작습니다.")
        verdict_line = (f"- 그래서 '누가 더 벌었나(CAGR)'가 아니라 **'위험 대비 얼마나 벌었나(Calmar·Sharpe)'** 로 "
                        f"봐야 하고, {'그 기준에선 '+STRAT_NAME+'가 앞섭니다.' if d_calmar > 0 else '이 점을 함께 봐야 합니다.'}")
    else:
        defend_line = (f"- 현재 사다리에선 최대 낙폭이 약 {m_pw['MDD']:.0%}로 Buy&Hold(약 {m_bh['MDD']:.0%})보다 "
                       "**큽니다** — 공격적 비중의 대가입니다.")
        verdict_line = "- 절대수익(CAGR)만이 아니라 **위험 대비 수익(Calmar·Sharpe)**과 최대 낙폭을 함께 봐야 합니다."
    st.markdown(
        invest_line + "\n" + defend_line + "\n" + verdict_line + "\n"
        "- 더 보수적으로 바꾸면 하락은 더 막지만 상승은 더 포기하고, 더 공격적(레버리지)으로 바꾸면 그 "
        "반대입니다. **정답은 내 위험 성향에 달려 있습니다.**")

st.caption("※ 2019~2021 구간은 모델 조기종료 선택에 일부 사용되어 약간 낙관적일 수 있습니다. "
           "엄밀한 검증(2022+ OOS·walk-forward)은 노트북을 참조하세요. "
           "모델의 검증된 강점은 하락 방어이며, 레버리지·공매도는 약한 상승 예측력을 손실로 증폭할 수 있습니다.")
