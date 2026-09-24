"""S5 공통: 데이터, 포트폴리오, 창(window) 실행, 적립식 지표, 금 드리프트 조정."""
import sys, pathlib, json
import numpy as np, pandas as pd
from scipy.optimize import brentq

S5 = pathlib.Path(__file__).resolve().parent.parent
ROOT = S5.parent
sys.path.insert(0, str(ROOT/'s4'/'code'))
sys.path.insert(0, str(S5/'code'))
import s4_lib as L                                   # noqa: E402  (기존 S4 정의 재사용)
import metrics as M                                  # noqa: E402
from dca_engine import simulate, calendar_struct     # noqa: E402

RB4 = ROOT/'s4'/'rebuilt'
_R = pd.read_csv(RB4/'s4_returns.csv', parse_dates=['date']).set_index('date')
_DV = pd.read_csv(RB4/'s4_divyield.csv', parse_dates=['date']).set_index('date')
CAL_ALL = _R.index
FX_ALL = L.FX.reindex(CAL_ALL).ffill()
T0 = pd.Timestamp('2000-01-03')
TEND = pd.Timestamp('2026-09-16')

INIT_KRW = 150_000_000.0
MONTHLY_KRW = 4_500_000.0
GOAL, GOAL2 = 2_500_000_000.0, 1_500_000_000.0

# 한국 CPI: s5/data_ext/KR_CPI.csv (date,value 월간) 가 있으면 실질 모드, 없으면 명목 고정
KR_CPI_FILE = S5/'data_ext'/'KR_CPI.csv'
REAL_MODE = KR_CPI_FILE.exists()

BOPT = {'SCHD': .15, 'SPY': .55, 'QQQ': .30}
CORE_P3 = {'SPY': .60, 'SCHD': .20, 'IEF': .10, 'GLD': .10}
SAT_TQ = {'TQQQ': .60, 'SGOV': .40}
PORTS = {
    'P3':    (L.spec_cs('P3', CORE_P3, .70, SAT_TQ), 'TQQQ'),
    'A80':   (L.spec_cs('A80', CORE_P3, .80, SAT_TQ), 'TQQQ'),
    'C100':  (L.spec_c1(), None),
    'B-opt': (L.spec_b(BOPT, 'B-opt'), None),
    'B1':    (L.spec_b({'SCHD': .2, 'SPY': .3, 'QQQ': .5}, 'B1'), None),
    'SPY':   (L.spec_single('SPY'), None),
    'QQQ':   (L.spec_single('QQQ'), None),
}
LABEL = {'C100': '코어100', 'SPY': 'SPY100', 'QQQ': 'QQQ100'}
GOLD_PORTS = ['P3', 'A80', 'C100']


def frames(wh=True, lev='TQQQ', gold_drift=0.0):
    """엔진 열 이름의 수익률·배당수익률 표 (전체 캘린더)."""
    R = L.returns(wh=wh, cal=False, lev=lev).reindex(CAL_ALL)
    D = pd.DataFrame({k: _DV[{'TQQQ': lev}.get(k, k)] for k in R.columns}).reindex(CAL_ALL).fillna(0.0)
    if gold_drift:
        R['GLD'] = R['GLD'] + gold_drift
    return R, D


def window(s, e):
    m = (CAL_ALL >= pd.Timestamp(s)) & (CAL_ALL <= pd.Timestamp(e))
    return CAL_ALL[m]


def arrays(spec, R, D, cal):
    A = list(spec.assets)
    Rv = np.nan_to_num(R.loc[cal, A].to_numpy(dtype=float), nan=0.0)
    Dv = np.nan_to_num(D.loc[cal, A].to_numpy(dtype=float), nan=0.0)
    return Rv, Dv


def contrib_series(cs, monthly=MONTHLY_KRW):
    return np.where(cs['is_contrib'], monthly, 0.0)


def xirr(dates, flows):
    t = np.array([(d - dates[0]).days/365.0 for d in dates])
    f = np.array(flows, dtype=float)

    def npv(r):
        return float((f/(1.0+r)**t).sum())
    try:
        return brentq(npv, -0.99, 2.0)
    except ValueError:
        return np.nan


def dca_metrics(out, cs, FXv):
    cal = cs['cal']
    val = out['eq']*FXv                          # 원화 평가액 (명목)
    paid = out['paid']
    under = val < paid
    best = cur = 0
    for u in under:
        cur = cur+1 if u else 0
        best = max(best, cur)
    loss = (paid - val).max()
    reach = np.nonzero(val >= GOAL)[0]
    flows_d = [cal[0]] + [cal[i] for i in np.nonzero(cs['is_contrib'])[0]] + [cal[-1]]
    flows = [-INIT_KRW] + [-MONTHLY_KRW]*int(cs['is_contrib'].sum()) + [val[-1]]
    dd = val/np.maximum.accumulate(val) - 1.0
    # 흐름 중립 단위가치(TWR) 낙폭: 적립 직전 가치로 연결
    u = np.empty(len(val)); u[0] = 1.0
    add = np.where(cs['is_contrib'], MONTHLY_KRW, 0.0); add[0] = 0.0
    for i in range(1, len(val)):
        u[i] = u[i-1]*(val[i]-add[i])/val[i-1]
    return dict(최종자산_명목=val[-1], 누적납입=paid[-1], 누적세금=out['tax_krw'], 환전비용=out['fx_cost'],
                XIRR_명목=xirr(flows_d, flows), 평가액MDD=dd.min(), 단위가치MDD=(u/np.maximum.accumulate(u)-1).min(),
                원금하회_최장거래일=int(best), 원금대비_최대평가손실=max(0.0, loss),
                목표25억_최초도달=(str(cal[reach[0]].date()) if len(reach) else '미도달'),
                리밸런싱매도_원화=out['reb_sell_krw'], 리밸런싱실현이익_원화=out['reb_gain_krw'],
                TQQQ연말최대매수_원화=out['tq_buy_max_krw'], 그때_TQQQ고점대비=out['tq_buy_dd'],
                TQQQ연말최대매수_일자=out['tq_buy_date'])


def syn3_first_below(cal, thr=0.01):
    s = L.syn_series(3, False).reindex(cal).fillna(0.0).to_numpy()
    w = np.cumprod(1.0+s)
    k = np.nonzero(w < thr)[0]
    return int(k[0]) if len(k) else -1


def freeze_window(level, cal):
    """레버리지 슬롯 가격지수가 직전 고점 대비 -90% 처음 도달 -> 36개월 후 첫 연말 체결일까지."""
    peak = np.maximum.accumulate(level)
    k = np.nonzero(level/peak <= 0.10)[0]
    if not len(k):
        return None
    h = int(k[0])
    lim = cal[h] + pd.DateOffset(months=36)
    cs = calendar_struct(cal)
    ex = [i+1 for i in np.nonzero(cs['is_sig'])[0] if i+1 < len(cal) and cal[i+1] >= lim]
    return (h, ex[0] if ex else len(cal))


def gold_drift_for(target_cagr, R):
    g = R['GLD'].loc[T0:TEND].fillna(0.0).to_numpy()[1:]
    n = len(g)

    def f(dl):
        return np.exp(np.log1p(g+dl).sum()*252.0/n) - 1.0 - target_cagr
    return brentq(f, -0.01, 0.01)
