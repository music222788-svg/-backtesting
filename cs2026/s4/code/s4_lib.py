"""S4 공통 모듈: 데이터 로드, 포트폴리오 Spec 생성, 실행, 지표.

기존 엔진(code/engine.py, engine_fast.py)을 수정 없이 사용한다.
  - B 계열(SCHD/SPY/QQQ 고정비중, 밴드 없이 매년 전량 복원):
      Spec(top_mode='none', core_w=1.0) + 코어 밴드를 [목표, 목표] 로 두어
      점검일에 조금이라도 목표와 다르면 '이탈' -> 체결일에 목표비중으로 전량 복원.
  - A 계열(코어·위성): P3 와 같은 Spec(top_mode='restore') 에 코어 목표/밴드만 교체.
      밴드 5/25 규칙 = 목표 ± min(5%p, 목표의 25%) (P3: SPY 55~65, SCHD 15~25, IEF/GLD 7.5~12.5 와 일치)
  - QLD 위성: 엔진의 레버리지 슬롯('TQQQ' 열)에 QLD 수익률을 넣어 실행한다.
      (엔진의 TQQQ=0 스트레스 로직을 QLD 에 그대로 적용하기 위함. 결과 표기는 QLD)
"""
import sys, pathlib
import numpy as np, pandas as pd

S4 = pathlib.Path(__file__).resolve().parent.parent
ROOT = S4.parent
sys.path.insert(0, str(ROOT/'code'))
from engine import Spec, CORE_ASSETS            # noqa: E402
from engine_fast import FastBacktest            # noqa: E402
import metrics as M                              # noqa: E402

RB = S4/'rebuilt'
PER = {'A': ('2000-01-03', '2019-12-31'), 'B': ('2020-01-02', '2026-09-16'),
       'C': ('2000-01-03', '2026-09-16'),
       'S1': ('2000-01-03', '2012-12-31'), 'S2': ('2013-01-02', '2021-12-31'),
       'S3': ('2022-01-03', '2026-09-16'), 'OOS2': ('2013-01-02', '2026-09-16')}
PER_LABEL = {'A': 'A 2000–2019', 'B': 'B 2020–2026', 'C': 'C 2000–2026',
             'S1': '2000–2012', 'S2': '2013–2021', 'S3': '2022–2026', 'OOS2': '2013–2026'}

_R = pd.read_csv(RB/'s4_returns.csv', parse_dates=['date']).set_index('date')
_P = pd.read_csv(RB/'s1_proxy_returns.csv', parse_dates=['date']).set_index('date')
_MAC = pd.read_csv(RB/'s1_macro.csv', parse_dates=['date']).set_index('date')
_SYN = pd.read_csv(RB/'s4_syn.csv', parse_dates=['date']).set_index('date')
FX = _MAC['krw']
RF = _P['SGOV_syn']
CAL = _R.index


def returns(wh=True, cal=False, lev='TQQQ'):
    """엔진 열 이름(SPY,QQQ,SCHD,IEF,GLD,TQQQ,SGOV)으로 된 수익률 표.
    lev='QLD' 이면 'TQQQ' 열에 QLD 수익률을 넣는다."""
    suf = '_wh' if wh else ''
    src = {'SPY': 'SPY', 'QQQ': 'QQQ', 'SCHD': 'SCHD', 'IEF': 'IEF', 'GLD': 'GLD', 'SGOV': 'SGOV',
           'TQQQ': (lev + ('_cal' if cal else ''))}
    return pd.DataFrame({k: _R[v+suf] for k, v in src.items()})


def syn_series(L=3, cal=False):
    c = ('TQQQ' if L == 3 else 'QLD') + '_syn' + ('_cal' if cal else '')
    return _SYN[c]


def band525(t):
    return {a: (w - min(0.05, 0.25*w), w + min(0.05, 0.25*w)) for a, w in t.items()}


def spec_b(w, name=None):
    """w: {'SCHD':..,'SPY':..,'QQQ':..} 합 1. 0 인 자산은 제외."""
    w = {a: float(v) for a, v in w.items() if v > 1e-12}
    assets = [a for a in ['SCHD', 'SPY', 'QQQ'] if a in w]
    name = name or 'B ' + ':'.join(f'{a}{round(w.get(a, 0)*100)}' for a in ['SCHD', 'SPY', 'QQQ'])
    return Spec(name, assets, assets, [], 1.0, 'none',
                core_tgt={a: w[a] for a in assets},
                core_band={a: (w[a], w[a]) for a in assets})


def spec_cs(name, core_tgt, core_w, sat_tgt):
    core = list(core_tgt)
    sat = list(sat_tgt)
    return Spec(name, core+sat, core, sat, core_w, 'restore',
                core_tgt=dict(core_tgt), core_band=band525(core_tgt), sat_tgt=dict(sat_tgt))


def spec_single(a):
    return Spec(f'{a} 100%', [a], [], [], None, 'single', rebalance=False)


def spec_c1():
    t = {'SPY': .60, 'SCHD': .20, 'IEF': .10, 'GLD': .10}
    return Spec('코어100', CORE_ASSETS, CORE_ASSETS, [], 1.0, 'none', core_tgt=t, core_band=band525(t))


def run(spec, rets, per, tax=False, zero=None, keep=False):
    s, e = PER[per] if per in PER else per
    return FastBacktest(spec, rets, FX, s, e, tax=tax, tqqq_zero_date=zero).run(keep_holdings=keep)


def full_metrics(pre, post):
    eq, eqp = pre['equity'], post['equity']
    d = M.summarize('', eq, eqp, FX, RF, pre['trades'], post['taxes'])
    d.pop('전략')
    d['Sharpe_세후'] = M.sharpe(eqp, RF)
    d['최악롤링10년'] = M.worst_rolling_cagr(eq, 10)
    d['최악롤링3년_세후'] = M.worst_rolling_cagr(eqp, 3)
    return d


def evaluate(spec, rets, per, zero=None):
    pre = run(spec, rets, per, tax=False, zero=zero)
    post = run(spec, rets, per, tax=True, zero=zero)
    return full_metrics(pre, post), pre, post


def monthly_starts(first='2000-01-01', years=10, end='2026-09-16'):
    s = pd.Series(CAL, index=CAL)
    fm = s.groupby([CAL.year, CAL.month]).min()
    out = []
    for d in fm:
        if d < pd.Timestamp(first):
            continue
        tgt = d + pd.DateOffset(years=years)
        if tgt > pd.Timestamp(end) + pd.Timedelta(days=1):
            break
        e = CAL[CAL < tgt][-1]
        out.append((d, e))
    return out


def rolling10_after_tax(spec, rets, starts):
    vals = []
    for s, e in starts:
        post = FastBacktest(spec, rets, FX, s, e, tax=True).run(keep_holdings=False)
        vals.append(M.cagr(post['equity']))
    return np.array(vals)


BLOCK, NBOOT, SEED = 252, 1000, 20260917       # 기존 S3 와 동일


def boot_indices(per):
    """기존 s3_boot.py 와 같은 난수 흐름으로 블록 시작점을 만든다."""
    s, e = PER[per]
    cal = CAL[(CAL >= pd.Timestamp(s)) & (CAL <= pd.Timestamp(e))]
    T = len(cal)
    nblk = int(np.ceil(T/BLOCK))
    rng = np.random.default_rng(SEED)
    idxs = []
    for _ in range(NBOOT):
        st = rng.integers(0, T-BLOCK+1, size=nblk)
        idxs.append(np.concatenate([np.arange(x, x+BLOCK) for x in st])[:T])
    return cal, idxs
