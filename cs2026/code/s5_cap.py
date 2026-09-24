"""P3 + 연중 위성 비중 상한 — 독립 구현.

engine.py 를 수정하지 않는다 (다른 전략용 밴드 로직과 섞이지 않도록).
CAP=None 으로 두면 기존 P3 와 완전히 동일해야 하며, self-test 가 이를 검증한다.

규칙
----
[연말]  12월 마지막 거래일 종가 점검 -> 다음 거래일 종가 체결
        · 최상위 코어:위성 = 70:30 복원
        · 코어 내부: 5/25 밴드 이탈 시 60/20/10/10 복원, 아니면 비례 유지
        · 위성 내부: 60/40 무조건 복원
[연중]  점검 주기(daily|quarter)의 점검일 종가에 위성비중 > CAP 이면
        -> 다음 거래일 종가에 최상위 비중만 조정
        · 조치 'target' : 위성을 30% 로
        · 조치 'cap'    : 위성을 CAP 으로
        · 코어/위성 내부 비중은 비례 유지 (상한 효과만 분리)
        · 연말 점검일은 연중 점검에서 제외 (연말 규칙 우선)

비용·세금은 0단계 확정 조건과 동일.
"""
import sys, pathlib
import numpy as np, pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent
OUT = ROOT/'out'

CORE = ['SPY', 'SCHD', 'IEF', 'GLD']
SAT = ['TQQQ', 'SGOV']
ASSETS = CORE + SAT
CORE_TGT = np.array([.60, .20, .10, .10])
CORE_LO = np.array([.55, .15, .075, .075])
CORE_HI = np.array([.65, .25, .125, .125])
SAT_TGT = np.array([.60, .40])
CORE_W = 0.70
SAT_W = 0.30
COST = 0.0010
TAX_RATE = 0.22
DED_KRW = 2_500_000.0


def _last_of(cal, by):
    s = pd.Series(cal, index=cal)
    return set(s.groupby(by(s)).max())


class CapP3:
    def __init__(self, rets, fx, start, end, cap=None, freq='daily', action='target',
                 init=10_000.0, tax=False, liquidate=True, tqqq_zero_date=None):
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        self.R = rets.loc[start:end, ASSETS]          # 구간 밖 데이터는 넘기지 않는다
        self.FX = fx.loc[start:end]
        self.cal = self.R.index
        assert self.cal[0] == start
        self.end = self.cal[-1]
        self.cap, self.action = cap, action
        self.init, self.tax_on, self.liq = float(init), tax, liquidate
        self.tz = pd.Timestamp(tqqq_zero_date) if tqqq_zero_date else None
        # TQQQ=0 시나리오에서는 위성 목표가 SGOV 100% 가 된다 (기존 엔진과 동일)
        self.sat_tgt = np.array([0.0, 1.0]) if self.tz is not None else SAT_TGT

        self.yend = _last_of(self.cal, lambda s: s.dt.year)
        dec = pd.Series(self.cal, index=self.cal)
        dec = dec[dec.dt.month == 12]
        self.sig_year = set(d for d in dec.groupby(dec.dt.year).max() if d != self.end)
        if freq == 'daily':
            mid = set(self.cal)
        elif freq == 'quarter':
            mid = _last_of(self.cal, lambda s: [s.dt.year, s.dt.quarter])
        elif freq == 'month':
            mid = _last_of(self.cal, lambda s: [s.dt.year, s.dt.month])
        else:
            raise ValueError(freq)
        # 연말 점검일과 마지막 날은 연중 점검에서 제외
        self.sig_mid = set(d for d in mid if d not in self.sig_year and d != self.end)

        self.Rv = np.nan_to_num(self.R.to_numpy(dtype=float), nan=0.0)
        self.FXv = self.FX.to_numpy(dtype=float)
        self.ci = np.arange(4)
        self.si = np.arange(4, 6)
        self.w0 = np.concatenate([CORE_W*CORE_TGT, SAT_W*SAT_TGT])
        self.is_y = np.array([d in self.sig_year for d in self.cal])
        self.is_m = np.array([d in self.sig_mid for d in self.cal])
        self.is_ye = np.array([d in self.yend for d in self.cal])

    def run(self, keep_holdings=False):
        c = COST
        v = self.init*self.w0/(1.0+c)
        b = self.init*self.w0.copy()
        T = len(self.cal)
        eq = np.empty(T)
        H = np.empty((T, 6)) if keep_holdings else None
        trades, taxes = [], []
        pend = None                     # (kind, breach) ; kind in {'year','mid'}
        gy = 0.0
        zi = int(self.cal.searchsorted(self.tz)) if self.tz is not None else -1
        zeroed = False

        for i in range(T):
            if i > 0:
                v = v*(1.0 + self.Rv[i])
            if zi >= 0 and not zeroed and i >= zi:
                gy += -b[4]; v[4] = 0.0; b[4] = 0.0; zeroed = True

            if pend is not None and pend[0] == i:
                kind, breach = pend[1], pend[2]
                V = v.sum()
                core_v, sat_v = v[self.ci].sum(), v[self.si].sum()
                t = np.zeros(6)
                if kind == 'year':
                    core_t, sat_t = CORE_W*V, SAT_W*V
                    t[self.ci] = core_t*CORE_TGT if breach else core_t*(v[self.ci]/core_v)
                    t[self.si] = sat_t*self.sat_tgt
                else:                                   # 연중 상한 조치
                    sat_t = (SAT_W if self.action == 'target' else self.cap)*V
                    core_t = V - sat_t
                    t[self.ci] = core_t*(v[self.ci]/core_v) if core_v > 0 else core_t*CORE_TGT
                    t[self.si] = sat_t*(v[self.si]/sat_v) if sat_v > 0 else sat_t*self.sat_tgt
                turn = float(np.maximum(0.0, v-t).sum())
                eps = 1e-10*max(V, 1.0)
                if turn > eps:
                    scale = (V - 2.0*c*turn)/V
                    new = t*scale
                    d = new - v
                    sell, buy = d < -eps, d > eps
                    vs, s = v[sell], -d[sell]
                    frac = np.where(vs > 0, s/np.where(vs > 0, vs, 1.0), 0.0)
                    gy += float((s - b[sell]*frac - c*s).sum())
                    b[sell] = b[sell]*(1.0-frac)
                    b[buy] = b[buy] + d[buy]*(1.0+c)
                    v = new
                    trades.append(dict(date=self.cal[i], kind=kind,
                                       n=int(sell.sum()+buy.sum()), realized=gy))
                pend = None

            # ---- 점검 (종가 기준, 이후 데이터 일절 미사용) ----
            if pend is None and i+1 < T:
                V = v.sum()
                if self.is_y[i]:
                    cv = v[self.ci].sum()
                    br = bool(cv > 0 and (((v[self.ci]/cv) < CORE_LO)
                                          | ((v[self.ci]/cv) > CORE_HI)).any())
                    pend = (i+1, 'year', br)
                elif self.cap is not None and self.is_m[i]:
                    if V > 0 and v[self.si].sum()/V > self.cap:
                        pend = (i+1, 'mid', False)

            if i == T-1 and self.liq:
                s = v.copy()
                gy += float((s - b - c*s).sum())
                b[:] = 0.0
                v = s*(1.0-c)

            if self.tax_on and (self.is_ye[i] or i == T-1):
                tx = TAX_RATE*max(0.0, gy - DED_KRW/self.FXv[i])
                V = v.sum()
                paid = 0.0
                if tx > 0 and V > 0:
                    paid = min(tx, V)
                    f = paid/V
                    v = v*(1.0-f); b = b*(1.0-f)
                taxes.append(dict(date=self.cal[i], realized=gy, tax=paid))
                gy = 0.0

            eq[i] = v.sum()
            if keep_holdings:
                H[i] = v

        return dict(equity=pd.Series(eq, index=self.cal),
                    holdings=(pd.DataFrame(H, index=self.cal, columns=ASSETS)
                              if keep_holdings else None),
                    trades=pd.DataFrame(trades), taxes=pd.DataFrame(taxes))
