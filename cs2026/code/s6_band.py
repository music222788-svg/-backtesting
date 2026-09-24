"""P3 연말 리밸런싱 규칙 비교 — 밴드 조건부 vs 무조건 복원.

현행 P3 는 최상위(코어:위성)와 위성 내부(60/40)를 밴드 없이 매년 무조건 복원한다.
여기서는 세 그룹 각각에 5/25 밴드를 적용했을 때의 차이를 본다.

그룹별 밴드
  최상위    : 위성 비중 25~35% (30±5pp)
  코어 내부 : SPY 55~65, SCHD 15~25, IEF 7.5~12.5, GLD 7.5~12.5 (원 조건 그대로)
  위성 내부 : TQQQ 55~65% (60±5pp)

변형
  V1 현행        : 최상위 무조건 · 코어 밴드 · 위성내부 무조건   (기존 P3 와 동일해야 함)
  V2 밴드-부분   : 세 그룹 모두 밴드, 이탈한 그룹만 조정
  V3 밴드-전체   : 세 그룹 모두 밴드, 하나라도 이탈하면 전부 목표로 복원
  V4 최상위만밴드: 최상위 밴드 · 코어 밴드 · 위성내부 무조건
  V5 위성만밴드  : 최상위 무조건 · 코어 밴드 · 위성내부 밴드
  V6 무리밸런싱  : 연말에 아무것도 하지 않음 (극단 대조군)

시점·비용·세금 규칙은 0단계 확정 조건과 동일.
"""
import numpy as np, pandas as pd

CORE = ['SPY', 'SCHD', 'IEF', 'GLD']
SAT = ['TQQQ', 'SGOV']
ASSETS = CORE + SAT
CORE_TGT = np.array([.60, .20, .10, .10])
CORE_LO = np.array([.55, .15, .075, .075])
CORE_HI = np.array([.65, .25, .125, .125])
SAT_TGT = np.array([.60, .40])
SAT_IN_LO, SAT_IN_HI = .55, .65        # 위성 내부 TQQQ 밴드
TOP_LO, TOP_HI = .25, .35              # 최상위 위성비중 밴드
CORE_W, SAT_W = 0.70, 0.30
COST, TAX_RATE, DED_KRW = 0.0010, 0.22, 2_500_000.0

VARIANTS = {
    'V1 현행(최상위·위성 무조건)': dict(top='always', core='band', sat='always', joint=False),
    'V2 밴드-부분':               dict(top='band',   core='band', sat='band',   joint=False),
    'V3 밴드-전체':               dict(top='band',   core='band', sat='band',   joint=True),
    'V4 최상위만 밴드':            dict(top='band',   core='band', sat='always', joint=False),
    'V5 위성내부만 밴드':          dict(top='always', core='band', sat='band',   joint=False),
    'V6 무리밸런싱':              dict(top='never',  core='never', sat='never',  joint=False),
    # V7: 현행 규칙을 그대로 쓰되 '아무 그룹도 이탈하지 않은 해'에만 매매를 건너뛴다.
    #     V1 과 오직 그 해에서만 달라지므로 질문을 정확히 분리한다.
    'V7 무이탈 해만 건너뜀':        dict(top='ifany',  core='band', sat='ifany',  joint=False),
}


class BandP3:
    def __init__(self, rets, fx, start, end, rule, init=10_000.0, tax=False,
                 liquidate=True, tqqq_zero_date=None, band_scale=1.0):
        # band_scale: 최상위(30±5pp)·위성내부(60±5pp) 밴드 폭 배수. 코어 밴드는 원 조건 고정.
        self.top_lo = max(0.0, SAT_W - 0.05*band_scale)
        self.top_hi = min(1.0, SAT_W + 0.05*band_scale)
        self.sin_lo = max(0.0, 0.60 - 0.05*band_scale)
        self.sin_hi = min(1.0, 0.60 + 0.05*band_scale)
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        self.R = rets.loc[start:end, ASSETS]        # 구간 밖 데이터는 넘기지 않는다
        self.FX = fx.loc[start:end]
        self.cal = self.R.index
        assert self.cal[0] == start
        self.rule = rule
        self.init, self.tax_on, self.liq = float(init), tax, liquidate
        self.tz = pd.Timestamp(tqqq_zero_date) if tqqq_zero_date else None
        self.sat_tgt = np.array([0.0, 1.0]) if self.tz is not None else SAT_TGT

        s = pd.Series(self.cal, index=self.cal)
        self.yend = set(s.groupby(s.dt.year).max())
        dec = s[s.dt.month == 12]
        self.sig = set(d for d in dec.groupby(dec.dt.year).max() if d != self.cal[-1])
        self.Rv = np.nan_to_num(self.R.to_numpy(dtype=float), nan=0.0)
        self.FXv = self.FX.to_numpy(dtype=float)
        self.ci, self.si = np.arange(4), np.arange(4, 6)
        self.w0 = np.concatenate([CORE_W*CORE_TGT, SAT_W*SAT_TGT])
        self.is_sig = np.array([d in self.sig for d in self.cal])
        self.is_ye = np.array([d in self.yend for d in self.cal])

    def _flags(self, v):
        V = v.sum()
        cv, sv = v[self.ci].sum(), v[self.si].sum()
        sw = sv/V if V > 0 else 0.0
        top_b = bool(sw < self.top_lo or sw > self.top_hi)
        core_b = False
        if cv > 0:
            w = v[self.ci]/cv
            core_b = bool(((w < CORE_LO) | (w > CORE_HI)).any())
        sat_b = False
        if sv > 0 and self.tz is None:
            tw = v[4]/sv
            sat_b = bool(tw < self.sin_lo or tw > self.sin_hi)
        elif sv > 0:
            sat_b = bool(v[4]/sv > 1e-12)      # TQQQ=0 시나리오
        return top_b, core_b, sat_b, sw

    def _do(self, kind, breach, anyb):
        m = self.rule[kind]
        if m == 'always':
            return True
        if m == 'never':
            return False
        if m == 'ifany':
            return anyb
        return anyb if self.rule['joint'] else breach

    def run(self, keep_holdings=False):
        c = COST
        v = self.init*self.w0/(1.0+c)
        b = self.init*self.w0.copy()
        T = len(self.cal)
        eq = np.empty(T)
        H = np.empty((T, 6)) if keep_holdings else None
        trades, taxes, sigs = [], [], []
        pend = None
        gy = 0.0
        zi = int(self.cal.searchsorted(self.tz)) if self.tz is not None else -1
        zeroed = False

        for i in range(T):
            if i > 0:
                v = v*(1.0 + self.Rv[i])
            if zi >= 0 and not zeroed and i >= zi:
                gy += -b[4]; v[4] = 0.0; b[4] = 0.0; zeroed = True

            if pend is not None and pend[0] == i:
                dt, dc, ds = pend[1]
                V = v.sum()
                cv, sv = v[self.ci].sum(), v[self.si].sum()
                core_t = CORE_W*V if dt else cv
                sat_t = V - core_t
                t = np.zeros(6)
                t[self.ci] = core_t*(CORE_TGT if dc or cv <= 0 else v[self.ci]/cv)
                t[self.si] = sat_t*(self.sat_tgt if ds or sv <= 0 else v[self.si]/sv)
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
                    trades.append(dict(date=self.cal[i], n=int(sell.sum()+buy.sum()),
                                       turnover=turn/V, realized=gy))
                pend = None

            if self.is_sig[i] and i+1 < T:
                tb, cb, sb, sw = self._flags(v)
                anyb = tb or cb or sb
                acts = (self._do('top', tb, anyb), self._do('core', cb, anyb),
                        self._do('sat', sb, anyb))
                pend = (i+1, acts)
                sigs.append(dict(date=self.cal[i], sat_w=sw, top_breach=tb,
                                 core_breach=cb, sat_breach=sb, any_breach=anyb,
                                 acted=any(acts)))

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
                    paid = min(tx, V); f = paid/V
                    v = v*(1.0-f); b = b*(1.0-f)
                taxes.append(dict(date=self.cal[i], realized=gy, tax=paid))
                gy = 0.0

            eq[i] = v.sum()
            if keep_holdings:
                H[i] = v

        return dict(equity=pd.Series(eq, index=self.cal),
                    holdings=(pd.DataFrame(H, index=self.cal, columns=ASSETS)
                              if keep_holdings else None),
                    trades=pd.DataFrame(trades), taxes=pd.DataFrame(taxes),
                    signals=pd.DataFrame(sigs))
