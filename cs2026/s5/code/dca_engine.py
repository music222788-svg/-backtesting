"""S5 적립식 엔진.

기존 engine_fast.FastBacktest 의 시점·비용·세금 규칙을 그대로 옮기고 다음을 추가한다.
  - 원화 현금흐름: 초기 투자금(원화)과 매월 첫 거래일 적립(원화). 적립일 DEXKOUS 환율로 환전,
    환전비용 편도 fx_cost, 같은 날 종가에 매수(매수 수수료 cost 는 취득원가 산입, 기존과 동일)
  - 적립금 배분: M1(미달 자산 우선) / M2(목표비중)
  - 세금: tax='usd' (기존 S4 방식: USD 양도차익, 공제 250만원/연말환율)
          tax='krw' (원화 양도차익: 취득가액은 취득일 환율, 양도가액은 양도일 환율, 이동평균법)
  - 배당(분배) 원화 합계 추적 (금융소득 종합과세 점검용, 원천징수 전 총액)
  - 스트레스: S1 위성 청산(liq_i 일에 레버리지 슬롯 전량 매도 -> SGOV), S2 매수 동결(freeze 구간)
적립금 0·초기 USD 투입·tax='usd' 이면 FastBacktest 와 같은 경로를 만든다(test_s5_dca.py 에서 검증).

같은 날 순서: 수익률 반영 -> (S1 청산) -> 적립 -> 체결(연말 리밸런싱) -> 신호 -> (종료일 청산) -> 연말 세금
"""
import numpy as np
import pandas as pd

from engine import initial_weights, tgt_of, band_of, sat_of


def calendar_struct(cal):
    cal = pd.DatetimeIndex(cal)
    T = len(cal)
    s = pd.Series(np.arange(T), index=cal)
    dec = s[cal.month == 12]
    sig = set(dec.groupby(dec.index.year).max().values) - {T-1}
    yend = set(s.groupby(cal.year).max().values)
    ym = cal.year*12 + cal.month
    contrib = np.r_[True, ym[1:] != ym[:-1]]
    return dict(cal=cal, T=T, is_sig=np.array([i in sig for i in range(T)]),
                is_yend=np.array([i in yend for i in range(T)]), is_contrib=contrib)


class Groups:
    """적립 배분용 그룹 구조 (최상위 코어/위성 -> 그룹 내부)."""

    def __init__(self, spec, A):
        self.A = A
        self.single = spec.top_mode == 'single'
        ct = tgt_of(spec)
        self.core_i = np.array([A.index(a) for a in spec.core], dtype=int)
        self.sat_i = np.array([A.index(a) for a in spec.sat], dtype=int)
        self.core_t = np.array([ct[a] for a in spec.core]) if spec.core else np.zeros(0)
        st = sat_of(spec) if spec.sat else {}
        self.sat_t = np.array([st[a] for a in spec.sat]) if spec.sat else np.zeros(0)
        self.core_w = spec.core_w if spec.core_w is not None else 1.0
        self.two = len(self.sat_i) > 0
        self.tq_in_sat = (list(np.array(spec.sat)[:]).index('TQQQ') if 'TQQQ' in spec.sat else -1)
        self.sg_in_sat = (list(spec.sat).index('SGOV') if 'SGOV' in spec.sat else -1)

    def sat_targets(self, frozen):
        t = self.sat_t.copy()
        if frozen and self.tq_in_sat >= 0:
            t[self.sg_in_sat] += t[self.tq_in_sat]
            t[self.tq_in_sat] = 0.0
        return t

    def full_w(self, frozen):
        n = len(self.A)
        w = np.zeros(n)
        if self.single:
            w[0] = 1.0
            return w
        w[self.core_i] = self.core_w*self.core_t
        if self.two:
            w[self.sat_i] = (1.0-self.core_w)*self.sat_targets(frozen)
        return w


def _split(A, d, w):
    D = d.sum()
    if D >= A and D > 0:
        return A*d/D
    return d + (A-D)*w/w.sum()


def allocate(G, v, A, method, frozen=False):
    """시장가치 A 를 자산별로 배분 (매수 수수료 제외 금액)."""
    n = len(v)
    if G.single:
        out = np.zeros(n); out[0] = A
        return out
    if method == 'M2':
        return A*G.full_w(frozen)
    Vp = v.sum() + A
    out = np.zeros(n)
    groups = [(G.core_i, G.core_w, G.core_t)]
    if G.two:
        groups.append((G.sat_i, 1.0-G.core_w, G.sat_targets(frozen)))
    gv = np.array([v[ix].sum() for ix, _, _ in groups])
    gw = np.array([w for _, w, _ in groups])
    ga = _split(A, np.maximum(0.0, gw*Vp-gv), gw) if len(groups) > 1 else np.array([A])
    for (ix, _, t), a, g in zip(groups, ga, gv):
        tg = (g+a)*t
        d = np.maximum(0.0, tg - v[ix])
        if t.sum() <= 0:
            continue
        # 동결 중 TQQQ(목표 0)는 부족분이 0 으로 계산되어 받지 않는다
        out[ix] = _split(a, d*(t > 0), t)
    return out


def simulate(spec, Rv, Dv, FXv, cs, *, init_usd=None, init_krw=None, contrib_krw=None,
             cost=0.0010, fx_cost=0.0010, tax='krw', tax_rate=0.22, ded_krw=2_500_000.0,
             method='M1', liq=True, s1_i=-1, freeze=None, record=False, tq_level=None):
    """Rv/Dv: (T,n) 수익률/배당수익률 (열 순서 = spec.assets), FXv: (T,) 원/달러, cs: calendar_struct.
    freeze: (hit_i, end_i) 또는 None — hit_i <= i < end_i 동안 TQQQ 매수 금지.
    tq_level: (T,) 레버리지 슬롯 가격지수 (행동지표용 낙폭 계산)."""
    A = list(spec.assets)
    n = len(A)
    T = cs['T']
    c = cost
    single = spec.top_mode == 'single'
    G = Groups(spec, A)
    ci, si = G.core_i, G.sat_i
    ct_ = tgt_of(spec)
    core_tgt = np.array([ct_[a] for a in spec.core]) if spec.core else np.zeros(0)
    inner_band = spec.top_mode not in ('absband', 'absband_top')
    cb = band_of(spec)
    band_lo = np.array([cb[a][0] for a in spec.core]) if spec.core else np.zeros(0)
    band_hi = np.array([cb[a][1] for a in spec.core]) if spec.core else np.zeros(0)
    tq = A.index('TQQQ') if 'TQQQ' in A else -1
    sg = A.index('SGOV') if 'SGOV' in A else -1
    w0d = initial_weights(spec)
    w0 = np.array([w0d.get(a, 0.0) for a in A])

    fxc_tot = 0.0
    paid_in = 0.0
    if init_usd is not None:
        u0 = float(init_usd)
    else:
        paid_in += init_krw
        fxc_tot += init_krw*fx_cost
        u0 = init_krw/FXv[0]*(1.0-fx_cost)
    v = u0*w0/(1.0+c)
    b = u0*w0.copy()
    bk = b*FXv[0]

    eq = np.empty(T)
    paid = np.empty(T) if record else None
    gy = 0.0; gk = 0.0; divk = 0.0
    years = []
    pend_i, pend_breach, pend_satw = -1, False, 0.0
    reb_sell_usd = 0.0; reb_sell_krw = 0.0; reb_gain_usd = 0.0; reb_gain_krw = 0.0
    tq_buy_max_krw, tq_buy_dd, tq_buy_i = 0.0, np.nan, -1
    tax_krw_tot = 0.0
    hit_i, end_i = freeze if freeze else (-1, -1)
    if tq_level is not None:
        tq_peak = np.maximum.accumulate(tq_level)
    for i in range(T):
        fx = FXv[i]
        if i > 0:
            divk += float((v*Dv[i]).sum())*fx
            v *= (1.0 + Rv[i])
        frozen = hit_i >= 0 and hit_i <= i < end_i
        if i == s1_i and tq >= 0 and v[tq] > 0:          # S1: 위성 청산 -> SGOV
            s = v[tq]
            gy += s - b[tq] - c*s
            gk += (s - c*s)*fx - bk[tq]
            pr = s*(1.0-c)
            v[sg] += pr/(1.0+c); b[sg] += pr; bk[sg] += pr*fx
            v[tq] = 0.0; b[tq] = 0.0; bk[tq] = 0.0
        if contrib_krw is not None and cs['is_contrib'][i] and contrib_krw[i] > 0:
            k = contrib_krw[i]
            paid_in += k
            fxc_tot += k*fx_cost
            usd = k/fx*(1.0-fx_cost)
            a = allocate(G, v, usd/(1.0+c), method, frozen)
            v += a
            b += a*(1.0+c)
            bk += a*(1.0+c)*fx
        if i == pend_i:
            V = v.sum()
            if single:
                t = v.copy()
            else:
                core_v = v[ci].sum()
                sat_v = v[si].sum() if len(si) else 0.0
                if spec.top_mode == 'restore':
                    core_t = spec.core_w*V
                    sat_t = V - core_t
                elif spec.top_mode == 'cap' and pend_satw > spec.sat_cap:
                    sat_t = spec.sat_cap*V
                    core_t = V - sat_t
                else:
                    core_t, sat_t = core_v, sat_v
                t = np.zeros(n)
                if core_v > 0 and not pend_breach:
                    t[ci] = core_t*(v[ci]/core_v)
                else:
                    t[ci] = core_t*core_tgt
                if len(si):
                    t[si] = sat_t*G.sat_t
                if frozen and tq >= 0:
                    keep = min(t[tq], v[tq])
                    t[sg] += t[tq]-keep
                    t[tq] = keep
            turn = float(np.maximum(0.0, v-t).sum())
            eps = 1e-10*max(V, 1.0)
            if turn > eps:
                scale = (V - 2.0*c*turn)/V
                new = t*scale
                d = new - v
                sell = d < -eps
                buy = d > eps
                vs = v[sell]
                s = -d[sell]
                frac = np.where(vs > 0, s/np.where(vs > 0, vs, 1.0), 0.0)
                g_u = float((s - b[sell]*frac - c*s).sum())
                g_k = float(((s - c*s)*fx - bk[sell]*frac).sum())
                gy += g_u; gk += g_k
                reb_sell_usd += float(s.sum()); reb_sell_krw += float(s.sum())*fx
                reb_gain_usd += g_u; reb_gain_krw += g_k
                b[sell] = b[sell]*(1.0-frac)
                bk[sell] = bk[sell]*(1.0-frac)
                b[buy] = b[buy] + d[buy]*(1.0+c)
                bk[buy] = bk[buy] + d[buy]*(1.0+c)*fx
                if tq >= 0 and buy[tq]:
                    amt = d[tq]*(1.0+c)*fx
                    if amt > tq_buy_max_krw:
                        tq_buy_max_krw = amt
                        tq_buy_i = i
                        if tq_level is not None:
                            tq_buy_dd = tq_level[i]/tq_peak[i]-1.0
                v = new
            pend_i = -1
        if spec.rebalance and cs['is_sig'][i]:
            V = v.sum()
            breach = False
            if len(ci) and inner_band:
                cv = v[ci].sum()
                if cv > 0:
                    w = v[ci]/cv
                    breach = bool(((w < band_lo) | (w > band_hi)).any())
            pend_satw = float(v[si].sum()/V) if (len(si) and V > 0) else 0.0
            pend_breach = breach
            pend_i = i+1 if i+1 < T else -1
        if i == T-1 and liq:
            s = v.copy()
            gy += float((s - b - c*s).sum())
            gk += float(((s - c*s)*fx).sum() - bk.sum())
            b[:] = 0.0; bk[:] = 0.0
            v = s*(1.0-c)
        if tax and (cs['is_yend'][i] or i == T-1):
            if tax == 'usd':
                tx = tax_rate*max(0.0, gy - ded_krw/fx)
            else:
                tx = tax_rate*max(0.0, gk - ded_krw)/fx
            V = v.sum()
            p = 0.0
            if tx > 0 and V > 0:
                p = min(tx, V)
                f = p/V
                v = v*(1.0-f); b = b*(1.0-f); bk = bk*(1.0-f)
            tax_krw_tot += p*fx
            if record:
                years.append(dict(year=int(cs['cal'][i].year), gain_usd=gy, gain_krw=gk,
                                  tax_usd=p, tax_krw=p*fx, div_krw=divk, fx=fx))
            gy = 0.0; gk = 0.0; divk = 0.0
        eq[i] = v.sum()
        if record:
            paid[i] = paid_in
    out = dict(eq=eq, paid_in=paid_in, fx_cost=fxc_tot, tax_krw=tax_krw_tot,
               reb_sell_usd=reb_sell_usd, reb_sell_krw=reb_sell_krw,
               reb_gain_usd=reb_gain_usd, reb_gain_krw=reb_gain_krw,
               tq_buy_max_krw=tq_buy_max_krw, tq_buy_dd=tq_buy_dd,
               tq_buy_date=(str(cs['cal'][tq_buy_i].date()) if tq_buy_i >= 0 else ''))
    if record:
        out['paid'] = paid
        out['years'] = pd.DataFrame(years)
    return out
