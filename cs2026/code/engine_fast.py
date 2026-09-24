"""고속 구현 (블록 부트스트랩·시작일 스캔용).

로직은 engine.Backtest.run 과 동일해야 한다.
code/test_s3_fast.py 가 참조 구현과의 완전 일치를 검증한다.
"""
import numpy as np
import pandas as pd

from engine import (Backtest, SPECS, CORE_TGT, CORE_BAND, SAT_TGT, initial_weights,
                    tgt_of, band_of, sat_of)


class FastBacktest(Backtest):
    def _prep(self):
        sp = self.spec
        self.A = list(sp.assets)
        self.n = len(self.A)
        self.ci = np.array([self.A.index(a) for a in sp.core], dtype=int)
        self.si = np.array([self.A.index(a) for a in sp.sat], dtype=int)
        ct = self._drop_tqqq(tgt_of(sp), sp.core)
        self.core_tgt = np.array([ct[a] for a in sp.core]) if sp.core else np.zeros(0)
        # absband 계열은 그룹 내부 밴드 규칙이 없다 -> 절대 발동하지 않는 밴드를 넣는다
        self.inner_band = sp.top_mode not in ('absband', 'absband_top')
        cb = band_of(sp) if self.inner_band else None
        self.band_lo = (np.array([cb[a][0] for a in sp.core]) if (sp.core and self.inner_band)
                        else np.zeros(len(sp.core)))
        self.band_hi = (np.array([cb[a][1] for a in sp.core]) if (sp.core and self.inner_band)
                        else np.ones(len(sp.core)))
        if sp.sat:
            sw = self._drop_tqqq(sat_of(sp), sp.sat)
            self.sat_tgt = np.array([sw[a] for a in sp.sat])
        else:
            self.sat_tgt = np.zeros(0)
        self.Rv = np.nan_to_num(self.R[self.A].to_numpy(dtype=float), nan=0.0)
        self.FXv = self.FX.to_numpy(dtype=float)
        w = initial_weights(sp)
        self.w0 = np.array([w.get(a, 0.0) for a in self.A])
        self.tq = self.A.index('TQQQ') if 'TQQQ' in self.A else -1
        self.is_sig = np.array([d in self.sig_days for d in self.cal])
        self.is_yend = np.array([d in self.yend_days for d in self.cal])

    def run(self, keep_holdings=True):
        if not hasattr(self, 'Rv'):
            self._prep()
        sp = self.spec
        c = self.cost
        single = (sp.top_mode == 'single')
        v = self.init*self.w0/(1.0+c)
        b = self.init*self.w0.copy()
        T = len(self.cal)
        eq = np.empty(T)
        H = np.empty((T, self.n)) if keep_holdings else None
        trades, tax_rows = [], []
        pend_i, pend_breach, pend_satw, pend_top = -1, False, 0.0, False
        gy = 0.0
        zeroed = False
        zi = -1
        if self.tqqq_zero_date is not None and self.tq >= 0:
            zi = int(self.cal.searchsorted(self.tqqq_zero_date))

        for i in range(T):
            if i > 0:
                v *= (1.0 + self.Rv[i])

            if zi >= 0 and not zeroed and i >= zi:
                gy += -b[self.tq]
                v[self.tq] = 0.0
                b[self.tq] = 0.0
                zeroed = True

            if i == pend_i:
                V = v.sum()
                if single:
                    t = v.copy()
                else:
                    core_v = v[self.ci].sum()
                    sat_v = v[self.si].sum() if len(self.si) else 0.0
                    if sp.top_mode in ('absband', 'absband_top'):
                        if not pend_top:
                            t = v.copy()
                        else:
                            core_t = sp.core_w*V
                            sat_t = V - core_t
                            t = np.zeros(self.n)
                            if sp.top_mode == 'absband':
                                t[self.ci] = core_t*self.core_tgt
                                if len(self.si):
                                    t[self.si] = sat_t*self.sat_tgt
                            else:
                                t[self.ci] = (core_t*(v[self.ci]/core_v) if core_v > 0
                                              else core_t*self.core_tgt)
                                if len(self.si):
                                    t[self.si] = (sat_t*(v[self.si]/sat_v) if sat_v > 0
                                                  else sat_t*self.sat_tgt)
                    else:
                        if sp.top_mode == 'restore':
                            core_t = sp.core_w*V
                            sat_t = V - core_t
                        elif sp.top_mode == 'cap' and pend_satw > sp.sat_cap:
                            sat_t = sp.sat_cap*V
                            core_t = V - sat_t
                        else:
                            core_t, sat_t = core_v, sat_v
                        t = np.zeros(self.n)
                        if core_v > 0 and not pend_breach:
                            t[self.ci] = core_t*(v[self.ci]/core_v)
                        else:
                            t[self.ci] = core_t*self.core_tgt
                        if len(self.si):
                            t[self.si] = sat_t*self.sat_tgt
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
                    gy += float((s - b[sell]*frac - c*s).sum())
                    b[sell] = b[sell]*(1.0-frac)
                    b[buy] = b[buy] + d[buy]*(1.0+c)
                    v = new
                    trades.append(dict(date=self.cal[i],
                                       n_assets=int(sell.sum()+buy.sum()), realized=gy))
                pend_i = -1

            if sp.rebalance and self.is_sig[i]:
                V = v.sum()
                breach = False
                if len(self.ci) and self.inner_band:
                    cv = v[self.ci].sum()
                    if cv > 0:
                        w = v[self.ci]/cv
                        breach = bool(((w < self.band_lo) | (w > self.band_hi)).any())
                pend_satw = float(v[self.si].sum()/V) if (len(self.si) and V > 0) else 0.0
                pend_top = bool(sp.sat_band is not None
                                and (pend_satw < sp.sat_band[0] or pend_satw > sp.sat_band[1]))
                pend_breach = breach
                pend_i = i+1 if i+1 < T else -1

            if i == T-1 and self.liq:
                s = v.copy()
                gy += float((s - b - c*s).sum())
                b[:] = 0.0
                v = s*(1.0-c)

            if self.tax_on and (self.is_yend[i] or i == T-1):
                ded = self.ded_krw/self.FXv[i]
                tx = self.tax_rate*max(0.0, gy-ded)
                V = v.sum()
                paid = 0.0
                if tx > 0 and V > 0:
                    paid = min(tx, V)
                    f = paid/V
                    v = v*(1.0-f)
                    b = b*(1.0-f)
                tax_rows.append(dict(date=self.cal[i], realized=gy, tax=paid))
                gy = 0.0

            eq[i] = v.sum()
            if keep_holdings:
                H[i] = v

        return dict(equity=pd.Series(eq, index=self.cal, name=sp.name),
                    holdings=(pd.DataFrame(H, index=self.cal, columns=self.A)
                              if keep_holdings else None),
                    trades=pd.DataFrame(trades), taxes=pd.DataFrame(tax_rows))


def run_fast(key, rets, fx, start, end, schd='SCHD_a', tax=False,
             tqqq_zero_date=None, init=10_000.0, cost=0.0010, keep_holdings=False):
    sp = SPECS[key]
    df = rets
    if 'SCHD' in sp.assets and 'SCHD' not in rets.columns:
        df = rets.copy()
        df['SCHD'] = df[schd]
    elif 'SCHD' in sp.assets:
        df = rets.copy()
        df['SCHD'] = df[schd]
    bt = FastBacktest(sp, df, fx, start, end, init=init, cost=cost,
                      tax=tax, tqqq_zero_date=tqqq_zero_date)
    return bt.run(keep_holdings=keep_holdings)
