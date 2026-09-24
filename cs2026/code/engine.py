"""백테스트 엔진.

시점 규칙 (2단계):
  - 점검(신호)   : 매년 12월 마지막 거래일 '종가' 기준 비중
  - 체결         : 그 다음 거래일 '종가'
  - 각 구간은 시작일 이전 데이터를 성과 계산에 사용하지 않는다
    (수익률 배열을 [start, end] 로 슬라이스한 뒤에만 엔진에 넘긴다)

비용/세금 (0단계 확정 조건):
  - 편도 0.10% (거래비용 0.05% + 슬리피지 0.05%), 매수·매도 각각. 최초 편입에도 부과
  - 취득원가: 이동평균법 (D8)
  - 매도 수수료는 양도차익에서 공제, 매수 수수료는 취득원가에 산입
  - 연간 통산, 손실 이월 없음 (D5)
  - 기본공제 250만원 / 해당 연말 USD/KRW (D8 조건표 8번)
  - 세율 22%, 실현 연도 마지막 거래일에 포트폴리오에서 차감 (D4)
  - 구간 종료일에 전량 청산 가정 (D1)
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

CORE_ASSETS = ['SPY', 'SCHD', 'IEF', 'GLD']
SAT_ASSETS = ['TQQQ', 'SGOV']
CORE_TGT = {'SPY': .60, 'SCHD': .20, 'IEF': .10, 'GLD': .10}
CORE_BAND = {'SPY': (.55, .65), 'SCHD': (.15, .25), 'IEF': (.075, .125), 'GLD': (.075, .125)}
SAT_TGT = {'TQQQ': .60, 'SGOV': .40}


@dataclass
class Spec:
    name: str
    assets: list
    core: list = field(default_factory=list)
    sat: list = field(default_factory=list)
    core_w: float = None          # 최상위 코어 목표비중 (P1~P4)
    top_mode: str = 'none'        # 'restore' | 'cap' | 'none' | 'single'
    sat_cap: float = None         # top_mode='cap' 일 때 위성 상한
    rebalance: bool = True        # 연말 점검 자체를 수행하는지
    core_tgt: dict = None         # None 이면 모듈 기본값 CORE_TGT
    core_band: dict = None        # None 이면 모듈 기본값 CORE_BAND
    sig_freq: str = 'annual'      # 'annual'(12월 마지막 거래일) | 'daily'(매 거래일)
    sat_tgt: dict = None          # None 이면 모듈 기본값 SAT_TGT
    sat_band: tuple = None        # 위험자산(위성) 비중의 절대 밴드 (lo, hi)


def tgt_of(spec):
    return spec.core_tgt if spec.core_tgt else CORE_TGT


def band_of(spec):
    return spec.core_band if spec.core_band else CORE_BAND


def sat_of(spec):
    return spec.sat_tgt if spec.sat_tgt else SAT_TGT


SPECS = {
    'P1': Spec('P1 코어80:위성20 복원', CORE_ASSETS+SAT_ASSETS, CORE_ASSETS, SAT_ASSETS, .80, 'restore'),
    'P2': Spec('P2 코어80:위성20 무복원(30%상한)', CORE_ASSETS+SAT_ASSETS, CORE_ASSETS, SAT_ASSETS, .80, 'cap', .30),
    'P3': Spec('P3 코어70:위성30 복원', CORE_ASSETS+SAT_ASSETS, CORE_ASSETS, SAT_ASSETS, .70, 'restore'),
    'P4': Spec('P4 코어70:위성30 무복원', CORE_ASSETS+SAT_ASSETS, CORE_ASSETS, SAT_ASSETS, .70, 'none'),
    'C1': Spec('C1 코어100', CORE_ASSETS, CORE_ASSETS, [], 1.0, 'none'),
    'B1': Spec('B1 SPY 100%', ['SPY'], [], [], None, 'single', rebalance=False),
    'B2': Spec('B2 QQQ 100%', ['QQQ'], [], [], None, 'single', rebalance=False),
    # /stock V4 (2026-08-30 규칙): TQQQ:현금 = 40:60, 밴드 ±30%p 절대기준,
    # 이탈 시 목표 40% 완전 복원. 현금은 SGOV.
    'S1': Spec('S1 /stock V4 TQQQ40:현금60 연말점검', ['TQQQ', 'SGOV'], ['TQQQ', 'SGOV'], [],
               1.0, 'none',
               core_tgt={'TQQQ': .40, 'SGOV': .60},
               core_band={'TQQQ': (.10, .70), 'SGOV': (.30, .90)}),
    'S2': Spec('S2 /stock V4 TQQQ40:현금60 상시점검', ['TQQQ', 'SGOV'], ['TQQQ', 'SGOV'], [],
               1.0, 'none',
               core_tgt={'TQQQ': .40, 'SGOV': .60},
               core_band={'TQQQ': (.10, .70), 'SGOV': (.30, .90)},
               sig_freq='daily'),
    # 포트폴리오 A: Gold56 / SCHD26 / TQQQ14 / SPTL4
    #  위험자산(SCHD+TQQQ) 40% , 안전자산(GLD+SPTL) 60%
    #  위험자산 비중이 40%±12%p (28%~52%) 를 벗어나면 목표비중으로 복원
    'A1': Spec('A1 포트A 위험40:안전60 ±12%p 연말점검', ['GLD', 'SPTL', 'SCHD', 'TQQQ'],
               ['GLD', 'SPTL'], ['SCHD', 'TQQQ'], .60, 'absband',
               core_tgt={'GLD': 56/60, 'SPTL': 4/60},
               sat_tgt={'SCHD': 26/40, 'TQQQ': 14/40},
               sat_band=(.28, .52)),
    'A2': Spec('A2 포트A 위험40:안전60 ±12%p 상시점검', ['GLD', 'SPTL', 'SCHD', 'TQQQ'],
               ['GLD', 'SPTL'], ['SCHD', 'TQQQ'], .60, 'absband',
               core_tgt={'GLD': 56/60, 'SPTL': 4/60},
               sat_tgt={'SCHD': 26/40, 'TQQQ': 14/40},
               sat_band=(.28, .52), sig_freq='daily'),
    'A3': Spec('A3 포트A 최상위만 복원(내부 드리프트 유지) 연말점검',
               ['GLD', 'SPTL', 'SCHD', 'TQQQ'],
               ['GLD', 'SPTL'], ['SCHD', 'TQQQ'], .60, 'absband_top',
               core_tgt={'GLD': 56/60, 'SPTL': 4/60},
               sat_tgt={'SCHD': 26/40, 'TQQQ': 14/40},
               sat_band=(.28, .52)),
}


def initial_weights(spec):
    w = {}
    if spec.top_mode == 'single':
        w[spec.assets[0]] = 1.0
        return w
    cw = spec.core_w
    ct = tgt_of(spec)
    for a in spec.core:
        w[a] = cw * ct[a]
    sw = sat_of(spec)
    for a in spec.sat:
        w[a] = (1.0 - cw) * sw[a]
    return w


def dec_last_trading_days(cal):
    """각 연도 12월의 마지막 거래일."""
    s = pd.Series(cal, index=cal)
    dec = s[s.dt.month == 12]
    return list(dec.groupby(dec.dt.year).max())


def year_last_trading_days(cal):
    s = pd.Series(cal, index=cal)
    return list(s.groupby(s.dt.year).max())


class Backtest:
    def __init__(self, spec, rets, fx, start, end, init=10_000.0, cost=0.0010,
                 tax=False, tax_rate=0.22, deduction_krw=2_500_000.0,
                 tqqq_zero_date=None, liquidate_at_end=True):
        self.spec = spec
        self.cost = cost
        self.init = float(init)
        self.tax_on = tax
        self.tax_rate = tax_rate
        self.ded_krw = deduction_krw
        self.tqqq_zero_date = pd.Timestamp(tqqq_zero_date) if tqqq_zero_date else None
        self.liq = liquidate_at_end

        start, end = pd.Timestamp(start), pd.Timestamp(end)
        # ---- 구간 밖 데이터는 애초에 넘기지 않는다 (미래참조/과거참조 차단) ----
        self.R = rets.loc[start:end, spec.assets].copy()
        self.FX = fx.loc[start:end].copy()
        self.cal = self.R.index
        assert self.cal[0] == start, f'start {start} 가 거래일이 아님'
        self.start, self.end = start, self.cal[-1]

        if spec.sig_freq == 'daily':
            self.sig_days = set(d for d in self.cal if d != self.end)
        else:
            self.sig_days = set(d for d in dec_last_trading_days(self.cal) if d != self.end)
        self.yend_days = set(year_last_trading_days(self.cal))
        self.pos = {d: i for i, d in enumerate(self.cal)}

    # ---------- 내부 헬퍼 ----------
    def _drop_tqqq(self, w, group):
        """소멸 시나리오: 목표비중에서 TQQQ 를 빼고 나머지로 재정규화."""
        if self.tqqq_zero_date is None or 'TQQQ' not in group:
            return w
        rest = sum(w[a] for a in group if a != 'TQQQ')
        return {a: (0.0 if a == 'TQQQ' else (w[a]/rest if rest > 0 else 0.0)) for a in group}

    def _targets(self, v, sig):
        """체결일 종가 가치 v 와 신호일에 산출한 sig 로 목표 가치 벡터 생성."""
        sp = self.spec
        V = sum(v.values())
        if sp.top_mode == 'single':
            return dict(v)

        core_v = sum(v[a] for a in sp.core)
        sat_v = sum(v[a] for a in sp.sat)
        ct = self._drop_tqqq(tgt_of(sp), sp.core)
        st = self._drop_tqqq(sat_of(sp), sp.sat) if sp.sat else {}

        # ---- 위험자산 비중 절대밴드 (포트폴리오 A) ----
        if sp.top_mode in ('absband', 'absband_top'):
            if not sig['top_breach']:
                return dict(v)                       # 밴드 안 -> 아무것도 하지 않음
            core_t, sat_t = sp.core_w*V, (1.0-sp.core_w)*V
            t = {}
            if sp.top_mode == 'absband':             # 전체 목표비중으로 완전 복원
                for a in sp.core:
                    t[a] = core_t*ct[a]
                for a in sp.sat:
                    t[a] = sat_t*st[a]
            else:                                    # 최상위 비중만 복원, 내부는 드리프트 유지
                for a in sp.core:
                    t[a] = core_t*(v[a]/core_v) if core_v > 0 else core_t*ct[a]
                for a in sp.sat:
                    t[a] = sat_t*(v[a]/sat_v) if sat_v > 0 else sat_t*st[a]
            return t

        if sp.top_mode == 'restore':
            core_t, sat_t = sp.core_w*V, (1.0-sp.core_w)*V
        elif sp.top_mode == 'cap' and sig['sat_w'] > sp.sat_cap:
            sat_t = sp.sat_cap*V
            core_t = V - sat_t
        else:
            core_t, sat_t = core_v, sat_v

        t = {}
        if core_v > 0 and not sig['core_band_breach']:
            for a in sp.core:
                t[a] = core_t*(v[a]/core_v)
        else:
            for a in sp.core:
                t[a] = core_t*ct[a]
        for a in sp.sat:
            t[a] = sat_t*st[a]
        return t

    def _signal(self, v):
        """신호일 종가 비중으로 점검 결과 산출 (신호일 이후 데이터 일절 사용 안 함)."""
        sp = self.spec
        V = sum(v.values())
        core_v = sum(v[a] for a in sp.core)
        sat_v = sum(v[a] for a in sp.sat)
        cb = band_of(sp)
        breach = False
        # absband 계열은 그룹 내부 밴드 규칙이 없다 (최상위 위험자산 밴드만 존재)
        if core_v > 0 and sp.top_mode not in ('absband', 'absband_top'):
            for a in sp.core:
                w = v[a]/core_v
                lo, hi = cb[a]
                if w < lo or w > hi:
                    breach = True
        sw = sat_v/V if V > 0 else 0.0
        tb = bool(sp.sat_band is not None and (sw < sp.sat_band[0] or sw > sp.sat_band[1]))
        return dict(core_band_breach=breach, sat_w=sw, top_breach=tb,
                    core_w=(core_v/V if V > 0 else 0.0))

    def _trade(self, v, b, t, date):
        """목표 t 로 리밸런싱. v/b(원가) 갱신, 실현손익 반환."""
        V = sum(v.values())
        turn = sum(max(0.0, v[a]-t[a]) for a in t)
        # 임계값은 상대기준. 절대 1e-12 로 두면 자산가치(~1e4)의 부동소수점
        # 잔차(~1e-11)를 매매로 오인한다.
        eps = 1e-10*max(V, 1.0)
        if turn <= eps:
            return 0.0, 0
        fee = 2.0*self.cost*turn
        Vn = V - fee
        scale = Vn/V if V > 0 else 0.0
        gain = 0.0
        n = 0
        for a in t:
            new = t[a]*scale
            d = new - v[a]
            if d < -eps:                                    # 순매도
                s = -d
                frac = s/v[a] if v[a] > 0 else 0.0
                gain += s - b[a]*frac - self.cost*s          # 매도수수료 공제
                b[a] -= b[a]*frac
                n += 1
            elif d > eps:                                   # 순매수
                b[a] += d*(1.0+self.cost)                    # 매수수수료 산입
                n += 1
            v[a] = new
        return gain, n

    def _liquidate(self, v, b):
        gain = 0.0
        for a in list(v):
            s = v[a]
            gain += s - b[a] - self.cost*s
            b[a] = 0.0
            v[a] = s*(1.0-self.cost)
        return gain

    def _pay_tax(self, v, b, gain_year, fx_rate):
        ded = self.ded_krw/fx_rate
        taxable = max(0.0, gain_year - ded)
        t = self.tax_rate*taxable
        V = sum(v.values())
        if t <= 0 or V <= 0:
            return 0.0
        t = min(t, V)
        f = t/V
        for a in v:                       # 추가 실현 없이 안분 차감 (단순화)
            v[a] -= v[a]*f
            b[a] -= b[a]*f
        return t

    # ---------- 실행 ----------
    def run(self):
        sp = self.spec
        w0 = initial_weights(sp)
        v = {a: 0.0 for a in sp.assets}
        b = {a: 0.0 for a in sp.assets}
        for a, w in w0.items():
            # 현금 init 를 전액 투입: 체결금액 = init*w/(1+c), 수수료 = 나머지
            v[a] = self.init*w/(1.0+self.cost)
            b[a] = self.init*w                      # 매수수수료 산입 취득원가

        equity, rows, trades = [], [], []
        pending = None
        gain_year = 0.0
        tax_rows = []
        zeroed = False

        for i, d in enumerate(self.cal):
            if i > 0:                                        # 당일 수익률 반영
                r = self.R.iloc[i]
                for a in sp.assets:
                    v[a] *= (1.0 + (0.0 if np.isnan(r[a]) else r[a]))

            # 합성 3배 소멸 처리 (조건부 병기 시나리오)
            if (self.tqqq_zero_date is not None and not zeroed
                    and 'TQQQ' in v and d >= self.tqqq_zero_date):
                gain_year += -b['TQQQ']                      # 잔여 원가 전액 실현손실
                v['TQQQ'] = 0.0
                b['TQQQ'] = 0.0
                zeroed = True

            if pending is not None and pending[0] == d:      # 체결일
                t = self._targets(v, pending[1])
                g, n = self._trade(v, b, t, d)
                gain_year += g
                if n > 0:                    # 무거래 체결일은 매매 횟수에 넣지 않는다
                    trades.append(dict(date=d, n_assets=n, realized=g))
                pending = None

            if sp.rebalance and d in self.sig_days:          # 신호일
                sig = self._signal(v)
                j = self.pos[d]+1
                if j < len(self.cal):
                    pending = (self.cal[j], sig)             # 반드시 '다음 거래일'

            if d == self.end and self.liq:
                gain_year += self._liquidate(v, b)

            if self.tax_on and (d in self.yend_days or d == self.end):
                paid = self._pay_tax(v, b, gain_year, float(self.FX.loc[d]))
                tax_rows.append(dict(date=d, realized=gain_year, tax=paid))
                gain_year = 0.0

            equity.append(sum(v.values()))
            rows.append({a: v[a] for a in sp.assets})

        eq = pd.Series(equity, index=self.cal, name=sp.name)
        return dict(equity=eq, holdings=pd.DataFrame(rows, index=self.cal),
                    trades=pd.DataFrame(trades), taxes=pd.DataFrame(tax_rows))


def run_one(key, rets, fx, start, end, schd='SCHD_a', tax=False,
            tqqq_zero_date=None, init=10_000.0, cost=0.0010):
    """SCHD 프록시 변형 선택 후 실행."""
    sp = SPECS[key]
    df = rets.copy()
    if 'SCHD' in sp.assets:
        df['SCHD'] = df[schd]
    bt = Backtest(sp, df, fx, start, end, init=init, cost=cost,
                  tax=tax, tqqq_zero_date=tqqq_zero_date)
    return bt.run()
