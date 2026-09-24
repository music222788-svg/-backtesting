"""4단계: P3 vs /stock V4 비교 (구간 A·B·C).

/stock V4 규칙 (redev 프로젝트 메모리 2026-08-30 기준):
  - TQQQ : 현금 = 40 : 60
  - 밴드 ±30%p 절대기준 -> TQQQ 10%~70%
  - 이탈 시 밴드 경계가 아니라 목표 40%까지 완전 복원 (부분 복원 금지)
  - Tier / 4단계 분할매수 / Free Ride 는 폐지

점검 주기가 규칙에 명시되어 있지 않아 두 가지를 병기한다.
  S1 = 연말(12월 마지막 거래일) 점검   <- 본 연구의 다른 전략과 동일한 관행
  S2 = 매 거래일 점검                  <- 밴드 규칙의 자연스러운 해석
체결은 두 경우 모두 '신호일 다음 거래일 종가'.
현금은 SGOV(=3개월 국채). 무이자 현금 변형도 함께 산출한다.
"""
import sys, pathlib
import numpy as np, pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent
OUT = ROOT/'out'
import engine as E
from engine_fast import run_fast
import metrics as M

R = pd.read_csv(OUT/'s1_returns.csv', parse_dates=['date']).set_index('date')
P = pd.read_csv(OUT/'s1_proxy_returns.csv', parse_dates=['date']).set_index('date')
MAC = pd.read_csv(OUT/'s1_macro.csv', parse_dates=['date']).set_index('date')
FX = MAC['krw']
RF = P['SGOV_syn']

PER = {'A': ('2000-01-03', '2019-12-31'),
       'B': ('2020-01-02', '2026-09-16'),
       'C': ('2000-01-03', '2026-09-16')}
KEYS = ['P3', 'S1', 'S2', 'C1', 'B1', 'B2']
TQQQ_ZERO = '2001-08-30'

SUB_A = [('2000-2002 닷컴붕괴', '2000-01-03', '2002-12-31'),
         ('2003-2007 회복장', '2003-01-02', '2007-12-31'),
         ('2008-2009 금융위기', '2008-01-02', '2009-12-31'),
         ('2010-2019 강세장', '2010-01-04', '2019-12-31')]
SUB_B = [('2020 코로나폭락', '2020-02-19', '2020-03-23'),
         ('2020-04~2021 회복장', '2020-04-01', '2021-12-31'),
         ('2022 약세장', '2022-01-03', '2022-12-30'),
         ('2023~최근', '2023-01-03', '2026-09-16')]


def run_set(per, rets=R, tqqq_zero=False):
    s, e = PER[per]
    z = TQQQ_ZERO if tqqq_zero else None
    out = {}
    for k in KEYS:
        out[k] = dict(
            pre=E.run_one(k, rets, FX, s, e, tax=False, tqqq_zero_date=z),
            post=E.run_one(k, rets, FX, s, e, tax=True, tqqq_zero_date=z))
    return out


def table(res, per):
    rows = []
    for k in KEYS:
        r = res[k]
        d = M.summarize(k, r['pre']['equity'], r['post']['equity'], FX, RF,
                        r['pre']['trades'], r['post']['taxes'])
        d['구간'] = per; d['key'] = k
        rows.append(d)
    return pd.DataFrame(rows)


def show(t, title):
    v = t[['key', '최종자산_세전', '최종자산_세후', 'CAGR_세전', 'CAGR_세후', 'CAGR_원화', 'MDD',
           '최장회복_거래일', '연변동성', 'Sharpe', 'Sortino', 'Calmar', '최악의연도', '최악연도',
           '최악롤링3년', '최악롤링5년', '연평균매매', '연평균세금']].copy()
    for c in ['CAGR_세전', 'CAGR_세후', 'CAGR_원화', 'MDD', '연변동성', '최악의연도',
              '최악롤링3년', '최악롤링5년']:
        v[c] = (v[c]*100).round(2)
    for c in ['최종자산_세전', '최종자산_세후', '연평균세금']:
        v[c] = v[c].round(0)
    for c in ['Sharpe', 'Sortino', 'Calmar', '연평균매매']:
        v[c] = v[c].round(2)
    print(f'\n===== {title} =====')
    print(v.to_string(index=False))


if __name__ == '__main__':
    pd.set_option('display.width', 260)
    tabs, eqs = [], {}
    for per in ['A', 'B', 'C']:
        res = run_set(per)
        t = table(res, per); tabs.append(t)
        eq = pd.DataFrame({k: res[k]['pre']['equity'] for k in KEYS})
        eq.to_csv(OUT/f's4_equity_{per}.csv')
        eqs[per] = eq
        show(t, f'구간 {per} ({PER[per][0]} ~ {PER[per][1]})  기본 시나리오')
    TAB = pd.concat(tabs, ignore_index=True)
    TAB.to_csv(OUT/'s4_summary.csv', index=False, encoding='utf-8-sig')

    # ---------- 연간 수익률 ----------
    ann = []
    for per in ['A', 'B']:
        a = pd.DataFrame({k: M.annual_returns(eqs[per][k]) for k in KEYS})*100
        a.round(1).to_csv(OUT/f's4_annual_{per}.csv', encoding='utf-8-sig')
        print(f'\n===== 구간 {per} 연간 수익률 (%) ====='); print(a.round(1).to_string())

    # ---------- 세부 구간 ----------
    sub = []
    for per, subs in [('A', SUB_A), ('B', SUB_B)]:
        rows = []
        for lab, s, e in subs:
            for k in KEYS:
                x = eqs[per][k].loc[s:e]
                rows.append(dict(구간=per, 세부=lab, key=k,
                                 누적수익=float(x.iloc[-1]/x.iloc[0]-1.0),
                                 구간내MDD=M.mdd(x)))
        d = pd.DataFrame(rows); sub.append(d)
        order = [x[0] for x in subs]
        print(f'\n===== 구간 {per} 세부 누적수익 (%) =====')
        print((d.pivot(index='key', columns='세부', values='누적수익')[order]*100)
              .reindex(KEYS).round(1).to_string())
        print(f'[구간 {per} 세부 MDD %]')
        print((d.pivot(index='key', columns='세부', values='구간내MDD')[order]*100)
              .reindex(KEYS).round(1).to_string())
    pd.concat(sub).to_csv(OUT/'s4_subperiods.csv', index=False, encoding='utf-8-sig')

    # ---------- 코로나 ----------
    pk = pd.Timestamp('2020-02-19')
    rows = []
    for k in KEYS:
        x = eqs['B'][k]
        seg = x.loc[pk:'2020-03-23']
        rd, days = M.recovery_days(x, pk)
        rows.append(dict(key=k, 낙폭=float(seg.min()/x.loc[pk]-1.0),
                         저점일=str(seg.idxmin().date()),
                         회복일=str(rd.date()) if rd is not None else '미회복',
                         회복_거래일=days))
    cv = pd.DataFrame(rows); cv.to_csv(OUT/'s4_covid.csv', index=False, encoding='utf-8-sig')
    cv['낙폭'] = (cv['낙폭']*100).round(2)
    print('\n===== 코로나 폭락 (2020-02-19 ~ 03-23) ====='); print(cv.to_string(index=False))

    # ---------- TQQQ=0 시나리오 ----------
    zt = []
    for per in ['A', 'C']:
        res = run_set(per, tqqq_zero=True)
        t = table(res, per); zt.append(t)
        pd.DataFrame({k: res[k]['pre']['equity'] for k in KEYS}).to_csv(OUT/f's4_equity_{per}_tqqq0.csv')
    Z = pd.concat(zt, ignore_index=True)
    Z.to_csv(OUT/'s4_summary_tqqq0.csv', index=False, encoding='utf-8-sig')
    print('\n===== TQQQ=0 (2001-08-30 이후) 시나리오 =====')
    for per in ['A', 'C']:
        b = TAB[TAB['구간'] == per].set_index('key').reindex(KEYS)
        z = Z[Z['구간'] == per].set_index('key').reindex(KEYS)
        d = pd.DataFrame({'CAGR기본': (b['CAGR_세전']*100).round(2),
                          'CAGR_TQQQ0': (z['CAGR_세전']*100).round(2),
                          'MDD기본': (b['MDD']*100).round(2),
                          'MDD_TQQQ0': (z['MDD']*100).round(2),
                          '최종기본': b['최종자산_세전'].round(0),
                          '최종_TQQQ0': z['최종자산_세전'].round(0)})
        print(f'--- 구간 {per} ---'); print(d.to_string())

    # ---------- 무이자 현금 변형 ----------
    R0 = R.copy(); R0['SGOV'] = 0.0
    rows = []
    for per in ['A', 'B', 'C']:
        s, e = PER[per]
        for k in ['S1', 'S2']:
            a = run_fast(k, R, FX, s, e)['equity']
            b = run_fast(k, R0, FX, s, e)['equity']
            rows.append(dict(구간=per, key=k, CAGR_SGOV현금=M.cagr(a)*100,
                             CAGR_무이자현금=M.cagr(b)*100,
                             차이pp=(M.cagr(b)-M.cagr(a))*100))
    z = pd.DataFrame(rows); z.round(3).to_csv(OUT/'s4_cash0.csv', index=False, encoding='utf-8-sig')
    print('\n===== 현금 이자 가정 민감도 ====='); print(z.round(2).to_string(index=False))

    # ---------- 구간 B 시작월 의존성 ----------
    cal = R.index
    ss = pd.Series(cal, index=cal)
    ss = ss[(ss >= '2020-01-01') & (ss <= '2020-12-31')]
    starts = list(ss.groupby([ss.dt.year, ss.dt.month]).min())
    rows = []
    for st in starts:
        for k in KEYS:
            eq = run_fast(k, R, FX, st, '2026-09-16')['equity']
            rows.append(dict(start=str(st.date()), key=k, cagr=M.cagr(eq)*100, mdd=M.mdd(eq)*100))
    sd = pd.DataFrame(rows); sd.to_csv(OUT/'s4_startdep_B.csv', index=False, encoding='utf-8-sig')
    piv = sd.pivot(index='start', columns='key', values='cagr')[KEYS]
    print('\n===== 구간 B 시작월별 CAGR (%) ====='); print(piv.round(2).to_string())
    print('\n[순위]'); print(piv.rank(axis=1, ascending=False).astype(int).to_string())

    print('\nsaved -> out/s4_*.csv')
