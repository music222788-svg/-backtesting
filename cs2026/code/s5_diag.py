"""5단계-c: 포트폴리오 A 진단 — 금 56% 의존도, 시작일 의존성, 자산별 기여."""
import sys, pathlib
import numpy as np, pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent
OUT = ROOT/'out'
from engine_fast import run_fast
import metrics as M

R = pd.read_csv(OUT/'s1_returns.csv', parse_dates=['date']).set_index('date')
MAC = pd.read_csv(OUT/'s1_macro.csv', parse_dates=['date']).set_index('date')
FX = MAC['krw']
CAL = R.index
PER = {'A': ('2000-01-03', '2019-12-31'), 'B': ('2020-01-02', '2026-09-16'),
       'C': ('2000-01-03', '2026-09-16')}
KEYS = ['A1', 'A2', 'A3', 'P3', 'C1', 'B1', 'B2']

if __name__ == '__main__':
    pd.set_option('display.width', 250)

    # ---------- 1) 자산별 단순보유 CAGR ----------
    rows = []
    for per, (s, e) in PER.items():
        for a in ['GLD', 'SPTL', 'SCHD_a', 'TQQQ', 'SPY', 'QQQ', 'IEF', 'SGOV']:
            x = R.loc[s:e, a].iloc[1:]
            rows.append(dict(구간=per, 자산=a, CAGR=((1+x).prod()**(252/len(x))-1)*100,
                             MDD=M.mdd((1+R.loc[s:e, a].fillna(0)).cumprod())*100))
    z = pd.DataFrame(rows)
    z.round(2).to_csv(OUT/'s5_asset_cagr.csv', index=False, encoding='utf-8-sig')
    print('===== 자산별 단순보유 CAGR (%) =====')
    print(z.pivot(index='자산', columns='구간', values='CAGR').round(2).to_string())
    print('\n[자산별 MDD %]')
    print(z.pivot(index='자산', columns='구간', values='MDD').round(1).to_string())

    # ---------- 2) 금 수익률 할인 스윕 (손익분기점 탐색) ----------
    cuts = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12]
    rows = []
    for cut in cuts:
        Rg = R.copy(); Rg['GLD'] = Rg['GLD'] - cut/100.0/252.0
        for per in ['A', 'C']:
            s, e = PER[per]
            for k in KEYS:
                rows.append(dict(구간=per, cut=cut, key=k,
                                 CAGR=M.cagr(run_fast(k, Rg, FX, s, e)['equity'])*100))
    g = pd.DataFrame(rows)
    g.round(3).to_csv(OUT/'s5_gold_sweep.csv', index=False, encoding='utf-8-sig')
    print('\n===== 금 수익률 연 -N pp 할인 시 CAGR (%) =====')
    for per in ['A', 'C']:
        x = g[g['구간'] == per].pivot(index='key', columns='cut', values='CAGR').reindex(KEYS)
        print(f'--- 구간 {per} ---'); print(x.round(2).to_string())
        be = {}
        for k in ['A1', 'A2', 'A3']:
            d = x.loc[k] - x.loc['P3']
            lost = [c for c in cuts if d[c] < 0]
            be[k] = (min(lost) if lost else '>12pp')
        print(f'  P3 에 역전당하는 최소 금 할인폭: {be}')

    # ---------- 3) 금을 다른 자산으로 대체 ----------
    rows = []
    for nm, col in [('실제 금', None), ('금->SGOV(현금)', 'SGOV'), ('금->SPY', 'SPY'),
                    ('금->IEF', 'IEF')]:
        Rg = R.copy()
        if col:
            Rg['GLD'] = R[col]
        for per in ['A', 'C']:
            s, e = PER[per]
            for k in ['A1', 'A2', 'A3', 'P3']:
                eq = run_fast(k, Rg, FX, s, e)['equity']
                rows.append(dict(구간=per, 가정=nm, key=k, CAGR=M.cagr(eq)*100, MDD=M.mdd(eq)*100))
    sw = pd.DataFrame(rows)
    sw.round(2).to_csv(OUT/'s5_gold_swap.csv', index=False, encoding='utf-8-sig')
    print('\n===== 금을 다른 자산으로 바꾸면 (CAGR %) =====')
    for per in ['A', 'C']:
        print(f'--- 구간 {per} ---')
        print(sw[sw['구간'] == per].pivot(index='key', columns='가정', values='CAGR')
              .reindex(['A1', 'A2', 'A3', 'P3']).round(2).to_string())

    # ---------- 4) 구간 A 시작일 의존성 ----------
    ss = pd.Series(CAL, index=CAL)
    def month_starts(lo, hi):
        x = ss[(ss >= lo) & (ss <= hi)]
        return list(x.groupby([x.dt.year, x.dt.month]).min())

    rows = []
    for yrs, lo, hi in [(5, '2000-01-01', '2014-12-31'), (10, '2000-01-01', '2009-12-31')]:
        for st in month_starts(lo, hi):
            tgt = st + pd.DateOffset(years=yrs)
            cand = CAL[CAL <= min(tgt, pd.Timestamp('2019-12-31'))]
            en = cand[-1]
            if (CAL.get_loc(en)-CAL.get_loc(st))/252.0 < yrs*0.98:
                continue
            for k in KEYS:
                eq = run_fast(k, R, FX, st, en)['equity']
                rows.append(dict(years=yrs, start=str(st.date()), key=k,
                                 cagr=M.cagr(eq)*100, mdd=M.mdd(eq)*100))
    sd = pd.DataFrame(rows)
    sd.to_csv(OUT/'s5_startdep_A.csv', index=False, encoding='utf-8-sig')
    print('\n===== 구간 A 매월 시작 보유 CAGR 분포 (%) =====')
    for yrs in [5, 10]:
        d = sd[sd['years'] == yrs]
        t = pd.DataFrame({'중앙값': d.groupby('key')['cagr'].median(),
                          '하위10%': d.groupby('key')['cagr'].quantile(.10),
                          '최악': d.groupby('key')['cagr'].min(),
                          '최고': d.groupby('key')['cagr'].max(),
                          '음수비율%': d.assign(n=d.cagr < 0).groupby('key')['n'].mean()*100,
                          'MDD중앙': d.groupby('key')['mdd'].median()}).reindex(KEYS)
        print(f'--- {yrs}년 보유 ({d["start"].nunique()}개 시작월) ---')
        print(t.round(2).to_string())
        wins = {}
        for k in ['A1', 'A2', 'A3']:
            a = d[d.key == k].set_index('start')['cagr']
            p = d[d.key == 'P3'].set_index('start')['cagr']
            wins[k] = round(float((a > p).mean()*100), 1)
        print(f'  P3 대비 승률(시작월 기준) %: {wins}')
    print('\nsaved -> out/s5_asset_cagr.csv, s5_gold_sweep.csv, s5_gold_swap.csv, s5_startdep_A.csv')
