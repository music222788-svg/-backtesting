"""P3 연중 위성 상한 그리드 실행."""
import sys, pathlib
import numpy as np, pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent
OUT = ROOT/'out'
from s5_cap import CapP3, SAT
import metrics as M

R = pd.read_csv(OUT/'s1_returns.csv', parse_dates=['date']).set_index('date')
R['SCHD'] = R['SCHD_a']
P = pd.read_csv(OUT/'s1_proxy_returns.csv', parse_dates=['date']).set_index('date')
MAC = pd.read_csv(OUT/'s1_macro.csv', parse_dates=['date']).set_index('date')
FX, RF = MAC['krw'], P['SGOV_syn']
PER = {'A': ('2000-01-03', '2019-12-31'), 'B': ('2020-01-02', '2026-09-16'),
       'C': ('2000-01-03', '2026-09-16')}

VARIANTS = [('P3 기본(상한없음)', None, None, None)]
for cap in (0.35, 0.40):
    for freq in ('daily', 'quarter'):
        for act in ('target', 'cap'):
            lbl = f'{int(cap*100)}%/{"상시" if freq=="daily" else "분기"}/{"30%복원" if act=="target" else "상한까지"}'
            VARIANTS.append((lbl, cap, freq, act))


def one(per, cap, freq, act, tqqq_zero=None):
    s, e = PER[per]
    kw = dict(cap=cap, freq=freq or 'daily', action=act or 'target', tqqq_zero_date=tqqq_zero)
    pre = CapP3(R, FX, s, e, **kw).run(keep_holdings=True)
    post = CapP3(R, FX, s, e, tax=True, **kw).run()
    eq, eqp = pre['equity'], post['equity']
    h = pre['holdings']
    w = h[SAT].sum(axis=1)/h.sum(axis=1)
    tq = h['TQQQ']/h.sum(axis=1)
    yrs = (len(eq)-1)/252.0
    lr, _ = M.longest_recovery(eq)
    tr = pre['trades']
    nmid = int((tr['kind'] == 'mid').sum()) if len(tr) else 0
    return dict(
        최종_세전=float(eq.iloc[-1]), 최종_세후=float(eqp.iloc[-1]),
        CAGR=M.cagr(eq)*100, CAGR_세후=M.cagr(eqp)*100, MDD=M.mdd(eq)*100,
        Sharpe=M.sharpe(eq, RF), Sortino=M.sortino(eq, RF), Calmar=M.calmar(eq),
        변동성=M.vol(eq)*100, 최장회복년=lr/252.0,
        최악롤링5년=M.worst_rolling_cagr(eq, 5)*100,
        위성최대=w.max()*100, 위성평균=w.mean()*100, TQQQ최대=tq.max()*100,
        위성35초과일수비율=float((w > 0.35).mean()*100),
        총체결=len(tr), 연중체결=nmid, 연평균체결=len(tr)/yrs,
        총세금=float(post['taxes']['tax'].sum()) if len(post['taxes']) else 0.0)


if __name__ == '__main__':
    pd.set_option('display.width', 260)
    rows = []
    for per in ['A', 'B', 'C']:
        for lbl, cap, freq, act in VARIANTS:
            r = one(per, cap, freq, act)
            r.update(구간=per, 변형=lbl)
            rows.append(r)
    d = pd.DataFrame(rows)
    cols = ['구간', '변형', 'CAGR', 'CAGR_세후', 'MDD', 'Sharpe', 'Calmar', '변동성',
            '최장회복년', '최악롤링5년', '위성최대', '위성평균', 'TQQQ최대',
            '위성35초과일수비율', '연평균체결', '연중체결', '총세금', '최종_세전']
    d[cols].round(3).to_csv(OUT/'s5_cap_grid.csv', index=False, encoding='utf-8-sig')
    for per in ['A', 'B', 'C']:
        x = d[d['구간'] == per].set_index('변형')
        v = x[['CAGR', 'MDD', 'Sharpe', 'Calmar', '최장회복년', '최악롤링5년',
               '위성최대', '위성평균', '연평균체결', '연중체결']].copy()
        for c in ['CAGR', 'MDD', '최악롤링5년', '위성최대', '위성평균']:
            v[c] = v[c].round(2)
        for c in ['Sharpe', 'Calmar', '최장회복년', '연평균체결']:
            v[c] = v[c].round(2)
        print(f'\n===== 구간 {per} =====')
        print(v.to_string())

    # TQQQ=0 시나리오
    z = []
    for per in ['A', 'C']:
        for lbl, cap, freq, act in VARIANTS:
            r = one(per, cap, freq, act, tqqq_zero='2001-08-30')
            r.update(구간=per, 변형=lbl)
            z.append(r)
    zd = pd.DataFrame(z)
    zd[cols].round(3).to_csv(OUT/'s5_cap_grid_tqqq0.csv', index=False, encoding='utf-8-sig')
    print('\n===== TQQQ=0 시나리오 CAGR (%) =====')
    print(zd.pivot(index='변형', columns='구간', values='CAGR').round(2).to_string())
    print('\nsaved -> out/s5_cap_grid.csv, out/s5_cap_grid_tqqq0.csv')
