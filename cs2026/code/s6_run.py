"""연말 리밸런싱 규칙 6변형 비교 + 쌍대 블록 부트스트랩."""
import sys, pathlib, time
import numpy as np, pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent
OUT = ROOT/'out'
from s6_band import BandP3, VARIANTS, ASSETS
import metrics as M

R = pd.read_csv(OUT/'s1_returns.csv', parse_dates=['date']).set_index('date')
R['SCHD'] = R['SCHD_a']
P = pd.read_csv(OUT/'s1_proxy_returns.csv', parse_dates=['date']).set_index('date')
MAC = pd.read_csv(OUT/'s1_macro.csv', parse_dates=['date']).set_index('date')
FX, RF = MAC['krw'], P['SGOV_syn']
PER = {'A': ('2000-01-03', '2019-12-31'), 'B': ('2020-01-02', '2026-09-16'),
       'C': ('2000-01-03', '2026-09-16')}
BLOCK, NBOOT, SEED = 252, 1000, 20260920


def one(per, rule, tz=None):
    s, e = PER[per]
    pre = BandP3(R, FX, s, e, rule, tqqq_zero_date=tz).run(keep_holdings=True)
    post = BandP3(R, FX, s, e, rule, tax=True, tqqq_zero_date=tz).run()
    eq, eqp = pre['equity'], post['equity']
    h = pre['holdings']
    w = h[ASSETS[4:]].sum(axis=1)/h.sum(axis=1)
    yrs = (len(eq)-1)/252.0
    lr, _ = M.longest_recovery(eq)
    tr, sg = pre['trades'], pre['signals']
    return dict(
        CAGR=M.cagr(eq)*100, CAGR_세후=M.cagr(eqp)*100, MDD=M.mdd(eq)*100,
        Sharpe=M.sharpe(eq, RF), Sortino=M.sortino(eq, RF), Calmar=M.calmar(eq),
        변동성=M.vol(eq)*100, 최장회복년=lr/252.0,
        최악롤링5년=M.worst_rolling_cagr(eq, 5)*100,
        위성최대=w.max()*100, 위성평균=w.mean()*100, 위성최소=w.min()*100,
        점검횟수=len(sg), 체결횟수=len(tr),
        평균회전율=float(tr['turnover'].mean()*100) if len(tr) else 0.0,
        연평균체결=len(tr)/yrs, 총세금=float(post['taxes']['tax'].sum()),
        최종_세전=float(eq.iloc[-1]), 최종_세후=float(eqp.iloc[-1]))


def boot(per):
    s, e = PER[per]
    sub = R.loc[s:e, ASSETS]
    cal = sub.index
    T = len(cal)
    arr = np.nan_to_num(sub.to_numpy(dtype=float), nan=0.0)
    nblk = int(np.ceil(T/BLOCK))
    rng = np.random.default_rng(SEED)
    rows = []
    t0 = time.time()
    for it in range(NBOOT):
        st = rng.integers(0, T-BLOCK+1, size=nblk)
        idx = np.concatenate([np.arange(x, x+BLOCK) for x in st])[:T]
        bs = pd.DataFrame(arr[idx], index=cal, columns=ASSETS)
        rec = {'it': it}
        for i, (lbl, rule) in enumerate(VARIANTS.items()):
            eq = BandP3(bs, FX, s, e, rule).run()['equity']
            rec[f'v{i+1}_cagr'] = M.cagr(eq)
            rec[f'v{i+1}_mdd'] = M.mdd(eq)
        rows.append(rec)
        if (it+1) % 250 == 0:
            el = time.time()-t0
            print(f'  [{per}] {it+1}/{NBOOT} {el:.0f}s (eta {el/(it+1)*(NBOOT-it-1):.0f}s)', flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT/f's6_boot_{per}.csv', index=False)
    return d


if __name__ == '__main__':
    pd.set_option('display.width', 250)
    rows = []
    for per in ['A', 'B', 'C']:
        for lbl, rule in VARIANTS.items():
            r = one(per, rule); r.update(구간=per, 변형=lbl); rows.append(r)
    d = pd.DataFrame(rows)
    d.round(3).to_csv(OUT/'s6_grid.csv', index=False, encoding='utf-8-sig')
    for per in ['A', 'B', 'C']:
        x = d[d['구간'] == per].set_index('변형')
        v = x[['CAGR', 'CAGR_세후', 'MDD', 'Sharpe', 'Calmar', '최장회복년', '최악롤링5년',
               '위성최대', '위성평균', '체결횟수', '평균회전율', '총세금']]
        print(f'\n===== 구간 {per} =====')
        print(v.round(2).to_string())
        base = v.loc['V1 현행(최상위·위성 무조건)']
        print('[V1 현행 대비 차이]')
        print((v-base)[['CAGR', 'MDD', 'Sharpe', '최장회복년', '위성최대', '체결횟수', '총세금']]
              .round(2).to_string())

    # 연말 점검 이탈 통계
    sg_all = []
    for per in ['A', 'B', 'C']:
        s, e = PER[per]
        sg = BandP3(R, FX, s, e, VARIANTS['V2 밴드-부분']).run()['signals']
        sg['구간'] = per
        sg_all.append(sg)
    sa = pd.concat(sg_all)
    sa.to_csv(OUT/'s6_signals.csv', index=False, encoding='utf-8-sig')
    print('\n===== 연말 점검 이탈 빈도 =====')
    g = sa.groupby('구간')[['top_breach', 'core_breach', 'sat_breach', 'any_breach']].sum()
    g['점검횟수'] = sa.groupby('구간').size()
    g['무이탈'] = g['점검횟수'] - g['any_breach']
    print(g.to_string())

    print('\n===== 쌍대 블록 부트스트랩 =====', flush=True)
    names = list(VARIANTS)
    for per in ['A', 'C']:
        print(f'--- 구간 {per} ---', flush=True)
        b = boot(per)
        out = []
        for i, lbl in enumerate(names):
            k = f'v{i+1}'
            c, m = b[f'{k}_cagr']*100, b[f'{k}_mdd']*100
            gc = (b[f'{k}_cagr']-b['v1_cagr'])*100
            gm = (b[f'{k}_mdd']-b['v1_mdd'])*100
            out.append(dict(변형=lbl, CAGR중앙=c.median(), CAGR하위10=c.quantile(.10),
                            MDD중앙=m.median(), MDD하위10=m.quantile(.10),
                            CAGR차평균pp=gc.mean(), CAGR개선승률=(gc > 1e-12).mean()*100,
                            MDD차평균pp=gm.mean(), MDD완화승률=(gm > 1e-12).mean()*100))
        t = pd.DataFrame(out).set_index('변형')
        t.round(3).to_csv(OUT/f's6_boot_summary_{per}.csv', encoding='utf-8-sig')
        print(t.round(2).to_string())
    print('\nsaved -> out/s6_grid.csv, s6_signals.csv, s6_boot_*.csv')
