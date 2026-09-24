"""밴드 폭에 따른 성과 차이가 실재하는지 쌍대 블록 부트스트랩으로 검증."""
import sys, pathlib, time
import numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE.parent/'out'
from s6_band import BandP3, VARIANTS, ASSETS
import metrics as M

R = pd.read_csv(OUT/'s1_returns.csv', parse_dates=['date']).set_index('date')
R['SCHD'] = R['SCHD_a']
FX = pd.read_csv(OUT/'s1_macro.csv', parse_dates=['date']).set_index('date')['krw']
PER = {'A': ('2000-01-03', '2019-12-31'), 'C': ('2000-01-03', '2026-09-16')}
V1, V2, V6 = (VARIANTS['V1 현행(최상위·위성 무조건)'], VARIANTS['V2 밴드-부분'],
              VARIANTS['V6 무리밸런싱'])
CASES = [('base', V1, 1.0), ('pm5', V2, 1.0), ('pm10', V2, 2.0),
         ('pm15', V2, 3.0), ('pm20', V2, 4.0), ('none', V6, 1.0)]
BLOCK, NBOOT, SEED = 252, 1000, 20260921

if __name__ == '__main__':
    pd.set_option('display.width', 220)
    for per in ['A', 'C']:
        s, e = PER[per]
        sub = R.loc[s:e, ASSETS]
        cal = sub.index; T = len(cal)
        arr = np.nan_to_num(sub.to_numpy(dtype=float), nan=0.0)
        nblk = int(np.ceil(T/BLOCK))
        rng = np.random.default_rng(SEED)
        rows = []; t0 = time.time()
        for it in range(NBOOT):
            st = rng.integers(0, T-BLOCK+1, size=nblk)
            idx = np.concatenate([np.arange(x, x+BLOCK) for x in st])[:T]
            bs = pd.DataFrame(arr[idx], index=cal, columns=ASSETS)
            rec = {'it': it}
            for nm, rule, k in CASES:
                r = BandP3(bs, FX, s, e, rule, band_scale=k).run(keep_holdings=True)
                eq = r['equity']; h = r['holdings']
                rec[f'{nm}_cagr'] = M.cagr(eq); rec[f'{nm}_mdd'] = M.mdd(eq)
                rec[f'{nm}_satavg'] = float((h[ASSETS[4:]].sum(axis=1)/h.sum(axis=1)).mean())
            rows.append(rec)
            if (it+1) % 250 == 0:
                el = time.time()-t0
                print(f'  [{per}] {it+1}/{NBOOT} {el:.0f}s (eta {el/(it+1)*(NBOOT-it-1):.0f}s)', flush=True)
        d = pd.DataFrame(rows); d.to_csv(OUT/f's6_bw_boot_{per}.csv', index=False)
        out = []
        for nm, _, _ in CASES:
            c, m = d[f'{nm}_cagr']*100, d[f'{nm}_mdd']*100
            gc = (d[f'{nm}_cagr']-d['base_cagr'])*100
            gm = (d[f'{nm}_mdd']-d['base_mdd'])*100
            out.append(dict(case=nm, CAGR_med=c.median(), CAGR_p10=c.quantile(.10),
                            MDD_med=m.median(), MDD_p10=m.quantile(.10),
                            satavg=d[f'{nm}_satavg'].mean()*100,
                            dCAGR=gc.mean(), winCAGR=(gc > 1e-12).mean()*100,
                            dMDD=gm.mean(), winMDD=(gm > 1e-12).mean()*100))
        t = pd.DataFrame(out).set_index('case')
        t.round(3).to_csv(OUT/f's6_bw_boot_summary_{per}.csv', encoding='utf-8-sig')
        print(f'\n== period {per} : paired bootstrap vs base ==')
        print(t.round(2).to_string(), flush=True)
