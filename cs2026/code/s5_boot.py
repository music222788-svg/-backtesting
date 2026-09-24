"""상한 변형의 개선이 단일 경로 우연인지 블록 부트스트랩으로 검증.

동일한 리샘플 경로에 기본 P3 와 상한 변형을 함께 돌려 '쌍대 비교'한다.
"""
import sys, pathlib, time
import numpy as np, pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent
OUT = ROOT/'out'
from s5_cap import CapP3, ASSETS
import metrics as M

R = pd.read_csv(OUT/'s1_returns.csv', parse_dates=['date']).set_index('date')
R['SCHD'] = R['SCHD_a']
MAC = pd.read_csv(OUT/'s1_macro.csv', parse_dates=['date']).set_index('date')
FX = MAC['krw']
PER = {'A': ('2000-01-03', '2019-12-31'), 'C': ('2000-01-03', '2026-09-16')}
BLOCK, NBOOT, SEED = 252, 1000, 20260918

VAR = {
    'base':       dict(cap=None),
    'c35_d_tgt':  dict(cap=0.35, freq='daily',   action='target'),
    'c35_d_cap':  dict(cap=0.35, freq='daily',   action='cap'),
    'c35_q_tgt':  dict(cap=0.35, freq='quarter', action='target'),
    'c40_d_tgt':  dict(cap=0.40, freq='daily',   action='target'),
}


def run(per):
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
        for k, kw in VAR.items():
            eq = CapP3(bs, FX, s, e, **kw).run()['equity']
            rec[f'{k}_cagr'] = M.cagr(eq)
            rec[f'{k}_mdd'] = M.mdd(eq)
        rows.append(rec)
        if (it+1) % 200 == 0:
            el = time.time()-t0
            print(f'  [{per}] {it+1}/{NBOOT} {el:.0f}s (eta {el/(it+1)*(NBOOT-it-1):.0f}s)', flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT/f's5_boot_{per}.csv', index=False)
    return d


if __name__ == '__main__':
    pd.set_option('display.width', 240)
    for per in ['A', 'C']:
        print(f'=== 구간 {per} 부트스트랩 (블록 {BLOCK}일, {NBOOT}회) ===', flush=True)
        d = run(per)
        out = []
        for k in VAR:
            c = d[f'{k}_cagr']*100
            m = d[f'{k}_mdd']*100
            gc = (d[f'{k}_cagr']-d['base_cagr'])*100
            gm = (d[f'{k}_mdd']-d['base_mdd'])*100          # 양수면 낙폭 완화
            out.append(dict(변형=k, CAGR중앙=c.median(), CAGR하위10=c.quantile(.10),
                            MDD중앙=m.median(), MDD하위10=m.quantile(.10),
                            CAGR개선평균pp=gc.mean(), CAGR개선승률=(gc > 0).mean()*100,
                            MDD완화평균pp=gm.mean(), MDD완화승률=(gm > 1e-9).mean()*100))
        t = pd.DataFrame(out).set_index('변형')
        t.round(3).to_csv(OUT/f's5_boot_summary_{per}.csv', encoding='utf-8-sig')
        print(f'\n[구간 {per}] 기본 대비 쌍대 비교')
        print(t.round(2).to_string())
        print()
