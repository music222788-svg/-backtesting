"""3단계-c: 블록 부트스트랩 (블록 = 252 거래일 = 1년), 1,000회. 구간 A, C."""
import sys, pathlib, time
import numpy as np, pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent
OUT = ROOT/'out'
from engine_fast import run_fast
import metrics as M

R = pd.read_csv(OUT/'s1_returns.csv', parse_dates=['date']).set_index('date')
P = pd.read_csv(OUT/'s1_proxy_returns.csv', parse_dates=['date']).set_index('date')
MAC = pd.read_csv(OUT/'s1_macro.csv', parse_dates=['date']).set_index('date')
FX = MAC['krw']

PER = {'A': ('2000-01-03', '2019-12-31'), 'C': ('2000-01-03', '2026-09-16')}
KEYS = ['P1', 'P2', 'P3', 'P4', 'B1', 'B2', 'C1']
COLS = ['SPY', 'QQQ', 'SCHD_a', 'IEF', 'GLD', 'TQQQ', 'SGOV']
BLOCK = 252
NBOOT = 1000
SEED = 20260917


def bootstrap(period):
    s, e = PER[period]
    sub = R.loc[s:e, COLS].copy()
    sub['SCHD'] = sub['SCHD_a']
    tq = P.loc[s:e, 'TQQQ_syn']
    cal = sub.index
    T = len(cal)
    arr = sub.to_numpy(dtype=float)
    tqa = np.nan_to_num(tq.to_numpy(dtype=float), nan=0.0)
    arr = np.nan_to_num(arr, nan=0.0)
    nblk = int(np.ceil(T/BLOCK))
    maxstart = T - BLOCK
    rng = np.random.default_rng(SEED)
    cols = list(sub.columns)

    rows = []
    t0 = time.time()
    for it in range(NBOOT):
        st = rng.integers(0, maxstart+1, size=nblk)
        idx = np.concatenate([np.arange(x, x+BLOCK) for x in st])[:T]
        bs = pd.DataFrame(arr[idx], index=cal, columns=cols)
        # 합성 3배 지수가 원금 1% 미만으로 떨어지는지 (이 경로에서의 소멸 위험)
        w = np.cumprod(1.0+tqa[idx])
        wipe = bool((w < 0.01).any())
        rec = dict(it=it, tqqq_wipeout=wipe, tqqq_min=float(w.min()))
        for k in KEYS:
            eq = run_fast(k, bs, FX, s, e, keep_holdings=False)['equity']
            rec[f'{k}_cagr'] = M.cagr(eq)
            rec[f'{k}_mdd'] = M.mdd(eq)
        rows.append(rec)
        if (it+1) % 100 == 0:
            el = time.time()-t0
            print(f'  [{period}] {it+1}/{NBOOT}  {el:.0f}s  (eta {el/(it+1)*(NBOOT-it-1):.0f}s)',
                  flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT/f's3_bootstrap_{period}.csv', index=False)
    return df


def summarize(df, period):
    rows = []
    for k in KEYS:
        c = df[f'{k}_cagr']*100
        m = df[f'{k}_mdd']*100
        rows.append(dict(구간=period, key=k,
                         CAGR_중앙=c.median(), CAGR_평균=c.mean(),
                         CAGR_하위10=c.quantile(.10), CAGR_하위5=c.quantile(.05),
                         CAGR_최악=c.min(), CAGR_상위10=c.quantile(.90),
                         CAGR_음수비율=float((c < 0).mean()*100),
                         MDD_중앙=m.median(), MDD_하위10=m.quantile(.10),
                         MDD_최악=m.min()))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    outs = []
    for per in ['A', 'C']:
        print(f'=== 블록 부트스트랩 구간 {per} (블록 {BLOCK}일, {NBOOT}회) ===', flush=True)
        df = bootstrap(per)
        print(f'  [{per}] 합성 3배 소멸(원금 1% 미만) 경로 비율: '
              f'{df["tqqq_wipeout"].mean()*100:.1f}%', flush=True)
        outs.append(summarize(df, per))
    res = pd.concat(outs, ignore_index=True)
    res.round(2).to_csv(OUT/'s3_bootstrap_summary.csv', index=False, encoding='utf-8-sig')
    pd.set_option('display.width', 250)
    print(res.round(2).to_string(index=False))
