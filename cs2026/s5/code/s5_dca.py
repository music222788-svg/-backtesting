"""S5 검증 1: 적립식 시뮬레이션 (명목 고정 모드 — 한국 CPI 미확보 시).

python s5_dca.py fixed     -> s5_dca_fixed.csv, s5_tax_check.csv, cache/fixed_paths.pkl
python s5_dca.py rolling   -> s5_dca_rolling.csv
python s5_dca.py stress    -> s5_dca_stress.csv
python s5_dca.py boot      -> s5_dca_boot.csv (+ cache/boot_paths_*.csv)
"""
import sys, pathlib, pickle, time
from multiprocessing import Pool
import numpy as np, pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s5_lib as S
from dca_engine import simulate, calendar_struct

OUT = S.S5
FIXED = {'2000-01~2019-12': ('2000-01-03', '2019-12-31'),
         '2000-01~2026-09': ('2000-01-03', '2026-09-16'),
         '2020-01~2026-09': ('2020-01-02', '2026-09-16')}
ORDER = ['P3', 'A80', 'C100', 'B-opt', 'B1', 'SPY', 'QQQ']


def run_window(key, cal, method='M1', tax='krw', R=None, D=None, s1=None, s2=False, record=True):
    """s1: None | 'syn3' | 'qld'  (위성 청산 후 대체 상품)."""
    sp, lev = S.PORTS[key]
    if R is None:
        R, D = S.frames()
    s1_i = -1
    if s1 and lev:
        s1_i = S.syn3_first_below(cal)
        if s1_i >= 0 and s1 == 'qld':
            Rq, Dq = S.frames(lev='QLD')
            R = R.copy(); D = D.copy()
            after = cal[s1_i:]
            R.loc[after, 'TQQQ'] = Rq.loc[after, 'TQQQ']
            D.loc[after, 'TQQQ'] = Dq.loc[after, 'TQQQ']
    cs = calendar_struct(cal)
    Rv, Dv = S.arrays(sp, R, D, cal)
    FXv = S.FX_ALL.loc[cal].to_numpy()
    lvl = None; fr = None
    if lev:
        j = list(sp.assets).index('TQQQ')
        lvl = np.cumprod(1.0+Rv[:, j])
        if s2:
            fr = S.freeze_window(lvl, cal)
    out = simulate(sp, Rv, Dv, FXv, cs, init_krw=S.INIT_KRW, contrib_krw=S.contrib_series(cs),
                   tax=tax, method=method, s1_i=s1_i, freeze=fr, record=True, tq_level=lvl)
    m = S.dca_metrics(out, cs, FXv)
    m['S1_청산일'] = str(cal[s1_i].date()) if s1_i >= 0 else '해당 없음'
    m['S2_동결'] = (f'{cal[fr[0]].date()}~{cal[min(fr[1], len(cal)-1)].date()}' if fr else '해당 없음')
    return m, (out, cs, FXv)


def fixed():
    rows, paths, tax_rows = [], {}, []
    for per, (s, e) in FIXED.items():
        cal = S.window(s, e)
        for k in ORDER:
            for method in ['M1', 'M2']:
                for tax in ['krw', 'usd']:
                    m, (out, cs, FXv) = run_window(k, cal, method, tax)
                    m.update(key=k, 구간=per, 배분=method, 과세기준=('원화' if tax == 'krw' else 'USD'))
                    rows.append(m)
                    if tax == 'krw' and method == 'M1':
                        paths[(k, per)] = dict(val=out['eq']*FXv, paid=out['paid'], cal=cs['cal'])
                    if method == 'M1':
                        y = out['years'].copy()
                        y['key'] = k; y['구간'] = per; y['과세기준'] = '원화' if tax == 'krw' else 'USD'
                        tax_rows.append(y)
    df = pd.DataFrame(rows)
    df.to_csv(OUT/'s5_dca_fixed.csv', index=False, encoding='utf-8-sig')
    pickle.dump(paths, open(OUT/'cache'/'fixed_paths.pkl', 'wb'))
    ty = pd.concat(tax_rows, ignore_index=True)
    k = ty[ty.과세기준 == '원화'].copy()
    u = ty[ty.과세기준 == 'USD'][['key', '구간', 'year', 'gain_usd', 'tax_usd', 'fx']].rename(
        columns={'gain_usd': 'USD기준_양도차익_USD', 'tax_usd': 'USD기준_세액_USD', 'fx': 'fx_usd'})
    t = k.merge(u, on=['key', '구간', 'year'])
    t['USD기준_세액_원'] = t['USD기준_세액_USD']*t['fx']
    t = t.rename(columns={'gain_krw': '원화기준_양도차익_원', 'tax_krw': '원화기준_세액_원',
                          'gain_usd': '참고_같은경로_USD양도차익', 'div_krw': '배당이자_원',
                          'fx': '연말환율'})
    t['종합과세_초과'] = t['배당이자_원'] > 20_000_000
    t['종합과세_초과액_원'] = (t['배당이자_원']-20_000_000).clip(lower=0)
    t['비고'] = np.where(t['종합과세_초과'], '과소 계상 가능(종합과세 추가세액 미계산)', '')
    cols = ['key', '구간', 'year', '연말환율', '원화기준_양도차익_원', '원화기준_세액_원',
            'USD기준_양도차익_USD', 'USD기준_세액_USD', 'USD기준_세액_원', '배당이자_원',
            '종합과세_초과', '종합과세_초과액_원', '비고']
    t[cols].to_csv(OUT/'s5_tax_check.csv', index=False, encoding='utf-8-sig')
    return df


def starts_for(years):
    out = []
    for s, e in S.L.monthly_starts(years=years):
        out.append((s, e))
    return out


def _roll_job(args):
    k, years, method = args
    R, D = S.frames()
    rows = []
    for s, e in starts_for(years):
        cal = S.window(s, e)
        m, _ = run_window(k, cal, method, 'krw', R, D)
        m.update(key=k, 보유년=years, 시작=str(s.date()), 종료=str(e.date()), 배분=method)
        rows.append(m)
    return rows


def rolling():
    jobs = [(k, y, m) for k in ORDER for y in (14, 19) for m in ('M1', 'M2')]
    with Pool(4) as p:
        res = p.map(_roll_job, jobs)
    df = pd.DataFrame([r for rs in res for r in rs])
    df.to_csv(OUT/'s5_dca_rolling.csv', index=False, encoding='utf-8-sig')
    return df


def _stress_job(args):
    k, scen, win = args
    R, D = S.frames()
    rows = []
    wins = [(per, *FIXED[per]) for per in FIXED] if win == 'fixed' else \
        [(f'롤링{win}년 {s.date()}', s, e) for s, e in starts_for(win)]
    for lab, s, e in wins:
        cal = S.window(s, e)
        if scen == '기본':
            m, _ = run_window(k, cal, 'M1', 'krw', R, D)
        elif scen == 'S1-i 합성3배 재시작':
            m, _ = run_window(k, cal, 'M1', 'krw', R, D, s1='syn3')
        elif scen == 'S1-ii QLD 대체':
            m, _ = run_window(k, cal, 'M1', 'krw', R, D, s1='qld')
        else:
            m, _ = run_window(k, cal, 'M1', 'krw', R, D, s2=True)
        m.update(key=k, 시나리오=scen, 구간=lab, 창=str(win))
        rows.append(m)
    return rows


def stress():
    scen = ['기본', 'S1-i 합성3배 재시작', 'S1-ii QLD 대체', 'S2 규칙이탈(36개월 매수중단)']
    jobs = [(k, sc, w) for k in ['P3', 'A80'] for sc in scen for w in ('fixed', 14, 19)]
    jobs += [('C100', '기본', w) for w in ('fixed', 14, 19)]
    with Pool(4) as p:
        res = p.map(_stress_job, jobs)
    df = pd.DataFrame([r for rs in res for r in rs])
    df.to_csv(OUT/'s5_dca_stress.csv', index=False, encoding='utf-8-sig')
    return df


# ---------------- 부트스트랩 ----------------
BLOCK, NBOOT, SEED = 252, 1000, 20260917


def boot_paths(years):
    pool = S.window('2000-01-03', '2026-09-16')
    Tp = len(pool)
    T = years*252
    nblk = int(np.ceil(T/BLOCK))
    rng = np.random.default_rng(SEED)
    return [np.concatenate([np.arange(x, x+BLOCK) for x in rng.integers(0, Tp-BLOCK+1, size=nblk)])[:T]
            for _ in range(NBOOT)]


def _boot_job(args):
    k, years, gold_drift, tag = args
    sp, lev = S.PORTS[k]
    R, D = S.frames(gold_drift=gold_drift)
    pool = S.window('2000-01-03', '2026-09-16')
    Rp, Dp = S.arrays(sp, R, D, pool)
    fxr = S.FX_ALL.loc[pool].pct_change().fillna(0.0).to_numpy()
    syn = S.L.syn_series(3, False).reindex(pool).fillna(0.0).to_numpy()
    fx0 = float(S.FX_ALL.loc[S.TEND])
    T = years*252
    cal = S.CAL_ALL[S.CAL_ALL >= S.T0][:T]            # 구조용 캘린더(월초·연말 위치)
    cs = calendar_struct(cal)
    ck = S.contrib_series(cs)
    rows = []
    for it, idx in enumerate(boot_paths(years)):
        fx = fx0*np.cumprod(np.r_[1.0, 1.0+fxr[idx[1:]]])
        out = simulate(sp, Rp[idx], Dp[idx], fx, cs, init_krw=S.INIT_KRW, contrib_krw=ck,
                       tax='krw', method='M1')
        w3 = np.cumprod(1.0+syn[idx])
        rows.append(dict(it=it, key=k, 보유년=years, G=tag, 최종자산=out['eq'][-1]*fx[-1],
                         붕괴3배=bool((w3 < 0.01).any()), 누적납입=out['paid_in']))
    return rows


def boot_summary(df):
    rows = []
    for (k, y, g), x in df.groupby(['key', '보유년', 'G'], sort=False):
        f = x['최종자산']
        cl = x['붕괴3배']
        rows.append(dict(key=k, 보유년=y, G=g, 중앙값=f.median(), 하위10=f.quantile(.10),
                         하위25=f.quantile(.25), 상위10=f.quantile(.90),
                         P_25억이상=(f >= S.GOAL).mean()*100, P_15억이상=(f >= S.GOAL2).mean()*100,
                         합성3배_붕괴경로비율=cl.mean()*100,
                         붕괴경로_중앙값=f[cl].median() if cl.any() else np.nan,
                         비붕괴경로_중앙값=f[~cl].median() if (~cl).any() else np.nan,
                         누적납입=x['누적납입'].iloc[0]))
    return pd.DataFrame(rows)


def boot():
    jobs = [(k, y, 0.0, 'G0') for k in ORDER for y in (14, 19)]
    with Pool(4) as p:
        res = p.map(_boot_job, jobs)
    df = pd.DataFrame([r for rs in res for r in rs])
    df.to_csv(OUT/'cache'/'boot_paths_G0.csv', index=False)
    s = boot_summary(df)
    s.to_csv(OUT/'s5_dca_boot.csv', index=False, encoding='utf-8-sig')
    return s


if __name__ == '__main__':
    t0 = time.time()
    what = sys.argv[1]
    r = {'fixed': fixed, 'rolling': rolling, 'stress': stress, 'boot': boot}[what]()
    print(what, 'done', round(time.time()-t0), 's', flush=True)
