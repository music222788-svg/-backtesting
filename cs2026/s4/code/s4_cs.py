"""S4 작업 2: 코어·위성 변형 비교.

필요: s4/cache/bopt.json  ({"SCHD":..,"SPY":..,"QQQ":..}) — 작업 1 에서 확정한 B-opt
산출: s4/s4_summary.csv, s4/s4_stress.csv, s4/s4_boot_cs.csv(+요약), s4/cache/cs_equity.pkl
"""
import sys, pathlib, json, pickle, time
from multiprocessing import Pool
import numpy as np, pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s4_lib as L
import metrics as M

S4 = L.S4
BOPT = json.load(open(S4/'cache'/'bopt.json'))
CORE_P3 = {'SPY': .60, 'SCHD': .20, 'IEF': .10, 'GLD': .10}
SAT_TQ = {'TQQQ': .60, 'SGOV': .40}

# key -> (표시명, spec, 레버리지 종류 None|'TQQQ'|'QLD')
PORTS = {
    'P3':    ('P3 코어70(SPY60/SCHD20/IEF10/GLD10):위성30(TQQQ60/SGOV40)',
              L.spec_cs('P3', CORE_P3, .70, SAT_TQ), 'TQQQ'),
    'A80':   ('A80 P3 구성, 코어80:위성20', L.spec_cs('A80', CORE_P3, .80, SAT_TQ), 'TQQQ'),
    'A-Q1':  ('A-Q1 위성 QLD60/SGOV40', L.spec_cs('A-Q1', CORE_P3, .70, SAT_TQ), 'QLD'),
    'A-Q2':  ('A-Q2 위성 QLD90/SGOV10', L.spec_cs('A-Q2', CORE_P3, .70, {'TQQQ': .90, 'SGOV': .10}), 'QLD'),
    'C':     ('C 코어70=B-opt:위성30(TQQQ60/SGOV40)',
              L.spec_cs('C', {a: BOPT[a] for a in ['SCHD', 'SPY', 'QQQ'] if BOPT[a] > 0}, .70, SAT_TQ), 'TQQQ'),
    'B-opt': ('B-opt SCHD:SPY:QQQ={SCHD:.0%}:{SPY:.0%}:{QQQ:.0%}'.format(**BOPT), L.spec_b(BOPT), None),
    'B1':    ('B1 SCHD:SPY:QQQ=2:3:5', L.spec_b({'SCHD': .2, 'SPY': .3, 'QQQ': .5}), None),
    'B2':    ('B2 SCHD:SPY:QQQ=1:3:6', L.spec_b({'SCHD': .1, 'SPY': .3, 'QQQ': .6}), None),
    'B3':    ('B3 SCHD:SPY:QQQ=3:3:4', L.spec_b({'SCHD': .3, 'SPY': .3, 'QQQ': .4}), None),
    'SPY':   ('SPY 100%', L.spec_single('SPY'), None),
    'QQQ':   ('QQQ 100%', L.spec_single('QQQ'), None),
    'C100':  ('코어100 (SPY60/SCHD20/IEF10/GLD10, 5/25 밴드)', L.spec_c1(), None),
}
PERIODS = ['A', 'B', 'C', 'S1', 'S2', 'S3']
VARIANTS = [(wh, cal) for wh in (False, True) for cal in (False, True)]
ZERO = {'TQQQ': '2001-08-30', 'QLD': None}      # QLD: 합성 QLD 가 원금 1% 미만으로 떨어진 날 없음


def rets_for(lev, wh, cal):
    return L.returns(wh=wh, cal=cal, lev=(lev or 'TQQQ'))


def summary():
    rows, eqs = [], {}
    for wh, cal in VARIANTS:
        for k, (nm, sp, lev) in PORTS.items():
            R = rets_for(lev, wh, cal)
            for p in PERIODS:
                d, pre, post = L.evaluate(sp, R, p)
                d.update(key=k, 포트폴리오=nm, 구간=p, 구간명=L.PER_LABEL[p],
                         원천징수=('적용' if wh else '미적용'), 합성=('보정' if cal else '기본'))
                rows.append(d)
                if p in ('A', 'B', 'C'):
                    eqs[(wh, cal, k, p)] = (pre['equity'], post['equity'])
    df = pd.DataFrame(rows)
    return df, eqs


def stress():
    rows = []
    for wh, cal in [(True, False), (True, True), (False, False)]:
        for k, (nm, sp, lev) in PORTS.items():
            R = rets_for(lev, wh, cal)
            for p in ['A', 'C']:
                base, _, _ = L.evaluate(sp, R, p)
                z = ZERO.get(lev) if lev else None
                rec = dict(key=k, 포트폴리오=nm, 구간=p, 원천징수=('적용' if wh else '미적용'),
                           합성=('보정' if cal else '기본'),
                           붕괴일=(z if z else ('해당 없음(합성 QLD 최저 원금의 1.90%, 2009-03-09)'
                                          if lev == 'QLD' else '비레버리지')),
                           기본_CAGR_세전=base['CAGR_세전'], 기본_CAGR_세후=base['CAGR_세후'],
                           기본_최종_세전=base['최종자산_세전'], 기본_MDD=base['MDD'])
                if z:
                    d, _, _ = L.evaluate(sp, R, p, zero=z)
                else:
                    d = base
                rec.update(붕괴_CAGR_세전=d['CAGR_세전'], 붕괴_CAGR_세후=d['CAGR_세후'],
                           붕괴_최종_세전=d['최종자산_세전'], 붕괴_MDD=d['MDD'],
                           붕괴_최장회복=d['최장회복_거래일'])
                rows.append(rec)
    df = pd.DataFrame(rows)
    for (wh, cal, p), g in df.groupby(['원천징수', '합성', '구간']):
        df.loc[g.index, '기본_순위'] = g['기본_CAGR_세전'].rank(ascending=False).astype(int)
        df.loc[g.index, '붕괴_순위'] = g['붕괴_CAGR_세전'].rank(ascending=False).astype(int)
    return df


# ---------------- 부트스트랩 ----------------
def _boot_job(args):
    per, cal_flag, wh, keys = args
    s, e = L.PER[per]
    cal, idxs = L.boot_indices(per)
    syn3 = np.nan_to_num(L.syn_series(3, cal_flag).loc[s:e].to_numpy(dtype=float), nan=0.0)
    syn2 = np.nan_to_num(L.syn_series(2, cal_flag).loc[s:e].to_numpy(dtype=float), nan=0.0)
    arrs = {}
    for k in keys:
        nm, sp, lev = PORTS[k]
        R = rets_for(lev, wh, cal_flag)
        arrs[k] = np.nan_to_num(R.loc[s:e, sp.assets].to_numpy(dtype=float), nan=0.0)
    out = []
    for it, idx in enumerate(idxs):
        w3 = np.cumprod(1.0+syn3[idx]); w2 = np.cumprod(1.0+syn2[idx])
        rec = dict(it=it, wipe3=bool((w3 < 0.01).any()), wipe2=bool((w2 < 0.01).any()),
                   min3=float(w3.min()), min2=float(w2.min()))
        for k in keys:
            sp = PORTS[k][1]
            bs = pd.DataFrame(arrs[k][idx], index=cal, columns=sp.assets)
            eq = L.FastBacktest(sp, bs, L.FX, s, e).run(keep_holdings=False)['equity']
            rec[f'{k}_cagr'] = M.cagr(eq); rec[f'{k}_mdd'] = M.mdd(eq)
        out.append(rec)
    d = pd.DataFrame(out)
    d['구간'] = per; d['합성'] = '보정' if cal_flag else '기본'; d['원천징수'] = '적용' if wh else '미적용'
    return d


def bootstrap():
    keys = list(PORTS)
    jobs = [(p, c, True, keys) for p in ['A', 'C'] for c in (False, True)]
    jobs += [(p, False, False, ['P3', 'C100']) for p in ['A', 'C']]   # 기존 S3 재현 확인용
    with Pool(4) as pool:
        ds = pool.map(_boot_job, jobs)
    return pd.concat(ds, ignore_index=True)


def boot_summary(bd):
    rows = []
    for (per, cal, wh), g in bd.groupby(['구간', '합성', '원천징수']):
        for k in PORTS:
            if f'{k}_cagr' not in g or g[f'{k}_cagr'].isna().all():
                continue
            lev = PORTS[k][2]
            c = g[f'{k}_cagr']*100; m = g[f'{k}_mdd']*100
            flag = g['wipe2'] if lev == 'QLD' else g['wipe3']
            rows.append(dict(구간=per, 합성=cal, 원천징수=wh, key=k,
                             CAGR_중앙=c.median(), CAGR_하위10=c.quantile(.10), CAGR_음수비율=(c < 0).mean()*100,
                             MDD_중앙=m.median(), MDD_하위10=m.quantile(.10),
                             붕괴기준=('합성2배' if lev == 'QLD' else '합성3배'),
                             붕괴경로비율=flag.mean()*100,
                             붕괴경로_CAGR_중앙=c[flag].median() if flag.any() else np.nan,
                             비붕괴경로_CAGR_중앙=c[~flag].median() if (~flag).any() else np.nan,
                             합성3배_붕괴비율=g['wipe3'].mean()*100, 합성2배_붕괴비율=g['wipe2'].mean()*100))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    t0 = time.time()
    df, eqs = summary()
    df.to_csv(S4/'s4_summary.csv', index=False, encoding='utf-8-sig')
    pickle.dump(eqs, open(S4/'cache'/'cs_equity.pkl', 'wb'))
    print('summary', time.time()-t0, flush=True)
    st = stress()
    st.to_csv(S4/'s4_stress.csv', index=False, encoding='utf-8-sig')
    print('stress', time.time()-t0, flush=True)
    bd = bootstrap()
    bd.to_csv(S4/'cache'/'s4_boot_cs_paths.csv', index=False)
    bs = boot_summary(bd)
    bs.to_csv(S4/'s4_boot_cs.csv', index=False, encoding='utf-8-sig')
    print('boot', time.time()-t0, flush=True)
    pd.set_option('display.width', 250)
    print(bs.round(2).to_string(index=False))
