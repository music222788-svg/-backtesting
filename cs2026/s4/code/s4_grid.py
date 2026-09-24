"""S4 작업 1: SCHD/SPY/QQQ 비율 그리드 (배당 원천징수 15% 적용).

사용: python s4_grid.py base      -> 10%p 간격 66개 조합
      python s4_grid.py refine    -> s4/cache/refine_list.json 의 5%p 조합
조합마다 s4/cache/grid/<key>.pkl 로 저장(재실행 시 건너뜀).
"""
import sys, pathlib, pickle, json, time, itertools
from multiprocessing import Pool
import numpy as np, pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s4_lib as L
import metrics as M

CACHE = L.S4/'cache'/'grid'
CACHE.mkdir(parents=True, exist_ok=True)
PERIODS = ['A', 'B', 'C', 'S1', 'S2', 'S3', 'OOS2']
R = L.returns(wh=True)
STARTS = L.monthly_starts()
BOOT = {p: L.boot_indices(p) for p in ['A', 'C']}


def key_of(w):
    return 'S{:03d}_P{:03d}_Q{:03d}'.format(*(int(round(w[a]*100)) for a in ['SCHD', 'SPY', 'QQQ']))


def base_combos():
    out = []
    for s, q in itertools.product(range(0, 11), range(0, 11)):
        if s+q <= 10:
            out.append({'SCHD': s/10, 'SPY': (10-s-q)/10, 'QQQ': q/10})
    return out


def one(w):
    k = key_of(w)
    f = CACHE/f'{k}.pkl'
    if f.exists():
        return k
    t0 = time.time()
    sp = L.spec_b(w)
    res = {'w': w, 'key': k, 'per': {}}
    for p in PERIODS:
        d, pre, post = L.evaluate(sp, R, p)
        res['per'][p] = d
    res['roll10'] = L.rolling10_after_tax(sp, R, STARTS)
    for p, (cal, idxs) in BOOT.items():
        s, e = L.PER[p]
        sub = R.loc[s:e, sp.assets].to_numpy(dtype=float)
        sub = np.nan_to_num(sub, nan=0.0)
        cg, md = [], []
        for idx in idxs:
            bs = pd.DataFrame(sub[idx], index=cal, columns=sp.assets)
            eq = L.FastBacktest(sp, bs, L.FX, s, e).run(keep_holdings=False)['equity']
            cg.append(M.cagr(eq)); md.append(M.mdd(eq))
        res[f'boot_{p}'] = (np.array(cg), np.array(md))
    pickle.dump(res, open(f, 'wb'))
    print(f'{k} done {time.time()-t0:.0f}s', flush=True)
    return k


if __name__ == '__main__':
    mode = sys.argv[1]
    combos = base_combos() if mode == 'base' else json.load(open(L.S4/'cache'/'refine_list.json'))
    print(f'{mode}: {len(combos)} combos', flush=True)
    with Pool(4) as pool:
        for _ in pool.imap_unordered(one, combos):
            pass
    print('ALL DONE', flush=True)
