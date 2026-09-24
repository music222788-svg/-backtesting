"""고속 구현(engine_fast) 이 참조 구현(engine) 과 완전히 동일한지 검증."""
import sys, pathlib, time
import numpy as np, pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent
import engine as E
from engine_fast import run_fast

R = pd.read_csv(ROOT/'out'/'s1_returns.csv', parse_dates=['date']).set_index('date')
MAC = pd.read_csv(ROOT/'out'/'s1_macro.csv', parse_dates=['date']).set_index('date')
FX = MAC['krw']

PER = {'A': ('2000-01-03', '2019-12-31'),
       'B': ('2020-01-02', '2026-09-16'),
       'C': ('2000-01-03', '2026-09-16')}
KEYS = ['P1', 'P2', 'P3', 'P4', 'B1', 'B2', 'C1', 'S1', 'S2', 'A1', 'A2', 'A3']

if __name__ == '__main__':
    bad = 0
    n = 0
    for per, (s, e) in PER.items():
        for k in KEYS:
            for tax in (False, True):
                for z in (None, '2001-08-30'):
                    if z and per == 'B':
                        continue
                    for schd in (['SCHD_a'] if (tax or z) else ['SCHD_a', 'SCHD_b', 'SCHD_c']):
                        a = E.run_one(k, R, FX, s, e, schd=schd, tax=tax, tqqq_zero_date=z)
                        b = run_fast(k, R, FX, s, e, schd=schd, tax=tax, tqqq_zero_date=z,
                                     keep_holdings=True)
                        n += 1
                        if not np.allclose(a['equity'].values, b['equity'].values,
                                           rtol=0, atol=1e-9):
                            d = np.abs(a['equity'].values-b['equity'].values).max()
                            print(f'  MISMATCH equity {per} {k} tax={tax} z={z} {schd} maxdiff={d:.3e}')
                            bad += 1
                        if len(a['trades']) != len(b['trades']):
                            print(f'  MISMATCH trades {per} {k} tax={tax} z={z} '
                                  f'{len(a["trades"])} vs {len(b["trades"])}')
                            bad += 1
                        ta = float(a['taxes']['tax'].sum()) if len(a['taxes']) else 0.0
                        tb = float(b['taxes']['tax'].sum()) if len(b['taxes']) else 0.0
                        if abs(ta-tb) > 1e-9:
                            print(f'  MISMATCH tax {per} {k} {ta} vs {tb}')
                            bad += 1
    print(f'비교 {n} 조합, 불일치 {bad} 건')

    t = time.time()
    for _ in range(20):
        run_fast('P1', R, FX, '2000-01-03', '2026-09-16')
    dt = (time.time()-t)/20
    t2 = time.time()
    for _ in range(3):
        E.run_one('P1', R, FX, '2000-01-03', '2026-09-16')
    dt2 = (time.time()-t2)/3
    print(f'참조 {dt2*1000:.1f} ms -> 고속 {dt*1000:.2f} ms ({dt2/dt:.0f}배)')
    print(f'부트스트랩 1000회 x 7전략 x 2구간 예상 {1000*7*2*dt/60:.2f} 분')
    sys.exit(1 if bad else 0)
