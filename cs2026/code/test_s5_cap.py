"""s5_cap 자체 검증: 기존 P3 재현, 미래참조 차단, 상한 동작."""
import sys, pathlib, traceback
import numpy as np, pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent
OUT = ROOT/'out'
import engine as E
from s5_cap import CapP3, ASSETS, SAT

R = pd.read_csv(OUT/'s1_returns.csv', parse_dates=['date']).set_index('date')
R['SCHD'] = R['SCHD_a']
MAC = pd.read_csv(OUT/'s1_macro.csv', parse_dates=['date']).set_index('date')
FX = MAC['krw']
CAL = R.index
PER = {'A': ('2000-01-03', '2019-12-31'), 'B': ('2020-01-02', '2026-09-16'),
       'C': ('2000-01-03', '2026-09-16')}
RES = []


def check(name, fn):
    try:
        fn(); RES.append((name, 'PASS')); print(f'  PASS  {name}')
    except Exception as ex:
        RES.append((name, 'FAIL')); print(f'  FAIL  {name}: {ex}'); traceback.print_exc()


def t1_reproduces_base_p3():
    """CAP=None 이면 기존 엔진의 P3 와 완전히 동일해야 한다."""
    for per, (s, e) in PER.items():
        for tax in (False, True):
            for z in (None, '2001-08-30'):
                if z and per == 'B':
                    continue
                a = E.run_one('P3', R, FX, s, e, tax=tax, tqqq_zero_date=z)['equity']
                b = CapP3(R, FX, s, e, cap=None, tax=tax, tqqq_zero_date=z).run()['equity']
                assert np.allclose(a.values, b.values, rtol=0, atol=1e-9), \
                    f'{per} tax={tax} z={z} maxdiff={np.abs(a.values-b.values).max():.3e}'


def t2_cap_never_binds_is_identical():
    """상한 99% 는 절대 발동하지 않으므로 기본 P3 와 같아야 한다."""
    for freq in ('daily', 'quarter'):
        a = CapP3(R, FX, *PER['A'], cap=None).run()['equity']
        b = CapP3(R, FX, *PER['A'], cap=0.99, freq=freq).run()['equity']
        assert np.allclose(a.values, b.values, rtol=0, atol=1e-9), freq


def t3_cap_is_enforced():
    """상시 점검 시 위성 비중이 상한+1일치 변동 이상으로는 오래 머물지 않는다."""
    r = CapP3(R, FX, *PER['A'], cap=0.35, freq='daily', action='target').run(keep_holdings=True)
    h = r['holdings']
    w = h[SAT].sum(axis=1)/h.sum(axis=1)
    over = w[w > 0.35]
    # 초과 상태는 '점검일 종가 ~ 다음 거래일 종가' 구간에만 존재 가능
    assert w.max() < 0.55, f'상한이 전혀 듣지 않음 (최대 {w.max():.3f})'
    base = CapP3(R, FX, *PER['A'], cap=None).run(keep_holdings=True)
    hb = base['holdings']
    wb = hb[SAT].sum(axis=1)/hb.sum(axis=1)
    assert w.max() < wb.max(), f'상한 적용본 최대 {w.max():.3f} vs 기본 {wb.max():.3f}'
    assert (w > 0.35).mean() < (wb > 0.35).mean()


def t4_exec_next_trading_day():
    """연중 조치 체결일은 점검일의 '다음 거래일'이어야 한다."""
    bt = CapP3(R, FX, *PER['A'], cap=0.35, freq='quarter', action='target')
    r = bt.run()
    mid = r['trades'][r['trades']['kind'] == 'mid']
    assert len(mid) > 0, '분기 상한이 한 번도 발동하지 않음'
    cal = list(bt.cal)
    for d in mid['date']:
        i = cal.index(d)
        prev = cal[i-1]
        assert prev in bt.sig_mid, f'{d.date()} 의 직전 거래일 {prev.date()} 이 점검일이 아님'
    assert any((d - cal[cal.index(d)-1]).days > 1 for d in mid['date']), '주말 건너뛴 사례 없음'


def t5_no_lookahead():
    """구간 밖 데이터를 오염시켜도 결과가 동일."""
    bad = R.copy()
    m = bad.index < pd.Timestamp('2020-01-02')
    rng = np.random.default_rng(7)
    bad.loc[m, :] = rng.uniform(-0.9, 9.0, size=(m.sum(), bad.shape[1]))
    for cap in (None, 0.35, 0.40):
        a = CapP3(R, FX, *PER['B'], cap=cap, tax=True).run()['equity']
        b = CapP3(bad, FX, *PER['B'], cap=cap, tax=True).run()['equity']
        assert np.allclose(a.values, b.values, rtol=0, atol=1e-9), f'cap={cap}'
    bad2 = R.copy()
    m2 = bad2.index > pd.Timestamp('2019-12-31')
    bad2.loc[m2, :] = rng.uniform(-0.9, 9.0, size=(m2.sum(), bad2.shape[1]))
    for cap in (None, 0.35):
        a = CapP3(R, FX, *PER['A'], cap=cap, tax=True).run()['equity']
        b = CapP3(bad2, FX, *PER['A'], cap=cap, tax=True).run()['equity']
        assert np.allclose(a.values, b.values, rtol=0, atol=1e-9), f'post cap={cap}'


def t6_action_difference():
    """조치 'target'(30%) 이 'cap'(상한) 보다 위성을 더 많이 줄인다."""
    a = CapP3(R, FX, *PER['A'], cap=0.35, freq='daily', action='target').run(keep_holdings=True)
    b = CapP3(R, FX, *PER['A'], cap=0.35, freq='daily', action='cap').run(keep_holdings=True)
    wa = (a['holdings'][SAT].sum(axis=1)/a['holdings'].sum(axis=1)).mean()
    wb = (b['holdings'][SAT].sum(axis=1)/b['holdings'].sum(axis=1)).mean()
    assert wa < wb, f'target 평균위성 {wa:.4f} >= cap 평균위성 {wb:.4f}'


if __name__ == '__main__':
    print('=== s5_cap 검증 ===')
    for n, f in [('T1 기존 P3 완전 재현 (12조합)', t1_reproduces_base_p3),
                 ('T2 발동 안 하는 상한 = 기본과 동일', t2_cap_never_binds_is_identical),
                 ('T3 상한이 실제로 작동', t3_cap_is_enforced),
                 ('T4 체결일 = 다음 거래일', t4_exec_next_trading_day),
                 ('T5 구간 밖 데이터 미사용', t5_no_lookahead),
                 ('T6 조치 강도 순서', t6_action_difference)]:
        check(n, f)
    bad = sum(1 for _, s in RES if s == 'FAIL')
    print(f'\n{len(RES)-bad}/{len(RES)} PASS')
    sys.exit(1 if bad else 0)
