"""S5 적립식 엔진 검증.
(1) 적립 0·원천징수 적용·USD 과세·초기 10,000 USD -> S4 일시투자 세후/세전 최종자산과 일치 (허용 $1)
(3) 손계산 예제: 적립일 체결, 미달 자산 배분, 원화 기준 과세, 환전비용
(2) 기존 test_s2_timing.py 21/21 은 s5_run_tests.sh 에서 별도 실행
"""
import sys, pathlib, traceback
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import s5_lib as S
from dca_engine import simulate, calendar_struct, allocate, Groups
from engine import Spec

RES = []


def check(name, fn):
    try:
        fn(); RES.append((name, 'PASS')); print('  PASS ', name)
    except Exception as ex:
        RES.append((name, 'FAIL')); print('  FAIL ', name, ex); traceback.print_exc()


def t1_match_s4():
    s4 = pd.read_csv(S.ROOT/'s4'/'s4_summary.csv')
    s4 = s4[(s4.원천징수 == '적용') & (s4.합성 == '기본')]
    R, D = S.frames()
    worst = 0.0
    for key, s4key in [('P3', 'P3'), ('A80', 'A80'), ('C100', 'C100'), ('B-opt', 'B-opt')]:
        sp, _ = S.PORTS[key]
        for per in ['A', 'B', 'C']:
            cal = S.window(*S.L.PER[per])
            cs = calendar_struct(cal)
            Rv, Dv = S.arrays(sp, R, D, cal)
            FXv = S.FX_ALL.loc[cal].to_numpy()
            post = simulate(sp, Rv, Dv, FXv, cs, init_usd=10_000.0, tax='usd')['eq'][-1]
            pre = simulate(sp, Rv, Dv, FXv, cs, init_usd=10_000.0, tax=None)['eq'][-1]
            row = s4[(s4.key == s4key) & (s4.구간 == per)].iloc[0]
            e1, e2 = abs(post-row['최종자산_세후']), abs(pre-row['최종자산_세전'])
            worst = max(worst, e1, e2)
            print(f'     {key:6s} {per}  세후 {post:,.2f} vs S4 {row["최종자산_세후"]:,.2f}  (차 {e1:.2e}) '
                  f' 세전 차 {e2:.2e}')
            assert e1 < 1.0 and e2 < 1.0
    print(f'     최대 차이 ${worst:.2e}')


def mini(n_days=70, start='2021-01-04'):
    cal = S.CAL_ALL[S.CAL_ALL >= pd.Timestamp(start)][:n_days]
    return cal, calendar_struct(cal)


SP2 = Spec('t', ['SPY', 'QQQ'], ['SPY', 'QQQ'], [], 1.0, 'none',
           core_tgt={'SPY': .5, 'QQQ': .5}, core_band={'SPY': (.5, .5), 'QQQ': (.5, .5)})


def t2_contrib_day_execution():
    cal, cs = mini()
    T = len(cal)
    Rv = np.zeros((T, 2)); Dv = np.zeros((T, 2)); FX = np.full(T, 1000.0)
    k = np.where(cs['is_contrib'], 1_000_000.0, 0.0)
    out = simulate(SP2, Rv, Dv, FX, cs, init_krw=10_000_000.0, contrib_krw=k, tax=None,
                   method='M2', liq=False, record=True)
    days = np.nonzero(cs['is_contrib'])[0]
    # 첫 거래일(시작일 포함) = 2021-01-04, 02-01, 03-01, 04-01
    assert [str(cal[i].date()) for i in days] == ['2021-01-04', '2021-02-01', '2021-03-01', '2021-04-01'], \
        [str(cal[i].date()) for i in days]
    per = 1_000_000/1000*(1-0.001)/(1+0.001)       # 적립 1회 시장가치 (USD)
    init = 10_000_000/1000*(1-0.001)/(1+0.001)
    eq = out['eq']
    for j, i in enumerate(days):
        expect = init + per*(j+1)
        assert abs(eq[i]-expect) < 1e-9, (i, eq[i], expect)     # 같은 날 종가에 반영
        if i > 0:
            assert abs(eq[i-1]-(init+per*j)) < 1e-9             # 전날에는 미반영


def t3_underweight_allocation():
    G = Groups(SP2, ['SPY', 'QQQ'])
    v = np.array([700.0, 300.0])
    # 적립 100: 목표 = (1000+100)*0.5 = 550 -> QQQ 부족 250 >= 100 -> 전부 QQQ
    a = allocate(G, v, 100.0, 'M1')
    assert np.allclose(a, [0.0, 100.0]), a
    # 적립 600: 목표 800 -> 부족 SPY 100, QQQ 500 (합 600) -> 정확히 부족분
    a = allocate(G, v, 600.0, 'M1')
    assert np.allclose(a, [100.0, 500.0]), a
    # 적립 1000: 목표 1000 -> 부족 SPY 300, QQQ 700 = 1000 -> 그대로
    a = allocate(G, np.array([700.0, 300.0]), 1000.0, 'M1')
    assert np.allclose(a, [300.0, 700.0]), a
    # 부족분 합(목표 = 1100*0.5=550 -> QQQ 250)이 적립 400 보다 작음 -> 250 + 나머지 150 을 50:50
    a = allocate(G, v, 400.0, 'M1')
    # 목표 700: SPY 부족 0, QQQ 400 -> 전부 QQQ
    assert np.allclose(a, [0.0, 400.0]), a
    a = allocate(G, np.array([500.0, 500.0]), 200.0, 'M1')      # 균형 상태 -> 목표비중
    assert np.allclose(a, [100.0, 100.0]), a
    # M2 는 항상 목표비중
    assert np.allclose(allocate(G, v, 100.0, 'M2'), [50.0, 50.0])
    # A 계열: 최상위 부족분 먼저 (코어 70 / 위성 30)
    sp = S.PORTS['P3'][0]
    GA = Groups(sp, list(sp.assets))
    v = np.array([420.0, 140.0, 70.0, 70.0, 100.0, 100.0])   # 코어 700, 위성 200 (목표 30% 미달)
    a = allocate(GA, v, 100.0, 'M1')
    # V'=1000: 코어 목표 700 부족 0, 위성 목표 300 부족 100 -> 위성 100 -> 위성 목표 TQQQ 180 부족 80, SGOV 120 부족 20
    assert np.allclose(a, [0, 0, 0, 0, 80.0, 20.0]), a
    # 동결(S2) 중에는 TQQQ 몫을 SGOV 로
    a = allocate(GA, v, 100.0, 'M1', frozen=True)
    assert np.allclose(a, [0, 0, 0, 0, 0, 100.0]), a


def t4_krw_tax():
    """1자산 단순보유: 원화 기준 과세 손계산."""
    sp = S.PORTS['SPY'][0]
    cal, cs = mini(n_days=30, start='2021-12-01')            # 2021-12-01 ~ 2022-01-12
    T = len(cal)
    Rv = np.zeros((T, 1)); Dv = np.zeros((T, 1))
    Rv[5, 0] = 1.0                                           # 가격 2배
    FX = np.full(T, 1000.0); FX[-1] = 1300.0                 # 종료일 환율 1300
    FX[cal.year == 2022] = 1300.0
    c, fxc = 0.001, 0.001
    out = simulate(sp, Rv, Dv, FX, cs, init_krw=100_000_000.0, tax='krw', record=True)
    u0 = 100_000_000/1000*(1-fxc)                             # 99,900 USD
    basis_k = u0*1000                                        # 취득가액(원) = 매수대금(수수료 포함) x 취득일 환율
    val = u0/(1+c)*2                                         # 종료일 평가 (USD)
    proceeds_k = val*(1-c)*1300
    gain_k = proceeds_k - basis_k
    tax_k = 0.22*max(0, gain_k-2_500_000)
    y = out['years']
    assert abs(y.iloc[0]['tax_krw']) < 1e-6                   # 2021 연말: 실현 없음
    assert abs(y.iloc[-1]['gain_krw']-gain_k) < 1e-3, (y.iloc[-1]['gain_krw'], gain_k)
    assert abs(y.iloc[-1]['tax_krw']-tax_k) < 1e-3, (y.iloc[-1]['tax_krw'], tax_k)
    final_usd = val*(1-c) - tax_k/1300
    assert abs(out['eq'][-1]-final_usd) < 1e-6
    # USD 과세와의 차이: 환율 상승분도 과세
    o2 = simulate(sp, Rv, Dv, FX, cs, init_krw=100_000_000.0, tax='usd', record=True)
    gain_u = val*(1-c) - u0
    assert abs(o2['years'].iloc[-1]['gain_usd']-gain_u) < 1e-6
    assert abs(o2['years'].iloc[-1]['tax_usd'] - 0.22*max(0, gain_u-2_500_000/1300)) < 1e-6


def t5_fx_cost():
    cal, cs = mini()
    T = len(cal)
    Rv = np.zeros((T, 2)); Dv = np.zeros((T, 2)); FX = np.linspace(1100, 1200, T)
    k = np.where(cs['is_contrib'], 4_500_000.0, 0.0)
    out = simulate(SP2, Rv, Dv, FX, cs, init_krw=150_000_000.0, contrib_krw=k, tax=None, liq=False)
    n = int(cs['is_contrib'].sum())
    assert abs(out['fx_cost'] - 0.001*(150_000_000+4_500_000*n)) < 1e-6
    assert abs(out['paid_in'] - (150_000_000+4_500_000*n)) < 1e-6
    usd = 150_000_000/FX[0]*(0.999)/(1.001) + sum(4_500_000/FX[i]*0.999/1.001 for i in np.nonzero(cs['is_contrib'])[0])
    assert abs(out['eq'][-1]-usd) < 1e-6


def t6_div_income():
    sp = S.PORTS['SPY'][0]
    cal, cs = mini(n_days=30, start='2021-12-01')
    T = len(cal)
    Rv = np.zeros((T, 1)); Dv = np.zeros((T, 1)); Dv[3, 0] = 0.01
    FX = np.full(T, 1200.0)
    out = simulate(sp, Rv, Dv, FX, cs, init_usd=10_000.0, tax='krw', record=True, liq=False)
    v0 = 10_000/1.001
    assert abs(out['years'].iloc[0]['div_krw'] - v0*0.01*1200) < 1e-6


if __name__ == '__main__':
    print('=== S5 적립식 엔진 검증 ===')
    check('V1  적립0·USD과세 = S4 일시투자 (P3/A80/코어100/B-opt × A/B/C, 허용 $1)', t1_match_s4)
    check('V3a 적립일 = 매월 첫 거래일, 같은 날 종가 반영', t2_contrib_day_execution)
    check('V3b 미달 자산 우선 배분(M1)·목표비중(M2)·최상위→그룹·동결', t3_underweight_allocation)
    check('V3c 원화 기준 양도차익·세액 손계산 (USD 기준 병기)', t4_krw_tax)
    check('V3d 환전비용 0.1% 누계·환전 후 매수금액', t5_fx_cost)
    check('V3e 배당 원화 합계(종합과세 점검용)', t6_div_income)
    n = sum(r == 'PASS' for _, r in RES)
    print(f'\n{n}/{len(RES)} PASS')
