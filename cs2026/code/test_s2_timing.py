"""2단계 단위테스트: 신호·체결 시점 정렬과 미래참조 차단 검증.

실행: python code/test_s2_timing.py
"""
import sys, pathlib, traceback
import numpy as np, pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import engine as E

ROOT = pathlib.Path(__file__).resolve().parent.parent
R = pd.read_csv(ROOT/'out'/'s1_returns.csv', parse_dates=['date']).set_index('date')
M = pd.read_csv(ROOT/'out'/'s1_macro.csv', parse_dates=['date']).set_index('date')
FX = M['krw']
CAL = R.index

RESULTS = []


def check(name, fn):
    try:
        fn()
        RESULTS.append((name, 'PASS', ''))
        print(f'  PASS  {name}')
    except Exception as ex:
        RESULTS.append((name, 'FAIL', repr(ex)))
        print(f'  FAIL  {name}: {ex}')
        traceback.print_exc()


def synth(start, end, spec_assets, const=0.0):
    """구간 내 모든 자산이 const 수익률인 합성 데이터."""
    idx = CAL[(CAL >= pd.Timestamp(start)) & (CAL <= pd.Timestamp(end))]
    return pd.DataFrame({a: const for a in spec_assets}, index=idx)


# ---------------------------------------------------------------- 1
def t1_exec_is_next_trading_day():
    """체결일은 반드시 캘린더상 '다음 거래일'이어야 한다 (+1일 아님)."""
    res = E.run_one('P1', R, FX, '2020-01-02', '2026-09-16')
    sig = [d for d in E.dec_last_trading_days(CAL[(CAL >= '2020-01-02') & (CAL <= '2026-09-16')])
           if d != pd.Timestamp('2026-09-16')]
    exec_days = list(res['trades']['date'])
    assert len(sig) == len(exec_days), f'신호 {len(sig)} vs 체결 {len(exec_days)}'
    for s, e in zip(sig, exec_days):
        i = list(CAL).index(s)
        assert e == CAL[i+1], f'신호 {s.date()} 의 체결일이 {e.date()}, 기대 {CAL[i+1].date()}'
        assert (e - s).days >= 1
    # 실제로 달력상 하루 뒤가 아닌 경우(주말/휴장)가 포함되어야 의미있는 검증
    assert any((e - s).days > 1 for s, e in zip(sig, exec_days)), '주말 건너뛴 사례가 없음'


# ---------------------------------------------------------------- 2
def t2_decision_uses_signal_day_only():
    """신호일 종가 비중으로만 판단. 체결일의 폭등은 '판단'을 바꾸면 안 된다."""
    assets = E.CORE_ASSETS
    idx = CAL[(CAL >= '2020-01-02') & (CAL <= '2021-06-30')]
    base = pd.DataFrame(0.0, index=idx, columns=assets)
    S = [d for d in E.dec_last_trading_days(idx)][0]      # 2020-12-31
    Eday = idx[list(idx).index(S)+1]                      # 2021-01-04

    # 신호일까지 비중은 정확히 목표(무변동) -> 밴드 이탈 없음.
    # 체결일에 SPY +100% : '체결일 기준'으로 보면 밴드를 크게 벗어난다.
    d = base.copy()
    d.loc[Eday, 'SPY'] = 1.0
    bt = E.Backtest(E.SPECS['C1'], d, FX, idx[0], idx[-1], liquidate_at_end=False)
    r = bt.run()
    assert len(r['trades']) == 0, '체결일 가격으로 판단해 리밸런싱이 일어남 (미래참조)'
    h = r['holdings'].loc[Eday]
    w = h/h.sum()
    assert w['SPY'] > 0.70, f'체결일 폭등이 반영되지 않음: SPY 비중 {w["SPY"]:.4f}'

    # 대조군: 신호일 '이전'에 같은 폭등 -> 반드시 리밸런싱 발생
    d2 = base.copy()
    d2.loc[idx[list(idx).index(S)-1], 'SPY'] = 1.0
    r2 = E.Backtest(E.SPECS['C1'], d2, FX, idx[0], idx[-1], liquidate_at_end=False).run()
    assert len(r2['trades']) == 1, '신호일 이전 이탈인데 리밸런싱이 없음'
    h2 = r2['holdings'].loc[Eday]; w2 = h2/h2.sum()
    assert abs(w2['SPY']-0.60) < 1e-9 and abs(w2['SCHD']-0.20) < 1e-9, f'목표 미복원 {w2.to_dict()}'


# ---------------------------------------------------------------- 3
def t3_execution_at_exec_day_close():
    """체결 금액은 체결일 종가 기준이어야 한다."""
    assets = E.CORE_ASSETS
    idx = CAL[(CAL >= '2020-01-02') & (CAL <= '2021-06-30')]
    S = [d for d in E.dec_last_trading_days(idx)][0]
    Eday = idx[list(idx).index(S)+1]
    iS = list(idx).index(S)

    def build(exec_move):
        d = pd.DataFrame(0.0, index=idx, columns=assets)
        d.iloc[iS-1, d.columns.get_loc('SPY')] = 1.0     # 신호일 이전 이탈 -> 리밸런싱 확정
        d.loc[Eday, 'GLD'] = exec_move                   # 체결일 GLD 변동
        return d

    tot = {}
    for mv in (0.0, 0.5):
        r = E.Backtest(E.SPECS['C1'], build(mv), FX, idx[0], idx[-1], liquidate_at_end=False).run()
        h = r['holdings'].loc[Eday]
        tot[mv] = h.sum()
        w = h/h.sum()
        assert abs(w['GLD']-0.10) < 1e-9, f'체결 후 목표비중 아님 {w.to_dict()}'
    # 체결일 시점 GLD 비중 = 0.1/1.6 = 6.25% -> +50% 이면 총액 약 +3.1%
    ratio = tot[0.5]/tot[0.0]
    assert 1.025 < ratio < 1.040, f'체결일 종가가 체결 규모에 반영되지 않음 (비율 {ratio:.5f})'


# ---------------------------------------------------------------- 4
def t4_no_pre_start_data():
    """구간 시작일 이전 데이터를 오염시켜도 결과가 동일해야 한다."""
    bad = R.copy()
    m = bad.index < pd.Timestamp('2020-01-02')
    rng = np.random.default_rng(0)
    bad.loc[m, :] = rng.uniform(-0.9, 9.0, size=(m.sum(), bad.shape[1]))
    for key in ['P1', 'P2', 'P3', 'P4', 'C1', 'B1', 'B2']:
        a = E.run_one(key, R, FX, '2020-01-02', '2026-09-16', tax=True)['equity']
        b = E.run_one(key, bad, FX, '2020-01-02', '2026-09-16', tax=True)['equity']
        assert np.allclose(a.values, b.values, rtol=0, atol=1e-9), f'{key}: 시작일 이전 데이터 사용됨'


# ---------------------------------------------------------------- 5
def t5_no_post_end_data():
    """구간 종료일 이후 데이터를 오염시켜도 결과가 동일해야 한다."""
    bad = R.copy()
    m = bad.index > pd.Timestamp('2019-12-31')
    rng = np.random.default_rng(1)
    bad.loc[m, :] = rng.uniform(-0.9, 9.0, size=(m.sum(), bad.shape[1]))
    for key in ['P1', 'P2', 'P3', 'P4', 'C1', 'B1', 'B2']:
        a = E.run_one(key, R, FX, '2000-01-03', '2019-12-31', tax=True)['equity']
        b = E.run_one(key, bad, FX, '2000-01-03', '2019-12-31', tax=True)['equity']
        assert np.allclose(a.values, b.values, rtol=0, atol=1e-9), f'{key}: 종료일 이후 데이터 사용됨'


# ---------------------------------------------------------------- 6
def t6_no_signal_execution_on_last_day():
    """구간 마지막 거래일이 12월 말일이어도 그날 신호는 체결되지 않는다."""
    res = E.run_one('P1', R, FX, '2000-01-03', '2019-12-31')
    tr = res['trades']
    assert pd.Timestamp('2019-12-31') not in set(tr['date']), '종료일에 체결 발생'
    # 신호 19회(2000~2018) -> 체결 19회, 2019-12-31 신호는 체결 없음
    sig_all = E.dec_last_trading_days(CAL[(CAL >= '2000-01-03') & (CAL <= '2019-12-31')])
    assert len(tr) == len(sig_all)-1, f'체결 {len(tr)}회, 기대 {len(sig_all)-1}회'
    assert tr['date'].iloc[-1] == pd.Timestamp('2019-01-02')


# ---------------------------------------------------------------- 7
def t7_band_boundary():
    """밴드 경계: 내부이면 무거래, 외부이면 거래."""
    assets = E.CORE_ASSETS
    idx = CAL[(CAL >= '2020-01-02') & (CAL <= '2021-06-30')]
    S = [d for d in E.dec_last_trading_days(idx)][0]
    iS = list(idx).index(S)

    def run_with_spy_move(mv):
        d = pd.DataFrame(0.0, index=idx, columns=assets)
        d.iloc[iS-1, d.columns.get_loc('SPY')] = mv
        return E.Backtest(E.SPECS['C1'], d, FX, idx[0], idx[-1], liquidate_at_end=False).run()

    # SPY 비중 = 0.6(1+m) / (0.6(1+m)+0.4).  상단 0.65 도달 m* = 0.2381
    lo = run_with_spy_move(0.230)      # SPY 비중 0.6485 -> 밴드 내
    hi = run_with_spy_move(0.245)      # SPY 비중 0.6514 -> 밴드 이탈
    assert len(lo['trades']) == 0, '밴드 내부인데 리밸런싱 발생'
    assert len(hi['trades']) == 1, '밴드 이탈인데 리밸런싱 없음'


# ---------------------------------------------------------------- 8
def t8_no_drift_no_cost():
    """전 자산 동일 수익률이면 비중이 변하지 않아 매매·비용이 없어야 한다."""
    idx = CAL[(CAL >= '2000-01-03') & (CAL <= '2019-12-31')]
    d = pd.DataFrame(0.0005, index=idx, columns=E.CORE_ASSETS+E.SAT_ASSETS)
    for key in ['P1', 'P2', 'P3', 'P4', 'C1']:
        r = E.Backtest(E.SPECS[key], d, FX, idx[0], idx[-1], liquidate_at_end=False).run()
        assert len(r['trades']) == 0, f'{key}: 드리프트 없는데 매매 {len(r["trades"])}회'
        exp = 10_000.0/1.001*(1.0005**(len(idx)-1))
        got = r['equity'].iloc[-1]
        assert abs(got-exp)/exp < 1e-10, f'{key}: {got} vs {exp}'


# ---------------------------------------------------------------- 9
def t9_buyhold_and_tax_hand_check():
    """B1 은 매매 0회. 청산·세금 계산을 손계산과 대조."""
    r = E.run_one('B1', R, FX, '2020-01-02', '2026-09-16', tax=True)
    assert len(r['trades']) == 0, 'B1 에서 매매 발생'

    c = 0.0010
    sub = R.loc['2020-01-02':'2026-09-16', 'SPY']
    grow = float((1.0+sub.iloc[1:]).prod())
    v_end = 10_000.0/(1.0+c)*grow                     # 청산 직전 평가액
    gain = v_end - 10_000.0 - c*v_end                 # 취득원가 10,000, 매도수수료 공제
    fx_end = float(FX.loc['2026-09-16'])
    tax = 0.22*max(0.0, gain - 2_500_000.0/fx_end)
    exp = v_end*(1.0-c) - tax
    got = float(r['equity'].iloc[-1])
    assert abs(got-exp) < 1e-6, f'손계산 {exp:.6f} vs 엔진 {got:.6f}'

    pre = float(E.run_one('B1', R, FX, '2020-01-02', '2026-09-16', tax=False)['equity'].iloc[-1])
    assert abs(pre - v_end*(1.0-c)) < 1e-6
    assert abs((pre-got) - tax) < 1e-6, '세금 차감액 불일치'


# ---------------------------------------------------------------- 10
def t10_moving_average_basis():
    """이동평균 원가: 절반 매도 시 원가도 절반만 소멸해야 한다."""
    assets = E.CORE_ASSETS
    idx = CAL[(CAL >= '2020-01-02') & (CAL <= '2021-06-30')]
    S = [d for d in E.dec_last_trading_days(idx)][0]
    iS = list(idx).index(S)
    d = pd.DataFrame(0.0, index=idx, columns=assets)
    d.iloc[iS-1, d.columns.get_loc('SPY')] = 1.0      # SPY 2배 -> 밴드 이탈
    r = E.Backtest(E.SPECS['C1'], d, FX, idx[0], idx[-1], liquidate_at_end=False).run()
    g = float(r['trades']['realized'].iloc[0])

    c = 0.0010
    V0 = 10_000.0/(1.0+c)
    v = {'SPY': V0*0.60*2, 'SCHD': V0*0.20, 'IEF': V0*0.10, 'GLD': V0*0.10}
    b = {'SPY': 10_000.0*0.60, 'SCHD': 10_000.0*0.20, 'IEF': 10_000.0*0.10, 'GLD': 10_000.0*0.10}
    V = sum(v.values())
    t = {a: V*E.CORE_TGT[a] for a in assets}
    turn = sum(max(0.0, v[a]-t[a]) for a in assets)
    scale = (V-2*c*turn)/V
    exp = 0.0
    for a in assets:
        dd = t[a]*scale - v[a]
        if dd < 0:
            s = -dd
            exp += s - b[a]*(s/v[a]) - c*s
    assert abs(g-exp) < 1e-9, f'이동평균 원가 손계산 {exp:.8f} vs 엔진 {g:.8f}'


# ---------------------------------------------------------------- 11
def t11_loss_no_carryforward():
    """연간 손실은 이월되지 않는다: 손실 연도 세금 0, 다음 해 공제 없음."""
    v = {'X': 100.0}; b = {'X': 100.0}
    bt = E.Backtest(E.SPECS['B1'], R.loc['2020-01-02':'2026-09-16'], FX,
                    '2020-01-02', '2026-09-16', tax=True)
    assert bt._pay_tax(dict(v), dict(b), -50_000.0, 1300.0) == 0.0, '손실인데 세금 발생'
    fx = 1300.0
    g = 2_500_000.0/fx + 1000.0
    vv, bb = {'X': 1_000_000.0}, {'X': 500_000.0}
    paid = bt._pay_tax(vv, bb, g, fx)
    assert abs(paid - 0.22*1000.0) < 1e-9, f'공제 후 세액 불일치: {paid}'


# ---------------------------------------------------------------- 12
def t12_tqqq_zero_variant():
    """TQQQ=0 시나리오: 지정일 이후 TQQQ 보유가 0 으로 유지된다."""
    z = '2001-08-30'
    r = E.run_one('P1', R, FX, '2000-01-03', '2019-12-31', tqqq_zero_date=z)
    h = r['holdings']
    assert (h.loc[pd.Timestamp(z):, 'TQQQ'].abs() < 1e-12).all(), '지정일 이후 TQQQ 잔존'
    assert h.loc[:pd.Timestamp(z)-pd.Timedelta(days=1), 'TQQQ'].max() > 0, '지정일 이전에도 0'
    base = E.run_one('P1', R, FX, '2000-01-03', '2019-12-31')['equity']
    assert r['equity'].iloc[-1] < base.iloc[-1], 'TQQQ 소멸인데 성과가 더 좋음'


# ---------------------------------------------------------------- 13~15
def _rule_setup(spy_mv, tqqq_mv):
    """신호일 직전에 SPY/TQQQ 를 움직인 합성 데이터와 (신호일, 체결일) 반환."""
    cols = E.CORE_ASSETS + E.SAT_ASSETS
    idx = CAL[(CAL >= '2020-01-02') & (CAL <= '2021-06-30')]
    S = E.dec_last_trading_days(idx)[0]
    iS = list(idx).index(S)
    d = pd.DataFrame(0.0, index=idx, columns=cols)
    d.iloc[iS-1, d.columns.get_loc('SPY')] = spy_mv
    d.iloc[iS-1, d.columns.get_loc('TQQQ')] = tqqq_mv
    return d, idx, idx[iS+1]


def t13_p1_restore_keeps_core_internal():
    """P1: 최상위는 80:20 복원, 코어 밴드 내이면 코어 내부 비중은 그대로."""
    d, idx, Eday = _rule_setup(0.10, 1.00)
    r = E.Backtest(E.SPECS['P1'], d, FX, idx[0], idx[-1], liquidate_at_end=False).run()
    h = r['holdings'].loc[Eday]; V = h.sum()
    core = h[E.CORE_ASSETS].sum(); sat = h[E.SAT_ASSETS].sum()
    assert abs(core/V-0.80) < 1e-9, f'코어 비중 {core/V:.6f}'
    assert abs(sat/V-0.20) < 1e-9, f'위성 비중 {sat/V:.6f}'
    assert abs(h['TQQQ']/sat-0.60) < 1e-9, '위성 내부 60/40 미복원'
    # 코어 내부는 60/20/10/10 이 아니라 드리프트된 0.528/0.848 = 0.62264 여야 한다
    assert abs(h['SPY']/core-0.528/0.848) < 1e-9, f'코어 내부가 초기화됨 {h["SPY"]/core:.6f}'


def t14_p2_cap_moves_only_excess():
    """P2: 위성 30% 초과 시 정확히 30%로만 낮추고 나머지는 건드리지 않는다."""
    d, idx, Eday = _rule_setup(0.0, 3.00)
    r = E.Backtest(E.SPECS['P2'], d, FX, idx[0], idx[-1], liquidate_at_end=False).run()
    h = r['holdings'].loc[Eday]; V = h.sum()
    sat = h[E.SAT_ASSETS].sum()
    assert abs(sat/V-0.30) < 1e-9, f'상한 적용 후 위성 {sat/V:.6f}, 기대 0.30'
    assert abs(h['TQQQ']/sat-0.60) < 1e-9
    core = h[E.CORE_ASSETS].sum()
    assert abs(h['SPY']/core-0.60) < 1e-9, '코어 내부 비중이 변형됨'

    # 상한 미만이면 최상위는 손대지 않는다
    d2, idx2, E2 = _rule_setup(0.0, 0.20)
    r2 = E.Backtest(E.SPECS['P2'], d2, FX, idx2[0], idx2[-1], liquidate_at_end=False).run()
    h2 = r2['holdings'].loc[E2]; V2 = h2.sum()
    sat2 = h2[E.SAT_ASSETS].sum()
    exp = (0.12*1.2+0.08)/(0.8+0.12*1.2+0.08)
    assert abs(sat2/V2-exp) < 1e-9, f'상한 미만인데 최상위 조정됨 {sat2/V2:.6f} vs {exp:.6f}'


def t15_p4_has_no_cap():
    """P4: 상한이 없어 위성이 30% 를 넘어도 그대로 둔다."""
    d, idx, Eday = _rule_setup(0.0, 3.00)
    r = E.Backtest(E.SPECS['P4'], d, FX, idx[0], idx[-1], liquidate_at_end=False).run()
    h = r['holdings'].loc[Eday]; V = h.sum()
    sat = h[E.SAT_ASSETS].sum()
    exp = (0.18*4+0.12)/(0.70+0.18*4+0.12)      # P4 는 코어70 -> 위성 TQQQ 0.18 / SGOV 0.12
    assert abs(sat/V-exp) < 1e-9, f'P4 위성 {sat/V:.6f}, 기대 {exp:.6f} (상한 적용된 듯)'
    assert sat/V > 0.30, 'P4 에서 위성이 30% 이하로 억제됨'
    assert abs(h['TQQQ']/sat-0.60) < 1e-9, '위성 내부 60/40 미복원'


# ---------------------------------------------------------------- 16~18
def _stock_setup(day_offset, tqqq_mv, key):
    """TQQQ 를 특정일에 움직인 합성 데이터로 /stock V4 실행."""
    idx = CAL[(CAL >= '2020-01-02') & (CAL <= '2021-06-30')]
    d = pd.DataFrame(0.0, index=idx, columns=['TQQQ', 'SGOV'])
    d.iloc[day_offset, d.columns.get_loc('TQQQ')] = tqqq_mv
    r = E.Backtest(E.SPECS[key], d, FX, idx[0], idx[-1], liquidate_at_end=False).run()
    return r, idx


def t16_stock_band_boundary():
    """/stock V4 밴드: TQQQ 비중 70% 이내 무거래, 초과 시 40% 완전 복원."""
    idx0 = CAL[(CAL >= '2020-01-02') & (CAL <= '2021-06-30')]
    iS = list(idx0).index(E.dec_last_trading_days(idx0)[0])
    # w = 0.4(1+m) / (0.4(1+m)+0.6);  상단 0.70 도달 m* = 2.50
    lo, _ = _stock_setup(iS-1, 2.45, 'S1')      # w = 0.6970 -> 밴드 내
    hi, idx = _stock_setup(iS-1, 2.55, 'S1')    # w = 0.7030 -> 이탈
    assert len(lo['trades']) == 0, f'밴드 내부인데 거래 {len(lo["trades"])}회'
    assert len(hi['trades']) == 1, f'밴드 이탈인데 거래 {len(hi["trades"])}회'
    h = hi['holdings'].loc[idx[iS+1]]
    w = h['TQQQ']/h.sum()
    assert abs(w-0.40) < 1e-9, f'목표 40% 완전복원 아님: {w:.6f}'   # 부분복원 금지
    # 하단 0.10 도달 m* = -0.8333
    lo2, _ = _stock_setup(iS-1, -0.82, 'S1')    # w = 0.1071 -> 밴드 내
    hi2, idx2 = _stock_setup(iS-1, -0.85, 'S1')  # w = 0.0909 -> 이탈
    assert len(lo2['trades']) == 0, '하단 밴드 내부인데 거래 발생'
    assert len(hi2['trades']) == 1, '하단 밴드 이탈인데 거래 없음'
    h2 = hi2['holdings'].loc[idx2[iS+1]]
    assert abs(h2['TQQQ']/h2.sum()-0.40) < 1e-9, '하단 이탈 후 40% 미복원'


def t17_stock_daily_vs_annual():
    """S2(상시점검)는 이탈 다음 거래일 체결, S1(연말점검)은 다음 1월까지 대기."""
    idx0 = CAL[(CAL >= '2020-01-02') & (CAL <= '2021-06-30')]
    off = 100                                   # 2020년 중반, 12월이 아님
    a1, idx = _stock_setup(off, 2.55, 'S1')
    a2, _ = _stock_setup(off, 2.55, 'S2')
    assert len(a2['trades']) >= 1, 'S2 가 상시 밴드 이탈에 반응하지 않음'
    assert a2['trades']['date'].iloc[0] == idx[off+1],         f'S2 체결일 {a2["trades"]["date"].iloc[0].date()}, 기대 {idx[off+1].date()}'
    assert len(a1['trades']) == 1, 'S1 체결 횟수 이상'
    assert a1['trades']['date'].iloc[0].year == 2021,         f'S1 이 연말 전에 체결함: {a1["trades"]["date"].iloc[0].date()}'


def t18_stock_initial_weights():
    """/stock V4 초기 비중은 TQQQ 40 / SGOV 60."""
    for key in ['S1', 'S2']:
        w = E.initial_weights(E.SPECS[key])
        assert abs(w['TQQQ']-0.40) < 1e-12 and abs(w['SGOV']-0.60) < 1e-12, w
        r, idx = _stock_setup(50, 0.0, key)
        h = r['holdings'].iloc[0]
        assert abs(h['TQQQ']/h.sum()-0.40) < 1e-12


# ---------------------------------------------------------------- 19~21
PA = ['GLD', 'SPTL', 'SCHD', 'TQQQ']


def _pa_setup(key, tqqq_mv=0.0, risk_mv=0.0):
    """신호일 직전에 TQQQ 또는 위험자산 전체를 움직인 합성 데이터."""
    idx = CAL[(CAL >= '2020-01-02') & (CAL <= '2021-06-30')]
    S = E.dec_last_trading_days(idx)[0]
    iS = list(idx).index(S)
    d = pd.DataFrame(0.0, index=idx, columns=PA)
    if tqqq_mv:
        d.iloc[iS-1, d.columns.get_loc('TQQQ')] = tqqq_mv
    if risk_mv:
        for a in ('SCHD', 'TQQQ'):
            d.iloc[iS-1, d.columns.get_loc(a)] = risk_mv
    r = E.Backtest(E.SPECS[key], d, FX, idx[0], idx[-1], liquidate_at_end=False).run()
    return r, idx[iS+1]


def t19_pa_initial_weights():
    """포트폴리오 A 초기 비중 = GLD 56 / SPTL 4 / SCHD 26 / TQQQ 14."""
    exp = {'GLD': .56, 'SPTL': .04, 'SCHD': .26, 'TQQQ': .14}
    for key in ['A1', 'A2', 'A3']:
        w = E.initial_weights(E.SPECS[key])
        for a, e in exp.items():
            assert abs(w[a]-e) < 1e-12, f'{key} {a}: {w[a]} vs {e}'
        r, _ = _pa_setup(key)
        h = r['holdings'].iloc[0]
        for a, e in exp.items():
            assert abs(h[a]/h.sum()-e) < 1e-12


def t20_pa_band_boundary():
    """위험자산(SCHD+TQQQ) 비중 28~52% 밖일 때만 목표비중으로 복원."""
    # 상단: risk = (26+14(1+m)) / (86+14(1+m)) = 0.52  ->  m* = 1.7857
    lo, _ = _pa_setup('A1', tqqq_mv=1.77)      # risk 0.5191 -> 밴드 내
    hi, Ed = _pa_setup('A1', tqqq_mv=1.80)     # risk 0.5208 -> 이탈
    assert len(lo['trades']) == 0, f'밴드 내부인데 거래 {len(lo["trades"])}회'
    assert len(hi['trades']) == 1, f'밴드 이탈인데 거래 {len(hi["trades"])}회'
    h = hi['holdings'].loc[Ed]; w = h/h.sum()
    for a, e in {'GLD': .56, 'SPTL': .04, 'SCHD': .26, 'TQQQ': .14}.items():
        assert abs(w[a]-e) < 1e-9, f'복원 후 {a} 비중 {w[a]:.6f}, 기대 {e}'
    # 하단: risk 40(1+m)/(60+40(1+m)) = 0.28  ->  m* = -0.41667
    lo2, _ = _pa_setup('A1', risk_mv=-0.41)    # risk 0.2823 -> 밴드 내
    hi2, Ed2 = _pa_setup('A1', risk_mv=-0.43)  # risk 0.2754 -> 이탈
    assert len(lo2['trades']) == 0, '하단 밴드 내부인데 거래 발생'
    assert len(hi2['trades']) == 1, '하단 밴드 이탈인데 거래 없음'
    h2 = hi2['holdings'].loc[Ed2]; w2 = h2/h2.sum()
    assert abs(w2['GLD']-0.56) < 1e-9 and abs(w2['TQQQ']-0.14) < 1e-9


def t21_pa_top_only_variant():
    """A3: 밴드 이탈 시 위험:안전 40:60 만 맞추고 그룹 내부는 드리프트 유지."""
    r1, Ed = _pa_setup('A1', tqqq_mv=1.80)
    r3, _ = _pa_setup('A3', tqqq_mv=1.80)
    h1 = r1['holdings'].loc[Ed]; h3 = r3['holdings'].loc[Ed]
    for h in (h1, h3):
        risk = h[['SCHD', 'TQQQ']].sum()/h.sum()
        assert abs(risk-0.40) < 1e-9, f'위험자산 비중 {risk:.6f}'
    assert abs(h1['TQQQ']/h1[['SCHD', 'TQQQ']].sum()-0.35) < 1e-9, 'A1 내부 미복원'
    # A3 내부는 26 : 14*2.8 = 26 : 39.2 -> TQQQ 0.60123
    exp = (14*2.80)/(26+14*2.80)
    got = h3['TQQQ']/h3[['SCHD', 'TQQQ']].sum()
    assert abs(got-exp) < 1e-9, f'A3 내부 비중 {got:.6f}, 기대 {exp:.6f}'
    # 안전자산 내부도 무변동이면 56:4 그대로
    assert abs(h3['GLD']/h3[['GLD', 'SPTL']].sum()-56/60) < 1e-9


if __name__ == '__main__':
    print('=== 2단계 시점 정렬 / 미래참조 차단 단위테스트 ===')
    for n, f in [
        ('T1  체결일 = 다음 거래일', t1_exec_is_next_trading_day),
        ('T2  판단은 신호일 종가만 사용', t2_decision_uses_signal_day_only),
        ('T3  체결금액은 체결일 종가 기준', t3_execution_at_exec_day_close),
        ('T4  시작일 이전 데이터 미사용', t4_no_pre_start_data),
        ('T5  종료일 이후 데이터 미사용', t5_no_post_end_data),
        ('T6  종료일 신호는 미체결', t6_no_signal_execution_on_last_day),
        ('T7  5/25 밴드 경계 동작', t7_band_boundary),
        ('T8  무드리프트 시 무매매·무비용', t8_no_drift_no_cost),
        ('T9  단순보유 청산·세금 손계산 대조', t9_buyhold_and_tax_hand_check),
        ('T10 이동평균 취득원가', t10_moving_average_basis),
        ('T11 손실 이월 없음·기본공제', t11_loss_no_carryforward),
        ('T12 TQQQ=0 시나리오', t12_tqqq_zero_variant),
        ('T13 P1 최상위복원·코어내부유지', t13_p1_restore_keeps_core_internal),
        ('T14 P2 30% 상한 초과분만 이전', t14_p2_cap_moves_only_excess),
        ('T15 P4 상한 없음', t15_p4_has_no_cap),
        ('T16 /stock V4 밴드 경계·완전복원', t16_stock_band_boundary),
        ('T17 /stock 상시점검 vs 연말점검', t17_stock_daily_vs_annual),
        ('T18 /stock 초기비중 40:60', t18_stock_initial_weights),
        ('T19 포트A 초기비중 56/4/26/14', t19_pa_initial_weights),
        ('T20 포트A 위험자산 ±12%p 밴드', t20_pa_band_boundary),
        ('T21 포트A 최상위만 복원 변형', t21_pa_top_only_variant),
    ]:
        check(n, f)
    n_fail = sum(1 for _, s, _ in RESULTS if s == 'FAIL')
    pd.DataFrame(RESULTS, columns=['test', 'status', 'detail']).to_csv(
        ROOT/'out'/'s2_unittests.csv', index=False, encoding='utf-8-sig')
    print(f'\n{len(RESULTS)-n_fail}/{len(RESULTS)} PASS')
    sys.exit(1 if n_fail else 0)
