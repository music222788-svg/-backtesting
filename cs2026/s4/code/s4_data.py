"""S4-1: S4 수익률 데이터 구축.

산출 (s4/rebuilt/):
  s4_returns.csv   자산 일간 수익률. 열 이름 규칙
      <자산>          : 기존 s1_returns.csv 와 동일(원천징수 미적용). SCHD = SCHD_a
      <자산>_wh       : 배당 원천징수 15% 적용(세후 배당 재투자)
      TQQQ_cal, QLD   : 레버리지 접합 시리즈 (QLD 는 기본 합성, *_cal 은 보정 합성)
  s4_syn.csv       순수 합성 시리즈 TQQQ_syn / QLD_syn / TQQQ_syn_cal / QLD_syn_cal
  s4_divyield.csv  자산별 일간 배당(분배) 수익률 추정치
  s4_calib.json    차입 스프레드 보정 결과
  s4_tracking_annual.csv  합성 vs 실제 연간 괴리 (TQQQ, QLD; 기본·보정)

규칙
  - 합성 공식은 기존 s1_build.py 와 동일: L*NDX_TR - (L-1)*(BEY_전일+스프레드)*경과일/365 - 보수*경과일/365
    TQQQ 보수 0.84%(기존), QLD 보수 0.95%(ProShares 팩트시트 2026-06-30 기준 순보수)
  - 보정 스프레드: 실제 상품 존재 구간(TQQQ 2010-02-11~, QLD 동일 창 2010~2026)의
    달력연도 수익률 괴리 제곱합을 최소화하는 연율 스프레드
  - QLD 실데이터(Yahoo, 2026-09-10 다운로드본)는 2026-09-10 까지. 2026-09-11~16 4거래일은 합성으로 채운다
  - 배당 수익률 = (1+조정종가수익률)/(1+종가수익률) - 1 (Yahoo close 는 분할만 조정, adjclose 는 배당까지 조정)
      SPY  : SPY.csv 가 Drive 사본에 없어 ^SP500TR - ^GSPC 일간 수익률 차로 대체 (지수 배당)
      SCHD : 실데이터 구간은 SCHD.csv, 프록시(a=SPY) 구간은 SPY 배당
      IEF  : 실데이터 구간은 IEF.csv, 프록시 구간은 쿠폰 경과이자(전일 8.5년 금리 x 경과일/365)
      SGOV : 실데이터 구간은 SGOV.csv, 프록시 구간은 수익 전액(이자)
      QQQ / TQQQ / QLD : 실데이터 구간은 각 CSV. 합성 레버리지 구간은 분배 0 으로 가정
      GLD  : 분배 없음
  - 원천징수 적용 수익률 = 원수익률 - 0.15 x 배당수익률 (배당락일에 차감)
"""
import json, pathlib
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar

S4 = pathlib.Path(__file__).resolve().parent.parent
ROOT = S4.parent
D, OUT, RB, EXT = ROOT/'data', ROOT/'out', S4/'rebuilt', S4/'data_ext'
WH = 0.15
EXP_TQQQ, EXP_QLD = 0.0084, 0.0095
END = pd.Timestamp('2026-09-16')

R1 = pd.read_csv(OUT/'s1_returns.csv', parse_dates=['date']).set_index('date')
P1 = pd.read_csv(RB/'s1_proxy_returns.csv', parse_dates=['date']).set_index('date')
MAC = pd.read_csv(RB/'s1_macro.csv', parse_dates=['date']).set_index('date')
CAL = R1.index
days = MAC['days']
dt = days/365.0
bey_prev = MAC['bey'].shift(1)
ndx = P1['NDX_TR']


def ycsv(path):
    return pd.read_csv(path, parse_dates=['date']).set_index('date').sort_index()


def divy(df):
    a = df['adjclose'].reindex(CAL).pct_change()
    c = df['close'].reindex(CAL).pct_change()
    return ((1+a)/(1+c)-1).clip(lower=0)


def syn(L, spread, exp):
    return L*ndx - (L-1)*(bey_prev+spread)*dt - exp*dt


def real_ret(path):
    return ycsv(path)['adjclose'].reindex(CAL).pct_change()


def cum(r):
    return float((1+r).prod()-1)


def annual_gap(real, s):
    m = real.notna() & s.notna()
    a, b = real[m], s[m]
    g = pd.DataFrame({'real': a, 'syn': b})
    return g.groupby(g.index.year).apply(lambda x: pd.Series({'n': len(x), 'real': cum(x.real),
                                                                'syn': cum(x.syn)}))


def calibrate(L, exp, real, lo='2010-02-11'):
    rr = real.loc[lo:]

    def obj(sp):
        t = annual_gap(rr, syn(L, sp, exp).loc[lo:])
        return float(((t.syn-t.real)**2).sum())
    res = minimize_scalar(obj, bounds=(-0.05, 0.10), method='bounded', options={'xatol': 1e-7})
    return float(res.x)


def splice(real, s, real_start, like_s1=False):
    """like_s1=True: 기존 s1_build.splice 와 동일(실데이터 첫날 NaN 을 그대로 둠 -> 엔진에서 0).
    False: 실데이터가 NaN 인 날은 합성값 사용."""
    r = s.copy()
    m = (CAL >= real_start)
    if not like_s1:
        m = m & real.notna()
    r[m] = real[m]
    return r


if __name__ == '__main__':
    tq_real = R1['TQQQ'].where(CAL >= pd.Timestamp('2010-02-11'))
    tq_real.loc[pd.Timestamp('2010-02-11')] = np.nan
    qld_df = ycsv(EXT/'QLD.csv')
    qld_real = qld_df['adjclose'].reindex(CAL).pct_change()
    qld_start = qld_df.index[0]
    qld_real.loc[qld_start] = np.nan

    s_tq = calibrate(3, EXP_TQQQ, tq_real)
    s_ql = calibrate(2, EXP_QLD, qld_real)

    SYN = pd.DataFrame({'TQQQ_syn': syn(3, 0.0, EXP_TQQQ), 'QLD_syn': syn(2, 0.0, EXP_QLD),
                        'TQQQ_syn_cal': syn(3, s_tq, EXP_TQQQ), 'QLD_syn_cal': syn(2, s_ql, EXP_QLD)})
    assert (SYN['TQQQ_syn']-P1['TQQQ_syn']).abs().max() < 1e-12
    SYN.to_csv(RB/'s4_syn.csv')

    # 추적 연간표
    rows, summ = [], []
    for name, real, cols, s0 in [('TQQQ', tq_real, ('TQQQ_syn', 'TQQQ_syn_cal'), '2010-02-11'),
                                 ('QLD', qld_real, ('QLD_syn', 'QLD_syn_cal'), str(qld_start.date()))]:
        for ver, c in zip(('기본', '보정'), cols):
            t = annual_gap(real, SYN[c])
            for y, r in t.iterrows():
                rows.append(dict(asset=name, version=ver, year=int(y), n=int(r.n), real=r.real,
                                 syn=r.syn, gap_pp=(r.syn-r.real)*100))
            m = real.notna() & SYN[c].notna()
            a, b = real[m], SYN[c][m]
            n = len(a)
            summ.append(dict(asset=name, version=ver, overlap=f'{a.index[0].date()}~{a.index[-1].date()}',
                             n=n, real_cagr=((1+a).prod()**(252/n)-1)*100,
                             syn_cagr=((1+b).prod()**(252/n)-1)*100,
                             cagr_gap_pp=(((1+b).prod()**(252/n))-((1+a).prod()**(252/n)))*100,
                             corr=float(np.corrcoef(a, b)[0, 1]),
                             te_ann_pct=float((b-a).std()*np.sqrt(252)*100),
                             same_sign_years=None))
    TR = pd.DataFrame(rows)
    TR.to_csv(RB/'s4_tracking_annual.csv', index=False, encoding='utf-8-sig')
    SM = pd.DataFrame(summ)
    for i, r in SM.iterrows():
        g = TR[(TR.asset == r.asset) & (TR.version == r.version)]
        SM.loc[i, 'same_sign_years'] = f'{int((g.gap_pp > 0).sum())}/{len(g)} 과대'
    SM.to_csv(RB/'s4_tracking_summary.csv', index=False, encoding='utf-8-sig')

    # 접합 시리즈
    qld_real_full = qld_real.where(CAL <= pd.Timestamp('2026-09-10'))
    RET = pd.DataFrame(index=CAL)
    for a in ['SPY', 'QQQ', 'IEF', 'GLD', 'SGOV']:
        RET[a] = R1[a]
    RET['SCHD'] = R1['SCHD_a']
    RET['TQQQ'] = R1['TQQQ']
    RET['TQQQ_cal'] = splice(R1['TQQQ'], SYN['TQQQ_syn_cal'], pd.Timestamp('2010-02-11'), like_s1=True)
    RET['QLD'] = splice(qld_real_full, SYN['QLD_syn'], qld_start)
    RET['QLD_cal'] = splice(qld_real_full, SYN['QLD_syn_cal'], qld_start)
    # 기본 TQQQ 접합이 기존과 동일한지 확인
    assert np.allclose(splice(R1['TQQQ'], SYN['TQQQ_syn'], pd.Timestamp('2010-02-11'), like_s1=True).loc['2000':],
                       R1['TQQQ'].loc['2000':], equal_nan=True)

    # 배당 수익률
    trx = ycsv(EXT/'SP500TR.csv') if 'date' in open(EXT/'SP500TR.csv').readline() else None
    if trx is None:
        t = pd.read_csv(EXT/'SP500TR.csv')
        t.columns = ['date', 'close']
        t['date'] = pd.to_datetime(t['date'])
        trx = t.set_index('date')['close']
    gspc = ycsv(D/'GSPC.csv')['close']
    spy_div = (trx.reindex(CAL).pct_change() - gspc.reindex(CAL).pct_change())
    spy_div = spy_div.fillna(0.0)
    real_start = {'SCHD': pd.Timestamp('2011-10-20'), 'IEF': pd.Timestamp('2002-07-30'),
                  'SGOV': pd.Timestamp('2020-06-01'), 'TQQQ': pd.Timestamp('2010-02-11')}
    DV = pd.DataFrame(index=CAL)
    DV['SPY'] = spy_div
    DV['QQQ'] = divy(ycsv(D/'QQQ.csv')).fillna(0.0)
    d_schd = divy(ycsv(D/'SCHD.csv')).fillna(0.0)
    DV['SCHD'] = np.where(CAL >= real_start['SCHD'], d_schd, spy_div)
    d_ief = divy(ycsv(D/'IEF.csv')).fillna(0.0)
    cpn = (MAC['y85'].shift(1)*dt).fillna(0.0)
    DV['IEF'] = np.where(CAL >= real_start['IEF'], d_ief, cpn)
    DV['GLD'] = 0.0
    d_sg = divy(ycsv(D/'SGOV.csv')).fillna(0.0)
    DV['SGOV'] = np.where(CAL >= real_start['SGOV'], d_sg, P1['SGOV_syn'].fillna(0.0).clip(lower=0))
    d_tq = divy(ycsv(EXT/'TQQQ.csv')).fillna(0.0)
    DV['TQQQ'] = np.where(CAL >= real_start['TQQQ'], d_tq, 0.0)
    DV['TQQQ_cal'] = DV['TQQQ']
    d_ql = divy(qld_df).fillna(0.0)
    DV['QLD'] = np.where(CAL >= qld_start, d_ql, 0.0)
    DV['QLD_cal'] = DV['QLD']
    DV.to_csv(RB/'s4_divyield.csv')

    for c in list(RET.columns):
        RET[c+'_wh'] = RET[c] - WH*DV[c]
    RET.to_csv(RB/'s4_returns.csv')

    json.dump(dict(spread_TQQQ=s_tq, spread_QLD=s_ql, window='2010-02-11 ~ 실데이터 최종일, 달력연도 괴리 제곱합 최소',
                   exp_TQQQ=EXP_TQQQ, exp_QLD=EXP_QLD, QLD_real_start=str(qld_start.date()),
                   QLD_real_end='2026-09-10', QLD_fill='2026-09-11~2026-09-16 합성(기본/보정 각각)',
                   withholding=WH),
              open(RB/'s4_calib.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)

    pd.set_option('display.width', 200)
    print('spread TQQQ %.4f%%  QLD %.4f%%' % (s_tq*100, s_ql*100))
    print(SM.round(3).to_string(index=False))
    y = DV.loc['2000':'2026'].groupby(DV.loc['2000':'2026'].index.year).sum()
    print('\n연간 배당수익률 합(%) 일부'); print((y*100).round(2).loc[[2000, 2005, 2010, 2015, 2020, 2025]].to_string())
