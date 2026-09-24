"""3단계-b: 세부구간, 코로나 폭락·회복, 시작일 의존성."""
import sys, pathlib
import numpy as np, pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent
OUT = ROOT/'out'
import engine as E
from engine_fast import run_fast
import metrics as M

R = pd.read_csv(OUT/'s1_returns.csv', parse_dates=['date']).set_index('date')
MAC = pd.read_csv(OUT/'s1_macro.csv', parse_dates=['date']).set_index('date')
FX = MAC['krw']
CAL = R.index
KEYS = ['P1', 'P2', 'P3', 'P4', 'B1', 'B2', 'C1']

EQ = {p: pd.read_csv(OUT/f's3_equity_{p}.csv', parse_dates=['date']).set_index('date')
      for p in ['A', 'B', 'C']}

SUB_A = [('2000-2002 닷컴붕괴', '2000-01-03', '2002-12-31'),
         ('2003-2007 회복장', '2003-01-02', '2007-12-31'),
         ('2008-2009 금융위기', '2008-01-02', '2009-12-31'),
         ('2010-2019 강세장', '2010-01-04', '2019-12-31')]
SUB_B = [('2020 코로나폭락', '2020-02-19', '2020-03-23'),
         ('2020-04~2021 회복장', '2020-04-01', '2021-12-31'),
         ('2022 약세장', '2022-01-03', '2022-12-30'),
         ('2023~최근', '2023-01-03', '2026-09-16')]


def sub_table(per, subs):
    rows = []
    eq = EQ[per]
    for label, s, e in subs:
        for k in KEYS:
            x = eq[k].loc[s:e]
            if len(x) < 2:
                continue
            yrs = (len(x)-1)/252.0
            rows.append(dict(구간=per, 세부=label, key=k, 시작=s, 종료=e, 거래일=len(x),
                             누적수익=float(x.iloc[-1]/x.iloc[0]-1.0),
                             CAGR=(float((x.iloc[-1]/x.iloc[0])**(1/yrs)-1.0) if yrs >= 0.5 else np.nan),
                             구간내MDD=M.mdd(x)))
    return pd.DataFrame(rows)


def covid_table():
    """코로나 폭락 낙폭과 전고점 회복 소요 거래일. 구간 B 자산곡선 기준."""
    eq = EQ['B']
    peak_d = pd.Timestamp('2020-02-19')
    rows = []
    for k in KEYS:
        x = eq[k]
        pk = float(x.loc[peak_d])
        seg = x.loc[peak_d:'2020-03-23']
        tr_d = seg.idxmin()
        tr = float(seg.min())
        rec_d, days = M.recovery_days(x, peak_d)
        rows.append(dict(key=k, 전고점일=str(peak_d.date()), 저점일=str(tr_d.date()),
                         낙폭=tr/pk-1.0,
                         폭락거래일=int(x.index.get_loc(tr_d)-x.index.get_loc(peak_d)),
                         회복일=str(rec_d.date()) if rec_d is not None else '미회복',
                         회복소요_거래일=days,
                         회복소요_달력일=((rec_d-peak_d).days if rec_d is not None else None)))
    return pd.DataFrame(rows)


def month_starts(lo, hi):
    s = pd.Series(CAL, index=CAL)
    s = s[(s >= pd.Timestamp(lo)) & (s <= pd.Timestamp(hi))]
    return list(s.groupby([s.dt.year, s.dt.month]).min())


def hold_scan(starts, years, hard_end):
    """각 시작일에서 years 년 보유한 결과."""
    rows = []
    hard_end = pd.Timestamp(hard_end)
    for s in starts:
        tgt = s + pd.DateOffset(years=years)
        cand = CAL[(CAL <= min(tgt, hard_end))]
        e = cand[-1]
        if e <= s:
            continue
        actual_yrs = (CAL.get_loc(e)-CAL.get_loc(s))/252.0
        if actual_yrs < years*0.98:
            continue
        for k in KEYS:
            eq = run_fast(k, R, FX, s, e, keep_holdings=False)['equity']
            rows.append(dict(start=str(s.date()), end=str(e.date()), years=years, key=k,
                             total=float(eq.iloc[-1]/eq.iloc[0]-1.0),
                             cagr=M.cagr(eq), mdd=M.mdd(eq)))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    pd.set_option('display.width', 250)

    sa = sub_table('A', SUB_A)
    sb = sub_table('B', SUB_B)
    pd.concat([sa, sb]).to_csv(OUT/'s3_subperiods.csv', index=False, encoding='utf-8-sig')
    for nm, d in [('구간 A 세부', sa), ('구간 B 세부', sb)]:
        print(f'\n===== {nm} =====')
        p = d.pivot(index='key', columns='세부', values='누적수익')*100
        m = d.pivot(index='key', columns='세부', values='구간내MDD')*100
        print('[누적수익 %]'); print(p.round(1).to_string())
        print('[구간내 MDD %]'); print(m.round(1).to_string())

    cv = covid_table()
    cv.to_csv(OUT/'s3_covid.csv', index=False, encoding='utf-8-sig')
    print('\n===== 코로나 폭락(2020-02-19 ~ 2020-03-23) =====')
    c = cv.copy(); c['낙폭'] = (c['낙폭']*100).round(2)
    print(c.to_string(index=False))

    # ---------- 시작일 의존성 ----------
    print('\n===== 시작일 의존성 스캔 =====', flush=True)
    s5 = hold_scan(month_starts('2000-01-01', '2014-12-31'), 5, '2019-12-31')
    print(f'  5년 보유: 시작월 {s5["start"].nunique()}개', flush=True)
    s10 = hold_scan(month_starts('2000-01-01', '2009-12-31'), 10, '2019-12-31')
    print(f'  10년 보유: 시작월 {s10["start"].nunique()}개', flush=True)
    sB = hold_scan(month_starts('2020-01-01', '2020-12-31'), 0, '2026-09-16')
    pd.concat([s5, s10]).to_csv(OUT/'s3_startdep_A.csv', index=False, encoding='utf-8-sig')

    for nm, d in [('구간 A · 5년 보유', s5), ('구간 A · 10년 보유', s10)]:
        g = d.groupby('key')['cagr']
        t = pd.DataFrame({'중앙값': g.median()*100, '하위10%': g.quantile(.10)*100,
                          '최악': g.min()*100, '최고': g.max()*100,
                          '음수비율%': d.assign(neg=d.cagr < 0).groupby('key')['neg'].mean()*100})
        print(f'\n[{nm}] CAGR 분포 (%)'); print(t.round(2).to_string())

    # 구간 B: 2020년 1~12월 시작
    rows = []
    for s in month_starts('2020-01-01', '2020-12-31'):
        for k in KEYS:
            eq = run_fast(k, R, FX, s, '2026-09-16', keep_holdings=False)['equity']
            rows.append(dict(start=str(s.date()), key=k, cagr=M.cagr(eq), mdd=M.mdd(eq),
                             final=float(eq.iloc[-1])))
    bdep = pd.DataFrame(rows)
    bdep.to_csv(OUT/'s3_startdep_B.csv', index=False, encoding='utf-8-sig')
    piv = bdep.pivot(index='start', columns='key', values='cagr')*100
    print('\n===== 구간 B 시작월별 CAGR (%) =====')
    print(piv.round(2).to_string())
    rk = piv.rank(axis=1, ascending=False).astype(int)
    print('\n[시작월별 수익률 순위]'); print(rk.to_string())
    print('\n순위가 시작월에 따라 바뀌는 전략:',
          {k: sorted(rk[k].unique().tolist()) for k in rk.columns if rk[k].nunique() > 1})
    print('\nsaved -> out/s3_subperiods.csv, s3_covid.csv, s3_startdep_A.csv, s3_startdep_B.csv')
