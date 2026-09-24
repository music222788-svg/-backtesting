"""S5 검증 2: 금 수익률 정상화 민감도.

GLD(프록시 포함, 원천징수 적용 수익률) 일간 수익률에 상수 드리프트 δ 를 더해
2000-01-03 ~ 2026-09-16 연율 수익률이 목표값이 되게 한다. 변동성·상관은 유지(상수 이동).
  G0 실제 / G1 미국 CPI 연율 + 1%p / G2 미국 CPI 연율 / G3 실제 CAGR 의 절반
미국 CPI: FRED CPIAUCSL (s5/data_ext, Drive 투자/data 사본, 2026-09-10 수정본, 최종 관측 2026-07)
"""
import sys, pathlib, json, time
from multiprocessing import Pool
import numpy as np, pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s5_lib as S
import s5_dca as DCA
from dca_engine import calendar_struct
L = S.L
M = S.M
OUT = S.S5
ORDER = DCA.ORDER


def drifts():
    cpi = pd.read_csv(OUT/'data_ext'/'FRED_CPIAUCSL.csv')
    cpi.columns = ['date', 'v']
    cpi['date'] = pd.to_datetime(cpi['date'])
    cpi = cpi.dropna().set_index('date')['v']
    c0, c1 = cpi.loc['2000-01-01'], cpi.index[-1]
    months = (c1.year-2000)*12 + (c1.month-1)
    us_cpi = (cpi.iloc[-1]/c0)**(12.0/months) - 1.0
    R, _ = S.frames()
    g = R['GLD'].loc[S.T0:S.TEND].fillna(0.0).to_numpy()[1:]
    actual = np.exp(np.log1p(g).sum()*252.0/len(g)) - 1.0
    tg = {'G0': actual, 'G1': us_cpi+0.01, 'G2': us_cpi, 'G3': actual/2}
    d = {k: (0.0 if k == 'G0' else S.gold_drift_for(v, R)) for k, v in tg.items()}
    # 구간별 영향
    per = {'2000–2012': ('2000-01-04', '2012-12-31'), '2013–2019': ('2013-01-02', '2019-12-31'),
           '2020–2026': ('2020-01-02', '2026-09-16'), '2000–2026': ('2000-01-04', '2026-09-16')}
    tab = {}
    for k, dl in d.items():
        x = R['GLD'].fillna(0.0) + dl
        tab[k] = {p: float(np.exp(np.log1p(x.loc[s:e]).sum()*252.0/len(x.loc[s:e]))-1.0) for p, (s, e) in per.items()}
    info = dict(us_cpi_series='FRED CPIAUCSL (월간, 계절조정)', cpi_start='2000-01', cpi_end=str(c1.date())[:7],
                cpi_start_value=float(c0), cpi_end_value=float(cpi.iloc[-1]), months=int(months),
                cpi_missing_note='원자료 2025-10 값 공란(제외)',
                us_cpi_annual=us_cpi, gld_actual_cagr=actual,
                target_cagr=tg, daily_drift=d, annual_drift_approx={k: v*252 for k, v in d.items()},
                gld_cagr_by_period=tab)
    json.dump(info, open(OUT/'s5_gold_drift.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    return d, info


def lump(d):
    rows = []
    for g, dl in d.items():
        for k in ORDER:
            sp, lev = S.PORTS[k]
            R, _ = S.frames(gold_drift=dl)
            for per in ['A', 'B', 'C']:
                m, _, _ = L.evaluate(sp, R, per)
                rows.append(dict(G=g, key=k, 구간=per, CAGR_세후=m['CAGR_세후'], Sharpe_세후=m['Sharpe_세후'],
                                 Sharpe=m['Sharpe'], MDD=m['MDD'], 최장회복_거래일=m['최장회복_거래일'],
                                 최종자산_세후=m['최종자산_세후']))
    df = pd.DataFrame(rows)
    for (g, p), x in df.groupby(['G', '구간']):
        df.loc[x.index, '순위_Sharpe세후'] = x['Sharpe_세후'].rank(ascending=False).astype(int)
        df.loc[x.index, '순위_CAGR세후'] = x['CAGR_세후'].rank(ascending=False).astype(int)
        df.loc[x.index, '순위_MDD'] = x['MDD'].rank(ascending=False).astype(int)
    df.to_csv(OUT/'s5_gold_lump.csv', index=False, encoding='utf-8-sig')
    return df


def _dca_job(args):
    g, dl, k = args
    R, D = S.frames(gold_drift=dl)
    rows = []
    for per, (s, e) in DCA.FIXED.items():
        m, _ = DCA.run_window(k, S.window(s, e), 'M1', 'krw', R, D)
        m.update(G=g, key=k, 구간=per); rows.append(m)
    for s, e in DCA.starts_for(19):
        m, _ = DCA.run_window(k, S.window(s, e), 'M1', 'krw', R, D)
        m.update(G=g, key=k, 구간=f'롤링19년 {s.date()}'); rows.append(m)
    return rows


def dca(d):
    jobs = [(g, dl, k) for g, dl in d.items() for k in ORDER if (g == 'G0' or k in S.GOLD_PORTS)]
    with Pool(4) as p:
        res = p.map(_dca_job, jobs)
    df = pd.DataFrame([r for rs in res for r in rs])
    # 금 비보유 포트폴리오는 G 무관 -> G0 값 복사
    add = []
    for g in d:
        if g == 'G0':
            continue
        x = df[(df.G == 'G0') & ~df.key.isin(S.GOLD_PORTS)].copy(); x['G'] = g; add.append(x)
    df = pd.concat([df] + add, ignore_index=True)
    bjobs = [(k, 19, dl, g) for g, dl in d.items() for k in ORDER if (g != 'G0' and k in S.GOLD_PORTS)]
    with Pool(4) as p:
        bres = p.map(DCA._boot_job, bjobs)
    bd = pd.DataFrame([r for rs in bres for r in rs])
    b0 = pd.read_csv(OUT/'cache'/'boot_paths_G0.csv')
    b0 = b0[b0.보유년 == 19]
    allb = [b0]
    for g in d:
        if g == 'G0':
            continue
        x = b0[~b0.key.isin(S.GOLD_PORTS)].copy(); x['G'] = g; allb.append(x)
    allb.append(bd)
    bs = DCA.boot_summary(pd.concat(allb, ignore_index=True))
    fixed = df[~df.구간.str.startswith('롤링')]
    roll = df[df.구간.str.startswith('롤링')].groupby(['G', 'key']).agg(
        롤링19_중앙=('최종자산_명목', 'median'), 롤링19_하위10=('최종자산_명목', lambda x: x.quantile(.1)),
        롤링19_최소=('최종자산_명목', 'min'), 롤링19_XIRR중앙=('XIRR_명목', 'median'),
        롤링19_종료시25억이상비율=('최종자산_명목', lambda x: (x >= S.GOAL).mean()*100)).reset_index()
    out = []
    for _, r in fixed.iterrows():
        out.append(dict(G=r.G, key=r.key, 구분='고정구간 '+r.구간, 최종자산_명목=r['최종자산_명목'],
                        XIRR_명목=r['XIRR_명목'], 원금하회_최장거래일=r['원금하회_최장거래일']))
    for _, r in roll.iterrows():
        out.append(dict(G=r.G, key=r.key, 구분='롤링19년(93개)', 최종자산_명목=r['롤링19_중앙'],
                        하위10=r['롤링19_하위10'], 최소=r['롤링19_최소'], XIRR_명목=r['롤링19_XIRR중앙'],
                        P_25억이상=r['롤링19_종료시25억이상비율']))
    for _, r in bs.iterrows():
        out.append(dict(G=r.G, key=r.key, 구분='부트스트랩19년(1000회)', 최종자산_명목=r['중앙값'],
                        하위10=r['하위10'], P_25억이상=r['P_25억이상'], P_15억이상=r['P_15억이상']))
    res = pd.DataFrame(out)
    for (g, c), x in res.groupby(['G', '구분']):
        res.loc[x.index, '순위_최종자산'] = x['최종자산_명목'].rank(ascending=False).astype(int)
    res.to_csv(OUT/'s5_gold_dca.csv', index=False, encoding='utf-8-sig')
    return res


if __name__ == '__main__':
    t0 = time.time()
    d, info = drifts()
    print(json.dumps({k: info[k] for k in ['us_cpi_annual', 'gld_actual_cagr', 'target_cagr', 'annual_drift_approx', 'gld_cagr_by_period']}, indent=1, ensure_ascii=False))
    lump(d); print('lump', round(time.time()-t0))
    dca(d); print('dca', round(time.time()-t0))
