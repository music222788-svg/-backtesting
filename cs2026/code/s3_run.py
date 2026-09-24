"""3단계-a: 구간 A/B/C 핵심 지표 + 연간 수익률 + 세부구간."""
import sys, pathlib, json
import numpy as np, pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent
OUT = ROOT/'out'
import engine as E
import metrics as M

R = pd.read_csv(OUT/'s1_returns.csv', parse_dates=['date']).set_index('date')
P = pd.read_csv(OUT/'s1_proxy_returns.csv', parse_dates=['date']).set_index('date')
MAC = pd.read_csv(OUT/'s1_macro.csv', parse_dates=['date']).set_index('date')
FX = MAC['krw']
RF = P['SGOV_syn']                      # 무위험수익률 = 3M 국채 BEY 일할

PERIODS = {'A': ('2000-01-03', '2019-12-31'),
           'B': ('2020-01-02', '2026-09-16'),
           'C': ('2000-01-03', '2026-09-16')}
KEYS = ['P1', 'P2', 'P3', 'P4', 'B1', 'B2', 'C1']
TQQQ_ZERO = '2001-08-30'                # 1단계에서 산출한 합성 3배 소멸일


def run_set(period, schd='SCHD_a', tqqq_zero=False):
    s, e = PERIODS[period]
    z = TQQQ_ZERO if tqqq_zero else None
    if z is not None and pd.Timestamp(z) < pd.Timestamp(s):
        z = s
    res = {}
    for k in KEYS:
        pre = E.run_one(k, R, FX, s, e, schd=schd, tax=False, tqqq_zero_date=z)
        post = E.run_one(k, R, FX, s, e, schd=schd, tax=True, tqqq_zero_date=z)
        res[k] = dict(pre=pre, post=post)
    return res


def table(res, period):
    rows = []
    for k in KEYS:
        r = res[k]
        rows.append(M.summarize(f'{k} {E.SPECS[k].name.split(" ",1)[1]}',
                                r['pre']['equity'], r['post']['equity'], FX, RF,
                                r['pre']['trades'], r['post']['taxes']))
        rows[-1]['구간'] = period
        rows[-1]['key'] = k
    return pd.DataFrame(rows)


if __name__ == '__main__':
    all_tabs, all_ann, all_eq = [], [], {}
    for per in ['A', 'B', 'C']:
        res = run_set(per)
        t = table(res, per); all_tabs.append(t)
        for k in KEYS:
            eq = res[k]['pre']['equity']
            all_eq[(per, k)] = eq
            a = M.annual_returns(eq); a.name = k
            a = a.to_frame(); a['구간'] = per; a['key'] = k
            a = a.rename(columns={k: 'ret'}).reset_index().rename(columns={'index': 'year'})
            all_ann.append(a)
        # 자산곡선 저장
        pd.DataFrame({k: res[k]['pre']['equity'] for k in KEYS}).to_csv(OUT/f's3_equity_{per}.csv')
        pd.DataFrame({k: res[k]['post']['equity'] for k in KEYS}).to_csv(OUT/f's3_equity_{per}_aftertax.csv')
        # 연간 세금·매매
        tx = []
        for k in KEYS:
            d = res[k]['post']['taxes']
            if len(d):
                d = d.copy(); d['key'] = k; d['구간'] = per; tx.append(d)
            tr = res[k]['pre']['trades']
            if len(tr):
                tr = tr.copy(); tr['key'] = k; tr['구간'] = per
        if tx:
            pd.concat(tx).to_csv(OUT/f's3_tax_{per}.csv', index=False)
        pd.concat([res[k]['pre']['trades'].assign(key=k) for k in KEYS if len(res[k]['pre']['trades'])]
                  ).to_csv(OUT/f's3_trades_{per}.csv', index=False)

    TAB = pd.concat(all_tabs, ignore_index=True)
    TAB.to_csv(OUT/'s3_summary.csv', index=False, encoding='utf-8-sig')
    ANN = pd.concat(all_ann, ignore_index=True)
    ANN.to_csv(OUT/'s3_annual_returns.csv', index=False, encoding='utf-8-sig')

    # ---------- TQQQ=0 변형 (구간 A, C) ----------
    zt = []
    for per in ['A', 'C']:
        res = run_set(per, tqqq_zero=True)
        t = table(res, per+'-TQQQ0'); zt.append(t)
        pd.DataFrame({k: res[k]['pre']['equity'] for k in KEYS}).to_csv(OUT/f's3_equity_{per}_tqqq0.csv')
    pd.concat(zt, ignore_index=True).to_csv(OUT/'s3_summary_tqqq0.csv', index=False, encoding='utf-8-sig')

    # ---------- SCHD 프록시 민감도 (구간 A, B, C) ----------
    st = []
    for schd in ['SCHD_a', 'SCHD_b', 'SCHD_c']:
        for per in ['A', 'B', 'C']:
            res = run_set(per, schd=schd)
            t = table(res, per); t['schd'] = schd; st.append(t)
    pd.concat(st, ignore_index=True).to_csv(OUT/'s3_summary_schd.csv', index=False, encoding='utf-8-sig')

    pd.set_option('display.width', 250)
    for per in ['A', 'B', 'C']:
        d = TAB[TAB['구간'] == per]
        print(f'\n===== 구간 {per} ({PERIODS[per][0]} ~ {PERIODS[per][1]}) =====')
        v = d[['key', '최종자산_세전', '최종자산_세후', 'CAGR_세전', 'CAGR_세후', 'CAGR_원화',
               'MDD', '최장회복_거래일', '회복완료', '연변동성', 'Sharpe', 'Sortino', 'Calmar',
               '최악의연도', '최악연도', '최악롤링3년', '최악롤링5년', '연평균매매', '연평균세금']].copy()
        for c in ['CAGR_세전', 'CAGR_세후', 'CAGR_원화', 'MDD', '연변동성', '최악의연도',
                  '최악롤링3년', '최악롤링5년']:
            v[c] = (v[c]*100).round(2)
        for c in ['최종자산_세전', '최종자산_세후', '연평균세금']:
            v[c] = v[c].round(0)
        for c in ['Sharpe', 'Sortino', 'Calmar', '연평균매매']:
            v[c] = v[c].round(2)
        print(v.to_string(index=False))
    print('\nsaved -> out/s3_summary.csv, s3_annual_returns.csv, s3_summary_tqqq0.csv, s3_summary_schd.csv')
