"""3단계-e: 순위표 (수익률 / 위험조정 / 최대낙폭) 및 SCHD 프록시 민감도."""
import pathlib
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT/'out'
KEYS = ['P1', 'P2', 'P3', 'P4', 'B1', 'B2', 'C1']

S = pd.read_csv(OUT/'s3_summary.csv')
Z = pd.read_csv(OUT/'s3_summary_tqqq0.csv')
H = pd.read_csv(OUT/'s3_summary_schd.csv')


def ranks(d):
    d = d.set_index('key').reindex(KEYS)
    return pd.DataFrame({
        'CAGR%': (d['CAGR_세전']*100).round(2),
        '수익률순위': d['CAGR_세전'].rank(ascending=False).astype(int),
        'Sharpe': d['Sharpe'].round(3),
        '위험조정순위': d['Sharpe'].rank(ascending=False).astype(int),
        'Calmar': d['Calmar'].round(3),
        'Calmar순위': d['Calmar'].rank(ascending=False).astype(int),
        'MDD%': (d['MDD']*100).round(2),
        'MDD순위': d['MDD'].rank(ascending=False).astype(int),   # 얕을수록 1위
    })


if __name__ == '__main__':
    pd.set_option('display.width', 250)
    out = {}
    for per in ['A', 'B', 'C']:
        out[per] = ranks(S[S['구간'] == per])
        print(f'\n===== 구간 {per} 순위 =====')
        print(out[per].to_string())

    cmp = pd.DataFrame({
        'A_수익률': out['A']['수익률순위'], 'B_수익률': out['B']['수익률순위'],
        'A_위험조정': out['A']['위험조정순위'], 'B_위험조정': out['B']['위험조정순위'],
        'A_MDD': out['A']['MDD순위'], 'B_MDD': out['B']['MDD순위'],
    })
    cmp['수익률순위변화'] = cmp['B_수익률']-cmp['A_수익률']
    cmp['위험조정순위변화'] = cmp['B_위험조정']-cmp['A_위험조정']
    cmp.to_csv(OUT/'s3_rank_AB.csv', encoding='utf-8-sig')
    print('\n===== 구간 A vs B 순위 비교 (숫자가 작을수록 상위) =====')
    print(cmp.to_string())

    # TQQQ=0 순위
    print('\n===== TQQQ=0 시나리오 순위 =====')
    zr = {}
    for per in ['A', 'C']:
        zr[per] = ranks(Z[Z['구간'] == per+'-TQQQ0'])
        print(f'\n--- 구간 {per} (TQQQ=0) ---')
        print(zr[per].to_string())
    fl = pd.DataFrame({'기본_순위': out['A']['수익률순위'], 'TQQQ0_순위': zr['A']['수익률순위'],
                       '기본_CAGR%': out['A']['CAGR%'], 'TQQQ0_CAGR%': zr['A']['CAGR%']})
    fl['순위변화'] = fl['TQQQ0_순위']-fl['기본_순위']
    fl.to_csv(OUT/'s3_rank_tqqq0.csv', encoding='utf-8-sig')
    print('\n[구간 A 기본 vs TQQQ=0]'); print(fl.to_string())

    # SCHD 프록시 민감도
    print('\n===== SCHD 프록시 민감도 =====')
    allr = {}
    for per in ['A', 'B', 'C']:
        tab = {}
        for schd in ['SCHD_a', 'SCHD_b', 'SCHD_c']:
            d = H[(H['구간'] == per) & (H['schd'] == schd)]
            r = ranks(d)
            tab[(schd, 'CAGR%')] = r['CAGR%']
            tab[(schd, '순위')] = r['수익률순위']
            tab[(schd, 'Sharpe순위')] = r['위험조정순위']
        t = pd.DataFrame(tab)
        allr[per] = t
        print(f'\n--- 구간 {per} ---')
        print(t.to_string())
        rk = t.xs('순위', axis=1, level=1)
        chg = {k: sorted(set(rk.loc[k])) for k in KEYS if rk.loc[k].nunique() > 1}
        print('  수익률 순위가 프록시에 따라 바뀌는 전략:', chg if chg else '없음')
        rs = t.xs('Sharpe순위', axis=1, level=1)
        chg2 = {k: sorted(set(rs.loc[k])) for k in KEYS if rs.loc[k].nunique() > 1}
        print('  위험조정 순위가 바뀌는 전략:', chg2 if chg2 else '없음')
    pd.concat(allr, axis=1).to_csv(OUT/'s3_rank_schd.csv', encoding='utf-8-sig')
    print('\nsaved -> out/s3_rank_AB.csv, s3_rank_tqqq0.csv, s3_rank_schd.csv')
