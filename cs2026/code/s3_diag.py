"""3단계-f: 진단(기본 시나리오를 바꾸지 않는 부가 분석).

(1) 합성 TQQQ 의 측정된 상방 편향(+0.8614 bp/일)을 프록시 구간에만 제거했을 때
    구간 A/C 결과가 얼마나 달라지는가 — 확정 조건 D10 은 '보정 없이 보고' 이므로
    기본 결과는 그대로 두고, 편향의 크기만 별도로 보고한다.
(2) 자산별 기여도: 코어의 SPY 대비 우위가 어디서 왔는가.
"""
import sys, pathlib
import numpy as np, pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent
OUT = ROOT/'out'
from engine_fast import run_fast
import metrics as M

R = pd.read_csv(OUT/'s1_returns.csv', parse_dates=['date']).set_index('date')
MAC = pd.read_csv(OUT/'s1_macro.csv', parse_dates=['date']).set_index('date')
FX = MAC['krw']
TRK = pd.read_csv(OUT/'s1_tracking.csv').set_index('asset')
BIAS = float(TRK.loc['TQQQ', 'daily_mean_diff_bp'])/1e4     # 일간 평균 초과분
PROXY_END = pd.Timestamp('2010-02-10')                       # TQQQ 실데이터 직전일
KEYS = ['P1', 'P2', 'P3', 'P4', 'B1', 'B2', 'C1']
PER = {'A': ('2000-01-03', '2019-12-31'), 'C': ('2000-01-03', '2026-09-16')}

if __name__ == '__main__':
    pd.set_option('display.width', 220)
    print(f'측정된 합성 TQQQ 일간 상방 편향: {BIAS*1e4:.4f} bp/일 '
          f'(= 연 {((1+BIAS)**252-1)*100:.2f}pp)')

    adj = R.copy()
    m = adj.index <= PROXY_END
    adj.loc[m, 'TQQQ'] = adj.loc[m, 'TQQQ'] - BIAS

    rows = []
    for per, (s, e) in PER.items():
        for k in KEYS:
            a = run_fast(k, R, FX, s, e)['equity']
            b = run_fast(k, adj, FX, s, e)['equity']
            rows.append(dict(구간=per, key=k,
                             CAGR_기본=M.cagr(a)*100, CAGR_편향제거=M.cagr(b)*100,
                             차이pp=(M.cagr(b)-M.cagr(a))*100,
                             최종_기본=float(a.iloc[-1]), 최종_편향제거=float(b.iloc[-1])))
    d = pd.DataFrame(rows)
    d.round(3).to_csv(OUT/'s3_diag_tqqq_bias.csv', index=False, encoding='utf-8-sig')
    for per in ['A', 'C']:
        x = d[d['구간'] == per].set_index('key').reindex(KEYS)
        x['기본순위'] = x['CAGR_기본'].rank(ascending=False).astype(int)
        x['편향제거순위'] = x['CAGR_편향제거'].rank(ascending=False).astype(int)
        print(f'\n===== 구간 {per}: 합성 TQQQ 편향 제거 시 =====')
        print(x[['CAGR_기본', 'CAGR_편향제거', '차이pp', '기본순위', '편향제거순위']].round(2).to_string())

    # (2) 자산별 단순보유 기여 확인
    print('\n===== 자산별 단순보유 CAGR (%) =====')
    rows = []
    for per, (s, e) in [('A', PER['A']), ('C', PER['C'])]:
        for a in ['SPY', 'QQQ', 'SCHD_a', 'SCHD_b', 'SCHD_c', 'IEF', 'GLD', 'TQQQ', 'SGOV']:
            r = R.loc[s:e, a].iloc[1:]
            g = float((1+r).prod()); n = len(r)
            rows.append(dict(구간=per, 자산=a, CAGR=(g**(252/n)-1)*100, 배수=g))
    z = pd.DataFrame(rows)
    z.round(3).to_csv(OUT/'s3_asset_cagr.csv', index=False, encoding='utf-8-sig')
    print(z.pivot(index='자산', columns='구간', values='CAGR').round(2).to_string())
