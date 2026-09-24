"""예시 계산(확정조건 아님): 인출이 있을 때 순위가 어떻게 바뀌는가.

가정 (이 계산 한정, 승인받은 조건이 아님):
  - 시작 5억원, 원화 기준 (USD 자산곡선 x 일간 USD/KRW)
  - 매월 말 정액 인출, 연 인출률 x 5억 / 12 (물가연동 없음, 명목 고정)
  - 세전. 인출은 포트폴리오에서 안분 매도 -> 비중/규칙 불변이므로
    자산곡선 수익률에 인출을 적용하는 것으로 정확히 동일
"""
import pathlib
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT/'out'
KEYS = ['P1', 'P2', 'P3', 'P4', 'B1', 'B2', 'C1']
INIT = 5e8


def krw_paths(per):
    eq = pd.read_csv(OUT/f's3_equity_{per}.csv', parse_dates=['date']).set_index('date')
    fx = pd.read_csv(OUT/'s1_macro.csv', parse_dates=['date']).set_index('date')['krw']
    return eq.mul(fx.reindex(eq.index).ffill(), axis=0)


def simulate(ret, rate):
    """ret: 일간 수익률. 매월 마지막 거래일에 정액 인출."""
    w = INIT*rate/12.0
    idx = ret.index
    month_end = set(pd.Series(idx, index=idx).groupby([idx.year, idx.month]).max())
    v = INIT
    path = []
    depleted = None
    for d, r in ret.items():
        v *= (1.0+r)
        if d in month_end:
            v -= w
            if v <= 0 and depleted is None:
                depleted = d
                v = 0.0
        path.append(v)
    s = pd.Series(path, index=idx)
    return s, depleted


if __name__ == '__main__':
    pd.set_option('display.width', 220)
    rows = []
    for per in ['A', 'C']:
        K = krw_paths(per)
        for rate in [0.00, 0.04, 0.05]:
            for k in KEYS:
                ret = K[k].pct_change().dropna()
                s, dep = simulate(ret, rate)
                rows.append(dict(구간=per, 인출률=f'{rate*100:.0f}%', key=k,
                                 최종억=s.iloc[-1]/1e8, 최저억=s.min()/1e8,
                                 최저일=str(s.idxmin().date()),
                                 고갈=('' if dep is None else str(dep.date())),
                                 원금대비=s.iloc[-1]/INIT))
    d = pd.DataFrame(rows)
    d.round(3).to_csv(OUT/'s4_withdraw.csv', index=False, encoding='utf-8-sig')
    for per in ['A', 'C']:
        print(f'\n===== 구간 {per} · 5억원 시작 · 원화 기준 · 세전 =====')
        for rate in ['0%', '4%', '5%']:
            x = d[(d['구간'] == per) & (d['인출률'] == rate)].set_index('key').reindex(KEYS)
            t = pd.DataFrame({'최종(억)': x['최종억'].round(2), '최저(억)': x['최저억'].round(2),
                              '최저시점': x['최저일'], '고갈': x['고갈'],
                              '순위': x['최종억'].rank(ascending=False).astype(int)})
            print(f'\n[연 인출률 {rate}]')
            print(t.to_string())
