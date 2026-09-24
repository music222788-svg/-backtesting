"""S5 보조 계산.
1) 금 순위 역전점: 일시투자 구간 C 세후 Sharpe 기준으로 두 포트폴리오가 같아지는 금 연율 수익률
2) 가정 인플레이션 부트스트랩(한국 CPI 미확보 보완): 연 2%(한국은행 물가안정목표), 3%
   - 초기 1.5억은 현재가치, 매월 적립 450만원을 월 복리 π/12 로 명목 증액, 최종자산은 (1+π)^년 으로 할인
   - 이것은 CPI 실측이 아니라 '가정'이다
"""
import sys, pathlib, json
from multiprocessing import Pool
import numpy as np, pandas as pd
from scipy.optimize import brentq

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s5_lib as S
import s5_dca as DCA
from dca_engine import simulate, calendar_struct
L = S.L
OUT = S.S5


def sharpe_C(key, gold_cagr):
    sp, _ = S.PORTS[key]
    R, _ = S.frames()
    dl = S.gold_drift_for(gold_cagr, R)
    R2, _ = S.frames(gold_drift=dl)
    m, _, _ = L.evaluate(sp, R2, 'C')
    return m['Sharpe_세후']


def flips():
    info = json.load(open(OUT/'s5_gold_drift.json', encoding='utf-8'))
    bopt = sharpe_C('B-opt', 0.05)   # 금 비보유: 금 수익률과 무관
    spy = sharpe_C('SPY', 0.05)
    res = {}
    for k in ['A80', 'P3', 'C100']:
        for other, val in [('B-opt', bopt), ('SPY', spy)]:
            f = lambda g: sharpe_C(k, g) - val
            lo, hi = -0.15, info['gld_actual_cagr']
            if f(lo)*f(hi) > 0:
                res[f'{k}=={other}'] = f'-15%까지 역전 없음 (금 -15%에서 차이 {f(lo):.3f})'
            else:
                res[f'{k}=={other}'] = brentq(f, lo, hi, xtol=1e-5)
    res['B-opt_Sharpe_C'] = bopt; res['SPY_Sharpe_C'] = spy
    json.dump(res, open(OUT/'cache'/'gold_flip.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    return res


def _job(args):
    k, years, infl = args
    sp, lev = S.PORTS[k]
    R, D = S.frames()
    pool = S.window('2000-01-03', '2026-09-16')
    Rp, Dp = S.arrays(sp, R, D, pool)
    fxr = S.FX_ALL.loc[pool].pct_change().fillna(0.0).to_numpy()
    fx0 = float(S.FX_ALL.loc[S.TEND])
    T = years*252
    cal = S.CAL_ALL[S.CAL_ALL >= S.T0][:T]
    cs = calendar_struct(cal)
    mi = np.cumsum(cs['is_contrib']) - 1                      # 월 번호 (0부터)
    ck = np.where(cs['is_contrib'], S.MONTHLY_KRW*(1+infl)**(mi/12.0), 0.0)
    rows = []
    for it, idx in enumerate(DCA.boot_paths(years)):
        fx = fx0*np.cumprod(np.r_[1.0, 1.0+fxr[idx[1:]]])
        out = simulate(sp, Rp[idx], Dp[idx], fx, cs, init_krw=S.INIT_KRW, contrib_krw=ck, tax='krw', method='M1')
        rows.append(out['eq'][-1]*fx[-1]/(1+infl)**years)
    f = np.array(rows)
    return dict(key=k, 보유년=years, 가정인플레=infl, 실질중앙값=np.median(f), 하위10=np.quantile(f, .1),
                하위25=np.quantile(f, .25), 상위10=np.quantile(f, .9), P_25억이상=(f >= S.GOAL).mean()*100,
                P_15억이상=(f >= S.GOAL2).mean()*100)


if __name__ == '__main__':
    print(flips())
    jobs = [(k, y, pi) for k in DCA.ORDER for y in (14, 19) for pi in (0.02, 0.03)]
    with Pool(4) as p:
        r = p.map(_job, jobs)
    df = pd.DataFrame(r)
    df.to_csv(OUT/'s5_dca_boot_assumed_inflation.csv', index=False, encoding='utf-8-sig')
    x = df.copy()
    for c in ['실질중앙값', '하위10', '하위25', '상위10']:
        x[c] = (x[c]/1e8).round(2)
    pd.set_option('display.width', 200)
    print(x.round(1).to_string(index=False))
