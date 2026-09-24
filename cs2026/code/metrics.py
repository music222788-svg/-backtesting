"""성과·위험 지표.

- 위험지표(MDD·변동성·Sharpe·Sortino·Calmar)는 세전 경로에서 산출
- 세전/세후 모두 구간 종료일 전량 청산을 포함 (조건 D1) -> 차이는 순수하게 세금
- 무위험수익률: 3개월 미국채 BEY 일할수익 (out/s1_proxy_returns.csv 의 SGOV_syn)
"""
import numpy as np
import pandas as pd

TD = 252.0


def dret(eq):
    return eq.pct_change().dropna()


def cagr(eq):
    n = len(eq)-1
    if n <= 0 or eq.iloc[0] <= 0 or eq.iloc[-1] <= 0:
        return np.nan
    return float((eq.iloc[-1]/eq.iloc[0])**(TD/n)-1.0)


def drawdown(eq):
    return eq/eq.cummax()-1.0


def mdd(eq):
    return float(drawdown(eq).min())


def mdd_dates(eq):
    dd = drawdown(eq)
    t = dd.idxmin()
    p = eq[:t].idxmax()
    return p, t


def longest_recovery(eq):
    """전고점 -> 회복까지 최장 거래일수. 미회복이면 (일수, False)."""
    peak = eq.cummax()
    under = eq < peak*(1-1e-12)
    best, cur, start_i = 0, 0, None
    unrecovered = 0
    idx = eq.index
    for i, u in enumerate(under.values):
        if u:
            if cur == 0:
                start_i = i
            cur += 1
        else:
            best = max(best, cur); cur = 0
    if cur > 0:
        unrecovered = cur
    if unrecovered > best:
        return unrecovered, False
    return best, True


def vol(eq):
    return float(dret(eq).std()*np.sqrt(TD))


def sharpe(eq, rf):
    r = dret(eq)
    ex = r - rf.reindex(r.index).fillna(0.0)
    s = ex.std()
    return float(ex.mean()/s*np.sqrt(TD)) if s > 0 else np.nan


def sortino(eq, rf):
    r = dret(eq)
    ex = r - rf.reindex(r.index).fillna(0.0)
    d = ex[ex < 0]
    ds = np.sqrt((d**2).sum()/len(ex)) if len(ex) else np.nan
    return float(ex.mean()/ds*np.sqrt(TD)) if ds and ds > 0 else np.nan


def calmar(eq):
    m = mdd(eq)
    return float(cagr(eq)/abs(m)) if m < 0 else np.nan


def annual_returns(eq):
    """달력연도 수익률. 첫 해는 시작일부터, 마지막 해는 종료일까지의 부분연도."""
    out = {}
    for y, g in eq.groupby(eq.index.year):
        prev = eq[eq.index < g.index[0]]
        base = prev.iloc[-1] if len(prev) else g.iloc[0]
        out[int(y)] = float(g.iloc[-1]/base-1.0)
    return pd.Series(out)


def worst_rolling_cagr(eq, years):
    w = int(round(years*TD))
    if len(eq) <= w:
        return np.nan
    r = (eq.shift(-w)/eq)**(1.0/years)-1.0
    r = r.dropna()
    return float(r.min()) if len(r) else np.nan


def period_return(eq, s, e):
    x = eq.loc[s:e]
    if len(x) < 2:
        return np.nan
    return float(x.iloc[-1]/x.iloc[0]-1.0)


def recovery_days(eq, peak_date, from_date=None):
    """peak_date 의 고점을 다시 넘어서기까지 걸린 거래일수."""
    lvl = float(eq.loc[peak_date])
    after = eq.loc[peak_date:]
    hit = after[after >= lvl]
    hit = hit[hit.index > peak_date]
    if len(hit) == 0:
        return None, len(after)-1
    d = hit.index[0]
    return d, int(eq.index.get_loc(d) - eq.index.get_loc(peak_date))


def summarize(name, eq_pre, eq_post, fx, rf, trades, taxes):
    """eq_pre: 세전(청산포함) 경로, eq_post: 세후(청산포함) 경로."""
    eq_krw = eq_pre*fx.reindex(eq_pre.index).ffill()
    lr, rec = longest_recovery(eq_pre)
    ar = annual_returns(eq_pre)
    yrs = (len(eq_pre)-1)/TD
    ntr = len(trades)
    tax_tot = float(taxes['tax'].sum()) if len(taxes) else 0.0
    p, t = mdd_dates(eq_pre)
    return dict(
        전략=name,
        최종자산_세전=float(eq_pre.iloc[-1]),
        최종자산_세후=float(eq_post.iloc[-1]),
        최종자산_원화=float(eq_krw.iloc[-1]),
        CAGR_세전=cagr(eq_pre), CAGR_세후=cagr(eq_post), CAGR_원화=cagr(eq_krw),
        MDD=mdd(eq_pre), MDD고점=str(p.date()), MDD저점=str(t.date()),
        최장회복_거래일=int(lr), 회복완료=rec,
        연변동성=vol(eq_pre), Sharpe=sharpe(eq_pre, rf), Sortino=sortino(eq_pre, rf),
        Calmar=calmar(eq_pre),
        최악의연도=float(ar.min()), 최악연도=int(ar.idxmin()),
        최악롤링3년=worst_rolling_cagr(eq_pre, 3), 최악롤링5년=worst_rolling_cagr(eq_pre, 5),
        총매매횟수=ntr, 연평균매매=ntr/yrs,
        총세금=tax_tot, 연평균세금=tax_tot/yrs,
        세금비중=tax_tot/float(eq_pre.iloc[-1]) if eq_pre.iloc[-1] else np.nan,
    )
