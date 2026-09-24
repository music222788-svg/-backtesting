"""1단계-c: 프록시 추적오차, 데이터 품질, 교차검증, 실데이터 비중."""
import json, pathlib
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
D, OUT, LOG = ROOT/'data', ROOT/'out', ROOT/'logs'
R  = pd.read_csv(OUT/'s1_returns.csv', parse_dates=['date']).set_index('date')
P  = pd.read_csv(OUT/'s1_proxy_returns.csv', parse_dates=['date']).set_index('date')
CAL = R.index

def raw(sym):
    return pd.read_csv(D/f'{sym}.csv', parse_dates=['date']).set_index('date').sort_index()

def cum(r):
    return float((1.0+r.dropna()).prod()-1.0)

def ann(r):
    n = r.dropna().shape[0]
    return float((1.0+r.dropna()).prod()**(252.0/n)-1.0) if n > 20 else np.nan

# ========== 1) 합성 vs 실제 추적오차 ==========
pairs = [('TQQQ','TQQQ_syn'), ('IEF','IEF_syn'), ('GLD','GLD_syn'), ('SGOV','SGOV_syn')]
rows, annual_rows = [], []
for sym, syn in pairs:
    rr = raw(sym)['adjclose'].reindex(CAL).pct_change()
    a = rr.dropna(); b = P[syn].reindex(a.index)
    m = a.notna() & b.notna(); a = a[m]; b = b[m]
    if len(a) < 10:
        continue
    diff = b - a
    rows.append(dict(asset=sym, overlap_start=str(a.index[0].date()), overlap_end=str(a.index[-1].date()),
                     n=len(a), real_cum=cum(a), syn_cum=cum(b), real_cagr=ann(a), syn_cagr=ann(b),
                     cagr_gap_pp=(ann(b)-ann(a))*100, corr=float(np.corrcoef(a, b)[0, 1]),
                     daily_mean_diff_bp=float(diff.mean()*1e4),
                     te_ann_pct=float(diff.std()*np.sqrt(252)*100)))
    for y, g in pd.DataFrame({'real': a, 'syn': b}).groupby(a.index.year):
        annual_rows.append(dict(asset=sym, year=int(y), n=len(g), real=cum(g['real']), syn=cum(g['syn']),
                                gap_pp=(cum(g['syn'])-cum(g['real']))*100))
track = pd.DataFrame(rows); track.to_csv(OUT/'s1_tracking.csv', index=False)
track_y = pd.DataFrame(annual_rows); track_y.to_csv(OUT/'s1_tracking_annual.csv', index=False)

# ========== 2) 데이터 품질 ==========
q = []
for sym in ['SPY','QQQ','SCHD','IEF','GLD','TQQQ','SGOV','IWD','DVY','VEIPX','VWNFX']:
    df = raw(sym)
    ar = df['adjclose'].pct_change(); cr = df['close'].pct_change()
    gap = (cr - ar).abs()
    q.append(dict(asset=sym, rows=len(df),
                  na_adj=int(df['adjclose'].isna().sum()), na_close=int(df['close'].isna().sum()),
                  zero_or_neg=int((df['adjclose'] <= 0).sum()),
                  dup_dates=int(df.index.duplicated().sum()),
                  gaps_gt5d=int((pd.Series(df.index).diff().dt.days > 5).sum()),
                  max_abs_dret=float(ar.abs().max()),
                  n_dret_gt20pct=int((ar.abs() > 0.20).sum()),
                  n_close_adj_gap_gt10pct=int((gap > 0.10).sum()),
                  max_close_adj_gap=float(gap.max())))
qual = pd.DataFrame(q); qual.to_csv(OUT/'s1_quality.csv', index=False)

# 분할 후보(원종가 수익률과 조정종가 수익률 괴리 > 30%)
sp = []
for sym in ['SPY','QQQ','SCHD','IEF','GLD','TQQQ','SGOV','IWD','DVY']:
    df = raw(sym); ar = df['adjclose'].pct_change(); cr = df['close'].pct_change()
    bad = (cr - ar).abs() > 0.30
    for dte in df.index[bad]:
        sp.append(dict(asset=sym, date=str(dte.date()), close_ret=float(cr[dte]), adj_ret=float(ar[dte]),
                       implied_ratio=float(df['close'].shift(1)[dte]/df['close'][dte])))
splits = pd.DataFrame(sp); splits.to_csv(OUT/'s1_split_candidates.csv', index=False)

# 이상치
out = []
for c in R.columns:
    s = R[c].dropna(); lim = 0.35 if c == 'TQQQ' else 0.15
    for dte, v in s[s.abs() > lim].items():
        out.append(dict(series=c, date=str(dte.date()), ret=float(v)))
pd.DataFrame(out).to_csv(OUT/'s1_outliers.csv', index=False)

# ========== 3) 교차검증: 알려진 낙폭 ==========
def mdd(r, s, e):
    x = r.loc[s:e].dropna(); w = (1+x).cumprod()
    dd = w/w.cummax() - 1.0
    return float(dd.min()), str(dd.idxmin().date()), str(w[:dd.idxmin()].idxmax().date())

# 1차 기준: 동일 기간 공식 지수(^GSPC, ^NDX) 가격지수 낙폭을 독립 시리즈로 계산
IDX = {'SPY': raw('GSPC')['close'].reindex(CAL).ffill().pct_change(),
       'QQQ': raw('NDX')['close'].reindex(CAL).ffill().pct_change()}
# 2차 기준: 널리 인용되는 공표 수치 (기억 기반 참고값 — 지수 계산값이 우선)
PUB = {('SPY','2000-01-03~2002-12-31'): -0.491, ('QQQ','2000-01-03~2002-12-31'): -0.830,
       ('SPY','2007-01-03~2009-12-31'): -0.568, ('QQQ','2007-01-03~2009-12-31'): -0.537,
       ('SPY','2020-02-19~2020-03-23'): -0.339, ('QQQ','2020-02-19~2020-03-23'): -0.281,
       ('SPY','2022-01-03~2022-12-30'): -0.254, ('QQQ','2022-01-03~2022-12-30'): -0.353}
wins = [('2000-01-03','2002-12-31','닷컴 붕괴'), ('2007-01-03','2009-12-31','금융위기'),
        ('2020-02-19','2020-03-23','코로나 폭락'), ('2022-01-03','2022-12-30','2022 약세장')]
cc = []
for sym in ['SPY','QQQ']:
    for s_, e_, note in wins:
        v, td, pk = mdd(R[sym], s_, e_)
        iv, itd, ipk = mdd(IDX[sym], s_, e_)
        w = f'{s_}~{e_}'
        cc.append(dict(asset=sym, window=w, note=note,
                       etf_tr_mdd=v, etf_peak=pk, etf_trough=td,
                       index_price_mdd=iv, index_peak=ipk, index_trough=itd,
                       tr_minus_price_pp=(v-iv)*100,
                       published_ref=PUB[(sym, w)], etf_vs_pub_pp=(v-PUB[(sym, w)])*100))
cross = pd.DataFrame(cc); cross.to_csv(OUT/'s1_crosscheck.csv', index=False)

# ========== 4) 실데이터 비중 (구간 A) ==========
meta = json.load(open(LOG/'series_meta.json', encoding='utf-8'))
real_start = {k: pd.Timestamp(v['real_start']) for k, v in meta.items()}
TW = {
    'P1': {'SPY': .48, 'SCHD': .16, 'IEF': .08, 'GLD': .08, 'TQQQ': .12, 'SGOV': .08},
    'P3': {'SPY': .42, 'SCHD': .14, 'IEF': .07, 'GLD': .07, 'TQQQ': .18, 'SGOV': .12},
    'C1': {'SPY': .60, 'SCHD': .20, 'IEF': .10, 'GLD': .10},
    'B1': {'SPY': 1.0}, 'B2': {'QQQ': 1.0},
}
rs = dict(SPY=real_start['SPY'], QQQ=real_start['QQQ'], IEF=real_start['IEF'],
          GLD=real_start['GLD'], TQQQ=real_start['TQQQ'], SGOV=real_start['SGOV'],
          SCHD=real_start['SCHD_a'])
A = CAL[(CAL >= pd.Timestamp('2000-01-03')) & (CAL <= pd.Timestamp('2019-12-31'))]
rows = []
for y, g in pd.Series(A, index=A).groupby(A.year):
    rec = {'year': int(y)}
    for pf, w in TW.items():
        rec[pf] = sum(wt*float((g.index >= rs[a]).mean()) for a, wt in w.items())
    rows.append(rec)
pd.DataFrame(rows).to_csv(OUT/'s1_realdata_share_A.csv', index=False)

# 자산별 실/프록시 구간표
dl = {r['symbol']: r for r in json.load(open(LOG/'download_log.json', encoding='utf-8'))}
label = {'SPY': 'SPY', 'QQQ': 'QQQ', 'SCHD_a': 'SCHD (프록시 a)', 'SCHD_b': 'SCHD (프록시 b)',
         'SCHD_c': 'SCHD (프록시 c)', 'IEF': 'IEF', 'GLD': 'GLD', 'TQQQ': 'TQQQ', 'SGOV': 'SGOV'}
src = {'SPY': 'Yahoo SPY', 'QQQ': 'Yahoo QQQ', 'SCHD_a': 'Yahoo SCHD / SPY',
       'SCHD_b': 'Yahoo SCHD / IWD / VWNFX', 'SCHD_c': 'Yahoo SCHD / DVY / VEIPX',
       'IEF': 'Yahoo IEF / FRED DGS7,DGS10', 'GLD': 'Yahoo GLD / LBMA gold PM',
       'TQQQ': 'Yahoo TQQQ / Yahoo QQQ + FRED DTB3', 'SGOV': 'Yahoo SGOV / FRED DTB3'}
cov = []
for k, lab in label.items():
    st = real_start[k]
    cov.append(dict(자산=lab, 실데이터=f'{st.date()} ~ 2026-09-16',
                    프록시=('없음' if st <= pd.Timestamp('2000-01-03')
                          else f'2000-01-03 ~ {(st-pd.Timedelta(days=1)).date()}'),
                    프록시방식=meta[k]['proxy'], 출처=src[k], 다운로드일=dl['SPY']['downloaded'][:10]))
pd.DataFrame(cov).to_csv(OUT/'s1_coverage.csv', index=False, encoding='utf-8-sig')

# ========== 5) 합성 3배 상품 원금 1% 미만 점검 ==========
chk = []
for start in ['2000-01-03', '2020-01-02']:
    s = P['TQQQ_syn'].loc[start:'2026-09-16'].dropna()
    w = (1+s).cumprod()
    below = w[w < 0.01]
    chk.append(dict(start=start, min_index=float(w.min()), min_date=str(w.idxmin().date()),
                    days_below_1pct=int(len(below)),
                    first_below=str(below.index[0].date()) if len(below) else '',
                    n_daily_ret_le_m100pct=int((s <= -1.0).sum()),
                    worst_daily=float(s.min()), worst_daily_date=str(s.idxmin().date())))
pd.DataFrame(chk).to_csv(OUT/'s1_3x_wipeout.csv', index=False)

print('=== TRACKING ===');  print(track.to_string(index=False))
print('\n=== 3X WIPEOUT CHECK ==='); print(pd.DataFrame(chk).to_string(index=False))
print('\n=== CROSSCHECK ===')
print(cross.to_string(index=False))
print('\n=== QUALITY ==='); print(qual.to_string(index=False))
print('\n=== SPLIT CANDIDATES ==='); print(splits.to_string(index=False) if len(splits) else '(none)')
