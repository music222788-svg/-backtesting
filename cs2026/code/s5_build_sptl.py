"""5단계-a: SPTL 시리즈 추가 (실데이터 + VUSTX 프록시) → out/s1_returns.csv 에 열 추가.

- SPTL 설정일 2007-05-23 / 첫 거래일 2007-05-30, 보수 0.03%,
  벤치마크 Bloomberg Long U.S. Treasury Index(잔존 10년 이상), 평균만기 21.69년
  (ssga.com 제품 페이지, 2026-09-18 확인)
- 상장 전(2000-01-03 ~ 2007-05-29) 프록시: VUSTX (Vanguard Long-Term Treasury,
  설정 1986-05-19, 평균만기 ~22년). 지수의 사후 백필이 아니라 실제 거래된 펀드다.
- 기존 7자산 결과는 SPTL 을 쓰지 않으므로 영향받지 않는다 (열 추가만).
"""
import json, pathlib
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
D, OUT, LOG = ROOT/'data', ROOT/'out', ROOT/'logs'

R = pd.read_csv(OUT/'s1_returns.csv', parse_dates=['date']).set_index('date')
CAL = R.index


def adj(sym):
    return pd.read_csv(D/f'{sym}.csv', parse_dates=['date']).set_index('date')['adjclose']


sptl_raw, vustx_raw = adj('SPTL'), adj('VUSTX')
sptl = sptl_raw.reindex(CAL).pct_change()
vustx = vustx_raw.reindex(CAL).pct_change()

real_start = sptl_raw.index[0]
s = vustx.copy()
s[CAL >= real_start] = sptl[CAL >= real_start]
R['SPTL'] = s
R.to_csv(OUT/'s1_returns.csv')

# 추적오차 기록
a = sptl.dropna(); b = vustx.reindex(a.index)
m = a.notna() & b.notna(); a, b = a[m], b[m]


def ann(x):
    x = x.dropna()
    return float((1+x).prod()**(252/len(x))-1)


rec = dict(asset='SPTL', overlap_start=str(a.index[0].date()), overlap_end=str(a.index[-1].date()),
           n=int(len(a)), real_cagr=ann(a), proxy_cagr=ann(b),
           cagr_gap_pp=(ann(b)-ann(a))*100, corr=float(np.corrcoef(a, b)[0, 1]),
           daily_mean_diff_bp=float((b-a).mean()*1e4),
           te_ann_pct=float((b-a).std()*np.sqrt(252)*100))
pd.DataFrame([rec]).to_csv(OUT/'s5_sptl_tracking.csv', index=False, encoding='utf-8-sig')

# 품질 점검
q = dict(rows=int(len(sptl_raw)), na=int(sptl_raw.isna().sum()),
         zero_or_neg=int((sptl_raw <= 0).sum()),
         max_abs_dret=float(sptl_raw.pct_change().abs().max()),
         dup=int(sptl_raw.index.duplicated().sum()))
json.dump(dict(tracking=rec, quality=q, real_start=str(real_start.date()),
               proxy='VUSTX 1986-05-19~2007-05-29',
               source='Yahoo SPTL / Yahoo VUSTX', downloaded='2026-09-18'),
          open(LOG/'sptl_meta.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)

print(f"SPTL 실데이터 {real_start.date()} ~ {CAL[-1].date()} / 프록시 VUSTX 2000-01-03 ~ "
      f"{(real_start - pd.Timedelta(days=1)).date()}")
print(f"추적오차: 실제 CAGR {rec['real_cagr']*100:.2f}% vs 프록시 {rec['proxy_cagr']*100:.2f}% "
      f"({rec['cagr_gap_pp']:+.2f}pp), 상관 {rec['corr']:.4f}, TE {rec['te_ann_pct']:.2f}%")
print('품질:', q)
for per, (st, en) in {'A': ('2000-01-03', '2019-12-31'),
                      'C': ('2000-01-03', '2026-09-16')}.items():
    x = R.loc[st:en, 'SPTL'].iloc[1:]
    print(f'  구간 {per} SPTL 단순보유 CAGR {((1+x).prod()**(252/len(x))-1)*100:.2f}%')
print('saved -> out/s1_returns.csv (SPTL 열 추가), out/s5_sptl_tracking.csv')
