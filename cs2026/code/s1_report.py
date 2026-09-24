"""1단계-d: 운용사 확인 메타데이터 결합 + 1단계 요약표 저장."""
import json, pathlib
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT, LOG = ROOT/'out', ROOT/'logs'

# 운용사 공식 페이지에서 2026-09-17 확인한 값 (웹 확인 결과)
ISSUER = [
    dict(ticker='SPY',   name='SPDR S&P 500 ETF Trust',            issuer='State Street',
         inception='1993-01-22', first_trade='1993-01-29', expense='0.0945%',
         source='Yahoo firstTradeDate (운용사 페이지 미확인)', verified='부분'),
    dict(ticker='QQQ',   name='Invesco QQQ Trust',                 issuer='Invesco',
         inception='1999-03-10', first_trade='1999-03-10', expense='0.18% (현재)',
         source='invesco.com QQQ 제품 페이지', verified='확인'),
    dict(ticker='SCHD',  name='Schwab U.S. Dividend Equity ETF',   issuer='Schwab',
         inception='2011-10-20', first_trade='2011-10-20', expense='0.060%',
         source='schwabassetmanagement.com/products/schd', verified='확인'),
    dict(ticker='IEF',   name='iShares 7-10 Year Treasury Bond ETF', issuer='BlackRock',
         inception='2002-07-22', first_trade='2002-07-30', expense='0.15%',
         source='ishares.com 제품 239456', verified='확인'),
    dict(ticker='GLD',   name='SPDR Gold Shares',                  issuer='World Gold Trust Services',
         inception='2004-11-18', first_trade='2004-11-18', expense='0.40%',
         source='ssga.com GLD 제품 페이지', verified='확인'),
    dict(ticker='TQQQ',  name='ProShares UltraPro QQQ',            issuer='ProShares',
         inception='2010-02-09', first_trade='2010-02-11', expense='0.82% 순 / 0.97% 총',
         source='proshares.com TQQQ 제품 페이지', verified='확인'),
    dict(ticker='SGOV',  name='iShares 0-3 Month Treasury Bond ETF', issuer='BlackRock',
         inception='2020-05-26', first_trade='2020-06-01', expense='0.09%',
         source='ishares.com 제품 314116', verified='확인'),
    dict(ticker='IWD',   name='iShares Russell 1000 Value ETF',    issuer='BlackRock',
         inception='2000-05-22', first_trade='2000-05-26', expense='0.18%',
         source='ishares.com 제품 239708', verified='확인'),
    dict(ticker='DVY',   name='iShares Select Dividend ETF',       issuer='BlackRock',
         inception='2003-11-03', first_trade='2003-11-07', expense='0.38%',
         source='ishares.com 제품 239500', verified='확인'),
    dict(ticker='VEIPX', name='Vanguard Equity Income Fund Inv',   issuer='Vanguard',
         inception='1988-03-21', first_trade='1988-03-21', expense='0.26%',
         source='advisors.vanguard.com VEIPX', verified='확인'),
    dict(ticker='VWNFX', name='Vanguard Windsor II Fund Inv',      issuer='Vanguard',
         inception='1985-06-24', first_trade='1985-06-24', expense='0.33%',
         source='advisors.vanguard.com VWNFX', verified='확인'),
]
iss = pd.DataFrame(ISSUER)
iss.to_csv(OUT/'s1_issuer_meta.csv', index=False, encoding='utf-8-sig')

# TQQQ 소멸(원금 1% 미만) 플래그 저장
P = pd.read_csv(OUT/'s1_proxy_returns.csv', parse_dates=['date']).set_index('date')
flags = {}
for start in ['2000-01-03', '2020-01-02']:
    s = P['TQQQ_syn'].loc[start:'2026-09-16'].dropna()
    w = (1+s).cumprod()
    b = w[w < 0.01]
    flags[start] = dict(wipeout=bool(len(b)), first_below=str(b.index[0].date()) if len(b) else None,
                        min_index=float(w.min()), min_date=str(w.idxmin().date()))
json.dump(flags, open(LOG/'tqqq_wipeout_flags.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)

print(iss.to_string(index=False))
print()
print(json.dumps(flags, indent=1, ensure_ascii=False))
