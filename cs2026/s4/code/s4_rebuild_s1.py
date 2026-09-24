"""S4-0: Drive 사본에 없는 out/s1_macro.csv, out/s1_proxy_returns.csv 재생성.

기존 code/s1_build.py 의 공식을 그대로 옮긴다. 차이는 단 하나:
마스터 캘린더(SPY 거래일)를 data/SPY.csv 대신 out/s1_returns.csv 의 인덱스에서 가져온다
(Drive 사본에 SPY.csv 가 없음). 결과는 s4/rebuilt/ 에 저장하며 기존 out/ 은 건드리지 않는다.

검증: 재생성한 TQQQ_syn·SGOV_syn·IEF_syn·GLD_syn 이 s1_returns.csv 의 프록시 구간
(실데이터 시작 이전) 값과 일치하는지 확인한다.
"""
import pathlib
import numpy as np, pandas as pd

S4 = pathlib.Path(__file__).resolve().parent.parent
ROOT = S4.parent
D, OUT, RB = ROOT/'data', ROOT/'out', S4/'rebuilt'
RB.mkdir(exist_ok=True)

EXP_QQQ, EXP_TQQQ, EXP_IEF, EXP_GLD, IEF_MAT = 0.0020, 0.0084, 0.0015, 0.0040, 8.5

R1 = pd.read_csv(OUT/'s1_returns.csv', parse_dates=['date']).set_index('date')
CAL = R1.index


def ycsv(name):
    return pd.read_csv(D/name, parse_dates=['date']).set_index('date').sort_index()


def fred(sid):
    df = pd.read_csv(D/f'FRED_{sid}.csv')
    df.columns = ['date', 'value']
    df['date'] = pd.to_datetime(df['date'])
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    return df.dropna().drop_duplicates('date').set_index('date')['value'].sort_index()


def to_cal(s):
    return s.reindex(CAL)


dtb3 = to_cal(fred('DTB3')).ffill()
dgs7 = to_cal(fred('DGS7')).ffill()
dgs10 = to_cal(fred('DGS10')).ffill()
krw = to_cal(fred('DEXKOUS')).ffill()
d = dtb3/100.0
price_bill = 1.0 - d*91.0/360.0
bey = (1.0 - price_bill)/price_bill*365.0/91.0
days = pd.Series(CAL, index=CAL).diff().dt.days.astype('float')
sgov_proxy_ret = (bey.shift(1)*days/365.0)
sgov_proxy_ret.iloc[0] = np.nan

y85 = (dgs7 + (IEF_MAT-7.0)/(10.0-7.0)*(dgs10-dgs7))/100.0


def par_bond_price(y, c, M):
    n = np.maximum(np.ceil(M*2.0), 1.0)
    frac = n - M*2.0
    i = y/2.0
    k = np.arange(1, int(np.nanmax(n))+1)[:, None]
    mask = (k <= n[None, :])
    disc = np.where(mask, (1.0+i[None, :])**(-(k-frac[None, :])), 0.0)
    pv_cpn = (c*100.0/2.0)*disc.sum(axis=0)
    pv_prn = 100.0*(1.0+i)**(-(n-frac))
    return pv_cpn + pv_prn


yv = y85.to_numpy()
dt_yr = days.to_numpy()/365.0
c_prev = np.roll(yv, 1)
M_now = IEF_MAT - dt_yr
P_now = par_bond_price(np.nan_to_num(yv, nan=0.03), np.nan_to_num(c_prev, nan=0.03),
                       np.nan_to_num(M_now, nan=IEF_MAT))
ief_proxy_ret = pd.Series((P_now-100.0)/100.0, index=CAL)
ief_proxy_ret = ief_proxy_ret.where(~(np.isnan(yv) | np.isnan(np.roll(yv, 1)) | np.isnan(dt_yr)))
ief_proxy_ret.iloc[0] = np.nan
ief_proxy_ret = ief_proxy_ret - EXP_IEF*dt_yr

gold = pd.read_csv(D/'LBMA_gold_pm.csv', parse_dates=['date']).set_index('date')['usd_per_oz']
gold = to_cal(gold).ffill()
gld_proxy_ret = gold.pct_change() - EXP_GLD*dt_yr

qqq_ret = to_cal(ycsv('QQQ.csv')['adjclose']).pct_change()
ndx_tr_ret = qqq_ret + EXP_QQQ*dt_yr
fin = 2.0*(bey.shift(1))*(days/365.0)
tqqq_proxy_ret = 3.0*ndx_tr_ret - fin - EXP_TQQQ*dt_yr

PROXY = pd.DataFrame({'TQQQ_syn': tqqq_proxy_ret, 'IEF_syn': ief_proxy_ret,
                      'GLD_syn': gld_proxy_ret, 'SGOV_syn': sgov_proxy_ret,
                      'NDX_TR': ndx_tr_ret})
PROXY.to_csv(RB/'s1_proxy_returns.csv')
pd.DataFrame({'krw': krw, 'bey': bey, 'y85': y85, 'days': days}).to_csv(RB/'s1_macro.csv')

# ---------------- 검증 ----------------
chk = [('TQQQ', 'TQQQ_syn', '2010-02-11'), ('SGOV', 'SGOV_syn', '2020-06-01'),
       ('IEF', 'IEF_syn', '2002-07-30'), ('GLD', 'GLD_syn', '2004-11-18'),
       ('QQQ', None, None)]
for a, p, real in chk:
    if p is None:
        x, y = R1[a], qqq_ret
    else:
        m = CAL < pd.Timestamp(real)
        x, y = R1.loc[m, a], PROXY.loc[m, p]
    both = x.notna() & y.notna()
    diff = (x[both]-y[both]).abs().max()
    print(f'{a:5s} vs {p or "QQQ.csv"}: n={int(both.sum())}  max|diff|={diff:.3e}  '
          f'NaN불일치={int((x.notna() != y.notna()).sum())}')
print('saved ->', RB)
