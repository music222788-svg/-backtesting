"""1단계-b: 거래일 캘린더 정렬, 프록시 합성, 자산별 총수익 지수 생성."""
import json, pathlib
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
D, OUT, LOG = ROOT/'data', ROOT/'out', ROOT/'logs'
OUT.mkdir(exist_ok=True); LOG.mkdir(exist_ok=True)

# 승인된 조건(D9~D13)에 따른 상수
EXP_QQQ  = 0.0020   # QQQ 운용보수 -> NDX 총수익 복원에 환원
EXP_TQQQ = 0.0084   # TQQQ 순보수 (D10)
EXP_IEF  = 0.0015   # IEF 운용보수 (D11)
EXP_GLD  = 0.0040   # GLD 운용보수 (D12)
IEF_MAT  = 8.5      # ICE 7-10Y 평균만기 근사 (D11)

def ycsv(name):
    df = pd.read_csv(D/name, parse_dates=['date']).set_index('date').sort_index()
    return df

def fred(sid):
    df = pd.read_csv(D/f'FRED_{sid}.csv')
    df.columns = ['date','value']
    df['date'] = pd.to_datetime(df['date'])
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    return df.dropna().drop_duplicates('date').set_index('date')['value'].sort_index()

# ---------- 마스터 거래일 캘린더 = SPY ----------
spy = ycsv('SPY.csv')
CAL = spy.index

def ret(sym_df):
    """조정종가 일간 수익률 (해당 종목 자체 거래일 기준)."""
    return sym_df['adjclose'].pct_change()

def to_cal(s):
    """마스터 캘린더로 정렬. 종목 자체 거래일이 캘린더에 없으면 버림(수익률 누락 방지 위해 재계산)."""
    return s.reindex(CAL)

RAW = {k: ycsv(f'{k}.csv') for k in
       ['SPY','QQQ','SCHD','IEF','GLD','TQQQ','SGOV','IWD','DVY','VEIPX','VWNFX','NDX','GSPC']}

# 조정종가를 마스터 캘린더에 정렬 후 수익률 산출(캘린더 밖 거래일은 제외되므로 가격 기준 재계산)
ADJ = {k: to_cal(v['adjclose']) for k, v in RAW.items()}
RET = {k: ADJ[k].pct_change() for k in ADJ}

# ---------- 금리 시리즈 ----------
dtb3  = to_cal(fred('DTB3')).ffill()      # 3M T-bill, 할인율(discount basis), %
dgs7  = to_cal(fred('DGS7')).ffill()      # %
dgs10 = to_cal(fred('DGS10')).ffill()     # %
krw   = to_cal(fred('DEXKOUS')).ffill()   # USD/KRW

# 할인율 -> 투자수익률(BEY, act/365) 환산: P = 100*(1 - d*91/360)
d = dtb3/100.0
price_bill = 1.0 - d*91.0/360.0
bey = (1.0 - price_bill)/price_bill * 365.0/91.0     # 연율, act/365

# ---------- SGOV 프록시: 3M 국채 일할 수익 (달력일 경과분 accrual) ----------
days = pd.Series(CAL, index=CAL).diff().dt.days.astype('float')
sgov_proxy_ret = (bey.shift(1) * days / 365.0)        # 전일 금리로 당일까지 경과 이자
sgov_proxy_ret.iloc[0] = np.nan

# ---------- IEF 프록시: 8.5년 constant-maturity par bond 총수익 ----------
y85 = (dgs7 + (IEF_MAT-7.0)/(10.0-7.0)*(dgs10-dgs7))/100.0

def par_bond_price(y, c, M):
    """반기 쿠폰 채권 가격(액면 100). y,c: 연율 소수. M: 잔존연수(배열)."""
    n = np.maximum(np.ceil(M*2.0), 1.0)                    # 남은 쿠폰 횟수
    frac = n - M*2.0                                       # 다음 쿠폰까지 (반기 단위) 잔여
    i = y/2.0
    k = np.arange(1, int(np.nanmax(n))+1)[:, None]         # (K,1)
    mask = (k <= n[None, :])
    # 쿠폰 시점(반기 단위) = k - frac  (k=1..n), 원금 시점 = n - frac = M*2
    disc = np.where(mask, (1.0+i[None, :])**(-(k-frac[None, :])), 0.0)
    pv_cpn = (c*100.0/2.0) * disc.sum(axis=0)
    pv_prn = 100.0*(1.0+i)**(-(n-frac))
    return pv_cpn + pv_prn

yv = y85.to_numpy()
dt_yr = (days.to_numpy())/365.0                            # 경과 연수(달력일 기준)
c_prev = np.roll(yv, 1)                                    # 전일 par 쿠폰 = 전일 금리
M_now = IEF_MAT - dt_yr                                    # 하루 롤다운
P_now = par_bond_price(np.nan_to_num(yv, nan=0.03),
                       np.nan_to_num(c_prev, nan=0.03),
                       np.nan_to_num(M_now, nan=IEF_MAT))
# P_now 는 더티가격(경과이자 포함) 이므로 별도 accr 을 더하면 이중계상이 된다
ief_proxy_ret = pd.Series((P_now - 100.0)/100.0, index=CAL)
ief_proxy_ret = ief_proxy_ret.where(~(np.isnan(yv) | np.isnan(np.roll(yv,1)) | np.isnan(dt_yr)))
ief_proxy_ret.iloc[0] = np.nan
ief_proxy_ret = ief_proxy_ret - EXP_IEF*dt_yr              # 운용보수 차감

# ---------- GLD 프록시: LBMA PM fix - 운용보수 ----------
gold = pd.read_csv(D/'LBMA_gold_pm.csv', parse_dates=['date']).set_index('date')['usd_per_oz']
gold = to_cal(gold).ffill()
gld_proxy_ret = gold.pct_change() - EXP_GLD*dt_yr

# ---------- NDX 총수익 (QQQ 조정종가 + 보수 환원) ----------
ndx_tr_ret = RET['QQQ'] + EXP_QQQ*dt_yr

# ---------- TQQQ 프록시 ----------
fin = 2.0 * (bey.shift(1)) * (days/365.0)                  # 2배 차입 비용
tqqq_proxy_ret = 3.0*ndx_tr_ret - fin - EXP_TQQQ*dt_yr

# ---------- 시리즈 결합(splice): 실데이터 우선, 그 이전은 프록시 ----------
def splice(real_ret, proxy_ret, real_start):
    r = proxy_ret.copy()
    m = CAL >= real_start
    r[m] = real_ret[m]
    return r

def first_valid(sym):
    return RAW[sym].index[0]

SRC = {}   # 자산 -> (실데이터 시작일, 프록시 설명)
series = {}

series['SPY'] = RET['SPY'];  SRC['SPY'] = (first_valid('SPY'), '없음(전 구간 실데이터)')
series['QQQ'] = RET['QQQ'];  SRC['QQQ'] = (first_valid('QQQ'), '없음(전 구간 실데이터)')

ief_start = first_valid('IEF')
series['IEF'] = splice(RET['IEF'], ief_proxy_ret, ief_start)
SRC['IEF'] = (ief_start, 'FRED DGS7/DGS10 보간 8.5년 par bond 총수익 - 0.15%')

gld_start = first_valid('GLD')
series['GLD'] = splice(RET['GLD'], gld_proxy_ret, gld_start)
SRC['GLD'] = (gld_start, 'LBMA 금 PM fix - 0.40%')

tq_start = first_valid('TQQQ')
series['TQQQ'] = splice(RET['TQQQ'], tqqq_proxy_ret, tq_start)
SRC['TQQQ'] = (tq_start, '3x NDX총수익 - 2x3M금리 - 0.84%')

sg_start = first_valid('SGOV')
series['SGOV'] = splice(RET['SGOV'], sgov_proxy_ret, sg_start)
SRC['SGOV'] = (sg_start, '3M 국채 BEY 일할수익')

# SCHD 3종 프록시
schd_start = first_valid('SCHD')
pa = RET['SPY'].copy()
pb = RET['IWD'].copy(); pb[CAL < first_valid('IWD')] = RET['VWNFX'][CAL < first_valid('IWD')]
pc = RET['DVY'].copy(); pc[CAL < first_valid('DVY')] = RET['VEIPX'][CAL < first_valid('DVY')]
series['SCHD_a'] = splice(RET['SCHD'], pa, schd_start)
series['SCHD_b'] = splice(RET['SCHD'], pb, schd_start)
series['SCHD_c'] = splice(RET['SCHD'], pc, schd_start)
SRC['SCHD_a'] = (schd_start, 'SPY')
SRC['SCHD_b'] = (schd_start, f"IWD (>= {first_valid('IWD').date()}) / 그 이전 VWNFX")
SRC['SCHD_c'] = (schd_start, f"DVY (>= {first_valid('DVY').date()}) / 그 이전 VEIPX")

RETS = pd.DataFrame(series)
RETS.to_csv(OUT/'s1_returns.csv')

# 검증용 순수 프록시(실데이터 구현에도 프록시를 계속 계산한 것)
PROXY = pd.DataFrame({'TQQQ_syn': tqqq_proxy_ret, 'IEF_syn': ief_proxy_ret,
                      'GLD_syn': gld_proxy_ret, 'SGOV_syn': sgov_proxy_ret,
                      'NDX_TR': ndx_tr_ret})
PROXY.to_csv(OUT/'s1_proxy_returns.csv')
pd.DataFrame({'krw': krw, 'bey': bey, 'y85': y85, 'days': days}).to_csv(OUT/'s1_macro.csv')

meta = {k: dict(real_start=str(v[0].date()), proxy=v[1]) for k, v in SRC.items()}
json.dump(meta, open(LOG/'series_meta.json','w',encoding='utf-8'), indent=1, ensure_ascii=False)
print('calendar', CAL[0].date(), '~', CAL[-1].date(), 'n=', len(CAL))
print(RETS.notna().sum())
print('saved: out/s1_returns.csv, out/s1_proxy_returns.csv, out/s1_macro.csv')
