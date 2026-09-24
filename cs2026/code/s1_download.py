"""1단계-a: 원천 데이터 전량 재다운로드. 기존 ../data 는 사용하지 않는다."""
import requests, json, pathlib, time, datetime as dt
import pandas as pd

H = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
D = pathlib.Path(__file__).resolve().parent.parent / 'data'
L = pathlib.Path(__file__).resolve().parent.parent / 'logs'
D.mkdir(parents=True, exist_ok=True); L.mkdir(parents=True, exist_ok=True)
STAMP = dt.datetime.now().astimezone().isoformat(timespec='seconds')
log = []

def yahoo(sym, out):
    u = (f'https://query1.finance.yahoo.com/v8/finance/chart/{sym}'
         f'?period1=0&period2=9999999999&interval=1d&events=div%2Csplit')
    for a in range(5):
        try:
            r = requests.get(u, headers=H, timeout=60); r.raise_for_status(); j = r.json(); break
        except Exception as e:
            print('  retry', sym, repr(e)[:80]); time.sleep(4)
    else:
        raise RuntimeError('download failed: ' + sym)
    res = j['chart']['result'][0]; meta = res['meta']
    ts = res['timestamp']; q = res['indicators']['quote'][0]
    adj = res['indicators'].get('adjclose', [{}])[0].get('adjclose', q['close'])
    df = pd.DataFrame({
        'date': pd.to_datetime(ts, unit='s', utc=True).tz_convert('America/New_York').normalize().tz_localize(None),
        'close': q['close'], 'adjclose': adj, 'volume': q['volume']})
    df = df.dropna(subset=['adjclose']).drop_duplicates('date').set_index('date').sort_index()
    df.to_csv(D / out)
    ftd = meta.get('firstTradeDate')
    ftd = (dt.datetime(1970, 1, 1) + dt.timedelta(seconds=ftd)).date().isoformat() if ftd else None
    rec = dict(source='Yahoo Finance chart API v8', symbol=sym, file=out, rows=len(df),
               first=str(df.index[0].date()), last=str(df.index[-1].date()),
               yahoo_firstTradeDate=ftd, currency=meta.get('currency'), downloaded=STAMP)
    log.append(rec); print(f"  {sym:8s} n={len(df):6d} {rec['first']}~{rec['last']} ftd={ftd}")
    return rec

def fred(sid):
    u = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}'
    for a in range(5):
        try:
            r = requests.get(u, headers=H, timeout=60); r.raise_for_status(); break
        except Exception as e:
            print('  retry', sid, repr(e)[:80]); time.sleep(4)
    else:
        raise RuntimeError('download failed: ' + sid)
    p = D / f'FRED_{sid}.csv'
    p.write_bytes(r.content)
    df = pd.read_csv(p)
    df.columns = ['date', 'value']
    df['date'] = pd.to_datetime(df['date'])
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna()
    rec = dict(source='FRED (fredgraph.csv)', symbol=sid, file=p.name, rows=len(df),
               first=str(df['date'].iloc[0].date()), last=str(df['date'].iloc[-1].date()),
               downloaded=STAMP)
    log.append(rec); print(f"  {sid:10s} n={len(df):6d} {rec['first']}~{rec['last']}")
    return rec

def lbma():
    u = 'https://prices.lbma.org.uk/json/gold_pm.json'
    r = requests.get(u, headers=H, timeout=60); r.raise_for_status(); j = r.json()
    rows = [(x['d'], x['v'][0]) for x in j if x.get('v') and x['v'][0] is not None]
    df = pd.DataFrame(rows, columns=['date', 'usd_per_oz'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.drop_duplicates('date').set_index('date').sort_index()
    df.to_csv(D / 'LBMA_gold_pm.csv')
    rec = dict(source='LBMA (prices.lbma.org.uk/json/gold_pm.json), PM fix USD/oz',
               symbol='GOLD_PM', file='LBMA_gold_pm.csv', rows=len(df),
               first=str(df.index[0].date()), last=str(df.index[-1].date()), downloaded=STAMP)
    log.append(rec); print(f"  GOLD_PM  n={len(df):6d} {rec['first']}~{rec['last']}")
    return rec

if __name__ == '__main__':
    print('=== Yahoo ===')
    for sym, out in [('SPY','SPY.csv'), ('QQQ','QQQ.csv'), ('SCHD','SCHD.csv'), ('IEF','IEF.csv'),
                     ('GLD','GLD.csv'), ('TQQQ','TQQQ.csv'), ('SGOV','SGOV.csv'),
                     ('IWD','IWD.csv'), ('DVY','DVY.csv'), ('VEIPX','VEIPX.csv'),
                     ('VWNFX','VWNFX.csv'), ('^NDX','NDX.csv'), ('^GSPC','GSPC.csv')]:
        yahoo(sym, out)
    print('=== FRED ===')
    for s in ['DEXKOUS', 'DGS7', 'DGS10', 'DTB3']:
        fred(s)
    print('=== LBMA ===')
    lbma()
    json.dump(log, open(L / 'download_log.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print('\nlog ->', L / 'download_log.json')
