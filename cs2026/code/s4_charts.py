"""4단계 그래프: P3 vs /stock V4."""
import sys, pathlib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent
OUT, FIG = ROOT/'out', ROOT/'fig'
import engine as E
import metrics as M

for f in ['Malgun Gothic', 'NanumGothic', 'Gulim']:
    if any(f == ff.name for ff in font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = f
        break
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 110

KEYS = ['P3', 'S1', 'S2', 'C1', 'B1', 'B2']
LBL = {'P3': 'P3 코어70:위성30 복원', 'S1': '/stock V4 (연말점검)', 'S2': '/stock V4 (상시점검)',
       'C1': 'C1 코어 100%', 'B1': 'B1 SPY 100%', 'B2': 'B2 QQQ 100%'}
COL = {'P3': '#d62728', 'S1': '#1f77b4', 'S2': '#17becf',
       'C1': '#8c564b', 'B1': '#2ca02c', 'B2': '#9467bd'}
PERNAME = {'A': '구간 A (2000-01-03 ~ 2019-12-31)',
           'B': '구간 B (2020-01-02 ~ 2026-09-16)',
           'C': '구간 C (2000-01-03 ~ 2026-09-16)'}

R = pd.read_csv(OUT/'s1_returns.csv', parse_dates=['date']).set_index('date')
MAC = pd.read_csv(OUT/'s1_macro.csv', parse_dates=['date']).set_index('date')
FX = MAC['krw']


def eq(per, suf=''):
    return pd.read_csv(OUT/f's4_equity_{per}{suf}.csv', parse_dates=['date']).set_index('date')


for per in ['A', 'B', 'C']:
    d = eq(per)
    fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                           gridspec_kw={'height_ratios': [2, 1]})
    for k in KEYS:
        lw = 1.9 if k in ('P3', 'S1') else 1.1
        ax[0].plot(d.index, d[k], lw=lw, color=COL[k], label=LBL[k])
        ax[1].plot(d.index, M.drawdown(d[k])*100, lw=lw, color=COL[k])
    ax[0].set_yscale('log')
    ax[0].set_title(f'P3 vs /stock V4 — {PERNAME[per]}  (10,000 USD 일시투자, 세전)')
    ax[0].set_ylabel('USD (로그축)'); ax[0].grid(alpha=.3, which='both')
    ax[0].legend(fontsize=8, ncol=2)
    ax[1].set_ylabel('낙폭 %'); ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIG/f's4_equity_{per}.png'); plt.close(fig)

for per in ['A', 'C']:
    a, b = eq(per), eq(per, '_tqqq0')
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for k in ['P3', 'S1', 'C1']:
        ax.plot(a.index, a[k], lw=1.7, color=COL[k], label=f'{LBL[k]} 기본')
        ax.plot(b.index, b[k], lw=1.2, color=COL[k], ls='--', label=f'{LBL[k]} TQQQ=0')
    ax.set_yscale('log')
    ax.set_title(f'TQQQ 소멸(2001-08-30 이후 0) 시나리오 — {PERNAME[per]}')
    ax.set_ylabel('USD (로그축)'); ax.grid(alpha=.3, which='both')
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(FIG/f's4_tqqq0_{per}.png'); plt.close(fig)

# /stock 의 TQQQ 비중 추이와 밴드
fig, axes = plt.subplots(2, 1, figsize=(12, 7.5))
for ax, per, rng in zip(axes, ['A', 'B'],
                        [('2000-01-03', '2019-12-31'), ('2020-01-02', '2026-09-16')]):
    for k, ls in [('S1', '-'), ('S2', '--')]:
        r = E.run_one(k, R, FX, rng[0], rng[1])
        h = r['holdings']
        w = (h['TQQQ']/h.sum(axis=1))*100
        ax.plot(w.index, w, lw=1.2, ls=ls, color=COL[k], label=LBL[k])
        for dte in r['trades']['date']:
            ax.axvline(dte, color=COL[k], alpha=.35, lw=.9, ls=':')
    ax.axhline(40, color='k', lw=1.0, label='목표 40%')
    ax.axhline(70, color='r', lw=1.0, ls='--', label='밴드 상단 70%')
    ax.axhline(10, color='r', lw=1.0, ls='--', label='밴드 하단 10%')
    ax.set_ylabel('TQQQ 비중 %'); ax.grid(alpha=.3)
    ax.set_title(f'/stock V4 의 TQQQ 실제 비중과 ±30%p 밴드 — {PERNAME[per]} '
                 f'(점선 세로줄 = 체결일)')
    ax.legend(fontsize=7.5, ncol=3)
fig.tight_layout(); fig.savefig(FIG/'s4_stock_weight.png'); plt.close(fig)

# 구간 B 시작월 의존성
sd = pd.read_csv(OUT/'s4_startdep_B.csv')
piv = sd.pivot(index='start', columns='key', values='cagr')[KEYS]
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(piv))
for k in KEYS:
    lw = 2.0 if k in ('P3', 'S1') else 1.1
    ax.plot(x, piv[k], marker='o', ms=4, lw=lw, color=COL[k], label=LBL[k])
ax.set_xticks(x); ax.set_xticklabels([s[:7] for s in piv.index], rotation=45, fontsize=8)
ax.set_title('구간 B 시작월별 CAGR — P3 vs /stock V4 (종료 2026-09-16 고정, 세전)')
ax.set_ylabel('CAGR %'); ax.grid(alpha=.3); ax.legend(fontsize=8, ncol=2)
fig.tight_layout(); fig.savefig(FIG/'s4_startdep_B.png'); plt.close(fig)

print('saved ->', sorted(p.name for p in FIG.glob('s4_*.png')))
