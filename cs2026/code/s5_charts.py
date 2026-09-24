"""5단계 그래프: 포트폴리오 A vs P3."""
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

KEYS = ['A1', 'A2', 'A3', 'P3', 'C1', 'B1', 'B2']
LBL = {'A1': '포트A ±12%p 연말점검', 'A2': '포트A ±12%p 상시점검',
       'A3': '포트A 최상위만 복원', 'P3': 'P3 코어70:위성30',
       'C1': 'C1 코어100', 'B1': 'B1 SPY 100%', 'B2': 'B2 QQQ 100%'}
COL = {'A1': '#e6550d', 'A2': '#fd8d3c', 'A3': '#fdbe85', 'P3': '#d62728',
       'C1': '#8c564b', 'B1': '#2ca02c', 'B2': '#9467bd'}
PERNAME = {'A': '구간 A (2000-01-03 ~ 2019-12-31)',
           'B': '구간 B (2020-01-02 ~ 2026-09-16)',
           'C': '구간 C (2000-01-03 ~ 2026-09-16)'}

R = pd.read_csv(OUT/'s1_returns.csv', parse_dates=['date']).set_index('date')
MAC = pd.read_csv(OUT/'s1_macro.csv', parse_dates=['date']).set_index('date')
FX = MAC['krw']


def eq(per, suf=''):
    return pd.read_csv(OUT/f's5_equity_{per}{suf}.csv', parse_dates=['date']).set_index('date')


for per in ['A', 'B', 'C']:
    d = eq(per)
    fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                           gridspec_kw={'height_ratios': [2, 1]})
    for k in KEYS:
        lw = 2.0 if k in ('A1', 'P3') else 1.1
        ax[0].plot(d.index, d[k], lw=lw, color=COL[k], label=LBL[k])
        ax[1].plot(d.index, M.drawdown(d[k])*100, lw=lw, color=COL[k])
    ax[0].set_yscale('log')
    ax[0].set_title(f'포트폴리오 A vs P3 — {PERNAME[per]}  (10,000 USD 일시투자, 세전)')
    ax[0].set_ylabel('USD (로그축)'); ax[0].grid(alpha=.3, which='both')
    ax[0].legend(fontsize=8, ncol=2)
    ax[1].set_ylabel('낙폭 %'); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG/f's5_equity_{per}.png'); plt.close(fig)

# 위험자산 비중 추이 + 밴드
fig, axes = plt.subplots(2, 1, figsize=(12, 7.5))
for ax, per, rng in zip(axes, ['A', 'B'],
                        [('2000-01-03', '2019-12-31'), ('2020-01-02', '2026-09-16')]):
    for k, ls in [('A1', '-'), ('A2', '--')]:
        r = E.run_one(k, R, FX, rng[0], rng[1])
        h = r['holdings']
        w = (h[['SCHD', 'TQQQ']].sum(axis=1)/h.sum(axis=1))*100
        ax.plot(w.index, w, lw=1.2, ls=ls, color=COL[k], label=LBL[k])
        for dte in r['trades']['date']:
            ax.axvline(dte, color=COL[k], alpha=.35, lw=.9, ls=':')
    ax.axhline(40, color='k', lw=1.0, label='목표 40%')
    ax.axhline(52, color='r', lw=1.0, ls='--', label='밴드 상단 52%')
    ax.axhline(28, color='r', lw=1.0, ls='--', label='밴드 하단 28%')
    ax.set_ylabel('위험자산(SCHD+TQQQ) 비중 %'); ax.grid(alpha=.3)
    ax.set_title(f'포트폴리오 A 의 위험자산 비중과 ±12%p 밴드 — {PERNAME[per]} (세로 점선 = 체결일)')
    ax.legend(fontsize=7.5, ncol=3)
fig.tight_layout(); fig.savefig(FIG/'s5_risk_weight.png'); plt.close(fig)

# 금 할인 스윕
g = pd.read_csv(OUT/'s5_gold_sweep.csv')
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, per in zip(axes, ['A', 'C']):
    x = g[g['구간'] == per].pivot(index='cut', columns='key', values='CAGR')
    for k in ['A1', 'A2', 'A3', 'P3', 'C1']:
        ax.plot(x.index, x[k], marker='o', ms=4, color=COL[k], label=LBL[k],
                lw=2.0 if k in ('A1', 'P3') else 1.1)
    ax.set_xlabel('금 수익률 연 할인폭 (pp)'); ax.set_ylabel('CAGR %')
    ax.set_title(f'금 수익률을 깎았을 때 — 구간 {per}')
    ax.grid(alpha=.3); ax.legend(fontsize=7.5)
fig.tight_layout(); fig.savefig(FIG/'s5_gold_sweep.png'); plt.close(fig)

# 구간 A 시작일 분포
sd = pd.read_csv(OUT/'s5_startdep_A.csv')
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, yrs in zip(axes, [5, 10]):
    sub = sd[sd['years'] == yrs]
    data = [sub[sub.key == k]['cagr'].values for k in KEYS]
    bp = ax.boxplot(data, tick_labels=KEYS, patch_artist=True, whis=(10, 90), showfliers=True)
    for p, k in zip(bp['boxes'], KEYS):
        p.set_facecolor(COL[k]); p.set_alpha(.6)
    ax.axhline(0, color='k', lw=.8); ax.grid(alpha=.3, axis='y')
    ax.set_ylabel('CAGR %')
    ax.set_title(f'구간 A · 매월 시작 {yrs}년 보유 CAGR ({sub["start"].nunique()}개 시작월)')
fig.tight_layout(); fig.savefig(FIG/'s5_startdep_A.png'); plt.close(fig)

# TQQQ=0
for per in ['A', 'C']:
    a, b = eq(per), eq(per, '_tqqq0')
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for k in ['A1', 'P3', 'C1']:
        ax.plot(a.index, a[k], lw=1.8, color=COL[k], label=f'{LBL[k]} 기본')
        ax.plot(b.index, b[k], lw=1.2, color=COL[k], ls='--', label=f'{LBL[k]} TQQQ=0')
    ax.set_yscale('log'); ax.grid(alpha=.3, which='both')
    ax.set_title(f'TQQQ 소멸 시나리오 — {PERNAME[per]}')
    ax.set_ylabel('USD (로그축)'); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(FIG/f's5_tqqq0_{per}.png'); plt.close(fig)

if (OUT/'s5_bootstrap_A.csv').exists():
    for per in ['A', 'C']:
        p = OUT/f's5_bootstrap_{per}.csv'
        if not p.exists():
            continue
        d = pd.read_csv(p)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, sfx, t in zip(axes, ['_cagr', '_mdd'], ['CAGR %', 'MDD %']):
            bp = ax.boxplot([d[f'{k}{sfx}'].values*100 for k in KEYS], tick_labels=KEYS,
                            patch_artist=True, whis=(10, 90), showfliers=False)
            for b_, k in zip(bp['boxes'], KEYS):
                b_.set_facecolor(COL[k]); b_.set_alpha(.6)
            ax.set_ylabel(t); ax.grid(alpha=.3, axis='y')
        axes[0].axhline(0, color='k', lw=.8)
        axes[0].set_title(f'블록 부트스트랩 CAGR — 구간 {per} (블록 1년, 1,000회)')
        axes[1].set_title(f'블록 부트스트랩 MDD — 구간 {per}')
        fig.tight_layout(); fig.savefig(FIG/f's5_bootstrap_{per}.png'); plt.close(fig)

print('saved ->', sorted(p.name for p in FIG.glob('s5_*.png')))
