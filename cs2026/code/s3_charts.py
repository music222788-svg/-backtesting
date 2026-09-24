"""3단계-d: 그래프 저장 (자산곡선, 낙폭곡선, 연간수익률, 민감도 분포)."""
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
FIG.mkdir(exist_ok=True)
import metrics as M

for f in ['Malgun Gothic', 'NanumGothic', 'Gulim', 'Batang']:
    if any(f == ff.name for ff in font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = f
        break
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 110

KEYS = ['P1', 'P2', 'P3', 'P4', 'B1', 'B2', 'C1']
LBL = {'P1': 'P1 코어80:위성20 복원', 'P2': 'P2 코어80:위성20 무복원(30%상한)',
       'P3': 'P3 코어70:위성30 복원', 'P4': 'P4 코어70:위성30 무복원',
       'B1': 'B1 SPY 100%', 'B2': 'B2 QQQ 100%', 'C1': 'C1 코어 100%'}
COL = {'P1': '#1f77b4', 'P2': '#17becf', 'P3': '#d62728', 'P4': '#ff7f0e',
       'B1': '#2ca02c', 'B2': '#9467bd', 'C1': '#8c564b'}
PERNAME = {'A': '구간 A (2000-01-03 ~ 2019-12-31)',
           'B': '구간 B (2020-01-02 ~ 2026-09-16)',
           'C': '구간 C (2000-01-03 ~ 2026-09-16)'}


def load(p, suffix=''):
    return pd.read_csv(OUT/f's3_equity_{p}{suffix}.csv', parse_dates=['date']).set_index('date')


def fig_equity(per, suffix='', tag=''):
    eq = load(per, suffix)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for k in KEYS:
        ax.plot(eq.index, eq[k], lw=1.3, color=COL[k], label=LBL[k])
    ax.set_yscale('log')
    ax.set_title(f'누적 자산 곡선 — {PERNAME[per]}{tag}   (10,000 USD 일시투자, 세전)')
    ax.set_ylabel('USD (로그축)')
    ax.grid(alpha=.3, which='both')
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG/f'equity_{per}{suffix}.png')
    plt.close(fig)


def fig_drawdown(per, suffix='', tag=''):
    eq = load(per, suffix)
    fig, ax = plt.subplots(figsize=(11, 5))
    for k in KEYS:
        dd = M.drawdown(eq[k])*100
        ax.plot(dd.index, dd, lw=1.1, color=COL[k], label=LBL[k])
    ax.set_title(f'낙폭(drawdown) 곡선 — {PERNAME[per]}{tag}')
    ax.set_ylabel('%')
    ax.grid(alpha=.3)
    ax.legend(fontsize=8, ncol=2, loc='lower left')
    fig.tight_layout()
    fig.savefig(FIG/f'drawdown_{per}{suffix}.png')
    plt.close(fig)


def fig_annual(per):
    eq = load(per)
    ar = pd.DataFrame({k: M.annual_returns(eq[k]) for k in KEYS})*100
    fig, ax = plt.subplots(figsize=(max(11, len(ar)*0.62), 5))
    x = np.arange(len(ar))
    w = 0.115
    for i, k in enumerate(KEYS):
        ax.bar(x+(i-3)*w, ar[k], width=w, color=COL[k], label=LBL[k])
    ax.set_xticks(x)
    ax.set_xticklabels(ar.index, rotation=0, fontsize=8)
    ax.axhline(0, color='k', lw=.8)
    ax.set_title(f'연간 수익률 — {PERNAME[per]} (세전)')
    ax.set_ylabel('%')
    ax.grid(alpha=.3, axis='y')
    ax.legend(fontsize=7.5, ncol=4)
    fig.tight_layout()
    fig.savefig(FIG/f'annual_{per}.png')
    plt.close(fig)


def fig_tqqq0(per):
    a = load(per)
    b = load(per, '_tqqq0')
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for k in ['P1', 'P3', 'B1', 'C1']:
        ax.plot(a.index, a[k], lw=1.5, color=COL[k], label=f'{k} 기본')
        ax.plot(b.index, b[k], lw=1.2, color=COL[k], ls='--', label=f'{k} TQQQ=0')
    ax.set_yscale('log')
    ax.set_title(f'합성 3배 소멸(2001-08-30 이후 TQQQ=0) 시나리오 비교 — {PERNAME[per]}')
    ax.set_ylabel('USD (로그축)')
    ax.grid(alpha=.3, which='both')
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG/f'tqqq0_{per}.png')
    plt.close(fig)


def fig_startdep():
    d = pd.read_csv(OUT/'s3_startdep_A.csv')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, yr in zip(axes, [5, 10]):
        sub = d[d['years'] == yr]
        data = [sub[sub.key == k]['cagr'].values*100 for k in KEYS]
        bp = ax.boxplot(data, tick_labels=KEYS, showfliers=True, patch_artist=True, whis=(10, 90))
        for p, k in zip(bp['boxes'], KEYS):
            p.set_facecolor(COL[k]); p.set_alpha(.55)
        ax.axhline(0, color='k', lw=.8)
        ax.set_title(f'구간 A · 매월 시작 {yr}년 보유 CAGR 분포 '
                     f'({sub["start"].nunique()}개 시작월)')
        ax.set_ylabel('CAGR %')
        ax.grid(alpha=.3, axis='y')
    fig.tight_layout()
    fig.savefig(FIG/'startdep_A.png')
    plt.close(fig)

    b = pd.read_csv(OUT/'s3_startdep_B.csv')
    piv = b.pivot(index='start', columns='key', values='cagr')*100
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(piv))
    for k in KEYS:
        ax.plot(x, piv[k], marker='o', ms=4, color=COL[k], label=LBL[k])
    ax.set_xticks(x); ax.set_xticklabels([s[:7] for s in piv.index], rotation=45, fontsize=8)
    ax.set_title('구간 B 시작월별 CAGR (종료 2026-09-16 고정, 세전)')
    ax.set_ylabel('CAGR %')
    ax.grid(alpha=.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG/'startdep_B.png')
    plt.close(fig)


def fig_bootstrap():
    for per in ['A', 'C']:
        p = OUT/f's3_bootstrap_{per}.csv'
        if not p.exists():
            continue
        d = pd.read_csv(p)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].boxplot([d[f'{k}_cagr'].values*100 for k in KEYS], tick_labels=KEYS,
                        showfliers=False, patch_artist=True, whis=(10, 90))
        axes[1].boxplot([d[f'{k}_mdd'].values*100 for k in KEYS], tick_labels=KEYS,
                        showfliers=False, patch_artist=True, whis=(10, 90))
        for ax, t in zip(axes, ['CAGR %', 'MDD %']):
            ax.grid(alpha=.3, axis='y'); ax.set_ylabel(t)
        axes[0].axhline(0, color='k', lw=.8)
        axes[0].set_title(f'블록 부트스트랩 CAGR 분포 — 구간 {per} (블록 1년, 1,000회)')
        axes[1].set_title(f'블록 부트스트랩 MDD 분포 — 구간 {per}')
        fig.tight_layout()
        fig.savefig(FIG/f'bootstrap_{per}.png')
        plt.close(fig)


if __name__ == '__main__':
    for per in ['A', 'B', 'C']:
        fig_equity(per); fig_drawdown(per); fig_annual(per)
    for per in ['A', 'C']:
        if (OUT/f's3_equity_{per}_tqqq0.csv').exists():
            fig_equity(per, '_tqqq0', ' · TQQQ=0 시나리오')
            fig_drawdown(per, '_tqqq0', ' · TQQQ=0 시나리오')
            fig_tqqq0(per)
    fig_startdep()
    fig_bootstrap()
    print('saved figs ->', sorted(p.name for p in FIG.glob('*.png')))
