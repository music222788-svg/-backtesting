"""S4 작업 2 그림 + 롤링 10년 분포(작업 2 포트폴리오).

산출: fig/equity_*.png, fig/drawdown_*.png, fig/roll10_box.png, s4_roll10_cs.csv
기준: 배당 원천징수 적용, 합성 기본(보정판은 표로 비교), 세전 자산곡선
"""
import sys, pathlib, pickle
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s4_lib as L
from s4_cs import PORTS, rets_for

S4 = L.S4
FIG = S4/'fig'
fm.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
plt.rcParams.update({'font.family': 'WenQuanYi Zen Hei', 'axes.unicode_minus': False,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#b5b4ad', 'axes.labelcolor': '#52514e',
                     'xtick.color': '#52514e', 'ytick.color': '#52514e', 'grid.color': '#e6e5df'})
# 고정 색 (엔티티 고정, 순서 고정 — 레퍼런스 팔레트 8색)
COL = {'P3': '#2a78d6', 'A80': '#eb6834', 'A-Q1': '#1baf7a', 'A-Q2': '#eda100', 'C': '#e87ba4',
       'B-opt': '#2a78d6', 'B1': '#eb6834', 'B2': '#1baf7a', 'B3': '#eda100',
       'SPY': '#4a3aa7', 'QQQ': '#e34948', 'C100': '#008300'}
GROUPS = {'coresat': ['P3', 'A80', 'A-Q1', 'A-Q2', 'C', 'SPY', 'C100'],
          'b': ['B-opt', 'B1', 'B2', 'B3', 'SPY', 'QQQ', 'C100']}
TITLE = {'coresat': '코어·위성 계열 (P3, A80, A-Q1, A-Q2, C) vs SPY·코어100',
         'b': 'B 계열 (B-opt, B1~B3) vs SPY·QQQ·코어100'}
LBL = {'C100': '코어100'}


def label_end(ax, series, keys, fmt):
    ends = sorted(((series[k].iloc[-1], k) for k in keys))
    ys = []
    lo, hi = ax.get_ylim()
    for v, k in ends:
        y = v
        if ax.get_yscale() == 'log':
            y = np.log10(v)
        ys.append([y, k, v])
    gap = (np.log10(hi)-np.log10(lo) if ax.get_yscale() == 'log' else hi-lo)*0.045
    for i in range(1, len(ys)):                      # 아래에서 위로 겹침 해소
        if ys[i][0]-ys[i-1][0] < gap:
            ys[i][0] = ys[i-1][0]+gap
    x = series[keys[0]].index[-1]
    for y, k, v in ys:
        yy = 10**y if ax.get_yscale() == 'log' else y
        ax.annotate(f'{LBL.get(k, k)} {fmt(v)}', (x, yy), xytext=(6, 0), textcoords='offset points',
                    va='center', fontsize=8, color='#0b0b0b', annotation_clip=False)


def main():
    eqs = pickle.load(open(S4/'cache'/'cs_equity.pkl', 'rb'))
    for per in ['A', 'B', 'C']:
        for g, keys in GROUPS.items():
            ser = {k: eqs[(True, False, k, per)][0] for k in keys}
            fig, ax = plt.subplots(figsize=(11, 5.6))
            for k in keys:
                ax.plot(ser[k].index, ser[k].values, lw=2 if k in ('P3', 'B-opt') else 1.4,
                        color=COL[k], label=LBL.get(k, k))
            ax.set_yscale('log')
            lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi*1.35)
            ax.grid(True, axis='y', lw=.6)
            ax.set_ylabel('평가액 (USD, 로그)')
            label_end(ax, ser, keys, lambda v: f'${v:,.0f}')
            ax.legend(loc='upper left', fontsize=8, frameon=False, ncol=4)
            ax.set_title(f'자산곡선 — {TITLE[g]} — 구간 {L.PER_LABEL[per]}\n'
                         '10,000 USD 일시투자, 세전, 배당 원천징수 15% 적용, 합성 레버리지 기본판',
                         fontsize=10, loc='left')
            fig.tight_layout(); fig.subplots_adjust(right=.84)
            fig.savefig(FIG/f'equity_{g}_{per}.png', dpi=150); plt.close(fig)

            fig, ax = plt.subplots(figsize=(11, 4.6))
            for k in keys:
                dd = (ser[k]/ser[k].cummax()-1)*100
                ax.plot(dd.index, dd.values, lw=1.8 if k in ('P3', 'B-opt') else 1.2, color=COL[k],
                        label=f'{LBL.get(k, k)} (MDD {dd.min():.1f}%)')
            ax.axhline(0, color='#b5b4ad', lw=.8)
            ax.grid(True, axis='y', lw=.6)
            ax.set_ylabel('고점 대비 낙폭 (%)')
            ax.legend(loc='lower left', fontsize=8, frameon=False, ncol=4)
            ax.set_title(f'낙폭 — {TITLE[g]} — 구간 {L.PER_LABEL[per]}', fontsize=10, loc='left')
            fig.tight_layout()
            fig.savefig(FIG/f'drawdown_{g}_{per}.png', dpi=150); plt.close(fig)

    # 롤링 10년 세후 CAGR (월별 시작) — 작업 2 포트폴리오
    starts = L.monthly_starts()
    rows = {}
    for k, (nm, sp, lev) in PORTS.items():
        rows[k] = L.rolling10_after_tax(sp, rets_for(lev, True, False), starts)*100
    RD = pd.DataFrame(rows, index=[s for s, _ in starts])
    RD.index.name = 'start'
    RD.to_csv(S4/'s4_roll10_cs.csv', encoding='utf-8-sig')
    order = RD.median().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(10, 6.2))
    bp = ax.boxplot([RD[k] for k in order], orientation='horizontal', widths=.55, whis=(0, 100), patch_artist=True,
                    medianprops=dict(color='#0b0b0b', lw=1.6))
    for patch, k in zip(bp['boxes'], order):
        patch.set_facecolor('#2a78d6' if k in ('P3', 'B-opt') else '#cfe0f6')
        patch.set_edgecolor('#52514e'); patch.set_linewidth(.8)
    for i, k in enumerate(order, start=1):
        x = RD[k]
        ax.annotate(f'최소 {x.min():.1f} / 10% {x.quantile(.1):.1f} / 중앙 {x.median():.1f}',
                    (x.max(), i), xytext=(6, 0), textcoords='offset points', va='center', fontsize=7.5,
                    color='#52514e')
    ax.set_yticks(range(1, len(order)+1)); ax.set_yticklabels([LBL.get(k, k) for k in order], fontsize=9)
    ax.axvline(0, color='#b5b4ad', lw=.8)
    ax.grid(True, axis='x', lw=.6)
    ax.set_xlabel('10년 세후 CAGR (%)')
    ax.set_title('롤링 10년 세후 CAGR 분포 — 시작월 2000-01 ~ 2016-09 (201개), 수염 = 최소~최대\n'
                 '배당 원천징수 적용, 합성 레버리지 기본판', fontsize=10, loc='left')
    fig.tight_layout(); fig.subplots_adjust(right=.78)
    fig.savefig(FIG/'roll10_box.png', dpi=150); plt.close(fig)
    print(RD.describe(percentiles=[.1, .5]).T.round(2).to_string())


if __name__ == '__main__':
    main()
