"""S5 그림 (명목 원화 — 한국 CPI 미확보)."""
import sys, pathlib, pickle, logging
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s5_lib as S
OUT = S.S5; FIG = OUT/'fig'
fm.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
plt.rcParams.update({'font.family': 'WenQuanYi Zen Hei', 'axes.unicode_minus': False,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#b5b4ad',
                     'axes.labelcolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
                     'grid.color': '#e6e5df'})
ORDER = ['P3', 'A80', 'C100', 'B-opt', 'B1', 'SPY', 'QQQ']
COL = {'P3': '#2a78d6', 'A80': '#eb6834', 'C100': '#008300', 'B-opt': '#e87ba4', 'B1': '#eda100',
       'SPY': '#4a3aa7', 'QQQ': '#e34948'}
LB = {'C100': '코어100', 'SPY': 'SPY100', 'QQQ': 'QQQ100'}
NOTE = '명목 원화(한국 CPI 미확보: 초기 1.5억·월 450만원 명목 고정). M1 배분, 원화 기준 과세, 원천징수 15%.'


def lab(k):
    return LB.get(k, k)


def end_labels(ax, ends, fmt, log=False):
    ys = sorted([(np.log10(v) if log else v, k, v) for k, v in ends.items()])
    lo, hi = ax.get_ylim()
    gap = ((np.log10(hi)-np.log10(lo)) if log else (hi-lo))*0.045
    ys = [list(t) for t in ys]
    for i in range(1, len(ys)):
        if ys[i][0]-ys[i-1][0] < gap:
            ys[i][0] = ys[i-1][0]+gap
    for y, k, v in ys:
        ax.annotate(f'{lab(k)} {fmt(v)}', (1.0, 10**y if log else y), xycoords=('axes fraction', 'data'),
                    xytext=(6, 0), textcoords='offset points', va='center', fontsize=8, annotation_clip=False)


def equity():
    P = pickle.load(open(OUT/'cache'/'fixed_paths.pkl', 'rb'))
    for per in ['2000-01~2026-09', '2000-01~2019-12', '2020-01~2026-09']:
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 8.6), sharex=True, gridspec_kw=dict(height_ratios=[3, 2]))
        ends, ends2 = {}, {}
        for k in ORDER:
            p = P[(k, per)]
            a1.plot(p['cal'], p['val']/1e8, color=COL[k], lw=2 if k in ('P3', 'A80') else 1.3, label=lab(k))
            a2.plot(p['cal'], (p['val']-p['paid'])/1e8, color=COL[k], lw=1.8 if k in ('P3', 'A80') else 1.1)
            ends[k] = p['val'][-1]/1e8; ends2[k] = (p['val'][-1]-p['paid'][-1])/1e8
        p = P[('P3', per)]
        a1.plot(p['cal'], p['paid']/1e8, color='#0b0b0b', lw=1.2, ls='--', label='누적 납입원금')
        a1.axhline(25, color='#52514e', lw=.9, ls=':')
        a1.annotate('25억', (p['cal'][0], 25), xytext=(2, 3), textcoords='offset points', fontsize=8, color='#52514e')
        a1.set_yscale('log'); a1.grid(True, axis='y', lw=.6); a1.set_ylabel('평가액 (억원, 로그)')
        lo, hi = a1.get_ylim(); a1.set_ylim(lo, hi*1.3)
        end_labels(a1, ends, lambda v: f'{v:,.1f}억', log=True)
        a1.legend(fontsize=8, frameon=False, ncol=4, loc='upper left')
        a1.set_title(f'적립식 평가액과 누적 원금 — {per}\n{NOTE}', fontsize=10, loc='left')
        a2.axhline(0, color='#0b0b0b', lw=.8)
        a2.grid(True, axis='y', lw=.6); a2.set_ylabel('원금 대비 평가손익 (억원)')
        end_labels(a2, ends2, lambda v: f'{v:+,.1f}억')
        fig.tight_layout(); fig.subplots_adjust(right=.84)
        fig.savefig(FIG/f'dca_equity_{per.replace("~", "_")}.png', dpi=150); plt.close(fig)


def box(ax, data, order, title, line=None):
    bp = ax.boxplot([data[k] for k in order], orientation='horizontal', widths=.55, whis=(10, 90),
                    patch_artist=True, showfliers=False, medianprops=dict(color='#0b0b0b', lw=1.6))
    for patch, k in zip(bp['boxes'], order):
        patch.set_facecolor('#2a78d6' if k in ('P3', 'A80') else '#cfe0f6'); patch.set_edgecolor('#52514e')
    for i, k in enumerate(order, 1):
        x = np.asarray(data[k])
        ax.plot([x.min(), x.max()], [i, i], ls='', marker='|', color='#8a8a86', ms=8)
    ax.set_yticks(range(1, len(order)+1)); ax.set_yticklabels([lab(k) for k in order], fontsize=9)
    ax.grid(True, axis='x', lw=.6)
    if line:
        ax.axvline(line, color='#e34948', lw=1.2, ls='--')
        ax.annotate('25억', (line, len(order)+.45), fontsize=8, color='#e34948', ha='left', xytext=(3, 0),
                    textcoords='offset points', annotation_clip=False)
    ax.set_title(title, fontsize=10, loc='left')


def rolling():
    r = pd.read_csv(OUT/'s5_dca_rolling.csv')
    r = r[r.배분 == 'M1']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, y in zip(axes, [14, 19]):
        x = r[r.보유년 == y]
        d = {k: x[x.key == k]['최종자산_명목']/1e8 for k in ORDER}
        order = sorted(ORDER, key=lambda k: d[k].median())
        box(ax, d, order, f'롤링 {y}년 최종자산 (억원) — 시작월 {len(x)//len(ORDER)}개', 25)
        ax.set_xlabel('최종자산 (억원, 명목)')
    fig.text(.01, .01, '상자 = 25~75%, 수염 = 10~90%, | = 최소·최대. 시작월 2000-01부터 매월(19년: ~2007-09, 14년: ~2012-09) — '
             '표본이 2000년대 초 시작에 집중. ' + NOTE, fontsize=7.5, color='#52514e')
    fig.tight_layout(rect=(0, .04, 1, 1)); fig.savefig(FIG/'dca_rolling_final.png', dpi=150); plt.close(fig)


def boot():
    b = pd.read_csv(OUT/'cache'/'boot_paths_G0.csv')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, y in zip(axes, [14, 19]):
        x = b[b.보유년 == y]
        d = {k: x[x.key == k]['최종자산']/1e8 for k in ORDER}
        order = sorted(ORDER, key=lambda k: d[k].median())
        box(ax, d, order, f'부트스트랩 {y}년 최종자산 (억원) — 1,000경로', 25)
        for i, k in enumerate(order, 1):
            p = (d[k] >= 25).mean()*100
            ax.annotate(f'P(≥25억) {p:.0f}%', (1.0, i), xycoords=('axes fraction', 'data'), xytext=(6, 0),
                        textcoords='offset points', va='center', fontsize=8, color='#0b0b0b', annotation_clip=False)
        ax.set_xscale('log'); ax.set_xlabel('최종자산 (억원, 명목, 로그)')
    fig.text(.01, .01, '블록 252거래일, 시드 20260917, 자산수익률·배당·환율 동일 블록 추출, 초기 환율 = 2026-09-16. ' + NOTE,
             fontsize=7.5, color='#52514e')
    fig.tight_layout(rect=(0, .04, 1, 1)); fig.subplots_adjust(right=.9, wspace=.45); fig.savefig(FIG/'dca_boot_final.png', dpi=150); plt.close(fig)


def gold():
    d = pd.read_csv(OUT/'s5_gold_lump.csv')
    g = pd.read_csv(OUT/'s5_gold_dca.csv')
    Gs = ['G0', 'G1', 'G2', 'G3']
    xl = ['G0 실제\n10.18%', 'G3 절반\n5.09%', 'G1 CPI+1\n3.58%', 'G2 CPI\n2.58%']
    go = ['G0', 'G3', 'G1', 'G2']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    x = d[d.구간 == 'C']
    ax = axes[0]
    for k in ORDER:
        y = [float(x[(x.G == G) & (x.key == k)]['Sharpe_세후'].iloc[0]) for G in go]
        ax.plot(range(4), y, marker='o', ms=6, color=COL[k], lw=2 if k in ('P3', 'A80', 'C100') else 1.2,
                mec='white', mew=1.5)
        ax.annotate(lab(k), (3, y[-1]), xytext=(8, 0), textcoords='offset points', va='center', fontsize=8,
                    annotation_clip=False)
    ax.set_xticks(range(4)); ax.set_xticklabels(xl, fontsize=8); ax.grid(True, axis='y', lw=.6)
    ax.set_ylabel('세후 Sharpe'); ax.set_title('일시투자 구간 C(2000–2026) 세후 Sharpe — 금 수익률별', fontsize=10, loc='left')
    ax = axes[1]
    x = g[g.구분.str.startswith('부트')]
    for k in ORDER:
        y = [float(x[(x.G == G) & (x.key == k)]['P_25억이상'].iloc[0]) for G in go]
        ax.plot(range(4), y, marker='o', ms=6, color=COL[k], lw=2 if k in ('P3', 'A80', 'C100') else 1.2,
                mec='white', mew=1.5)
        ax.annotate(lab(k), (3, y[-1]), xytext=(8, 0), textcoords='offset points', va='center', fontsize=8,
                    annotation_clip=False)
    ax.set_xticks(range(4)); ax.set_xticklabels(xl, fontsize=8); ax.grid(True, axis='y', lw=.6)
    ax.set_ylabel('P(최종 ≥ 25억) %'); ax.set_title('적립식 부트스트랩 19년 P(≥25억, 명목) — 금 수익률별', fontsize=10, loc='left')
    fig.text(.01, .01, '금 수익률 = 2000-01~2026-09 GLD(프록시 포함) 연율, 상수 일간 드리프트로 조정(변동성·상관 유지). 금 비보유 포트폴리오는 수평선.',
             fontsize=7.5, color='#52514e')
    fig.tight_layout(rect=(0, .04, 1, 1)); fig.subplots_adjust(wspace=.3, right=.93)
    fig.savefig(FIG/'gold_sensitivity_rank.png', dpi=150); plt.close(fig)


if __name__ == '__main__':
    equity(); rolling(); boot(); gold(); print('figs ok')
