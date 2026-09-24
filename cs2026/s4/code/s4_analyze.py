"""S4 작업 1 분석: 표, 안정 영역, 표본 외 검증, 세분화 목록, B-opt, 히트맵.

python s4_analyze.py stage1   -> 66개로 안정 영역 판정, cache/refine_list.json 작성
python s4_analyze.py stage2   -> 세분화 포함 최종 판정, B-opt 확정(cache/bopt.json),
                                 s4_grid.csv / s4_oos.csv / fig 히트맵
안정 영역 기준 (사용자 지시: 구간 A·B·C, 롤링 10년 최솟값, 부트스트랩 하위 10% CAGR)
  - 구간 A·B·C: 세후 Sharpe (세후 자산곡선, 무위험 = 3M 국채)  [주 기준]
  - 롤링 10년: 월별 시작 10년 세후 CAGR 의 최솟값
  - 부트스트랩: 구간 A, 구간 C 각각의 CAGR 하위 10% (둘 다 요구)
  => 6개 기준 모두 66개 중 상위 1/3(22위 이내)
  민감도: 구간 A·B·C 기준을 세후 CAGR 로 바꾼 판정도 병기
세분화 조합은 66개 기준으로 정한 '상위 1/3 경계값'을 동일하게 적용해 판정한다.
"""
import sys, pathlib, pickle, json, itertools
import numpy as np, pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s4_lib as L

S4 = L.S4
CACHE = S4/'cache'/'grid'
FIG = S4/'fig'
PERIODS = ['A', 'B', 'C', 'S1', 'S2', 'S3', 'OOS2']
REF_B = {'B1': (.2, .3, .5), 'B2': (.1, .3, .6), 'B3': (.3, .3, .4)}
MAIN = [('A', 'Sharpe_세후'), ('B', 'Sharpe_세후'), ('C', 'Sharpe_세후')]
ALT = [('A', 'CAGR_세후'), ('B', 'CAGR_세후'), ('C', 'CAGR_세후')]


def load():
    recs = [pickle.load(open(f, 'rb')) for f in sorted(CACHE.glob('*.pkl'))]
    long, wide = [], []
    for r in recs:
        w = r['w']
        base = {'SCHD': w['SCHD'], 'SPY': w['SPY'], 'QQQ': w['QQQ'], 'key': r['key'],
                'grid': '10%p' if all(abs(w[a]*10-round(w[a]*10)) < 1e-9 for a in w) else '5%p'}
        row = dict(base)
        for p in PERIODS:
            d = r['per'][p]
            long.append({**base, '구간': p, '구간명': L.PER_LABEL[p], **d})
            for m in ['CAGR_세전', 'CAGR_세후', 'CAGR_원화', 'Sharpe', 'Sharpe_세후', 'MDD',
                      'Calmar', '연변동성', '최장회복_거래일', '최악의연도']:
                row[f'{p}_{m}'] = d[m]
        rr = r['roll10']
        row.update(roll10_min=rr.min(), roll10_p10=np.quantile(rr, .10), roll10_med=np.median(rr),
                   roll10_n=len(rr))
        for p in ['A', 'C']:
            cg, md = r[f'boot_{p}']
            row.update({f'boot{p}_cagr_med': np.median(cg), f'boot{p}_cagr_p10': np.quantile(cg, .10),
                        f'boot{p}_mdd_med': np.median(md), f'boot{p}_mdd_p10': np.quantile(md, .10)})
        long.append({**base, '구간': 'ROLL10', '구간명': '롤링10년(세후, 월별시작)',
                     'roll10_min': rr.min(), 'roll10_p10': np.quantile(rr, .10), 'roll10_med': np.median(rr),
                     'roll10_n': len(rr)})
        for p in ['A', 'C']:
            cg, md = r[f'boot_{p}']
            long.append({**base, '구간': f'BOOT_{p}', '구간명': f'부트스트랩 {p} (블록252, 1000회)',
                         'boot_cagr_med': np.median(cg), 'boot_cagr_p10': np.quantile(cg, .10),
                         'boot_mdd_med': np.median(md), 'boot_mdd_p10': np.quantile(md, .10)})
        wide.append(row)
    return pd.DataFrame(long), pd.DataFrame(wide), {r['key']: r for r in recs}


def criteria(crit_abc):
    return [(f'{p}_{m}', True) for p, m in crit_abc] + [('roll10_min', True),
                                                        ('bootA_cagr_p10', True), ('bootC_cagr_p10', True)]


def stable(W, crit_abc, tag):
    base = W[W.grid == '10%p']
    n_top = int(np.ceil(len(base)/3))            # 66 -> 22
    ok = pd.Series(True, index=W.index)
    thr = {}
    for c, hi in criteria(crit_abc):
        cut = base[c].sort_values(ascending=not hi).iloc[n_top-1]
        thr[c] = cut
        W[f'rank_{c}'] = W[c].rank(ascending=not hi, method='min')
        W[f'rank10_{c}'] = base[c].rank(ascending=not hi, method='min').reindex(W.index)
        ok &= (W[c] >= cut) if hi else (W[c] <= cut)
    W[f'stable_{tag}'] = ok
    return thr


def neighbors5(W):
    st = W[(W.grid == '10%p') & W.stable_main]
    pts = set()
    for s, q in itertools.product(range(0, 21), range(0, 21)):
        if s+q > 20:
            continue
        w = np.array([s, 20-s-q, q])/20
        for _, r in st.iterrows():
            if np.max(np.abs(w-np.array([r.SCHD, r.SPY, r.QQQ]))) <= 0.05+1e-9:
                if not (s % 2 == 0 and q % 2 == 0):
                    pts.add((s, q))
                break
    return [{'SCHD': s/20, 'SPY': (20-s-q)/20, 'QQQ': q/20} for s, q in sorted(pts)]


def oos(W):
    base = W[W.grid == '10%p'].copy()
    out = []
    for sel, val in [('S1', 'OOS2'), ('OOS2', 'S1')]:
        base[f'rk_{sel}'] = base[f'{sel}_Sharpe_세후'].rank(ascending=False, method='min')
        base[f'rk_{val}'] = base[f'{val}_Sharpe_세후'].rank(ascending=False, method='min')
        top = base.sort_values(f'rk_{sel}').head(5)
        for _, r in top.iterrows():
            out.append(dict(선택구간=L.PER_LABEL[sel], 검증구간=L.PER_LABEL[val], key=r.key,
                            SCHD=r.SCHD, SPY=r.SPY, QQQ=r.QQQ,
                            선택구간_세후Sharpe=r[f'{sel}_Sharpe_세후'], 선택구간_순위=int(r[f'rk_{sel}']),
                            검증구간_세후Sharpe=r[f'{val}_Sharpe_세후'], 검증구간_순위=int(r[f'rk_{val}']),
                            검증구간_CAGR_세후=r[f'{val}_CAGR_세후'], 검증구간_MDD=r[f'{val}_MDD'],
                            검증구간_Sharpe_중앙값_66개=base[f'{val}_Sharpe_세후'].median()))
    rho = base['S1_Sharpe_세후'].rank().corr(base['OOS2_Sharpe_세후'].rank())
    rho_c = base['S1_CAGR_세후'].rank().corr(base['OOS2_CAGR_세후'].rank())
    return pd.DataFrame(out), rho, rho_c


# ---------------- 그림 ----------------
def heatmaps(W, bopt=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    fm.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
    plt.rcParams['font.family'] = 'WenQuanYi Zen Hei'
    plt.rcParams['axes.unicode_minus'] = False
    base = W[W.grid == '10%p']
    specs = [('CAGR_세후', '세후 CAGR (%)', 100, 'Blues', True),
             ('Sharpe_세후', '세후 Sharpe', 1, 'Blues', True),
             ('MDD', 'MDD (%)  — 진할수록 낙폭이 작음', 100, 'Blues', True)]
    fname = {'CAGR_세후': 'heatmap_cagr_aftertax.png', 'Sharpe_세후': 'heatmap_sharpe_aftertax.png',
             'MDD': 'heatmap_mdd.png'}

    def panel(ax, val, title, cmap, fmt, vmin, vmax):
        Z = np.full((11, 11), np.nan)
        for _, r in base.iterrows():
            Z[int(round(r.QQQ*10)), int(round(r.SCHD*10))] = val[r.name]
        im = ax.imshow(Z, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, extent=(-5, 105, -5, 105))
        for _, r in base.iterrows():
            v = val[r.name]
            x, y = r.SCHD*100, r.QQQ*100
            dark = (v-vmin)/(vmax-vmin+1e-12) > .6
            ax.text(x, y, fmt(v), ha='center', va='center', fontsize=6.3,
                    color='white' if dark else '#222222')
            if r.stable_main:
                ax.add_patch(plt.Rectangle((x-5, y-5), 10, 10, fill=False, ec='#eb6834', lw=1.6))
        for k, (s, p, q) in REF_B.items():
            ax.plot(s*100, q*100, marker='o', ms=13, mfc='none', mec='#111111', mew=1.2)
            ax.annotate(k, (s*100, q*100), xytext=(7, 7), textcoords='offset points', fontsize=8,
                        color='#111111', fontweight='bold')
        if bopt:
            ax.plot(bopt['SCHD']*100, bopt['QQQ']*100, marker='*', ms=15, mfc='#eb6834', mec='white', mew=1)
        ax.set_xlim(-5, 105); ax.set_ylim(-5, 105)
        ax.set_xticks(range(0, 101, 10)); ax.set_yticks(range(0, 101, 10))
        ax.tick_params(labelsize=7)
        ax.set_xlabel('SCHD 비중 (%)', fontsize=8); ax.set_ylabel('QQQ 비중 (%)', fontsize=8)
        ax.set_title(title, fontsize=9)
        for sp in ax.spines.values():
            sp.set_visible(False)
        return im

    note = ('SPY = 100 − SCHD − QQQ.  주황 테두리 = 안정 영역(6개 기준 모두 상위 1/3).  '
            '○ = 기존 B 후보.  ★ = B-opt.  원천징수 15% 적용, 10%p 격자 66개.')
    for m, lab, mul, cmap, hi in specs:
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))
        vals = {p: base[f'{p}_{m}']*mul for p in 'ABC'}
        for ax, p in zip(axes, 'ABC'):
            v = vals[p]
            im = panel(ax, v, f'구간 {L.PER_LABEL[p]}', cmap,
                       (lambda x: f'{x:.2f}') if mul == 1 else (lambda x: f'{x:.1f}'), v.min(), v.max())
            fig.colorbar(im, ax=ax, fraction=.046, pad=.03).ax.tick_params(labelsize=7)
        fig.suptitle(f'SCHD×QQQ 히트맵 — {lab}', fontsize=12, x=.01, ha='left')
        fig.text(.01, .01, note, fontsize=8, color='#52514e')
        fig.tight_layout(rect=(0, .04, 1, .95))
        fig.savefig(FIG/fname[m], dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    for ax, (c, t) in zip(axes, [('roll10_min', '롤링 10년 세후 CAGR 최솟값 (%)'),
                                 ('roll10_med', '롤링 10년 세후 CAGR 중앙값 (%)')]):
        v = base[c]*100
        im = panel(ax, v, t, 'Blues', lambda x: f'{x:.1f}', v.min(), v.max())
        fig.colorbar(im, ax=ax, fraction=.046, pad=.03).ax.tick_params(labelsize=7)
    fig.suptitle('SCHD×QQQ 히트맵 — 롤링 10년 (시작월 2000-01 ~ 2016-09, 201개)', fontsize=12, x=.01, ha='left')
    fig.text(.01, .01, note, fontsize=8, color='#52514e')
    fig.tight_layout(rect=(0, .04, 1, .95))
    fig.savefig(FIG/'heatmap_roll10.png', dpi=150); plt.close(fig)


if __name__ == '__main__':
    stage = sys.argv[1]
    LONG, W, RECS = load()
    thr = stable(W, MAIN, 'main')
    stable(W, ALT, 'alt')
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    cols = ['key', 'grid', 'A_Sharpe_세후', 'B_Sharpe_세후', 'C_Sharpe_세후', 'roll10_min',
            'bootA_cagr_p10', 'bootC_cagr_p10', 'C_CAGR_세후', 'C_MDD', 'stable_main', 'stable_alt']
    if stage == 'stage1':
        print('기준 경계값(상위 1/3):', {k: round(v, 4) for k, v in thr.items()})
        print(W[W.stable_main][cols].round(4).to_string(index=False))
        print('\nalt 안정:', list(W[W.stable_alt].key))
        ref = neighbors5(W)
        json.dump(ref, open(S4/'cache'/'refine_list.json', 'w'))
        print('refine points:', len(ref))
        o, rho, rho_c = oos(W)
        print(o.round(3).to_string(index=False)); print('spearman sharpe', rho, 'cagr', rho_c)
    else:
        st = W[W.stable_main].copy()
        cen = st[['SCHD', 'SPY', 'QQQ']].mean()
        st['dist'] = np.sqrt(((st[['SCHD', 'SPY', 'QQQ']]-cen)**2).sum(axis=1))
        rk = [f'rank10_{c}' for c, _ in criteria(MAIN)]
        # 세분화 조합은 66개 순위가 없으므로 66개 기준 분포에서의 백분위로 대신 비교
        base = W[W.grid == '10%p']
        for c, _ in criteria(MAIN):
            st[f'pct_{c}'] = st[c].apply(lambda v: (base[c] > v).mean()*100)   # 상위 몇 %
        st['worst_pct'] = st[[f'pct_{c}' for c, _ in criteria(MAIN)]].max(axis=1)
        st = st.sort_values(['dist', 'worst_pct'])
        b = st.iloc[0]
        bopt = {'SCHD': round(float(b.SCHD), 4), 'SPY': round(float(b.SPY), 4), 'QQQ': round(float(b.QQQ), 4)}
        json.dump(bopt, open(S4/'cache'/'bopt.json', 'w'))
        json.dump(dict(centroid=cen.round(4).to_dict(), n_stable=int(len(st)),
                       n_stable_10=int((st.grid == '10%p').sum()), thresholds=thr, bopt=bopt,
                       stable=st[['key', 'SCHD', 'SPY', 'QQQ', 'grid', 'dist', 'worst_pct']
                                 + [f'pct_{c}' for c, _ in criteria(MAIN)]].round(4).to_dict('records')),
                  open(S4/'cache'/'stable_region.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print('centroid', cen.round(4).to_dict(), '-> B-opt', bopt)
        print(st[['key', 'grid', 'dist', 'worst_pct'] + [f'pct_{c}' for c, _ in criteria(MAIN)]].round(2).to_string(index=False))
        LONG = LONG.merge(W[['key', 'stable_main', 'stable_alt']], on='key')
        LONG.to_csv(S4/'s4_grid.csv', index=False, encoding='utf-8-sig')
        W.to_csv(S4/'cache'/'s4_grid_wide.csv', index=False, encoding='utf-8-sig')
        o, rho, rho_c = oos(W)
        o['Spearman_세후Sharpe_순위상관_66개'] = rho
        o['Spearman_세후CAGR_순위상관_66개'] = rho_c
        o.to_csv(S4/'s4_oos.csv', index=False, encoding='utf-8-sig')
        heatmaps(W, bopt)
        print('saved')
