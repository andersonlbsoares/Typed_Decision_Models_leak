import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
GR = pd.read_csv('results/controlled_grid.csv'); KS = pd.read_csv('results/kev_by_source.csv'); KV = pd.read_csv('results/kev_label_standardized.csv')
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.75), gridspec_kw=dict(width_ratios=[1.15, 0.85, 1.25], wspace=0.55))
ax = axs[0]; cols = {'emotion': '#4C72B0', 'agnews': '#55A868', 'yelp': '#8172B2'}
S_ = GR[(GR.model == 'DeBERTa-v3') & (GR.probe == 'A') & (GR.scenario == 'S1') & GR.reference_corrected & GR.grid.isin(['main', 'small_n'])]
for ds, c in cols.items():
    for ep, ls, mk, dx in [(1, '--', 'o', 0.95), (3, '-', 's', 1.05)]:
        d = S_[(S_.dataset == ds) & (S_.epochs == ep)].sort_values('n_exposed'); x = d.n_exposed * dx
        ax.plot(x, d.est, ls=ls, color=c, lw=0.9)
        ax.errorbar(x, d.est, yerr=[d.est - d.lo, d.hi - d.est], fmt=mk, color=c, ms=3, lw=0.6, mfc=c if ep == 3 else 'white')
ax.set_xscale('log'); ax.set_xticks([25, 50, 100, 250, 1000]); ax.set_xticklabels(['25', '50', '100', '250', '1k']); ax.minorticks_off()
ax.axhline(0, color=META_GREY, lw=0.6); ax.set_xlabel('Exposed training items (log scale)'); ax.set_ylabel('Probe A vs. pre-FT checkpoint (pp)')
ax.set_title('Probe A by fine-tuning set size')
h = [Line2D([], [], color=c, lw=1, label={'emotion': 'Emotion', 'agnews': 'AG News', 'yelp': 'Yelp'}[k]) for k, c in cols.items()] + \
    [Line2D([], [], color='k', ls='--', marker='o', mfc='white', ms=3, lw=0.8, label='1 epoch'), Line2D([], [], color='k', ls='-', marker='s', ms=3, lw=0.8, label='3 epochs')]
ax.legend(handles=h, fontsize=6, loc='upper right')
ax = axs[1]
for i, m in enumerate(['kev-0.8b', 'kev-4b', 'kev-9b']):
    d = KS[KS.model == m]; ax.scatter(i + np.linspace(-0.22, 0.22, len(d)), d.est, s=6, color=META_GREY, zorder=2)
    r = KV[KV.model == m].iloc[0]; ax.errorbar(i, r.std_mf, yerr=[[r.std_mf - r.lo], [r.hi - r.std_mf]], fmt='D', color='#DD8452', ms=5, lw=1.2, zorder=3)
ax.axhline(0, color=META_GREY, lw=0.6); ax.set_xticks(range(3)); ax.set_xticklabels(['0.8B', '4B', '9B']); ax.set_xlabel('Kev size'); ax.set_ylabel('Members − fresh (pp)')
ax.set_title('Kev (documented)'); ax.margins(x=0.15)
ax.legend(handles=[Line2D([], [], marker='D', color='#DD8452', ls='', ms=4, label='all sources'), Line2D([], [], marker='o', color=META_GREY, ls='', ms=3, label='per source')], fontsize=6, loc='upper right')
ax = axs[2]; M = GR[(GR.grid == 'main') & GR.reference_corrected]
xpos = {('A', 'S1'): 0, ('A', 'S2'): 1, ('B', 'S1'): 2.4, ('B', 'S2'): 3.4}
for (pr, sc), x0 in xpos.items():
    for mname, col, mk, dx in [('DeBERTa-v3', '#4C72B0', 's', -0.17), ('Julia-1', '#C8102E', 'o', 0.17)]:
        d = M[(M.probe == pr) & (M.scenario == sc) & (M.model == mname)].reset_index(drop=True)
        for xx, (_, r) in zip(x0 + dx + np.linspace(-0.09, 0.09, len(d)), d.iterrows()):
            ax.errorbar(xx, r.est, yerr=[[r.est - r.lo], [r.hi - r.est]], fmt=mk, color=col, ms=2.6, lw=0.5, mfc=col if r.lo > 0 else 'white', alpha=0.9)
ax.axhline(0, color=META_GREY, lw=0.6)
ax.set_xticks(list(xpos.values())); ax.set_xticklabels(['train\nonly', 'train\n+test'] * 2)
for xx, t in [(0.5, 'probe A'), (2.9, 'probe B')]:
    ax.text(xx, -0.24, t, transform=ax.get_xaxis_transform(), ha='center', fontsize=7)
ax.set_ylabel('Contrast vs. pre-FT checkpoint (pp)'); ax.set_title('Two exposure scenarios')
ax.legend(handles=[Line2D([], [], marker='s', color='#4C72B0', ls='', ms=3, label='DeBERTa-v3 FT'), Line2D([], [], marker='o', color='#C8102E', ls='', ms=3, label='Julia-1 FT'),
                   Line2D([], [], marker='o', color='k', mfc='white', ls='', ms=3, label='CI includes 0')], fontsize=6, loc='upper left')
for a_, l in zip(axs, 'abc'):
    panel_letter(a_, l)
fig.savefig('fig_validation.pdf'); fig.savefig('fig_validation.png'); plt.close(fig)
