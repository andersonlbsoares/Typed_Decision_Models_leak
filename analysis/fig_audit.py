"""Figure 2 (probe A by dataset and model). Reference models on the complete samples (third revision).
Run from the artifact root: python analysis/fig_audit.py -> paper/figs/fig_audit.{pdf,png}"""
import os
import gzip
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

META_GREY = "#888888"


def apply_figure_style(*, frame="open", font=None, sizes=(8, 7, 6), grid=False):
    import matplotlib as mpl
    if frame not in ("open", "boxed", "none"):
        raise ValueError(f"frame must be 'open'|'boxed'|'none', got {frame!r}")

    try:
        import os, sys, glob, matplotlib.font_manager as fm
        fdir = os.path.join(os.environ.get("CONDA_PREFIX") or sys.prefix, "fonts")
        if os.path.isdir(fdir):
            known = {f.fname for f in fm.fontManager.ttflist}
            for f in glob.glob(os.path.join(fdir, "*.ttf")):
                if f not in known:
                    fm.fontManager.addfont(f)
    except Exception:
        pass
    base, secondary, tick = sizes
    boxed = (frame == "boxed")
    rc = {
        "font.family": "sans-serif",
        "font.size": base,
        "axes.labelsize": base,
        "axes.titlesize": base,
        "legend.fontsize": secondary,
        "xtick.labelsize": tick,
        "ytick.labelsize": tick,
        "axes.linewidth": 0.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": boxed, "axes.spines.right": boxed,
        "axes.spines.left": frame != "none", "axes.spines.bottom": frame != "none",
        "axes.grid": bool(grid),
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.titleweight": "normal",
        "axes.titlelocation": "left",
        "axes.labelweight": "normal",
        "lines.linewidth": 1.2,
        "patch.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    if font:
        rc["font.sans-serif"] = [font, "DejaVu Sans"]
    mpl.rcParams.update(rc)


def panel_letter(ax, letter, dx=-0.18, dy=1.02, case="lower", fontsize=None):
    import matplotlib.pyplot as plt
    if fontsize is None:
        fontsize = plt.rcParams.get("font.size", 8) + 1
    s = letter.lower() if case == "lower" else letter.upper()
    ax.text(dx, dy, s, transform=ax.transAxes,
            fontweight="bold", fontsize=fontsize, va="bottom", ha="left")


# core contrast functions
def _cells(v, y):
    v = np.asarray(v, float); y = np.asarray(y)
    out = {}
    for c in np.unique(y):
        m = y == c
        u, k = np.unique(v[m], return_counts=True)
        out[c] = (u, k, int(m.sum()))
    return out


def _boot_means(u, k, n, B, rng):
    if len(u) == 1:
        return np.full(B, u[0])
    draws = rng.multinomial(n, k / n, size=B)
    return draws @ u / n


def contrast(v1, y1, v2, y2, B=50000, rng=None, return_boot=False):
    rng = rng or np.random.default_rng(0)
    c1, c2 = _cells(v1, y1), _cells(v2, y2)
    C = sorted(set(c1) & set(c2), key=str)
    n = np.array([c1[c][2] + c2[c][2] for c in C], float)
    w = n / n.sum()
    m1 = np.array([(c1[c][0] * c1[c][1]).sum() / c1[c][2] for c in C])
    m2 = np.array([(c2[c][0] * c2[c][1]).sum() / c2[c][2] for c in C])
    T = float(w @ (m1 - m2))
    bs = np.zeros(B)
    for wc, c in zip(w, C):
        bs += wc * (_boot_means(*c1[c], B, rng) - _boot_means(*c2[c], B, rng))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = (1 + np.sum(np.abs(bs - T) >= abs(T) - 1e-12)) / (B + 1)
    n1, n2 = len(np.asarray(y1)), len(np.asarray(y2))
    kept1 = sum(c1[c][2] for c in C); kept2 = sum(c2[c][2] for c in C)
    res = dict(est=100 * T, lo=100 * lo, hi=100 * hi, se=100 * bs.std(ddof=1), p=float(p), B=B,
               n1=n1, n2=n2, strata_1=len(c1), strata_2=len(c2), strata_used=len(C),
               excluded=";".join(str(c) for c in sorted(set(c1) ^ set(c2), key=str)),
               coverage_1=kept1 / n1, coverage_2=kept2 / n2)
    if return_boot:
        res["_boot"] = 100 * bs
    return res


def raw_diff(v1, v2):
    return 100 * (np.mean(v1) - np.mean(v2))


def acc_vec(recs):
    ids = [r[0] for r in recs]
    return ids, np.array([float(r[1] == r[2]) for r in recs]), np.array([r[1] for r in recs])


apply_figure_style(sizes=(8, 7, 6))

GA = pd.read_csv('results/audit_gaps.csv')
TY = pd.read_csv('results/typed_probeA.csv')

lay6 = json.load(gzip.open('predictions/r8_e3_laya_full.json.gz', 'rt'))

# Laya-EN rotten gap
_, a_, ya_ = acc_vec(lay6['english']['rotten']['A_train'])
_, b_, yb_ = acc_vec(lay6['english']['rotten']['A_test'])
layen_rot = contrast(a_, ya_, b_, yb_, B=20000)


names = {'typed': 'typed-decisions', 'emotion': 'DAIR Emotion', 'agnews': 'AG News',
         'rotten': 'Rotten Tomatoes', 'yahoo': 'Yahoo Answers', 'banking77': 'Banking77',
         'amazon': 'Amazon Polarity', 'imdb': 'IMDB', 'yelp': 'Yelp Polarity',
         'massive': 'MASSIVE', 'wikitoxic': 'WikiToxic'}
mods = [('julia', 'Julia-1', '#C8102E', 'o', 0),
        ('jev', 'Jev 1.13 (closed, via API)', '#222222', 'X', 4),
        ('msd1', 'Microsoft-Decision-1 (closed, via API)', '#CC79A7', 'h', 5),
        ('deberta_c', 'DeBERTa-c (reference)', '#4C72B0', 's', 1),
        ('qwen7b', 'Qwen2.5-7B-Instruct', '#55A868', 'D', 2),
        ('laya_ml', 'Laya multilingual', '#8172B2', '^', 3),
        ('laya_en', 'Laya English', '#8172B2', 'v', 3),
        ('laya_typed', 'Laya typed-decisions (tuned on train split)', '#DD8452', 'P', 1)]
rows = []
for _, r in TY[TY.kind == 'gap'].iterrows():
    rows.append(('typed', r.model, r.est, r.lo, r.hi))
for _, r in GA.iterrows():
    if r.model != 'deberta_all33':
        rows.append((r.dataset, r.model, r.est, r.lo, r.hi))
rows.append(('rotten', 'laya_en', layen_rot['est'], layen_rot['lo'], layen_rot['hi']))
# Jev (second audited model; analysis/jev.py and analysis/jev_typed.py)
JG = pd.read_csv('results/jev_audit_gaps.csv'); JT = pd.read_csv('results/jev_typed_probeA.csv')
r = JT[(JT.kind == 'gap') & (JT.model == 'jev')].iloc[0]; rows.append(('typed', 'jev', r.est, r.lo, r.hi))
for _, r in JG[JG.model == 'jev'].iterrows():
    rows.append((r.dataset, 'jev', r.est, r.lo, r.hi))
MG = pd.read_csv('results/msd1_audit_gaps.csv'); MT = pd.read_csv('results/msd1_typed_probeA.csv')
r = MT[(MT.kind == 'gap') & (MT.model == 'msd1')].iloc[0]; rows.append(('typed', 'msd1', r.est, r.lo, r.hi))
for _, r in MG[MG.model == 'msd1'].iterrows():
    rows.append((r.dataset, 'msd1', r.est, r.lo, r.hi))
P = pd.DataFrame(rows, columns=['ds', 'model', 'est', 'lo', 'hi'])
order = ['typed', 'emotion', 'agnews', 'rotten', 'yahoo', 'banking77', 'amazon', 'imdb', 'yelp', 'massive', 'wikitoxic']
fig, ax = plt.subplots(figsize=(6.9, 3.9))
off = {0: 0, 1: -0.14, 2: 0.14, 3: 0.28, 4: -0.28, 5: -0.42}
seen = set()
for i, ds in enumerate(order):
    for key, lab, col, mk, o in mods:
        q = P[(P.ds == ds) & (P.model == key)]
        if q.empty:
            continue
        r = q.iloc[0]; y = i + (-0.14 if key == 'laya_typed' else off[o])
        foc = key in ('julia', 'jev', 'msd1')
        ax.errorbar(r.est, y, xerr=[[r.est - r.lo], [r.hi - r.est]], fmt=mk, color=col,
                    ms=5 if foc else 3.5, lw=1.6 if foc else 0.8, alpha=1 if foc else 0.85,
                    mfc=col if foc else 'white', mec=col,
                    label=lab if lab not in seen else None, zorder=3 if foc else 2)
        seen.add(lab)
ax.axvline(0, color=META_GREY, lw=0.6, zorder=0)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([names[o] for o in order])
ax.invert_yaxis()
ax.set_xlabel('Label-standardized train − test accuracy (pp), 95% CI')
ax.set_title("Julia-1 shows a large training-split advantage; the closed Jev and Microsoft-Decision-1 do not")
jt = P[(P.ds == 'typed') & (P.model == 'julia')].iloc[0]
ax.text(jt.est, -0.45, f'+{jt.est:.1f}', color='#C8102E', ha='center', va='bottom', fontsize=7)
ax.legend(loc='lower right', fontsize=6)
ax.margins(y=0.03)
ax.set_ylim(len(order) - 0.4, -0.95)
os.makedirs('paper/figs', exist_ok=True)
fig.savefig('paper/figs/fig_audit.pdf'); fig.savefig('paper/figs/fig_audit.png')
plt.close(fig)