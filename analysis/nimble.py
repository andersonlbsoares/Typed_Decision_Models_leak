"""Nimble-9B (original-2676 adapter): item-weighted accuracy on its 2,676 published training examples minus its
324 frozen evaluation examples. Training and evaluation examples come from disjoint task families
(34 training, 6 evaluation families), so the contrast mixes exposure with task-family transfer.
Primary interval (as reported): task families are resampled as clusters INDEPENDENTLY IN BOTH GROUPS
(34 training and 6 evaluation families); the statistic is the ratio-of-sums accuracy difference.
Also reported: an interval conditional on the observed training set (only evaluation families resampled).
Two-sided shift-test p, B = 50,000. Output: results/nimble.csv."""
import gzip, json
import numpy as np, pandas as pd
L = lambda g: pd.DataFrame(json.load(gzip.open(f'predictions/nimble/nimble_{g}.json.gz', 'rt'))['rows'])
tr, ev = L('train'), L('eval')
tr['ok'] = tr.correct.astype(float); ev['ok'] = ev.correct.astype(float)
assert not set(tr.family) & set(ev.family)
def fam(d):
    g = d.groupby('family').ok.agg(['sum', 'count']); return g['sum'].values, g['count'].values
B = 50_000; rng = np.random.default_rng(2676)
(sa, ca), (sb, cb) = fam(tr), fam(ev)
est = sa.sum() / ca.sum() - sb.sum() / cb.sum()
ib = rng.integers(0, len(sb), (B, len(sb))); eb = sb[ib].sum(1) / cb[ib].sum(1)
ia = rng.integers(0, len(sa), (B, len(sa))); ea = sa[ia].sum(1) / ca[ia].sum(1)
out = []
for name, bs in [('both groups resampled (34 training + 6 evaluation families)', ea - eb),
                 ('conditional on training set (6 evaluation families resampled)', sa.sum() / ca.sum() - eb)]:
    lo, hi = np.percentile(bs, [2.5, 97.5]); p = (1 + np.sum(np.abs(bs - est) >= abs(est))) / (B + 1)
    out.append(dict(bootstrap=name, n_train=len(tr), n_eval=len(ev), fam_train=len(sa), fam_eval=len(sb),
                    acc_train=tr.ok.mean(), acc_eval=ev.ok.mean(), est=100 * est, lo=100 * lo, hi=100 * hi, p=p, B=B))
R = pd.DataFrame(out); R.to_csv('results/nimble.csv', index=False); print(R.round(3).to_string())
