"""Kev members vs fresh / dev-cal (Table 3 and supplementary Kev table).
Strata: source x question x label; clusters: underlying item (`group` field); bootstrap resamples clusters
within source (B = 10,000 overall, 5,000 per source). A stratum absent from a replicate is dropped and the
remaining fixed weights renormalized; diagnostics (fraction of replicates with an absent stratum, weight mass
removed) and a sensitivity estimator restricted to strata with >= MIN_CL clusters in each group are reported.
Outputs: results/kev_label_standardized.csv, results/kev_by_source.csv, results/kev_bootstrap_diagnostics.csv."""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, 'analysis')
from core import holm

def kev_rows(d):
    out = {}
    for g in ['members', 'fresh', 'heldout']:
        rr = json.load(open(f'{d}/{g}/rows.json'))
        out[g] = pd.DataFrame([dict(id=r['id'], grp=r['group'], src=r['source'], q=r['question'], label=str(r['label']),
                                    ok=float(np.argmax(json.loads(r['p']) if isinstance(r['p'], str) else r['p']) == int(r['label']))
                                    if r['label'] not in (None, 'None') and r['p'] not in (None, 'None') else np.nan) for r in rr])
    return out

def cl_contrast(d1, d2, B=10000, rng=None, min_cl=None):
    for d in (d1, d2): d['st'] = d.src + '|' + d.q + '|' + d.label
    strata = sorted(set(d1.st) & set(d2.st))
    if min_cl:
        strata = [s for s in strata if d1[d1.st == s].grp.nunique() >= min_cl and d2[d2.st == s].grp.nunique() >= min_cl]
    col = {s: i for i, s in enumerate(strata)}
    def agg(d):
        g = d[d.st.isin(col)]; gi = pd.factorize(g.grp)[0]; ng = gi.max() + 1
        S = np.zeros((ng, len(strata))); N = np.zeros_like(S)
        np.add.at(S, (gi, g.st.map(col).values), g.ok.values); np.add.at(N, (gi, g.st.map(col).values), 1)
        src = g.groupby(gi).src.first().values; return S, N, src
    S1, N1, s1 = agg(d1); S2, N2, s2 = agg(d2)
    n = N1.sum(0) + N2.sum(0); w = n / n.sum(); est = w @ (S1.sum(0) / N1.sum(0) - S2.sum(0) / N2.sum(0))
    bs, dropped, empty = [], [], 0
    for b0 in range(0, B, 1000):
        W1 = np.zeros((1000, len(s1))); W2 = np.zeros((1000, len(s2)))
        for W, s in ((W1, s1), (W2, s2)):
            for u in np.unique(s):
                ix = np.where(s == u)[0]; W[:, ix] = rng.multinomial(len(ix), np.ones(len(ix)) / len(ix), size=1000)
        a1, c1, a2, c2 = W1 @ S1, W1 @ N1, W2 @ S2, W2 @ N2; ok = (c1 > 0) & (c2 > 0)
        empty += int((~ok).any(1).sum()); dropped.append(((~ok) * w).sum(1))
        dd = np.where(ok, a1 / np.where(c1 > 0, c1, 1) - a2 / np.where(c2 > 0, c2, 1), 0); ww = np.where(ok, w, 0); ww /= ww.sum(1, keepdims=True)
        bs.append((ww * dd).sum(1))
    bs = np.concatenate(bs); dropped = np.concatenate(dropped)
    lo, hi = np.percentile(bs, [2.5, 97.5]); p = (1 + np.sum(np.abs(bs - est) >= abs(est))) / (B + 1)
    return dict(est=100 * est, lo=100 * lo, hi=100 * hi, p=p, B=B, strata=len(strata), cov1=N1.sum() / len(d1), cov2=N2.sum() / len(d2),
                frac_replicates_empty=empty / B, mean_weight_dropped=float(dropped.mean()), max_weight_dropped=float(dropped.max()))

if __name__ == '__main__':
    K = {m: kev_rows(f'predictions/kev/{m}') for m in ['kev-0.8b', 'kev-4b', 'kev-9b']}
    rng = np.random.default_rng(1)          # one stream shared by all overall contrasts (as in the reported run)
    rows, diag = [], []
    for m, k in K.items():
        r1 = cl_contrast(k['members'].copy(), k['fresh'].copy(), rng=rng); r2 = cl_contrast(k['members'].copy(), k['heldout'].copy(), rng=rng)
        rows.append(dict(model=m, acc_m=k['members'].ok.mean(), acc_f=k['fresh'].ok.mean(), acc_h=k['heldout'].ok.mean(),
                         raw_mf=100 * (k['members'].ok.mean() - k['fresh'].ok.mean()), std_mf=r1['est'], lo=r1['lo'], hi=r1['hi'], p=r1['p'],
                         cov_m=r1['cov1'], cov_f=r1['cov2'], std_mh=r2['est'], mh_lo=r2['lo'], mh_hi=r2['hi'], cov_h=r2['cov2']))
        for nm, r in [('members-fresh', r1), ('members-devcal', r2)]:
            diag.append(dict(model=m, contrast=nm, estimator='primary (renormalize absent strata)', **r))
    KV = pd.DataFrame(rows); KV.to_csv('results/kev_label_standardized.csv', index=False)
    rs = np.random.default_rng(71); MIN_CL = 10
    for m, k in K.items():
        for nm, g in [('members-fresh', 'fresh'), ('members-devcal', 'heldout')]:
            r = cl_contrast(k['members'].copy(), k[g].copy(), rng=rs, min_cl=MIN_CL)
            diag.append(dict(model=m, contrast=nm, estimator=f'sensitivity (strata with >= {MIN_CL} clusters per group)', **r))
    D = pd.DataFrame(diag); D.to_csv('results/kev_bootstrap_diagnostics.csv', index=False)
    rows = []
    for m, k in K.items():
        for s in sorted(k['members'].src.unique()):
            a = k['members'][k['members'].src == s].copy(); b = k['fresh'][k['fresh'].src == s].copy()
            r = cl_contrast(a, b, B=5000, rng=np.random.default_rng(len(rows)))
            rows.append(dict(model=m, source=s, n_members=len(a), acc_fresh=b.ok.mean(),
                             **{k_: r[k_] for k_ in ['est', 'lo', 'hi', 'p', 'strata', 'cov1', 'cov2', 'frac_replicates_empty', 'mean_weight_dropped', 'max_weight_dropped']}))
    KS = pd.DataFrame(rows); KS['p_holm'] = holm(KS.p); KS.to_csv('results/kev_by_source.csv', index=False)
    print(KV.round(3).to_string()); print(D.round(4).to_string())
    print(KS.groupby('model').apply(lambda d: int((d.p_holm < .05).sum())).to_dict())
