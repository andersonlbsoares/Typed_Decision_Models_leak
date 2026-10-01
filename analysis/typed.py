"""typed-decisions probe A with state-level cluster bootstrap stratified by workflow.
Strata: workflow x question x gold. Weights fixed at pooled observed counts on common support.
Because states (not questions) are resampled, a rare workflow x question x gold stratum can be absent
from a replicate. Primary estimator: such a replicate drops that stratum and renormalizes the remaining
fixed weights (so the replicate estimand differs slightly). Diagnostics reported per contrast: fraction of
replicates with >=1 absent stratum, mean and maximum fixed-weight mass removed. Sensitivity estimator:
restrict the common support to strata with >= MIN_CL distinct states in each split (fixed weights on
that support), which makes absent strata negligible while keeping state clusters intact.
Results: results/typed_probeA.csv (primary + diagnostics), results/typed_bootstrap_sensitivity.csv."""
import gzip, json, sys
import numpy as np, pandas as pd
sys.path.insert(0, 'analysis')
from core import holm
L = lambda p: json.load(gzip.open(p, 'rt')) if p.endswith('.gz') else json.load(open(p))
r2 = L('predictions/r2_julia_nli.json.gz'); q3 = L('predictions/r3_qwen_typed.json'); ly = L('predictions/r2_laya.json.gz')
MODELS = {'julia': r2['typed'], 'qwen7b': q3, 'laya_en': ly['typed']['laya_en'],
          'laya_ml': ly['typed']['laya_ml'], 'laya_typed': ly['typed']['laya_typed']}
B = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
rng = np.random.default_rng(7)

def table(split):
    base = MODELS['julia'][split]
    ids = [r['id'] for r in base]
    meta = {r['id']: r for r in base}
    for m, d in MODELS.items():
        s = {r['id'] for r in d[split]}
        assert set(ids) <= s, (m, split)
    corr = {m: np.array([float({r['id']: r for r in d[split]}[i]['pred'] == meta[i]['gold']) for i in ids]) for m, d in MODELS.items()}
    for m, d in MODELS.items():
        gm = {r['id']: r['gold'] for r in d[split]}
        assert all(gm[i] == meta[i]['gold'] for i in ids), m
    state = np.array([i.rsplit(':', 1)[0] for i in ids]); q = [i.rsplit(':', 1)[1] for i in ids]
    wf = np.array([meta[i]['workflow'] for i in ids])
    strat = np.array([f"{meta[i]['workflow']}|{qq}|{meta[i]['gold']}" for i, qq in zip(ids, q)])
    return dict(ids=ids, state=state, wf=wf, strat=strat, corr=corr)

T = {s: table(s) for s in ['train', 'test']}

def agg(t, v, strata):
    """state x stratum sums and counts"""
    st, si = np.unique(t['state'], return_inverse=True)
    col = {c: k for k, c in enumerate(strata)}
    S = np.zeros((len(st), len(strata))); N = np.zeros_like(S)
    for i, (s, c) in enumerate(zip(si, t['strat'])):
        if c in col:
            S[s, col[c]] += v[i]; N[s, col[c]] += 1
    swf = np.array([t['wf'][np.where(si == k)[0][0]] for k in range(len(st))])
    return S, N, swf

def boot_weights(swf, B, rng):
    W = np.zeros((B, len(swf)))
    for w in np.unique(swf):
        idx = np.where(swf == w)[0]
        W[:, idx] = rng.multinomial(len(idx), np.ones(len(idx)) / len(idx), size=B)
    return W

def n_states(t, c):
    return len(set(t['state'][t['strat'] == c]))

def cluster_contrast(vtr, vte, B=B, rng=rng, min_cl=None):
    ctr = set(T['train']['strat']); cte = set(T['test']['strat'])
    strata = sorted(ctr & cte)
    if min_cl:
        strata = [c for c in strata if n_states(T['train'], c) >= min_cl and n_states(T['test'], c) >= min_cl]
    S1, N1, wf1 = agg(T['train'], vtr, strata); S2, N2, wf2 = agg(T['test'], vte, strata)
    n = N1.sum(0) + N2.sum(0); w = n / n.sum()
    est = w @ (S1.sum(0) / N1.sum(0) - S2.sum(0) / N2.sum(0))
    bs = np.empty(B); empty = 0; dropped = np.empty(B)
    for b0 in range(0, B, 2000):
        bb = min(2000, B - b0)
        W1 = boot_weights(wf1, bb, rng); W2 = boot_weights(wf2, bb, rng)
        s1, n1, s2, n2 = W1 @ S1, W1 @ N1, W2 @ S2, W2 @ N2
        ok = (n1 > 0) & (n2 > 0); empty += int((~ok).any(1).sum())
        with np.errstate(invalid='ignore', divide='ignore'):
            d = np.where(ok, s1 / np.where(n1 > 0, n1, 1) - s2 / np.where(n2 > 0, n2, 1), 0)
        dropped[b0:b0 + bb] = ((~ok) * w).sum(1)
        ww = np.where(ok, w, 0); ww = ww / ww.sum(1, keepdims=True)
        bs[b0:b0 + bb] = (ww * d).sum(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = (1 + np.sum(np.abs(bs - est) >= abs(est))) / (B + 1)
    excl = len((ctr | cte) - set(strata))
    cov1 = N1.sum() / len(vtr); cov2 = N2.sum() / len(vte)
    return dict(est=100 * est, lo=100 * lo, hi=100 * hi, p=p, B=B, strata_used=len(strata), strata_excluded=excl,
                coverage_train=cov1, coverage_test=cov2, replicates_with_empty_stratum=empty,
                frac_replicates_empty=empty / B, mean_weight_dropped=float(dropped.mean()), max_weight_dropped=float(dropped.max()),
                min_states_per_stratum=int(min(min(n_states(T['train'], c), n_states(T['test'], c)) for c in strata)))

rows = []
for m in MODELS:
    a, b = T['train']['corr'][m], T['test']['corr'][m]
    rows.append(dict(kind='gap', model=m, acc_train=a.mean(), acc_test=b.mean(), raw=100 * (a.mean() - b.mean()), **cluster_contrast(a, b)))
for m in ['qwen7b', 'laya_en', 'laya_ml', 'laya_typed']:
    a = T['train']['corr']['julia'] - T['train']['corr'][m]; b = T['test']['corr']['julia'] - T['test']['corr'][m]
    rows.append(dict(kind='did', model='julia-' + m, **cluster_contrast(a, b)))
R = pd.DataFrame(rows)
R.to_csv('results/typed_probeA.csv', index=False)
print(R.round(4).to_string())
# sensitivity: restricted support with >= MIN_CL states per stratum in each split (separate RNG stream)
MIN_CL = 10; rs = np.random.default_rng(70)
sens = []
for _, r in R.iterrows():
    if r.kind == 'gap':
        a, b = T['train']['corr'][r.model], T['test']['corr'][r.model]
    else:
        m = r.model.replace('julia-', '')
        a = T['train']['corr']['julia'] - T['train']['corr'][m]; b = T['test']['corr']['julia'] - T['test']['corr'][m]
    x = cluster_contrast(a, b, rng=rs, min_cl=MIN_CL)
    sens.append(dict(kind=r.kind, model=r.model, primary_est=r.est, primary_lo=r.lo, primary_hi=r.hi, min_states=MIN_CL,
                     **{k: x[k] for k in ['est', 'lo', 'hi', 'p', 'B', 'strata_used', 'coverage_train', 'coverage_test',
                                          'frac_replicates_empty', 'mean_weight_dropped', 'max_weight_dropped']}))
SR = pd.DataFrame(sens); SR.to_csv('results/typed_bootstrap_sensitivity.csv', index=False)
print(SR.round(4).to_string())
print('states', len(set(T['train']['state'])), len(set(T['test']['state'])), 'questions', len(T['train']['ids']), len(T['test']['ids']))
