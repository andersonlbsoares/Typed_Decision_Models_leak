import gzip, json, sys
import numpy as np, pandas as pd
sys.path.insert(0, 'analysis')
from core import contrast, raw_diff, acc_vec, paired, holm, bh

L = lambda p: json.load(gzip.open(p, 'rt'))
e4 = L('predictions/r4_eval.json.gz')
# Reference models on the COMPLETE 2,000/2,000 samples (third revision; eval_scripts/r8). Qwen: the stored
# top-20 admissibility rule (field 4, 'pred_top20'); Laya: multilingual checkpoint, rev. 55cf4c4e.
qw = {'A': {k: {s: [[r[0], r[1], r[4]] for r in v[s]] for s in v} for k, v in L('predictions/r8_e3_qwen_full.json.gz')['A'].items()}}
lay = L('predictions/r8_e3_laya_full.json.gz')
ro = L('predictions/r6_rotten_eval.json.gz')
B = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000
DS = ["emotion", "agnews", "rotten", "yahoo", "banking77", "amazon", "imdb", "yelp", "massive", "wikitoxic"]

def qrec(lst):  # qwen records have no conf
    return [r[:3] + [None] for r in lst]

def recs(model, ds, split):
    if ds == 'rotten':
        if model in ('julia', 'deberta_c', 'deberta_all33'):
            return ro[model]['A_' + split]
        if model == 'qwen7b':
            return qrec(qw['A']['rotten'][split])
        return None
    if model in ('julia', 'deberta_c', 'deberta_all33'):
        return e4[model][ds]['A_' + split]
    if model == 'qwen7b':
        return qrec(qw['A'][ds][split])
    if model == 'laya_ml':
        return lay['multilingual'][ds]['A_' + split]

rng = np.random.default_rng(20260930)
gaps, dids = [], []
for ds in DS:
    for m in ['julia', 'deberta_c', 'deberta_all33', 'qwen7b', 'laya_ml']:
        tr, te = recs(m, ds, 'train'), recs(m, ds, 'test')
        if tr is None:
            continue
        _, a, ya = acc_vec(tr); _, b, yb = acc_vec(te)
        r = contrast(a, ya, b, yb, B=B, rng=rng)
        rej_tr = sum(x[2] is None for x in tr); rej_te = sum(x[2] is None for x in te)
        gaps.append(dict(dataset=ds, model=m, acc_train=a.mean(), acc_test=b.mean(), raw_gap=raw_diff(a, b),
                         rejected_train=rej_tr, rejected_test=rej_te, **r))
    jtr, jte = recs('julia', ds, 'train'), recs('julia', ds, 'test')
    for m in ['deberta_c', 'deberta_all33', 'qwen7b', 'laya_ml']:
        tr, te = recs(m, ds, 'train'), recs(m, ds, 'test')
        if tr is None:
            continue
        _, j1, c1, y1 = paired(jtr, tr); _, j2, c2, y2 = paired(jte, te)
        r = contrast(j1 - c1, y1, j2 - c2, y2, B=B, rng=rng)
        dids.append(dict(dataset=ds, control=m, julia_gap_paired=contrast(j1, y1, j2, y2, B=2000, rng=rng)['est'],
                         control_gap_paired=contrast(c1, y1, c2, y2, B=2000, rng=rng)['est'], **r))
G = pd.DataFrame(gaps); Dd = pd.DataFrame(dids)
assert len(Dd) == 39, len(Dd)
Dd['p_holm'] = holm(Dd.p); Dd['q_bh'] = bh(Dd.p)
G.to_csv('results/audit_gaps.csv', index=False); Dd.to_csv('results/audit_did_39.csv', index=False)
print(Dd[['dataset', 'control', 'est', 'lo', 'hi', 'p', 'p_holm', 'n1', 'n2', 'strata_used', 'coverage_1', 'coverage_2']].round(4).to_string())
print(G[['dataset', 'model', 'raw_gap', 'est', 'lo', 'hi', 'n1', 'n2', 'strata_used', 'excluded', 'coverage_1', 'coverage_2', 'rejected_train', 'rejected_test']].round(3).to_string())
