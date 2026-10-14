"""Analyses added in response to the pre-submission reviews of the TMLR version. Run from the artifact root:
    python analysis/revision_tmlr.py [B]      (default B = 10,000 for the controlled grid; 50,000 elsewhere)
Writes
  results/controlled_grid_probeC.csv   probe C under controlled exposure (same configurations as grid.py, Qwen
                                       paraphrases of the same items; pooled over seeds; minus the pre-FT checkpoint)
  results/laya_ml_rotten_outside_family.csv
                                       audited-model gap minus Laya-ML gap on Rotten Tomatoes (complete samples).
                                       Reported OUTSIDE the fixed 39-contrast families: those families were defined when
                                       the only Laya Rotten Tomatoes predictions came from a positive-only training subset
                                       (predictions/r6_laya.json.gz), and the Jev/Decision-1 plans fixed the same 39.
  results/three_split_common_weights.csv
                                       Emotion train/test/validation point estimates for Julia-1 with ONE label
                                       distribution pooled over the three samples (additive by construction), next to the
                                       pairwise-weight estimates of three_split.csv (Eq. 1 per pair; not additive).
  results/dilution_baseline_vs_audit.csv
                                       Julia-1's pre-fine-tuning Emotion/AG News decisions in the dilution harness
                                       (r8_e1_*.json.gz 'base'; texts cut at 1,200 characters, FP32 weights with bf16
                                       autocast) versus the stored audit run on the same items, gold labels and weights.
"""
import gzip, json, sys
import numpy as np, pandas as pd
sys.path.insert(0, 'analysis')
from core import contrast, paired, acc_vec

L = lambda p: json.load(gzip.open(p, 'rt')) if p.endswith('.gz') else json.load(open(p))
B = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
R4 = L('predictions/r4_eval.json.gz')
corr_vec = lambda recs: np.array([float(r[1] == r[2]) for r in recs])
lab = lambda ds, s: np.array([x[1] for x in R4['julia'][ds]['A_' + s]])

# ---------------------------------------------------------------- probe C under controlled exposure
CT = {d: L(f'predictions/r6_ctrl_{d}.json.gz') for d in ['emotion', 'agnews', 'yelp']}
JF = {}
for d in ['yelp', 'yahoo']:
    JF[d] = L(f'predictions/r6_juliaft_{d}.json.gz')
    for k, v in L(f'predictions/r7_juliaft_{d}_seed2.json.gz').items():
        if k != 'base':
            JF[d][k] = v
rng = np.random.default_rng(4242)
rows = []
for mname, ds, res, bkey in [('DeBERTa-v3', d, CT[d], 'deberta_c') for d in CT] + [('Julia-1', d, JF[d], 'julia') for d in JF]:
    ytr, yte = lab(ds, 'train'), lab(ds, 'test')
    if bkey == 'julia':   # pre-FT Julia-1 on the same 1,200-character texts and paraphrases
        bb = res['base']; f = lambda k: np.array(bb[k], float)
        bdrop_tr = f('train_seen') - f('train_seen_para')
        bdrop_te = np.concatenate([f('test_seen') - f('test_seen_para'), f('test_unseen') - f('test_unseen_para')])
    else:                 # pre-FT DeBERTa-c on the probe-C items (first 1,000 per split, same ids and order)
        g = R4['deberta_c'][ds]
        assert [r[0] for r in g['B_train_orig']] == [r[0] for r in R4['deberta_c'][ds]['A_train'][:1000]]
        bdrop_tr = corr_vec(g['B_train_orig']) - corr_vec(g['B_train_para'])
        bdrop_te = corr_vec(g['B_test_orig']) - corr_vec(g['B_test_para'])
    for n in [250, 1000]:
        for sc in ['S1', 'S2']:
            seeds = [s for s in range(3) if f'{s}|{n}|{sc}' in res]
            for ep in ['1', '3']:
                g2 = 'test_unseen' if sc == 'S1' else 'test_seen'      # same groups as probe A in grid.py
                i1 = np.arange(n); i2 = np.arange(500, 1000) if sc == 'S1' else np.arange(500)
                drop = lambda grp: np.stack([np.array(res[f'{s}|{n}|{sc}'][ep][grp], float)
                                             - np.array(res[f'{s}|{n}|{sc}'][ep][grp + '_para'], float) for s in seeds]).mean(0)
                D1, D2 = drop('train_seen'), drop(g2)
                for corr in [False, True]:
                    r = contrast(D1 - (bdrop_tr[i1] if corr else 0), ytr[i1], D2 - (bdrop_te[i2] if corr else 0), yte[i2], B=B, rng=rng)
                    rows.append(dict(model=mname, dataset=ds, n_exposed=n, scenario=sc, epochs=int(ep), probe='C',
                                     paraphraser='Qwen2.5-7B-Instruct', n_seeds=len(seeds), reference_corrected=corr,
                                     est=r['est'], lo=r['lo'], hi=r['hi'], p=r['p'], strata_used=r['strata_used'],
                                     coverage_1=r['coverage_1'], coverage_2=r['coverage_2']))
C = pd.DataFrame(rows); C['positive_ci'] = C.lo > 0
C.to_csv('results/controlled_grid_probeC.csv', index=False)
print(C.groupby(['model', 'scenario', 'reference_corrected']).agg(k=('positive_ci', 'sum'), n=('positive_ci', 'size')).to_string())

# ---------------------------------------------------------------- Laya-ML on Rotten Tomatoes, outside the families
rng = np.random.default_rng(20261014)
lay = L('predictions/r8_e3_laya_full.json.gz')['multilingual']['rotten']
ro = L('predictions/r6_rotten_eval.json.gz')
J = L('predictions/jev_1.13.json.gz')['A']['rotten']; Dm = L('predictions/msd1.json.gz')['A']['rotten']
out = []
_, a, ya = acc_vec(lay['A_train']); _, b, yb = acc_vec(lay['A_test'])
r = contrast(a, ya, b, yb, B=50_000, rng=rng)
out.append(dict(audited_model='', reference='laya_ml', kind='gap', **{k: r[k] for k in ('est', 'lo', 'hi', 'p', 'n1', 'n2', 'strata_used', 'coverage_1', 'coverage_2')}))
for name, src in [('julia', ro['julia']), ('jev', J), ('msd1', Dm)]:
    _, j1, c1, y1 = paired(src['A_train'], lay['A_train']); _, j2, c2, y2 = paired(src['A_test'], lay['A_test'])
    r = contrast(j1 - c1, y1, j2 - c2, y2, B=50_000, rng=rng)
    out.append(dict(audited_model=name, reference='laya_ml', kind='gap minus reference gap',
                    **{k: r[k] for k in ('est', 'lo', 'hi', 'p', 'n1', 'n2', 'strata_used', 'coverage_1', 'coverage_2')}))
O = pd.DataFrame(out); O['dataset'] = 'rotten'; O['family'] = 'outside the 39-contrast families'
O.to_csv('results/laya_ml_rotten_outside_family.csv', index=False); print(O.round(3).to_string())

# ---------------------------------------------------------------- three-split with common weights (Julia-1, Emotion)
v5 = L('predictions/r5_val.json.gz')
S = {'train': R4['julia']['emotion']['A_train'], 'test': R4['julia']['emotion']['A_test'], 'val': v5['julia']['emotion']}
acc = {s: np.array([np.mean([r[2] == c for r in S[s] if r[1] == c]) for c in range(6)]) for s in S}
cnt = {s: np.bincount([r[1] for r in S[s]], minlength=6) for s in S}
w3 = sum(cnt.values()) / sum(c.sum() for c in cnt.values())
rows = []
for a_, b_ in [('train', 'test'), ('test', 'val'), ('train', 'val')]:
    wp = (cnt[a_] + cnt[b_]) / (cnt[a_] + cnt[b_]).sum()
    rows.append(dict(model='julia', dataset='emotion', contrast=f'{a_}-{b_}', est_pairwise_weights=100 * wp @ (acc[a_] - acc[b_]),
                     est_common_weights=100 * w3 @ (acc[a_] - acc[b_])))
T = pd.DataFrame(rows); T.to_csv('results/three_split_common_weights.csv', index=False); print(T.round(2).to_string())

# ---------------------------------------------------------------- dilution baseline vs audit run
rows = []; rng = np.random.default_rng(9)
for key in ['emotion', 'agnews']:
    E = L(f'predictions/r8_e1_{key}.json.gz'); M = E['meta']
    for s in ['train', 'test']:
        au = R4['julia'][key]['A_' + s]
        assert [r[0] for r in au] == M[f'{s}_ids'] and [r[1] for r in au] == M[f'{s}_gold']
    ytr, yte = np.array(M['train_gold']), np.array(M['test_gold'])
    ptr = np.array([r[2] for r in R4['julia'][key]['A_train']]); pte = np.array([r[2] for r in R4['julia'][key]['A_test']])
    btr, bte = np.array(E['base']['train']), np.array(E['base']['test'])
    ra = contrast((ptr == ytr).astype(float), ytr, (pte == yte).astype(float), yte, B=50_000, rng=rng)
    rb = contrast((btr == ytr).astype(float), ytr, (bte == yte).astype(float), yte, B=50_000, rng=rng)
    rows.append(dict(dataset=key, decisions_changed_train=int((ptr != btr).sum()), decisions_changed_test=int((pte != bte).sum()),
                     audit_acc_train=(ptr == ytr).mean(), audit_acc_test=(pte == yte).mean(), audit_est=ra['est'], audit_lo=ra['lo'], audit_hi=ra['hi'],
                     dilution_base_acc_train=(btr == ytr).mean(), dilution_base_acc_test=(bte == yte).mean(),
                     dilution_base_est=rb['est'], dilution_base_lo=rb['lo'], dilution_base_hi=rb['hi']))
D = pd.DataFrame(rows); D.to_csv('results/dilution_baseline_vs_audit.csv', index=False); print(D.round(4).to_string())
