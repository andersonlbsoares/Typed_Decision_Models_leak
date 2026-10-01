"""Controlled-exposure grid. Evaluation items are shared across the 3 seeds, so each item's
value is its mean correctness over seeds (resampling item identities jointly keeps all
seed-specific predictions of an item together). Reference correction: minus the
pre-fine-tuning checkpoint on the same item. Stratified multinomial bootstrap (core.contrast).
Also reports each seed separately."""
import gzip, json, sys
import numpy as np, pandas as pd
sys.path.insert(0, 'analysis')
from core import contrast
L = lambda p: json.load(gzip.open(p, 'rt'))
R4 = L('predictions/r4_eval.json.gz')
H6a = H6b = H6c = 'predictions/'
CT = {'emotion': L('predictions/r6_ctrl_emotion.json.gz'), 'agnews': L('predictions/r6_ctrl_agnews.json.gz'), 'yelp': L('predictions/r6_ctrl_yelp.json.gz')}
JF = {'yelp': L('predictions/r6_juliaft_yelp.json.gz'), 'yahoo': L('predictions/r6_juliaft_yahoo.json.gz')}
for k in JF:
    s2 = L(f'predictions/r7_juliaft_{k}_seed2.json.gz')
    for kk, v in s2.items():
        if kk != 'base':
            JF[k][kk] = v
SM = {d: L(f'predictions/r7_sens_small_{d}.json.gz') for d in ['emotion', 'agnews', 'yelp']}
BB = {d: L(f'predictions/r7_sens_bigB_{d}.json.gz') for d in ['emotion', 'agnews', 'yelp']}
B = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
rng = np.random.default_rng(42)

lab = lambda ds, s: np.array([x[1] for x in R4['julia'][ds]['A_' + s]])
def idx_of(g, n):
    return {'train_seen': np.arange(0, n), 'train_unseen': (np.arange(n, 1000) if n < 1000 else np.arange(1000, 2000)),
            'test_seen': np.arange(0, 500), 'test_unseen': np.arange(500, 1000)}[g]
PROBES = {'A': lambda sc: ('train_seen', 'test_unseen') if sc == 'S1' else ('train_seen', 'test_seen'),
          'B': lambda sc: ('test_seen', 'test_unseen')}

def run(cfgs, n_list, sc_list, probes, grid, test_big=False):
    rows = []
    for mname, ds, res, bkey in cfgs:
        ytr, yte = lab(ds, 'train'), lab(ds, 'test')
        if bkey == 'julia':
            bb = res['base']; btr = np.concatenate([bb['train_seen'], bb['train_unseen']]).astype(float)
            bte = np.concatenate([bb['test_seen'], bb['test_unseen']]).astype(float)
        else:
            btr = np.array([float(x[1] == x[2]) for x in R4[bkey][ds]['A_train']])
            bte = np.array([float(x[1] == x[2]) for x in R4[bkey][ds]['A_test']])
        for n in n_list:
            for sc in sc_list:
                seeds = [s for s in range(3) if f'{s}|{n}|{sc}' in res]
                if not seeds:
                    continue
                for ep in [1, 3]:
                    for pr in probes:
                        g1, g2 = PROBES[pr](sc)
                        i1, i2 = idx_of(g1, n), idx_of(g2, n)
                        if test_big and g2 == 'test_unseen':
                            i2 = np.arange(500, 2000)
                        y1 = (ytr if g1.startswith('train') else yte)[i1]; y2 = (ytr if g2.startswith('train') else yte)[i2]
                        M1 = np.stack([np.array(res[f'{s}|{n}|{sc}'][str(ep)][g1], float) for s in seeds])
                        M2 = np.stack([np.array(res[f'{s}|{n}|{sc}'][str(ep)][g2], float) for s in seeds])
                        assert M1.shape[1] == len(i1) and M2.shape[1] == len(i2)
                        b1 = (btr if g1.startswith('train') else bte)[i1]; b2 = (btr if g2.startswith('train') else bte)[i2]
                        assert len(b2) == len(i2), (len(b2), len(i2))
                        base = dict(grid=grid, model=mname, dataset=ds, n_exposed=n, scenario=sc, epochs=ep, probe=pr,
                                    n_seeds=len(seeds), n_ref=len(i2))
                        for corr in [False, True]:
                            v1 = M1.mean(0) - (b1 if corr else 0); v2 = M2.mean(0) - (b2 if corr else 0)
                            r = contrast(v1, y1, v2, y2, B=B, rng=rng)
                            per = [contrast(M1[k] - (b1 if corr else 0), y1, M2[k] - (b2 if corr else 0), y2, B=2000, rng=rng) for k in range(len(seeds))]
                            rows.append(dict(base, reference_corrected=corr, est=r['est'], lo=r['lo'], hi=r['hi'], p=r['p'],
                                             strata_used=r['strata_used'], coverage_1=r['coverage_1'], coverage_2=r['coverage_2'],
                                             seed_est=';'.join(f"{x['est']:.1f}" for x in per),
                                             seeds_positive_ci=sum(x['lo'] > 0 for x in per)))
    return pd.DataFrame(rows)

cfg_main = [('DeBERTa-v3', d, CT[d], 'deberta_c') for d in ['emotion', 'agnews', 'yelp']] + [('Julia-1', d, JF[d], 'julia') for d in ['yelp', 'yahoo']]
G = run(cfg_main, [250, 1000], ['S1', 'S2'], ['A', 'B'], 'main')
S = run([('DeBERTa-v3', d, SM[d], 'deberta_c') for d in SM], [25, 50, 100], ['S1'], ['A'], 'small_n')
BG = run([('DeBERTa-v3', d, BB[d], 'deberta_c') for d in BB], [250], ['S2'], ['B'], 'heldout_1500', test_big=True)
A = pd.concat([G, S, BG]); A['positive_ci'] = A.lo > 0
A.to_csv('results/controlled_grid.csv', index=False)
# summaries
def summ(df):
    return df.groupby(['model', 'probe', 'scenario', 'reference_corrected']).agg(k=('positive_ci', 'sum'), n=('positive_ci', 'size')).reset_index()
G = A[A.grid == 'main']; print(summ(G).to_string())
sensA = A[(A.model == 'DeBERTa-v3') & (A.probe == 'A') & (A.scenario == 'S1') & A.reference_corrected & A.grid.isin(['main', 'small_n'])]
print('DeBERTa probe A S1 corrected, all n:', int(sensA.positive_ci.sum()), '/', len(sensA))
print(sensA[['dataset', 'n_exposed', 'epochs', 'est', 'lo', 'hi', 'seed_est', 'seeds_positive_ci']].round(1).sort_values(['dataset', 'n_exposed', 'epochs']).to_string())
BB_ = A[(A.probe == 'B') & (A.model == 'DeBERTa-v3') & (A.scenario == 'S2') & A.reference_corrected & (A.n_exposed == 250)]
print(BB_[['grid', 'dataset', 'epochs', 'est', 'lo', 'hi']].round(1).to_string())
J = A[(A.model == 'Julia-1') & A.reference_corrected]
print(J[['dataset', 'n_exposed', 'scenario', 'epochs', 'probe', 'est', 'lo', 'hi', 'seed_est']].round(1).to_string())
