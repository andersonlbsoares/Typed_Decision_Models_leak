import gzip, json, sys
import numpy as np, pandas as pd
sys.path.insert(0, 'analysis')
from core import contrast, acc_vec, paired, holm, bh
L = lambda p: json.load(gzip.open(p, 'rt')) if p.endswith('.gz') else json.load(open(p))
e4 = L('predictions/r4_eval.json.gz'); v5 = L('predictions/r5_val.json.gz'); P2 = L('predictions/r7_para2.json.gz')
qw = L('predictions/r6_qwen.json.gz')
B = 50_000; rng = np.random.default_rng(11)

# ---- three-split probe on datasets with a validation split
rows = []
for ds in ['emotion', 'massive', 'wikitoxic']:
    for m in ['julia', 'deberta_c', 'deberta_all33']:
        S = {'train': e4[m][ds]['A_train'], 'test': e4[m][ds]['A_test'], 'val': v5[m][ds]}
        for a, b in [('test', 'val'), ('train', 'val'), ('train', 'test')]:
            _, x, yx = acc_vec(S[a]); _, z, yz = acc_vec(S[b])
            rows.append(dict(dataset=ds, model=m, contrast=f'{a}-{b}', **contrast(x, yx, z, yz, B=B, rng=rng)))
    for m in ['deberta_c', 'deberta_all33']:
        for a, b in [('test', 'val'), ('train', 'val')]:
            ja = e4['julia'][ds]['A_' + a] if a != 'val' else v5['julia'][ds]
            jb = v5['julia'][ds] if b == 'val' else e4['julia'][ds]['A_' + b]
            ca = e4[m][ds]['A_' + a] if a != 'val' else v5[m][ds]
            cb = v5[m][ds] if b == 'val' else e4[m][ds]['A_' + b]
            _, j1, c1, y1 = paired(ja, ca); _, j2, c2, y2 = paired(jb, cb)
            rows.append(dict(dataset=ds, model='julia-' + m, contrast=f'{a}-{b} DiD', **contrast(j1 - c1, y1, j2 - c2, y2, B=B, rng=rng)))
T3 = pd.DataFrame(rows); T3.to_csv('results/three_split.csv', index=False)
print(T3[['dataset', 'model', 'contrast', 'est', 'lo', 'hi', 'p', 'n1', 'n2']].round(3).to_string())

# ---- probe C: paired orig-minus-paraphrase drop, common label weights across splits
def drops(src, m, ds):
    g = src[m][ds]
    out = {}
    for s in ['train', 'test']:
        ids, co, cp, y = paired(g[f'B_{s}_orig'], g[f'B_{s}_para'])
        out[s] = (ids, co - cp, y, co, cp)
    return out

def qdrops(ds):
    out = {}
    for s in ['train', 'test']:
        o = [r[:3] + [None] for r in qw['Borig'][ds][s]]; p = [r[:3] + [None] for r in qw['Bpara'][ds][s]]
        ids, co, cp, y = paired(o, p); out[s] = (ids, co - cp, y, co, cp)
    return out

rowsC = []
for para, src in [('Qwen2.5-7B-Instruct', e4), ('Phi-3.5-mini-instruct', P2)]:
    for ds in ['emotion', 'agnews', 'banking77', 'yelp', 'yahoo']:
        dm = {m: drops(src, m, ds) for m in ['julia', 'deberta_c']}
        if para.startswith('Qwen'):
            dm['qwen7b'] = qdrops(ds)
        for m, d in dm.items():
            r = contrast(d['train'][1], d['train'][2], d['test'][1], d['test'][2], B=B, rng=rng)
            rowsC.append(dict(paraphraser=para, dataset=ds, model=m, kind='within-model',
                              drop_train=100 * d['train'][1].mean(), drop_test=100 * d['test'][1].mean(), **r))
        for ctl in [c for c in dm if c != 'julia']:
            J, Cc = dm['julia'], dm[ctl]
            vv = {}
            for s in ['train', 'test']:
                mc = dict(zip(Cc[s][0], Cc[s][1])); ids = [i for i in J[s][0] if i in mc]
                mj = dict(zip(J[s][0], J[s][1])); my = dict(zip(J[s][0], J[s][2]))
                vv[s] = (np.array([mj[i] - mc[i] for i in ids]), np.array([my[i] for i in ids]))
            r = contrast(vv['train'][0], vv['train'][1], vv['test'][0], vv['test'][1], B=B, rng=rng)
            rowsC.append(dict(paraphraser=para, dataset=ds, model='julia-' + ctl, kind='between-model DiD', **r))
C = pd.DataFrame(rowsC)
fam = (C.kind == 'between-model DiD') & (C.model == 'julia-deberta_c')
C.loc[fam, 'family'] = 'C-confirmatory (10)'
C.loc[fam, 'p_holm'] = holm(C.loc[fam, 'p']); C.loc[fam, 'q_bh'] = bh(C.loc[fam, 'p'])
C.to_csv('results/probeC.csv', index=False)
print(C[['paraphraser', 'dataset', 'model', 'kind', 'est', 'lo', 'hi', 'p', 'p_holm', 'n1', 'n2', 'strata_used']].round(4).to_string())
