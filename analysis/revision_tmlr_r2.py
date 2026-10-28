"""Pre-submission review round 28/10/2026.
R01: probe A on a fixed evaluation set across scenarios.
In S1, probe A uses H_te (test items 501-1,000); in S2 it uses E_te (test items 1-500). Here probe A is
recomputed with E_te in BOTH scenarios, so S1 and S2 differ only in the fine-tuning data, and the
between-scenario change A_S1 - A_S2 is estimated on the same items (per-item S1-minus-S2 values,
same stratified bootstrap as grid.py; the pre-fine-tuning reference cancels in the change).
Output: results/controlled_grid_fixed_Ete.csv
R04/R05: API repeat checks and decision ties -> results/api_repeat_checks.csv, results/api_decision_ties.csv
R10: Kev group overlaps -> results/kev_group_overlap.csv
Run from the artifact root: python analysis/revision_tmlr_r2.py [B]"""
import gzip, json, sys
import numpy as np, pandas as pd
sys.path.insert(0, 'analysis')
from core import contrast
L = lambda p: json.load(gzip.open(p, 'rt'))
R4 = L('predictions/r4_eval.json.gz')
CT = {d: L(f'predictions/r6_ctrl_{d}.json.gz') for d in ['emotion', 'agnews', 'yelp']}
JF = {'yelp': L('predictions/r6_juliaft_yelp.json.gz'), 'yahoo': L('predictions/r6_juliaft_yahoo.json.gz')}
for k in JF:
    for kk, v in L(f'predictions/r7_juliaft_{k}_seed2.json.gz').items():
        if kk != 'base':
            JF[k][kk] = v
B = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
rng = np.random.default_rng(2810)
lab = lambda ds, s: np.array([x[1] for x in R4['julia'][ds]['A_' + s]])
rows = []
cfgs = [('DeBERTa-v3', d, CT[d], 'deberta_c') for d in ['emotion', 'agnews', 'yelp']] + [('Julia-1', d, JF[d], 'julia') for d in ['yelp', 'yahoo']]
for mname, ds, res, bkey in cfgs:
    ytr, yte = lab(ds, 'train'), lab(ds, 'test')
    if bkey == 'julia':
        bb = res['base']; btr = np.concatenate([bb['train_seen'], bb['train_unseen']]).astype(float)
        bte = np.concatenate([bb['test_seen'], bb['test_unseen']]).astype(float)
    else:
        btr = np.array([float(x[1] == x[2]) for x in R4[bkey][ds]['A_train']])
        bte = np.array([float(x[1] == x[2]) for x in R4[bkey][ds]['A_test']])
    for n in [250, 1000]:
        i1, i2 = np.arange(0, n), np.arange(0, 500)
        y1, y2 = ytr[i1], yte[i2]
        M = {}
        for sc in ['S1', 'S2']:
            seeds = [s for s in range(3) if f'{s}|{n}|{sc}' in res]
            M[sc] = seeds
        seeds = sorted(set(M['S1']) & set(M['S2']))
        if not seeds:
            continue
        for ep in [1, 3]:
            V = {}
            for sc in ['S1', 'S2']:
                a = np.stack([np.array(res[f'{s}|{n}|{sc}'][str(ep)]['train_seen'], float) for s in seeds]).mean(0)
                b = np.stack([np.array(res[f'{s}|{n}|{sc}'][str(ep)]['test_seen'], float) for s in seeds]).mean(0)
                V[sc] = (a, b)
                r = contrast(a - btr[i1], y1, b - bte[i2], y2, B=B, rng=rng)
                rows.append(dict(model=mname, dataset=ds, n_exposed=n, epochs=ep, n_seeds=len(seeds), quantity=f'A_{sc}(E_tr,E_te)_ref_corrected',
                                 est=r['est'], lo=r['lo'], hi=r['hi'], p=r['p']))
            r = contrast(V['S1'][0] - V['S2'][0], y1, V['S1'][1] - V['S2'][1], y2, B=B, rng=rng)
            rows.append(dict(model=mname, dataset=ds, n_exposed=n, epochs=ep, n_seeds=len(seeds), quantity='A_S1-A_S2(E_tr,E_te)',
                             est=r['est'], lo=r['lo'], hi=r['hi'], p=r['p']))
D = pd.DataFrame(rows); D['positive_ci'] = D.lo > 0; D['negative_ci'] = D.hi < 0
D.to_csv('results/controlled_grid_fixed_Ete.csv', index=False)
print(D.groupby(['model', 'quantity']).agg(pos=('positive_ci', 'sum'), neg=('negative_ci', 'sum'), n=('positive_ci', 'size')).to_string())


# ---------------------------------------------------------------- R04/R05: API repeat checks and decision ties
import collections
sys.path.insert(0, 'analysis')
from jev import decide          # decision rules of the analysis plans (identical in msd1.py)

def decide_alt(body):
    """Sensitivity rule, independent of gold: p = 0.5 -> 'false'; score ties -> lowest level."""
    if not isinstance(body, dict) or not body.get('answers'):
        return None
    a = list(body['answers'].values())[0]; t = a.get('type')
    if t == 'noul':
        p = a.get('noul'); return None if p is None else ('true' if p > 0.5 else 'false')
    P = a.get('probabilities') or {}
    if t == 'score':
        if not P: return None
        m = max(P.values()); return min([k for k, v in P.items() if v == m], key=float)
    return a.get('choice')

def load_raw(path):
    reps = collections.defaultdict(dict)
    for l in gzip.open(path, 'rt'):
        try: r = json.loads(l)
        except ValueError: continue
        if r.get('status') == 200: reps[r['hash']].setdefault(r.get('rep', 1), r)
    return reps

api_rows, rep_rows = [], []
for model, raw, idx in [('jev', 'predictions/jev_raw.jsonl.gz', 'predictions/jev_request_index.jsonl.gz'),
                        ('msd1', 'predictions/msd1_raw.jsonl.gz', 'predictions/msd1_request_index.jsonl.gz')]:
    reps = load_raw(raw); Q = [json.loads(l) for l in gzip.open(idx, 'rt')]
    cnt = collections.Counter(); typed = {r: {'train': [], 'test': []} for r in ['plan', 'alt']}
    for q in Q:
        t = q['task'].split(':'); body = reps.get(q['hash'], {}).get(1, {}).get('body')
        if t[1] != 'typed':
            continue
        a = list(body['answers'].values())[0]; P = a.get('probabilities') or {}
        cnt[(t[2], a['type'], 'n')] += 1
        if a['type'] == 'noul' and a.get('noul') == 0.5: cnt[(t[2], 'noul', 'p_eq_0.5')] += 1
        if a['type'] == 'score' and P and sum(v == max(P.values()) for v in P.values()) > 1: cnt[(t[2], 'score', 'tied_max')] += 1
        strat = f"{q['workflow']}|{q['id'].rsplit(':', 1)[1]}|{q['gold']}"
        typed['plan'][t[2]].append((float(decide(body)[0] == q['gold']), strat))
        typed['alt'][t[2]].append((float(decide_alt(body) == q['gold']), strat))
    for (split, qt, what), v in sorted(cnt.items()):
        api_rows.append(dict(model=model, kind='typed_count', split=split, qtype=qt, what=what, value=v))
    for rule in ['plan', 'alt']:
        tr, te = typed[rule]['train'], typed[rule]['test']
        r = contrast(np.array([x[0] for x in tr]), np.array([x[1] for x in tr]), np.array([x[0] for x in te]), np.array([x[1] for x in te]), B=200, rng=rng)
        api_rows.append(dict(model=model, kind='typed_gap_point_estimate', split='train-test', qtype='all', what=rule, value=round(r['est'], 3)))
    qmap = {q['hash']: q for q in Q}
    for h, d in reps.items():
        if 1 in d and 2 in d:
            q = qmap[h]; t = q['task'].split(':')
            def corr(body):
                k = decide(body)[0]
                if t[1] == 'typed': return k == q['gold']
                o = q['opts']; return (None if k is None or k not in [str(j) for j in range(len(o))] else o[int(k)]) == q['gold']
            k1, k2 = decide(d[1]['body'])[0], decide(d[2]['body'])[0]
            rep_rows.append(dict(model=model, task=q['task'], flip=k1 != k2, correct_1=corr(d[1]['body']), correct_2=corr(d[2]['body']),
                                 minutes_between=(d[2]['t'] - d[1]['t']) / 60))
RP = pd.DataFrame(rep_rows); RP.to_csv('results/api_repeat_checks.csv', index=False)
print(RP.groupby('model').agg(n=('flip', 'size'), flips=('flip', 'sum'),
      c_to_i=('correct_1', lambda s: int((s & ~RP.loc[s.index, 'correct_2']).sum())),
      i_to_c=('correct_1', lambda s: int((~s & RP.loc[s.index, 'correct_2']).sum())),
      med_min=('minutes_between', 'median')).to_string())
print(RP.groupby(['model', 'task']).size().to_string())
AP = pd.DataFrame(api_rows); AP.to_csv('results/api_decision_ties.csv', index=False); print(AP.to_string())

# ---------------------------------------------------------------- R10: Kev group overlaps (underlying-item identifiers)
ov = []
for m in ['kev-0.8b', 'kev-4b', 'kev-9b']:
    G = {g: json.load(open(f'predictions/kev/{m}/{g}/rows.json')) for g in ['members', 'fresh', 'heldout']}
    for a, b in [('members', 'fresh'), ('members', 'heldout'), ('fresh', 'heldout')]:
        ga, gb = {r['group'] for r in G[a]}, {r['group'] for r in G[b]}
        ov.append(dict(model=m, group_1=a, group_2=b, items_1=len(ga), items_2=len(gb), shared_items=len(ga & gb),
                       shared_question_ids=len({r['id'] for r in G[a]} & {r['id'] for r in G[b]})))
OV = pd.DataFrame(ov); OV.to_csv('results/kev_group_overlap.csv', index=False); print(OV.to_string())
