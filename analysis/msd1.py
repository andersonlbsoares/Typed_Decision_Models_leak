"""Microsoft-Decision-1 as a third audited model (pre-registered plan: prereg_decision1.md; same code as jev.py).
Run from the artifact root after predictions/msd1.json.gz exists (built by `msd1.py collect`).

  python analysis/msd1.py collect predictions/msd1_request_index.jsonl.gz predictions/msd1_raw.jsonl.gz   # -> predictions/msd1.json.gz
  python analysis/msd1.py [B]                                               # -> results/msd1_*.csv

Records use the same layout as the other models: classification [id, gold, pred, p_gold];
typed-decisions {id, type, gold, pred, workflow, p_gold}. The first send (rep 1) of every request is used.
Decision rules (prereg section 7): choice -> returned key; noul -> 'true' if p > 0.5, 'false' if
p < 0.5, scored incorrect if p == 0.5; score -> argmax of the returned probabilities (ties go to the
key closest to the returned expected score, then to the lower key).
Same estimators, strata, bootstraps and seeds as audit.py, typed.py and probe_bc.py, with Jev in
place of Julia-1; Holm adjustment within Jev's own families.
"""
import gzip, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, 'analysis')
from core import contrast, raw_diff, acc_vec, paired, holm, bh

L = lambda p: json.load(gzip.open(p, 'rt')) if p.endswith('.gz') else json.load(open(p))
OUT = 'predictions/msd1.json.gz'


def decide(body):
    if not isinstance(body, dict) or not body.get('answers'):
        return None, None
    a = list(body['answers'].values())[0]; t = a.get('type')
    if t == 'noul':
        p = a.get('noul')
        if p is None:
            return None, None
        return ('true' if p > 0.5 else 'false' if p < 0.5 else None), {'true': p, 'false': 1 - p}
    P = a.get('probabilities') or {}
    if t == 'score':
        if not P:
            return None, None
        m = max(P.values()); top = [k for k, v in P.items() if v == m]
        s = a.get('score')
        k = top[0] if len(top) == 1 or s is None else min(top, key=lambda k: (abs(float(k) - s), float(k)))
        return k, P
    return a.get('choice'), P


def collect(req_path, raw_path):
    first = {}
    for l in (gzip.open(raw_path, 'rt') if raw_path.endswith('.gz') else open(raw_path)):
        try:
            r = json.loads(l)
        except ValueError:
            continue
        if r.get('rep', 1) == 1 and r.get('status') == 200 and r['hash'] not in first:
            first[r['hash']] = r['body']
    out = {'model': 'microsoft/microsoft-decision-1', 'snapshots': {}, 'missing': 0, 'A': {}, 'typed': {'train': [], 'test': []},
           'val': {}, 'C': {}}
    for l in gzip.open(req_path, 'rt'):
        q = json.loads(l); body = first.get(q['hash'])
        if body is None:
            out['missing'] += 1
        else:
            out['snapshots'][body.get('model')] = out['snapshots'].get(body.get('model'), 0) + 1
        key, P = decide(body)
        task = q['task'].split(':')
        if task[1] == 'typed':
            pg = None if P is None else P.get(q['gold'])
            out['typed'][task[2]].append(dict(id=q['id'], type=q['qtype'], gold=q['gold'], pred=key,
                                              workflow=q['workflow'], p_gold=pg))
            continue
        opts, gold = q['opts'], q['gold']
        pred = None if key is None or key not in [str(j) for j in range(len(opts))] else opts[int(key)]
        pg = None if P is None else P.get(str(opts.index(gold)))
        rec = [q['id'], gold, pred, pg]
        if task[0] == 'A':
            out['A'].setdefault(task[1], {}).setdefault('A_' + task[2], []).append(rec)
        elif task[0] == 'B':
            out['val'].setdefault(task[1], []).append([q['id'].replace(':validation:', ':val:')] + rec[1:])
        else:                                    # C:<ds>:<split>:<orig_qwen|para_qwen|para_phi>
            out['C'].setdefault(task[1], {}).setdefault(f'{task[2]}:{task[3]}', []).append(rec)
    with gzip.open(OUT, 'wt') as f:
        json.dump(out, f)
    print('missing', out['missing'], 'snapshots', out['snapshots'])


def analyse(B):
    jv = L(OUT)
    e4 = L('predictions/r4_eval.json.gz'); ro = L('predictions/r6_rotten_eval.json.gz')
    qw = {k: {s: [[r[0], r[1], r[4], None] for r in v[s]] for s in v} for k, v in L('predictions/r8_e3_qwen_full.json.gz')['A'].items()}
    lay = L('predictions/r8_e3_laya_full.json.gz')
    DS = ["emotion", "agnews", "rotten", "yahoo", "banking77", "amazon", "imdb", "yelp", "massive", "wikitoxic"]

    def recs(model, ds, split):
        if model == 'msd1':
            return jv['A'][ds]['A_' + split]
        if model in ('julia', 'deberta_c', 'deberta_all33'):
            return (ro[model] if ds == 'rotten' else e4[model][ds])['A_' + split]
        if model == 'qwen7b':
            return qw[ds][split]
        if model == 'laya_ml':
            return None if ds == 'rotten' else lay['multilingual'][ds]['A_' + split]

    # ---------------- probe A, classification (audit.py with Jev as the audited model)
    rng = np.random.default_rng(20260930)
    gaps, dids, agree = [], [], []
    for ds in DS:
        for m in ['msd1', 'julia']:
            tr, te = recs(m, ds, 'train'), recs(m, ds, 'test')
            _, a, ya = acc_vec(tr); _, b, yb = acc_vec(te)
            gaps.append(dict(dataset=ds, model=m, acc_train=a.mean(), acc_test=b.mean(), raw_gap=raw_diff(a, b),
                             none_train=sum(x[2] is None for x in tr), none_test=sum(x[2] is None for x in te),
                             **contrast(a, ya, b, yb, B=B, rng=rng)))
        jtr, jte = recs('msd1', ds, 'train'), recs('msd1', ds, 'test')
        for s, rr in [('train', jtr), ('test', jte)]:
            mj = {r[0]: r[2] for r in recs('julia', ds, s)}
            agree.append(dict(dataset=ds, split=s, n=len(rr), agree_julia=float(np.mean([mj[r[0]] == r[2] for r in rr]))))
        for m in ['deberta_c', 'deberta_all33', 'qwen7b', 'laya_ml']:
            tr, te = recs(m, ds, 'train'), recs(m, ds, 'test')
            if tr is None:
                continue
            _, j1, c1, y1 = paired(jtr, tr); _, j2, c2, y2 = paired(jte, te)
            dids.append(dict(dataset=ds, control=m, msd1_gap_paired=contrast(j1, y1, j2, y2, B=2000, rng=rng)['est'],
                             control_gap_paired=contrast(c1, y1, c2, y2, B=2000, rng=rng)['est'],
                             **contrast(j1 - c1, y1, j2 - c2, y2, B=B, rng=rng)))
    G = pd.DataFrame(gaps); D = pd.DataFrame(dids)
    assert len(D) == 39, len(D)
    D['p_holm'] = holm(D.p); D['q_bh'] = bh(D.p)
    G.to_csv('results/msd1_audit_gaps.csv', index=False); D.to_csv('results/msd1_audit_did_39.csv', index=False)
    pd.DataFrame(agree).to_csv('results/msd1_julia_agreement.csv', index=False)

    # ---------------- exploratory: probe A on p(gold) (not in any Holm family)
    rng = np.random.default_rng(20261009)
    pg = []
    for ds in DS:
        tr, te = recs('msd1', ds, 'train'), recs('msd1', ds, 'test')
        a = np.array([r[3] if r[3] is not None else 0.0 for r in tr]); b = np.array([r[3] if r[3] is not None else 0.0 for r in te])
        pg.append(dict(dataset=ds, mean_pgold_train=a.mean(), mean_pgold_test=b.mean(),
                       **contrast(a, [r[1] for r in tr], b, [r[1] for r in te], B=B, rng=rng)))
    pd.DataFrame(pg).to_csv('results/msd1_pgold_probeA.csv', index=False)

    # ---------------- probe B: Emotion three-split (probe_bc.py)
    v5 = L('predictions/r5_val.json.gz'); rng = np.random.default_rng(11); rows = []
    S = {m: {'train': recs(m, 'emotion', 'train'), 'test': recs(m, 'emotion', 'test'),
             'val': jv['val']['emotion'] if m == 'msd1' else v5[m]['emotion']} for m in ['msd1', 'deberta_c', 'deberta_all33']}
    for a, b in [('test', 'val'), ('train', 'val'), ('train', 'test')]:
        _, x, yx = acc_vec(S['msd1'][a]); _, z, yz = acc_vec(S['msd1'][b])
        rows.append(dict(dataset='emotion', model='msd1', contrast=f'{a}-{b}', **contrast(x, yx, z, yz, B=B, rng=rng)))
    for m in ['deberta_c', 'deberta_all33']:
        for a, b in [('test', 'val'), ('train', 'val')]:
            _, j1, c1, y1 = paired(S['msd1'][a], S[m][a]); _, j2, c2, y2 = paired(S['msd1'][b], S[m][b])
            rows.append(dict(dataset='emotion', model='msd1-' + m, contrast=f'{a}-{b} DiD',
                             **contrast(j1 - c1, y1, j2 - c2, y2, B=B, rng=rng)))
    pd.DataFrame(rows).to_csv('results/msd1_three_split.csv', index=False)

    # ---------------- probe C (probe_bc.py): orig - paraphrase drop, train vs test
    P2 = L('predictions/r7_para2.json.gz'); rowsC = []
    for para, src, tag in [('Qwen2.5-7B-Instruct', e4, 'para_qwen'), ('Phi-3.5-mini-instruct', P2, 'para_phi')]:
        for ds in ['emotion', 'agnews', 'banking77', 'yelp', 'yahoo']:
            dm = {}
            for m in ['msd1', 'deberta_c']:
                dm[m] = {}
                for s in ['train', 'test']:
                    if m == 'msd1':
                        o, p = jv['C'][ds][f'{s}:orig_qwen'], jv['C'][ds][f'{s}:{tag}']
                    else:
                        o, p = src[m][ds][f'B_{s}_orig'], src[m][ds][f'B_{s}_para']
                    ids, co, cp, y = paired(o, p); dm[m][s] = (ids, co - cp, y)
            for m, d in dm.items():
                rowsC.append(dict(paraphraser=para, dataset=ds, model=m, kind='within-model',
                                  drop_train=100 * d['train'][1].mean(), drop_test=100 * d['test'][1].mean(),
                                  **contrast(d['train'][1], d['train'][2], d['test'][1], d['test'][2], B=B, rng=rng)))
            J, Cc = dm['msd1'], dm['deberta_c']; vv = {}
            for s in ['train', 'test']:
                mc = dict(zip(Cc[s][0], Cc[s][1])); ids = [i for i in J[s][0] if i in mc]
                mj = dict(zip(J[s][0], J[s][1])); my = dict(zip(J[s][0], J[s][2]))
                vv[s] = (np.array([mj[i] - mc[i] for i in ids]), np.array([my[i] for i in ids]))
            rowsC.append(dict(paraphraser=para, dataset=ds, model='msd1-deberta_c', kind='between-model DiD',
                              **contrast(vv['train'][0], vv['train'][1], vv['test'][0], vv['test'][1], B=B, rng=rng)))
    C = pd.DataFrame(rowsC); fam = C.kind == 'between-model DiD'
    C.loc[fam, 'p_holm'] = holm(C.loc[fam, 'p']); C.loc[fam, 'q_bh'] = bh(C.loc[fam, 'p'])
    C.to_csv('results/msd1_probeC.csv', index=False)
    print(D[['dataset', 'control', 'est', 'lo', 'hi', 'p_holm']].round(3).to_string())


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'collect':
        collect(sys.argv[2], sys.argv[3])
    else:
        analyse(int(sys.argv[1]) if len(sys.argv) > 1 else 50_000)
