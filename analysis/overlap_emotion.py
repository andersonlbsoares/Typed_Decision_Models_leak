"""DAIR Emotion split-overlap sensitivity (Section 7). Needs dair-ai/emotion rev. cab853a1 (config `split`) parquet files
in data/emotion_{train,test,validation}.parquet. Matching uses the Appendix normalization (lower-case, punctuation
removed, whitespace collapsed); verbatim matching is reported as well. Contrasts use the primary estimator
(core.contrast, split x label stratified bootstrap, B = 50,000) on the current audit samples.
Outputs: results/emotion_split_overlap.csv (contrasts), results/emotion_overlap_items.csv (matched items)."""
import gzip, json, re, sys, collections
import numpy as np, pandas as pd
sys.path.insert(0, 'analysis')
from core import contrast, acc_vec, paired
L = lambda p: json.load(gzip.open(p, 'rt'))
e4 = L('predictions/r4_eval.json.gz'); qw = L('predictions/r6_qwen.json.gz'); lay = L('predictions/r6_laya.json.gz')
E = {s: pd.read_parquet(f'data/emotion_{s}.parquet') for s in ['train', 'test', 'validation']}
def norm(t):
    t = t.lower(); t = re.sub(r'[^\w\s]', '', t); return re.sub(r'\s+', ' ', t).strip()
items, counts = [], []
for mode, f in [('verbatim', lambda t: t), ('normalized', norm)]:
    trmap = collections.defaultdict(set)
    for t, l in zip(E['train'].text, E['train'].label): trmap[f(t)].add(int(l))
    for s in ['test', 'validation']:
        h = [(i, int(l), sorted(trmap[f(t)])) for i, (t, l) in enumerate(zip(E[s].text, E[s].label)) if f(t) in trmap]
        counts.append(dict(matching=mode, split=s, n_matched=len(h), n_label_conflict=sum(l not in ls for _, l, ls in h)))
        if mode == 'normalized':
            items += [dict(split=s, idx=i, label=l, train_labels=';'.join(map(str, ls))) for i, l, ls in h]
print(pd.DataFrame(counts).to_string())
tr_idx = [int(r[0].rsplit(':', 1)[1]) for r in e4['julia']['emotion']['A_train']]
other = set(norm(t) for t in E['test'].text) | set(norm(t) for t in E['validation'].text)
ov_train = {f'emotion:train:{i}' for i in tr_idx if norm(E['train'].text[i]) in other}
ov_test = {f"emotion:test:{it['idx']}" for it in items if it['split'] == 'test'}
items += [dict(split='train_sample', idx=int(i.rsplit(':', 1)[1]), label=None, train_labels=None) for i in sorted(ov_train)]
def recs(m, sp):
    if m == 'qwen7b': return [r[:3] + [None] for r in qw['A']['emotion'][sp]]
    if m == 'laya_ml': return lay['multilingual']['emotion']['A_' + sp]
    return e4[m]['emotion']['A_' + sp]
rng = np.random.default_rng(20261001); out = []
for variant, ex_tr, ex_te in [('all items', set(), set()), ('excluding test items matched in the training split', set(), ov_test),
                              ('also excluding training-sample items matched in test/validation', ov_train, ov_test)]:
    tr = [r for r in recs('julia', 'train') if r[0] not in ex_tr]; te = [r for r in recs('julia', 'test') if r[0] not in ex_te]
    _, a, ya = acc_vec(tr); _, b, yb = acc_vec(te)
    out.append(dict(variant=variant, contrast='Julia-1 gap', **contrast(a, ya, b, yb, rng=rng)))
    for m in ['deberta_c', 'deberta_all33', 'qwen7b', 'laya_ml']:
        _, j1, c1, y1 = paired(tr, recs(m, 'train')); _, j2, c2, y2 = paired(te, recs(m, 'test'))
        out.append(dict(variant=variant, contrast='Julia-1 minus ' + m, **contrast(j1 - c1, y1, j2 - c2, y2, rng=rng)))
R = pd.DataFrame(out); R.to_csv('results/emotion_split_overlap.csv', index=False)
pd.DataFrame(items).to_csv('results/emotion_overlap_items.csv', index=False)
pd.DataFrame(counts).to_csv('results/emotion_overlap_counts.csv', index=False)
print(R[['variant', 'contrast', 'est', 'lo', 'hi', 'p', 'n1', 'n2']].round(2).to_string())
