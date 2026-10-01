"""MASSIVE (English) label universe: audit (58 gold labels) vs BTZSC hypotheses (59 options).
mteb/amazon_massive_intent rev. 940fd47a (en) has 60 intents. BTZSC (btzsc/btzsc rev. fef2a2ac, `massive`) has
59 hypotheses: it has none for `general_quirky`. Our sampler (eval_scripts/r4_data.py) maps dataset label names to
BTZSC class names read from the BTZSC test file; `cooking_query` has a hypothesis but no BTZSC test example, so its
name is not recovered and its 4 training items are left unmapped (it remains available as a distractor option).
Hence 58 gold labels in the audit, 59 candidate hypotheses in the 20-option subsets and in the BTZSC table.
Needs data/btzsc_massive_test.parquet (btzsc/btzsc @ fef2a2ac, massive/test-00000-of-00001.parquet) and
data/massive_en_{train,test}.json.gz (mteb/amazon_massive_intent @ 940fd47a, {split}/en.json.gz).
Writes results/massive_label_reconciliation.csv."""
import collections, gzip, json, re
import pandas as pd, pyarrow.parquet as pq
norm = lambda s: re.sub(r"[^a-z0-9]+", "", str(s).lower())
t = pq.read_table('data/btzsc_massive_test.parquet').to_pydict()
k = next(i for i in range(1, len(t['text'])) if t['text'][i] != t['text'][0])
hyps = [t['hypothesis'][i] for i in range(k)]; names = [None] * k
for s in range(len(t['text']) // k):
    for c in range(k):
        if t['labels'][s * k + c] == 1 and names[c] is None:
            names[c] = t['label_text'][s * k + c]
rows = []
for split in ['train', 'test']:
    p = f'data/massive_en_{split}.json.gz'
    raw = gzip.open(p, 'rt').read().strip()
    recs = json.loads(raw) if raw.startswith('[') else [json.loads(l) for l in raw.splitlines()]
    cnt = collections.Counter(r['label_text'] for r in recs)
    mapped = {norm(n) for n in names if n}
    for lab, n in sorted(cnt.items()):
        rows.append(dict(split=split, label=lab, n_items=n, mapped_to_btzsc_class=norm(lab) in mapped))
R = pd.DataFrame(rows); R.to_csv('results/massive_label_reconciliation.csv', index=False)
print('BTZSC hypotheses:', k, '| hypotheses without a recovered class name:', [hyps[i] for i in range(k) if names[i] is None])
print('dataset intents:', R.label.nunique(), '| mapped:', R[R.mapped_to_btzsc_class].label.nunique())
print(R[~R.mapped_to_btzsc_class].to_string(index=False))
