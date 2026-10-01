"""Sample order and label support (Appendix, Data).  eval_scripts/run_julia.balanced_indices selects 2,000 items per
split by class-balanced round-robin over per-class shuffled lists and then RETURNS THE SELECTED INDICES SORTED BY
SOURCE-FILE POSITION; the stored samples (and all prediction files) follow that order.  The Qwen-7B / Laya-ML
subsets are the first 1,000 stored records, i.e. the 1,000 selected items with the smallest source positions, so in
source files grouped by label (IMDB, Banking77, MASSIVE, Rotten Tomatoes train) they lose labels.  The controlled
fine-tuning grid uses the first n stored training records and test records 0-499 / 500-999 (500-1,999).
Outputs: results/sample_order_check.csv, results/label_support_subsets.csv, results/label_counts_by_subset.csv."""
import gzip, json, collections
import pandas as pd
L = lambda p: json.load(gzip.open(p, 'rt'))
e4 = L('predictions/r4_eval.json.gz'); qw = L('predictions/r6_qwen.json.gz'); lay = L('predictions/r6_laya.json.gz'); ro = L('predictions/r6_rotten_eval.json.gz')
DS = ['emotion', 'agnews', 'rotten', 'yahoo', 'banking77', 'amazon', 'imdb', 'yelp', 'massive', 'wikitoxic']
full = lambda ds, sp: (ro['julia'] if ds == 'rotten' else e4['julia'][ds])['A_' + sp]
order, sup, cnt = [], [], []
for ds in DS:
    for sp in ['train', 'test']:
        f = full(ds, sp); idx = [int(r[0].rsplit(':', 1)[1]) for r in f]
        q = [r[0] for r in qw['A'][ds][sp]]
        l = None if ds == 'rotten' else [r[0] for r in lay['multilingual'][ds]['A_' + sp]]
        order.append(dict(dataset=ds, split=sp, n=len(f), sorted_by_source_position=all(a < b for a, b in zip(idx, idx[1:])),
                          qwen_subset_is_first_1000=q == [r[0] for r in f[:len(q)]], n_qwen=len(q),
                          laya_subset_is_first_1000=None if l is None else l == [r[0] for r in f[:len(l)]]))
    task = set(r[1] for sp in ['train', 'test'] for r in full(ds, sp))
    for sub, k in [('full sample (2,000)', None), ('Qwen-7B / Laya-ML subset (first 1,000)', 1000)]:
        s = {sp: set(r[1] for r in full(ds, sp)[:k]) for sp in ['train', 'test']}
        sup.append(dict(dataset=ds, subset=sub, labels_task=len(task), labels_train=len(s['train']), labels_test=len(s['test']),
                        labels_common=len(s['train'] & s['test'])))
        for sp in ['train', 'test']:
            for lab, n in sorted(collections.Counter(r[1] for r in full(ds, sp)[:k]).items()):
                cnt.append(dict(dataset=ds, split=sp, subset=sub, label=lab, n=n))
for ds in ['emotion', 'agnews', 'yelp', 'yahoo']:
    tr, te = full(ds, 'train'), full(ds, 'test'); task = set(r[1] for r in tr)
    subs = [(f'grid: exposed training items (first {n})', 'train', tr[:n]) for n in [25, 50, 100, 250, 1000]] + \
           [('grid: test items 0-499 (exposed in S2)', 'test', te[:500]), ('grid: test items 500-999 (held out)', 'test', te[500:1000]),
            ('grid: test items 500-1,999 (held out, 1,500)', 'test', te[500:2000])]
    for name, sp, recs in subs:
        c = collections.Counter(r[1] for r in recs)
        sup.append(dict(dataset=ds, subset=name, labels_task=len(task), labels_train=len(c) if sp == 'train' else None,
                        labels_test=len(c) if sp == 'test' else None, labels_common=None, min_count=min(c.values())))
        for lab, n in sorted(c.items()):
            cnt.append(dict(dataset=ds, split=sp, subset=name, label=lab, n=n))
pd.DataFrame(order).to_csv('results/sample_order_check.csv', index=False)
pd.DataFrame(sup).to_csv('results/label_support_subsets.csv', index=False)
pd.DataFrame(cnt).to_csv('results/label_counts_by_subset.csv', index=False)
print(pd.DataFrame(order).to_string()); print(pd.DataFrame(sup).to_string())
