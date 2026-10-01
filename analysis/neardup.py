"""typed-decisions: TF-IDF (char_wb 5-gram) max similarity of each test state to the 1,200 training
states, train->train with self-matches excluded (1,199 candidates). Association between Julia-1
test correctness and similarity: within-workflow OLS slope (per 0.1 similarity) on question-level
correctness, state-cluster bootstrap stratified by workflow; same for Qwen and for Julia minus Qwen."""
import gzip, json
import numpy as np, pandas as pd, pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
TT = {s: pq.read_table(f'data/typed_{s}.parquet').to_pylist() for s in ['train', 'test']}
stx = lambda c: json.dumps(json.loads(c['state']), ensure_ascii=False, sort_keys=True)
tr = [stx(c) for c in TT['train']]; te = [stx(c) for c in TT['test']]
vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(5, 5), min_df=2, sublinear_tf=True).fit(tr + te)
Xtr, Xte = vec.transform(tr), vec.transform(te)
sim_te = (Xte @ Xtr.T).toarray().max(1)
Srr = (Xtr @ Xtr.T).toarray(); np.fill_diagonal(Srr, 0); sim_tr = Srr.max(1)
exact = sum(t in set(tr) for t in te)
sid = [c['id'] for c in TT['test']]
r2 = json.load(gzip.open('predictions/r2_julia_nli.json.gz', 'rt')); q3 = json.load(open('predictions/r3_qwen_typed.json'))
def frame(recs):
    x = pd.DataFrame(recs); x['ok'] = (x.pred == x.gold).astype(float); x['state'] = x.id.str.rsplit(':', n=1).str[0]; return x
J = frame(r2['typed']['test']); Q = frame(q3['test'])
m = dict(zip(sid, sim_te)); J['sim'] = J.state.map(m); Q['sim'] = Q.state.map(m)
assert J.sim.notna().all()
J = J.merge(Q[['id', 'ok']].rename(columns={'ok': 'ok_q'}), on='id')
J['diff'] = J.ok - J.ok_q
wf = J.groupby('state').workflow.first()
states = wf.index.values
def slope(df, col):
    d = df.copy(); d['x'] = d.sim - d.groupby('workflow').sim.transform('mean'); d['y'] = d[col] - d.groupby('workflow')[col].transform('mean')
    return 0.1 * (d.x * d.y).sum() / (d.x ** 2).sum()
rng = np.random.default_rng(3); B = 5000
g = {s: J.index[J.state == s].values for s in states}
bys = {w: states[wf.values == w] for w in wf.unique()}
out = []
for col, name in [('ok', 'Julia-1'), ('ok_q', 'Qwen2.5-7B'), ('diff', 'Julia-1 minus Qwen')]:
    est = slope(J, col); bs = []
    for _ in range(B):
        pick = np.concatenate([rng.choice(v, len(v)) for v in bys.values()])
        idx = np.concatenate([g[s] for s in pick]); d = J.loc[idx].reset_index(drop=True)
        bs.append(slope(d, col))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    out.append(dict(model=name, slope_per_0p1=100 * est, lo=100 * lo, hi=100 * hi))
R = pd.DataFrame(out); R.to_csv('results/neardup_slope.csv', index=False)
q = pd.qcut(pd.Series(sim_te, index=sid), 4, labels=False)
J['q'] = J.state.map(q)
qq = J.groupby('q').agg(lo=('sim', 'min'), hi=('sim', 'max'), julia=('ok', 'mean'), qwen=('ok_q', 'mean'), n=('ok', 'size'))
qq.to_csv('results/neardup_quartiles.csv')
print(R.round(2).to_string()); print(qq.round(3).to_string())
print('exact', exact, 'median test->train', np.median(sim_te).round(3), 'median train->train(self-excluded)', np.median(sim_tr).round(3))
print('quantiles test', np.percentile(sim_te, [0, 25, 50, 75, 100]).round(3), 'train', np.percentile(sim_tr, [0, 25, 50, 75, 100]).round(3))
