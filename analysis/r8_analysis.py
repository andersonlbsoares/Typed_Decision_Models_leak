"""Third-revision analyses (runs in eval_scripts/r8). Run from the artifact root.
Usage: python analysis/r8_analysis.py [B]      (default B = 50,000)
Writes results/r8_e3_overlap_agreement.csv, r8_e4_agreement.csv, r8_e4_probeA_by_mode.csv,
r8_e1_power.csv, r8_e1_membership_by_seed.csv, r8_e1_base_vs_filler.csv, r8_summary.json.

E3  The 39-contrast family itself is computed by audit.py on the complete samples; here only the
    decision agreement with the earlier 1,000-record reference runs on the overlapping items.
E4  Decision agreement between execution modes and against the stored audit run; label-standardized
    probe A per mode (typed-decisions: workflow x question x gold strata, state-cluster bootstrap
    stratified by workflow, as typed.py; classification: split x gold, as audit.py).
E1  Probe A on the audited sample after fine-tuning with exposed fraction rho, as a difference in
    differences against the same-budget rho = 0 control of the same seed (item values averaged over
    seeds before the stratified bootstrap, as grid.py); also against the pre-FT checkpoint, and the
    within-sample membership contrast (exposed vs unexposed audited training items, per seed)."""
import gzip, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, 'analysis')
from core import contrast, acc_vec, paired, holm, bh

B = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000
L = lambda p: json.load(gzip.open(p, 'rt')) if p.endswith('.gz') else json.load(open(p))
P = lambda f: os.path.join('predictions', f)
os.makedirs('results', exist_ok=True)
DS = ["emotion", "agnews", "rotten", "yahoo", "banking77", "amazon", "imdb", "yelp", "massive", "wikitoxic"]
e4 = L('predictions/r4_eval.json.gz'); ro = L('predictions/r6_rotten_eval.json.gz')
qw_old = L('predictions/r6_qwen.json.gz'); lay_old = L('predictions/r6_laya.json.gz')
out_summary = {}


def agree(a, b):
    mb = {r[0]: r[2] for r in b}
    ids = [r[0] for r in a if r[0] in mb]
    return len(ids), float(np.mean([dict((r[0], r[2]) for r in a)[i] == mb[i] for i in ids])) if ids else np.nan


# ---------------------------------------------------------------- E3
def e3():
    qn = L(P('r8_e3_qwen_full.json.gz'))['A']; ln = L(P('r8_e3_laya_full.json.gz'))['multilingual']
    ov = []
    for ds in DS:
        for split in ['train', 'test']:
            n, a = agree([[r[0], r[1], r[4]] for r in qn[ds][split]], qw_old['A'][ds][split]); ov.append(dict(dataset=ds, split=split, model='qwen7b', n_overlap=n, agreement=a))
            n, a = agree(ln[ds]['A_' + split], lay_old['multilingual'][ds]['A_' + split]); ov.append(dict(dataset=ds, split=split, model='laya_ml', n_overlap=n, agreement=a))
            ov.append(dict(dataset=ds, split=split, model='qwen7b_top20_vs_argmax', n_overlap=len(qn[ds][split]),
                           agreement=float(np.mean([r[2] == r[4] for r in qn[ds][split]]))))
    O = pd.DataFrame(ov); O.to_csv('results/r8_e3_overlap_agreement.csv', index=False)
    out_summary['e3'] = dict(min_overlap_agreement=O.groupby('model').agreement.min().round(4).to_dict(),
                             mean_overlap_agreement=O.groupby('model').agreement.mean().round(4).to_dict())
    print(O.groupby('model').agreement.describe().round(4))


# ---------------------------------------------------------------- E4
def cluster_contrast(tr, te, vtr, vte, B, rng):
    """typed.py primary estimator: state-cluster bootstrap within workflow; absent strata dropped."""
    def agg(t, v, strata):
        st, si = np.unique(t['state'], return_inverse=True); col = {c: k for k, c in enumerate(strata)}
        S = np.zeros((len(st), len(strata))); N = np.zeros_like(S)
        for i, (s, c) in enumerate(zip(si, t['strat'])):
            if c in col:
                S[s, col[c]] += v[i]; N[s, col[c]] += 1
        swf = np.array([t['wf'][np.where(si == k)[0][0]] for k in range(len(st))])
        return S, N, swf

    def bw(swf, b):
        W = np.zeros((b, len(swf)))
        for w in np.unique(swf):
            idx = np.where(swf == w)[0]; W[:, idx] = rng.multinomial(len(idx), np.ones(len(idx)) / len(idx), size=b)
        return W
    strata = sorted(set(tr['strat']) & set(te['strat']))
    S1, N1, wf1 = agg(tr, vtr, strata); S2, N2, wf2 = agg(te, vte, strata)
    n = N1.sum(0) + N2.sum(0); w = n / n.sum()
    est = w @ (S1.sum(0) / N1.sum(0) - S2.sum(0) / N2.sum(0)); bs = np.empty(B)
    for b0 in range(0, B, 2000):
        bb = min(2000, B - b0); W1, W2 = bw(wf1, bb), bw(wf2, bb)
        s1, n1, s2, n2 = W1 @ S1, W1 @ N1, W2 @ S2, W2 @ N2; ok = (n1 > 0) & (n2 > 0)
        with np.errstate(invalid='ignore', divide='ignore'):
            d = np.where(ok, s1 / np.where(n1 > 0, n1, 1) - s2 / np.where(n2 > 0, n2, 1), 0)
        ww = np.where(ok, w, 0); ww = ww / ww.sum(1, keepdims=True); bs[b0:b0 + bb] = (ww * d).sum(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(est=100 * est, lo=100 * lo, hi=100 * hi, strata_used=len(strata))


def e4_():
    R = {}
    for tag in ['gpu', 'cpu']:
        if os.path.exists(P(f'r8_e4_{tag}.json.gz')):
            R.update(L(P(f'r8_e4_{tag}.json.gz')))
    r2 = L('predictions/r2_julia_nli.json.gz')['typed']
    stored = {'typed/train': [[x['id'], x['gold'], x['pred']] for x in r2['train']],
              'typed/test': [[x['id'], x['gold'], x['pred']] for x in r2['test']]}
    for t in ['emotion', 'agnews']:
        for s in ['train', 'test']:
            stored[f'{t}/{s}'] = e4['julia'][t]['A_' + s]
    meta = {x['id']: x for s in ['train', 'test'] for x in r2[s]}
    modes = [m for m in ['gpu_bs64', 'gpu_bs1', 'gpu_fp32_bs1', 'cpu_fp32_bs1'] if m in R]
    rows, gaps = [], []
    rng = np.random.default_rng(11)
    ref_mode = 'cpu_fp32_bs1' if 'cpu_fp32_bs1' in modes else modes[-1]
    for k in stored:
        for m in modes:
            if k not in R[m]:
                continue
            n, a_st = agree(R[m][k], stored[k]); _, a_ref = agree(R[m][k], R[ref_mode][k]) if k in R[ref_mode] else (0, np.nan)
            acc = float(np.mean([r[2] == r[1] for r in R[m][k]]))
            rows.append(dict(task=k, mode=m, n=n, acc=acc, agree_stored_audit=a_st, agree_ref=a_ref, ref=ref_mode,
                             seconds=R[m]['seconds'].get(k)))
    for m in modes + ['stored']:
        src = stored if m == 'stored' else R[m]
        for t in ['emotion', 'agnews']:
            if f'{t}/train' in src and f'{t}/test' in src:
                _, a, ya = acc_vec(src[f'{t}/train']); _, b, yb = acc_vec(src[f'{t}/test'])
                gaps.append(dict(task=t, mode=m, **{k: v for k, v in contrast(a, ya, b, yb, B=B, rng=rng).items() if k in ('est', 'lo', 'hi', 'strata_used')}))
        if 'typed/train' in src and 'typed/test' in src:
            T, V = {}, {}
            for s in ['train', 'test']:
                rec = src[f'typed/{s}']; ids = [r[0] for r in rec]
                T[s] = dict(state=np.array([i.rsplit(':', 1)[0] for i in ids]), wf=np.array([meta[i]['workflow'] for i in ids]),
                            strat=np.array([f"{meta[i]['workflow']}|{i.rsplit(':', 1)[1]}|{meta[i]['gold']}" for i in ids]))
                V[s] = np.array([float(r[2] == r[1]) for r in rec])
            gaps.append(dict(task='typed', mode=m, **cluster_contrast(T['train'], T['test'], V['train'], V['test'], min(B, 20000), rng)))
    A, Gp = pd.DataFrame(rows), pd.DataFrame(gaps)
    A.to_csv('results/r8_e4_agreement.csv', index=False); Gp.to_csv('results/r8_e4_probeA_by_mode.csv', index=False)
    out_summary['e4'] = dict(modes=modes, min_agree_gpu_bs64_vs_stored=float(A[A['mode'] == 'gpu_bs64'].agree_stored_audit.min()) if 'gpu_bs64' in modes else None,
                             flip_rate_stored_vs_ref=float(1 - A[A['mode'] == ref_mode].agree_stored_audit.mean()))
    print(A.round(4).to_string()); print(Gp.round(2).to_string())


# ---------------------------------------------------------------- E1
def e1():
    rows, mem = [], []
    for key in ['emotion', 'agnews', 'yelp', 'yahoo']:
        f = P(f'r8_e1_{key}.json.gz')
        if not os.path.exists(f):
            continue
        R = L(f); M = R['meta']; ytr, yte = np.array(M['train_gold']), np.array(M['test_gold'])
        corr = lambda p, y: (np.array(p) == y).astype(float)
        base_tr, base_te = corr(R['base']['train'], ytr), corr(R['base']['test'], yte)
        runs = {k: v for k, v in R.items() if '|' in k}
        seeds = sorted({int(k.split('|')[0]) for k in runs}); rhos = sorted({float(k.split('|')[1]) for k in runs})
        rng = np.random.default_rng(5)
        for ep in [str(e) for e in M['epochs']]:
            for rho in rhos:
                ss = [s for s in seeds if f'{s}|{rho:g}' in runs and f'{s}|0' in runs and ep in runs[f'{s}|{rho:g}']]
                if not ss:
                    continue
                Ctr = np.stack([corr(runs[f'{s}|{rho:g}'][ep]['train'], ytr) for s in ss]); Cte = np.stack([corr(runs[f'{s}|{rho:g}'][ep]['test'], yte) for s in ss])
                Ztr = np.stack([corr(runs[f'{s}|0'][ep]['train'], ytr) for s in ss]); Zte = np.stack([corr(runs[f'{s}|0'][ep]['test'], yte) for s in ss])
                did0 = contrast((Ctr - Ztr).mean(0), ytr, (Cte - Zte).mean(0), yte, B=B, rng=rng)
                didb = contrast(Ctr.mean(0) - base_tr, ytr, Cte.mean(0) - base_te, yte, B=B, rng=rng)
                raw = contrast(Ctr.mean(0), ytr, Cte.mean(0), yte, B=B, rng=rng)
                per = [contrast(Ctr[i] - Ztr[i], ytr, Cte[i] - Zte[i], yte, B=5000, rng=rng) for i in range(len(ss))]
                rows.append(dict(dataset=key, epochs=int(ep), rho=rho, n_seeds=len(ss), acc_train=Ctr.mean(), acc_test=Cte.mean(),
                                 probeA=raw['est'], probeA_lo=raw['lo'], probeA_hi=raw['hi'],
                                 did_rho0=did0['est'], did_rho0_lo=did0['lo'], did_rho0_hi=did0['hi'], did_rho0_p=did0['p'],
                                 did_base=didb['est'], did_base_lo=didb['lo'], did_base_hi=didb['hi'],
                                 seeds_detected=sum(r['lo'] > 0 for r in per), seed_sd=float(np.std([r['est'] for r in per], ddof=1)) if len(per) > 1 else np.nan))
                if 0 < rho < 1:
                    for i, s in enumerate(ss):
                        ex = np.zeros(len(ytr), bool); ex[runs[f'{s}|{rho:g}']['exposed']] = True
                        c = Ctr[i]; r = contrast(c[ex], ytr[ex], c[~ex], ytr[~ex], B=5000, rng=rng)
                        mem.append(dict(dataset=key, epochs=int(ep), rho=rho, seed=s, n_exposed=int(ex.sum()), **{k: r[k] for k in ('est', 'lo', 'hi', 'coverage_1', 'coverage_2')}))
    Dp, Mm = pd.DataFrame(rows), pd.DataFrame(mem)
    Dp.to_csv('results/r8_e1_power.csv', index=False); Mm.to_csv('results/r8_e1_membership_by_seed.csv', index=False)
    if len(Dp):
        print(Dp[['dataset', 'epochs', 'rho', 'n_seeds', 'probeA', 'did_rho0', 'did_rho0_lo', 'did_rho0_hi', 'did_base', 'seeds_detected']].round(2).to_string())
        out_summary['e1'] = dict(detected_pooled=Dp.assign(det=Dp.did_rho0_lo > 0).groupby(['epochs', 'rho']).det.sum().to_dict().__repr__())
    return Dp, Mm


def e1_base():
    """Julia-1's own probe A in the E1 setting (text[:1200]) before fine-tuning and after fine-tuning on
    filler items only (rho = 0, epoch 3; item values averaged over the 5 seeds)."""
    rows = []; rng = np.random.default_rng(9)
    for key in ['emotion', 'agnews', 'yelp', 'yahoo']:
        f = P(f'r8_e1_{key}.json.gz')
        if not os.path.exists(f):
            continue
        R = L(f); M = R['meta']; ytr, yte = np.array(M['train_gold']), np.array(M['test_gold'])
        corr = lambda p, y: (np.array(p) == y).astype(float)
        b0 = contrast(corr(R['base']['train'], ytr), ytr, corr(R['base']['test'], yte), yte, B=B, rng=rng)
        seeds = sorted({int(k.split('|')[0]) for k in R if k.endswith('|0')})
        Ztr = np.mean([corr(R[f'{s}|0']['3']['train'], ytr) for s in seeds], 0); Zte = np.mean([corr(R[f'{s}|0']['3']['test'], yte) for s in seeds], 0)
        z = contrast(Ztr, ytr, Zte, yte, B=B, rng=rng)
        rows.append(dict(dataset=key, base_acc_train=corr(R['base']['train'], ytr).mean(), base_acc_test=corr(R['base']['test'], yte).mean(),
                         base_est=b0['est'], base_lo=b0['lo'], base_hi=b0['hi'], rho0_acc_train=Ztr.mean(), rho0_acc_test=Zte.mean(),
                         rho0_est=z['est'], rho0_lo=z['lo'], rho0_hi=z['hi'], n_seeds=len(seeds)))
    pd.DataFrame(rows).to_csv('results/r8_e1_base_vs_filler.csv', index=False)
    print(pd.DataFrame(rows).round(3).to_string())


if __name__ == '__main__':
    for name, fn in [('e3', e3), ('e4', e4_), ('e1', e1), ('e1_base', e1_base)]:
        try:
            fn()
        except FileNotFoundError as ex:
            print(f'[{name}] skipped: {ex}')
    json.dump(out_summary, open('results/r8_summary.json', 'w'), indent=1, default=str)
    print(json.dumps(out_summary, indent=1, default=str))
