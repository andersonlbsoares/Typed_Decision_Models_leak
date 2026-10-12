import os, numpy as np, pandas as pd
os.makedirs('paper/tables', exist_ok=True)
def s(x, d=1):
    x = float(x); v = round(x, d)
    if v == 0 and abs(x) >= 0.005:
        # do not hide the sign of a value that rounds to zero at d decimals (e.g. a CI limit of -0.04)
        v = round(x, 2); t = f"{abs(v):.2f}"
        return ('+' if v > 0 else '$-$') + t
    if v == 0: v = 0.0
    t = f"{abs(v):.{d}f}"
    return ('+' if v > 0 else ('$-$' if v < 0 else '')) + t
TS = '\\footnotesize'   # table font: 8 pt in the 10 pt SCITEPRESS layout (no \resizebox anywhere)
def ci(lo, hi): return f"[{s(lo)}, {s(hi)}]"
def pf(p):
    return '$<$0.001' if p < 0.001 else f"{p:.3f}"
W = lambda name, txt: open(f'paper/tables/{name}.tex', 'w').write(txt.replace('\\centering\\scriptsize', '\\centering' + TS))
DSN = {'emotion': 'DAIR Emotion', 'agnews': 'AG News', 'rotten': 'Rotten Tomatoes', 'yahoo': 'Yahoo Answers', 'banking77': 'Banking77',
       'amazon': 'Amazon Polarity', 'imdb': 'IMDB', 'yelp': 'Yelp Polarity', 'massive': 'MASSIVE', 'wikitoxic': 'WikiToxic'}
CTL = {'deberta_c': 'DeBERTa-c', 'deberta_all33': 'DeBERTa all-33', 'qwen7b': 'Qwen-7B', 'laya_ml': 'Laya-ML'}
DSS = {'emotion': 'Emotion', 'agnews': 'AG News', 'rotten': 'Rotten T.', 'yahoo': 'Yahoo', 'banking77': 'Banking77',
       'amazon': 'Amazon', 'imdb': 'IMDB', 'yelp': 'Yelp', 'massive': 'MASSIVE', 'wikitoxic': 'WikiToxic'}
CTS = {'deberta_c': 'DeB-c', 'deberta_all33': 'all-33', 'qwen7b': 'Qwen', 'laya_ml': 'Laya'}

# ---------------- grid
G = pd.read_csv('results/controlled_grid.csv'); M = G[G.grid == 'main']
def cnt(model, probe, sc, corr):
    d = M[(M.model == model) & (M.probe == probe) & (M.scenario == sc) & (M.reference_corrected == corr)]
    return f"{int(d.positive_ci.sum())}/{len(d)}"
rows = []
for model, lab in [('DeBERTa-v3', 'DeBERTa-v3'), ('Julia-1', 'Julia-1')]:
    for probe, pl, fire, other in [('A', 'A', 'S1', 'S2'), ('B', 'B', 'S2', 'S1')]:
        rows.append(f"{lab if probe=='A' else ''} & {pl} & {cnt(model,probe,fire,False)} & {cnt(model,probe,other,False)} & {cnt(model,probe,fire,True)} & {cnt(model,probe,other,True)} \\\\")
    rows.append('\\midrule')
rows = rows[:-1]
W('tab_grid', r"""\begin{table}[t]
\centering\scriptsize
\caption{Controlled exposure, label-standardized: number of configurations whose 95\% CI is above zero. $n\in\{250,1000\}$ exposed training items, 1 or 3 epochs, 3 seeds; DeBERTa-v3 on Emotion, AG News and Yelp (12 configurations per cell), Julia-1 on Yelp and Yahoo (8). Probe A: exposed training vs.\ unexposed test items; probe B: exposed vs.\ held-out test items. S1: only training items exposed; S2: training items plus the first 500 test items. ``Target'' is the scenario each probe is designed for (A: S1; B: S2); ``other'' is the opposite scenario, in which the compared splits receive the same or no additional exposure, so a positive contrast there is not a false positive for the target exposure and these counts are not estimates of a false-positive rate. ``vs.\ pre-FT'' subtracts the pre-fine-tuning checkpoint's statistic on the same items.}
\label{tab:grid}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llcccc}
\toprule
& & \multicolumn{2}{c}{Uncorrected} & \multicolumn{2}{c}{vs.\ pre-FT} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}
Model & Probe & target & other & target & other \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------- sensitivity (probe A S1 across n) + probe B held-out size
Sx = G[(G.model == 'DeBERTa-v3') & (G.probe == 'A') & (G.scenario == 'S1') & G.reference_corrected & G.grid.isin(['main', 'small_n'])]
lines = []
for ds in ['emotion', 'agnews', 'yelp']:
    cells = []
    for n in [25, 50, 100, 250, 1000]:
        for ep in [1, 3]:
            r = Sx[(Sx.dataset == ds) & (Sx.n_exposed == n) & (Sx.epochs == ep)].iloc[0]
            v = s(r.est); cells.append(f"\\textbf{{{v}}}" if r.lo > 0 else v)
    lines.append(DSN[ds] + ' & ' + ' & '.join(cells) + ' \\\\')
Bx = G[(G.model == 'DeBERTa-v3') & (G.probe == 'B') & (G.scenario == 'S2') & G.reference_corrected & (G.n_exposed == 250)]
lb = []
for ds in ['emotion', 'agnews', 'yelp']:
    cells = []
    for grid in ['main', 'heldout_1500']:
        for ep in [1, 3]:
            r = Bx[(Bx.dataset == ds) & (Bx.grid == grid) & (Bx.epochs == ep)].iloc[0]
            v = s(r.est); cells.append((f"\\textbf{{{v}}}" if r.lo > 0 else v) + ' ' + ci(r.lo, r.hi))
    lb.append(DSN[ds] + ' & ' + ' & '.join(cells) + ' \\\\')
W('tab_sens', r"""\begin{table*}[t]
\centering\scriptsize
\caption{Sensitivity under controlled exposure (DeBERTa-v3, 3 seeds, label-standardized contrast against the pre-fine-tuning checkpoint, pp). Top: probe A (exposed training items vs.\ 500 unexposed test items, S1) by number of exposed items $n$ and epochs $e$; the audited training sample consists entirely of exposed items, so these values do not measure sensitivity to a small exposed fraction of a larger split. Bottom: probe B in S2 ($n=250$) with 500 vs.\ 1{,}500 held-out test items. Bold: 95\% CI above zero.}
\label{tab:sens}
\begin{tabular}{lcccccccccc}
\toprule
& \multicolumn{2}{c}{$n=25$} & \multicolumn{2}{c}{$n=50$} & \multicolumn{2}{c}{$n=100$} & \multicolumn{2}{c}{$n=250$} & \multicolumn{2}{c}{$n=1000$}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}
Probe A & $e=1$ & $e=3$ & $e=1$ & $e=3$ & $e=1$ & $e=3$ & $e=1$ & $e=3$ & $e=1$ & $e=3$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}

\vspace{4pt}
\begin{tabular}{lcccc}
\toprule
& \multicolumn{2}{c}{Held-out 500} & \multicolumn{2}{c}{Held-out 1{,}500}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Probe B (S2, $n=250$) & $e=1$ & $e=3$ & $e=1$ & $e=3$\\
\midrule
""" + "\n".join(lb) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# ---------------- Kev
KV = pd.read_csv('results/kev_label_standardized.csv'); KS = pd.read_csv('results/kev_by_source.csv')
lk = []
for _, r in KV.iterrows():
    nsig = int((KS[KS.model == r.model].p_holm < 0.05).sum())
    lab = {'kev-0.8b': 'Kev-0.8B', 'kev-4b': 'Kev-4B', 'kev-9b': 'Kev-9B$^\\ddagger$'}[r.model]
    lk.append(f"{lab} & {r.acc_m:.3f} & {r.acc_f:.3f} & {r.acc_h:.3f} & {s(r.std_mf)} {ci(r.lo, r.hi)} & {s(r.std_mh)} & {nsig}/10 \\\\")
NB = pd.read_csv('results/nimble.csv').iloc[0]
lk.append(f"Nimble-9B & {NB.acc_train:.3f} & {NB.acc_eval:.3f} & -- & {s(NB.est)} {ci(NB.lo, NB.hi)}$^\\S$ & -- & -- \\\\")
KD = pd.read_csv('results/kev_bootstrap_diagnostics.csv'); KDp = KD[KD.estimator.str.startswith('primary') & (KD.contrast == 'members-fresh')]
KDs = KD[KD.estimator.str.startswith('sensitivity') & (KD.contrast == 'members-fresh')]
kev_diag = (f"Rare strata absent from a replicate are dropped with renormalized weights (mean weight removed {100*KDp.mean_weight_dropped.mean():.1f}\\%, max {100*KDp.max_weight_dropped.max():.1f}\\%); "
            f"restricting the support to strata with $\\ge$10 items per group gives " + ', '.join(s(x) for x in KDs.est) + " pp.")
W('tab_kev', r"""\begin{table*}[t]
\centering\scriptsize
\caption{Released models with documented training data. Kev: 3{,}900 published training records (members; 3{,}000 underlying items), 3{,}900 records newly generated from the same source pools with training and calibration states excluded (fresh; same source$\times$question counts), and the 1{,}820 official development/calibration records (dev/cal). Gaps are label-standardized (source$\times$question$\times$label strata); 95\% CIs from a bootstrap over underlying items stratified by source ($B=10{,}000$). """ + kev_diag + r""" Last column: sources with Holm-adjusted $p<0.05$ over the 30 source$\times$model accuracy contrasts. $^\ddagger$bf16 (fp32 exceeds host RAM). $^\S$Nimble: item-weighted raw accuracy gap, 2{,}676 training vs.\ 324 evaluation examples from disjoint task families; task families resampled as clusters in both groups (34 training, 6 evaluation families; $B=50{,}000$); conditional on the observed training set (evaluation families only): """ + ci(pd.read_csv('results/nimble.csv').iloc[1].lo, pd.read_csv('results/nimble.csv').iloc[1].hi) + r""".}
\label{tab:kev}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccccc}
\toprule
& \multicolumn{3}{c}{Accuracy} & Members$-$ & Members$-$ & Sources \\
Model & members & fresh & dev/cal & fresh [95\% CI] & dev/cal & $p_{\text{Holm}}{<}.05$\\
\midrule
""" + "\n".join(lk) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")
P = KS.pivot(index='source', columns='model', values='est'); Pp = KS.pivot(index='source', columns='model', values='p_holm'); Pf = KS.pivot(index='source', columns='model', values='acc_fresh')
nm = KS.groupby('source').n_members.first()
lk2 = []
for src in P.index:
    cells = []
    for m in ['kev-0.8b', 'kev-4b', 'kev-9b']:
        v = s(P.loc[src, m]); cells.append((f"\\textbf{{{v}}}" if Pp.loc[src, m] < 0.05 else v) + f" ({Pf.loc[src, m]:.2f})")
    lk2.append(f"{src} & {nm[src]} & " + ' & '.join(cells) + ' \\\\')
W('tab_supp_kev', r"""\begin{table}[!tb]
\centering\scriptsize
\caption{Kev members$-$fresh gap by source (label-standardized, pp; fresh accuracy in parentheses); $n$: member records (fresh: same source$\times$question counts); bold: $p_{\text{Holm}}<0.05$ over 30 cells; bootstrap $B=5{,}000$ per cell.}
\label{tab:supp_kev}
\setlength{\tabcolsep}{1.5pt}
\begin{tabular}{lrccc}
\toprule
Source & $n$ & Kev-0.8B & Kev-4B & Kev-9B \\
\midrule
""" + "\n".join(lk2) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------- typed
T = pd.read_csv('results/typed_probeA.csv')
lab = {'julia': 'Julia-1', 'qwen7b': 'Qwen2.5-7B', 'laya_en': 'Laya (English)', 'laya_ml': 'Laya (multilingual)', 'laya_typed': 'Laya (typed-dec.)$^\\dagger$'}
lt = []
JTY = pd.read_csv('results/jev_typed_probeA.csv')   # Jev: second audited model (analysis/jev_typed.py, B = 50,000)
MTY = pd.read_csv('results/msd1_typed_probeA.csv')  # Microsoft-Decision-1: third audited model (analysis/msd1_typed.py)
def _row(name, r, dj=None, dv=None, dm=None):
    f = lambda d: '' if d is None else f"{s(d.est)} {ci(d.lo, d.hi)}"
    return f"{name} & {r.acc_train:.3f} & {r.acc_test:.3f} & {s(r.est)} {ci(r.lo, r.hi)} & {f(dj)} & {f(dv)} & {f(dm)} \\\\"
for _, r in T[T.kind == 'gap'].iterrows():
    if r.model == 'julia':
        lt.append(_row(lab['julia'], r))
        lt.append(_row('Jev 1.13', JTY[(JTY.kind == 'gap') & (JTY.model == 'jev')].iloc[0]))
        lt.append(_row('Decision-1', MTY[(MTY.kind == 'gap') & (MTY.model == 'msd1')].iloc[0]))
    else:
        lt.append(_row(lab[r.model], r, T[T.model == 'julia-' + r.model].iloc[0], JTY[JTY.model == 'jev-' + r.model].iloc[0],
                       MTY[MTY.model == 'msd1-' + r.model].iloc[0]))
TD = T[T.model == 'julia'].iloc[0]; TSn = pd.read_csv('results/typed_bootstrap_sensitivity.csv')
TSj = TSn[TSn.model == 'julia'].iloc[0]; TSq = TSn[TSn.model == 'julia-qwen7b'].iloc[0]
typed_diag = (f"States are resampled within workflow ($B={int(TD.B)//1000}{{,}}000$), keeping all questions about a state together; in {100*T.frac_replicates_empty.mean():.0f}\\% of replicates "
              f"some stratum is absent and is dropped with renormalized weights (mean weight removed {100*T.mean_weight_dropped.mean():.2f}\\%, max {100*T.max_weight_dropped.max():.1f}\\%). "
              f"Restricting the support to strata with $\\ge$10 states per split ({int(TSj.strata_used)} strata) gives {s(TSj.est)} {ci(TSj.lo, TSj.hi)} for Julia-1 and "
              f"{s(TSq.est)} {ci(TSq.lo, TSq.hi)} for Julia-1 minus Qwen.")
W('tab_typed', r"""\begin{table*}[t]
\centering\scriptsize
\caption{Probe A on typed-decisions (complete splits: 6{,}000 training and 2{,}000 test questions from 1{,}200 and 400 states). Columns 2--4: accuracies and label-standardized gaps; columns 5--7: the audited model's standardized gap minus the reference model's on the same questions. Strata: workflow$\times$question$\times$gold (68 common strata, 99.95\% of training questions retained). 95\% CIs: state-cluster bootstrap, same states for both models in contrasts. """ + typed_diag + r""" Jev and Decision-1 (Microsoft-Decision-1) rows: $B=50{,}000$. $^\dagger$Checkpoint documented as tuned on the training split.}
\label{tab:typed}
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{lcccccc}
\toprule
 & & & & \multicolumn{3}{c}{Audited model minus reference} \\
\cmidrule(lr){5-7}
Model & Acc.\ tr. & Acc.\ te. & Std.\ [95\% CI] & Julia-1 & Jev & Decision-1 \\
\midrule
""" + "\n".join(lt) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# ---------------- audit (main)
GA = pd.read_csv('results/audit_gaps.csv'); DD = pd.read_csv('results/audit_did_39.csv')
JGA = pd.read_csv('results/jev_audit_gaps.csv'); JDD = pd.read_csv('results/jev_audit_did_39.csv')
MGA = pd.read_csv('results/msd1_audit_gaps.csv'); MDD = pd.read_csv('results/msd1_audit_did_39.csv')
la = []
for ds in ['emotion', 'agnews', 'rotten', 'yahoo', 'banking77', 'amazon', 'imdb', 'yelp', 'massive', 'wikitoxic']:
    j = GA[(GA.dataset == ds) & (GA.model == 'julia')].iloc[0]
    jv = JGA[(JGA.dataset == ds) & (JGA.model == 'jev')].iloc[0]
    mv = MGA[(MGA.dataset == ds) & (MGA.model == 'msd1')].iloc[0]
    cells = [f"{s(j.est)} {ci(j.lo, j.hi)}", f"{s(jv.est)} {ci(jv.lo, jv.hi)}", f"{s(mv.est)} {ci(mv.lo, mv.hi)}"]
    for c in ['deberta_c', 'deberta_all33', 'qwen7b', 'laya_ml']:
        d = DD[(DD.dataset == ds) & (DD.control == c)]
        if d.empty: cells.append('--'); continue
        d = d.iloc[0]; v = s(d.est)   # CIs of all 39 contrasts: tab_supp_audit
        if d.p_holm < 0.05: v = f"\\textbf{{{s(d.est)}}}$^*$"
        if d.strata_used < GA[(GA.dataset == ds) & (GA.model == 'julia')].iloc[0].strata_used: v += '$^\\flat$'
        cells.append(v)
    la.append(DSN[ds] + ' & ' + ' & '.join(cells) + ' \\\\')
W('tab_audit', r"""\begin{table*}[t]
\centering\scriptsize
\caption{Probe A on ten classification datasets (BTZSC hypotheses as options), pp. Columns 2--4: label-standardized train$-$test gaps of Julia-1, Jev and Microsoft-Decision-1. Columns 5--8: Julia-1's gap minus the reference model's gap on the same items (paired, same strata and weights; CIs, and Jev's and Decision-1's own 39 contrasts, in Table~\ref{tab:supp_audit}). 95\% CIs from a bootstrap stratified by split$\times$label (50{,}000 resamples). $^*$Holm-adjusted $p<0.05$ over the 39 contrasts (shift test; Section~\ref{sec:method}). All contrasts use the complete 2{,}000/2{,}000 samples (Rotten Tomatoes: 2{,}000/1{,}066) and cover every label.}
\label{tab:audit}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llllrrrr}
\toprule
 & \multicolumn{3}{c}{Gap} & \multicolumn{4}{c}{Julia-1 gap minus reference gap} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-8}
Dataset & Julia-1 & Jev & Decision-1 & DeBERTa-c & all-33 & Qwen-7B & Laya-ML \\
\midrule
""" + "\n".join(la) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# ---------------- supp audit (all 39; p_raw, q_BH, n, strata and coverage stay in results/audit_did_39.csv)
ph = lambda x: f"{x:.2f}" if x >= 0.005 else '$<$0.01'
ls_ = {}
for ds in DSN:
    first = True; ls_[ds] = []
    for c in ['deberta_c', 'deberta_all33', 'qwen7b', 'laya_ml']:
        d = DD[(DD.dataset == ds) & (DD.control == c)]
        if d.empty: continue
        d = d.iloc[0]
        flat = '$^\\flat$' if d.strata_used < GA[(GA.dataset == ds) & (GA.model == 'julia')].iloc[0].strata_used else ''
        jd = JDD[(JDD.dataset == ds) & (JDD.control == c)].iloc[0]
        jv = (f"\\textbf{{{s(jd.est)}}}$^*$" if jd.p_holm < 0.05 else s(jd.est))
        md = MDD[(MDD.dataset == ds) & (MDD.control == c)].iloc[0]
        mv = (f"\\textbf{{{s(md.est)}}}$^*$" if md.p_holm < 0.05 else s(md.est))
        jt = (f"\\textbf{{{s(d.est)}}}$^*$" if d.p_holm < 0.05 else s(d.est))
        ls_[ds].append(f"{DSS[ds] if first else ''} & {CTS[c]}{flat} & {jt} {ci(d.lo, d.hi)} & {ph(d.p_holm)} & {jv} & {mv}")
        first = False
_h1 = [r for k in list(DSN)[:5] for r in ls_[k]]; _h2 = [r for k in list(DSN)[5:] for r in ls_[k]]
_h1 += [' & & & & & '] * (len(_h2) - len(_h1))
ls_ = [a + ' & ' + b + ' \\\\' for a, b in zip(_h1, _h2)]
W('tab_supp_audit', r"""\begin{table*}[!tb]
\centering\scriptsize
\caption{All 39 classification-audit contrasts per audited model: its standardized train$-$test gap minus the reference's on paired items (pp; shift test, $B=50{,}000$; Holm within each model's 39); $^*p_{\text{Holm}}<0.05$. Julia-1 with 95\% CI; Jev and Decision-1 (Microsoft-Decision-1) as estimates (CIs, raw $p$ and BH $q$ for all models in \texttt{results/*audit\_did\_39.csv}). DeB-c: DeBERTa-c; all-33: DeBERTa all-33; Qwen: Qwen-7B; Laya: Laya-ML.}
\label{tab:supp_audit}
\setlength{\tabcolsep}{2.2pt}
\begin{tabular}{llllll@{\hspace{8pt}}llllll}
\toprule
Dataset & Ref. & Julia-1 & $p_{\text{H}}$ & Jev & Dec.-1 & Dataset & Ref. & Julia-1 & $p_{\text{H}}$ & Jev & Dec.-1 \\
\midrule
""" + "\n".join(ls_) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# ---------------- diluted exposure (third revision, E1)
E1 = pd.read_csv('results/r8_e1_power.csv'); E1M = pd.read_csv('results/r8_e1_membership_by_seed.csv')
E1K = ['yahoo', 'agnews', 'yelp', 'emotion']
ld = []
for rho in [0.05, 0.1, 0.25, 0.5, 1.0]:
    cells = []
    for k in E1K:
        r = E1[(E1.dataset == k) & (E1.epochs == 3) & (E1.rho == rho)].iloc[0]
        v = s(r.did_rho0)
        cells.append((f"\\textbf{{{v}}}" if r.did_rho0_lo > 0 else v) + f"$^{{{int(r.seeds_detected)}}}$")
    ld.append(f"{rho:g} & " + ' & '.join(cells) + ' \\\\')
mem = E1M[E1M.epochs == 3].groupby('dataset').est.mean()
ld.append('\\midrule Per item & ' + ' & '.join(s(mem[k]) for k in E1K) + ' \\\\')
W('tab_dilution', r"""\begin{table}[!tb]
\centering\scriptsize
\caption{Probe A under diluted exposure: Julia-1 fine-tuned (3 epochs, 5 seeds) on 2{,}000 items, a fraction $\rho$ of the 2{,}000 audited training items plus filler items from the same training split. Entries: probe A minus that of the same-seed $\rho=0$ run (pp); bold: pooled 95\% CI above zero (item values averaged over seeds; $B=50{,}000$); superscript: seeds whose own CI ($B=5{,}000$) is above zero. Per item: exposed minus unexposed audited items (mean over seeds and $0<\rho<1$).}
\label{tab:dilution}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccc}
\toprule
$\rho$ & Yahoo & AG News & Yelp & Emotion \\
\midrule
""" + "\n".join(ld) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

C = pd.read_csv('results/probeC.csv'); CS = pd.read_csv('results/probeC_phi_complete_only.csv')
lc = []
for ds in ['emotion', 'agnews', 'banking77', 'yelp', 'yahoo']:
    cells = []
    for para in ['Qwen2.5-7B-Instruct', 'Phi-3.5-mini-instruct']:
        for m in ['julia', 'deberta_c', 'julia-deberta_c']:
            r = C[(C.paraphraser == para) & (C.dataset == ds) & (C.model == m)].iloc[0]
            cells.append((s(r.est), ci(r.lo, r.hi)))
    lc.append(DSN[ds] + ' & ' + ' & '.join(c[0] for c in cells) + ' \\\\')
    lc.append(' & ' + ' & '.join(c[1] for c in cells) + ' \\\\[1pt]')
W('tab_para', r"""\begin{table*}[t]
\centering\scriptsize
\caption{Probe C (pp): train$-$test difference in the paired original-minus-paraphrase accuracy drop, with one common label distribution for both splits (Eq.~\ref{eq:c}); 1{,}000 training and 1{,}000 test items per dataset; 95\% CIs from a bootstrap stratified by split$\times$label. ``Julia$-$DeBERTa'' is the paired between-model contrast on the same items. The ten between-model contrasts form the probe-C family; none has Holm-adjusted $p<0.05$ (smallest: Yahoo/Phi, $p=0.017$, $p_{\text{Holm}}=0.17$). Using the same items with a second paraphraser is a sensitivity analysis, not an independent replication.}
\label{tab:para}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{lcccccc}
\toprule
& \multicolumn{3}{c}{Qwen2.5-7B paraphrases} & \multicolumn{3}{c}{Phi-3.5-mini paraphrases}\\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Dataset & Julia-1 & DeBERTa-c & Julia$-$DeBERTa & Julia-1 & DeBERTa-c & Julia$-$DeBERTa\\
\midrule
""" + "\n".join(lc) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# ---------------- neardup
Q = pd.read_csv('results/neardup_quartiles.csv'); NS = pd.read_csv('results/neardup_slope.csv')
lq = [f"Q{int(r.q)+1} ({r.lo:.2f}--{r.hi:.2f}) & {r.julia:.3f} & {r.qwen:.3f} \\\\" for _, r in Q.iterrows()]
sl = {r.model: r for _, r in NS.iterrows()}
W('tab_neardup', r"""\begin{table}[!tb]
\centering\scriptsize
\caption{typed-decisions: test accuracy by quartile of the test state's maximum TF-IDF similarity (character 5-grams, full sorted-key JSON state) to the 1{,}200 training states; ranges are the empirical minimum and maximum in each quartile; 100 states $=$ 500 questions per quartile. Slope: OLS of question-level correctness (0/1) on maximum similarity, both demeaned within workflow (unweighted questions), pp per 0.1; 95\% CI from a state-cluster bootstrap stratified by workflow ($B=5{,}000$). Julia-1 """ + f"{s(sl['Julia-1'].slope_per_0p1)} pp {ci(sl['Julia-1'].lo, sl['Julia-1'].hi)}; Qwen {s(sl['Qwen2.5-7B'].slope_per_0p1)} {ci(sl['Qwen2.5-7B'].lo, sl['Qwen2.5-7B'].hi)}; Julia-1 minus Qwen {s(sl['Julia-1 minus Qwen'].slope_per_0p1)} {ci(sl['Julia-1 minus Qwen'].lo, sl['Julia-1 minus Qwen'].hi)}" + r"""; this interval includes zero.}
\label{tab:neardup}
\begin{tabular}{lcc}
\toprule
Similarity quartile (range) & Julia-1 & Qwen2.5-7B \\
\midrule
""" + "\n".join(lq) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------- length / truncation
LT = pd.read_csv('results/length_truncation_by_split.csv')
LI = pd.read_csv('results/len_items.csv')
QT = LI.groupby(['dataset', 'split']).apply(lambda d: int((d.chars.values[:1000] > 1500).sum())).rename('q1000').reset_index()
LT = LT.merge(QT, on=['dataset', 'split'])
ll = []
for ds in ['imdb', 'yelp', 'wikitoxic', 'yahoo']:
    a = LT[(LT.dataset == ds) & (LT.split == 'train')].iloc[0]; b = LT[(LT.dataset == ds) & (LT.split == 'test')].iloc[0]
    ll.append(f"{DSN[ds]} & {int(a.med_tok)}/{int(b.med_tok)} & {int(a.julia_trunc)}/{int(b.julia_trunc)} & {int(a.nli_trunc)}/{int(b.nli_trunc)} & {int(a.q1000)}/{int(b.q1000)} \\\\")
W('tab_len', r"""\begin{table}[!tb]
\centering\scriptsize
\caption{Input length and truncation by split (train/test, 2{,}000 items each): median Julia-1 (mmBERT) tokens; items truncated by Julia-1 ($>$700 tokens), the NLI references (text$+$longest hypothesis $>$320 tokens) and the Qwen prompt ($>$1{,}500 characters; first 1{,}000 items). Other datasets have no truncated item (Rotten Tomatoes not re-tokenized); no Julia-1 request was rejected.}
\label{tab:len}
\setlength{\tabcolsep}{2pt}
\begin{tabular}{lcccc}
\toprule
Dataset & Median tok. & Julia-1 & NLI & Qwen \\
\midrule
""" + "\n".join(ll) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
print('tables written')
