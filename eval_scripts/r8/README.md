# Third-revision runs (r8)

GPU runs added in the third revision, executed on one NVIDIA H100 80 GB (Slurm, offline compute nodes).
Outputs are in `predictions/r8_*.json.gz`; analyses in `analysis/audit.py` (complete-sample references)
and `analysis/r8_analysis.py` (E1, E3 overlap, E4).

| Script | Run | Output |
|---|---|---|
| `e3_qwen_full.py` | Qwen2.5-7B-Instruct on the complete 2,000/2,000 audit samples (Rotten Tomatoes 2,000/1,066); transformers bf16, one forward pass, next-token log-probabilities of the option letters; `pred_top20` keeps the earlier top-20 admissibility rule | `r8_e3_qwen_full.json.gz` |
| `e3_laya_full.py` | Laya multilingual and English (rev. 55cf4c4e) on the same samples | `r8_e3_laya_full.json.gz` |
| `e4_julia_det.py` | Julia-1 on typed-decisions (full splits), DAIR Emotion and AG News: GPU bs64 / bs1 / fp32 bs1 and CPU fp32 bs1 | `r8_e4_{gpu,cpu}.json.gz` |
| `e1_julia_dilution.py` | Julia-1 fine-tuned on 2,000 items: a fraction rho in {0, .05, .1, .25, .5, 1} of the 2,000 audited training items plus class-balanced filler items from the same training split outside the audit sample; 5 seeds; exposed sets nested across rho; evaluation after epochs 1 and 3 on all audited training and test items | `r8_e1_<dataset>.json.gz` |

Layout expected by the scripts: `<R8>/r8/` (these files), `<R8>/code/` (copy `eval_scripts/run_julia.py`,
`r2_data.py`, `r4_data.py`), `<R8>/ref/` (copy `predictions/r4_eval.json.gz`, `r6_rotten_eval.json.gz`,
`r2_julia_nli.json.gz`, `r4_para.json`, `r6_qwen.json.gz`, `r6_laya.json.gz`), `<R8>/jobs/`.
`stage_download.py` resolves the pinned short revisions to full commits and writes each split to parquet in
source order; `build_samples.py` rebuilds the audit samples from the stored item ids and aborts if any recomputed
gold label, paraphrase-source text or typed-decisions question id differs. Partition names in `jobs/` are placeholders.
