# Typed-decision models: output-only exposure diagnostics

Anonymized research artifact for the submission *Output-Only Diagnostics of Training-Split Exposure in Typed-Decision Models*.
It contains the evaluation scripts, per-item predictions, fixed sample IDs, the analysis code and the machine-readable results behind every table and figure. Public benchmark data are not redistributed; scripts that need them say which file and revision to place in `data/`.

Quick start (all tables from the stored predictions, no GPU needed):

```
pip install -r requirements.txt
python analysis/audit.py 50000 && python analysis/typed.py 20000 && python analysis/probe_bc.py
python analysis/grid.py 10000 && python analysis/kev.py && python analysis/nimble.py && python analysis/sample_support.py
python analysis/r8_analysis.py 50000   # third revision: dilution (E1), reference overlap (E3), deterministic re-run (E4)
python analysis/jev.py 50000 && python analysis/jev_typed.py 50000   # Jev as a second audited model
python analysis/make_tables.py && python analysis/fig_audit.py   # paper/tables/*.tex, paper/figs/fig_audit.*
```

Bootstrap sizes for every analysis are listed in `results/bootstrap_settings.csv`.

Run from this directory with Python 3.11 (numpy, pandas, scipy, statsmodels, scikit-learn, pyarrow, matplotlib).

- `predictions/` — per-item predictions (`[id, gold, pred, confidence]`) for every model, split and paraphrase collection; Kev `rows.json` per group; controlled fine-tuning outputs (per seed, per epoch, per item correctness, item order = sample order).
- `analysis/core.py` — estimator (Eq. 1, common support, pooled weights), split x stratum bootstrap (exact multinomial form), shift-test p-values, Holm and BH.
- `analysis/audit.py` (39-contrast probe-A family), `typed.py` (typed-decisions, state-cluster bootstrap), `probe_bc.py` (test/validation contrasts; probe C family), `grid.py` (controlled exposure, pooled and per-seed), `neardup.py` (needs the typed-decisions parquet files at revision c76749ec in `data/`), `kev.py` (Kev members/fresh/dev-cal, with bootstrap diagnostics and restricted-support sensitivity), `nimble.py` (Nimble-9B train-eval gap, family-cluster bootstrap in both groups and conditional variant), `overlap_emotion.py` (DAIR Emotion split-overlap sensitivity; needs `dair-ai/emotion` rev. cab853a1, config `split`, as `data/emotion_{train,test,validation}.parquet`), `sample_support.py` (sample order and per-label counts of every subset), `massive_labels.py` (MASSIVE: 58 audit labels vs 59 BTZSC hypotheses; needs the files named in its docstring in `data/`), `make_tables.py`, `fig_validation.py`. Tables are no longer rescaled (`resize_tables.py` is a no-op kept for old pipelines).
- `results/` — machine-readable outputs of every table, including all 39 probe-A contrasts with n, strata, excluded strata, coverage, SE, raw/Holm/BH p; per-seed grid estimates; token-length and truncation statistics; paraphrase completeness.
- `eval_scripts/` — scripts used to produce the predictions on the GPU host (data sampling, Julia-1/NLI/Qwen/Laya scoring, fine-tuning, paraphrase generation, Kev data).

## Revision notes (second review round)

- **Sample order.** `eval_scripts/run_julia.balanced_indices` selects items by class-balanced round-robin and returns the selected indices **sorted by source-file position**; the stored samples follow that order. Reference subsets (Qwen-7B, Laya-ML) are the first 1,000 stored records. See `results/sample_order_check.csv`, `results/label_support_subsets.csv`, `results/label_counts_by_subset.csv`.
- **Cluster bootstraps.** `results/typed_probeA.csv` and `results/kev_bootstrap_diagnostics.csv` report B, the fraction of replicates with an absent stratum, and the mean/max fixed-weight mass removed; `results/typed_bootstrap_sensitivity.csv` and the `sensitivity` rows of the Kev file restrict the support to strata with >= 10 clusters per group.
- **Qwen scoring.** A request with no admissible letter is stored as `pred = null` and counted as incorrect; there are 0 such requests in `r6_qwen.json.gz` (40,000) and `r3_qwen_typed.json` (8,000).
- **Human paraphrase check.** `results/paraphrase_human_check.csv`: 60 pairs, all from the Qwen2.5-7B-Instruct collection (`predictions/r4_para.json`); 7 pairs (all Yelp) were displayed truncated at 400 characters.
- **Emotion overlap.** `results/emotion_overlap_counts.csv`, `results/emotion_overlap_items.csv`, `results/emotion_split_overlap.csv`.
- **Nimble.** `predictions/nimble/` (per-example correctness, family and kind), `results/nimble.csv`.
- **MASSIVE label universe.** The English split has 60 intents; BTZSC has 59 hypotheses (none for `general_quirky`). `cooking_query` has a hypothesis but no BTZSC test example, so the name-based mapping in `eval_scripts/r4_data.py` leaves its 4 training items unmapped; it occurs only as a distractor option. The audit therefore has 58 gold labels and 59 candidate hypotheses (`results/massive_label_reconciliation.csv`).
- `eval_scripts/` reference the GPU host's working directory only through relative paths; dataset and checkpoint revisions are pinned in the scripts and in the paper's Appendix.

## Revision notes (third revision)

- **Complete-sample references.** Qwen-7B and Laya-ML were re-run on the complete 2,000/2,000 audit samples
  (`predictions/r8_e3_*`; scripts in `eval_scripts/r8/`). `analysis/audit.py` now uses them, so all 39 probe-A
  contrasts cover every label; the restricted-support table was removed. On the items shared with the earlier
  1,000-record runs, decisions agree on 98.5-100% (Qwen; transformers instead of vLLM) and >= 98.1% (Laya-ML):
  `results/r8_e3_overlap_agreement.csv`. `sample_support.py` and the `label_support_*` files describe the earlier subsets.
- **Deterministic re-run (E4).** `results/r8_e4_agreement.csv`, `results/r8_e4_probeA_by_mode.csv`.
- **Diluted exposure (E1).** `results/r8_e1_power.csv`, `r8_e1_membership_by_seed.csv`, `r8_e1_base_vs_filler.csv`; table `tab_dilution`.
- Samples were rebuilt on the GPU cluster from the stored item ids (`eval_scripts/r8/build_samples.py`), with every gold
  label and every paraphrase-source text checked against the stored files.

## Revision notes (Jev)

- **Jev 1.13 as a second audited model** (closed; OpenRouter `typesafe/jev-1.13`, snapshot `jev-1.13-20260917`), with its own
  Holm families and never used as a reference. Same states, questions, option subsets and 700-token truncation as Julia-1,
  one question per request. Collection code and the pre-registered plan: `eval_scripts/jev/`; predictions
  `predictions/jev_1.13.json.gz`; raw responses `predictions/jev_raw.jsonl.gz`; request index (no texts)
  `predictions/jev_request_index.jsonl.gz`.
- Results: `results/jev_audit_gaps.csv`, `jev_audit_did_39.csv`, `jev_typed_probeA.csv`, `jev_typed_bootstrap_sensitivity.csv`,
  `jev_three_split.csv`, `jev_probeC.csv`, `jev_pgold_probeA.csv` (exploratory), `jev_julia_agreement.csv` (descriptive).
- BTZSC task-transfer accuracies (former appendix table) moved to `results/btzsc_transfer.csv`.
