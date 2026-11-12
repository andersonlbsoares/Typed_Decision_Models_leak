# Typed-decision models: output-only exposure diagnostics

Research artifact for the paper *Output-Only Diagnostics of Benchmark Split Exposure in Typed-Decision Models* (Anderson Luis Bento Soares, Gabriel Bianchin de Oliveira, Ruben Interian and Zanoni Dias; Institute of Computing, Unicamp).
It contains the evaluation scripts, per-item predictions, fixed sample IDs, the analysis code and the machine-readable results behind every table and figure. Public benchmark data are not redistributed; scripts that need them say which file and revision to place in `data/`.

Quick start (all tables from the stored predictions, no GPU needed):

```
pip install -r requirements.txt
python analysis/audit.py 50000 && python analysis/typed.py 20000 && python analysis/probe_bc.py
python analysis/grid.py 10000 && python analysis/kev.py && python analysis/nimble.py && python analysis/sample_support.py
python analysis/r8_analysis.py 50000   # third revision: dilution (E1), reference overlap (E3), deterministic re-run (E4)
python analysis/jev.py 50000 && python analysis/jev_typed.py 50000   # Jev as a second audited model
python analysis/msd1.py 50000 && python analysis/msd1_typed.py 50000   # Microsoft-Decision-1 as a third audited model
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

## Revision notes (Microsoft-Decision-1)

- **Microsoft-Decision-1 as a third audited model** (closed; OpenRouter `microsoft/microsoft-decision-1`, snapshot
  `microsoft-decision-1-20261009`; post-trained from Qwen3.5-9B, training data not listed), with its own Holm families and
  never used as a reference. Same requests as Jev (69,382 distinct requests for 79,066 decisions; 500 repeated pilot
  requests for the determinism check, 2.2% of decisions changed; first response used). Collection code and the
  pre-registered plan: `eval_scripts/decision1/`; predictions `predictions/msd1.json.gz`; raw responses
  `predictions/msd1_raw.jsonl.gz`; request index (no texts) `predictions/msd1_request_index.jsonl.gz`.
- Results: `results/msd1_audit_gaps.csv`, `msd1_audit_did_39.csv`, `msd1_typed_probeA.csv`, `msd1_typed_bootstrap_sensitivity.csv`,
  `msd1_three_split.csv`, `msd1_probeC.csv`, `msd1_pgold_probeA.csv` (exploratory), `msd1_julia_agreement.csv` (descriptive).
- `analysis/msd1.py` and `analysis/msd1_typed.py` are the Jev scripts with the model's file names substituted.
- Paper-length trims: the per-split length/truncation table is `results/length_truncation_by_split.csv`.

## Table and figure map (TMLR version)

| Paper item | Command (from the artifact root) | Result files |
|---|---|---|
| Fig. 1 (validation) | `python analysis/grid.py 10000 && python analysis/kev.py && python analysis/fig_validation.py` | `controlled_grid.csv`, `kev_*.csv` |
| Table 1 (controlled exposure counts, probes A/B) | `python analysis/grid.py 10000` | `controlled_grid.csv` (pooled and per-seed estimates, `seeds_positive_ci`) |
| Table 1 (probe C rows) | `python analysis/revision_tmlr.py 10000` | `controlled_grid_probeC.csv` |
| Table 2 (sensitivity grid) | `python analysis/grid.py 10000` | `controlled_grid.csv` (`grid` = `small_n`, `heldout_1500`) |
| Table 3 (diluted exposure) | `python analysis/r8_analysis.py 50000` | `r8_e1_power.csv`, `r8_e1_membership_by_seed.csv`, `r8_e1_base_vs_filler.csv` |
| Table 4 (Kev, Nimble) | `python analysis/kev.py && python analysis/nimble.py` | `kev_label_standardized.csv`, `kev_bootstrap_diagnostics.csv`, `nimble.csv` |
| Table 5 (typed-decisions) | `python analysis/typed.py 20000 && python analysis/jev_typed.py 50000 && python analysis/msd1_typed.py 50000` | `typed_probeA.csv`, `jev_typed_probeA.csv`, `msd1_typed_probeA.csv` (+ `*_bootstrap_sensitivity.csv`) |
| Fig. 2, Tables 6 and 8 (classification audit) | `python analysis/audit.py 50000 && python analysis/jev.py 50000 && python analysis/msd1.py 50000 && python analysis/fig_audit.py` | `audit_gaps.csv`, `audit_did_39.csv`, `jev_audit_*.csv`, `msd1_audit_*.csv` |
| Laya-ML on Rotten Tomatoes (outside the families) | `python analysis/revision_tmlr.py` | `laya_ml_rotten_outside_family.csv` |
| Test vs validation (Emotion) | `python analysis/probe_bc.py`, `python analysis/revision_tmlr.py` | `three_split.csv`, `jev_three_split.csv`, `msd1_three_split.csv`, `three_split_common_weights.csv` |
| Table 7 (probe C) | `python analysis/probe_bc.py` (Jev, Decision-1: `jev.py`, `msd1.py`) | `probeC.csv`, `probeC_phi_complete_only.csv`, `jev_probeC.csv`, `msd1_probeC.csv` |
| Table 9 (Kev per source) | `python analysis/kev.py` | `kev_by_source.csv` |
| Table 10 (lengths) | stored | `length_truncation_by_split.csv`, `len_items.csv` |
| Table 11 (near-duplicates) | `python analysis/neardup.py` (needs typed-decisions parquet files in `data/`) | `neardup_quartiles.csv`, `neardup_slope.csv` |
| Table 12 (BTZSC task transfer) | stored | `btzsc_transfer.csv` |
| Emotion overlap, deterministic re-run | `python analysis/overlap_emotion.py`, `python analysis/r8_analysis.py 50000` | `emotion_overlap_*.csv`, `r8_e4_*.csv` |
| Dilution baseline vs audit run | `python analysis/revision_tmlr.py` | `dilution_baseline_vs_audit.csv` |
| Probe A on fixed items, S1 vs S2 (Section 4.1) | `python analysis/revision_tmlr_r2.py 10000` | `controlled_grid_fixed_Ete.csv` |
| API decision ties, tie-rule sensitivity, repeat checks (Appendix) | `python analysis/revision_tmlr_r2.py` | `api_decision_ties.csv`, `api_repeat_checks.csv` |
| Kev group disjointness (Section 4.3) | `python analysis/revision_tmlr_r2.py` | `kev_group_overlap.csv` |
| API score-tie branches and decision reconstruction from raw responses (Appendix) | `python analysis/revision_tmlr_r3.py r05` | `api_tie_resolution.csv`, `api_tie_examples.csv` |
| Qwen typed-decisions prompt lengths, pinned tokenizer (Appendix) | `python analysis/revision_tmlr_r3.py r06` | `typed_input_lengths_qwen.csv` |

LaTeX tables are produced by `python analysis/make_tables.py` (conference layout); the TMLR tables use the same result files.
Table 10's Qwen column counts all 2,000 items per split (`qwen_trunc` in `length_truncation_by_split.csv`).

## Revision notes (TMLR pre-submission reviews)

- **Probe C under controlled exposure.** `analysis/revision_tmlr.py` computes probe C on the controlled-exposure checkpoints
  (Qwen2.5-7B paraphrases of the same items, groups of probe A, minus the pre-FT checkpoint):
  `results/controlled_grid_probeC.csv`.
- **Laya-ML on Rotten Tomatoes.** The 39-contrast families were fixed when the only Laya Rotten Tomatoes predictions came from
  a positive-only 1,000-record training subset (`predictions/r6_laya.json.gz`). The complete-sample run
  (`predictions/r8_e3_laya_full.json.gz`) is reported outside the families: `results/laya_ml_rotten_outside_family.csv`.
- **Three-split weights.** `results/three_split_common_weights.csv` shows that the Emotion train/test/validation contrasts are
  additive with one pooled label distribution; the paper's pairwise contrasts use the weights of each pair (Eq. 1).
- **Dilution baseline.** `results/dilution_baseline_vs_audit.csv`: same items, labels and weights as the audit; 3 training and
  3 test decisions differ on Emotion (+9.2 vs +8.9 pp).
- **API identifiers.** Every Jev and Microsoft-Decision-1 response in `predictions/*_raw.jsonl.gz` carries the returned
  `body.model` (`typesafe/jev-1.13-20260917`, `microsoft/microsoft-decision-1-20261009`) and `body.provider`, and the
  collection timestamp `t` (Unix seconds).
- **Kev fresh records.** `eval_scripts/r7_kev_data.py`: members are 300 clean records per source sampled (seed 0) from the
  released v7 training partition; fresh records come from `kev.suite --exclude-states-from` the frozen v7 suite (exact
  normalized-text hashes of all 12,576 training and 968 calibration records), with records sharing an id removed.

## Revision notes (second TMLR pre-submission review)

- **Fixed evaluation items across scenarios.** Probe A uses test items 501-1,000 in S1 and 1-500 in S2. `analysis/revision_tmlr_r2.py`
  recomputes it on test items 1-500 in both scenarios, so S1 and S2 differ only in the fine-tuning data, and estimates the
  S1-minus-S2 change on the same items (paired: per-item S1-minus-S2 values, same stratified bootstrap): positive with a 95% interval
  wholly above zero in 12/12 DeBERTa and 8/8 Julia-1 configurations (`results/controlled_grid_fixed_Ete.csv`, rows `A_S1-A_S2(E_tr,E_te)`).
- **API decision rules.** Choice: the returned key. Yes/no: `true` if p > 0.5, `false` if p < 0.5, scored incorrect if p = 0.5.
  Score: the most probable level, with ties resolved towards the returned expected score and then the lower level (`analysis/jev.py`).
  `results/api_decision_ties.csv` counts p = 0.5 and tied score answers and gives typed-decisions gaps under a gold-independent
  alternative rule; `results/api_repeat_checks.csv` lists the 500 repeated requests per model (seeded sample) and their transitions.
- **Kev groups.** `results/kev_group_overlap.csv`: no underlying source item is shared by any pair of members, fresh and dev/cal.
- **Banking77 label universe.** The audit uses the 72 of 77 intents with a BTZSC hypothesis; excluded: `card_linking`,
  `exchange_via_app`, `get_physical_card`, `supported_cards_and_currencies`, `top_up_by_bank_transfer_charge`.

## Revision notes (third TMLR pre-submission review)

- **Score ties (Jev, Microsoft-Decision-1).** With p_j the returned probability of level j, s the returned expected score and
  M the set of levels whose returned probability equals the maximum exactly, the decision is the level in M closest to s, then the
  lowest (`decide` in `analysis/jev.py` and `analysis/msd1.py`); no step uses the gold label. `analysis/revision_tmlr_r3.py r05`
  re-applies the rule to the stored first responses, reproduces all 47,066 stored typed-decisions and probe-A decisions of each
  model, and counts the branch used: Jev 7 ties (4 settled by s, 3 with two maximal levels equidistant from s, lowest taken),
  Microsoft-Decision-1 53 ties (all settled by s); every tied answer carried s.
- **Qwen inputs on typed-decisions.** The 1,500-character cut (`eval_scripts/r6_qwen.py`, `e3_qwen_full.py`) applies to
  classification texts before prompt assembly. `eval_scripts/r3_qwen.py` passes the full JSON state; `revision_tmlr_r3.py r06`
  rebuilds its prompts and counts them with the pinned Qwen tokenizer: at most 674 (train) and 639 (test) tokens, none at or
  above vLLM's max_model_len of 4,096. Julia-1 runs with strict encoding (`FastEngine(strict_encoding=True)`), which raises
  instead of truncating; no typed-decisions request failed.

## License

Code (`analysis/`, `eval_scripts/`) is released under the MIT License (see [`LICENSE`](LICENSE)).
Predictions, result tables and recorded API responses are released under CC BY 4.0.
Benchmark texts are not covered by these licenses: the probe-C paraphrase files
(`predictions/r4_para.json`, `predictions/r7_para_second.json`) contain original benchmark texts
and remain subject to the terms of the source datasets. See [`DATA_LICENSE.md`](DATA_LICENSE.md).
