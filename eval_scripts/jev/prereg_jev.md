# Pre-registered analysis plan: Jev (typesafe/jev-1.13) as a second audited model

*Anonymized copy: two sentences naming hosts or quoting an author were reworded; the analysis content is unchanged.*

Written 2026-10-09, before any audit item was sent to Jev. Before this point, Jev had answered only
four requests, all on invented sentences that are not audit items (`jev_test_call_1.json` and
`jev_test_calls_2.json`). The plan is frozen once this file is saved. Any later deviation is
reported as a deviation.

## 1. Model and access
- Model: OpenRouter `typesafe/jev-1.13`. The resolved snapshot (e.g. `typesafe/jev-1.13-20260917`) is
  logged for every response; if more than one snapshot appears, results are reported per snapshot.
- Endpoint: `POST https://openrouter.ai/api/v1/systemone` (TypeSafe format). Each request carries one
  question, so every model in the paper answers the same unit (one row = one decision).
- Logged for every request: the raw request, the raw response, the generation id, latency and cost.
- Budget: hard ceiling of US$ 3.00 on spending within this collection, measured against the key's
  usage at session start. It is enforced client-side (the cost is reserved before sending) and
  checked against OpenRouter's key usage every 200 requests. If the ceiling stops the collection,
  unfinished tasks are reported as "not assessed".
- Transport failures (HTTP 429/5xx, timeouts) are retried up to 3 times with backoff. A request that
  still fails, or a response without a valid choice, is recorded as `pred = None` and counted as
  incorrect, as `pred = None` is for Julia-1. The count is reported per task.

## 2. Inputs, identical to the Julia-1 protocol
- **Classification (10 datasets).** Use the stored audit samples (2,000 training and 2,000 test items per
  dataset; Rotten Tomatoes 2,000/1,066), rebuilt from the stored item ids by `build_samples.py`,
  which checks every gold label.
  - Request: `state = {"text": <text truncated to 700 Julia-1/mmBERT tokens>}`; instructions
    "Which single label best describes the input text?"; `type = "choice"`.
  - Criteria: the BTZSC hypotheses of the options Julia-1 saw (`subset_opts`, i.e. all options when
    there are ≤ 20, otherwise the gold label plus 19 seeded distractors in a seeded order).
  - Criteria format: sent as a JSON object with keys `"0"…"k-1"` in that order. The returned key is
    mapped back to the class index.
- **typed-decisions.** Use the complete train and test splits (1,200 + 400 states, 6,000 + 2,000
  questions). The state and the question object (`type`, `instructions`, `criteria`) are sent
  verbatim, one question per request. The prediction is the returned choice; for `noul` questions it
  is the returned boolean mapped to `"true"`/`"false"`. Gold is defined as in `typed_rows`.
- **Emotion validation (probe B).** The complete DAIR Emotion validation split (2,000 items), built
  with the same label mapping.
- **Scope.** An earlier plan collected probe A only; on 2026-10-09 the scope was extended to probes B and C, with a
  hard ceiling of US$ 3.
- **Probe C.** For the same 5 datasets × 1,000 training and 1,000 test items as Julia-1, collect the
  originals (`text[:1200]`) plus the Qwen2.5-7B and Phi-3.5-mini paraphrases already used in the
  paper. The Phi-3.5 paraphrase texts were stored only on the GPU hosts. `jev_audit.py` builds
  the Qwen half now; the Phi half (`C:*:para_phi`) is added with the same builder once the texts are
  retrieved, and before any probe-C result is analysed.

## 3. Pilot (before the full collection)
- The pilot is the first 500 requests: 300 typed-decisions questions (150 train and 150 test, seeded
  sample) and 200 Emotion validation items. Each is sent twice.
- Reported: the share of decisions that change between the two sends, the mean |Δp| on the chosen
  option, and any format failures.
- Pilot items stay in the full collection, which uses the first send.
- The pilot changes the plan only if more than 5% of responses fail format validation; in that case
  the request format is fixed and the pilot repeated.

## 4. Primary analysis (same code paths as for Julia-1; Jev substituted for Julia-1)
- **Gaps.** Probe A per task: label-standardized train−test accuracy gap, with the same strata, the
  same stratified bootstrap (B = 50,000) and the same seeds.
  - Classification strata: split × gold label.
  - typed-decisions: workflow × question × gold strata, with a state-cluster bootstrap stratified by
    workflow.
- **Classification contrasts.** Paired difference-in-differences Jev − reference for each of the 10
  datasets against DeBERTa-c, DeBERTa all-33, Qwen2.5-7B (`pred_top20`) and Laya-ML (not on Rotten
  Tomatoes). That gives the same 39-contrast set as for Julia-1, with **Holm adjustment within Jev's
  own family of 39**. The Julia-1 family stays unchanged.
- **typed-decisions contrasts.** Jev − reference contrasts against the references used for Julia-1
  in Table `tab:typed`.
- **Jev − Julia-1.** Descriptive only; no tests.

## 5. Secondary and exploratory analyses
- **Probe B (Emotion):** test − validation and train − validation gaps for Jev, and their contrast
  with DeBERTa-c, as reported for Julia-1.
- **Probe C:** train − test asymmetry in paraphrase sensitivity for each paraphraser, and paired
  Jev − DeBERTa-c contrasts, as in Table `tab:para`.
- **Exploratory:** probe A on the probability assigned to the gold option (label-standardized
  train − test difference in mean p(gold)). This is not part of any Holm family.

## 6. Interpretation rules, fixed in advance
- A Holm-significant positive contrast is read as a training-split-specific advantage. As for
  Julia-1, it is not read as proof of training on the split.
- A non-significant result is read with the dilution table (`tab:dilution`): the diagnostic detects
  an expected contrast of about 1 pp or more, so a null means "no advantage at the detectable size",
  not "no exposure".
- Provenance: before the results are interpreted, we document whether `LocalLLaMA/typed-decisions`
  is linked to TypeSafe (Jev's developer). If it is, a typed-decisions gap is reported as expected by
  design rather than as evidence of undisclosed exposure.

## 7. Amendment after the pilot, before the full collection (2026-10-09)
Pilot: 500 requests sent twice. All 1,000 responses returned HTTP 200, with no format failures and
snapshot `typesafe/jev-1.13-20260917` throughout. Cost: US$ 0.0227. Accuracy on the pilot items was
not computed.
- **Determinism.** 7 of 500 decisions (1.4%) changed between the two sends: classification 1/200,
  choice 1/97, noul 1/78, score 4/125. Mean |Δp| was 0.009–0.022. The analysis uses the first send,
  as planned.
- **Decision rules, fixed from the response format observed in the pilot:**
  - `choice` (classification and typed-decisions): the returned `choice` key.
  - `noul`: the response is a probability (`"noul": p`), not a boolean. The prediction is `"true"`
    if p > 0.5 and `"false"` if p < 0.5. If p = 0.5 exactly, the answer is scored incorrect
    (1 of 500 in the pilot).
  - `score`: the response has `probabilities` over the keys `"0"…"k-1"` (rounded to 0.01) plus an
    expected `score`. The prediction is the argmax key, matching the argmax rule used for Julia-1.
    Ties go to the key closest to the returned `score`, then to the lower key.
- **Exploratory p(gold):** the probability of the gold key; for `noul`, p if gold = true, else 1 − p.
- **Request sharing:** byte-identical requests are sent once and their response is shared. This
  covers 79,066 analysis rows and 69,382 unique requests: probe-C originals that equal the probe-A
  text, plus duplicate texts within a dataset.
