# Data licensing

- **Code** (`analysis/`, `eval_scripts/`): MIT License (see `LICENSE`).
- **Our outputs** (`predictions/`, `results/`, including the recorded API responses
  `predictions/*_raw.jsonl.gz` and request indices `predictions/*_request_index.jsonl.gz`):
  Creative Commons Attribution 4.0 International (CC BY 4.0,
  https://creativecommons.org/licenses/by/4.0/).
  These files store item identifiers, gold labels and decisions; the API responses and
  request indices contain no input texts.
- **Benchmark texts.** The only files that contain benchmark texts in bulk are the probe-C
  paraphrase files `predictions/r4_para.json` (Qwen2.5-7B-Instruct paraphrases) and
  `predictions/r7_para_second.json` (Phi-3.5-mini-instruct paraphrases). They hold the original
  texts and paraphrases of the sampled AG News, DAIR Emotion, Banking77, Yelp Polarity and
  Yahoo Answers items. They are provided for research reproducibility only and remain subject
  to the licenses and terms of use of the source datasets (license fields on the Hugging Face
  dataset cards at the pinned revisions: AG News `unknown`, DAIR Emotion `other`,
  Banking77 (mteb) `mit`, Yelp Polarity none stated, Yahoo Answers Topics `unknown`).
  The CC BY 4.0 grant above does not cover these texts.
- All other benchmark inputs are rebuilt by the released scripts from the official dataset
  revisions listed in the paper (Appendix A.1).
