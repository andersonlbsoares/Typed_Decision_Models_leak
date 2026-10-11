# Jev (typesafe/jev-1.13) collection

Pre-registered plan: `prereg_jev.md` (sections 1-6 frozen before any audit item was sent; section 7 added after the
500-request pilot, before the full collection).

1. Build the audit samples with `eval_scripts/r8/build_samples.py` (writes `eval_scripts/r8/data/`), plus
   `data/emotion_validation.json.gz` (DAIR Emotion rev. cab853a1, config `split`, validation split; ids
   `emotion:validation:<row>`, labels mapped as in `r2_data.py`).
2. `python jev_audit.py build` writes the requests (contain benchmark texts, so not redistributed here).
3. `OPENROUTER_API_KEY=... python jev_audit.py pilot` and `... run` (spending ceiling in `jev_client.py`, ledger file).
4. `python analysis/jev.py collect predictions/jev_request_index.jsonl.gz predictions/jev_raw.jsonl.gz`, then
   `python analysis/jev.py 50000` and `python analysis/jev_typed.py 50000`.

`predictions/jev_request_index.jsonl.gz` maps every analysis row (task, item id, gold, option subset) to the SHA-256 of
its request; `predictions/jev_raw.jsonl.gz` holds every raw response (first send `rep = 1`; pilot repeats `rep = 2`).
All 79,066 rows were answered by snapshot `typesafe/jev-1.13-20260917`; cost US$ 1.60.
