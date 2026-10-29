"""Third TMLR pre-submission review (29/10/2026): implementation checks behind two descriptive fixes.

R05 (API score ties). Re-runs the decision rule of analysis/jev.py and analysis/msd1.py on the stored first
responses (predictions/<model>_raw.jsonl.gz + predictions/<model>_request_index.jsonl.gz), checks that it
reproduces every stored decision in predictions/jev_1.13.json.gz and predictions/msd1.json.gz, and counts
which branch of the rule decided each score question:
  unique_max       one level has the largest returned probability
  tie_by_score     several levels share the maximum; the one whose index is closest to the returned
                   expected score is unique
  tie_lowest       several maximal levels are equally close to the expected score; the lowest is taken
  tie_no_score     several maximal levels and no expected score returned (would take the first key)
Ties use exact equality of the probabilities as returned (no tolerance); no step uses the gold label.
Outputs: results/api_tie_resolution.csv, results/api_tie_examples.csv (request ids, returned
probabilities, expected score and decision; no benchmark text).

R06 (typed-decisions inputs to Qwen). Rebuilds the exact Qwen2.5-7B-Instruct prompts of
eval_scripts/r3_qwen.py (full JSON state, question, options, chat template) from the pinned
typed-decisions files and counts them with the pinned Qwen tokenizer against vLLM's max_model_len=4096.
The 1,500-character cut in eval_scripts/r6_qwen.py / e3_qwen_full.py applies to classification texts only.
Output: results/typed_input_lengths_qwen.csv

Run from the artifact root: python analysis/revision_tmlr_r3.py [r05|r06|all]
(r06 downloads the pinned dataset files and tokenizer from the Hugging Face Hub; needs transformers.)"""
import gzip, json, sys, types, importlib.util
from collections import Counter
import pandas as pd

sys.path.insert(0, 'analysis')
TYPED_REV = 'c76749ec58bd8c3d2ea706b31c333a9059c38f90'
QWEN_REV = 'a09a35458c702b33eeacc393d103063234e8bc28'
QWEN_MAX_MODEL_LEN = 4096


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m


def first_responses(raw_path):
    first = {}
    for l in gzip.open(raw_path, 'rt'):
        try:
            r = json.loads(l)
        except ValueError:
            continue
        if r.get('rep', 1) == 1 and r.get('status') == 200 and r['hash'] not in first:
            first[r['hash']] = r['body']
    return first


def branch(body):
    a = list(body['answers'].values())[0]
    if a.get('type') == 'noul':
        return 'noul_p_eq_0.5' if a.get('noul') == 0.5 else 'noul'
    if a.get('type') != 'score':
        return a.get('type')
    P = a.get('probabilities') or {}; m = max(P.values()); top = [k for k, v in P.items() if v == m]
    if len(top) == 1:
        return 'unique_max'
    s = a.get('score')
    if s is None:
        return 'tie_no_score'
    d = sorted(abs(float(k) - s) for k in top)
    return 'tie_lowest' if d[0] == d[1] else 'tie_by_score'


def stored_lookup(pred):
    out = {}
    for s in ['train', 'test']:
        for r in pred['typed'][s]:
            out[('typed', s, r['id'])] = r['pred']
    for ds, v in pred['A'].items():
        for k, recs in v.items():
            for r in recs:
                out[('A', ds, k, r[0])] = r[2]
    return out


def r05():
    rows, ex = [], []
    for model, mod, pred_path in [('jev', 'analysis/jev.py', 'predictions/jev_1.13.json.gz'),
                                  ('msd1', 'analysis/msd1.py', 'predictions/msd1.json.gz')]:
        M = load_module(mod, model + '_mod')
        first = first_responses(f'predictions/{model}_raw.jsonl.gz')
        stored = stored_lookup(json.load(gzip.open(pred_path, 'rt')))
        cnt, checked, mismatch = Counter(), 0, 0
        for l in gzip.open(f'predictions/{model}_request_index.jsonl.gz', 'rt'):
            q = json.loads(l); body = first.get(q['hash']); task = q['task'].split(':')
            key, P = M.decide(body)
            if task[1] == 'typed':
                sk = ('typed', task[2], q['id']); dec = key
                fam = 'typed-decisions'
            elif task[0] == 'A':
                opts = q['opts']
                dec = None if key is None or key not in [str(j) for j in range(len(opts))] else opts[int(key)]
                sk = ('A', task[1], 'A_' + task[2], q['id']); fam = 'classification probe A'
            else:
                sk = None; fam = 'classification probes B/C'
            if sk is not None and sk in stored:
                checked += 1; mismatch += int(stored[sk] != dec)
            b = branch(body) if body else 'missing'
            cnt[(fam, b)] += 1
            if b.startswith('tie') and len(ex) < 400:
                a = list(body['answers'].values())[0]
                ex.append(dict(model=model, task=q['task'], id=q['id'], branch=b, expected_score=a.get('score'),
                               probabilities=json.dumps(a.get('probabilities'), sort_keys=True), decision=key))
        for (fam, b), n in sorted(cnt.items()):
            rows.append(dict(model=model, family=fam, branch=b, n=n))
        rows.append(dict(model=model, family='all stored typed + probe-A decisions', branch='reproduced',
                         n=checked - mismatch))
        rows.append(dict(model=model, family='all stored typed + probe-A decisions', branch='mismatch', n=mismatch))
        print(model, 'checked', checked, 'mismatch', mismatch)
    D = pd.DataFrame(rows); D.to_csv('results/api_tie_resolution.csv', index=False); print(D.to_string())
    pd.DataFrame(ex).to_csv('results/api_tie_examples.csv', index=False)


def r06():
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    sys.modules.setdefault('vllm', types.SimpleNamespace(LLM=None, SamplingParams=None))
    Q = load_module('eval_scripts/r3_qwen.py', 'r3_qwen')
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct', revision=QWEN_REV)
    rows = []
    for split in ['train', 'test']:
        p = hf_hub_download('LocalLLaMA/typed-decisions', f'all/{split}-00000-of-00001.parquet',
                            repo_type='dataset', revision=TYPED_REV)
        cases = pd.read_parquet(p)
        items = [x for _, c in cases.iterrows() for x in Q.build(c)]
        prompts = [tok.apply_chat_template([{'role': 'user', 'content': x['msg']}], tokenize=False,
                                           add_generation_prompt=True) for x in items]
        n = [len(tok(s, add_special_tokens=False).input_ids) for s in prompts]
        st = [len(json.dumps(json.loads(c['state']), ensure_ascii=False)) for _, c in cases.iterrows()]
        rows.append(dict(split=split, cases=len(cases), prompts=len(items), max_state_chars=max(st),
                         max_user_message_chars=max(len(x['msg']) for x in items),
                         median_prompt_tokens=float(pd.Series(n).median()), max_prompt_tokens=max(n),
                         max_model_len=QWEN_MAX_MODEL_LEN, prompts_over_limit=sum(t >= QWEN_MAX_MODEL_LEN for t in n),
                         tokenizer_revision=QWEN_REV))
    D = pd.DataFrame(rows); D.to_csv('results/typed_input_lengths_qwen.csv', index=False); print(D.to_string())


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('r05', 'all'):
        r05()
    if what in ('r06', 'all'):
        r06()
