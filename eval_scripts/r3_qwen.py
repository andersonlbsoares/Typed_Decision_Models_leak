"""Strong negative control: Qwen2.5-7B-Instruct (never trained on typed-decisions) scored on train vs test.
Option letters scored by next-token logprob (vLLM, max_tokens=1)."""
import json, math, string, sys, time
import pandas as pd
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-7B-Instruct"
L = string.ascii_uppercase


def build(case):
    state = json.loads(case["state"]); gold = json.loads(case["gold"]); out = []
    for qid, q in json.loads(case["questions"]).items():
        kind = q["type"]; crit = q.get("criteria")
        if crit is None and kind == "noul":
            crit = {"false": "false", "true": "true"}
        if isinstance(crit, list):
            crit = {str(i): v for i, v in enumerate(crit)}
        keys = ["false", "true"] if kind == "noul" else list(crit)
        opts = "\n".join(f"{L[i]}. {crit[k]}" for i, k in enumerate(keys))
        extra = " The options are ordered from lowest to highest." if kind == "score" else ""
        msg = (f"State (JSON):\n{json.dumps(state, ensure_ascii=False)}\n\nQuestion: {q['instructions']}{extra}\n\n"
               f"Options:\n{opts}\n\nAnswer with the letter of the single best option only.")
        out.append(dict(id=case["id"] + ":" + qid, type=kind, keys=keys, gold=str(gold[qid]["label"]),
                        workflow=case["workflow"], msg=msg))
    return out


if __name__ == "__main__":
    llm = LLM(MODEL, max_model_len=4096, gpu_memory_utilization=0.85, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
    res = {}
    for split in ["train", "test"]:
        items = [x for _, c in pd.read_parquet(f"typed_{split}.parquet").iterrows() for x in build(c)]
        prompts = [tok.apply_chat_template([{"role": "user", "content": x["msg"]}], tokenize=False, add_generation_prompt=True)
                   for x in items]
        t = time.time(); outs = llm.generate(prompts, sp)
        recs = []
        for x, o in zip(items, outs):
            lp = o.outputs[0].logprobs[0]
            best = {}
            for tid, v in lp.items():
                s = (v.decoded_token or "").strip()
                if len(s) == 1 and s in L[:len(x["keys"])]:
                    best[s] = max(best.get(s, -1e9), v.logprob)
            if best:
                letter = max(best, key=best.get); pred = x["keys"][L.index(letter)]
                z = [best.get(L[i], -1e9) for i in range(len(x["keys"]))]; m = max(z)
                p = [math.exp(a - m) for a in z]; conf = max(p) / sum(p)
            else:
                pred, conf = None, None
            recs.append(dict(id=x["id"], type=x["type"], workflow=x["workflow"], gold=x["gold"], pred=pred, conf=conf))
        acc = sum(r["pred"] == r["gold"] for r in recs) / len(recs)
        print(split, len(recs), round(acc, 4), "unparsed", sum(r["pred"] is None for r in recs), round(time.time() - t, 1), "s", flush=True)
        res[split] = recs
    json.dump(res, open("r3_qwen_typed.json", "w"))
