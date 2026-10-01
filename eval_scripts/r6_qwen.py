"""Qwen2.5-7B-Instruct (never trained on these benchmarks' splits as a classifier head):
 (i) strong negative control on train vs test for 10 datasets (1,000 per split);
 (ii) the same classifier on the paraphrased texts (5 datasets);
 (iii) paraphrase-validity judge: does the paraphrase preserve meaning and label? P(Yes).
Letters scored by next-token logprob (max_tokens=1)."""
import gzip, hashlib, json, math, random, string, time
from vllm import LLM, SamplingParams
S = json.load(gzip.open("r4_samples.json.gz", "rt"))["datasets"]
S["rotten"] = json.load(gzip.open("r6_rotten_samples.json.gz", "rt"))
P = json.load(open("r4_para.json"))
PARA = ["agnews", "emotion", "banking77", "yelp", "yahoo"]
L = string.ascii_uppercase


def subset_opts(item_id, target, k_all, k=20):
    if k_all <= 20:
        return list(range(k_all))
    rng = random.Random(int(hashlib.md5(item_id.encode()).hexdigest()[:8], 16))
    opts = [target] + rng.sample([c for c in range(k_all) if c != target], k - 1)
    rng.shuffle(opts)
    return opts


llm = LLM("Qwen/Qwen2.5-7B-Instruct", max_model_len=4096, gpu_memory_utilization=0.88, enable_prefix_caching=True)
tok = llm.get_tokenizer()
sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)


def chat(msg):
    return tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True)


def cls_prompt(text, opts_txt):
    o = "\n".join(f"{L[i]}. {t}" for i, t in enumerate(opts_txt))
    return chat(f"Text:\n{text[:1500]}\n\nWhich option best describes the text?\n{o}\n\nAnswer with the letter of the single best option only.")


def letter_pick(out, k):
    best = {}
    for tid, v in out.outputs[0].logprobs[0].items():
        s = (v.decoded_token or "").strip()
        if len(s) == 1 and s in L[:k]:
            best[s] = max(best.get(s, -1e9), v.logprob)
    return L.index(max(best, key=best.get)) if best else None


jobs, meta = [], []
for key, d in S.items():
    for split in ["train", "test"]:
        for it in d["splits"][split][:1000]:
            o = subset_opts(it["id"], it["target"], len(d["hyps"]))
            jobs.append(cls_prompt(it["text"], [d["hyps"][c] for c in o])); meta.append(("A", key, split, it, o))
for key in PARA:
    d = S[key]
    for split in ["train", "test"]:
        for it in d["splits"][split][:1000]:
            o = subset_opts(it["id"], it["target"], len(d["hyps"]))
            jobs.append(cls_prompt(P[it["id"]]["para"] or P[it["id"]]["orig"], [d["hyps"][c] for c in o])); meta.append(("Bpara", key, split, it, o))
            jobs.append(cls_prompt(P[it["id"]]["orig"], [d["hyps"][c] for c in o])); meta.append(("Borig", key, split, it, o))
            gold = d["hyps"][it["target"]]
            jobs.append(chat(f"Original text:\n{P[it['id']]['orig']}\n\nParaphrase:\n{P[it['id']]['para']}\n\n"
                             f"The original is labelled: \"{gold}\". Does the paraphrase keep the same meaning and the same label? "
                             f"Answer Yes or No.")); meta.append(("J", key, split, it, None))
t = time.time(); outs = llm.generate(jobs, sp); print("generated", len(outs), round(time.time() - t), "s", flush=True)
OUT = {}
for (kind, key, split, it, o), out in zip(meta, outs):
    if kind == "J":
        lp = {(v.decoded_token or "").strip().lower(): v.logprob for v in out.outputs[0].logprobs[0].values()}
        py, pn = math.exp(lp.get("yes", -30)), math.exp(lp.get("no", -30))
        OUT.setdefault("judge", {}).setdefault(key, {}).setdefault(split, []).append([it["id"], py / (py + pn)])
    else:
        j = letter_pick(out, len(o))
        pred = None if j is None else o[j]
        OUT.setdefault(kind, {}).setdefault(key, {}).setdefault(split, []).append([it["id"], it["target"], pred])
with gzip.open("r6_qwen.json.gz", "wt") as f:
    json.dump(OUT, f)
for kind in ["A", "Borig", "Bpara"]:
    for key, v in OUT[kind].items():
        print(kind, key, {s: round(sum(r[2] == r[1] for r in x) / len(x), 4) for s, x in v.items()}, flush=True)
for key, v in OUT["judge"].items():
    print("judge", key, {s: round(sum(p > 0.5 for _, p in x) / len(x), 4) for s, x in v.items()}, flush=True)
# human-check sample
rng = random.Random(0); ids = rng.sample([i for i in P if i.split(":")[0] in PARA], 60)
with open("paraphrase_sample_for_human_check.md", "w") as f:
    f.write("| id | gold label | original | paraphrase |\n|---|---|---|---|\n")
    for i in ids:
        key = i.split(":")[0]; it = next(x for s in ["train", "test"] for x in S[key]["splits"][s][:1000] if x["id"] == i)
        clean = lambda s: s.replace("|", "/").replace("\n", " ")[:400]
        f.write(f"| {i} | {S[key]['names'][it['target']]} | {clean(P[i]['orig'])} | {clean(P[i]['para'])} |\n")
print("done", flush=True)
