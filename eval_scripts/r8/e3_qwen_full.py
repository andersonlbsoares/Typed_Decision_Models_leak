"""E3a: Qwen2.5-7B-Instruct on the COMPLETE audit samples (2,000/2,000; Rotten Tomatoes 2,000/1,066),
same prompt and option subsets as r6_qwen.py. Scoring: one forward pass, next-token log-probabilities
of the option letters (each letter as 'A' or ' A' token, max), argmax over admissible letters.
To mirror the stored vLLM protocol (letter must appear among the top-20 tokens), `pred_top20` is None
when no admissible letter is in the top 20; `pred` is the unrestricted argmax. Left padding with
explicit position ids. Output format matches r6_qwen['A'] plus confidence.
Usage: python r8/e3_qwen_full.py [--keys agnews,emotion] [--limit 256] [--tag full]"""
import argparse, os, string, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, jdump, jload, load_cls, revisions, subset_opts
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--keys", default=""); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--tag", default="full"); ap.add_argument("--tok-budget", type=int, default=48000)
a = ap.parse_args()
S = load_cls(); keys = a.keys.split(",") if a.keys else list(S)
path = revisions()["qwen"]["path"]
tok = AutoTokenizer.from_pretrained(path); tok.padding_side = "left"
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
print("loaded", round(time.time() - t0), "s", flush=True)
LET = string.ascii_uppercase
LID = []
for c in LET[:20]:
    ids = {e[0] for v in (c, " " + c) for e in [tok.encode(v, add_special_tokens=False)] if len(e) == 1}
    LID.append(sorted(ids))


def chat(msg):
    return tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True)


def cls_prompt(text, opts_txt):
    o = "\n".join(f"{LET[i]}. {t}" for i, t in enumerate(opts_txt))
    return chat(f"Text:\n{text[:1500]}\n\nWhich option best describes the text?\n{o}\n\nAnswer with the letter of the single best option only.")


outp = os.path.join(RES, f"r8_e3_qwen_{a.tag}.json.gz")
OUT = jload(outp) if os.path.exists(outp) else {"A": {}, "meta": dict(model=revisions()["qwen"], letter_ids=LID)}
for key in keys:
    d = S[key]
    for split in ["train", "test"]:
        if split in OUT["A"].get(key, {}):
            continue
        items = d["splits"][split][:a.limit] if a.limit else d["splits"][split]
        opts = [subset_opts(it["id"], it["target"], len(d["hyps"])) for it in items]
        enc = [tok(cls_prompt(it["text"], [d["hyps"][c] for c in o]), add_special_tokens=False)["input_ids"]
               for it, o in zip(items, opts)]
        order = np.argsort([-len(e) for e in enc]); res = [None] * len(items); t1 = time.time(); i = 0
        while i < len(order):
            L = len(enc[order[i]]); nb = max(1, min(64, a.tok_budget // L)); idx = order[i:i + nb]; i += nb
            ids = torch.full((len(idx), L), tok.pad_token_id); att = torch.zeros((len(idx), L), dtype=torch.long)
            for r, j in enumerate(idx):
                e = enc[j]; ids[r, L - len(e):] = torch.tensor(e); att[r, L - len(e):] = 1
            pos = (att.cumsum(-1) - 1).clamp(min=0)
            with torch.inference_mode():
                kw = dict(input_ids=ids.cuda(), attention_mask=att.cuda(), position_ids=pos.cuda())
                try:
                    lg = model(**kw, logits_to_keep=1).logits[:, -1].float()
                except TypeError:
                    lg = model(**kw).logits[:, -1].float()
                lp = torch.log_softmax(lg, -1); top20 = lp.topk(20, -1).values[:, -1]
            lp = lp.cpu().numpy(); top20 = top20.cpu().numpy()
            for r, j in enumerate(idx):
                k = len(opts[j]); sc = np.array([lp[r, LID[c]].max() for c in range(k)])
                b = int(sc.argmax()); p = np.exp(sc - sc.max()); p /= p.sum()
                res[j] = [items[j]["id"], items[j]["target"], opts[j][b], round(float(p[b]), 6),
                          opts[j][b] if sc[b] >= top20[r] else None]
        OUT["A"].setdefault(key, {})[split] = res
        acc = np.mean([r[2] == r[1] for r in res])
        print(key, split, len(res), round(float(acc), 4), f"{len(res) / (time.time() - t1):.1f} it/s", flush=True)
        jdump(OUT, outp)
print("done", flush=True)
