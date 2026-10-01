"""Controlled contamination v2 (DeBERTa-v3-base-zeroshot-v2.0-c, fp32).
Grid per dataset: seeds x exposure size n in {250, 1000} x scenario {S1 train-only, S2 train + test[:500]};
evaluated after epochs 1 and 3. Stores per-example correctness for every group."""
import gzip, json, os, random, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r4_eval as E
from transformers import AutoModelForSequenceClassification, AutoTokenizer

key = sys.argv[1]; seeds = [int(s) for s in sys.argv[2].split(",")]
MID = E.NLI["deberta_c"]; P = E.P


def groups_for(d, n):
    tr = d["splits"]["train"]; te = d["splits"]["test"]
    g = {"train_seen": tr[:n], "train_unseen": tr[n:1000] if n < 1000 else tr[1000:2000],
         "test_seen": te[:500], "test_unseen": te[500:1000]}
    out = {}
    for name, items in g.items():
        out[name] = [dict(it, text=P[it["id"]]["orig"] if it["id"] in P else it["text"][:1200]) for it in items]
        if all(it["id"] in P for it in items):
            out[name + "_para"] = [dict(it, text=P[it["id"]]["para"] or P[it["id"]]["orig"]) for it in items]
    return out


def run(seed, n, scenario, epochs=(1, 3), lr=1e-5, bs=16):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    tok = AutoTokenizer.from_pretrained(MID)
    model = AutoModelForSequenceClassification.from_pretrained(MID, dtype=torch.float32).to("cuda")
    ent = int([i for i, l in model.config.id2label.items() if l.lower().startswith("entail")][0]); ne = 1 - ent
    d = E.S[key]; labels = d["hyps"]; G = groups_for(d, n)
    seen = list(G["train_seen"]) + (list(G["test_seen"]) if scenario == "S2" else [])
    pairs = [(it["text"], labels[c], ent if c == it["target"] else ne)
             for it in seen for c in E.subset_opts(it["id"], it["target"], len(labels))]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    res = {}
    for ep in range(1, max(epochs) + 1):
        model.train(); random.shuffle(pairs)
        for i in range(0, len(pairs), bs):
            b = pairs[i:i + bs]
            enc = tok([x[0] for x in b], [x[1] for x in b], truncation="only_first", max_length=256,
                      padding=True, return_tensors="pt").to("cuda")
            loss = torch.nn.functional.cross_entropy(model(**enc).logits, torch.tensor([x[2] for x in b], device="cuda"))
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite loss")
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if ep in epochs:
            res[ep] = {g: [int(r["pred"] == r["target"]) for r in E.nli_eval(tok, model, ent, items, labels)]
                       for g, items in G.items()}
            print(key, seed, n, scenario, ep, {g: round(float(np.mean(v)), 3) for g, v in res[ep].items()}, flush=True)
    del model; torch.cuda.empty_cache()
    return res


if __name__ == "__main__":
    OUT = {}
    for seed in seeds:
        for n in [250, 1000]:
            for sc in ["S1", "S2"]:
                OUT[f"{seed}|{n}|{sc}"] = run(seed, n, sc)
                with gzip.open(f"r6_ctrl_{key}.json.gz", "wt") as f:
                    json.dump(OUT, f)
    print("done", key, flush=True)
