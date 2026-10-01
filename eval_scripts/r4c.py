"""Rerun of experiment C (controlled contamination) with fp32 training, grad clipping and loss logging."""
import gzip, json, math, random, sys, os, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r4_eval as E


def finetune_eval(key, scenario, epochs=(1, 3, 6), lr=1e-5, bs=16, seed=0):
    torch.manual_seed(seed); random.seed(seed)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    mid = E.NLI["deberta_c"]
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForSequenceClassification.from_pretrained(mid, dtype=torch.float32).to("cuda")
    ent = int([i for i, l in model.config.id2label.items() if l.lower().startswith("entail")][0]); not_ent = 1 - ent
    print("param dtype", next(model.parameters()).dtype, flush=True)
    d = E.S[key]; labels = d["hyps"]; P = E.P
    tr = d["splits"]["train"][:E.NP]; te = d["splits"]["test"][:E.NP]
    seen = list(tr) + (list(te[:500]) if scenario == "S2_train+test500" else [])
    pairs = []
    for it in seen:
        text = P[it["id"]]["orig"]
        for c in E.subset_opts(it["id"], it["target"], len(labels)):
            pairs.append((text, labels[c], ent if c == it["target"] else not_ent))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    evals = {"train_seen": [dict(it, text=P[it["id"]]["orig"]) for it in tr],
             "test_seen": [dict(it, text=P[it["id"]]["orig"]) for it in te[:500]],
             "test_unseen": [dict(it, text=P[it["id"]]["orig"]) for it in te[500:]]}
    base = list(evals)
    evals["train_unseen"] = [dict(it, text=it["text"][:1200]) for it in d["splits"]["train"][E.NP:2 * E.NP]]
    for g in base:
        evals[g + "_para"] = [dict(it, text=P[it["id"]]["para"] or P[it["id"]]["orig"]) for it in evals[g]]
    res = {0: {g: E.pack(E.nli_eval(tok, model, ent, items, labels)) for g, items in evals.items()}}
    print("C", key, scenario, "epoch 0", {g: E.acc([dict(id=a, target=b, pred=c) for a, b, c, _ in v]) for g, v in res[0].items()}, flush=True)
    for ep in range(1, max(epochs) + 1):
        model.train(); random.shuffle(pairs); losses = []
        for i in range(0, len(pairs), bs):
            b = pairs[i:i + bs]
            enc = tok([x[0] for x in b], [x[1] for x in b], truncation="only_first", max_length=256,
                      padding=True, return_tensors="pt").to("cuda")
            y = torch.tensor([x[2] for x in b], device="cuda")
            loss = torch.nn.functional.cross_entropy(model(**enc).logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite loss")
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            losses.append(loss.item())
        print("  epoch", ep, "mean loss", round(float(np.mean(losses)), 4), flush=True)
        if ep in epochs:
            res[ep] = {g: E.pack(E.nli_eval(tok, model, ent, items, labels)) for g, items in evals.items()}
            print("C", key, scenario, "epoch", ep, {g: E.acc([dict(id=a, target=b, pred=c) for a, b, c, _ in v]) for g, v in res[ep].items()}, flush=True)
    del model; torch.cuda.empty_cache()
    return res


if __name__ == "__main__":
    OUT = {}
    for key in ["emotion", "agnews"]:
        for sc in ["S1_train", "S2_train+test500"]:
            OUT.setdefault(key, {})[sc] = finetune_eval(key, sc)
            with gzip.open("r4c.json.gz", "wt") as f:
                json.dump(OUT, f)
    print("done", flush=True)
