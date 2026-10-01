"""Round 3 evaluation (julia venv).

A  train/test probe on 10 datasets: Julia-1, DeBERTa-zeroshot-v2.0-c (no task data),
   DeBERTa-zeroshot-v1.1-all-33 (documented: trained on these 10 tasks, ~51k texts over 28 tasks)
B  paraphrase probe: accuracy on original vs Qwen paraphrase, train[:1000] and test[:1000], 5 datasets
C  controlled contamination: fine-tune DeBERTa-v2.0-c ourselves with known exposure
   S1 train-exposed  : train[:1000]
   S2 test-contaminated: train[:1000] + test[:500]
   evaluated after epochs 1, 3, 6 on all groups, original and paraphrased text.
"""
import gzip, hashlib, json, os, random, sys, time, traceback
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_julia as RJ

S = json.load(gzip.open("r4_samples.json.gz", "rt"))["datasets"]
P = json.load(open("r4_para.json"))
PARA = ["agnews", "emotion", "banking77", "yelp", "yahoo"]
NP = 1000
OUT = {}
NLI = {"deberta_c": "MoritzLaurer/deberta-v3-base-zeroshot-v2.0-c",
       "deberta_all33": "MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33"}


def subset_opts(item_id, target, k_all, k=20):
    if k_all <= 20:
        return list(range(k_all))
    rng = random.Random(int(hashlib.md5(item_id.encode()).hexdigest()[:8], 16))
    opts = [target] + rng.sample([c for c in range(k_all) if c != target], k - 1)
    rng.shuffle(opts)
    return opts


def pack(recs):
    return [[r["id"], r["target"], r["pred"], round(r["conf"], 4) if r["conf"] is not None else None] for r in recs]


def acc(recs):
    return round(float(np.mean([r["pred"] == r["target"] for r in recs])), 4)


def groups():
    """Yield (dataset, group name, items(list of dict id/text/target)) for A and B."""
    for key, d in S.items():
        for split in ["train", "test"]:
            yield key, f"A_{split}", d["splits"][split]
    for key in PARA:
        d = S[key]
        for split in ["train", "test"]:
            base = d["splits"][split][:NP]
            yield key, f"B_{split}_orig", [dict(it, text=P[it["id"]]["orig"]) for it in base]
            yield key, f"B_{split}_para", [dict(it, text=P[it["id"]]["para"] or P[it["id"]]["orig"]) for it in base]


# ------------------------------------------------------------------ Julia
def julia_all():
    eng = RJ.load_engine("cuda", batch_size=64)
    for key, g, items in groups():
        labels = S[key]["hyps"]
        rows, opts = [], []
        for it in items:
            o = subset_opts(it["id"], it["target"], len(labels)); opts.append(o)
            rows.append(RJ.row(RJ.truncate(eng, it["text"])[0], [labels[c] for c in o]))
        lg = RJ.safe_logits(eng, rows)
        recs = []
        for it, o, z in zip(items, opts, lg):
            if z is None:
                recs.append(dict(id=it["id"], target=it["target"], pred=None, conf=None)); continue
            p = np.exp(np.array(z) - max(z)); p /= p.sum()
            recs.append(dict(id=it["id"], target=it["target"], pred=o[int(p.argmax())], conf=float(p.max())))
        OUT.setdefault("julia", {}).setdefault(key, {})[g] = pack(recs)
        print("julia", key, g, acc(recs), flush=True)
    del eng; torch.cuda.empty_cache()


# ------------------------------------------------------------------ NLI helpers
def load_nli(mid):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForSequenceClassification.from_pretrained(mid).to("cuda")
    ent = [i for i, l in model.config.id2label.items() if l.lower().startswith("entail")][0]
    return tok, model, int(ent)


@torch.inference_mode()
def nli_scores(tok, model, ent, texts, labels, bs=128, max_len=320):
    model.eval()
    pairs = [(t, h) for t in texts for h in labels]
    order = np.argsort([len(a) + len(b) for a, b in pairs]); res = np.zeros(len(pairs), np.float32)
    for i in range(0, len(pairs), bs):
        idx = order[i:i + bs]
        enc = tok([pairs[j][0] for j in idx], [pairs[j][1] for j in idx], truncation="only_first",
                  max_length=max_len, padding=True, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            res[idx] = model(**enc).logits.float()[:, ent].cpu().numpy()
    return res.reshape(len(texts), len(labels))


def nli_eval(tok, model, ent, items, labels):
    L = nli_scores(tok, model, ent, [it["text"] for it in items], labels)
    recs = []
    for it, z in zip(items, L):
        o = subset_opts(it["id"], it["target"], len(labels)); zz = z[o]; p = np.exp(zz - zz.max()); p /= p.sum()
        recs.append(dict(id=it["id"], target=it["target"], pred=o[int(p.argmax())], conf=float(p.max())))
    return recs


def nli_all():
    for mk, mid in NLI.items():
        tok, model, ent = load_nli(mid)
        for key, g, items in groups():
            recs = nli_eval(tok, model, ent, items, S[key]["hyps"])
            OUT.setdefault(mk, {}).setdefault(key, {})[g] = pack(recs)
            print(mk, key, g, acc(recs), flush=True)
        del model; torch.cuda.empty_cache()


# ------------------------------------------------------------------ C: controlled contamination
def finetune_eval(key, scenario, epochs=(1, 3, 6), lr=2e-5, bs=32, seed=0):
    torch.manual_seed(seed); random.seed(seed)
    tok, model, ent = load_nli(NLI["deberta_c"])
    not_ent = 1 - ent
    d = S[key]; labels = d["hyps"]
    tr = d["splits"]["train"][:NP]; te = d["splits"]["test"][:NP]
    seen = list(tr) + (list(te[:500]) if scenario == "S2_train+test500" else [])
    pairs = []
    for it in seen:
        text = P[it["id"]]["orig"] if it["id"] in P else it["text"]
        for c in subset_opts(it["id"], it["target"], len(labels)):
            pairs.append((text, labels[c], ent if c == it["target"] else not_ent))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    evals = {
        "train_seen": [dict(it, text=P[it["id"]]["orig"]) for it in tr],
        "test_seen": [dict(it, text=P[it["id"]]["orig"]) for it in te[:500]],
        "test_unseen": [dict(it, text=P[it["id"]]["orig"]) for it in te[500:]],
    }
    base_groups = list(evals)
    evals["train_unseen"] = [dict(it, text=it["text"][:1200]) for it in d["splits"]["train"][NP:2 * NP]]
    for g in base_groups:
        evals[g + "_para"] = [dict(it, text=P[it["id"]]["para"] or P[it["id"]]["orig"]) for it in evals[g]]
    res = {}
    done = 0
    for ep in range(1, max(epochs) + 1):
        model.train(); random.shuffle(pairs)
        for i in range(0, len(pairs), bs):
            b = pairs[i:i + bs]
            enc = tok([x[0] for x in b], [x[1] for x in b], truncation="only_first", max_length=256,
                      padding=True, return_tensors="pt").to("cuda")
            y = torch.tensor([x[2] for x in b], device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = torch.nn.functional.cross_entropy(model(**enc).logits.float(), y)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep in epochs:
            r = {}
            for g, items in evals.items():
                r[g] = pack(nli_eval(tok, model, ent, items, labels))
            res[ep] = r
            print("C", key, scenario, "epoch", ep, {g: acc([dict(id=a, target=b, pred=c) for a, b, c, _ in v]) for g, v in r.items()}, flush=True)
    del model; torch.cuda.empty_cache()
    return res


if __name__ == "__main__":
    t0 = time.time()
    for name, fn in [("julia", julia_all), ("nli", nli_all)]:
        try:
            fn()
        except Exception:
            traceback.print_exc()
        with gzip.open("r4_eval.json.gz", "wt") as f:
            json.dump(OUT, f)
        print("stage", name, round(time.time() - t0), "s", flush=True)
    for key in ["emotion", "agnews"]:
        for sc in ["S1_train", "S2_train+test500"]:
            try:
                OUT.setdefault("controlled", {}).setdefault(key, {})[sc] = finetune_eval(key, sc)
            except Exception:
                traceback.print_exc()
            with gzip.open("r4_eval.json.gz", "wt") as f:
                json.dump(OUT, f)
    print("done", round(time.time() - t0), "s", flush=True)
