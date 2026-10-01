"""Round 2: train/test probe, verbalization study and typed-decisions train/test for Julia-1 + NLI."""
import gzip, hashlib, json, os, random, sys, time
from collections import defaultdict
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_julia as RJ

S = json.load(gzip.open("r2_samples.json.gz", "rt"))
OUT = {}
NV = 1000


def subset_opts(item, k_all, k=20):
    """Deterministic 20-option subset (gold + 19 random) for >20-label datasets."""
    if k_all <= 20:
        return list(range(k_all))
    rng = random.Random(int(hashlib.md5(item["id"].encode()).hexdigest()[:8], 16))
    opts = [item["target"]] + rng.sample([c for c in range(k_all) if c != item["target"]], k - 1)
    rng.shuffle(opts)
    return opts


def julia_eval(eng, items, labels):
    rows, opts_l = [], []
    for it in items:
        o = subset_opts(it, len(labels)); opts_l.append(o)
        rows.append(RJ.row(RJ.truncate(eng, it["text"])[0], [labels[c] for c in o]))
    lg = RJ.safe_logits(eng, rows)
    recs = []
    for it, o, z in zip(items, opts_l, lg):
        if z is None:
            recs.append(dict(id=it["id"], target=it["target"], pred=None, conf=None)); continue
        p = np.exp(np.array(z) - max(z)); p /= p.sum()
        recs.append(dict(id=it["id"], target=it["target"], opts=o, pred=o[int(p.argmax())], conf=float(p.max()),
                         p_gold=float(p[o.index(it["target"])])))
    return recs


def summarize(recs):
    ok = [r for r in recs if r["pred"] is not None]
    return dict(n=len(recs), acc=round(sum(r["pred"] == r["target"] for r in ok) / len(recs), 4),
                mean_conf=round(float(np.mean([r["conf"] for r in ok])), 4))


def typed_rows(cases):
    rows, meta = [], []
    for case in cases:
        state = json.loads(case["state"]); gold = json.loads(case["gold"])
        for qid, q in json.loads(case["questions"]).items():
            kind = q["type"]; crit = q.get("criteria")
            if crit is None and kind == "noul":
                crit = {"false": "false", "true": "true"}
            if isinstance(crit, list):
                crit = {str(i): v for i, v in enumerate(crit)}
            keys = ["false", "true"] if kind == "noul" else list(crit)
            rows.append(dict(state=state, question=q["instructions"], type=kind, options=[crit[k] for k in keys]))
            meta.append(dict(id=case["id"] + ":" + qid, type=kind, keys=keys, gold=str(gold[qid]["label"]),
                             workflow=case.get("workflow") or case.get("config") or case["id"].split("-")[0]))
    return rows, meta


if __name__ == "__main__":
    eng = RJ.load_engine("cuda", batch_size=64)
    t0 = time.time()
    # --- typed-decisions train vs test
    for split in ["train", "test"]:
        rows, meta = typed_rows(S["typed"][split])
        lg = RJ.safe_logits(eng, rows, chunk=256)
        recs = []
        for m, z in zip(meta, lg):
            p = np.exp(np.array(z) - max(z)); p /= p.sum()
            recs.append(dict(**m, pred=m["keys"][int(p.argmax())], conf=float(p.max())))
        OUT.setdefault("typed", {})[split] = recs
        by = defaultdict(list)
        for r in recs:
            by[r["type"]].append(r["pred"] == r["gold"])
        print("typed", split, len(recs), {k: round(float(np.mean(v)), 4) for k, v in by.items()},
              "all", round(float(np.mean([r["pred"] == r["gold"] for r in recs])), 4), flush=True)
    # --- train/test probe
    for key, d in S["datasets"].items():
        for split in ["train", "test"]:
            recs = julia_eval(eng, d["splits"][split], d["hyps"])
            OUT.setdefault("probe", {}).setdefault(key, {})[split] = recs
            print("probe julia", key, split, summarize(recs), flush=True)
    # --- verbalization (test)
    for key, d in S["datasets"].items():
        items = d["splits"]["test"][:NV]
        for v in ["v0", "v1", "v2"]:
            labels = d["hyps"] if v == "v0" else d.get(v)
            if labels is None:
                continue
            recs = julia_eval(eng, items, labels)
            OUT.setdefault("verbal", {}).setdefault(key, {})[v] = recs
            print("verbal julia", key, v, summarize(recs), flush=True)
    del eng; torch.cuda.empty_cache()
    # --- NLI baseline (full label scoring, then same subsets)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import run_nli as RN
    mid = RN.MODELS["deberta_zs_v2c"]
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForSequenceClassification.from_pretrained(mid).to("cuda").eval()
    ent = [i for i, l in model.config.id2label.items() if l.lower().startswith("entail")][0]
    def nli_eval(items, labels):
        L = RN.score_examples(model, tok, int(ent), [it["text"] for it in items], labels)
        recs = []
        for it, z in zip(items, L):
            o = subset_opts(it, len(labels)); zz = z[o]; p = np.exp(zz - zz.max()); p /= p.sum()
            recs.append(dict(id=it["id"], target=it["target"], opts=o, pred=o[int(p.argmax())], conf=float(p.max()),
                             p_gold=float(p[o.index(it["target"])])))
        return recs
    for key, d in S["datasets"].items():
        for split in ["train", "test"]:
            recs = nli_eval(d["splits"][split], d["hyps"])
            OUT.setdefault("probe_nli", {}).setdefault(key, {})[split] = recs
            print("probe nli", key, split, summarize(recs), flush=True)
        items = d["splits"]["test"][:NV]
        for v in ["v0", "v1", "v2"]:
            labels = d["hyps"] if v == "v0" else d.get(v)
            if labels is None:
                continue
            if v == "v1":
                labels = [f"This example is about {x}." for x in labels]
            recs = nli_eval(items, labels)
            OUT.setdefault("verbal_nli", {}).setdefault(key, {})[v] = recs
            print("verbal nli", key, v, summarize(recs), flush=True)
    with gzip.open("r2_julia_nli.json.gz", "wt") as f:
        json.dump(OUT, f)
    print("done", round(time.time() - t0), "s", flush=True)
