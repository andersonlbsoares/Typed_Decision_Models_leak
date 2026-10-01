"""Round 3 for Laya (laya venv): A probe on 10 datasets (english: 1000/split, multilingual: 500/split)
and B paraphrase probe (english, 500 per group)."""
import gzip, hashlib, json, os, random, time, traceback
import numpy as np
os.environ["USE_TF"] = "0"
import laya
S = json.load(gzip.open("r4_samples.json.gz", "rt"))["datasets"]
P = json.load(open("r4_para.json"))
PARA = ["agnews", "emotion", "banking77", "yelp", "yahoo"]
Q = "Which single label best describes the input text?"
router = laya.Router(device="cuda", max_loaded=1)
OUT = {}


def subset_opts(item_id, target, k_all, k=20):
    if k_all <= 20:
        return list(range(k_all))
    rng = random.Random(int(hashlib.md5(item_id.encode()).hexdigest()[:8], 16))
    opts = [target] + rng.sample([c for c in range(k_all) if c != target], k - 1)
    rng.shuffle(opts)
    return opts


def run(model, items, labels):
    recs = []
    for it in items:
        o = subset_opts(it["id"], it["target"], len(labels))
        crit = {f"c{j}": labels[c] for j, c in enumerate(o)}
        try:
            a = router.predict({"text": it["text"][:4000]}, {"label": {"type": "choice", "instructions": Q, "criteria": crit}},
                               model=model)["answers"]["label"]
            pr = a.get("probabilities") or {}
            recs.append([it["id"], it["target"], o[int(a["choice"][1:])], float(pr.get(a["choice"], np.nan)) if pr else None])
        except Exception as ex:
            recs.append([it["id"], it["target"], None, None])
    return recs


def acc(r):
    return round(float(np.mean([x[2] == x[1] for x in r])), 4)


t0 = time.time()
for model, n, do_para in [("english", 1000, True), ("multilingual", 500, False)]:
    try:
        for key, d in S.items():
            for split in ["train", "test"]:
                r = run(model, d["splits"][split][:n], d["hyps"])
                OUT.setdefault(model, {}).setdefault(key, {})[f"A_{split}"] = r
                print(model, key, split, acc(r), round(time.time() - t0), flush=True)
        if do_para:
            for key in PARA:
                d = S[key]
                for split in ["train", "test"]:
                    base = d["splits"][split][:500]
                    for v in ["orig", "para"]:
                        items = [dict(it, text=(P[it["id"]][v] or P[it["id"]]["orig"])) for it in base]
                        r = run(model, items, d["hyps"])
                        OUT[model][key][f"B_{split}_{v}"] = r
                        print(model, key, split, v, acc(r), flush=True)
    except Exception:
        traceback.print_exc()
    with gzip.open("r4_laya.json.gz", "wt") as f:
        json.dump(OUT, f)
print("done", round(time.time() - t0), flush=True)
