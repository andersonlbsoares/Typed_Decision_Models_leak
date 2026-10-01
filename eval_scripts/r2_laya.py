"""Round 2 for Laya checkpoints: typed-decisions train/test (base, multilingual, typed-decisions
fine-tune = positive control) and the dataset train/test probe + verbalization (English base)."""
import gzip, hashlib, inspect, json, os, random, sys, time, traceback
from collections import defaultdict
import numpy as np
os.environ["USE_TF"] = "0"
import laya

S = json.load(gzip.open("r2_samples.json.gz", "rt"))
Q = "Which single label best describes the input text?"
OUT = {"raw_examples": []}


_ROUTER = None


class _M:
    def __init__(self, model):
        self.model = model

    def predict(self, state, qs):
        return _ROUTER.predict(state, qs, model=self.model)


def load(sub):
    """Documented CUDA path: Router(device="cuda").predict(..., model=<checkpoint>)."""
    global _ROUTER
    if _ROUTER is None:
        _ROUTER = laya.Router(device="cuda", max_loaded=1)
    return _M({None: "english", "typed-decisions": "typed-decisions", "multilingual": "multilingual"}[sub])


def subset_opts(item, k_all, k=20):
    if k_all <= 20:
        return list(range(k_all))
    rng = random.Random(int(hashlib.md5(item["id"].encode()).hexdigest()[:8], 16))
    opts = [item["target"]] + rng.sample([c for c in range(k_all) if c != item["target"]], k - 1)
    rng.shuffle(opts)
    return opts


def choice_eval(m, items, labels, tag):
    recs = []
    for it in items:
        o = subset_opts(it, len(labels))
        crit = {f"c{j}": labels[c] for j, c in enumerate(o)}
        try:
            a = m.predict({"text": it["text"][:4000]}, {"label": {"type": "choice", "instructions": Q, "criteria": crit}})["answers"]["label"]
            if len(OUT["raw_examples"]) < 6:
                OUT["raw_examples"].append({"tag": tag, "answer": a})
            ch = a["choice"]; pr = a.get("probabilities") or {}
            conf = float(pr.get(ch, a.get("confidence", np.nan))) if pr or "confidence" in a else None
            recs.append(dict(id=it["id"], target=it["target"], pred=o[int(ch[1:])], conf=conf))
        except Exception as ex:
            recs.append(dict(id=it["id"], target=it["target"], pred=None, conf=None, err=str(ex)[:120]))
    return recs


def typed_eval(m, cases, tag):
    recs = []
    for case in cases:
        state = json.loads(case["state"]); gold = json.loads(case["gold"]); qs = json.loads(case["questions"])
        try:
            ans = m.predict(state, qs)["answers"]
        except Exception as ex:
            ans = {}
            for qid, q in qs.items():
                try:
                    ans.update(m.predict(state, {qid: q})["answers"])
                except Exception as ex2:
                    pass
        if len(OUT["raw_examples"]) < 12:
            OUT["raw_examples"].append({"tag": tag, "answers": ans, "gold": gold})
        for qid, q in qs.items():
            a = ans.get(qid); g = str(gold[qid]["label"])
            if a is None:
                recs.append(dict(id=case["id"] + ":" + qid, type=q["type"], gold=g, pred=None)); continue
            pr = a.get("probabilities")
            if isinstance(pr, dict) and pr:
                pred = max(pr, key=pr.get)
            elif q["type"] == "choice":
                pred = a["choice"]
            elif q["type"] == "score":
                pred = str(int(round(a["score"])))
            else:
                pred = "true" if a["noul"] > 0.5 else "false"
            if q["type"] == "noul" and pred in ("True", "False", "1", "0", "yes", "no"):
                pred = {"True": "true", "False": "false", "1": "true", "0": "false", "yes": "true", "no": "false"}[pred]
            recs.append(dict(id=case["id"] + ":" + qid, type=q["type"], gold=g, pred=str(pred)))
    return recs


if __name__ == "__main__":
    OUT["laya_version"] = getattr(laya, "__version__", "?")
    OUT["router_signature"] = str(inspect.signature(laya.Router))
    t0 = time.time()
    for sub, name in [(None, "laya_en"), ("typed-decisions", "laya_typed"), ("multilingual", "laya_ml")]:
        try:
            m = load(sub)
            for split in ["train", "test"]:
                recs = typed_eval(m, S["typed"][split], f"{name}:typed:{split}")
                OUT.setdefault("typed", {}).setdefault(name, {})[split] = recs
                by = defaultdict(list)
                for r in recs:
                    by[r["type"]].append(r["pred"] == r["gold"])
                print(name, "typed", split, {k: round(float(np.mean(v)), 4) for k, v in by.items()},
                      "all", round(float(np.mean([r["pred"] == r["gold"] for r in recs])), 4), round(time.time() - t0), flush=True)
            if name == "laya_en":
                for key, d in S["datasets"].items():
                    for split in ["train", "test"]:
                        recs = choice_eval(m, d["splits"][split][::2], d["hyps"], f"{key}:{split}")
                        OUT.setdefault("probe", {}).setdefault(key, {})[split] = recs
                        print(name, "probe", key, split, round(float(np.mean([r["pred"] == r["target"] for r in recs])), 4), round(time.time() - t0), flush=True)
                    for v in ["v0", "v1", "v2"]:
                        labels = d["hyps"] if v == "v0" else d.get(v)
                        if labels is None:
                            continue
                        recs = choice_eval(m, d["splits"]["test"][:500], labels, f"{key}:{v}")
                        OUT.setdefault("verbal", {}).setdefault(key, {})[v] = recs
                        print(name, "verbal", key, v, round(float(np.mean([r["pred"] == r["target"] for r in recs])), 4), flush=True)
            del m
            import torch; torch.cuda.empty_cache()
        except Exception as ex:
            OUT[name + "_error"] = f"{type(ex).__name__}: {ex}"[:500]; traceback.print_exc()
        with gzip.open("r2_laya.json.gz", "wt") as f:
            json.dump(OUT, f)
    print("done", round(time.time() - t0), flush=True)
