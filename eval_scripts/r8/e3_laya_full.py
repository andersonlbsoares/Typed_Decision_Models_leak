"""E3b: Laya (multilingual and english checkpoints) on the COMPLETE audit samples, same call as
r4_laya.run (state {"text": text[:4000]}, one 'choice' question, criteria c0..c{k-1}).
Output format matches r6_laya.json.gz: {model: {key: {A_train|A_test: [[id, gold, pred, conf]]}}}.
Usage (laya env): python r8/e3_laya_full.py [--models multilingual,english] [--limit 100] [--tag full]"""
import argparse, os, sys, time
os.environ["USE_TF"] = "0"
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, jdump, jload, load_cls, subset_opts
import laya

ap = argparse.ArgumentParser()
ap.add_argument("--models", default="multilingual,english"); ap.add_argument("--keys", default="")
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--tag", default="full")
a = ap.parse_args()
Q = "Which single label best describes the input text?"
S = load_cls(); keys = a.keys.split(",") if a.keys else list(S)
router = laya.Router(device="cuda", max_loaded=1)
outp = os.path.join(RES, f"r8_e3_laya_{a.tag}.json.gz")
OUT = jload(outp) if os.path.exists(outp) else {"meta": {"laya_version": getattr(laya, "__version__", "?")}}


def run(model, items, labels):
    recs = []
    for it in items:
        o = subset_opts(it["id"], it["target"], len(labels))
        crit = {f"c{j}": labels[c] for j, c in enumerate(o)}
        try:
            ans = router.predict({"text": it["text"][:4000]}, {"label": {"type": "choice", "instructions": Q, "criteria": crit}},
                                 model=model)["answers"]["label"]
            pr = ans.get("probabilities") or {}
            recs.append([it["id"], it["target"], o[int(ans["choice"][1:])], float(pr.get(ans["choice"], np.nan)) if pr else None])
        except Exception:
            recs.append([it["id"], it["target"], None, None])
    return recs


for model in a.models.split(","):
    for key in keys:
        d = S[key]
        for split in ["train", "test"]:
            g = OUT.setdefault(model, {}).setdefault(key, {})
            if f"A_{split}" in g:
                continue
            t0 = time.time(); items = d["splits"][split][:a.limit] if a.limit else d["splits"][split]
            g[f"A_{split}"] = r = run(model, items, d["hyps"])
            print(model, key, split, len(r), round(float(np.mean([x[2] == x[1] for x in r])), 4),
                  "fail", sum(x[2] is None for x in r), f"{len(r) / (time.time() - t0):.1f} it/s", flush=True)
            jdump(OUT, outp)
print("done", flush=True)
