"""Three-split probe: accuracy on validation vs test (same sampling), for datasets with a validation split.
Train-only exposure -> val ~ test; test exposure -> test > val."""
import gzip, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2_data as R2, r4_data as R4, run_julia as RJ
VAL = {"emotion": "validation", "massive": "validation", "wikitoxic": "validation"}
out = {"samples": {}}
for key, vsplit in VAL.items():
    repo, cfg, tfn, lcol, bname, show = R4.SPECS[key]
    hyps, names = R2.btzsc_hyp_names(bname); idx = {R2.norm(n): i for i, n in enumerate(names)}
    ds, used = R4.load_split(repo, cfg, vsplit)
    feat = ds.features.get(lcol); lnames = feat.names if hasattr(feat, "names") else None
    items = []
    for i, r in enumerate(ds):
        lab = r[lcol]
        nm = r["label_text"] if "label_text" in r and r["label_text"] else (lnames[lab] if lnames else str(lab))
        nm = R4.ALIASES.get(key, {}).get(str(nm), nm); c = idx.get(R2.norm(nm))
        if c is None or not tfn(r):
            continue
        items.append(dict(id=f"{key}:val:{i}", text=str(tfn(r)), target=c))
    sel = RJ.balanced_indices([it["target"] for it in items], 2000, 20260929)
    out["samples"][key] = dict(hyps=hyps, items=[items[j] for j in sel])
    print(key, vsplit, len(items), "->", len(sel), flush=True)
import r4_eval as E
E.S = {k: dict(hyps=v["hyps"], splits={"train": [], "test": v["items"]}) for k, v in out["samples"].items()}
eng = RJ.load_engine("cuda", batch_size=64)
for key, v in out["samples"].items():
    rows, opts = [], []
    for it in v["items"]:
        o = E.subset_opts(it["id"], it["target"], len(v["hyps"])); opts.append(o)
        rows.append(RJ.row(RJ.truncate(eng, it["text"])[0], [v["hyps"][c] for c in o]))
    lg = RJ.safe_logits(eng, rows); recs = []
    for it, o, z in zip(v["items"], opts, lg):
        recs.append([it["id"], it["target"], None if z is None else o[int(np.argmax(z))], None])
    out.setdefault("julia", {})[key] = recs
    print("julia", key, round(float(np.mean([r[2] == r[1] for r in recs])), 4), flush=True)
del eng; torch.cuda.empty_cache()
for mk, mid in E.NLI.items():
    tok, model, ent = E.load_nli(mid)
    for key, v in out["samples"].items():
        recs = E.pack(E.nli_eval(tok, model, ent, v["items"], v["hyps"]))
        out.setdefault(mk, {})[key] = recs
        print(mk, key, round(float(np.mean([r[2] == r[1] for r in recs])), 4), flush=True)
    del model; torch.cuda.empty_cache()
out.pop("samples")
with gzip.open("r5_val.json.gz", "wt") as f:
    json.dump(out, f)
print("done", flush=True)
