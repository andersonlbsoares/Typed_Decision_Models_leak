"""Rebuild the exact audit samples from the stored item ids (no re-sampling), verify them, and draw
the E1 filler pool. Runs offline after stage_download.py (login node or any node).
Checks: every item's gold label recomputed from the dataset equals the stored gold; for the 5
paraphrase datasets the first 1,000 texts per split equal r4_para 'orig' (text[:1200]); typed-decisions
question ids match the stored predictions. Any mismatch aborts."""
import os, sys
import pyarrow.parquet as pq
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "code"))
os.environ.setdefault("JULIA_CKPT", "/nonexistent")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from common import DATA, REF, jload, jdump, revisions
import r2_data as R2, r4_data as R4, run_julia as RJ

E1_KEYS = ["emotion", "agnews", "yelp", "yahoo"]
FILLER_N, FILLER_SEED = 4000, 20261007


def gold_of(row_label, row_label_text, lnames, key, idx):
    """Label index under the r4_data mapping. Fallback for Rotten Tomatoes, whose ClassLabel names are
    neg/pos: also try str(label) through the aliases (0 -> negative, 1 -> positive). Stored gold is checked."""
    al = R4.ALIASES.get(key, {})
    cands = []
    if row_label_text:
        cands.append(row_label_text)
    if lnames:
        cands.append(lnames[row_label])
    cands.append(str(row_label))
    for nm in cands:
        c = idx.get(R2.norm(al.get(str(nm), nm)))
        if c is not None:
            return c
    return None


def main():
    rev = revisions()
    e4 = jload(os.path.join(REF, "r4_eval.json.gz"))["julia"]
    ro = jload(os.path.join(REF, "r6_rotten_eval.json.gz"))["julia"]
    para = jload(os.path.join(REF, "r4_para.json"))
    out, rotten, filler, report = {"datasets": {}}, None, {}, {}
    only = [k for k in os.environ.get("R8_ONLY", "").split(",") if k]  # dry runs only
    for key, spec in R4.SPECS.items():
        if only and key not in only:
            continue
        repo, cfg, tfn, lcol, bname, show = spec
        hyps, names = R2.btzsc_hyp_names(bname)
        idx = {R2.norm(n): i for i, n in enumerate(names)}
        meta = rev[key]; lnames = meta["label_names"]
        info = dict(hyps=hyps, names=names, showcased=show, source=meta["repo"], revision=meta["rev"], splits={})
        for split in ["train", "test"]:
            t = pq.read_table(meta["splits"][split]["file"])
            stored = (ro if key == "rotten" else e4[key])[f"A_{split}"]
            items, bad = [], 0
            for sid, gold, *_ in stored:
                i = int(sid.split(":")[2])
                row = t.slice(i, 1).to_pylist()[0]
                c = gold_of(row[lcol], row.get("label_text"), lnames, key, idx)
                bad += c != gold
                items.append(dict(id=sid, text=str(tfn(row)), target=gold))
            if bad:
                raise SystemExit(f"{key}/{split}: {bad} gold-label mismatches -> wrong revision or mapping")
            pmis = sum(1 for it in items[:1000] if it["id"] in para and it["text"][:1200] != para[it["id"]]["orig"])
            if pmis:
                raise SystemExit(f"{key}/{split}: {pmis} texts differ from r4_para")
            info["splits"][split] = items
            report[f"{key}/{split}"] = dict(n=len(items), gold_mismatch=0, para_checked=sum(it["id"] in para for it in items[:1000]))
            print(key, split, len(items), "ok", flush=True)
            if key in E1_KEYS and split == "train":
                used = {int(it["id"].split(":")[2]) for it in items}
                lab = t.column(lcol).to_pylist()
                lt = t.column("label_text").to_pylist() if "label_text" in t.column_names else None
                cand, tg = [], []
                for i, l in enumerate(lab):
                    if i in used:
                        continue
                    c = gold_of(l, lt[i] if lt else None, lnames, key, idx)
                    if c is not None:
                        cand.append(i); tg.append(c)
                sel = RJ.balanced_indices(tg, FILLER_N + 200, FILLER_SEED)
                pool = []
                for j in sel:
                    row = t.slice(cand[j], 1).to_pylist()[0]; txt = tfn(row)
                    if txt and str(txt).strip():
                        pool.append(dict(id=f"{key}:train:{cand[j]}", text=str(txt), target=tg[j]))
                filler[key] = pool[:FILLER_N]
                assert len(filler[key]) == FILLER_N and not used & {int(x["id"].split(":")[2]) for x in filler[key]}
                report[f"{key}/filler"] = dict(n=FILLER_N, candidates=len(cand))
        if key == "rotten":
            rotten = info
        else:
            out["datasets"][key] = info
    jdump(out, os.path.join(DATA, "r4_samples.json.gz"))
    jdump(rotten, os.path.join(DATA, "r6_rotten_samples.json.gz"))
    jdump(filler, os.path.join(DATA, "e1_filler.json.gz"))
    # typed-decisions: full splits, order as in the parquet files
    r2t = jload(os.path.join(REF, "r2_julia_nli.json.gz"))["typed"]
    typed = {s: pq.read_table(rev["typed_files"][s]).to_pylist() for s in ["train", "test"]}
    from common import typed_rows
    for s in ["train", "test"]:
        _, meta = typed_rows(typed[s])
        if [m["id"] for m in meta] != [r["id"] for r in r2t[s]]:
            raise SystemExit(f"typed/{s}: question ids differ from stored predictions")
        report[f"typed/{s}"] = dict(states=len(typed[s]), questions=len(meta))
        print("typed", s, len(typed[s]), len(meta), "ok", flush=True)
    jdump(typed, os.path.join(DATA, "typed_samples.json.gz"))
    jdump(report, os.path.join(DATA, "build_report.json"))
    print("all samples verified", flush=True)


if __name__ == "__main__":
    main()
