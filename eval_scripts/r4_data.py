"""Round 3 data: train/test samples for 10 datasets (+ texts to paraphrase for 5 of them)."""
import gzip, json, os, sys
from datasets import load_dataset
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2_data as R2
import run_julia as RJ

N = 2000
PARA = ["agnews", "emotion", "banking77", "yelp", "yahoo"]
NP = 1000
SPECS = dict(R2.SPECS)
SPECS.update({
    "amazon": ("fancyzhx/amazon_polarity", None, lambda r: r["content"], "label", "amazonpolarity", False),
    "rotten": ("cornell-movie-review-data/rotten_tomatoes", None, lambda r: r["text"], "label", "rottentomatoes", False),
    "massive": ("mteb/amazon_massive_intent", "en", lambda r: r["text"], "label", "massive", False),
    "wikitoxic": ("csv:OxAISH-AL-LLM/wiki_toxic", None, lambda r: r["comment_text"], "label", "wikitoxic_toxicaggregated", False),
})
ALIASES = dict(R2.ALIASES)
ALIASES.update({"amazon": {"0": "negative", "1": "positive"}, "rotten": {"0": "negative", "1": "positive"},
                "wikitoxic": {"0": "not_toxicaggregated", "1": "toxicaggregated"}})


def load_split(repo, cfg, split):
    if repo.startswith("csv:"):
        r = repo[4:]
        return load_dataset("csv", data_files={split: f"hf://datasets/{r}/{split}.csv"}, split=split), r
    return R2.load_split(repo, cfg, split)


def main():
    out = {"datasets": {}}
    for key, (repo, cfg, tfn, lcol, bname, show) in SPECS.items():
        hyps, names = R2.btzsc_hyp_names(bname)
        idx = {R2.norm(n): i for i, n in enumerate(names)}
        info = dict(hyps=hyps, names=names, showcased=show, splits={})
        for split in ["train", "test"]:
            ds, used = load_split(repo, cfg, split)
            feat = ds.features.get(lcol)
            lnames = feat.names if hasattr(feat, "names") else None
            items, miss = [], set()
            for i, r in enumerate(ds):
                lab = r[lcol]
                if "label_text" in r and r["label_text"]:
                    nm = r["label_text"]
                elif lnames:
                    nm = lnames[lab]
                else:
                    nm = str(lab)
                nm = ALIASES.get(key, {}).get(str(nm), nm)
                c = idx.get(R2.norm(nm))
                if c is None:
                    miss.add(nm); continue
                t = tfn(r)
                if not t or not str(t).strip():
                    continue
                items.append(dict(id=f"{key}:{split}:{i}", text=str(t), target=c))
            assert len(miss) <= 6, (key, split, sorted(miss)[:8])
            sel = RJ.balanced_indices([it["target"] for it in items], N, 20260928 + (split == "train"))
            info["splits"][split] = [items[j] for j in sel]
            info["source"] = used
            print(key, split, used, len(items), "->", len(sel), "unmapped:", sorted(miss), flush=True)
        out["datasets"][key] = info
    with gzip.open("r4_samples.json.gz", "wt") as f:
        json.dump(out, f)
    para = []
    for key in PARA:
        for split in ["train", "test"]:
            for it in out["datasets"][key]["splits"][split][:NP]:
                para.append(dict(id=it["id"], text=it["text"][:1200]))
    json.dump(para, open("r4_to_para.json", "w"))
    print("to paraphrase:", len(para), flush=True)


if __name__ == "__main__":
    main()
