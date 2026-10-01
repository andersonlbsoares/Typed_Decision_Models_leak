"""Round 2 data: train vs test samples from the original HF datasets, BTZSC hypotheses, and
label-verbalization variants. Writes r2_samples.json.gz consumed by r2_julia_nli.py / r2_laya.py.

Probe logic: a model never trained on a dataset should score about the same on its train and test
splits; a model trained on the train split scores noticeably higher on train.
"""
import gzip, json, random, re, sys, os
from collections import defaultdict
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("JULIA_CKPT", "/nonexistent")
import run_julia as RJ

N = 2000        # per split for the train/test probe
NV = 1000       # test examples for the verbalization study
TYPED_REV = RJ.TYPED_REV


def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def btzsc_hyp_names(name):
    """Return (hypotheses, class names) in BTZSC order."""
    path = hf_hub_download("btzsc/btzsc", f"{name}/test-00000-of-00001.parquet", repo_type="dataset",
                           revision=RJ.BTZSC_REV)
    t = pq.read_table(path).to_pydict()
    texts = t["text"]
    k = next(i for i in range(1, len(texts)) if texts[i] != texts[0])
    hyps = [t["hypothesis"][i] for i in range(k)]
    names = [None] * k
    for s in range(len(texts) // k):
        for c in range(k):
            if t["labels"][s * k + c] == 1 and names[c] is None:
                names[c] = t["label_text"][s * k + c]
        if all(names):
            break
    return hyps, names


ALIASES = {"yelp": {"1": "negative", "2": "positive", "0": "negative"},
           "imdb": {"neg": "negative", "pos": "positive"}}

SPECS = {  # name: (hf repo, config, text fn, label column, btzsc name, showcased?)
    "agnews": ("fancyzhx/ag_news", None, lambda r: r["text"], "label", "agnews", True),
    "emotion": ("dair-ai/emotion", "split", lambda r: r["text"], "label", "emotiondair", True),
    "banking77": ("PolyAI/banking77", None, lambda r: r["text"], "label", "banking77", True),
    "yelp": ("fancyzhx/yelp_polarity", None, lambda r: r["text"], "label", "yelpreviews", False),
    "imdb": ("stanfordnlp/imdb", "plain_text", lambda r: r["text"], "label", "imdb", False),
    "yahoo": ("community-datasets/yahoo_answers_topics", None,
              lambda r: (r["question_title"] + " " + r["question_content"]).strip(), "topic", "yahootopics", False),
}

SYN = {  # V2: reworded label descriptions with the same meaning
    "agnews": ("This example news text is about {} news.",
               {"World": "international", "Sports": "athletics", "Business": "economic", "Sci/Tech": "technology and research"}),
    "emotion": ("This example tweet expresses the emotion: {}",
                {"sadness": "sorrow", "joy": "happiness", "love": "affection", "anger": "rage", "fear": "anxiety",
                 "surprise": "astonishment"}),
    "yelp": ("The sentiment in this example review is {}.", {"negative": "unfavorable", "positive": "favorable"}),
    "imdb": ("The sentiment in this example review is {}.", {"negative": "unfavorable", "positive": "favorable"}),
    "yahoo": ("This example question is about {}.",
              {"Society & Culture": "society and cultural topics", "Science & Mathematics": "science and math",
               "Health": "medicine and wellbeing", "Education & Reference": "schooling and reference material",
               "Computers & Internet": "computing and the web", "Sports": "athletics",
               "Business & Finance": "the economy and money", "Entertainment & Music": "movies, media and music",
               "Family & Relationships": "family life and relationships",
               "Politics & Government": "politics and the state"}),
}


def load_split(repo, config, split):
    for r in ([repo, "mteb/banking77", "legacy-datasets/banking77"] if repo == "PolyAI/banking77" else [repo]):
        try:
            return load_dataset(r, config, split=split) if config else load_dataset(r, split=split), r
        except Exception as ex:
            print("load fail", r, split, type(ex).__name__, str(ex)[:150], flush=True)
    raise RuntimeError(f"cannot load {repo}")


def main():
    out = {"datasets": {}, "typed": {}}
    for key, (repo, cfg, tfn, lcol, bname, show) in SPECS.items():
        hyps, names = btzsc_hyp_names(bname)
        idx = {norm(n): i for i, n in enumerate(names)}
        ds_info = dict(hyps=hyps, names=names, showcased=show, splits={})
        for split in ["train", "test"]:
            ds, used = load_split(repo, cfg, split)
            feat = ds.features.get(lcol)
            lnames = feat.names if hasattr(feat, "names") else None
            items, miss = [], set()
            for i, r in enumerate(ds):
                lab = r[lcol]
                nm = r["label_text"] if "label_text" in r and r["label_text"] else (lnames[lab] if lnames else str(lab))
                nm = ALIASES.get(key, {}).get(str(nm), nm)
                c = idx.get(norm(nm))
                if c is None:
                    miss.add(nm); continue
                items.append(dict(id=f"{key}:{split}:{i}", text=tfn(r), target=c))
            assert len(miss) <= 6, (key, split, list(miss)[:8])  # BTZSC banking77 keeps 72 of 77 intents
            sel = RJ.balanced_indices([it["target"] for it in items], N, 20260928 + (split == "train"))
            ds_info["splits"][split] = [items[j] for j in sel]
            ds_info["source"] = used
            print(key, split, used, len(items), "->", len(sel), "unmapped:", sorted(miss), flush=True)
        if key in SYN:
            tmpl, m = SYN[key]
            ds_info["v2"] = [tmpl.format(m[n]) for n in names]
        ds_info["v1"] = [n.replace("_", " ") for n in names]
        out["datasets"][key] = ds_info
    for split in ["train", "test"]:
        p = hf_hub_download("LocalLLaMA/typed-decisions", f"all/{split}-00000-of-00001.parquet",
                            repo_type="dataset", revision=TYPED_REV)
        out["typed"][split] = pq.read_table(p).to_pylist()
        print("typed", split, len(out["typed"][split]), flush=True)
    with gzip.open("r2_samples.json.gz", "wt") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
