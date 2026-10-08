"""LOGIN NODE ONLY (needs internet). Resolves the pinned short revisions to full commit SHAs,
downloads checkpoints and dataset splits at those commits, and writes each split to parquet in
data/raw/ (row order preserved, so the item ids '<key>:<split>:<row>' keep their meaning).
Usage: python r8/stage_download.py [models|data|all]"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, jdump
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

MODELS = {"julia": ("SupersonicLabs/Julia-1", "a85b1273"),
          "qwen": ("Qwen/Qwen2.5-7B-Instruct", "a09a3545")}
# key: (repo, short rev, config, label column, train/test split names)
DATASETS = {
    "emotion": ("dair-ai/emotion", "cab853a1", "split", "label"),
    "agnews": ("fancyzhx/ag_news", "eb185aad", None, "label"),
    "banking77": ("mteb/banking77", "18072d26", None, "label"),
    "yelp": ("fancyzhx/yelp_polarity", "bbf1c97a", None, "label"),
    "imdb": ("stanfordnlp/imdb", "e6281661", "plain_text", "label"),
    "yahoo": ("community-datasets/yahoo_answers_topics", "6652a1e7", None, "topic"),
    "amazon": ("fancyzhx/amazon_polarity", "9d9c45c1", None, "label"),
    "rotten": ("cornell-movie-review-data/rotten_tomatoes", "aa13bc28", None, "label"),
    "massive": ("mteb/amazon_massive_intent", "940fd47a", "en", "label"),
    "wikitoxic": ("csv:OxAISH-AL-LLM/wiki_toxic", "872656a1", None, "label"),
}
FIXED = {"btzsc": ("btzsc/btzsc", "fef2a2ac62b69c58670047dddf045c53d7c3cb5e"),
         "typed": ("LocalLLaMA/typed-decisions", "c76749ec58bd8c3d2ea706b31c333a9059c38f90")}
BTZSC_NAMES = ["agnews", "emotiondair", "banking77", "yelpreviews", "imdb", "yahootopics", "amazonpolarity",
               "rottentomatoes", "massive", "wikitoxic_toxicaggregated"]
api = HfApi()


def resolve(repo, short, repo_type):
    if len(short) == 40:
        return short
    for c in api.list_repo_commits(repo, repo_type=repo_type):
        if c.commit_id.startswith(short):
            return c.commit_id
    raise SystemExit(f"revision {short} not found in {repo}")


def main(what):
    os.makedirs(os.path.join(DATA, "raw"), exist_ok=True)
    revp = os.path.join(DATA, "revisions_resolved.json")
    rev = json.load(open(revp)) if os.path.exists(revp) else {}
    if what in ("models", "all"):
        for k, (repo, short) in MODELS.items():
            full = resolve(repo, short, "model"); rev[k] = {"repo": repo, "rev": full}
            if k == "julia":
                path = snapshot_download(repo, revision=full, local_dir=os.environ["JULIA_CKPT"])
            else:
                path = snapshot_download(repo, revision=full)
            rev[k]["path"] = path; print(k, repo, full, path, flush=True)
            jdump(rev, revp)
    if what in ("data", "all"):
        from datasets import load_dataset
        for k, (repo, full) in FIXED.items():
            rev[k] = {"repo": repo, "rev": full}
        for n in BTZSC_NAMES:
            hf_hub_download("btzsc/btzsc", f"{n}/test-00000-of-00001.parquet", repo_type="dataset", revision=FIXED["btzsc"][1])
        for s in ["train", "test"]:
            p = hf_hub_download(FIXED["typed"][0], f"all/{s}-00000-of-00001.parquet", repo_type="dataset",
                                revision=FIXED["typed"][1])
            rev.setdefault("typed_files", {})[s] = p
        only = [k for k in os.environ.get("R8_ONLY", "").split(",") if k]  # dry runs only
        for key, (repo, short, cfg, lcol) in DATASETS.items():
            if only and key not in only:
                continue
            r = repo[4:] if repo.startswith("csv:") else repo
            full = resolve(r, short, "dataset")
            meta = {"repo": r, "rev": full, "config": cfg, "label_col": lcol, "splits": {}}
            for split in ["train", "test"]:
                out = os.path.join(DATA, "raw", f"{key}_{split}.parquet")
                if repo.startswith("csv:"):
                    ds = load_dataset("csv", data_files={split: f"hf://datasets/{r}@{full}/{split}.csv"}, split=split)
                else:
                    ds = load_dataset(r, cfg, split=split, revision=full) if cfg else load_dataset(r, split=split, revision=full)
                feat = ds.features.get(lcol)
                meta["label_names"] = feat.names if hasattr(feat, "names") else None
                if not os.path.exists(out):
                    ds.to_parquet(out)
                meta["splits"][split] = {"rows": ds.num_rows, "file": out}
                print(key, split, r, full[:8], ds.num_rows, flush=True)
            rev[key] = meta
            jdump(rev, revp)
    jdump(rev, revp)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
