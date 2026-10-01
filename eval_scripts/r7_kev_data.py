"""Build Kev ground-truth membership files (run inside the kev repo with its venv).
members : clean v7 train records from the 10 public sources (300 per source)   -> trained for 2 epochs
heldout : clean v7 development + calibration records from the same sources      -> never trained
fresh   : new records built by kev.suite from the same pools, excluding v7 train+calibration states"""
import json, random, subprocess, sys, collections, shutil
from pathlib import Path
from huggingface_hub import hf_hub_download
KEV = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
SUITE = KEV / "evals/v7/decision-v7"
p = hf_hub_download("jaredpalmer/kev-suites", "v7/decision-v7/train.jsonl", repo_type="dataset")
shutil.copy(p, SUITE / "train.jsonl")
SRC = ["agnews", "amazon", "banking77", "boolq", "dbpedia14", "imdb", "mnli", "sst5", "trec", "yelp"]
load = lambda f: [json.loads(l) for l in open(SUITE / f)]
rng = random.Random(0)
tr = [r for r in load("train.jsonl") if r["_meta"].get("source") in SRC and r["_meta"].get("variant", "clean") == "clean"]
by = collections.defaultdict(list)
for r in tr:
    by[r["_meta"]["source"]].append(r)
members = [r for s in SRC for r in rng.sample(by[s], min(300, len(by[s])))]
held = [r for f in ["development.jsonl", "calibration.jsonl"] for r in load(f)
        if r["_meta"].get("source") in SRC and r["_meta"].get("variant", "clean") == "clean"]
for name, rs in [("members", members), ("heldout", held)]:
    with open(OUT / f"{name}.jsonl", "w") as f:
        for r in rs:
            f.write(json.dumps(r) + "\n")
    print(name, len(rs), collections.Counter(r["_meta"]["source"] for r in rs), flush=True)
try:
    tmp = OUT / "fresh_suite"
    cmd = ["uv", "run", "python", "-m", "kev.suite", "--out", str(tmp), "--sources", ",".join(SRC), "--train", "300",
           "--calibration", "1", "--development", "1", "--test", "1", "--seed", "777", "--exclude-states-from", str(SUITE)]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=KEV, check=True, timeout=3600)
    member_ids = {r["_meta"]["id"] for r in load("train.jsonl")} | {r["_meta"]["id"] for r in load("calibration.jsonl")}
    fresh = [json.loads(l) for l in open(tmp / "train.jsonl")]
    fresh = [r for r in fresh if r["_meta"].get("id") not in member_ids and r["_meta"].get("source") in SRC
             and r["_meta"].get("variant", "clean") == "clean"]
    with open(OUT / "fresh.jsonl", "w") as f:
        for r in fresh:
            f.write(json.dumps(r) + "\n")
    print("fresh", len(fresh), collections.Counter(r["_meta"]["source"] for r in fresh), flush=True)
except Exception as ex:
    print("fresh suite failed:", type(ex).__name__, str(ex)[:300], flush=True)
