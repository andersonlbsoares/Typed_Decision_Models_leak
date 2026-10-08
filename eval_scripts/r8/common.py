"""Shared helpers for the r8 cluster runs (paths, samples, option subsets, atomic writes)."""
import gzip, hashlib, json, os, random

R8 = os.environ.get("R8", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(R8, "data")
RES = os.path.join(R8, "results")
REF = os.path.join(R8, "ref")
os.makedirs(RES, exist_ok=True)


def jload(p):
    return json.load(gzip.open(p, "rt")) if p.endswith(".gz") else json.load(open(p))


def jdump(obj, p):
    """Atomic write (tmp + rename) so a killed job never leaves a truncated file."""
    tmp = p + ".tmp"
    if p.endswith(".gz"):
        with gzip.open(tmp, "wt") as f:
            json.dump(obj, f)
    else:
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=1)
    os.replace(tmp, p)


def subset_opts(item_id, target, k_all, k=20):
    """Identical to r4_eval.subset_opts: all options if <=20, else gold + 19 seeded distractors, shuffled."""
    if k_all <= 20:
        return list(range(k_all))
    rng = random.Random(int(hashlib.md5(item_id.encode()).hexdigest()[:8], 16))
    opts = [target] + rng.sample([c for c in range(k_all) if c != target], k - 1)
    rng.shuffle(opts)
    return opts


def load_cls():
    S = jload(os.path.join(DATA, "r4_samples.json.gz"))["datasets"]
    S["rotten"] = jload(os.path.join(DATA, "r6_rotten_samples.json.gz"))
    return S


def revisions():
    return jload(os.path.join(DATA, "revisions_resolved.json"))


def typed_rows(cases):
    """Identical to r2_julia_nli.typed_rows: one Julia row per typed-decisions question."""
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
