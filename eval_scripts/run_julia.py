"""Initial independent evaluation of Julia-1 (SupersonicLabs).

Experiments
  E1 typed-decisions (pinned revision), CUDA BF16, full logits -> accuracy + calibration
  E2 Jev pilot-v1 manifest (300 BTZSC examples, sha-checked) -> acc/F1/Brier/NLL/ECE
  E3 BTZSC test sets (<=3000 per dataset, seeded) incl. >20-label sets via tournament
  E4 option-order sensitivity: all cyclic shifts of the label list (4 datasets x 1000)
  E5 option-count scaling on banking77/massive: k in {2,4,8,12,16,20}, random vs hard distractors
  E6 latency, batch size 1, GPU and CPU
Outputs: out/*.jsonl.gz (per-example logits) + out/summary_julia.json
"""
import gzip, hashlib, json, os, random, sys, time, traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

CKPT = Path(os.environ["JULIA_CKPT"])
OUT = Path("out"); OUT.mkdir(exist_ok=True)
BTZSC_REV = "fef2a2ac62b69c58670047dddf045c53d7c3cb5e"
TYPED_REV = "c76749ec58bd8c3d2ea706b31c333a9059c38f90"
TYPED_SHA = "4f294f218ea1da27f3efef936359389c62ea4d3973a41457732990f1d31b647c"
WEIGHTS_SHA = "df853bf7fe424420011f3d0c47a05d7341aa9eefa7fb9f203ea4aada4ad95b72"
MANIFEST_SHA = "ec064c52b149de458344cd4b4a44c158460f30b3bbb7fe8b2e7ec72d0abf3ba5"
QUESTION = "Which single label best describes the input text?"
CAP = 3000
SMALL_K = ["agnews", "amazonpolarity", "appreviews", "biasframes_intent", "biasframes_offensive",
           "biasframes_sex", "emotiondair", "financialphrasebank", "imdb", "rottentomatoes",
           "trueteacher", "wikitoxic_insult", "wikitoxic_obscene", "wikitoxic_threat",
           "wikitoxic_toxicaggregated", "yahootopics", "yelpreviews"]
LARGE_K = ["capsotu", "empathetic", "massive", "manifesto", "banking77"]
SUMMARY = {"started": time.strftime("%Y-%m-%d %H:%M:%S")}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def save_summary():
    (OUT / "summary_julia.json").write_text(json.dumps(SUMMARY, indent=2, default=str))


def dump(name, records):
    with gzip.open(OUT / f"{name}.jsonl.gz", "wt") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=float) + "\n")


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------------------------------------------------------- data
def load_btzsc(name):
    path = hf_hub_download("btzsc/btzsc", f"{name}/test-00000-of-00001.parquet",
                           repo_type="dataset", revision=BTZSC_REV)
    t = pq.read_table(path).to_pydict()
    texts, binary = [str(x) for x in t["text"]], [int(x) for x in t["labels"]]
    k = next(i for i in range(1, len(texts)) if texts[i] != texts[0])
    labels = [str(t["hypothesis"][i]) for i in range(k)]
    items = []
    for s in range(len(texts) // k):
        v = binary[s * k:(s + 1) * k]
        if sum(v) == 1:
            items.append(dict(id=f"{name}:{s}", text=texts[s * k], target=v.index(1)))
    return labels, items


def balanced_indices(targets, limit, seed):
    by = defaultdict(list)
    for i, t in enumerate(targets):
        by[t].append(i)
    rng = random.Random(seed)
    for v in by.values():
        rng.shuffle(v)
    chosen = []
    while len(chosen) < min(limit, len(targets)):
        prog = False
        for c in sorted(by):
            if by[c] and len(chosen) < limit:
                chosen.append(by[c].pop()); prog = True
        if not prog:
            break
    return sorted(chosen)


# ---------------------------------------------------------------- model helpers
def load_engine(device, batch_size=64):
    sys.path.insert(0, str(CKPT))
    from julia.router.engine import FastEngine
    return FastEngine(str(CKPT), device=device, transformer_backend="torch", strict_encoding=True,
                      max_length=1024, head_length=512, batch_size=batch_size, marker_only_head=False)


def truncate(engine, text, max_tokens=700):
    ids = engine.tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text, False
    return engine.tokenizer.decode(ids[:max_tokens]), True


def safe_logits(engine, rows, chunk=512):
    """Returns list of logits (or None when the request cannot be encoded)."""
    out = []
    for i in range(0, len(rows), chunk):
        part = rows[i:i + chunk]
        try:
            out.extend(engine.logits(part))
        except Exception:
            for r in part:
                try:
                    out.extend(engine.logits([r]))
                except Exception:
                    out.append(None)
    return out


def row(text, options, qtype="choice", question=QUESTION):
    return dict(state={"text": text}, question=question, options=list(options), type=qtype)


def tournament(engine, text, labels, width=20, survivors=2):
    """Group-wise narrowing like the package Router; returns final candidates + logits + trace."""
    cand = list(range(len(labels)))
    rounds, calls = 0, 0
    while len(cand) > width:
        rounds += 1
        g = int(np.ceil(len(cand) / width))
        groups = [cand[j::g] for j in range(g)]  # interleaved, balanced sizes
        rows = [row(text, [labels[c] for c in grp]) for grp in groups]
        lg = safe_logits(engine, rows)
        calls += len(rows)
        nxt = []
        for grp, z in zip(groups, lg):
            if z is None:
                continue
            order = np.argsort(z)[::-1][:survivors]
            nxt.extend(grp[o] for o in order)
        cand = sorted(nxt)
        if not cand:
            return None
    z = safe_logits(engine, [row(text, [labels[c] for c in cand])])[0]
    calls += 1
    if z is None:
        return None
    return dict(cand=cand, logits=z, rounds=rounds + 1, calls=calls)


def acc_of(records):
    ok = [r for r in records if r.get("pred") is not None]
    return dict(n=len(records), answered=len(ok),
                acc=sum(r["pred"] == r["target"] for r in ok) / max(1, len(records)))


# ---------------------------------------------------------------- experiments
def e1_typed(engine):
    path = hf_hub_download("LocalLLaMA/typed-decisions", "all/test-00000-of-00001.parquet",
                           repo_type="dataset", revision=TYPED_REV)
    SUMMARY["typed_sha_ok"] = sha_file(path) == TYPED_SHA
    Path("typed.parquet").write_bytes(Path(path).read_bytes())
    cases = pq.read_table(path).to_pylist()
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
            rows.append(dict(state=state, question=q["instructions"], type=kind,
                             options=[crit[k] for k in keys]))
            meta.append(dict(id=case["id"] + ":" + qid, type=kind, keys=keys,
                             gold=str(gold[qid]["label"])))
    lg = safe_logits(engine, rows, chunk=256)
    recs = []
    for m, z in zip(meta, lg):
        pred = None if z is None else m["keys"][int(np.argmax(z))]
        recs.append(dict(**m, logits=z, pred=pred))
    dump("E1_typed_cuda", recs)
    by = defaultdict(lambda: [0, 0])
    for r in recs:
        by[r["type"]][0] += 1; by[r["type"]][1] += int(r["pred"] == r["gold"])
    SUMMARY["E1_typed_cuda"] = {k: dict(n=v[0], correct=v[1], acc=v[1] / v[0]) for k, v in by.items()}
    SUMMARY["E1_typed_cuda"]["all"] = dict(n=len(recs), correct=sum(v[1] for v in by.values()))
    log("E1", SUMMARY["E1_typed_cuda"])


def e2_pilot(engine):
    tasks = {"agnews": "topic", "emotiondair": "emotion", "banking77": "intent"}
    examples = []
    for off, (name, task) in enumerate(tasks.items()):
        labels, items = load_btzsc(name)
        for p in balanced_indices([it["target"] for it in items], 100, 20260917 + off):
            it = items[p]
            examples.append({"dataset": name, "task": task, "example_id": it["id"], "text": it["text"],
                             "text_sha256": hashlib.sha256(it["text"].encode()).hexdigest(),
                             "labels": labels, "target_index": it["target"]})
    manifest = "".join(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n" for e in examples)
    SUMMARY["E2_manifest_sha_ok"] = hashlib.sha256(manifest.encode()).hexdigest() == MANIFEST_SHA
    recs = []
    direct = [e for e in examples if len(e["labels"]) <= 20]
    lg = safe_logits(engine, [row(e["text"], e["labels"]) for e in direct])
    for e, z in zip(direct, lg):
        recs.append(dict(dataset=e["dataset"], id=e["example_id"], target=e["target_index"], logits=z,
                         cand=list(range(len(e["labels"]))),
                         pred=None if z is None else int(np.argmax(z))))
    for e in [e for e in examples if len(e["labels"]) > 20]:
        t = tournament(engine, e["text"], e["labels"])
        pred = None if t is None else t["cand"][int(np.argmax(t["logits"]))]
        recs.append(dict(dataset=e["dataset"], id=e["example_id"], target=e["target_index"],
                         logits=None if t is None else t["logits"], cand=None if t is None else t["cand"],
                         rounds=None if t is None else t["rounds"], pred=pred))
    try:
        from julia.router import Router
        rt = Router(engine, survivors=2)
        bank = [e for e in examples if e["dataset"] == "banking77"]
        res = rt.route_many([row(e["text"], e["labels"]) for e in bank])
        SUMMARY["E2_banking_package_router"] = dict(
            correct=int(sum(r.index == e["target_index"] for r, e in zip(res, bank))), n=len(bank))
        dump("E2_banking_router", [dict(id=e["example_id"], target=e["target_index"], pred=int(r.index),
                                        cand=list(r.candidates), probs=list(r.probabilities))
                                   for r, e in zip(res, bank)])
    except Exception as ex:
        SUMMARY["E2_banking_package_router"] = f"unavailable: {type(ex).__name__}: {ex}"[:400]
    dump("E2_pilot", recs)
    SUMMARY["E2_pilot"] = {d: acc_of([r for r in recs if r["dataset"] == d]) for d in tasks}
    log("E2", SUMMARY["E2_pilot"], SUMMARY["E2_banking_package_router"])


def e3_btzsc(engine):
    SUMMARY["E3"] = {}
    for name in SMALL_K + LARGE_K:
        try:
            t0 = time.time()
            labels, items = load_btzsc(name)
            rng = random.Random(7)
            if len(items) > CAP:
                items = sorted(rng.sample(items, CAP), key=lambda x: int(x["id"].split(":")[1]))
            ntrunc, texts = 0, []
            for it in items:
                tx, tr = truncate(engine, it["text"]); texts.append(tx); ntrunc += tr
            recs = []
            if len(labels) <= 20:
                lg = safe_logits(engine, [row(tx, labels) for tx in texts])
                for it, z in zip(items, lg):
                    recs.append(dict(id=it["id"], target=it["target"], logits=z,
                                     pred=None if z is None else int(np.argmax(z))))
            else:
                for it, tx in zip(items, texts):
                    t = tournament(engine, tx, labels)
                    recs.append(dict(id=it["id"], target=it["target"],
                                     logits=None if t is None else t["logits"],
                                     cand=None if t is None else t["cand"],
                                     gold_survived=None if t is None else it["target"] in t["cand"],
                                     calls=None if t is None else t["calls"],
                                     pred=None if t is None else t["cand"][int(np.argmax(t["logits"]))]))
            dump(f"E3_{name}", recs)
            SUMMARY["E3"][name] = dict(k=len(labels), truncated=ntrunc, secs=round(time.time() - t0, 1),
                                       labels=labels, **acc_of(recs))
            log("E3", name, {k: v for k, v in SUMMARY["E3"][name].items() if k != "labels"})
        except Exception as ex:
            SUMMARY["E3"][name] = f"error: {ex}"; traceback.print_exc()
        save_summary()


def e4_order(engine):
    SUMMARY["E4"] = {}
    for name in ["agnews", "emotiondair", "financialphrasebank", "yahootopics"]:
        labels, items = load_btzsc(name)
        k = len(labels)
        sel = [items[i] for i in balanced_indices([it["target"] for it in items], 1000, 11)]
        texts = [truncate(engine, it["text"])[0] for it in sel]
        recs = []
        for shift in range(k):
            perm = [(j + shift) % k for j in range(k)]  # position p shows label perm[p]
            lg = safe_logits(engine, [row(tx, [labels[p] for p in perm]) for tx in texts])
            for it, z in zip(sel, lg):
                zc = None if z is None else [z[perm.index(c)] for c in range(k)]
                recs.append(dict(id=it["id"], target=it["target"], shift=shift, logits_canon=zc,
                                 chosen_pos=None if z is None else int(np.argmax(z)),
                                 gold_pos=perm.index(it["target"])))
        dump(f"E4_{name}", recs)
        accs = [float(np.mean([r["logits_canon"] is not None and int(np.argmax(r["logits_canon"])) == r["target"]
                               for r in recs if r["shift"] == s])) for s in range(k)]
        SUMMARY["E4"][name] = dict(k=k, acc_by_shift=[round(a, 4) for a in accs])
        log("E4", name, SUMMARY["E4"][name]); save_summary()


def e5_cardinality(engine):
    from sklearn.feature_extraction.text import TfidfVectorizer
    SUMMARY["E5"] = {}
    for name in ["banking77", "massive"]:
        labels, items = load_btzsc(name)
        sel = [items[i] for i in balanced_indices([it["target"] for it in items], 1000, 13)]
        tf = TfidfVectorizer().fit_transform(labels)
        sim = (tf @ tf.T).toarray(); np.fill_diagonal(sim, -1)
        recs = []
        for mode in ["random", "hard"]:
            for k in [2, 4, 8, 12, 16, 20]:
                rng = random.Random(1000 * k + (mode == "hard"))
                rows, meta = [], []
                for it in sel:
                    g = it["target"]
                    if mode == "random":
                        d = rng.sample([c for c in range(len(labels)) if c != g], k - 1)
                    else:
                        d = list(np.argsort(sim[g])[::-1][:k - 1])
                    opts = [g] + [int(x) for x in d]; rng.shuffle(opts)
                    rows.append(row(it["text"], [labels[o] for o in opts])); meta.append((it, opts))
                lg = safe_logits(engine, rows)
                for (it, opts), z in zip(meta, lg):
                    recs.append(dict(id=it["id"], target=it["target"], mode=mode, k=k, opts=opts, logits=z,
                                     pred=None if z is None else opts[int(np.argmax(z))]))
                a = acc_of([r for r in recs if r["mode"] == mode and r["k"] == k])["acc"]
                SUMMARY["E5"].setdefault(name, {}).setdefault(mode, {})[k] = round(a, 4)
        dump(f"E5_{name}", recs)
        log("E5", name, SUMMARY["E5"][name]); save_summary()


def e6_latency():
    import torch
    labels, items = load_btzsc("agnews")
    sel = items[:200]
    res = {}
    for dev, threads in [("cuda", None), ("cpu", 4), ("cpu", 12)]:
        if threads:
            torch.set_num_threads(threads)
        eng = load_engine(dev, batch_size=1)
        rows = [row(it["text"], labels) for it in sel]
        for r in rows[:10]:
            eng.logits([r])
        ts = []
        for r in rows:
            t = time.perf_counter(); eng.logits([r])
            if dev == "cuda":
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t) * 1000)
        res[f"{dev}{'' if threads is None else '_t' + str(threads)}"] = dict(
            median_ms=float(np.median(ts)), p90_ms=float(np.percentile(ts, 90)), n=len(ts))
        del eng
    SUMMARY["E6_latency_agnews_bs1"] = res
    log("E6", res)


if __name__ == "__main__":
    import torch, transformers
    SUMMARY["env"] = dict(torch=torch.__version__, transformers=transformers.__version__,
                          cuda=torch.cuda.is_available(),
                          gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    SUMMARY["weights_sha_ok"] = sha_file(CKPT / "model.safetensors") == WEIGHTS_SHA
    log(SUMMARY["env"], "weights ok:", SUMMARY["weights_sha_ok"])
    engine = load_engine("cuda")
    stages = [("E1", lambda: e1_typed(engine)), ("E2", lambda: e2_pilot(engine)),
              ("E4", lambda: e4_order(engine)), ("E5", lambda: e5_cardinality(engine)),
              ("E3", lambda: e3_btzsc(engine)), ("E6", e6_latency)]
    only = os.environ.get("ONLY")
    for name, fn in stages:
        if only and name not in only.split(","):
            continue
        t0 = time.time()
        try:
            fn()
        except Exception as ex:
            SUMMARY[name + "_error"] = f"{type(ex).__name__}: {ex}"; traceback.print_exc()
        SUMMARY.setdefault("stage_secs", {})[name] = round(time.time() - t0, 1)
        save_summary()
    SUMMARY["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_summary()
    log("done")
