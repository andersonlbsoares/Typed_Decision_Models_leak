"""Microsoft-Decision-1 audit collection (same protocol as the Jev collection; prereg_decision1.md).

  python jev_audit.py build             # writes jev_requests.jsonl.gz (no network, no cost)
  python jev_audit.py pilot             # 500 requests x 2 sends (prereg section 3)
  python jev_audit.py run [tasks...]    # full collection, resumable; default = all tasks

Every distinct request is identified by the SHA-256 of its canonical JSON. Identical requests
(e.g. a probe-C original that equals the probe-A text) are sent once and shared. Raw responses are
appended to jev_raw.jsonl (one line per send) and never rewritten, so a crash loses at most the
requests that were in flight. Spending is capped by jev_client (ledger cap_usd, US$ 3).
"""
import gzip, hashlib, json, os, random, sys, threading, time
HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.dirname(HERE), "r8")   # eval_scripts/r8 (common.py, data/ built by build_samples.py)
sys.path.insert(0, os.path.join(PKG, "r8")); sys.path.insert(0, HERE)
os.environ.setdefault("R8", PKG)
from common import jload, subset_opts   # identical option subsets to Julia-1 / Qwen / Laya

MODEL = "microsoft/microsoft-decision-1"
QUESTION = "Which single label best describes the input text?"
REQ = os.path.join(HERE, "dec_requests.jsonl.gz")
RAW = os.path.join(HERE, "dec_raw.jsonl")
DATA = os.path.join(PKG, "data")
PARA_DS = ["agnews", "emotion", "banking77", "yelp", "yahoo"]


def canon(p):
    return json.dumps(p, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def h(p):
    return hashlib.sha256(canon(p).encode()).hexdigest()


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("SupersonicLabs/Julia-1", subfolder="tokenizer", revision="a85b127321d580d65176c89ced8273f305745d85")


def truncate(tok, text, max_tokens=700):
    """Same rule as run_julia.truncate: first 700 Julia-1 (mmBERT) tokens, no special tokens."""
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return (text, False) if len(ids) <= max_tokens else (tok.decode(ids[:max_tokens]), True)


def cls_payload(text, hyps, opts):
    return {"model": MODEL, "state": {"text": text},
            "questions": {"label": {"type": "choice", "instructions": QUESTION,
                                    "criteria": {str(j): hyps[o] for j, o in enumerate(opts)}}}}


def build():
    tok = tokenizer()
    S = jload(os.path.join(DATA, "r4_samples.json.gz"))["datasets"]
    S["rotten"] = jload(os.path.join(DATA, "r6_rotten_samples.json.gz"))
    para = json.load(open(os.path.join(HERE, "..", "..", "predictions", "r4_para.json")))                 # Qwen2.5-7B
    phi = json.load(open(os.path.join(HERE, "..", "..", "predictions", "r7_para_second.json")))["items"]  # Phi-3.5-mini
    rows, ntr = [], {}

    def add(task, iid, payload, **meta):
        rows.append(dict(task=task, id=iid, hash=h(payload), payload=payload, **meta))

    for ds, d in S.items():
        hyps = d["hyps"]
        for split in ["train", "test"]:
            for it in d["splits"][split]:
                opts = subset_opts(it["id"], it["target"], len(hyps))
                t, tr = truncate(tok, it["text"]); ntr[ds] = ntr.get(ds, 0) + tr
                add(f"A:{ds}:{split}", it["id"], cls_payload(t, hyps, opts), gold=it["target"], opts=opts)
                if ds in PARA_DS and it["id"] in para:          # probe C, Qwen paraphrases
                    for kind in ["orig", "para"]:
                        tt = truncate(tok, para[it["id"]][kind])[0]
                        add(f"C:{ds}:{split}:{kind}_qwen", it["id"], cls_payload(tt, hyps, opts),
                            gold=it["target"], opts=opts)
                    assert phi[it["id"]]["orig"] == para[it["id"]]["orig"]
                    add(f"C:{ds}:{split}:para_phi", it["id"],
                        cls_payload(truncate(tok, phi[it["id"]]["para"])[0], hyps, opts), gold=it["target"], opts=opts)
    V = jload(os.path.join(DATA, "emotion_validation.json.gz"))
    for it in V["items"]:
        opts = subset_opts(it["id"], it["target"], len(V["hyps"]))
        add("B:emotion:validation", it["id"], cls_payload(truncate(tok, it["text"])[0], V["hyps"], opts),
            gold=it["target"], opts=opts)
    typed = jload(os.path.join(DATA, "typed_samples.json.gz"))
    for split in ["train", "test"]:
        for case in typed[split]:
            state = json.loads(case["state"]); gold = json.loads(case["gold"])
            wf = case.get("workflow") or case.get("config") or case["id"].split("-")[0]
            for qid, q in json.loads(case["questions"]).items():
                add(f"A:typed:{split}", case["id"] + ":" + qid,
                    {"model": MODEL, "state": state, "questions": {qid: q}},
                    gold=str(gold[qid]["label"]), qtype=q["type"], qid=qid, workflow=wf)
    with gzip.open(REQ + ".tmp", "wt") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(REQ + ".tmp", REQ)
    return rows, ntr


def load_requests():
    return [json.loads(l) for l in gzip.open(REQ, "rt")]


def done_hashes(rep=1):
    """Hashes with a successful (HTTP 200) response for send number `rep`."""
    out = set()
    if os.path.exists(RAW):
        for l in open(RAW):
            try:
                r = json.loads(l)
            except ValueError:
                continue                                   # a partially written last line
            if r.get("status") == 200 and r.get("rep", 1) == rep:
                out.add(r["hash"])
    return out


_wlock = threading.Lock()


def send(unique, rep, workers):
    """unique: list of (hash, payload). Appends every response to RAW."""
    import dec_client as J

    def on_result(hs, res):
        st, body, cost, lat = res
        with _wlock, open(RAW, "a") as f:
            f.write(json.dumps(dict(hash=hs, rep=rep, status=st, body=body, cost=cost, lat=round(lat, 3),
                                    t=round(time.time())), ensure_ascii=False) + "\n")
    return J.run(unique, workers=workers, on_result=on_result)


def unique_todo(rows, rep=1):
    done, seen, todo = done_hashes(rep), set(), []
    for r in rows:
        if r["hash"] not in done and r["hash"] not in seen:
            seen.add(r["hash"]); todo.append((r["hash"], r["payload"]))
    return todo


def pilot_rows(rows):
    rng = random.Random(20261009)
    tr = [r for r in rows if r["task"] == "A:typed:train"]; te = [r for r in rows if r["task"] == "A:typed:test"]
    va = [r for r in rows if r["task"] == "B:emotion:validation"]
    return rng.sample(tr, 150) + rng.sample(te, 150) + rng.sample(va, 200)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        rows, ntr = build()
        from collections import Counter
        print(json.dumps(dict(rows=len(rows), unique=len({r["hash"] for r in rows}),
                              by_task=Counter(r["task"] for r in rows), truncated=ntr), indent=0))
    elif cmd == "pilot":
        P = pilot_rows(load_requests())
        for rep in (1, 2):
            res, stop = send(unique_todo(P, rep), rep, workers=8)
            print("pilot rep", rep, len(res), "stop:", stop, flush=True)
            if stop:
                break
    elif cmd == "run":
        rows = load_requests()
        want = sys.argv[2:]
        if want:
            rows = [r for r in rows if any(r["task"].startswith(w) for w in want)]
        todo = unique_todo(rows)
        print("to send:", len(todo), flush=True)
        res, stop = send(todo, 1, workers=int(os.environ.get("JEV_WORKERS", "16")))
        print("sent", len(res), "stop:", stop, flush=True)
