"""Same-architecture positive control: continue training Julia-1 on a dataset it was not exposed to
(Yelp, Yahoo), with known exposure, then run the probes. Grid: seeds x n in {250,1000} x {S1,S2}; epochs 1,3."""
import copy, gzip, json, os, random, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r4_eval as E, run_julia as RJ
from r6_ctrl import groups_for

key = sys.argv[1]; seeds = [int(s) for s in sys.argv[2].split(",")]
base_engine = RJ.load_engine("cuda", batch_size=32)
base_state = copy.deepcopy(base_engine.model.state_dict())


def rows_for(items, labels, with_target=False):
    rows = []
    for it in items:
        o = E.subset_opts(it["id"], it["target"], len(labels))
        r = RJ.row(RJ.truncate(base_engine, it["text"])[0], [labels[c] for c in o])
        if with_target:
            r["target"] = o.index(it["target"])
        rows.append((r, o))
    return rows


def evaluate(items, labels):
    rs = rows_for(items, labels)
    lg = RJ.safe_logits(base_engine, [r for r, _ in rs])
    return [int(z is not None and o[int(np.argmax(z))] == it["target"]) for (r, o), z, it in zip(rs, lg, items)]


def run(seed, n, scenario, epochs=(1, 3), lr=1e-5, bs=16):
    torch.manual_seed(seed); random.seed(seed)
    m = base_engine.model; m.load_state_dict(base_state); base_engine.clear_cache()
    d = E.S[key]; labels = d["hyps"]; G = groups_for(d, n)
    seen = list(G["train_seen"]) + (list(G["test_seen"]) if scenario == "S2" else [])
    train_rows = [r for r, _ in rows_for(seen, labels, with_target=True)]
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    res = {}
    for ep in range(1, max(epochs) + 1):
        m.train(); random.shuffle(train_rows)
        for i in range(0, len(train_rows), bs):
            batch = base_engine.collate(train_rows[i:i + bs], include_targets=True)
            batch = {k: v.to("cuda") for k, v in batch.items()}
            y = batch.pop("labels")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                scores = m(**batch)
            loss = torch.nn.functional.cross_entropy(scores.float(), y)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        m.eval(); base_engine.clear_cache()
        if ep in epochs:
            res[ep] = {g: evaluate(items, labels) for g, items in G.items()}
            print(key, seed, n, scenario, ep, {g: round(float(np.mean(v)), 3) for g, v in res[ep].items()}, flush=True)
    return res


if __name__ == "__main__":
    m = base_engine.model; m.eval()
    OUT = {"base": {g: evaluate(items, E.S[key]["hyps"]) for g, items in groups_for(E.S[key], 1000).items()}}
    print(key, "base", {g: round(float(np.mean(v)), 3) for g, v in OUT["base"].items()}, flush=True)
    for seed in seeds:
        for n in [250, 1000]:
            for sc in ["S1", "S2"]:
                OUT[f"{seed}|{n}|{sc}"] = run(seed, n, sc)
                with gzip.open(f"r6_juliaft_{key}.json.gz", "wt") as f:
                    json.dump(OUT, f)
    print("done", key, flush=True)
