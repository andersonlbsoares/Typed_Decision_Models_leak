"""E1: power of probe A as a function of the exposed FRACTION of the audited training sample.
Fixed fine-tuning budget of N=2000 items per run: a fraction rho of the 2,000 audited training items
plus (1-rho)*N filler items drawn from the same training split outside the audit sample (class-balanced,
build_samples.py). Task learning is therefore held fixed across rho; only membership in the audited
sample varies. rho=0 is the same-budget control (task fine-tuning without membership).
Exposed sets are nested across rho within a seed. Evaluated after epochs 1 and 3 on the 2,000 audited
training and 2,000 test items (text[:1200], as in r6_julia_ft). Resumable: completed runs are skipped.
Usage: python r8/e1_julia_dilution.py emotion [--seeds 0,1,2,3,4] [--rhos 0,0.05,0.1,0.25,0.5,1]"""
import argparse, copy, os, random, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import julia_lib as J
from common import DATA, RES, jload, jdump, load_cls

ap = argparse.ArgumentParser()
ap.add_argument("key"); ap.add_argument("--seeds", default="0,1,2,3,4")
ap.add_argument("--rhos", default="0,0.05,0.1,0.25,0.5,1"); ap.add_argument("--epochs", default="1,3")
ap.add_argument("--n", type=int, default=2000); ap.add_argument("--max-runs", type=int, default=0)
a = ap.parse_args()
seeds = [int(s) for s in a.seeds.split(",")]; rhos = [float(r) for r in a.rhos.split(",")]
epochs = [int(e) for e in a.epochs.split(",")]; N = a.n; CUT = 1200

d = load_cls()[a.key]; labels = d["hyps"]
tr, te = d["splits"]["train"][:N], d["splits"]["test"][:N]
filler = jload(os.path.join(DATA, "e1_filler.json.gz"))[a.key]
outp = os.path.join(RES, f"r8_e1_{a.key}.json.gz")
OUT = jload(outp) if os.path.exists(outp) else {}
eng = J.engine("cuda", 32)
base_state = copy.deepcopy(eng.model.state_dict())
OUT.setdefault("meta", dict(key=a.key, N=N, cut=CUT, epochs=epochs, lr=1e-5, bs=16, engine=J.engine_info(eng),
                            train_ids=[x["id"] for x in tr], test_ids=[x["id"] for x in te],
                            train_gold=[x["target"] for x in tr], test_gold=[x["target"] for x in te]))


def preds(items):
    return [r[2] if r[2] is not None else -1 for r in J.eval_cls(eng, items, labels, cut=CUT)]


def acc(p, items):
    return round(sum(x == it["target"] for x, it in zip(p, items)) / len(items), 4)


if "base" not in OUT:
    OUT["base"] = {"train": preds(tr), "test": preds(te)}; jdump(OUT, outp)
print(a.key, "base", acc(OUT["base"]["train"], tr), acc(OUT["base"]["test"], te), flush=True)
done = 0
for seed in seeds:
    perm = random.Random(1000 + seed).sample(range(N), N)
    forder = random.Random(2000 + seed).sample(range(len(filler)), len(filler))
    for rho in rhos:
        rk = f"{seed}|{rho:g}"
        if rk in OUT and all(str(e) in OUT[rk] for e in epochs):
            continue
        if a.max_runs and done >= a.max_runs:
            break
        t0 = time.time(); m = round(rho * N); exposed = sorted(perm[:m])
        ft_items = [tr[i] for i in exposed] + [filler[j] for j in forder[:N - m]]
        torch.manual_seed(seed); random.seed(seed)
        eng.model.load_state_dict(base_state); eng.clear_cache()
        rows, _ = J.cls_rows(eng, ft_items, labels, cut=CUT, with_target=True)
        opt = J.make_opt(eng); res = {"exposed": exposed, "filler_ids": [filler[j]["id"] for j in forder[:N - m]]}
        for ep in range(1, max(epochs) + 1):
            J.train_epoch(eng, opt, rows)
            if ep in epochs:
                res[str(ep)] = {"train": preds(tr), "test": preds(te)}
                print(a.key, rk, "ep", ep, "train", acc(res[str(ep)]["train"], tr), "test", acc(res[str(ep)]["test"], te),
                      f"{time.time() - t0:.0f}s", flush=True)
        OUT[rk] = res; jdump(OUT, outp); done += 1
print("done", a.key, flush=True)
