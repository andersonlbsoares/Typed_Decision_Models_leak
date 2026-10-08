"""E4: Julia-1 decisions without batch/precision effects, on typed-decisions (full splits) and the
DAIR Emotion and AG News audit samples (full text, 700-token truncation, as in the audit).
Modes: gpu_bs64 (re-run of the audit setting; checks the environment against the stored predictions),
gpu_bs1, gpu_fp32_bs1 (weights fp32 + autocast disabled), cpu_fp32_bs1 (official reproduction setting).
Usage: python r8/e4_julia_det.py --modes gpu_bs64,gpu_bs1,gpu_fp32_bs1 --tag gpu
       python r8/e4_julia_det.py --modes cpu_fp32_bs1 --threads 12 --tag cpu"""
import argparse, contextlib, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import julia_lib as J
from common import DATA, RES, jload, jdump, load_cls

ap = argparse.ArgumentParser()
ap.add_argument("--modes", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--threads", type=int, default=12); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
S = load_cls(); typed = jload(os.path.join(DATA, "typed_samples.json.gz"))
L = (lambda x: x[:a.limit]) if a.limit else (lambda x: x)
outp = os.path.join(RES, f"r8_e4_{a.tag}.json.gz")
OUT = jload(outp) if os.path.exists(outp) else {}
for mode in a.modes.split(","):
    dev = "cpu" if mode.startswith("cpu") else "cuda"; bs = 64 if mode.endswith("bs64") else 1
    if dev == "cpu":
        torch.set_num_threads(a.threads)
    eng = J.engine(dev, bs); ctx = contextlib.nullcontext
    if "fp32" in mode:
        eng.model.float(); eng.clear_cache()
        if dev == "cuda":
            ctx = J.no_autocast
    R = OUT.setdefault(mode, {}); R["engine"] = J.engine_info(eng); R.setdefault("seconds", {})
    with ctx(), torch.inference_mode():
        for task in ["typed", "emotion", "agnews"]:
            for split in ["train", "test"]:
                k = f"{task}/{split}"
                if k in R:
                    continue
                t0 = time.time()
                if task == "typed":
                    R[k] = J.eval_typed(eng, L(typed[split]))
                else:
                    d = S[task]; R[k] = J.eval_cls(eng, L(d["splits"][split]), d["hyps"])
                R["seconds"][k] = round(time.time() - t0, 1)
                acc = sum(r[2] == r[1] for r in R[k]) / len(R[k])
                print(mode, k, len(R[k]), round(acc, 4), R["seconds"][k], "s", flush=True)
                jdump(OUT, outp)
    del eng
    if dev == "cuda":
        torch.cuda.empty_cache()
print("done", a.tag, flush=True)
