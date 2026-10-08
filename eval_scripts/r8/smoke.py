"""Smoke test (<gpu-dev-partition>, <=20 min): environment, checksums, agreement with the stored Casa
predictions, and throughput of every component, then projected wall time and UA per job.
Writes results/smoke_report.json. Run from the package root with the main env active."""
import json, os, re, subprocess, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import julia_lib as J, run_julia as RJ
from common import DATA, REF, RES, R8, jload, jdump, load_cls

rep = dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0))
import transformers; rep["transformers"] = transformers.__version__
rep["julia_weights_sha_ok"] = RJ.sha_file(RJ.CKPT / "model.safetensors") == RJ.WEIGHTS_SHA
S = load_cls(); typed = jload(os.path.join(DATA, "typed_samples.json.gz"))
ref4 = jload(os.path.join(REF, "r4_eval.json.gz"))["julia"]; ref2 = jload(os.path.join(REF, "r2_julia_nli.json.gz"))["typed"]
emo = S["emotion"]; ag = S["agnews"]


def agree(a, b):
    return round(float(np.mean([x[2] == y[2] for x, y in zip(a, b)])), 4)


def timed(f, *x):
    t = time.time(); r = f(*x); torch.cuda.synchronize() if torch.cuda.is_available() else None
    return r, time.time() - t


eng = J.engine("cuda", 64); rep["engine"] = J.engine_info(eng)
r, t = timed(J.eval_cls, eng, emo["splits"]["train"][:512], emo["hyps"])
rep["julia_gpu_bs64"] = dict(it_s=round(512 / t, 1), agree_stored=agree(r, ref4["emotion"]["A_train"][:512]))
r, t = timed(J.eval_typed, eng, typed["test"][:40])
st = {x["id"]: x["pred"] for x in ref2["test"]}
rep["julia_typed_gpu_bs64"] = dict(it_s=round(len(r) / t, 1), agree_stored=round(float(np.mean([x[2] == st[x[0]] for x in r])), 4))
eng1 = J.engine("cuda", 1); r1, t = timed(J.eval_cls, eng1, emo["splits"]["train"][:64], emo["hyps"])
rep["julia_gpu_bs1"] = dict(it_s=round(64 / t, 1), agree_bs64=agree(r1, ref4["emotion"]["A_train"][:64]))
eng1.model.float(); eng1.clear_cache()
with J.no_autocast():
    r2, t = timed(J.eval_cls, eng1, emo["splits"]["train"][:64], emo["hyps"])
rep["julia_gpu_fp32_bs1"] = dict(it_s=round(64 / t, 1), engine=J.engine_info(eng1), agree_bs1=agree(r2, r1),
                               conf_maxdiff=round(float(np.nanmax([abs((a[3] or 0) - (b[3] or 0)) for a, b in zip(r1, r2)])), 6))
del eng1; torch.cuda.empty_cache()
torch.set_num_threads(8)
engc = J.engine("cpu", 1); rc, t = timed(J.eval_cls, engc, emo["splits"]["train"][:16], emo["hyps"])
rep["julia_cpu_fp32_bs1_8thr"] = dict(it_s=round(16 / t, 2), agree_gpu_fp32=agree(rc, r2[:16]), engine=J.engine_info(engc))
del engc
# fine-tuning speed (r6 recipe), 30 steps on AG News
import copy
state = copy.deepcopy(eng.model.state_dict())
rows, _ = J.cls_rows(eng, ag["splits"]["train"][:480], ag["hyps"], cut=1200, with_target=True)
opt = J.make_opt(eng); t = time.time(); J.train_epoch(eng, opt, rows); torch.cuda.synchronize()
rep["julia_ft_s_per_step_agnews"] = round((time.time() - t) / 30, 4)
eng.model.load_state_dict(state); del eng; torch.cuda.empty_cache()
jdump(rep, os.path.join(RES, "smoke_report.json"))
# Qwen (separate process so its memory is released)
q = subprocess.run([sys.executable, "-u", os.path.join(R8, "r8", "e3_qwen_full.py"), "--keys", "emotion,yelp",
                    "--limit", "256", "--tag", "smoke"], capture_output=True, text=True)
print(q.stdout[-2000:], q.stderr[-2000:])
rates = [float(x) for x in re.findall(r"([\d.]+) it/s", q.stdout)]
rep["qwen_it_s"] = rates; rep["qwen_load_s"] = int((re.findall(r"loaded (\d+) s", q.stdout) or [-1])[0])
if os.path.exists(os.path.join(RES, "r8_e3_qwen_smoke.json.gz")):
    qs = jload(os.path.join(RES, "r8_e3_qwen_smoke.json.gz"))["A"]; qr = jload(os.path.join(REF, "r6_qwen.json.gz"))["A"]
    rep["qwen_agree_stored_vllm"] = {k: agree([x[:2] + [x[4]] for x in qs[k]["train"]], qr[k]["train"][:256]) for k in qs}
jdump(rep, os.path.join(RES, "smoke_report.json"))
print(json.dumps(rep, indent=1))
