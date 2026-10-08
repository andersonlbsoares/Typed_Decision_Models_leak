"""Julia-1 helpers shared by smoke.py, e1_julia_dilution.py and e4_julia_det.py.
Evaluation reproduces r4_eval.julia_all (classification) and r2_julia_nli (typed-decisions);
fine-tuning reproduces r6_julia_ft.run (AdamW lr 1e-5, wd 0.01, bs 16, bf16 autocast, clip 1.0)."""
import contextlib, inspect, os, random, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "code"))
import run_julia as RJ
from common import subset_opts, typed_rows


def engine(device, bs):
    return RJ.load_engine(device, batch_size=bs)


def _pick(z):
    p = np.exp(np.array(z, dtype=np.float64) - max(z)); p /= p.sum()
    return int(p.argmax()), float(p.max())


def cls_rows(eng, items, labels, cut=None, with_target=False):
    rows, opts = [], []
    for it in items:
        o = subset_opts(it["id"], it["target"], len(labels)); opts.append(o)
        t = it["text"] if cut is None else it["text"][:cut]
        r = RJ.row(RJ.truncate(eng, t)[0], [labels[c] for c in o])
        if with_target:
            r["target"] = o.index(it["target"])
        rows.append(r)
    return rows, opts


def eval_cls(eng, items, labels, cut=None, chunk=512):
    """-> [[id, gold, pred, conf]]; pred None when the request cannot be encoded (counted wrong)."""
    rows, opts = cls_rows(eng, items, labels, cut)
    out = []
    for it, o, z in zip(items, opts, RJ.safe_logits(eng, rows, chunk=chunk)):
        if z is None:
            out.append([it["id"], it["target"], None, None]); continue
        j, c = _pick(z); out.append([it["id"], it["target"], o[j], round(c, 6)])
    return out


def eval_typed(eng, cases, chunk=256):
    rows, meta = typed_rows(cases)
    out = []
    for m, z in zip(meta, RJ.safe_logits(eng, rows, chunk=chunk)):
        if z is None:
            out.append([m["id"], m["gold"], None, None]); continue
        j, c = _pick(z); out.append([m["id"], m["gold"], m["keys"][j], round(c, 6)])
    return out


@contextlib.contextmanager
def no_autocast():
    """Force autocast off (for the GPU FP32 variant). Validated against CPU FP32 in e4."""
    saved = (torch.autocast, torch.amp.autocast)

    class _Off(saved[0]):
        def __init__(self, *a, **k):
            k["enabled"] = False
            super().__init__(*a, **k)
    torch.autocast = torch.amp.autocast = _Off
    try:
        yield
    finally:
        torch.autocast, torch.amp.autocast = saved


def engine_info(eng):
    p = next(eng.model.parameters())
    try:
        src = inspect.getsource(sys.modules[type(eng).__module__])
    except Exception:
        src = ""
    return dict(engine=type(eng).__name__, param_dtype=str(p.dtype), device=str(p.device),
                engine_mentions_autocast="autocast" in src, engine_mentions_bfloat16="bfloat16" in src,
                torch=torch.__version__)


def train_epoch(eng, opt, rows, bs=16):
    m = eng.model; m.train(); random.shuffle(rows)
    for i in range(0, len(rows), bs):
        batch = eng.collate(rows[i:i + bs], include_targets=True)
        batch = {k: v.to("cuda") for k, v in batch.items()}
        y = batch.pop("labels")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            scores = m(**batch)
        loss = torch.nn.functional.cross_entropy(scores.float(), y)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite loss")
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    m.eval(); eng.clear_cache()


def make_opt(eng, lr=1e-5):
    return torch.optim.AdamW([p for p in eng.model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
