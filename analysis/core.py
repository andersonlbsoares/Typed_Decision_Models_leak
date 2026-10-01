"""Label-standardized split contrasts with fixed common-support weights,
stratified (within split x stratum) bootstrap, and a null-calibrated
shift test.  All estimands are linear in per-stratum means of discrete
per-item values, so the item bootstrap within a stratum is exactly a
multinomial draw over the observed value categories (fast, exact).

Estimand for samples S1, S2 with per-item values v and strata y:
    T = sum_{c in C} w_c (mean_{1c} v - mean_{2c} v),
    C = strata present in both samples, w_c = (n_1c + n_2c) / sum_{c' in C}(n_1c' + n_2c').
Weights and support are fixed at their observed values in every replicate
(stratified resampling keeps n_jc fixed, so no stratum is ever empty).
p-value (two-sided shift test, Efron & Tibshirani 1993 sec. 16.4):
    p = (1 + #{b : |T*_b - T| >= |T|}) / (B + 1).
"""
import numpy as np

B_DEFAULT = 50_000


def _cells(v, y):
    v = np.asarray(v, float); y = np.asarray(y)
    out = {}
    for c in np.unique(y):
        m = y == c
        u, k = np.unique(v[m], return_counts=True)
        out[c] = (u, k, int(m.sum()))
    return out


def _boot_means(u, k, n, B, rng):
    if len(u) == 1:
        return np.full(B, u[0])
    draws = rng.multinomial(n, k / n, size=B)
    return draws @ u / n


def contrast(v1, y1, v2, y2, B=B_DEFAULT, rng=None, return_boot=False):
    rng = rng or np.random.default_rng(0)
    c1, c2 = _cells(v1, y1), _cells(v2, y2)
    C = sorted(set(c1) & set(c2), key=str)
    n = np.array([c1[c][2] + c2[c][2] for c in C], float)
    w = n / n.sum()
    m1 = np.array([(c1[c][0] * c1[c][1]).sum() / c1[c][2] for c in C])
    m2 = np.array([(c2[c][0] * c2[c][1]).sum() / c2[c][2] for c in C])
    T = float(w @ (m1 - m2))
    bs = np.zeros(B)
    for wc, c in zip(w, C):
        bs += wc * (_boot_means(*c1[c], B, rng) - _boot_means(*c2[c], B, rng))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = (1 + np.sum(np.abs(bs - T) >= abs(T) - 1e-12)) / (B + 1)
    n1, n2 = len(np.asarray(y1)), len(np.asarray(y2))
    kept1 = sum(c1[c][2] for c in C); kept2 = sum(c2[c][2] for c in C)
    res = dict(est=100 * T, lo=100 * lo, hi=100 * hi, se=100 * bs.std(ddof=1), p=float(p), B=B,
               n1=n1, n2=n2, strata_1=len(c1), strata_2=len(c2), strata_used=len(C),
               excluded=";".join(str(c) for c in sorted(set(c1) ^ set(c2), key=str)),
               coverage_1=kept1 / n1, coverage_2=kept2 / n2)
    if return_boot:
        res["_boot"] = 100 * bs
    return res


def raw_diff(v1, v2):
    return 100 * (np.mean(v1) - np.mean(v2))


def acc_vec(recs):
    """records [id, gold, pred, conf] -> (ids, correct, gold); pred None counts as incorrect."""
    ids = [r[0] for r in recs]
    return ids, np.array([float(r[1] == r[2]) for r in recs]), np.array([r[1] for r in recs])


def paired(recs_a, recs_b):
    """Match two record lists on id -> (ids, corr_a, corr_b, gold)."""
    mb = {r[0]: r for r in recs_b}
    ids = [r[0] for r in recs_a if r[0] in mb]
    ma = {r[0]: r for r in recs_a}
    ca = np.array([float(ma[i][1] == ma[i][2]) for i in ids])
    cb = np.array([float(mb[i][1] == mb[i][2]) for i in ids])
    g = np.array([ma[i][1] for i in ids])
    gb = np.array([mb[i][1] for i in ids])
    assert (g == gb).all(), "gold mismatch between paired records"
    return ids, ca, cb, g


def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)
    adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1, (m - r) * p[i])); adj[i] = run
    return adj


def bh(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)[::-1]
    adj = np.empty(m); run = 1
    for r, i in enumerate(o):
        rank = m - r; run = min(run, p[i] * m / rank); adj[i] = run
    return adj
