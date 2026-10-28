"""The metric study: which set metric tracks training value? Spearman correlations of set metrics with ResNet-50 top-1
over the 186 template-prompt ImageNet-100 sets (results/metric_study.json), with 95% percentile bootstrap intervals
(10,000 resamples of sets; a numpy generator seeded per statistic, so every number is reproducible on its own):

  pooled            every set
  within family     each generator family with at least 8 sets
  family-centered   ranks within each family, centered and divided by the family's size, then correlated over all sets
                    (resampled within families, so family sizes stay fixed)
  held-out family   isotonic regression of top-1 on the metric fit on the other families (direction from the sign of
                    their Spearman), predicting the held-out family; Spearman of the out-of-fold predictions per family
                    with at least 5 sets, and pooled over all sets
  partial           aesthetic score and PickScore against top-1 given DINOv3 recall, family-centered
  equal-family      the mean of the within-family Spearman correlations of the four families with at least 8 sets,
                    against the family-centered statistic, on the same 10,000 paired within-family resamples

A resample in which a metric or top-1 is constant has no correlation; it is dropped and counted.

    python -m fau.metric_study [--inputs results/metric_study.json] [--out FILE.json] [--workers 8]
"""
import argparse
import json
import math
import random
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import rankdata

METRICS = [   # (key, label in the supplementary table); distances correlate negatively when they track value
    ("dino_recall", "recall, DINOv3"), ("clip_recall", "recall, CLIP"), ("clip_fd", "Fr\\'echet distance, CLIP"),
    ("dino_fd", "Fr\\'echet distance, DINOv3"), ("dino_coverage", "coverage, DINOv3"), ("clip_coverage", "coverage, CLIP"),
    ("dino_precision", "precision, DINOv3"), ("dino_density", "density, DINOv3"), ("dino_vendi", "Vendi, DINOv3"),
    ("clip_vendi", "Vendi, CLIP"), ("fid", "FID (Inception)"), ("inception_score", "Inception Score"),
    ("recog", "recognizability"), ("dino_linear", "frozen DINOv3, linear probe"), ("dino_knn", "frozen DINOv3, $k$-NN"),
    ("contrast", "contrast"), ("sharpness", "sharpness"), ("saturation", "saturation"), ("colorfulness", "colorfulness"),
    ("hf", "high-frequency share"), ("slope", "spectral slope"), ("aesthetic", "aesthetic score"), ("pickscore", "PickScore"),
]
PARTIAL = [("aesthetic", "dino_recall"), ("pickscore", "dino_recall")]
WEIGHTING = [("dino_recall", 1), ("clip_recall", 1), ("clip_fd", -1), ("fid", -1)]   # distances negated
SUBSET = "templates"   # the random streams are keyed by subset/metric/statistic, as when the study was computed
MIN_WITHIN, MIN_HELDOUT, B, SEED = 8, 5, 10000, 0


# Statistics ------------------------------------------------------------------------------------------------------------
def pearson(x, y):
    """Row-wise Pearson correlation of (..., n) arrays; nan where either row is constant."""
    x = x - x.mean(-1, keepdims=True)
    y = y - y.mean(-1, keepdims=True)
    den = np.sqrt((x * x).sum(-1) * (y * y).sum(-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 1e-12, (x * y).sum(-1) / np.where(den > 1e-12, den, 1), np.nan)


def rank(v):
    return rankdata(v, axis=-1)   # average ranks for ties


def centered(v, blocks):
    """Ranks within each family (column blocks), centered and divided by the family's size."""
    out = np.empty(v.shape)
    for b in blocks:
        out[..., b] = (rank(v[..., b]) - (len(b) + 1) / 2) / len(b)
    return out


def partial(rxy, rxz, ryz):
    with np.errstate(invalid="ignore", divide="ignore"):
        return (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))


def summary(point, boot, n):
    ok = boot[~np.isnan(boot)]
    lo, hi = np.percentile(ok, [2.5, 97.5]) if len(ok) else (math.nan, math.nan)
    return dict(r=float(point), lo=float(lo), hi=float(hi), n=int(n), undefined=int(len(boot) - len(ok)))


def rng_for(*keys):
    return np.random.default_rng([SEED] + [zlib.crc32(k.encode()) for k in keys])


def strat(rng, fams):
    """A (B, n) resampling index within each family; columns grouped by family -> (index, column blocks)."""
    idx, blocks, start = [], [], 0
    for f in sorted(set(fams)):
        mem = np.flatnonzero(fams == f)
        idx.append(mem[rng.integers(0, len(mem), size=(B, len(mem)))])
        blocks.append(np.arange(start, start + len(mem)))
        start += len(mem)
    return np.concatenate(idx, axis=1), blocks


def blocks_of(fams):
    return [np.flatnonzero(fams == f) for f in sorted(set(fams))]


def corr_pooled(x, y, key):
    i = rng_for(key).integers(0, len(x), size=(B, len(x)))
    return summary(pearson(rank(x), rank(y)), pearson(rank(x[i]), rank(y[i])), len(x))


def corr_centered(x, y, fams, key):
    keep = np.isin(fams, [f for f in set(fams) if (fams == f).sum() >= 2])
    x, y, fams = x[keep], y[keep], fams[keep]
    i, blocks = strat(rng_for(key), fams)
    pb = blocks_of(fams)
    return summary(pearson(centered(x, pb), centered(y, pb)), pearson(centered(x[i], blocks), centered(y[i], blocks)), len(x))


def iso_fit(xtr, ytr, xte):
    """Isotonic regression of y on x (direction from the sign of their Spearman); ties averaged with multiplicity
    weights; linear between knots, clipped at the ends."""
    from scipy.optimize import isotonic_regression
    inc = not pearson(rank(xtr), rank(ytr)) < 0
    xu, inv = np.unique(xtr, return_inverse=True)
    w = np.bincount(inv).astype(float)
    yu = np.bincount(inv, weights=ytr) / w
    return np.interp(xte, xu, isotonic_regression(yu, weights=w, increasing=inc).x)


def lofo_once(x, y, blocks, heldout):
    pred = np.empty(len(y))
    for b in blocks:
        tr = np.ones(len(y), bool)
        tr[b] = False
        pred[b] = iso_fit(x[tr], y[tr], x[b])
    per = [pearson(rank(pred[b]), rank(y[b])) for b, h in zip(blocks, heldout) if h]
    return per, pearson(rank(pred), rank(y))


def lofo(job):
    """Leave one family out, isotonic: Spearman of the out-of-fold predictions per held-out family and pooled."""
    x, y, fams, key = job
    order = sorted(set(fams))
    pb = blocks_of(fams)
    heldout = [len(b) >= MIN_HELDOUT for b in pb]
    p_per, p_pool = lofo_once(x, y, pb, heldout)
    i, blocks = strat(rng_for(key), fams)
    bs = [lofo_once(x[ii], y[ii], blocks, heldout) for ii in i]
    per = np.array([b[0] for b in bs]).reshape(B, -1)
    held = [(f, b) for f, b, h in zip(order, pb, heldout) if h]
    return key, dict(held_out={f: summary(p, per[:, k], len(b)) for k, ((f, b), p) in enumerate(zip(held, p_per))},
                     pooled=summary(p_pool, np.array([b[1] for b in bs]), len(y)))


def partial_centered(x, y, z, fams, key):
    def pc(a, b, c):
        return partial(pearson(a, b), pearson(a, c), pearson(b, c))
    keep = np.isin(fams, [f for f in set(fams) if (fams == f).sum() >= 2])
    x, y, z, fams = x[keep], y[keep], z[keep], fams[keep]
    i, blocks = strat(rng_for(key, "centered"), fams)
    pb = blocks_of(fams)
    return summary(pc(centered(x, pb), centered(y, pb), centered(z, pb)),
                   pc(centered(x[i], blocks), centered(y[i], blocks), centered(z[i], blocks)), len(y))


def arrays(sets, *keys):
    rows = [s for s in sets if all(not math.isnan(s["metrics"][k]) for k in keys) and not math.isnan(s["top1"])]
    return (np.array([s["top1"] for s in rows]), np.array([s["family"] for s in rows]),
            [np.array([s["metrics"][k] for s in rows]) for k in keys])


def analyze(sets, workers=4):
    """sets: {name: {family, top1, metrics}} in the study's order -> {metrics: {...}, partial: {...}}."""
    sel = list(sets.values())
    res, jobs = {"metrics": {}, "partial": {}}, []
    for key, _ in METRICS:
        y, fams, (x,) = arrays(sel, key)
        fam_n = {f: int((fams == f).sum()) for f in sorted(set(fams))}
        res["metrics"][key] = dict(
            n=len(y), families=fam_n, pooled=corr_pooled(x, y, f"{SUBSET}/{key}/pooled"),
            within={f: corr_pooled(x[fams == f], y[fams == f], f"{SUBSET}/{key}/within/{f}")
                    for f, k in fam_n.items() if k >= MIN_WITHIN},
            centered=corr_centered(x, y, fams, f"{SUBSET}/{key}/centered"))
        jobs.append((x, y, fams, f"{SUBSET}/{key}/lofo/isotonic"))
    for xk, zk in PARTIAL:
        y, fams, (x, z) = arrays(sel, xk, zk)
        res["partial"][f"{xk}|{zk}"] = dict(centered=partial_centered(x, y, z, fams, f"{SUBSET}/{xk}|{zk}"))
    with ProcessPoolExecutor(workers) as ex:
        for key, out in ex.map(lofo, jobs):
            res["metrics"][key.split("/")[1]]["lofo"] = out
    return res


# Equal-family sensitivity ----------------------------------------------------------------------------------------------
def ranks0(values):
    """Average ranks centered on zero, ties included."""
    order = sorted(range(len(values)), key=values.__getitem__)
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        for k in order[i:j]:
            out[k] = (i + j - 1) / 2 - (len(values) - 1) / 2
        i = j
    return out


def family_weighting(sets, resamples=B):
    """Family-centered ranks against the equal-weight mean of within-family Spearman correlations, for the four families
    with at least 8 sets, on the same paired within-family resamples (Python's random.Random(0))."""
    groups = {}
    for s in sets.values():
        groups.setdefault(s["family"], []).append(s)
    families = sorted(f for f, b in groups.items() if len(b) >= MIN_WITHIN)
    blocks = [groups[f] for f in families]

    def calculate(draws):
        covs, xvars, yvars, per = ([0.0] * len(WEIGHTING) for _ in range(4))
        for block, idx in zip(blocks, draws):
            n = len(idx)
            ry = ranks0([block[i]["top1"] for i in idx])
            vy = sum(v * v for v in ry)
            for k, (metric, direction) in enumerate(WEIGHTING):
                rx = ranks0([direction * block[i]["metrics"][metric] for i in idx])
                vx = sum(v * v for v in rx)
                if vx == 0 or vy == 0:
                    return None
                cov = sum(a * b for a, b in zip(rx, ry))
                per[k] += cov / math.sqrt(vx * vy) / len(blocks)
                covs[k] += cov / n ** 2
                xvars[k] += vx / n ** 2
                yvars[k] += vy / n ** 2
        return [(covs[k] / math.sqrt(xvars[k] * yvars[k]), per[k]) for k in range(len(WEIGHTING))]

    def quantiles(values):
        values = sorted(values)

        def q(p):
            i = (len(values) - 1) * p
            lo, hi = int(i), math.ceil(i)
            return values[lo] + (values[hi] - values[lo]) * (i - lo)
        return [q(.025), q(.975)]

    point = calculate([list(range(len(b))) for b in blocks])
    rng, boot, undefined = random.Random(0), [[] for _ in WEIGHTING], 0
    for _ in range(resamples):
        values = calculate([[rng.randrange(len(b)) for _ in b] for b in blocks])
        if values is None:
            undefined += 1
            continue
        for k, v in enumerate(values):
            boot[k].append(v)
    results = {m: dict(centered=dict(value=point[k][0], interval=quantiles([v[0] for v in boot[k]])),
                       equal_family=dict(value=point[k][1], interval=quantiles([v[1] for v in boot[k]])))
               for k, (m, _) in enumerate(WEIGHTING)}
    index = {m: k for k, (m, _) in enumerate(WEIGHTING)}
    fd, recall = index["clip_fd"], index["dino_recall"]
    paired = dict(value=point[fd][1] - point[recall][1],
                  interval=quantiles([a[1] - b[1] for a, b in zip(boot[fd], boot[recall])]))
    return dict(families={f: len(b) for f, b in zip(families, blocks)}, resamples=resamples, undefined_draws=undefined,
                results=results, clip_fd_minus_dino_recall=paired)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="results/metric_study.json")
    ap.add_argument("--out", default="metric-study-analysis.json")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    sets = json.load(open(a.inputs))["sets"]
    out = dict(correlations=analyze(sets, a.workers), family_weighting=family_weighting(sets))
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"METRIC STUDY: {len(sets)} sets -> {a.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
