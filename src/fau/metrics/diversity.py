"""Diversity and realism of generated sets in a frozen encoder's feature space (DINOv3 ViT-L/16), from the feature cache
that frozen.py fills (features of missing sets are computed and cached).

Per class, then averaged over classes:
  pairwise   mean cosine distance between two images of the class (higher = more diverse)
  vendi      Vendi score of the class's cosine-similarity kernel (effective number of distinct images)
  precision, recall, density, coverage   Naeem et al. 2020 (k = 5) against the class's real training images
And over the whole set: the Frechet distance to the real images' features.

Per-class recall is kept too (recall_per_class: the per-class analysis of the adapters' gains).

Usage: python -m fau.metrics.diversity --sets SPEC [SPEC ...] --real real500=work/real/in100_train:0:500 --out FILE.json
       --cache DIR [--encoder dinov3_vitl16]
"""
import argparse, json, os, time

import torch

from fau.devices import DEV, WORKERS, half
from fau.hub import snapshot
from fau.metrics.frozen import ENCODERS, cached


def per_class(x, y, ncls):
    return [x[y == c] for c in range(ncls)]


def pairwise(xc):
    n = len(xc)
    s = xc.sum(0)
    return 1 - ((s @ s) - n) / (n * (n - 1))


def vendi(xc):
    lam = torch.linalg.eigvalsh(xc @ xc.T / len(xc)).clamp(min=0)
    lam = lam[lam > 0]
    return torch.exp(-(lam * lam.log()).sum())


def prdc(real, fake, k=5):
    drr, dff, drf = torch.cdist(real, real), torch.cdist(fake, fake), torch.cdist(real, fake)
    rr = drr.kthvalue(k + 1, dim=1).values          # k-th neighbour, the point itself is the first
    rf = dff.kthvalue(k + 1, dim=1).values
    inside = drf <= rr[:, None]                     # fake j inside real i's ball
    return dict(precision=inside.any(0).float().mean().item(),
                recall=(drf <= rf[None, :]).any(1).float().mean().item(),
                density=(inside.float().sum(0) / k).mean().item(),
                coverage=(drf.min(1).values <= rr).float().mean().item())


def frechet(a, b):
    ma, mb = a.mean(0), b.mean(0)
    ca, cb = torch.cov(a.T), torch.cov(b.T)
    ev = torch.linalg.eigvals(ca @ cb).real.clamp(min=0)   # tr sqrt(ca cb) through the eigenvalues of ca cb
    return ((ma - mb) ** 2).sum() + ca.trace() + cb.trace() - 2 * ev.sqrt().sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--real", required=True, help="NAME=SPEC of the real reference images")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--encoder", default="dinov3_vitl16", choices=list(ENCODERS))
    ap.add_argument("--ncls", type=int, default=100)
    a = ap.parse_args()
    from transformers import AutoModel
    model = AutoModel.from_pretrained(snapshot(ENCODERS[a.encoder]), dtype=half(torch.bfloat16), disable_mmap=True).to(DEV).eval()
    cache = f"{a.cache}/{a.encoder}"
    rname, rspec = a.real.split("=", 1)
    xr, yr = cached(model, rname, rspec, cache)
    xr = torch.nn.functional.normalize(xr.float().to(DEV), dim=1); yr = yr.to(DEV)
    real = per_class(xr, yr, a.ncls)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for spec in [a.real] + a.sets:
        name, spec = spec.split("=", 1) if "=" in spec else (os.path.basename(spec.split(":")[0].rstrip("/")), spec)
        t0 = time.time()
        x, y = cached(model, name, spec, cache)
        x = torch.nn.functional.normalize(x.float().to(DEV), dim=1); y = y.to(DEV)
        fake = per_class(x, y, a.ncls)
        m = dict(pairwise=torch.stack([pairwise(f) for f in fake]).mean().item(),
                 vendi=torch.stack([vendi(f) for f in fake]).mean().item())
        if name != rname:
            p = [prdc(r, f) for r, f in zip(real, fake)]
            m.update({k: sum(d[k] for d in p) / len(p) for k in p[0]})
            m["recall_per_class"] = [d["recall"] for d in p]
            m["frechet"] = frechet(x.double(), xr.double()).item()
        res[name] = dict(m, encoder=a.encoder, n=len(y))
        print(f"DIVERSITY {name}: " + ", ".join(f"{k} {v:.4f}" for k, v in m.items()) + f" ({time.time() - t0:.0f} s)", flush=True)
        json.dump(res, open(a.out + ".tmp", "w"), indent=1); os.replace(a.out + ".tmp", a.out)


if __name__ == "__main__":
    main()
