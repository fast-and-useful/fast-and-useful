"""The frozen-feature learner, a metric of the metric study: how much a generated set teaches a linear classifier on a
pretrained encoder (DINOv3 ViT-L/16), next to the ResNet-50 trained from scratch (train.py). The encoder sees real
images only in pretraining, so this learner reads class content and is little affected by low-level image statistics.

For each set: encoder features (CLS token after the final norm, center crop 224) of its training images, cached; a
softmax-regression probe on standardized features, its L2 weight chosen on 10% of the set's own images (never on real
validation images), refit on all of it; top-1 on the real validation images. Also k-NN (k = 20, cosine) top-1 with the
set as the memory bank.

Sets are [NAME=]PREFIX[:domain[:N]] as in train.py (e.g. real500=work/real/in100_train:0:500); NAME defaults to
the prefix's basename.

Usage: python -m fau.metrics.frozen --sets SPEC [SPEC ...] --val VAL --out FILE.json --cache DIR
"""
import argparse, json, os, time

import torch
import torch.nn.functional as F
from torchvision.transforms import v2 as T

from fau.devices import DEV, WORKERS, half
from fau.hub import snapshot
from fau.packs import PackSet
from fau.train import build_sources

ENCODERS = {"dinov3_vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m"}


@torch.no_grad()
def features(model, spec, workers=WORKERS):
    tf = T.Compose([T.Resize(256, antialias=True), T.CenterCrop(224), T.ToImage(), T.ToDtype(torch.float32, scale=True),
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    ds = PackSet(build_sources([spec]), tf)
    dl = torch.utils.data.DataLoader(ds, batch_size=256, num_workers=workers, pin_memory=DEV == "cuda")
    out = [model(pixel_values=x.to(DEV, half(torch.bfloat16))).pooler_output.float().cpu() for x, _, _ in dl]
    return torch.cat(out), torch.from_numpy(ds.labels().copy())


def cached(model, name, spec, cache):
    f = f"{cache}/{name}.pt"
    if os.path.exists(f):
        return torch.load(f)
    x, y = features(model, spec)
    os.makedirs(cache, exist_ok=True)
    torch.save((x.half(), y), f + ".tmp"); os.replace(f + ".tmp", f)
    return x.half(), y


def fit(x, y, ncls, l2, iters=300):
    w = torch.zeros(x.shape[1], ncls, device=x.device, requires_grad=True)
    b = torch.zeros(ncls, device=x.device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=iters, line_search_fn="strong_wolfe", tolerance_grad=1e-6, tolerance_change=1e-9)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(x @ w + b, y) + l2 * (w ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach(), b.detach()


def linear_probe(xtr, ytr, xte, yte, ncls, seed=0):
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(xtr), generator=g).to(xtr.device)
    ho, tr = idx[: len(idx) // 10], idx[len(idx) // 10:]
    sel = {}
    for l2 in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
        w, b = fit(xtr[tr], ytr[tr], ncls, l2)
        sel[l2] = ((xtr[ho] @ w + b).argmax(1) == ytr[ho]).float().mean().item()
    best = max(sel, key=sel.get)
    w, b = fit(xtr, ytr, ncls, best)
    return ((xte @ w + b).argmax(1) == yte).float().mean().item(), best, sel


def knn(xtr, ytr, xte, yte, ncls, k=20, t=0.07):
    xtr, xte = F.normalize(xtr, dim=1), F.normalize(xte, dim=1)
    correct = 0
    for i in range(0, len(xte), 1024):
        s, j = (xte[i:i + 1024] @ xtr.T).topk(k, dim=1)
        votes = torch.zeros(len(s), ncls, device=xte.device).scatter_add_(1, ytr[j], (s / t).exp())
        correct += (votes.argmax(1) == yte[i:i + 1024]).sum().item()
    return correct / len(xte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", default="dinov3_vitl16", choices=list(ENCODERS))
    ap.add_argument("--cache", required=True, help="feature cache directory (cheap to recreate)")
    ap.add_argument("--ncls", type=int, help="classes (default: the validation pack's class list, else the largest label + 1)")
    a = ap.parse_args()
    from transformers import AutoModel
    model = AutoModel.from_pretrained(snapshot(ENCODERS[a.encoder]), dtype=half(torch.bfloat16), disable_mmap=True).to(DEV).eval()
    cache = f"{a.cache}/{a.encoder}"
    xte, yte = cached(model, "_val_" + os.path.basename(a.val.rstrip("/")), a.val, cache)
    xte, yte = xte.float().to(DEV), yte.to(DEV)
    classes = a.val.rstrip("/") + "_classes.json"   # Food-101 has 101 classes: a fixed 100 crashed its probe (2026-10-14)
    ncls = a.ncls or (len(json.load(open(classes))["wnids"]) if os.path.exists(classes) else int(yte.max()) + 1)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for spec in a.sets:
        name, spec = spec.split("=", 1) if "=" in spec else (os.path.basename(spec.split(":")[0].rstrip("/")), spec)
        t0 = time.time()
        xtr, ytr = cached(model, name, spec, cache)
        xtr, ytr = xtr.float().to(DEV), ytr.to(DEV)
        top1, l2, sel = linear_probe(xtr, ytr, xte, yte, ncls)
        res[name] = dict(linear=top1, knn=knn(xtr, ytr, xte, yte, ncls), l2=l2, holdout=sel, n=len(ytr), encoder=a.encoder)
        print(f"PROBE {name}: linear {top1:.4f}, k-NN {res[name]['knn']:.4f} (l2 {l2:g}, {len(ytr)} images, {time.time() - t0:.0f} s)",
              flush=True)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out + ".tmp", "w"), indent=1); os.replace(a.out + ".tmp", a.out)


if __name__ == "__main__":
    main()
