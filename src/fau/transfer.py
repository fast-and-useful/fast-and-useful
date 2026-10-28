"""Do the features of a ResNet-50 trained from scratch on generated images transfer? Linear probes on the frozen
penultimate features (2,048 per image, the classifier's fc removed) of the ImageNet-100 classifiers, on datasets none of
them saw: Food-101, Stanford Cars, Oxford-IIIT Pets (their full train and test splits) and ImageNet-900 (the 900
ImageNet-1k classes outside ImageNet-100: 100 training images per class, the 50 validation images per class).

Each target is read once (the training evaluation transform: resize 256, center crop 224) and passed through every
backbone. The probe: standardized features, softmax regression by L-BFGS, its L2 weight chosen on 10% of the target's
own training images, refit on all of them; top-1 and per-class top-1 on the target's test images.

  python -m fau.transfer --runs real500_s0 klein_s0 ... [--out work/probes/transfer.json] [--targets food101 ...]
"""
import argparse, fcntl, json, os, time

import torch
import torchvision.transforms.v2.functional as TF
from torchvision.transforms import v2 as T

from fau import PROBES, REAL, RUNS
from fau.metrics.frozen import fit
from fau.ood import MEAN, STD, load_net
from fau.packs import PackSet
from fau.prompts import class_list
from fau.train import build_sources

TARGETS = {"food101": ("food101_train", "food101_test"), "cars": ("cars_train", "cars_test"),
           "pets": ("pets_train", "pets_test"), "in900": ("in1k_train:0:100", "in1k_val")}


def images(spec, workers):
    ds = PackSet(build_sources([f"{REAL}/{spec}"]), T.Compose([T.Resize(256, antialias=True), T.CenterCrop(224), T.ToImage()]))
    dl = torch.utils.data.DataLoader(ds, batch_size=256, num_workers=workers)
    return torch.cat([x for x, _, _ in dl]).as_subclass(torch.Tensor), torch.from_numpy(ds.labels().copy())


def in900_keep(labels):
    """Labels of ImageNet-1k packs (in label order) outside ImageNet-100 -> (mask, labels renumbered 0..899)."""
    in1k = class_list("in1k")["wnids"]
    in100 = set(class_list("in100")["wnids"])
    keep = [k for k, w in enumerate(in1k) if w not in in100]
    assert len(keep) == 900, len(keep)
    new = torch.full((len(in1k),), -1, dtype=torch.long)
    new[keep] = torch.arange(900)
    y = new[labels]
    return y >= 0, y


@torch.no_grad()
def features(nets, x, bs=512):
    out = {r: [] for r in nets}
    for i in range(0, len(x), bs):
        b = TF.normalize(TF.to_dtype(x[i:i + bs].cuda(), torch.float32, scale=True), MEAN, STD).to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for r, net in nets.items():
                out[r].append(net(b).half().cpu())   # half on the host: 21 backbones x 100,000 images
    return {r: torch.cat(v) for r, v in out.items()}


def probe(xtr, ytr, xte, yte, ncls, seed=0):
    """metrics.frozen.linear_probe, returning the test predictions: -> (top-1, per-class top-1, chosen L2)."""
    xtr, xte, ytr, yte = xtr.cuda().float(), xte.cuda().float(), ytr.cuda(), yte.cuda()
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    idx = torch.randperm(len(xtr), generator=torch.Generator().manual_seed(seed)).cuda()
    ho, tr = idx[: len(idx) // 10], idx[len(idx) // 10:]
    sel = {}
    for l2 in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
        w, b = fit(xtr[tr], ytr[tr], ncls, l2)
        sel[l2] = ((xtr[ho] @ w + b).argmax(1) == ytr[ho]).float().mean().item()
    best = max(sel, key=sel.get)
    w, b = fit(xtr, ytr, ncls, best)
    hit = ((xte @ w + b).argmax(1) == yte).float()
    per = {str(c): hit[yte == c].mean().item() for c in range(ncls) if (yte == c).any()}
    return hit.mean().item(), per, best


def save(path, run, res):
    with open(path + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        allres = json.load(open(path)) if os.path.exists(path) else {}
        allres.setdefault(run, {}).update(res)
        json.dump(allres, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--targets", nargs="+", default=list(TARGETS))
    ap.add_argument("--out", default=str(PROBES / "transfer.json"))
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    nets = {}
    for r in a.runs:
        args = json.load(open(f"{RUNS}/{r}/metrics.json"))["args"]
        cands = (args.get("ckpt"), f"{RUNS}/{r}/model.pt", args.get("out") and f"{args['out']}/model.pt")
        ckpt = next(p for p in cands if p and os.path.exists(p))
        net = load_net(args, ckpt)
        net.fc = torch.nn.Identity()   # the 2,048 pooled features
        nets[r] = net
    print(f"TRANSFER: {len(nets)} backbones", flush=True)
    for t in a.targets:
        t0 = time.time()
        (xtr, ytr), (xte, yte) = (images(s, a.workers) for s in TARGETS[t])
        if t == "in900":
            (mtr, ytr), (mte, yte) = in900_keep(ytr), in900_keep(yte)
            xtr, ytr, xte, yte = xtr[mtr], ytr[mtr], xte[mte], yte[mte]
        ncls = int(max(ytr.max(), yte.max())) + 1
        ftr, fte = features(nets, xtr), features(nets, xte)
        for r in nets:
            top1, per, l2 = probe(ftr[r], ytr, fte[r], yte, ncls)
            save(a.out, r, {t: dict(top1=top1, per_class=per, l2=l2, train=len(ytr), test=len(yte), classes=ncls)})
            print(f"{t} {r}: {100 * top1:.1f} (L2 {l2:g})", flush=True)
        print(f"TRANSFER {t}: {len(ytr)} train / {len(yte)} test images, {ncls} classes, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
