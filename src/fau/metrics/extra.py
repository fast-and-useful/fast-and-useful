"""The other set metrics of the metric study, besides recall and coverage in DINOv3 space. One pass over a set's images (each decoded once) feeds three networks:

  CLIP ViT-L/14 (openai)   projected image embedding (768-d; resize 224 bicubic, center crop, CLIP normalization), cached;
                           then pairwise, Vendi, precision, recall, density, coverage (per class, k = 5) and the Frechet
                           distance of the whole set to the real reference, with diversity.py's functions, exactly as in
                           DINOv3 space (L2-normalized features)
  Inception-v3 (TF FID)    pool3 features (2048-d) with the FID weights (pytorch-fid's pt_inception-2015-12-05; the network
                           is reimplemented below), cached; FID of the whole set against the real reference, and the
                           Inception Score (1008 logits without the bias, as the original code; 10 splits, shuffled)
  ResNet-50 (ImageNet-1k)  torchvision IMAGENET1K_V2, outputs cached; recognizability: top-1 among the set's 100 classes
                           (`recog`, the headline number), top-1 among all 1,000 (`recog1k`), and the mean probability of
                           the intended class among the 100 (`conf`) and among the 1,000 (`conf1k`)

Every image is resized (short side) and center-cropped, never squashed; the packs store copies with the short side at 256
(stats.py), so the Inception input is upsampled 256 -> 299 with an antialiased bicubic filter. Inception and ResNet run
in fp32 without TF32. FID is computed from the cached fp16 features on every path (first pass and re-reads alike), with
diversity.py's Frechet function (unbiased covariances, the trace of the square root through the eigenvalues); --check
compares it with scipy's sqrtm and with fp32 features.

The real reference is real500 (500 real training images per class), as in diversity.py. Array task 0 also writes the
reference rows: real500 (CLIP pairwise and Vendi, Inception Score, recognizability, which is inflated: the classifier was
trained on these images) and in100_val (FID of the 5,000 real validation images against real500, Inception Score,
recognizability). Results are merged into OUT_DIR/{clip_vitl14,fid_inception,recognizability}.json under a lock, keyed by
set name; parallel array tasks share the files.

Usage: python -m fau.metrics.extra --sets SPEC [SPEC ...] --real real500=work/real/in100_train:0:500
           --val work/real/in100_val --out-dir work/probes --cache DIR [--chunk I/N] [--check]
(SPEC is [NAME=]PREFIX[:domain[:N]] as in train.py; NAME defaults to the prefix's basename)
"""
import argparse, fcntl, json, os, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.inception import InceptionA, InceptionC, InceptionE
from torchvision.transforms import v2 as T

from fau.metrics.diversity import frechet, pairwise, per_class, prdc, vendi
from fau.devices import DEV, WORKERS, half
from fau.hub import snapshot
from fau.prompts import class_list
from fau.packs import PackSet
from fau.train import build_sources

CLIP_L = "openai/clip-vit-large-patch14"
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
IMNET_MEAN, IMNET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
FID_WEIGHTS = "pt_inception-2015-12-05-6726825d.pth"   # pytorch-fid's release; fetched into $TORCH_HOME/hub/checkpoints
FID_URL = "https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/" + FID_WEIGHTS
CACHES = dict(clip="clip_vitl14", inception="inception_fid", cls="resnet50_in1k_v2")   # feature cache subdirectories
OUTS = dict(clip="clip_vitl14.json", fid="fid_inception.json", recog="recognizability.json")
BICUBIC = T.InterpolationMode.BICUBIC
VIEWS = (   # uint8 from the workers (shared memory), scaled and normalized on the GPU
    T.Compose([T.Resize(224, interpolation=BICUBIC, antialias=True), T.CenterCrop(224), T.ToImage()]),   # CLIP
    T.Compose([T.Resize(299, interpolation=BICUBIC, antialias=True), T.CenterCrop(299), T.ToImage()]),   # Inception
    T.Compose([T.Resize(232, antialias=True), T.CenterCrop(224), T.ToImage()]),   # ResNet-50 V2's own eval transform
)


class Views:
    def __call__(self, img):
        return tuple(tf(img) for tf in VIEWS)


# Inception-v3 as the TF FID code has it (the 2015-12-05 graph), after pytorch-fid's port: torchvision's Inception3 with
# 1008 classes, and the TF graph's pooling in the Mixed blocks' pool branches (average pooling without the padded zeros;
# max pooling in the last block).
class _FIDInceptionA(InceptionA):
    def forward(self, x):
        b5 = self.branch5x5_2(self.branch5x5_1(x))
        bd = self.branch3x3dbl_3(self.branch3x3dbl_2(self.branch3x3dbl_1(x)))
        bp = self.branch_pool(F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False))
        return torch.cat([self.branch1x1(x), b5, bd, bp], 1)


class _FIDInceptionC(InceptionC):
    def forward(self, x):
        b7 = self.branch7x7_3(self.branch7x7_2(self.branch7x7_1(x)))
        bd = x
        for m in (self.branch7x7dbl_1, self.branch7x7dbl_2, self.branch7x7dbl_3, self.branch7x7dbl_4, self.branch7x7dbl_5):
            bd = m(bd)
        bp = self.branch_pool(F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False))
        return torch.cat([self.branch1x1(x), b7, bd, bp], 1)


class _FIDInceptionE(InceptionE):
    def __init__(self, in_channels, pool_kind):
        super().__init__(in_channels)
        self.pool_kind = pool_kind   # "avg" in Mixed_7b; "max" in Mixed_7c, as in the TF graph

    def forward(self, x):
        b3 = self.branch3x3_1(x)
        b3 = torch.cat([self.branch3x3_2a(b3), self.branch3x3_2b(b3)], 1)
        bd = self.branch3x3dbl_2(self.branch3x3dbl_1(x))
        bd = torch.cat([self.branch3x3dbl_3a(bd), self.branch3x3dbl_3b(bd)], 1)
        bp = (F.max_pool2d(x, kernel_size=3, stride=1, padding=1) if self.pool_kind == "max" else
              F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False))
        return torch.cat([self.branch1x1(x), b3, bd, self.branch_pool(bp)], 1)


class FIDInception(nn.Module):
    """x in [0, 1] at 299 x 299 -> (pool3 features [N, 2048], the 1008 logits with the bias)."""
    BLOCKS = ("Mixed_5b", "Mixed_5c", "Mixed_5d", "Mixed_6a", "Mixed_6b", "Mixed_6c", "Mixed_6d", "Mixed_6e", "Mixed_7a",
              "Mixed_7b", "Mixed_7c")

    def __init__(self, weights):
        super().__init__()
        n = torchvision.models.inception_v3(weights=None, num_classes=1008, aux_logits=False, init_weights=False)
        n.Mixed_5b, n.Mixed_5c, n.Mixed_5d = (_FIDInceptionA(192, pool_features=32), _FIDInceptionA(256, pool_features=64),
                                              _FIDInceptionA(288, pool_features=64))
        n.Mixed_6b, n.Mixed_6c, n.Mixed_6d, n.Mixed_6e = (_FIDInceptionC(768, channels_7x7=c) for c in (128, 160, 160, 192))
        n.Mixed_7b, n.Mixed_7c = _FIDInceptionE(1280, "avg"), _FIDInceptionE(2048, "max")
        n.load_state_dict(torch.load(weights, map_location="cpu"))   # strict: every tensor present and of its shape
        self.net = n

    def forward(self, x):
        n = self.net
        x = n.Conv2d_2b_3x3(n.Conv2d_2a_3x3(n.Conv2d_1a_3x3(2 * x - 1)))
        x = n.Conv2d_4a_3x3(n.Conv2d_3b_1x1(F.max_pool2d(x, kernel_size=3, stride=2)))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        for b in self.BLOCKS:
            x = getattr(n, b)(x)
        f = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return f, n.fc(f)


class Nets:
    def __init__(self, inception_weights):
        from transformers import CLIPModel
        self.clip = CLIPModel.from_pretrained(snapshot(CLIP_L), dtype=half(torch.float16), disable_mmap=True).to(DEV).eval()
        self.inception = FIDInception(inception_weights).to(DEV).eval()
        self.cls = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2).to(DEV).eval()
        self.cmean, self.cstd = (torch.tensor(v, device=DEV).view(1, 3, 1, 1) for v in (CLIP_MEAN, CLIP_STD))
        self.imean, self.istd = (torch.tensor(v, device=DEV).view(1, 3, 1, 1) for v in (IMNET_MEAN, IMNET_STD))


@torch.no_grad()
def extract(nets, spec, idx100, workers):
    ds = PackSet(build_sources([spec]), Views())
    dl = torch.utils.data.DataLoader(ds, batch_size=200, num_workers=workers, pin_memory=DEV == "cuda")
    out = {k: [] for k in ("clip", "pool3", "logits100", "lse", "pred1k")}
    for (xc, xi, xr), _, _ in dl:
        xc = ((xc.to(DEV, non_blocking=True).float() / 255 - nets.cmean) / nets.cstd).to(half(torch.float16))
        o = nets.clip.get_image_features(pixel_values=xc)
        out["clip"].append((o.pooler_output if hasattr(o, "pooler_output") else o).float().cpu())   # projected embedding
        out["pool3"].append(nets.inception(xi.to(DEV, non_blocking=True).float() / 255)[0].cpu())
        lg = nets.cls((xr.to(DEV, non_blocking=True).float() / 255 - nets.imean) / nets.istd).float()
        out["logits100"].append(lg[:, idx100].cpu()); out["lse"].append(lg.logsumexp(1).cpu()); out["pred1k"].append(lg.argmax(1).cpu())
    out = {k: torch.cat(v) for k, v in out.items()}
    assert out["clip"].shape[1] == 768 and out["pool3"].shape[1] == 2048, (out["clip"].shape, out["pool3"].shape)
    return out, torch.from_numpy(ds.labels().copy())


def features(nets, name, spec, cache, idx100, workers):
    """(CLIP fp16, pool3 fp16, classifier outputs, labels, pool3 fp32 or None): the three caches, all computed in one pass
    if any is missing."""
    paths = {k: f"{cache}/{d}/{name}.pt" for k, d in CACHES.items()}
    if all(os.path.exists(p) for p in paths.values()):
        (clip, y), (pool3, _), cls = torch.load(paths["clip"]), torch.load(paths["inception"]), torch.load(paths["cls"])
        return clip, pool3, cls, y, None
    f, y = extract(nets, spec, idx100, workers)
    clip, pool3 = f["clip"].half(), f["pool3"].half()
    cls = dict(logits100=f["logits100"], lse=f["lse"], pred1k=f["pred1k"].short())
    for k, obj in (("clip", (clip, y)), ("inception", (pool3, y)), ("cls", cls)):
        os.makedirs(os.path.dirname(paths[k]), exist_ok=True)
        tmp = f"{paths[k]}.{os.getpid()}.tmp"
        torch.save(obj, tmp); os.replace(tmp, paths[k])
    return clip, pool3, cls, y, f["pool3"]


def clip_metrics(x, y, real, xr, ncls, reference):
    """As diversity.main: L2-normalized features; per class against the real class, then averaged; Frechet over the set."""
    x = F.normalize(x.float().to(DEV), dim=1); y = y.to(DEV)
    fake = per_class(x, y, ncls)
    m = dict(pairwise=torch.stack([pairwise(f) for f in fake]).mean().item(), vendi=torch.stack([vendi(f) for f in fake]).mean().item())
    if not reference:
        p = [prdc(r, f) for r, f in zip(real, fake)]
        m.update({k: sum(d[k] for d in p) / len(p) for k in p[0]})
        m["frechet"] = frechet(x.double(), xr.double()).item()
    counts = torch.bincount(y, minlength=ncls)
    return dict(m, encoder="clip_vitl14", n=len(y), per_class_n=[counts.min().item(), counts.max().item()])


def inception_score(pool3, fc_weight, splits=10, seed=0):
    p = (pool3.float().to(DEV) @ fc_weight.T).softmax(1).double()   # logits without the bias, as the original IS code
    p = p[torch.randperm(len(p), generator=torch.Generator().manual_seed(seed)).to(DEV)]   # packs are ordered by class
    s = [(part * (part.clamp_min(1e-30).log() - part.mean(0, keepdim=True).log())).sum(1).mean().exp().item() for part in p.chunk(splits)]
    return float(np.mean(s)), float(np.std(s))


def fid_metrics(pool3, ref_pool3, fc_weight, reference):
    m = {}
    if not reference:
        m["fid"] = frechet(pool3.double().to(DEV), ref_pool3.double().to(DEV)).item()
    m["inception_score"], m["inception_score_sd"] = inception_score(pool3, fc_weight)
    return dict(m, weights=FID_WEIGHTS, n=len(pool3))


def recog_metrics(cls, y, idx100, ncls):
    lg, y = cls["logits100"].float(), y.long()
    i = torch.arange(len(y))
    ok = (lg.argmax(1) == y).float()
    per = torch.zeros(ncls).index_add_(0, y, ok) / torch.bincount(y, minlength=ncls).clamp(min=1)
    return dict(recog=ok.mean().item(), recog1k=(cls["pred1k"].long() == torch.tensor(idx100)[y]).float().mean().item(),
                conf=lg.softmax(1)[i, y].mean().item(), conf1k=(lg[i, y] - cls["lse"]).exp().mean().item(),
                per_class=[round(v, 4) for v in per.tolist()], classifier="torchvision resnet50 IMAGENET1K_V2", n=len(y))


def save(path, name, entry):
    """Merges one set into the results file. Parallel array tasks share it: read-modify-write under a lock."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        res = json.load(open(path)) if os.path.exists(path) else {}
        res[name] = entry
        with open(path + ".tmp", "w") as f:
            json.dump(res, f, indent=1)
        os.replace(path + ".tmp", path)


def done(path, name):
    return os.path.exists(path) and name in json.load(open(path))


def class_index(classes):
    """ImageNet-1k indices of the sets' classes (label order), checked against the 1,000 classes in label order."""
    c, full = class_list(classes), class_list("in1k")["wnids"]
    idx = c["imagenet_index"]
    assert full == sorted(full) and len(full) == 1000 and all(full[i] == x for i, x in zip(idx, c["wnids"])), "class index mismatch"
    return idx


def check(name, pool3_32, pool3, ref_pool3):
    """Smoke checks of the FID path: fp16 cache vs fp32 features; the eigenvalue route vs scipy's sqrtm (pytorch-fid's)."""
    from scipy import linalg
    a16 = frechet(pool3.double().to(DEV), ref_pool3.double().to(DEV)).item()
    a32 = frechet(pool3_32.double().to(DEV), ref_pool3.double().to(DEV)).item()
    x, r = pool3.double().numpy(), ref_pool3.double().numpy()
    s1, s2 = np.cov(x, rowvar=False), np.cov(r, rowvar=False)
    cm = linalg.sqrtm(s1 @ s2)
    cm = cm[0] if isinstance(cm, tuple) else cm   # SciPy < 1.16 with disp=False returned (sqrtm, error estimate)
    d = x.mean(0) - r.mean(0)
    sp = float(d @ d + np.trace(s1) + np.trace(s2) - 2 * np.trace(cm.real))
    print(f"CHECK {name}: FID fp16 cache {a16:.4f}, fp32 features {a32:.4f} (delta {a32 - a16:+.4f}), scipy sqrtm {sp:.4f} "
          f"(delta {sp - a16:+.4f})", flush=True)
    assert abs(a32 - a16) < 0.05 and abs(sp - a16) < 0.05, f"FID check failed for {name}"   # fails the smoke job (afterok gate)


def purity(name, pool3, y, net, ncls):
    """Real images through the FID Inception's own 1008-way head: per class, the share of images on the class's most
    frequent prediction, and how many distinct predictions the classes' majorities take (100 = one per class)."""
    pred = net.fc(pool3.float().to(DEV)).argmax(1).cpu()
    maj, share = [], []
    for c in range(ncls):
        v = torch.bincount(pred[y == c], minlength=1008)
        maj.append(v.argmax().item()); share.append((v.max() / v.sum()).item())
    print(f"CHECK {name}: FID Inception 1008-way majority purity {np.mean(share):.3f}, {len(set(maj))} distinct majority "
          f"classes for {ncls} classes", flush=True)
    assert np.mean(share) > 0.6 and len(set(maj)) > 0.9 * ncls, "FID Inception does not classify the real images"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="*", default=[])
    ap.add_argument("--chunk", default="0/1", help="I/N: this task measures sets I, I+N, I+2N, ... (array task I of N)")
    ap.add_argument("--real", required=True, help="NAME=SPEC of the real reference images")
    ap.add_argument("--val", default=None, help="real validation images: the reference rows' FID and recognizability (task 0)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cache", required=True, help="feature cache directory (cheap to recreate: scratch)")
    ap.add_argument("--inception", default=None, help=f"FID weights (default: {FID_URL}, cached by torch.hub)")
    ap.add_argument("--classes", default="in100", help="the sets' class list")
    ap.add_argument("--ncls", type=int, default=100)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--force", action="store_true", help="recompute sets already in the results files")
    ap.add_argument("--check", action="store_true", help="smoke checks: FID by scipy and from fp32 features, Inception purity")
    a = ap.parse_args()
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    i, n = map(int, a.chunk.split("/"))
    if a.inception is None:
        torch.hub.load_state_dict_from_url(FID_URL, progress=False)   # downloads once into the torch hub cache
        a.inception = os.path.join(torch.hub.get_dir(), "checkpoints", FID_WEIGHTS)
    weights = a.inception
    idx100 = class_index(a.classes)
    outs = {k: f"{a.out_dir}/{f}" for k, f in OUTS.items()}
    nets = Nets(weights)
    fc_w = nets.inception.net.fc.weight.detach()

    rname, rspec = a.real.split("=", 1)
    t0 = time.time()
    rclip, rpool3, rcls, ry, rpool3_32 = features(nets, rname, rspec, a.cache, idx100, a.workers)
    xr = F.normalize(rclip.float().to(DEV), dim=1)
    real = per_class(xr, ry.to(DEV), a.ncls)
    print(f"REFERENCE {rname}: {len(ry)} images ({time.time() - t0:.0f} s)", flush=True)
    if a.check:
        purity(rname, rpool3, ry, nets.inception.net, a.ncls)

    todo = []   # (name, spec, kind): kind None for a generated set, else the reference row it is
    if i == 0:
        todo.append((rname, rspec, "train"))
        if a.val:
            todo.append((os.path.basename(a.val.rstrip("/")), a.val, "val"))
    for s in a.sets[i::n]:
        name, spec = s.split("=", 1) if "=" in s else (os.path.basename(s.split(":")[0].rstrip("/")), s)
        todo.append((name, spec, None))
    print(f"TASK {i}/{n}: {len(todo)} sets", flush=True)

    for name, spec, kind in todo:
        if not a.force and all(done(p, name) for p in outs.values()):
            print(f"SKIP {name}: in every results file", flush=True)
            continue
        t0 = time.time()
        clip, pool3, cls, y, pool3_32 = ((rclip, rpool3, rcls, ry, rpool3_32) if kind == "train" else
                                         features(nets, "_val_" + name if kind == "val" else name, spec, a.cache, idx100, a.workers))
        ref = kind is not None
        if kind != "val":   # CLIP diversity of the validation images is not needed
            save(outs["clip"], name, clip_metrics(clip, y, real, xr, a.ncls, ref))
        fm = fid_metrics(pool3, rpool3, fc_w, kind == "train")
        save(outs["fid"], name, dict(fm, ref=rname))
        rm = recog_metrics(cls, y, idx100, a.ncls)
        save(outs["recog"], name, dict(rm, classifier_trained_on_these_images=kind == "train"))
        if a.check and pool3_32 is not None and kind != "train":
            check(name, pool3_32, pool3, rpool3)
        cm = json.load(open(outs["clip"])).get(name, {}) if kind != "val" else {}
        print(f"METRICS {name}: CLIP recall {cm.get('recall', float('nan')):.4f}, coverage {cm.get('coverage', float('nan')):.4f}, "
              f"vendi {cm.get('vendi', float('nan')):.2f}; FID {fm.get('fid', float('nan')):.2f}, IS {fm['inception_score']:.2f}; "
              f"recog {rm['recog']:.4f} (1k {rm['recog1k']:.4f}) ({len(y)} images, {time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
