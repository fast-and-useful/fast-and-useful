"""The classifier recipe of Adamkiewicz et al. (CVPR 2026), reimplemented on our packs from the paper's description and
settings (their released code carries no license, so none of it is used): ResNet-50, 80 epochs, batch 1,024, Adam at
1e-3, five epochs of linear warm-up from 1e-5 of the rate, cosine annealing to a floor of 1e-8, stepped per batch, no
weight decay, gradient clipping at 1, CutMix on every batch (alpha 1); training images: random resized crop 224
(bicubic), then one of grayscale / solarize (threshold 0.5) / Gaussian blur (kernel 7, sigma 0.2-2), colour jitter 0.3,
horizontal flip; evaluation: resize 224 (bicubic), center crop 224; ImageNet normalization. The checkpoint with the
lowest loss on held-out images of the training domain is evaluated on the real test images.

Deviations: the held-out images are one in ten of each class's training images (every tenth record of a class, in pack
order), so the classifier trains on 450 images per class where the protocol has 500; augmentations on the 8-bit image
after the crop (theirs on float: equal up to rounding); bfloat16 autocast and channels-last memory (theirs float32).

Writes <out>/metrics.json as train.py does, plus the selected epoch and the per-epoch held-out loss.

Usage: python -m fau.train --config wp_kd_ad2 --seed 0        # dispatches here for the runs of this recipe
       python -m fau.train_prior --train SPEC [SPEC ...] --val VAL --out DIR --ncls 100 [--seed 0] [--accum 1]
"""
import argparse, copy, json, math, os, time

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import v2 as T

from fau.packs import PackSet, load_recs
from fau.train import MEAN, STD, autocast, build_net, build_sources, evaluate, expand


def split_sources(sources, every=10):
    """Every `every`-th record of each class (pack order) is held out for validation."""
    train, held = [], []
    for prefix, domain, keep in sources:
        seen, tr, ho = {}, set(), set()
        for i, r in enumerate(load_recs(prefix)):
            if keep is not None and i not in keep:
                continue
            c = r["label"]
            seen[c] = seen.get(c, 0) + 1
            (ho if seen[c] % every == 0 else tr).add(i)
        train.append((prefix, domain, tr)); held.append((prefix, domain, ho))
    return train, held


def lr_factor(step, total, warm, start=1e-5, floor=1e-8):
    if step < warm:
        return step / max(1, warm) * (1 - start) + start
    return max(floor, 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm))))


def val_loss(net, loader, device="cuda"):
    net.eval(); tot, n = 0.0, 0
    with torch.no_grad(), autocast(device):
        for x, y, _ in loader:
            out = net(x.to(device, non_blocking=True).to(memory_format=torch.channels_last)).float()
            tot += F.cross_entropy(out, y.to(device), reduction="sum").item(); n += len(y)
    net.train()
    return tot / n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--ncls", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--accum", type=int, default=1, help="micro-batches per step (memory); CutMix and BatchNorm per micro-batch")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=0, help="stop after N steps (smoke tests)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="as train.py")
    ap.add_argument("--resume_every", type=int, default=0, help="keep <out>/resume.pt every N epochs (0: never)")
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = True
    if a.device == "cpu":
        torch.set_num_threads(max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())) - a.workers))
    os.makedirs(a.out, exist_ok=True)
    bic = T.InterpolationMode.BICUBIC
    # their order (to float first, then crop and augment) up to rounding: crop and augment the 8-bit image (solarize at 128 =
    # their 0.5), 5x faster in the loader; batches stay 8-bit (4x less host memory) and are normalized on the GPU
    tr_tf = T.Compose([T.ToImage(), T.RandomResizedCrop(224, interpolation=bic, antialias=True),
                       T.RandomChoice([T.Grayscale(num_output_channels=3), T.RandomSolarize(threshold=128, p=1.0),
                                       T.GaussianBlur(kernel_size=7, sigma=(0.2, 2.0))]),
                       T.ColorJitter(0.3, 0.3, 0.3), T.RandomHorizontalFlip(0.5)])
    mean = torch.tensor(MEAN, device=a.device).view(1, 3, 1, 1) * 255
    std = torch.tensor(STD, device=a.device).view(1, 3, 1, 1) * 255
    ev_tf = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Resize(224, interpolation=bic, antialias=True),
                       T.CenterCrop(224), T.Normalize(MEAN, STD)])
    tr_src, ho_src = split_sources(build_sources(a.train))
    train, held = PackSet(tr_src, tr_tf), PackSet(ho_src, ev_tf)
    test = PackSet([(p, 0, None) for p in expand(a.val)], ev_tf)
    lab, dom = train.labels(), train.domains()
    counts = np.stack([np.bincount(lab[dom == d], minlength=a.ncls) for d in (0, 1)])
    micro = a.bs // a.accum
    pin = a.device == "cuda"
    dl = torch.utils.data.DataLoader(train, batch_size=micro, shuffle=True, num_workers=a.workers, pin_memory=pin, drop_last=True,
                                     persistent_workers=True, prefetch_factor=2)
    hl = torch.utils.data.DataLoader(held, batch_size=256, num_workers=a.workers, pin_memory=pin)
    tl = torch.utils.data.DataLoader(test, batch_size=256, num_workers=a.workers, pin_memory=pin)
    net = build_net("resnet50", a.ncls).to(a.device).to(memory_format=torch.channels_last)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    per_epoch = len(dl) // a.accum
    total, warm = a.epochs * per_epoch, a.warmup * per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, total, warm))
    cutmix = T.CutMix(num_classes=a.ncls)
    print(f"train {len(train)} images ({counts[0].sum()} real, {counts[1].sum()} generated), held out {len(held)}, test {len(test)}; "
          f"{total} steps of {a.bs} ({a.accum} x {micro})", flush=True)
    t0, curve, best, best_ep, step, start, before = time.time(), [], (float("inf"), None), 0, 0, 0, 0.0
    state = f"{a.out}/resume.pt"
    if a.resume_every and os.path.exists(state):
        st = torch.load(state, map_location=a.device, weights_only=False)
        net.load_state_dict(st["net"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        start, curve, best, best_ep, step, before = st["epoch"], st["curve"], st["best"], st["best_ep"], st["step"], st["minutes"]
        torch.set_rng_state(st["rng"]); np.random.set_state(st["np_rng"])
        print(f"resumed after epoch {start} ({before:.0f} min before)", flush=True)
    for ep in range(start, a.epochs):
        for k, (x, y, _) in enumerate(dl):
            x, y = cutmix((x.to(a.device, non_blocking=True).float() - mean) / std, y.to(a.device, non_blocking=True))
            with autocast(a.device):
                loss = F.cross_entropy(net(x.to(memory_format=torch.channels_last)).float(), y) / a.accum
            loss.backward()
            if (k + 1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), a.clip)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                if a.max_steps and step >= a.max_steps:
                    mem = torch.cuda.max_memory_allocated() / 2**30 if a.device == "cuda" else float("nan")
                    print(f"SMOKE: {step} steps, peak memory {mem:.1f} GB, "
                          f"{(time.time() - t0) / step:.2f} s/step", flush=True)
                    return
        vloss = val_loss(net, hl, a.device)
        curve.append((ep + 1, vloss))
        if vloss < best[0]:
            best, best_ep = (vloss, copy.deepcopy(net.state_dict())), ep + 1
        print(f"epoch {ep + 1}/{a.epochs}: train loss {loss.item() * a.accum:.3f}, held-out loss {vloss:.4f} (best epoch {best_ep}), "
              f"{before + (time.time() - t0) / 60:.1f} min", flush=True)
        if a.resume_every and (ep + 1) % a.resume_every == 0 and ep + 1 < a.epochs:
            torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=ep + 1, curve=curve,
                            best=best, best_ep=best_ep, step=step, minutes=before + (time.time() - t0) / 60,
                            rng=torch.get_rng_state(), np_rng=np.random.get_state()), state + ".tmp")
            os.replace(state + ".tmp", state)
    net.load_state_dict(best[1])
    acc, per = evaluate(net, tl, a.ncls, a.device)
    out = dict(top1=acc, per_class=per, curve=curve, best_epoch=best_ep, counts_real=counts[0].tolist(), counts_gen=counts[1].tolist(),
               args=dict(vars(a), recipe="prior", arch="resnet50"), minutes=before + (time.time() - t0) / 60)
    json.dump(out, open(f"{a.out}/metrics.json", "w"))
    ckpt = a.ckpt or f"{a.out}/model.pt"
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    torch.save(net.state_dict(), ckpt)
    if os.path.exists(state):
        os.remove(state)
    print(f"TRAIN DONE {a.out}: top-1 {acc:.4f} (checkpoint of epoch {best_ep})", flush=True)


if __name__ == "__main__":
    main()
