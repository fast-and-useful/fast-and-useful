"""The training-value probe: a classifier trained from scratch on packed image sets (packs.py), evaluated on real
validation images of the same classes. The paper's recipe (ResNet-50): 100 epochs, SGD with Nesterov momentum 0.9,
learning rate 0.2 at batch size 256, 5 warmup epochs then cosine decay, weight decay 1e-4, label smoothing 0.1, random
resized crops to 224 and flips, bf16 autocast and channels-last on one GPU; evaluation resizes to 256 and center-crops
224. The other learners (--arch convnext_tiny / vit_s16, with --opt adamw and --randaug) use the same loop.

Sources are PREFIX[:domain[:N]]: domain 0 real, 1 generated (recorded, not used by the loss); N caps each class (a
generated set keeps each class's N lowest image indices, so subsets are nested and every template is equally
represented; a real pack keeps each class's first N records). PREFIX names one pack or a directory of part*.json packs.

Writes <out>/metrics.json (top-1, per-class top-1, the validation curve, images per class and domain, the arguments)
and the weights to --ckpt (default <out>/model.pt).

Usage: python -m fau.train --config kd_ad2 --seed 0            # a run group of the paper (configs/runs.json)
       python -m fau.train --train SET:1 --val VAL --out DIR [--epochs 100] [--seed 0]
"""
import argparse, contextlib, json, math, os, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import v2 as T

from fau import REAL, ROOT, RUNS, SETS
from fau.packs import PackSet, load_recs

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def expand(prefix):
    """A prefix may name one pack, a directory of part*.json packs, or a <prefix>_part* family."""
    if os.path.isdir(prefix):
        return sorted(os.path.join(prefix, f[:-5]) for f in os.listdir(prefix) if f.startswith("part") and f.endswith(".json"))
    if not os.path.exists(prefix + ".json"):
        d, b = os.path.split(prefix)
        parts = sorted(os.path.join(d, f[:-5]) for f in os.listdir(d or ".") if f.startswith(b + "_part") and f.endswith(".json"))
        if parts:
            return parts
    return [prefix]


def build_sources(specs):
    """PREFIX[:domain[:N]] -> [(pack prefix, domain, record indices kept or None)]; N caps each class across the packs."""
    sources = []
    for s in specs:
        parts = s.split(":")
        prefix, domain = parts[0], int(parts[1]) if len(parts) > 1 else 0
        packs = expand(prefix)
        if len(parts) < 3:
            sources += [(p, domain, None) for p in packs]
            continue
        cap = int(parts[2])
        recs = [(p, i, r) for p in packs for i, r in enumerate(load_recs(p))]
        if recs and all("j" in r for _, _, r in recs):
            # A generated set keeps the N lowest image indices of each class: packs are written shard by shard, so the
            # first N records of a class can come from one shard and carry as few as two of the eight templates.
            recs.sort(key=lambda t: (t[2]["label"], t[2]["j"]))
        keep, seen = {p: set() for p in packs}, {}
        for p, i, r in recs:   # real packs (no image index): the first N records of each class, in pack order
            c = r["label"]
            if seen.get(c, 0) < cap:
                keep[p].add(i)
                seen[c] = seen.get(c, 0) + 1
        sources += [(p, domain, keep[p]) for p in packs]
    return sources


def source(spec):
    """A source of configs/runs.json ('set:<protocol>/<set>:...' or 'real:<pack>:...') -> PREFIX:... in $FAU_WORK."""
    kind, rest = spec.split(":", 1)
    return f"{SETS if kind == 'set' else REAL}/{rest}"


def from_config(group, seed):
    """The recorded run of a group of the paper (configs/runs.json) -> (trainer, arguments)."""
    import json
    cfg = json.loads((ROOT / "configs" / "runs.json").read_text())[group]
    assert "pooled" not in cfg, f"{group} pools the runs of {cfg.get('pooled')}: train those"
    out = RUNS / f"{group}_s{seed}"
    args = dict(train=[source(t) for t in cfg["train"]], val=str(REAL / cfg["val"]), ncls=cfg["ncls"], out=str(out),
                seed=seed, **cfg["recipe"])
    return cfg["trainer"], args


def resnet50(ncls):
    m = torchvision.models.resnet50(weights=None)
    m.fc = nn.Linear(2048, ncls)
    return m


def build_net(arch, ncls):
    """The learner, randomly initialized: resnet50 (the protocol), convnext_tiny, or vit_s16 (ViT-S/16, timm)."""
    if arch == "resnet50":
        return resnet50(ncls)
    if arch == "convnext_tiny":
        return torchvision.models.convnext_tiny(weights=None, num_classes=ncls)
    if arch == "vit_s16":
        import timm
        return timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=ncls)
    raise ValueError(arch)


def autocast(device):
    """bf16 autocast on the GPU; on CPUs without bf16 units plain float32."""
    return torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else contextlib.nullcontext()


def evaluate(net, loader, ncls, device="cuda"):
    net.eval(); correct = torch.zeros(ncls); total = torch.zeros(ncls)
    with torch.no_grad(), autocast(device):
        for x, y, _ in loader:
            pred = net(x.to(device, non_blocking=True).to(memory_format=torch.channels_last)).argmax(1).cpu()
            for c, ok in zip(y.tolist(), (pred == y).tolist()):
                total[c] += 1; correct[c] += ok
    net.train()
    return (correct.sum() / total.sum()).item(), (correct / total.clamp(min=1)).tolist()


def argv_of(args):
    """{argument: value} -> a command line for these parsers (lists as several values, True flags as flags)."""
    out = []
    for k, v in args.items():
        if v is False or v is None:
            continue
        out += [f"--{k}"] + ([] if v is True else [str(x) for x in v] if isinstance(v, list) else [str(v)])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="a run group of the paper (configs/runs.json): its recorded run with --seed")
    ap.add_argument("--train", nargs="+")
    ap.add_argument("--val")
    ap.add_argument("--out")
    ap.add_argument("--ckpt", default=None, help="where the weights go (default <out>/model.pt)")
    ap.add_argument("--ncls", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--ls", type=float, default=0.1)
    ap.add_argument("--arch", default="resnet50", choices=["resnet50", "convnext_tiny", "vit_s16"])
    ap.add_argument("--opt", default="sgd", choices=["sgd", "adamw"], help="adamw for ConvNeXt-T and ViT-S/16")
    ap.add_argument("--randaug", action="store_true", help="RandAugment (2, 9) before the crop, as in DeiT-style recipes")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                    help="cpu: float32, torch threads = SLURM_CPUS_PER_TASK (or all cores) - workers")
    ap.add_argument("--resume_every", type=int, default=0, help="keep <out>/resume.pt every N epochs (0: never)")
    a = ap.parse_args(argv)
    if a.config:   # the recorded run, with this invocation's workers, device and checkpointing
        trainer, args = from_config(a.config, a.seed)
        argv = argv_of(dict(args, workers=a.workers, device=a.device, resume_every=a.resume_every or None))
        if trainer == "prior":
            from fau import train_prior
            return train_prior.main(argv)
        a = ap.parse_args(argv)
    assert a.train and a.val and a.out, "give --config and --seed, or --train, --val and --out"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = True
    if a.device == "cpu":
        torch.set_num_threads(max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())) - a.workers))
    os.makedirs(a.out, exist_ok=True)
    tr_tf = T.Compose([T.RandomResizedCrop(224, antialias=True), T.RandomHorizontalFlip()] + ([T.RandAugment(2, 9)] if a.randaug else []) +
                      [T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize(MEAN, STD)])
    ev_tf = T.Compose([T.Resize(256, antialias=True), T.CenterCrop(224), T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize(MEAN, STD)])
    train = PackSet(build_sources(a.train), tr_tf)
    val = PackSet([(p, 0, None) for p in expand(a.val)], ev_tf)
    lab, dom = train.labels(), train.domains()
    counts = np.stack([np.bincount(lab[dom == d], minlength=a.ncls) for d in (0, 1)])
    pin = a.device == "cuda"
    dl = torch.utils.data.DataLoader(train, batch_size=a.bs, shuffle=True, num_workers=a.workers, pin_memory=pin, drop_last=True,
                                     persistent_workers=True, prefetch_factor=4)
    vl = torch.utils.data.DataLoader(val, batch_size=256, num_workers=a.workers, pin_memory=pin)
    net = build_net(a.arch, a.ncls).to(a.device).to(memory_format=torch.channels_last)
    split = {}   # weight decay on weights, none on biases and normalization parameters
    for n, p in net.named_parameters():
        split.setdefault(p.ndim > 1, []).append(p)
    groups = [dict(params=ps, lr=a.lr, weight_decay=a.wd if dec else 0) for dec, ps in split.items()]
    opt = torch.optim.SGD(groups, lr=a.lr, momentum=0.9, nesterov=True) if a.opt == "sgd" else torch.optim.AdamW(groups, lr=a.lr)
    steps, wsteps = a.epochs * len(dl), a.warmup * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / wsteps if s < wsteps else 0.5 * (1 + math.cos(math.pi * (s - wsteps) / (steps - wsteps))))
    print(f"train {len(train)} images ({counts[0].sum()} real, {counts[1].sum()} generated), val {len(val)}; {steps} steps "
          f"on {a.device}", flush=True)
    t0, curve, start, before = time.time(), [], 0, 0.0
    state = f"{a.out}/resume.pt"
    if a.resume_every and os.path.exists(state):
        st = torch.load(state, map_location=a.device, weights_only=False)
        net.load_state_dict(st["net"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        start, curve, before = st["epoch"], st["curve"], st["minutes"]
        torch.set_rng_state(st["rng"]); np.random.set_state(st["np_rng"])
        print(f"resumed after epoch {start} ({before:.0f} min before)", flush=True)
    for ep in range(start, a.epochs):
        for x, y, _ in dl:
            x = x.to(a.device, non_blocking=True).to(memory_format=torch.channels_last); y = y.to(a.device, non_blocking=True)
            with autocast(a.device):
                loss = F.cross_entropy(net(x).float(), y, label_smoothing=a.ls)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if (ep + 1) % 10 == 0 or ep + 1 == a.epochs:
            acc, per = evaluate(net, vl, a.ncls, a.device)
            curve.append((ep + 1, acc))
            print(f"epoch {ep + 1}/{a.epochs}: loss {loss.item():.3f}, val top-1 {acc:.4f}, "
                  f"{before + (time.time() - t0) / 60:.1f} min", flush=True)
        if a.resume_every and (ep + 1) % a.resume_every == 0 and ep + 1 < a.epochs:
            torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=ep + 1, curve=curve,
                            minutes=before + (time.time() - t0) / 60, rng=torch.get_rng_state(), np_rng=np.random.get_state()),
                       state + ".tmp")
            os.replace(state + ".tmp", state)
    out = dict(top1=acc, per_class=per, curve=curve, counts_real=counts[0].tolist(), counts_gen=counts[1].tolist(), args=vars(a),
               minutes=before + (time.time() - t0) / 60)
    json.dump(out, open(f"{a.out}/metrics.json", "w"))
    ckpt = a.ckpt or f"{a.out}/model.pt"
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    torch.save(net.state_dict(), ckpt)
    if os.path.exists(state):
        os.remove(state)
    print(f"TRAIN DONE {a.out}: top-1 {acc:.4f}", flush=True)


if __name__ == "__main__":
    main()
