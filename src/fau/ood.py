"""Out-of-distribution test sets for the classifiers of train.py, and their top-1 on them. ImageNet-V2, -Sketch,
-R and -A become packs (packs.py) restricted to a class list (default ImageNet-100), labels re-indexed to its order.

Sources (Hugging Face hub, revisions pinned in SOURCES, cached in $HF_HUB_CACHE), and how each one names its classes:
  v2      vaishaal/ImageNetV2, imagenetv2-matched-frequency.tar.gz (an uncompressed tar): 10,000 photos, 10 for each of
          the 1000 classes, in imagenetv2-matched-frequency-format-val/<k>/ with k the ImageNet-1k index in torchvision
          order (sorted wnids, which is the order of ILSVRC/imagenet-1k's classes.py: asserted). The one source without
          wnids in the data.
  sketch  songweig/imagenet_sketch, data/ImageNet-Sketch.zip: 50,889 sketches of the 1000 classes, sketch/<wnid>/*.JPEG
          (the HF loader's filter).
  r       axiong/imagenet-r, parquet: 30,000 renditions of 200 classes, column wnid.
  a       danjacobellis/imagenet-a, parquet: 7,500 adversarially filtered photos of 200 classes, column path
          (imagenet-a/<wnid>/<file>, the original tar layout) and label (ImageNet-1k index), asserted to agree.
Each image goes to its wnid, then to the wnid's position in the class list; other classes are left out, so R and A
(200 classes each) cover only part of a subset list.

pack: <out-dir>/<name>_ood_<set>.{bin,json} per set, name from the class list (in100, in1k). The
original image bytes (evaluation resizes); record metadata wnid, src (path in the source) and, where the source has
one, src_label (V2's folder index, A's ImageNet-1k label). Sidecar <prefix>_classes.json: the class list, the source,
images per class. Images PIL cannot decode are dropped and reported.
eval: per run (work/runs/<run>/metrics.json: arch, ncls, and the validation pack whose basename minus _val names the
class list), top-1 on every OOD pack of its class list and on its own validation pack (set val: must reproduce
metrics.json, a check of the weights and the transform). The argmax runs over all of the classifier's classes; sets
covering fewer classes also get top1_restricted, the argmax over the covered classes only. Evaluation transform of
train.py (resize 256, center crop 224, ImageNet mean/std; the crops are decoded once per pack as uint8 and
normalized on the GPU), bf16 autocast. Merged into --out as {run: {set: {top1, n, classes, per_class}}}.
Weights: args.ckpt or model.pt in the run directory.

Usage: python -m fau.ood download
       python -m fau.ood pack [--classes in100] [--sets v2 sketch r a]
       python -m fau.ood eval --runs RUN [RUN ...] [--out work/probes/ood.json]
"""
import argparse, fcntl, glob, io, json, os, tarfile, time, zipfile
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as TF

from fau import PROBES, REAL, RUNS
from fau.imagenet import imagenet_classes
from fau.packs import PackSet, PackWriter
from fau.prompts import class_list
from fau.train import MEAN, STD, build_net, expand

SOURCES = {   # set: repo, revision, files, (images, classes) in the source
    "v2": ("vaishaal/ImageNetV2", "d626240be2538720e83103a0e1178d24aca8b12c", ["imagenetv2-matched-frequency.tar.gz"], (10000, 1000)),
    "sketch": ("songweig/imagenet_sketch", "0d8b8b5380efe50baae075842505918ad43c88fe", ["data/ImageNet-Sketch.zip"], (50889, 1000)),
    "r": ("axiong/imagenet-r", "762a23f169e080bef0be7c9d96bf73a126cbb660", ["test/*.parquet"], (30000, 200)),
    "a": ("danjacobellis/imagenet-a", "c6fdab0b640eb713485d95b5701a55b85d1bda88", ["data/*.parquet"], (7500, 200)),
}


def local(s):
    """The source's snapshot directory (downloads it unless HF_HUB_OFFLINE). Offline, a snapshot directory can exist
    without its data files (e.g. only README.md fetched), hence the check."""
    repo, rev, files, _ = SOURCES[s]
    d = snapshot_download(repo, repo_type="dataset", revision=rev, allow_patterns=files)
    missing = [f for f in files if not glob.glob(f"{d}/{f}")]
    assert not missing, f"{repo}@{rev[:7]}: {missing} not in {d} (run the download mode first)"
    return d


# Readers yield (wnid, src, src_label, bytes) for every image of the source; bytes only for wnids in keep, else None.
def read_v2(d, keep, in1k):
    with tarfile.open(f"{d}/imagenetv2-matched-frequency.tar.gz", "r|*") as t:   # a plain tar despite the name
        for m in t:
            k = os.path.basename(os.path.dirname(m.name))
            if m.isfile() and k.isdigit() and not os.path.basename(m.name).startswith("."):
                w = in1k[int(k)]
                yield w, m.name, int(k), t.extractfile(m).read() if w in keep else None


def read_sketch(d, keep, in1k):
    with zipfile.ZipFile(f"{d}/data/ImageNet-Sketch.zip") as z:
        for i in z.infolist():   # as the HF loader: *.JPEG, hidden or __ files and folders skipped
            if i.filename.endswith(".JPEG") and not any(c.startswith((".", "__")) for c in i.filename.split("/")):
                w = os.path.basename(os.path.dirname(i.filename)).lower()
                yield w, i.filename, None, z.read(i) if w in keep else None


def read_r(d, keep, in1k):
    for f in sorted(glob.glob(f"{d}/test/*.parquet")):
        t = pq.read_table(f, columns=["image", "wnid"])
        for j, (img, w) in enumerate(zip(t.column("image").to_pylist(), t.column("wnid").to_pylist())):
            yield w, img.get("path") or f"{os.path.basename(f)}:{j}", None, img["bytes"] if w in keep else None


def read_a(d, keep, in1k):
    for f in sorted(glob.glob(f"{d}/data/*.parquet")):
        t = pq.read_table(f, columns=["image", "path", "label"])
        for img, p, lab in zip(*(t.column(c).to_pylist() for c in ("image", "path", "label"))):
            w = p.split("/")[-2]
            assert in1k[lab] == w, f"ImageNet-A: label {lab} ({in1k[lab]}) disagrees with the path {p}"
            yield w, p, lab, img["bytes"] if w in keep else None


READERS = dict(v2=read_v2, sketch=read_sketch, r=read_r, a=read_a)


def pack_one(job):
    s, classes, prefix = job
    repo, rev, _, (n_src, k_src) = SOURCES[s]
    cl = class_list(classes)
    pos = {w: k for k, w in enumerate(cl["wnids"])}
    in1k = [w for w, _ in imagenet_classes()]
    d, t0 = local(s), time.time()
    w, counts, seen, bad = PackWriter(prefix), np.zeros(len(pos), dtype=int), Counter(), []
    for wn, src, label, data in READERS[s](d, pos, in1k):
        seen[wn] += 1
        if wn not in pos:
            continue
        assert data, f"{s}: no image bytes for {src}"
        try:
            Image.open(io.BytesIO(data)).convert("RGB")   # decoded as PackSet will: fail here, not in an eval worker
        except Exception as e:
            bad.append(f"{src} ({type(e).__name__})")
            continue
        w.add(data, pos[wn], wnid=wn, src=src, **({} if label is None else dict(src_label=label)))
        counts[pos[wn]] += 1
    n = sum(seen.values())
    assert (n, len(seen)) == (n_src, k_src), f"{s}: {n} images of {len(seen)} classes in the source, expected {n_src} of {k_src}"
    assert set(seen) <= set(in1k), f"{s}: wnids outside ImageNet-1k: {sorted(set(seen) - set(in1k))[:5]}"
    side = dict(cl, set=s, repo=repo, revision=rev, counts=counts.tolist(), dropped=bad)
    with open(prefix + "_classes.json.tmp", "w") as f:
        json.dump(side, f, indent=1)
    os.replace(prefix + "_classes.json.tmp", prefix + "_classes.json")
    w.close()   # the pack appears last, with its sidecar already in place
    return (f"PACK {s}: {counts.sum()} images of {(counts > 0).sum()}/{len(pos)} classes (source {repo}@{rev[:7]}: {n} images of "
            f"{len(seen)} classes), {len(bad)} undecodable{': ' + ', '.join(bad) if bad else ''} -> {prefix} ({time.time() - t0:.0f} s)")


def crops(prefixes, workers):
    """Images through train.py's evaluation transform up to the uint8 crop; ToDtype and Normalize follow on the GPU."""
    ds = PackSet([(p, 0, None) for p in prefixes], T.Compose([T.Resize(256, antialias=True), T.CenterCrop(224), T.ToImage()]))
    dl = torch.utils.data.DataLoader(ds, batch_size=256, num_workers=workers)
    return torch.cat([x for x, _, _ in dl]).as_subclass(torch.Tensor), torch.from_numpy(ds.labels().copy())


@torch.no_grad()
def predict(net, x, bs=512):
    out = []
    for i in range(0, len(x), bs):
        b = TF.normalize(TF.to_dtype(x[i:i + bs].cuda(), torch.float32, scale=True), MEAN, STD)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(net(b.to(memory_format=torch.channels_last)).float().cpu())
    return torch.cat(out)


def load_net(args, ckpt):
    sd = torch.load(ckpt, map_location="cpu")
    net = build_net(args.get("arch", "resnet50"), args["ncls"])
    net.load_state_dict(sd)
    return net.cuda().to(memory_format=torch.channels_last).eval()


def save(path, run, res):
    """Merges one run into the results file. Parallel eval jobs share it: read-modify-write under a lock."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        allres = json.load(open(path)) if os.path.exists(path) else {}
        allres.setdefault(run, {}).update(res)
        with open(path + ".tmp", "w") as f:
            json.dump(allres, f, indent=1)
        os.replace(path + ".tmp", path)


def evaluate(a):
    info, problems = {}, []
    for run in a.runs:
        mf = f"{a.runs_dir}/{run}/metrics.json"
        if not os.path.exists(mf):
            problems.append(f"{run}: no {mf}"); continue
        m = json.load(open(mf))
        args = m["args"]
        cands = [args.get("ckpt"), f"{a.runs_dir}/{run}/model.pt", args.get("out") and f"{args['out']}/model.pt"]
        ckpt = next((p for p in cands if p and os.path.exists(p)), None)
        if ckpt is None:
            problems.append(f"{run}: no weights at {', '.join(p for p in cands if p)}"); continue
        val = args["val"].rstrip("/")
        info[run] = (m, ckpt, val, os.path.basename(val).removesuffix("_val"))
    for p in problems:   # evaluated are the others; the job still fails at the end
        print(f"OOD SKIPPED {p}", flush=True)
    cache, cur, failed = {}, None, []

    def load(prefixes):   # by real path: the same validation pack is linked from two places
        key = tuple(os.path.realpath(p + ".json") for p in prefixes)
        if key not in cache:
            cache[key] = crops(prefixes, a.workers)
        return cache[key]

    for run in sorted(info, key=lambda r: info[r][3]):
        m, ckpt, val, name = info[run]
        args, t0 = m["args"], time.time()
        packs = {s: f"{a.packs}/{name}_ood_{s}" for s in a.sets if os.path.exists(f"{a.packs}/{name}_ood_{s}.json")}
        if not packs:
            print(f"OOD {run}: no packs {a.packs}/{name}_ood_* (pack mode with the {name} class list), skipped", flush=True)
            continue
        if name != cur:   # one class list in memory at a time
            cache, cur = {}, name
        wnids = json.load(open(val + "_classes.json"))["wnids"] if os.path.exists(val + "_classes.json") else None
        for s, p in packs.items():
            pw = json.load(open(p + "_classes.json"))["wnids"]
            assert len(pw) == args["ncls"] and (wnids is None or wnids == pw), f"{p}: its class list is not the one of {run} ({val})"
        net, res = load_net(args, ckpt), {}
        for s, prefixes in [("val", expand(val))] + [(s, [p]) for s, p in packs.items()]:
            x, y = load(prefixes)
            z = predict(net, x)
            cov = torch.unique(y)
            hit = (z.argmax(1) == y).float()
            r = dict(top1=hit.mean().item(), n=len(y), classes=len(cov),
                     # each covered class's top-1: paired class-bootstrap gaps between run groups
                     per_class={str(int(c)): hit[y == c].mean().item() for c in cov})
            if len(cov) < args["ncls"]:
                r["top1_restricted"] = (cov[z[:, cov].argmax(1)] == y).float().mean().item()
            res[s] = r
        line = ", ".join(f"{s} {r['top1']:.4f}" + (f" (restricted {r['top1_restricted']:.4f}, {r['classes']} classes)"
                                                    if "top1_restricted" in r else "") for s, r in res.items())
        if abs(res["val"]["top1"] - m["top1"]) > 0.005:   # bf16 and cuDNN noise is a few images out of thousands
            failed.append(run)
            print(f"OOD {run}: NOT SAVED, val top-1 {res['val']['top1']:.4f} but {m['top1']:.4f} in metrics.json ({ckpt}): {line}", flush=True)
            continue
        save(a.out, run, res)
        print(f"OOD {run}: {line} ({time.time() - t0:.0f} s)", flush=True)
    if problems or failed:
        raise SystemExit(f"OOD: {len(problems)} runs skipped (no metrics.json or weights), {len(failed)} not saved "
                         f"(weights do not reproduce metrics.json: {' '.join(failed)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["download", "pack", "eval"])
    ap.add_argument("--sets", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    ap.add_argument("--classes", default="in100", help="pack: the class list (in100, in1k)")
    ap.add_argument("--out-dir", default=str(REAL), help="pack: where the packs go")
    ap.add_argument("--runs", nargs="+", help="eval: run names under --runs-dir")
    ap.add_argument("--runs-dir", default=str(RUNS))
    ap.add_argument("--packs", default=str(REAL), help="eval: where <name>_ood_<set> packs are")
    ap.add_argument("--out", default=str(PROBES / "ood.json"))
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    if a.mode == "download":
        for s in a.sets:
            t0 = time.time()
            d = local(s)
            gb = sum(os.path.getsize(f) for p in SOURCES[s][2] for f in glob.glob(f"{d}/{p}")) / 1e9
            print(f"DOWNLOADED {s}: {SOURCES[s][0]}@{SOURCES[s][1][:7]}, {gb:.2f} GB in {d} ({time.time() - t0:.0f} s)", flush=True)
    elif a.mode == "pack":
        in1k = [w for w, _ in imagenet_classes()]
        assert len(in1k) == 1000 and in1k == sorted(in1k), "classes.py is not in sorted-wnid order: V2's folder index would not map"
        wnids = class_list(a.classes)["wnids"]
        assert set(wnids) <= set(in1k), f"{a.classes}: wnids outside ImageNet-1k"
        name = os.path.basename(a.classes).removesuffix(".json").removesuffix("_classes").removesuffix("_val")
        out = a.out_dir
        with Pool(len(a.sets)) as p:
            for line in p.imap_unordered(pack_one, [(s, a.classes, f"{out}/{name}_ood_{s}") for s in a.sets]):
                print(line, flush=True)
    else:
        assert a.runs, "eval needs --runs"
        torch.backends.cudnn.benchmark = True
        evaluate(a)


if __name__ == "__main__":
    main()
