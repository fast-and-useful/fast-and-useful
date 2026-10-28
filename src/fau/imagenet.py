"""ImageNet class subsets as packs (packs.py), from the ILSVRC/imagenet-1k parquet shards (pinned in fau/hub.py). The
classes are a class list (data/classes: in100, the CMC ImageNet-100; tin200, the 200-class protocol), or `all` for the
1,000 classes in label order. Labels are re-indexed 0..K-1 in list order. --per_class N keeps the first N images of each
class in shard order (a label-only pass picks the rows before any image is decoded). Validation images keep their
original bytes (evaluation resizes to 256 and crops 224); training images are stored with the short side at 256.

Usage: python -m fau.imagenet --split validation --classes in100 --out work/real/in100_val
       python -m fau.imagenet --split train --classes in100 --out work/real/in100_train --workers 16
       python -m fau.imagenet --split train --classes all --out work/real/in1k_train --workers 16
"""
import argparse, glob, importlib.util, io, json, os
from concurrent.futures import ProcessPoolExecutor

import pyarrow.parquet as pq
from PIL import Image

from fau.hub import snapshot
from fau.packs import PackWriter, encode
from fau.prompts import class_list

IMAGENET = "ILSVRC/imagenet-1k"


def imagenet_classes():
    snap = snapshot(IMAGENET)
    spec = importlib.util.spec_from_file_location("in_classes", f"{snap}/classes.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return list(m.IMAGENET2012_CLASSES.items())     # [(wnid, "name, synonym, ...")] in label order


def shards(split):
    files = sorted(glob.glob(f"{snapshot(IMAGENET)}/data/{split}-*.parquet"))
    assert files, f"no {split} shards in {snapshot(IMAGENET)}"
    return files


def first_rows(files, keep, n):
    """The first n rows of every kept class, in shard order, read from the label column alone: {file: set of rows}."""
    seen, rows = {}, {}
    for f in files:
        rows[f] = set()
        for i, lab in enumerate(pq.read_table(f, columns=["label"]).column("label").to_pylist()):
            if lab in keep and seen.get(lab, 0) < n:
                seen[lab] = seen.get(lab, 0) + 1
                rows[f].add(i)
    return rows


def work(args):
    files, keep, prefix, short, rows = args
    w = PackWriter(prefix)
    for f in files:
        i = 0   # row index within the shard, as first_rows counts it
        # batch by batch, so a worker holds one batch of a shard, not the whole shard as Python objects (44 workers
        # reading whole train shards outgrew 90 GB on 2026-10-13, and the pool hung on its killed workers)
        for b in pq.ParquetFile(f).iter_batches(batch_size=256, columns=["image", "label"]):
            d = b.to_pydict()
            for img, lab in zip(d["image"], d["label"]):
                if lab in keep and (rows is None or i in rows[f]):
                    raw = img["bytes"]
                    data = raw if short is None else encode(Image.open(io.BytesIO(raw)), short=short)
                    w.add(data, keep[lab], src=os.path.basename(img.get("path") or ""))
                i += 1
    return w.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "validation"], required=True)
    ap.add_argument("--classes", required=True, help="a class list (in100, tin200) or `all`")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--per_class", type=int, default=0, help="keep the first N images of each class (0: all)")
    a = ap.parse_args()
    classes = imagenet_classes()
    wnids = [w for w, _ in classes] if a.classes == "all" else class_list(a.classes)["wnids"]
    pos = {w: i for i, (w, _) in enumerate(classes)}
    missing = [w for w in wnids if w not in pos]
    assert not missing, f"unknown wnids: {missing[:5]}"
    keep = {pos[w]: k for k, w in enumerate(wnids)}
    names = [classes[pos[w]][1] for w in wnids]
    json.dump(dict(wnids=wnids, names=names, imagenet_index=[pos[w] for w in wnids]), open(a.out + "_classes.json", "w"), indent=1)
    files = shards(a.split)
    short = None if a.split == "validation" else 256
    rows = first_rows(files, keep, a.per_class) if a.per_class else None
    n = max(1, min(a.workers, len(files)))
    jobs = [(files[i::n], keep, f"{a.out}_part{i}" if n > 1 else a.out, short, rows) for i in range(n)]
    with ProcessPoolExecutor(n) as ex:   # a worker killed (e.g. out of memory) raises BrokenProcessPool, never hangs
        counts = list(ex.map(work, jobs))
    if n > 1:   # one pack, so index lists (e.g. subsets of the real images) address a single file
        w = PackWriter(a.out)
        for i in range(n):
            part = f"{a.out}_part{i}"
            data = open(part + ".bin", "rb").read()
            for r in json.load(open(part + ".json")):
                w.add(data[r["off"]:r["off"] + r["len"]], r["label"], src=r.get("src", ""))
            os.remove(part + ".bin"); os.remove(part + ".json")
        w.close()
    print(f"{a.split}: {sum(counts)} images of {len(wnids)} classes from {len(files)} shards -> {a.out}")


if __name__ == "__main__":
    main()
