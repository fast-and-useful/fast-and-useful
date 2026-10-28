"""Fine-grained classification datasets as packs (packs.py): Oxford-IIIT Pets, Stanford Cars and Food-101, to train on
images generated for their class names and test on their real test images.

Sources (Hugging Face hub, revisions pinned in SOURCES). Labels are the source's own, in its class order (the ClassLabel
names in the parquet metadata, asserted identical across shards; Cars: classnames.txt):
  pets      timm/oxford-iiit-pet: the standard trainval (3,680, the source's train) and test (3,669) splits of the 37
            breeds, with a cat/dog column.
  cars      clip-benchmark/wds_cars (CLIP_benchmark's webdataset, written through torchvision's StanfordCars): train
            (8,144) and test (8,041) of the 196 classes, original bytes.
  food101   ethz/food101: train (75,750, label noise left in by design) and validation (25,250, the official test split)
            of the 101 dishes.

Class names (data/classes/<set>.json): comma-free, unique, and naming the kind of object wherever the bare label leaves
it open (the templates' "in its natural setting" and "a small / large" pull such a name toward something else):
  pets      the breed in title case + the species of its images: "abyssinian" -> "Abyssinian cat"
  cars      the year moved to the front, + "car": "Audi S4 Sedan 2012" -> "2012 Audi S4 Sedan car"
  food101   the label with spaces, "apple_pie" -> "apple pie"; the 8 labels that alone name an animal or a person get a
            dish word (FOOD_DISH): "hot_dog" -> "hot dog sandwich", "oysters" -> "oysters dish"
wnids are "<set>_<label:03d>": "pets_000".

pack: <out-dir>/<set>_test (the test split, original bytes) and <set>_train (all training images, short side 256),
labels 0..K-1, with a sidecar <prefix>_classes.json. check: every pack against its sidecar and a few images decoded.

Usage: python -m fau.finegrained download [--sets pets cars food101]
       python -m fau.finegrained pack [--sets ...] [--out-dir work/real] [--workers 16]
       python -m fau.finegrained check [--sets ...] [--out-dir work/real]
"""
import argparse, glob, io, json, os, re, tarfile, time
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from PIL import Image

from fau import REAL
from fau.packs import PackSet, PackWriter, encode, load_recs

SOURCES = {   # set: dataset, repo, revision, files, {pack: (source splits, images)}, classes
    "pets": ("Oxford-IIIT Pets", "timm/oxford-iiit-pet", "089695c834a7deb60505b7cc506672db1c31a6aa", ["data/*.parquet"],
             {"train": (["train"], 3680), "test": (["test"], 3669)}, 37),
    "cars": ("Stanford Cars", "clip-benchmark/wds_cars", "25cfe61abf8373d833d400d76983a9b389ebcdba",
             ["classnames.txt", "train/*.tar", "test/*.tar"], {"train": (["train"], 8144), "test": (["test"], 8041)}, 196),
    "food101": ("Food-101", "ethz/food101", "83488de741c1bd1ce27aa6a2b33e19c7bdf92ca9", ["data/*.parquet"],
                {"train": (["train"], 75750), "test": (["validation"], 25250)}, 101),
}
LABEL = {}   # the class column where it is not "label"
# Food-101 labels that alone name an animal or a person, not a dish
FOOD_DISH = {"chicken_wings": "chicken wings dish", "croque_madame": "croque madame sandwich", "escargots": "escargots dish",
             "hot_dog": "hot dog sandwich", "mussels": "mussels dish", "oysters": "oysters dish", "peking_duck": "peking duck dish",
             "scallops": "scallops dish"}
SIGNATURES = (b"\xff\xd8\xff", b"\x89PNG", b"GIF8", b"RIFF", b"BM")   # JPEG, PNG, GIF, WebP, BMP


def local(s):
    """The source's snapshot directory (downloads it unless HF_HUB_OFFLINE). Offline, a snapshot directory can exist
    without its data files, hence the check."""
    _, repo, rev, files, _, _ = SOURCES[s]
    d = snapshot_download(repo, repo_type="dataset", revision=rev, allow_patterns=files)
    missing = [f for f in files if not glob.glob(f"{d}/{f}")]
    assert not missing, f"{repo}@{rev[:7]}: {missing} not in {d} (run the download mode first)"
    return d


def shards(d, split):
    files = sorted(glob.glob(f"{d}/data/{split}-*.parquet"))
    assert files, f"no {split} shards in {d}/data"
    return files


def features(files):
    """The HF features in the parquet metadata (ClassLabel names), one set of them for all shards."""
    feats = [json.loads(pq.read_schema(f).metadata[b"huggingface"])["info"]["features"] for f in files]
    assert all(f == feats[0] for f in feats), f"shards disagree on their features: {files}"
    return feats[0]


def all_shards(s, d):
    return [f for src, _ in SOURCES[s][4].values() for sp in src for f in shards(d, sp)]


def per_class(s, d, col):
    """The value of a per-image class-label column for every class, asserted the same for all images of the class (a
    breed's species)."""
    lab, files, vals = LABEL.get(s, "label"), all_shards(s, d), {}
    feats = features(files)
    names, k_all = feats[col]["names"], len(feats[lab]["names"])
    for f in files:
        t = pq.read_table(f, columns=[lab, col])
        for k, v in zip(t.column(lab).to_pylist(), t.column(col).to_pylist()):
            vals.setdefault(k, set()).add(v)
    bad = {k: [names[v] for v in vs] for k, vs in vals.items() if len(vs) > 1}
    assert not bad and sorted(vals) == list(range(k_all)), f"{s}: classes with several or no {col} values: {bad}"
    return [names[min(vals[k])] for k in range(k_all)]


def class_list(s, d):
    """(source names, prompt names) in label order (see the module docstring for the rules)."""
    if s == "cars":   # webdataset: the names in classnames.txt, one per line in label order
        src = [l.strip() for l in open(f"{d}/classnames.txt") if l.strip()]
    else:
        src = features(all_shards(s, d))[LABEL.get(s, "label")]["names"]
    if s == "pets":
        names = [" ".join(w.capitalize() for w in n.split("_")) + " " + kind for n, kind in zip(src, per_class(s, d, "label_cat_dog"))]
    elif s == "cars":   # "<make> <model> <body> <year>"
        ms = [re.fullmatch(r"(.+) ((?:19|20)\d\d)", n) for n in src]
        assert all(ms), f"cars: names without a trailing year: {[n for n, m in zip(src, ms) if not m]}"
        names = [f"{m[2]} {m[1]} car" for m in ms]
    elif s == "food101":
        names = [FOOD_DISH.get(n, n.replace("_", " ")) for n in src]
    else:
        raise ValueError(s)
    return src, names


def read(s, d, split, src_names):
    """The images of a pack split as lists of (label, src, bytes, record metadata), one parquet shard per list. A pack made
    of several source splits (FGVC-Aircraft's train + validation) records each image's split."""
    lab, sps = LABEL.get(s, "label"), SOURCES[s][4][split][0]
    for sp in sps:
        meta = dict(split=sp) if len(sps) > 1 else {}
        if s == "cars":   # webdataset shards <split>/<i>.tar; a sample is <key>.jpg and <key>.cls (its label as text)
            for f in sorted(glob.glob(f"{d}/{sp}/*.tar"), key=lambda p: int(os.path.basename(p).split(".")[0])):
                smp = {}
                with tarfile.open(f) as t:
                    for m in t:
                        if m.isfile():
                            key, ext = m.name.rsplit(".", 1)
                            smp.setdefault(key, {})[ext] = t.extractfile(m).read()
                assert all({"jpg", "cls"} <= set(v) for v in smp.values()), f"{f}: samples without jpg or cls"
                yield [(int(v["cls"]), key, v["jpg"], meta) for key, v in smp.items()]
            continue
        for f in shards(d, sp):
            t = pq.read_table(f, columns=["image", lab])
            batch = []
            for i, (img, k) in enumerate(zip(t.column("image").to_pylist(), t.column(lab).to_pylist())):
                p = img.get("path") or f"{os.path.basename(f)}:{i}"
                batch.append((k, f"{src_names[k]}/{p}" if s == "food101" else p, img["bytes"], meta))   # Food-101: <class>/<id>.jpg
            yield batch


def prep(job):
    """Decodes an image as PackSet will (a truncated or corrupt file fails here, not in a training worker); training copies
    are re-encoded with the short side at `short`. -> (label, src, meta, bytes or None, format, error)"""
    (label, src, raw, meta), short = job
    fmt = "?"
    try:
        im = Image.open(io.BytesIO(raw))
        fmt = f"{im.format} {im.mode}"
        im = im.convert("RGB")
    except Exception as e:
        return label, src, meta, None, fmt, f"{type(e).__name__}: {str(e)[:80]}"
    return label, src, meta, raw if short is None else encode(im, short=short), fmt, None


def pack_one(s, split, prefix, workers):
    title, repo, rev, _, splits, k = SOURCES[s]
    d, t0 = local(s), time.time()
    src_names, names = class_list(s, d)
    wnids = [f"{s}_{i:03d}" for i in range(k)]
    assert len(src_names) == k, f"{s}: {len(src_names)} classes in the source, expected {k}"
    assert len(set(names)) == k and not any("," in n for n in names), f"{s}: class names not unique or with commas"
    short = None if split == "test" else 256
    w, counts, fmts, bad, n = PackWriter(prefix), np.zeros(k, dtype=int), Counter(), [], 0
    with Pool(workers) as p:
        for batch in read(s, d, split, src_names):   # one shard at a time: imap's feeder would read every shard ahead
            n += len(batch)
            for label, src, meta, data, fmt, err in p.imap(prep, [(r, short) for r in batch], chunksize=16):
                fmts[fmt] += 1
                if err:
                    bad.append(f"{src} ({err})")
                    continue
                w.add(data, label, src=src, **meta)
                counts[label] += 1
    src_splits, n_src = splits[split]
    assert n == n_src, f"{s} {split}: {n} images in {repo} {'+'.join(src_splits)}, expected {n_src}"
    assert counts.min() > 0, f"{s} {split}: classes without images: {np.flatnonzero(counts == 0).tolist()}"
    side = dict(wnids=wnids, names=names, dataset=title, source_names=src_names, repo=repo, revision=rev, splits=src_splits,
                counts=counts.tolist(), formats=dict(fmts.most_common()), dropped=bad)
    with open(prefix + "_classes.json.tmp", "w") as f:
        json.dump(side, f, indent=1)
    os.replace(prefix + "_classes.json.tmp", prefix + "_classes.json")
    w.close()   # the pack appears last, with its sidecar already in place
    return (f"PACK {s} {split}: {counts.sum()} images of {k} classes ({counts.min()}-{counts.max()} per class) from "
            f"{repo}@{rev[:7]} {'+'.join(src_splits)} ({n} images), {len(bad)} undecodable{': ' + ', '.join(bad) if bad else ''}; "
            f"formats {dict(fmts.most_common())} -> {prefix} ({os.path.getsize(prefix + '.bin') / 1e9:.2f} GB, {time.time() - t0:.0f} s)")


def check(s, out, rng):
    """Both packs of a set against their sidecars; a few images decoded through PackSet. -> the set's report line."""
    k, sides = SOURCES[s][5], {}
    for split in ("train", "test"):
        prefix = f"{out}/{s}_{split}"
        side, recs = json.load(open(prefix + "_classes.json")), load_recs(prefix)
        sides[split] = side
        lab = np.array([r["label"] for r in recs])
        assert side["wnids"] == [f"{s}_{i:03d}" for i in range(k)] and len(side["names"]) == k, f"{prefix}: class list"
        assert lab.min() >= 0 and lab.max() < k and np.bincount(lab, minlength=k).tolist() == side["counts"], f"{prefix}: labels vs sidecar counts"
        assert len(recs) + len(side["dropped"]) == SOURCES[s][4][split][1], f"{prefix}: images + dropped != source"
        offs = np.array([r["off"] for r in recs]); lens = np.array([r["len"] for r in recs])
        assert offs[0] == 0 and (offs[1:] == offs[:-1] + lens[:-1]).all() and offs[-1] + lens[-1] == os.path.getsize(prefix + ".bin"), \
            f"{prefix}: records do not tile the .bin"
        mm = np.memmap(prefix + ".bin", dtype=np.uint8, mode="r")
        sig = Counter(next((g for g in SIGNATURES if mm[o:o + len(g)].tobytes() == g), b"?") for o in offs)
        assert b"?" not in sig, f"{prefix}: {sig[b'?']} records without an image signature"
        ds = PackSet([(prefix, 0, None)], lambda im: im.size)
        sizes = [(label, size) for size, label, _ in (ds[i] for i in sorted(rng.choice(len(ds), 6, replace=False).tolist()))]
        if split == "train":
            assert all(min(sz) == 256 for _, sz in sizes), f"{prefix}: short side is not 256: {sizes}"
        print(f"CHECK {prefix}: {len(recs)} images, {k} classes, {min(side['counts'])}-{max(side['counts'])} per class, "
              f"signatures {dict((g.hex(), c) for g, c in sig.items())}, {len(side['dropped'])} dropped; decoded (label, size): {sizes}",
              flush=True)
    assert sides["train"]["names"] == sides["test"]["names"], f"{s}: train and test class lists differ"
    tr, te = sides["train"], sides["test"]
    return (f"{s}: {k} classes, train {sum(tr['counts'])} ({min(tr['counts'])}-{max(tr['counts'])} per class), test "
            f"{sum(te['counts'])} ({min(te['counts'])}-{max(te['counts'])}), dropped {len(tr['dropped'])} / {len(te['dropped'])}; "
            f"names e.g. {' | '.join(te['names'][i] for i in np.linspace(0, k - 1, 5).astype(int))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["download", "pack", "check"])
    ap.add_argument("--sets", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    ap.add_argument("--out-dir", default=str(REAL))
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    if a.mode == "download":
        for s in a.sets:
            t0 = time.time()
            d = local(s)
            gb = sum(os.path.getsize(f) for p in SOURCES[s][3] for f in glob.glob(f"{d}/{p}")) / 1e9
            print(f"DOWNLOADED {s}: {SOURCES[s][1]}@{SOURCES[s][2][:7]}, {gb:.2f} GB in {d} ({time.time() - t0:.0f} s)", flush=True)
    elif a.mode == "pack":
        os.makedirs(a.out_dir, exist_ok=True)
        for s in a.sets:
            for split in ("test", "train"):
                print(pack_one(s, split, f"{a.out_dir}/{s}_{split}", a.workers), flush=True)
    else:
        rng = np.random.default_rng(0)
        lines = [check(s, a.out_dir, rng) for s in a.sets]
        print("\n".join(["SUMMARY"] + lines), flush=True)


if __name__ == "__main__":
    main()
