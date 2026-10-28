"""Packed image sets for the classification experiments: one <prefix>.bin of concatenated JPEG bytes and one
<prefix>.json with a record per image (byte offset and length, label, and free metadata). Thousands of small files are
slow on network file systems; a pack is two files per writer.

PackSet reads any number of packs, each tagged with a domain id (0 = real, 1 = generated), and serves
(image, label, domain) with a torchvision transform.
"""
import io, json, os

import numpy as np
import torch
from PIL import Image


class PackWriter:
    def __init__(self, prefix):
        os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
        self.prefix, self.f, self.recs, self.off = prefix, open(prefix + ".bin.tmp", "wb"), [], 0

    def add(self, jpeg, label, **meta):
        self.f.write(jpeg)
        self.recs.append(dict(off=self.off, len=len(jpeg), label=int(label), **meta))
        self.off += len(jpeg)

    def close(self):
        self.f.close()
        json.dump(self.recs, open(self.prefix + ".json.tmp", "w"))
        os.replace(self.prefix + ".bin.tmp", self.prefix + ".bin")    # a pack exists only once it is complete
        os.replace(self.prefix + ".json.tmp", self.prefix + ".json")
        return len(self.recs)


def encode(img, short=None, quality=90):
    """RGB PIL image -> JPEG bytes, optionally resized so its short side is `short` (training copies)."""
    img = img.convert("RGB")
    if short and min(img.size) != short:
        s = short / min(img.size)
        img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BICUBIC)
    b = io.BytesIO()
    img.save(b, format="JPEG", quality=quality)
    return b.getvalue()


def load_recs(prefix):
    return json.load(open(prefix + ".json"))


class PackSet(torch.utils.data.Dataset):
    """sources: [(prefix, domain, keep)] where keep is None or a set of record indices to use."""

    def __init__(self, sources, transform):
        self.items, self.files, self.transform = [], [], transform
        for k, (prefix, domain, keep) in enumerate(sources):
            self.files.append(prefix + ".bin")
            for i, r in enumerate(load_recs(prefix)):
                if keep is None or i in keep:
                    self.items.append((k, r["off"], r["len"], r["label"], domain))
        self.items = np.array(self.items, dtype=np.int64)
        self.mm = None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        if self.mm is None:   # opened lazily, once per worker
            self.mm = [np.memmap(f, dtype=np.uint8, mode="r") for f in self.files]
        k, off, n, label, domain = (int(v) for v in self.items[i])
        img = Image.open(io.BytesIO(self.mm[k][off:off + n].tobytes())).convert("RGB")
        return self.transform(img), label, domain

    def labels(self):
        return self.items[:, 3]

    def domains(self):
        return self.items[:, 4]
