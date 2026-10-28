"""Relay check: does the first-step adapter reproduce the teacher's content seed by seed, and do the
distilled model's untouched steps carry it to the image? Every set draws image (k, j) from the same noise and prompt, so
two sets compare image by image: the cosine similarity of the two images' DINOv3 features (frozen.py's cache),
averaged over images. The different-noise baseline pairs image (k, j) of A with image (k, j + 8) of B: same class and
prompt template, other noise. same - different is the part of the similarity that the noise, i.e. the seed's own
commitment, explains.

Read: the adapter against its ceiling (the teacher's own state finished by the distilled model), next to the plain
distilled model, base-first and the base finishing its own trajectory against the same ceiling.

Usage (CPU job; the features must be cached): python -m fau.metrics.relay --pairs A:B [A:B ...] --out FILE.json --cache DIR
"""
import argparse, json, os

import torch

from fau import SETS
from fau.packs import load_recs
from fau.train import expand


def indexed(name, cache):
    """name -> (features [N, D] float, {(class, j): row}) in the probe's record order."""
    x, y = torch.load(f"{cache}/{name}.pt")
    keys = [(int(r["label"]), int(r["j"])) for p in expand(f"{SETS}/in100/{name}") for r in load_recs(p)]
    assert len(keys) == len(x) and all(k == int(c) for (k, _), c in zip(keys, y)), f"{name}: cache and records disagree"
    return torch.nn.functional.normalize(x.float(), dim=1), {k: i for i, k in enumerate(keys)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", required=True, help="frozen.py's feature cache of the encoder (work/cache/dinov3_vitl16)")
    ap.add_argument("--shift", type=int, default=8, help="different-noise partner: j + shift (a multiple of the 8 templates)")
    a = ap.parse_args()
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    sets = {}
    for p in a.pairs:
        A, B = p.split(":")
        for s in (A, B):
            if s not in sets:
                sets[s] = indexed(s, a.cache)
        (xa, ia), (xb, ib) = sets[A], sets[B]
        same = [(ia[k], ib[k]) for k in ia if k in ib]
        diff = [(ia[(c, j)], ib[(c, j + a.shift)]) for (c, j) in ia if (c, j + a.shift) in ib]
        cos = lambda pr: (xa[[i for i, _ in pr]] * xb[[i for _, i in pr]]).sum(1)
        s, d = cos(same), cos(diff)
        res[p] = dict(same=s.mean().item(), different=d.mean().item(), seed_share=(s.mean() - d.mean()).item(),
                      same_sd=s.std().item(), n_same=len(same), n_different=len(diff))
        print(f"RELAY {A} vs {B}: same noise {res[p]['same']:.4f}, different noise {res[p]['different']:.4f}, "
              f"seed share {res[p]['seed_share']:.4f} ({len(same)} images)", flush=True)
    json.dump(res, open(a.out + ".tmp", "w"), indent=1); os.replace(a.out + ".tmp", a.out)


if __name__ == "__main__":
    main()
