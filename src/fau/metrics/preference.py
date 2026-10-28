"""How pretty a generated set is, next to how useful it is (train.py): two standard scores per image, averaged
over n images spread evenly over the set.
  aesthetic   LAION improved aesthetic predictor: an MLP on normalized CLIP ViT-L/14 image embeddings (prompt-free)
  pickscore   PickScore v1: human preference for the image given its own prompt (CLIP-H, logit scale)

Usage: python -m fau.metrics.preference --sets SPEC [SPEC ...] --out FILE.json [--n 5000]
(SPEC as in train.py, optionally NAME=SPEC; images without a stored prompt get no PickScore)
"""
import argparse, io, json, os, time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from fau.devices import DEV, WORKERS, half
from fau.hub import snapshot
from fau.packs import load_recs
from fau.train import build_sources

CLIP_L = "openai/clip-vit-large-patch14"
AESTHETIC = ("trl-lib/ddpo-aesthetic-predictor", "aesthetic-model.pth")
PICK, PICK_PROC = "yuvalkirstain/PickScore_v1", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"


class AestheticMLP(nn.Module):
    """The improved-aesthetic-predictor head (sac+logos+ava1-l14-linearMSE)."""

    def __init__(self, d=768):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(d, 1024), nn.Dropout(0.2), nn.Linear(1024, 128), nn.Dropout(0.2),
                                    nn.Linear(128, 64), nn.Dropout(0.1), nn.Linear(64, 16), nn.Linear(16, 1))

    def forward(self, x):
        return self.layers(x)


def items(spec, n):
    recs = [(p + ".bin", r) for p, _, keep in build_sources([spec]) for i, r in enumerate(load_recs(p)) if keep is None or i in keep]
    return [recs[i] for i in np.linspace(0, len(recs) - 1, min(n, len(recs))).astype(int)]


def read(path, r):
    with open(path, "rb") as f:
        f.seek(r["off"]); return Image.open(io.BytesIO(f.read(r["len"]))).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--bs", type=int, default=128)
    a = ap.parse_args()
    from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor
    clip = CLIPModel.from_pretrained(snapshot(CLIP_L), dtype=half(torch.float16), disable_mmap=True).to(DEV).eval()
    cproc = CLIPProcessor.from_pretrained(snapshot(CLIP_L))
    head = AestheticMLP()
    head.load_state_dict(torch.load(os.path.join(snapshot(AESTHETIC[0]), AESTHETIC[1]), map_location="cpu", weights_only=True))
    head = head.to(DEV).eval()
    pick = AutoModel.from_pretrained(snapshot(PICK), dtype=half(torch.float16), disable_mmap=True).to(DEV).eval()
    pproc = AutoProcessor.from_pretrained(snapshot(PICK_PROC))
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for spec in a.sets:
        name, spec = spec.split("=", 1) if "=" in spec else (os.path.basename(spec.split(":")[0].rstrip("/")), spec)
        t0, aes, pk = time.time(), [], []
        its = items(spec, a.n)
        for b in range(0, len(its), a.bs):
            chunk = its[b:b + a.bs]
            ims = [read(p, r) for p, r in chunk]
            with torch.no_grad():
                x = cproc(images=ims, return_tensors="pt")["pixel_values"].to(DEV, half(torch.float16))
                e = clip.get_image_features(pixel_values=x).pooler_output.float()   # projected embedding
                aes += head(e / e.norm(dim=-1, keepdim=True)).squeeze(1).tolist()
                prompts = [r.get("prompt") for _, r in chunk]
                if all(prompts):
                    pi = pproc(images=ims, return_tensors="pt")["pixel_values"].to(DEV, half(torch.float16))
                    pt = pproc(text=prompts, padding=True, truncation=True, max_length=77, return_tensors="pt").to(DEV)
                    ie = pick.get_image_features(pixel_values=pi).pooler_output.float(); te = pick.get_text_features(**pt).pooler_output.float()
                    ie, te = ie / ie.norm(dim=-1, keepdim=True), te / te.norm(dim=-1, keepdim=True)
                    pk += (pick.logit_scale.exp().float() * (ie * te).sum(-1)).tolist()
        res[name] = dict(n=len(its), aesthetic=float(np.mean(aes)), aesthetic_sd=float(np.std(aes)),
                         pickscore=float(np.mean(pk)) if pk else None)
        print(f"PRETTY {name}: aesthetic {res[name]['aesthetic']:.3f}, PickScore "
              f"{res[name]['pickscore'] if pk else float('nan'):.3f} ({len(its)} images, {time.time() - t0:.0f} s)", flush=True)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out + ".tmp", "w"), indent=1); os.replace(a.out + ".tmp", a.out)


if __name__ == "__main__":
    main()
