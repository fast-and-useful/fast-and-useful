"""Low-level image statistics of a set: what the late denoising steps could change without changing what is drawn.
Every image is center-cropped to 256 x 256 (the stored copies have their short side at 256, JPEG quality 90, for real and
generated images alike), and per image:
  spectrum     the radially averaged log power spectrum of the luminance (32 bands), and its slope (real photos: about -2)
  hf           the share of luminance power above half the Nyquist frequency
  saturation   mean HSV saturation; colorfulness (Hasler and Suesstrunk 2003)
  contrast     standard deviation of the luminance; sharpness: variance of its Laplacian
Per set: the mean and standard deviation of each scalar over images, and the mean spectrum.

Usage: python -m fau.metrics.image_stats --sets SPEC [SPEC ...] --out FILE.json [--n 5000] [--workers 32]
(SPEC as in train.py, optionally NAME=SPEC; n images per set, spread evenly over the set)
"""
import argparse, io, json, os, time
from multiprocessing import Pool

import numpy as np
from PIL import Image

from fau.train import build_sources
from fau.packs import load_recs

BANDS = 32


def radial_index(n=256):
    y, x = np.indices((n, n)) - n // 2
    r = np.sqrt(x ** 2 + y ** 2) / (n // 2)          # 1 = Nyquist along the axes
    return np.minimum((r * BANDS).astype(int), BANDS)   # the corners (beyond Nyquist) fall in an extra band, dropped


RIDX = radial_index()
FREQ = (np.arange(BANDS) + 0.5) / BANDS


def one(args):
    path, off, n = args
    with open(path, "rb") as f:
        f.seek(off); b = f.read(n)
    im = Image.open(io.BytesIO(b)).convert("RGB")
    w, h = im.size
    l, t = (w - 256) // 2, (h - 256) // 2
    rgb = np.asarray(im.crop((l, t, l + 256, t + 256)), dtype=np.float64) / 255
    lum = rgb @ np.array([0.299, 0.587, 0.114])
    p = np.abs(np.fft.fftshift(np.fft.fft2(lum - lum.mean()))) ** 2
    bands = np.bincount(RIDX.ravel(), weights=p.ravel(), minlength=BANDS + 1)[:BANDS] / np.bincount(RIDX.ravel(), minlength=BANDS + 1)[:BANDS]
    spec = np.log10(bands + 1e-12)
    slope = np.polyfit(np.log10(FREQ[2:]), spec[2:], 1)[0]
    hf = bands[BANDS // 2:].sum() / bands.sum()
    mx, mn = rgb.max(2), rgb.min(2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0).mean()
    rg, yb = rgb[..., 0] - rgb[..., 1], 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]
    colorful = np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    lap = lum[1:-1, 1:-1] * 4 - lum[:-2, 1:-1] - lum[2:, 1:-1] - lum[1:-1, :-2] - lum[1:-1, 2:]
    return spec, dict(slope=slope, hf=hf, saturation=sat, colorfulness=colorful, contrast=lum.std(), sharpness=lap.var())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    with Pool(a.workers) as pool:
        for spec in a.sets:
            name, spec = spec.split("=", 1) if "=" in spec else (os.path.basename(spec.split(":")[0].rstrip("/")), spec)
            t0 = time.time()
            items = [(p + ".bin", r["off"], r["len"]) for p, _, keep in build_sources([spec])
                     for i, r in enumerate(load_recs(p)) if keep is None or i in keep]
            items = [items[i] for i in np.linspace(0, len(items) - 1, min(a.n, len(items))).astype(int)]
            out = pool.map(one, items, chunksize=16)
            specs = np.stack([o[0] for o in out])
            res[name] = dict(n=len(out), spectrum=specs.mean(0).round(4).tolist(),
                             **{k: float(np.mean([o[1][k] for o in out])) for k in out[0][1]},
                             **{f"{k}_sd": float(np.std([o[1][k] for o in out])) for k in out[0][1]})
            print(f"STATS {name}: " + ", ".join(f"{k} {res[name][k]:.4g}" for k in out[0][1]) + f" ({time.time() - t0:.0f} s)", flush=True)
            json.dump(res, open(a.out + ".tmp", "w"), indent=1); os.replace(a.out + ".tmp", a.out)


if __name__ == "__main__":
    main()
