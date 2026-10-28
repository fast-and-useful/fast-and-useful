"""Generic prompts for fitting an adapter: COCO-2017 training captions without ImageNet-100 class names.

The first caption of each image (lowest annotation id), images in id order; a caption is dropped when it contains an
ImageNet-100 class's prompt name (generate.class_name, case-insensitive, whole words, with or without a trailing "s"),
so the adapter never sees the evaluated classes' names. Writes a JSON list (adapter.py's --prompts file) and a
.meta.json with the counts. The paper's file, data/prompts/coco_16000.json.xz, is the first 16,000 (--n 16000).

Usage (CPU): python -m fau.prompts.coco --captions annotations/captions_train2017.json --n 16000 --out coco_16000.json
"""
import argparse, json, os, re

from fau.generate import class_name
from fau.prompts import class_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", required=True)
    ap.add_argument("--classes", default="in100")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    meta = class_list(a.classes)
    names = sorted({class_name(w, n).lower() for w, n in zip(meta["wnids"], meta["names"])}, key=len, reverse=True)
    pat = re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")s?\b", re.IGNORECASE)
    ann = json.load(open(a.captions))["annotations"]
    first = {}
    for x in sorted(ann, key=lambda x: x["id"]):
        first.setdefault(x["image_id"], " ".join(x["caption"].split()))
    kept, dropped, hits = [], 0, {}
    for img in sorted(first):
        c = first[img]
        m = pat.search(c)
        if m:
            dropped += 1
            hits[m.group(1).lower()] = hits.get(m.group(1).lower(), 0) + 1
            continue
        kept.append(c)
        if len(kept) == a.n:
            break
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(kept, open(a.out, "w"), indent=0)
    info = dict(source=os.path.basename(a.captions), images=len(first), kept=len(kept), dropped_before_n=dropped,
                top_dropped_names=sorted(hits.items(), key=lambda kv: -kv[1])[:20], classes=a.classes)
    json.dump(info, open(a.out.replace(".json", ".meta.json"), "w"), indent=1)
    print(f"COCO PROMPTS: {len(kept)} captions kept, {dropped} dropped for an ImageNet-100 name "
          f"(most often: {info['top_dropped_names'][:5]}) -> {a.out}")


if __name__ == "__main__":
    main()
