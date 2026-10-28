"""Real-image captions as prompts: one caption per real training image, written by an open vision-language model
(Qwen2.5-VL-7B-Instruct, greedy decoding) with the instruction Adamkiewicz et al. (2026) gave GPT-4.1-nano. Generation
then prompts "<class name>, <caption>" for image (class k, index j), so every class gets as many distinct prompts as it
has real images.

The images are the first N per class of a real pack, in pack order (the same images as the real-data baseline
<pack>:0:N). Shards split the (class, index) list; each writes <out>.part<i>.json, and --merge joins them into <out>
({"model", "instruction", "captions": [[caption per index] per class]}).

The paper's file is data/prompts/in100_captions.json.xz; this module rebuilds it.

Usage: python -m fau.prompts.captions --pack work/real/in100_train --per_class 500 --out FILE.json --shard i --nshards n
       python -m fau.prompts.captions --out FILE.json --merge --nshards n
"""
import argparse, io, json, os, time

import numpy as np
import torch
from PIL import Image

from fau.hub import snapshot
from fau.packs import load_recs

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
INSTRUCTION = ("You are an expert image captioning model. You will write a detailed 5-6 sentence caption that describes "
               "background, foreground, objects and camera angle of the image.")


def jobs(pack, per_class):
    """[(class, index, offset, length)] for the first per_class records of every class, in pack order."""
    seen, out = {}, []
    for r in load_recs(pack):
        c = r["label"]
        if seen.get(c, 0) < per_class:
            out.append((c, seen.get(c, 0), r["off"], r["len"])); seen[c] = seen.get(c, 0) + 1
    return out


def merge(a):
    caps = {}
    for i in range(a.nshards):
        d = json.load(open(f"{a.out}.part{i}.json"))
        for c, j, t in d["captions"]:
            caps.setdefault(c, {})[j] = t
    ncls = max(caps) + 1
    full = [[caps[c][j] for j in range(len(caps[c]))] for c in range(ncls)]
    json.dump(dict(model=MODEL, instruction=INSTRUCTION, captions=full), open(a.out, "w"))
    print(f"MERGED {a.out}: {ncls} classes, {sum(len(x) for x in full)} captions", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack")
    ap.add_argument("--per_class", type=int, default=500)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--bs", type=int, default=128)   # generation is bound by per-step overhead: large batches amortize it
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.merge:
        return merge(a)
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    todo = jobs(a.pack, a.per_class)[a.shard::a.nshards]
    if a.limit:
        todo = todo[:a.limit]
    part = f"{a.out}.part{a.shard}.json"
    if os.path.exists(part):
        print(f"{part} exists, nothing to do"); return
    proc = AutoProcessor.from_pretrained(snapshot(MODEL), padding_side="left")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(snapshot(MODEL), dtype=torch.bfloat16, attn_implementation="sdpa",
                                                               disable_mmap=True).to("cuda").eval()
    mm = np.memmap(a.pack + ".bin", dtype=np.uint8, mode="r")
    msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": INSTRUCTION}]}]
    text = proc.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
    out, t0 = [], time.time()
    for b in range(0, len(todo), a.bs):
        chunk = todo[b:b + a.bs]
        imgs = [Image.open(io.BytesIO(mm[o:o + n].tobytes())).convert("RGB") for _, _, o, n in chunk]
        inp = proc(text=[text] * len(imgs), images=imgs, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            ids = model.generate(**inp, max_new_tokens=a.max_new_tokens, do_sample=False)
        caps = proc.batch_decode(ids[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        out += [(c, j, " ".join(t.split())) for (c, j, _, _), t in zip(chunk, caps)]
        if (b // a.bs) % 20 == 0:
            print(f"shard {a.shard}: {len(out)}/{len(todo)} captions, {(time.time() - t0) / len(out):.3f} s each; e.g. {out[-1][2][:160]}",
                  flush=True)
    json.dump(dict(captions=out), open(part + ".tmp", "w")); os.replace(part + ".tmp", part)
    print(f"CAPTION DONE shard {a.shard}: {len(out)} captions in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
