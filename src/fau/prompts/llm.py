"""Rich prompts without real images: an LLM (Qwen2.5-VL-7B in text-only mode, the same model that wrote the image
captions) writes `per_class` varied photo descriptions per class, in the format of captions.py so generate.py --captions
takes them. Variety comes from sampling (temperature 1) and from a scene hint drawn per request from fixed lists (setting,
viewpoint, distance, light), the way prompt-diversification baselines work; no real image is looked at.

--start S draws a pool disjoint from another's: prompt indices S..S+per_class-1 per class, so other scene hints, and
another sampling seed per shard; the file still numbers its prompts from 0. The last shard to finish merges the parts
(it looks for the others' parts for 100 s: a network file system can hide a fresh file from another node).

The paper's files are data/prompts/in100_llm*.json.xz and in1k_llm.json.xz; this module rebuilds them.

Usage: python -m fau.prompts.llm --classes in100 --per_class 500 [--start S] --out FILE.json --shard i --nshards n
       python -m fau.prompts.llm --out FILE.json --merge --nshards n [--start S] (by hand, if a merge is missing)
"""
import argparse, json, os, random, time

import torch

from fau.generate import class_name
from fau.hub import snapshot
from fau.prompts import class_list
from fau.prompts.captions import MODEL

SETTINGS = ["indoors", "outdoors in nature", "in a city street", "in a home", "in a shop or market", "on a table",
            "in a field", "near water", "in a workshop", "in a garden", "in a studio", "in the wild"]
VIEWS = ["from the front", "from the side", "from above", "from a low angle", "at an angle", "from behind"]
DISTANCES = ["in a close-up", "at medium distance", "far away, small in the frame", "partly cut off by the frame"]
LIGHT = ["in daylight", "in dim light", "at night", "in harsh sunlight", "under artificial light", "on an overcast day"]
INSTRUCTION = ("Write a detailed 3-4 sentence description of a realistic photograph of a {name}, {setting}, seen {view}, "
               "{distance}, {light}. Describe the background, the foreground, other objects and the camera angle. "
               "Write only the description.")


def merge(a):
    caps = {}
    for i in range(a.nshards):
        for c, j, t in json.load(open(f"{a.out}.part{i}.json"))["captions"]:
            caps.setdefault(c, {})[j] = t
    full = [[caps[c][j] for j in range(len(caps[c]))] for c in range(max(caps) + 1)]
    meta = dict(model=MODEL, instruction=INSTRUCTION, hints=dict(settings=SETTINGS, views=VIEWS, distances=DISTANCES, light=LIGHT))
    if a.start:   # the first pool (start 0) keeps its original fields
        meta.update(start=a.start, classes=a.classes)
    tmp = f"{a.out}.{os.getpid()}.tmp"   # two shards finishing together both merge: each writes its own file, then renames
    json.dump(dict(meta, captions=full), open(tmp, "w")); os.replace(tmp, a.out)
    print(f"MERGED {a.out}: {len(full)} classes, {sum(len(x) for x in full)} prompts", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes")
    ap.add_argument("--per_class", type=int, default=500)
    ap.add_argument("--start", type=int, default=0, help="first prompt index per class: a pool disjoint from start 0's")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.merge:
        return merge(a)
    if not os.path.exists(a.out) and all(os.path.exists(f"{a.out}.part{i}.json") for i in range(a.nshards)):
        return merge(a)   # a rerun after the parts were written but the merge was lost
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    meta = class_list(a.classes)
    names = [class_name(w, n) for w, n in zip(meta["wnids"], meta["names"])]
    jobs = [(c, j) for c in range(len(names)) for j in range(a.start, a.start + a.per_class)][a.shard::a.nshards]
    part = f"{a.out}.part{a.shard}.json"
    if os.path.exists(part):
        print(f"{part} exists, nothing to do"); return
    torch.manual_seed(a.shard + 1000 * a.start)
    proc = AutoProcessor.from_pretrained(snapshot(MODEL), padding_side="left")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(snapshot(MODEL), dtype=torch.bfloat16, attn_implementation="sdpa",
                                                               disable_mmap=True).to("cuda").eval()
    out, t0 = [], time.time()
    for b in range(0, len(jobs), a.bs):
        chunk = jobs[b:b + a.bs]
        texts = []
        for c, j in chunk:
            r = random.Random(1_000_003 * c + j)   # the hint is fixed per (class, index), whatever the sharding
            msg = [{"role": "user", "content": [{"type": "text", "text": INSTRUCTION.format(
                name=names[c], setting=r.choice(SETTINGS), view=r.choice(VIEWS), distance=r.choice(DISTANCES), light=r.choice(LIGHT))}]}]
            texts.append(proc.apply_chat_template(msg, add_generation_prompt=True, tokenize=False))
        inp = proc(text=texts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            ids = model.generate(**inp, max_new_tokens=a.max_new_tokens, do_sample=True, temperature=1.0, top_p=0.95)
        caps = proc.batch_decode(ids[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        out += [(c, j - a.start, " ".join(t.split())) for (c, j), t in zip(chunk, caps)]
        if (b // a.bs) % 10 == 0:
            print(f"shard {a.shard}: {len(out)}/{len(jobs)} prompts, {(time.time() - t0) / len(out):.3f} s each; e.g. {out[-1][2][:160]}", flush=True)
    json.dump(dict(captions=out), open(part + ".tmp", "w")); os.replace(part + ".tmp", part)
    print(f"PROMPTS DONE shard {a.shard}: {len(out)} prompts in {time.time() - t0:.0f} s", flush=True)
    # The last shard to finish merges. On NFS a part written on another node can stay invisible for up to a minute (a
    # cached negative lookup): shards that end within seconds of each other can each miss the others' parts. So a shard
    # that misses a part looks again for 100 s; merging twice is harmless.
    for _ in range(10):
        if os.path.exists(a.out):
            break
        if all(os.path.exists(f"{a.out}.part{i}.json") for i in range(a.nshards)):
            merge(a)
            break
        time.sleep(10)


if __name__ == "__main__":
    main()
