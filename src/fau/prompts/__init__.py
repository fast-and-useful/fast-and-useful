"""Class lists and prompt files.

Class lists (data/classes/<name>.json): the ordered classes of a protocol, as wnids and names (ImageNet's synonyms, or
the fine-grained datasets' disambiguated names) with each class's ImageNet-1k index where it has one.
Prompt files (data/prompts/<name>.json.xz): {"captions": [[one description per image index] per class]} for the
real-image captions and LLM descriptions, or a list of captions (COCO) for fitting an adapter on captions.
"""
import json
import lzma
import os

from fau import ROOT


def class_list(ref):
    """A class list by name (in100, tin200, in1k, food101, cars, pets) or path."""
    path = ref if os.path.isfile(ref) else ROOT / "data" / "classes" / f"{ref}.json"
    return json.loads(open(path).read())


def load_prompts(ref):
    """A prompt file by name (data/prompts/<name>.json.xz) or path (.json or .json.xz)."""
    path = ref if os.path.isfile(ref) else ROOT / "data" / "prompts" / f"{ref}.json.xz"
    opener = lzma.open if str(path).endswith(".xz") else open
    with opener(path, "rt") as f:
        return json.load(f)
