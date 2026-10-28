"""SRPO as a diffusers FLUX transformer in bfloat16. tencent/SRPO ships FLUX.1-dev's transformer after SRPO preference
tuning as one float32 safetensors file (47.6 GB). It is read sequentially once (memory-mapped reads of large files can
be slow on network file systems), converted to bfloat16 and saved with save_pretrained, where generate.py loads it like
FLUX.1-dev's own transformer.

Usage: python -m fau.srpo [--out work/models/srpo-flux1-dev-bf16]     (generate.py reads $FAU_SRPO, else this default)
"""
import argparse, json, os, struct

import torch

from fau.generate import SRPO_DIR
from fau.hub import snapshot

DT = {"F32": torch.float32, "BF16": torch.bfloat16, "F16": torch.float16}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=SRPO_DIR)
    a = ap.parse_args()
    from diffusers import FluxTransformer2DModel
    path = os.path.join(snapshot("tencent/SRPO"), "diffusion_pytorch_model.safetensors")
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data = f.read()                  # one sequential read
    header.pop("__metadata__", None)
    sd = {}
    for k, v in header.items():
        s, e = v["data_offsets"]
        t = torch.frombuffer(bytearray(data[s:e]), dtype=DT[v["dtype"]]).reshape(v["shape"])
        sd[k] = t.to(torch.bfloat16)
    del data
    model = FluxTransformer2DModel.from_pretrained(snapshot("black-forest-labs/FLUX.1-dev"), subfolder="transformer",
                                                   torch_dtype=torch.bfloat16)
    model.load_state_dict(sd, strict=True)
    model.save_pretrained(a.out, max_shard_size="10GB")
    print(f"SRPO transformer: {len(sd)} tensors, bfloat16, saved to {a.out}")


if __name__ == "__main__":
    main()
