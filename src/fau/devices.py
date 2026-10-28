"""Where a set metric runs: the GPU when there is one, otherwise the CPU's cores in float32 (CPUs without bf16 or fp16
units). A set measured on both paths agrees closely; the paper's set metrics ran on both."""
import os

import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
WORKERS = 14 if DEV == "cuda" else 4   # loader workers; on a CPU node the rest of the cores compute
if DEV == "cpu":
    torch.set_num_threads(max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())) - WORKERS))


def half(dtype):
    """The reduced precision a probe uses on the GPU (bf16 or fp16); float32 on the CPU."""
    return dtype if DEV == "cuda" else torch.float32
