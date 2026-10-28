"""Fast and Useful: adapting distilled text-to-image generators so that their images train better classifiers.

The repository root (configs/, data/, results/) is found next to the package, or at $FAU_ROOT; working data (real and
generated image packs, teacher pairs, adapters, classifier runs, set metrics) goes to $FAU_WORK (default ./work).
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("FAU_ROOT") or Path(__file__).resolve().parents[2])
WORK = Path(os.environ.get("FAU_WORK") or "work")
