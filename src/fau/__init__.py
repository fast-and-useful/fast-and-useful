"""Fast and Useful: adapting distilled text-to-image generators so that their images train better classifiers.

The repository root (configs/, data/, results/) is found next to the package, or at $FAU_ROOT. Working data goes to
$FAU_WORK (default ./work):

  real/<name>                    real image packs (ImageNet subsets, fine-grained datasets, OOD test sets)
  sets/<protocol>/<set>/         generated sets (part*.bin / part*.json packs and gen.json)
  pairs/<id>/, adapters/<id>.pt  teacher pairs and fitted adapters
  runs/<group>_s<seed>/          classifier runs (metrics.json, model.pt)
  probes/                        set metrics, OOD and transfer accuracies
  cache/                         feature caches (cheap to recreate)
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("FAU_ROOT") or Path(__file__).resolve().parents[2])
WORK = Path(os.environ.get("FAU_WORK") or "work")
REAL, SETS, RUNS, PROBES, CACHE = (WORK / d for d in ("real", "sets", "runs", "probes", "cache"))
