"""The recorded measurements behind the paper (`results/`), with the run groups each table reads.

  sets.csv          one row per run group: top-1 mean and SD over its classifier runs (two decimals, as published),
                    the generation config and measured cost (NFE, H100 seconds per image) of its generated set, and the
                    set metrics (DINOv3 recall and Vendi, FID, aesthetic score, PickScore, recognizability)
  runs.csv          one row per classifier run: its group, seed and top-1
  per_class.json    per run group, the per-class top-1 of every classifier run (the input of every paired gap)
  ood.json          per classifier run, top-1 and per-class top-1 on ImageNet-V2, -Sketch, -R and -A
  transfer.json     per classifier run, linear-probe top-1 of its frozen features on Food-101, Cars, Pets and ImageNet-900
  metric_study.json the 186 template-prompt sets of the metric study: family, top-1 and every set metric
  costs.json        one-time preparation (H100-hours) of every adapter whose cost the paper reports
  diagnostics.json  the seed-relay similarities and the per-class recall of the per-class analysis

A pooled group (e.g. `kd_ad2_x3`) is every classifier run of its members (the same adapter trained three times);
its per-class record keys runs as `member:seed`.
"""
import csv
import json
import statistics
from pathlib import Path

POOLED = {   # one adapter trained three times (training seeds 0, 1, 2), every classifier run of each
    "kd_ad2_x3": ["kd_ad2", "kd_ad2b", "kd_ad2_s2"],
    "sdt_ad1_x3": ["sdt_ad1", "sdt_ad1_s1", "sdt_ad1_s2"],
    "zt_ad2_x3": ["zt_ad2", "zt_ad2_s1", "zt_ad2_s2"],
    "sdt_ad1_r128_x3": ["sdt_ad1_r128", "sdt_ad1_r128_s1", "sdt_ad1_r128_s2"],
    "dpdmd_ad0_x3": ["dpdmd_ad0", "dpdmd_ad0_s1", "dpdmd_ad0_s2"],
}
POOLED_DATASETS = {   # one sampler on three independently generated LLM-prompt datasets, every classifier run of each
    f"llmx_{arm}": [f"llm_{arm}", f"llm2_{arm}", f"llm3_{arm}"]
    for arm in ("klein", "kb_dd2_g1", "kd_ad2", "kd_llm", "sdt4", "sdt4_dd2_g1", "sdt_ad1", "sdt_llm")}


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class Results:
    """Read access to results/, recording every run group, OOD/transfer run and metric-study set that is read."""

    def __init__(self, root, configs=None):
        self.root = Path(root)
        self.configs = Path(configs) if configs else self.root.parent / "configs"
        self._sets = {r["group"]: r for r in csv.DictReader((self.root / "sets.csv").open())}
        self._runs = list(csv.DictReader((self.root / "runs.csv").open()))
        self._pc = json.loads((self.root / "per_class.json").read_text())
        self._cache = {}
        self.used = set()   # run groups and sets read; and the evaluations behind them (fau.plan)
        self.needs = dict(ood=set(), transfer=set(), metrics=set(), study=False, relay=False, recall=False)

    def _json(self, name):
        if name not in self._cache:
            self._cache[name] = json.loads((self.root / name).read_text())
        return self._cache[name]

    # Run groups ------------------------------------------------------------------------------------------------------
    def has(self, g):
        return g in self._sets

    def row(self, g):
        self.used.add(g)
        return self._sets[g]

    def mean(self, g):
        return num(self.row(g)["top1_mean"])

    def sd(self, g):
        return num(self.row(g)["top1_sd"])

    def col(self, g, c):
        if c in ("recall", "vendi", "fid", "recog", "aesthetic", "pickscore"):
            self.needs["metrics"].add(g)
        return num(self.row(g)[c])

    def runs(self, g):
        """Top-1 of each classifier run of a (not pooled) group, by increasing seed."""
        self.used.add(g)
        rs = sorted((r for r in self._runs if r["group"] == g), key=lambda r: int(r["seed"]))
        return [float(r["top1"]) for r in rs]

    def run(self, g, seed):
        self.used.add(g)
        return next(float(r["top1"]) for r in self._runs if r["group"] == g and int(r["seed"]) == seed)

    def per_class(self, g, classes=None):
        """{run: [per-class top-1]} of a group, optionally restricted to class indices."""
        self.used.add(g)
        pc = self._pc[g]
        return pc if classes is None else {s: [v[k] for k in classes] for s, v in pc.items()}

    def runs_from_runs_csv(self, g):
        """Mean and SD computed from the individual runs (the paper does so for two late reference cells)."""
        v = self.runs(g)
        return statistics.mean(v), statistics.stdev(v)

    # Other measurements ------------------------------------------------------------------------------------------------
    def _run_names(self, g, record):
        members = POOLED.get(g, [g])
        self.used.update(members)
        return sorted(r for r in record for m in members if r.rsplit("_s", 1)[0] == m and r.rsplit("_s", 1)[1].isdigit())

    def ood_top1(self, g, test):
        """Top-1 (points) of every run of group g on an OOD test set."""
        self.needs["ood"].add(g)
        ood = self._json("ood.json")
        return [100 * ood[r][test]["top1"] for r in self._run_names(g, ood) if test in ood[r]]

    def ood_per_class(self, g, test):
        self.needs["ood"].add(g)
        ood = self._json("ood.json")
        rs = [ood[r][test]["per_class"] for r in self._run_names(g, ood) if "per_class" in ood[r].get(test, {})]
        classes = set.intersection(*(set(r) for r in rs))
        return {c: statistics.mean(r[c] for r in rs) for c in classes}

    def transfer_top1(self, g, target):
        self.needs["transfer"].add(g)
        tr = self._json("transfer.json")
        v = [100 * tr[r][target]["top1"] for r in self._run_names(g, tr) if target in tr[r]]
        return statistics.mean(v), statistics.stdev(v)

    def transfer_per_class(self, g, target):
        self.needs["transfer"].add(g)
        tr = self._json("transfer.json")
        rs = [tr[r][target]["per_class"] for r in self._run_names(g, tr) if target in tr[r]]
        classes = set.intersection(*(set(r) for r in rs))
        return {c: statistics.mean(r[c] for r in rs) for c in classes}

    @property
    def costs(self):
        return self._json("costs.json")

    @property
    def metric_study(self):
        self.needs["study"] = True
        return self._json("metric_study.json")

    @property
    def diagnostics(self):
        self.needs["relay"] = self.needs["recall"] = True
        return self._json("diagnostics.json")

    def set_config(self, set_id):
        """A generated set's configuration (configs/sets.json, ids '<protocol>/<name>'): its generation arguments and
        the schedule it ran."""
        if "sets.json" not in self._cache:
            self._cache["sets.json"] = json.loads((self.configs / "sets.json").read_text())
        self.used.add(set_id.split("/", 1)[1])
        return self._cache["sets.json"][set_id]

    def classes(self, name):
        """A class list shipped with the code (data/classes/<name>.json)."""
        return json.loads((self.configs.parent / "data" / "classes" / f"{name}.json").read_text())
