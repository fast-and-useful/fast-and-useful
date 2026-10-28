"""Collect a reproduction's measurements from $FAU_WORK into a results directory laid out as results/, so that
`python -m fau.paper --results DIR --out DIR/paper --compare paper` compares it with the paper, table by table.

  runs.csv, per_class.json   every classifier run of the run groups in configs/runs.json (work/runs/<group>_s<seed>)
  sets.csv                   per run group: top-1 mean and SD (two decimals), its generated set's measured cost (NFE,
                             H100 seconds per image, from the pack records) and set metrics (work/probes/in100/*.json)
  ood.json, transfer.json    work/probes/ood.json and work/probes/transfer.json
  metric_study.json          the metric study's sets with their top-1 and metrics
  costs.json                 preparation of the adapters whose cost the paper reports (pair and fit times recorded by
                             fau.adapter; the released file's job-level hours are kept where no time was recorded)
  diagnostics.json           work/probes/relay.json and the per-class recall of the per-class analysis

Run groups or files that are missing are reported and left out; tables that read them then fail with their name.

    python -m fau.collect [--out results-repro]
"""
import argparse
import csv
import glob
import json
import os
import shutil
import statistics

from fau import PROBES, ROOT, RUNS, SETS
from fau.paper.data import POOLED, POOLED_DATASETS

SET_METRICS = {   # sets.csv column -> (probe file in work/probes/in100, field)
    "recall": ("diversity_dinov3_vitl16.json", "recall"), "vendi": ("diversity_dinov3_vitl16.json", "vendi"),
    "fid": ("fid_inception.json", "fid"), "recog": ("recognizability.json", "recog"),
    "aesthetic": ("preference.json", "aesthetic"), "pickscore": ("preference.json", "pickscore"),
}
STUDY_METRICS = {   # metric-study key -> (probe file, field)
    "dino_recall": ("diversity_dinov3_vitl16.json", "recall"), "dino_coverage": ("diversity_dinov3_vitl16.json", "coverage"),
    "dino_precision": ("diversity_dinov3_vitl16.json", "precision"), "dino_density": ("diversity_dinov3_vitl16.json", "density"),
    "dino_vendi": ("diversity_dinov3_vitl16.json", "vendi"), "dino_fd": ("diversity_dinov3_vitl16.json", "frechet"),
    "clip_recall": ("clip_vitl14.json", "recall"), "clip_coverage": ("clip_vitl14.json", "coverage"),
    "clip_vendi": ("clip_vitl14.json", "vendi"), "clip_fd": ("clip_vitl14.json", "frechet"),
    "fid": ("fid_inception.json", "fid"), "inception_score": ("fid_inception.json", "inception_score"),
    "recog": ("recognizability.json", "recog"), "dino_linear": ("dinov3_vitl16.json", "linear"),
    "dino_knn": ("dinov3_vitl16.json", "knn"), "aesthetic": ("preference.json", "aesthetic"),
    "pickscore": ("preference.json", "pickscore"),
    **{k: ("image_stats.json", k) for k in ("slope", "hf", "saturation", "colorfulness", "contrast", "sharpness")},
}
PREP = {"kd_ad2": "pairs_in100", "sdt_ad1": "pairs_sd_in100", "kd_llm": "pairs_llm_klein", "sdt_llm": "pairs_llm_sd"}


def load(path, default=None):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return default


def set_cost(set_id):
    """(images, mean NFE, mean H100 seconds per image, resolution) of a generated set, from its pack records."""
    recs = [r for p in sorted(glob.glob(f"{SETS}/{set_id}/part*.json")) for r in load(p, [])]
    gen = load(f"{SETS}/{set_id}/gen.json", {})
    if not recs:
        return {}
    return dict(images=len(recs), nfe=round(statistics.mean(r["nfe"] for r in recs), 2),
                sec=round(statistics.mean(r["sec"] for r in recs), 4), resolution=gen.get("resolution", ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results-repro")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    configs = json.loads((ROOT / "configs" / "runs.json").read_text())
    probes = {f: load(PROBES / "in100" / f, {}) for f in {v[0] for v in [*SET_METRICS.values(), *STUDY_METRICS.values()]}}
    runs, per_class, rows, missing = [], {}, {}, []
    for g, cfg in configs.items():
        if "pooled" in cfg:
            continue
        got = []
        for seed in cfg["seeds"]:
            m = load(f"{RUNS}/{g}_s{seed}/metrics.json")
            if m is None:
                missing.append(f"{g}_s{seed}")
                continue
            got.append((seed, m))
            runs.append(dict(run=f"{g}_s{seed}", group=g, seed=seed, top1=round(100 * m["top1"], 2), protocol=cfg["val"].split("_")[0],
                             learner=m["args"].get("arch", "resnet50") if m["args"].get("recipe") != "prior" else "resnet50, prior recipe",
                             epochs=m["args"]["epochs"], minutes=round(m.get("minutes", 0), 1)))
        if not got:
            continue
        top = [round(100 * m["top1"], 2) for _, m in got]
        row = dict(group=g, protocol=cfg["val"].split("_")[0], n=len(top), seeds=",".join(str(s) for s, _ in got),
                   top1_mean=round(statistics.mean(top), 2), top1_sd=round(statistics.stdev(top), 2) if len(top) > 1 else "")
        sets = [t.split(":")[1] for t in cfg["train"] if t.startswith("set:")]
        if sets:
            row.update(set_cost(sets[0]))
            name = sets[0].split("/")[1]
            if sets[0].startswith("in100/"):
                for col, (f, field) in SET_METRICS.items():
                    row[col] = probes[f].get(name, {}).get(field, "")
        rows[g] = row
        per_class[g] = {str(s): [round(v, 4) for v in m["per_class"]] for s, m in got}
    for g, members in {**POOLED, **POOLED_DATASETS}.items():
        if g in configs and all(m in rows for m in members):
            top = [r["top1"] for m in members for r in runs if r["group"] == m]
            rows[g] = dict(rows[members[0]], group=g, n=len(top), top1_mean=round(statistics.mean(top), 2),
                           top1_sd=round(statistics.stdev(top), 2), **{c: "" for c in SET_METRICS})
            per_class[g] = {f"{m}:{s}": v for m in members for s, v in per_class[m].items()}
    cols = ["group", "protocol", "n", "seeds", "top1_mean", "top1_sd", "images", "nfe", "sec", "resolution", *SET_METRICS]
    with open(f"{a.out}/sets.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows.values())
    with open(f"{a.out}/runs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run", "group", "seed", "top1", "protocol", "learner", "epochs", "minutes"])
        w.writeheader()
        w.writerows(runs)
    json.dump(per_class, open(f"{a.out}/per_class.json", "w"), separators=(",", ":"))
    for f in ("ood.json", "transfer.json"):
        if os.path.exists(PROBES / f):
            shutil.copyfile(PROBES / f, f"{a.out}/{f}")
        else:
            missing.append(str(PROBES / f))
    study = json.loads((ROOT / "results" / "metric_study.json").read_text())
    for name, s in study["sets"].items():   # the study's frozen set list and families; top-1 and metrics measured anew
        s["top1"] = float(rows[name]["top1_mean"]) if name in rows else float("nan")
        s["metrics"] = {k: probes[f].get(name, {}).get(field, float("nan")) for k, (f, field) in STUDY_METRICS.items()}
    json.dump(study, open(f"{a.out}/metric_study.json", "w"), indent=1)
    costs = json.loads((ROOT / "results" / "costs.json").read_text())
    for adapter, pairs in PREP.items():
        fit = load(f"{SETS.parent}/adapters/{adapter}.pt.json", {}).get("seconds")
        shards = [load(p, {}).get("seconds") for p in glob.glob(f"{SETS.parent}/pairs/{pairs}/pairs*.json")]
        if fit and shards and all(shards):
            costs["preparation"][adapter].update(pairs=sum(shards) / 3600, fit=fit / 3600)
    json.dump(costs, open(f"{a.out}/costs.json", "w"), indent=1)
    div = probes["diversity_dinov3_vitl16.json"]
    diag = dict(relay=load(PROBES / "relay.json", {}),
                per_class_recall={g: div.get(g, {}).get("recall_per_class") for g in ("klein", "kd_ad2", "sdt4", "sdt_ad1")})
    json.dump(diag, open(f"{a.out}/diagnostics.json", "w"), indent=1)
    if os.path.isdir(ROOT / "results" / "figure-images"):
        shutil.copytree(ROOT / "results" / "figure-images", f"{a.out}/figure-images", dirs_exist_ok=True)
    print(f"COLLECT: {len(runs)} classifier runs, {len(rows)} run groups -> {a.out}; {len(missing)} missing"
          + (f": {', '.join(missing[:12])}{' ...' if len(missing) > 12 else ''}" if missing else ""))


if __name__ == "__main__":
    main()
