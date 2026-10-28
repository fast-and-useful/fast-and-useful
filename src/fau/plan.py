"""What it takes to reproduce a table, a figure or the numbers in the text: every command, in order, with its recorded
H100 cost. The run groups, sets and evaluations are traced by building the target from results/ and recording what it
reads; configs/ then gives the sets, adapters and teacher pairs behind them.

    python -m fau.plan t2-ablation                 # a table (file name in paper/tables, or its label: tab:ablation)
    python -m fau.plan f3-comparison numbers       # a figure, and the numbers quoted in the text
    python -m fau.plan all --summary               # everything, as counts and H100-hours only

Each command needs one GPU (dataset packing and evaluations aside); the loops over shards and seeds are independent jobs,
so a scheduler can run them in parallel. Costs are the
recorded H100 hours of generation, classifier training and adapter preparation (where recorded), excluding model loading.
"""
import argparse
import csv
import json
import tempfile
from collections import OrderedDict

from fau import ROOT
from fau.paper import figures, numbers, tables
from fau.paper.data import POOLED, POOLED_DATASETS, Results

LABELS = {   # the paper's labels -> table and figure files
    "tab:adapter-config": "t0-method-config", "tab:main": "t1-scale-transfer", "tab:ablation": "t2-ablation",
    "tab:finegrained": "t3-finegrained", "tab:learners": "t5-learners", "tab:prompts": "t6-prompts",
    "tab:supp-deployment": "s14-deployment", "tab:supp-e50-costs": "s-e50-costs", "tab:supp-regeneration": "s-dataset-regeneration",
    "tab:supp-secondary-generators": "s-additional-generators", "tab:supp-main-complete": "t1-main", "tab:supp-dpdmd": "s10-dpdmd",
    "tab:supp-unseen-fullscale": "s12-unseen-fullscale", "tab:supp-crossdata": "s-crossdata-comparisons",
    "tab:supp-finegrained-references": "s-finegrained-references", "tab:transfer": "s9-transfer",
    "tab:supp-learners": "s-learner-references", "tab:supp-captions": "s5-captions", "tab:supp-e50-replication": "s-e50-replication",
    "tab:supp-e50-fitting": "s-e50-prompt-fitting", "tab:supp-e50-in1k": "s-e50-in1k", "tab:supp-ood": "s6-ood",
    "tab:supp-metrics": "s3-metrics", "tab:supp-heldout": "s11-heldout", "tab:supp-family-weighting": "s13-family-weighting",
    "tab:supp-quality": "s7-quality", "tab:supp-contrasts": "s15-contrasts", "fig:attribution": "f3-comparison",
    "fig:cost": "f4-generation-cost", "fig:window": "f3a-guidance-window", "fig:handoff": "f3b-handoff", "fig:method": "f5-method",
    "fig:supp-cost-full": "s-additional-operating-points", "fig:supp-scaling": "s-scaling"}
TABLES = {"t0-method-config": tables.t0_method_config, "t1-scale-transfer": tables.t1_scale_transfer,
          "t2-ablation": tables.t2_ablation, "t3-finegrained": tables.t3_finegrained, "t5-learners": tables.t5_learners,
          "t6-prompts": tables.t6_prompts, "s14-deployment": tables.s14_deployment, "s-e50-costs": tables.s_e50_costs,
          "s-dataset-regeneration": tables.s_dataset_regeneration, "s-additional-generators": tables.s_additional_generators,
          "t1-main": tables.t1_main, "s10-dpdmd": tables.s10_dpdmd, "s12-unseen-fullscale": tables.s12_unseen_fullscale,
          "s-crossdata-comparisons": tables.s_crossdata_comparisons, "s-finegrained-references": tables.s_finegrained_references,
          "s9-transfer": tables.s9_transfer, "s-learner-references": tables.s_learner_references, "s5-captions": tables.s5_captions,
          "s-e50-replication": tables.s_e50_replication, "s-e50-prompt-fitting": tables.s_e50_prompt_fitting,
          "s-e50-in1k": tables.s_e50_in1k, "s6-ood": tables.s6_ood, "s7-quality": tables.s7_quality,
          "s15-contrasts": tables.s15_contrasts}
STUDY = {"s3-metrics", "s11-heldout", "s13-family-weighting"}   # the metric study: its 186 sets, runs and set metrics
FIGURES = {"f3-comparison": figures.fig_comparison, "f4-generation-cost": figures.fig_cost,
           "s-additional-operating-points": figures.fig_operating_points,
           "f3a-guidance-window": lambda R, out: figures.fig_switch(R, out, figures.WINDOW, "f3a-guidance-window", ""),
           "f3b-handoff": lambda R, out: figures.fig_switch(R, out, figures.HANDOFF, "f3b-handoff", ""),
           "f5-method": lambda R, out: figures.fig_method(R, R.root / "figure-images", out), "s-scaling": figures.fig_scaling}
RELAY_SETS = ["kd_ad2", "klein", "kb_dd2_g1", "kb_or34_d1", "sdt_ad1", "sdt4", "sdt4_dd2_g1", "sdt_or30_d2"]
STUB_STUDY = dict(metrics={"dino_recall": dict(n=186)})
STUB_WEIGHTING = dict(clip_fd_minus_dino_recall=dict(value=0.0, interval=[0.0, 0.0]))


def trace(target):
    """Build a target from results/ -> the Results that recorded what it read."""
    R = Results(ROOT / "results")
    name = LABELS.get(target, target)
    if name in STUDY:
        R.needs["study"] = True
    elif name in TABLES:
        TABLES[name](R)
    elif name in FIGURES:
        figures.style()
        with tempfile.TemporaryDirectory() as d:
            FIGURES[name](R, d)
    elif name == "numbers":
        numbers.build(R, STUB_STUDY, STUB_WEIGHTING)
        R.needs["study"] = True   # the text quotes the metric study (its set count and the family-weighting gap)
    else:
        raise SystemExit(f"unknown target {target}: a table ({', '.join(TABLES)}, {', '.join(STUDY)}), a figure "
                         f"({', '.join(FIGURES)}), a label (tab:..., fig:...), numbers, or all")
    return R


def gpu_minutes(runs):
    """{(group, seed): H100 minutes} of the recorded classifier runs; a run made on CPU nodes counts as the median H100
    run of its learner, protocol and length."""
    key = lambda r: (r["learner"], r["protocol"], r["epochs"])
    gpu = {}
    for r in runs:
        if r["device"] != "cpu":
            gpu.setdefault(key(r), []).append(float(r["minutes"]))
    median = {k: sorted(v)[len(v) // 2] for k, v in gpu.items()}
    return {(r["group"], int(r["seed"])): float(r["minutes"]) if r["device"] != "cpu" else median.get(key(r), 0.0) for r in runs}


def repos(sets, pairs_cfg, pairs, needs, metric_sets):
    """The Hugging Face repositories a reproduction loads (generate.WEIGHTS, the adapters' teachers, the metric models)."""
    from fau.generate import AD_BASE, SDA, WEIGHTS
    from fau.hub import MODELS
    teachers = {"klein": ["klein_base", "klein"], "sd": ["sd21", "sd_turbo"], "zimage": ["zimage", "zimage_turbo"],
                "sd35m": ["sd35_medium", "dpdmd_sd35m"]}
    models = {f for p in pairs for f in teachers[pairs_cfg[p]["family"]]}
    out = set()
    for s in sets:
        args = cfg("sets")[s]["args"]
        models.add(args.get("model", "klein"))
        kind, _, ref = (args.get("other") or "").partition(":")
        models.add(AD_BASE.get(kind, "zimage_turbo" if kind == "zt_lokr" else kind))
        if kind == "zt_lokr":
            out.add(SDA[0])
        if args.get("method") == "mono":
            out.add("facebook/dinov2-base")
    if "srpo" in models:
        out |= {"black-forest-labs/FLUX.1-dev", "tencent/SRPO"}
    out |= {repo for m in models if m in WEIGHTS for repo, _ in WEIGHTS[m] if repo in MODELS}
    if metric_sets:
        out |= {"facebook/dinov3-vitl16-pretrain-lvd1689m"}
    if needs["study"] or needs["metrics"]:
        out |= {"openai/clip-vit-large-patch14", "trl-lib/ddpo-aesthetic-predictor", "yuvalkirstain/PickScore_v1",
                "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"}
    return sorted(out)


def traced(record):
    """The run groups of a tracer record: groups, or (group, column / test / target) cells."""
    return {x if isinstance(x, str) else x[0] for x in record}


def loop(cmd, flag, values):
    """A command once per value (shards, seeds): a shell loop, or the command itself for a single shard."""
    values = list(values)
    if values == [0] and flag == "--shard":
        return cmd
    return f"for i in {' '.join(map(str, values))}; do {cmd} {flag} $i; done"


def cfg(name):
    return json.loads((ROOT / "configs" / f"{name}.json").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="+")
    ap.add_argument("--summary", action="store_true", help="counts and H100-hours only")
    a = ap.parse_args()
    targets = [*TABLES, *STUDY, *FIGURES, "numbers"] if a.targets == ["all"] else a.targets
    runs_cfg, sets_cfg, ads_cfg, pairs_cfg = cfg("runs"), cfg("sets"), cfg("adapters"), cfg("pairs")
    groups, needs = set(), dict(ood=set(), transfer=set(), metrics=set(), timing=set(), study=False, relay=False, recall=False)
    for t in targets:
        R = trace(t)   # groups whose accuracy the target reads, or whose classifiers it evaluates (OOD, transfer)
        groups |= {g for g in R.needs["top1"] | traced(R.needs["ood"]) | traced(R.needs["transfer"]) if g in runs_cfg}
        for k in needs:   # the groups behind the evaluations and the measured generation times
            needs[k] = needs[k] | (traced(R.needs[k]) if isinstance(needs[k], set) else R.needs[k])
    study_sets = list(json.loads((ROOT / "results" / "metric_study.json").read_text())["sets"]) if needs["study"] else []
    groups |= {g for g in study_sets if g in runs_cfg}
    members = {m for g in groups for m in {**POOLED, **POOLED_DATASETS}.get(g, [g])}
    rows = {r["group"]: r for r in csv.DictReader((ROOT / "results" / "sets.csv").open())}
    minutes = gpu_minutes(list(csv.DictReader((ROOT / "results" / "runs.csv").open())))

    # what the runs train on, and what the evaluations read
    sets, real = OrderedDict(), set()
    for g in sorted(members):
        for src in runs_cfg[g]["train"]:
            kind, name = src.split(":")[:2]
            (sets.setdefault(name, g) if kind == "set" else real.add(name))
        real.add(runs_cfg[g]["val"])

    def train_set(g):   # the generated set a run group trains on
        src = runs_cfg.get(g, {}).get("train", [""])[0]
        return src.split(":")[1] if src.startswith("set:") else None
    metric_sets = sorted(({f"in100/{s}" for s in study_sets} | {train_set(g) for g in needs["metrics"]}) - {None})
    if needs["recall"]:
        metric_sets = sorted(set(metric_sets) | {f"in100/{s}" for s in ("klein", "kd_ad2", "sdt4", "sdt_ad1")})
    relay_sets = [f"in100/{s}" for s in RELAY_SETS] if needs["relay"] else []
    timed_sets = sorted({train_set(g) for g in needs["timing"]} - {None})   # generated to measure their time per image
    for s in [*metric_sets, *relay_sets, *timed_sets]:
        sets.setdefault(s, None)
    if metric_sets or relay_sets:
        real.add("in100_train")
    if needs["transfer"]:
        real |= {"food101_train", "food101_test", "cars_train", "cars_test", "pets_train", "pets_test", "in1k_train", "in1k_val"}
    ood_lists = sorted({runs_cfg[m]["val"].split("_")[0] for g in needs["ood"] for m in {**POOLED}.get(g, [g])})
    adapters = sorted({sets_cfg[s]["args"]["other"].split(":")[1] for s in sets
                       if "_ad:" in (sets_cfg[s]["args"].get("other") or "")})
    pairs = sorted({ads_cfg[x]["pairs"] for x in adapters})

    # recorded costs (H100-hours)
    def set_hours(s):
        g = next((g for g, r in rows.items() if r.get("images") and r.get("sec") and runs_cfg.get(g, {}).get("train", [""])[0]
                  .startswith(f"set:{s}:")), None)
        return float(rows[g]["images"]) * float(rows[g]["sec"]) / 3600 if g else None
    gen = {s: set_hours(s) for s in sets}
    train = {(g, s): minutes.get((g, s), 0) / 60 for g in sorted(members) for s in runs_cfg[g]["seeds"]}
    costs = json.loads((ROOT / "results" / "costs.json").read_text())["preparation"]   # keyed by the adapter's set
    prep = {x: sum(v.values()) for k, v in costs.items() for x in [sets_cfg[f"in100/{k}"]["args"]["other"].split(":")[1]]
            if x in adapters}
    total = sum(v for v in gen.values() if v) + sum(train.values()) + sum(prep.values())
    unknown = [s for s, v in gen.items() if v is None]
    print(f"# {' '.join(targets) if len(targets) < 6 else 'all'}: {len(members)} run groups ({len(train)} classifier runs), "
          f"{len(sets)} generated sets, {len(adapters)} adapters on {len(pairs)} teacher-pair sets; about {total:.0f} H100-hours "
          f"recorded (generation {sum(v for v in gen.values() if v):.0f}, classifiers {sum(train.values()):.0f}, adapter "
          f"preparation {sum(prep.values()):.0f}" + (f"; {len(unknown)} set{'s' * (len(unknown) > 1)} without a recorded time" if unknown else "") + ")")
    if a.summary:
        return
    if not sets:
        print("\nNothing to generate or train: python -m fau.paper rebuilds this target from configs/ and results/.")
        return

    print("\n## 1. Models and data (python -m fau.hub check lists what is missing)")
    print("python -m fau.hub download " + " ".join(repos(sets, pairs_cfg, pairs, needs, metric_sets)))
    if any(n.startswith(("in100", "tin200", "in1k")) for n in real):
        print("python -m fau.hub download ILSVRC/imagenet-1k")
    for name in sorted(real):
        if name.startswith(("in100", "tin200", "in1k")) and "_ood_" not in name:
            protocol, split = name.split("_", 1)
            split = "validation" if split == "val" else "train"
            classes = {"in100": "in100", "tin200": "tin200", "in1k": "all"}[protocol]
            print(f"python -m fau.imagenet --split {split} --classes {classes} --out work/real/{name}" + (" --workers 16" if split == "train" else ""))
    fine = sorted({n.split("_")[0] for n in real if n.startswith(("food101", "cars", "pets"))})
    if fine:
        print(f"python -m fau.finegrained pack --sets {' '.join(fine)}")
    for c in ood_lists:
        print(f"python -m fau.ood pack --classes {c}")
    if pairs:
        print("\n## 2. Teacher pairs and adapters (one GPU per shard)")
        for p in pairs:
            print(loop(f"python -m fau.adapter pairs --config {p}", "--shard", range(pairs_cfg[p]["shards"])))
        for x in adapters:
            print(f"python -m fau.adapter train --config {x}" + (f"   # {prep[x]:.1f} h of recorded preparation" if x in prep else ""))
    if any(sets_cfg[s]["args"].get("model") == "srpo" or sets_cfg[s]["args"].get("other") == "srpo" for s in sets):
        print("python -m fau.srpo   # SRPO's transformer in bfloat16, once")
    print("\n## 3. Generated sets (one GPU per shard)")
    for s in sets:
        c = sets_cfg[s]
        module = "baselines" if c["generator"] == "baselines" else "generate"
        print(loop(f"python -m fau.{module} --set {s}", "--shard", range(c["shards"])) + (f"   # {gen[s]:.1f} h" if gen[s] else ""))
    if members:
        print("\n## 4. Classifiers (one GPU per run)")
    for g in sorted(members):
        seeds = runs_cfg[g]["seeds"]
        print(loop(f"python -m fau.train --config {g}", "--seed", seeds) + f"   # {sum(train[(g, s)] for s in seeds):.1f} h")
    if needs["ood"] or needs["transfer"] or metric_sets or relay_sets:
        print("\n## 5. Evaluations")
    run_names = lambda gs: " ".join(f"{m}_s{s}" for g in sorted(gs) for m in {**POOLED}.get(g, [g]) for s in runs_cfg[m]["seeds"])
    if needs["ood"]:
        print(f"python -m fau.ood eval --runs {run_names(needs['ood'])}")
    if needs["transfer"]:
        print(f"python -m fau.transfer --runs {run_names(needs['transfer'])}")
    if metric_sets or relay_sets:
        specs = " ".join(f"work/sets/{s}" for s in metric_sets)
        real500 = "real500=work/real/in100_train:0:500"
        print(f"python -m fau.metrics.diversity --sets {specs} --real {real500} --cache work/cache "
              f"--out work/probes/in100/diversity_dinov3_vitl16.json")
        if needs["study"] or needs["metrics"]:
            print(f"python -m fau.metrics.extra --sets {specs} --real {real500} --val work/real/in100_val --out-dir work/probes/in100 "
                  f"--cache work/cache")
            print(f"python -m fau.metrics.preference --sets {specs} --out work/probes/in100/preference.json")
        if needs["study"]:
            print(f"python -m fau.metrics.frozen --sets {specs} --val work/real/in100_val --cache work/cache "
                  f"--out work/probes/in100/dinov3_vitl16.json")
            print(f"python -m fau.metrics.image_stats --sets {specs} --out work/probes/in100/image_stats.json")
        if relay_sets:
            print(f"python -m fau.metrics.frozen --sets {' '.join(f'work/sets/{s}' for s in relay_sets)} --val work/real/in100_val "
                  f"--cache work/cache --out work/probes/in100/dinov3_vitl16.json")
            print("python -m fau.metrics.relay --cache work/cache/dinov3_vitl16 --out work/probes/relay.json --pairs "
                  "kd_ad2:kb_or34_d1 klein:kb_or34_d1 kb_dd2_g1:kb_or34_d1 sdt_ad1:sdt_or30_d2 sdt4:sdt_or30_d2 sdt4_dd2_g1:sdt_or30_d2")
    print("\n## 6. Results and comparison with the paper")
    print("python -m fau.collect --out results-repro")
    print("python -m fau.paper --results results-repro --out results-repro/paper --compare paper")


if __name__ == "__main__":
    main()
