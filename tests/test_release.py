"""Checks that need no GPU: the configurations reference each other completely, the recorded sampling schedules follow
from the sets' settings, and results/ rebuild the committed paper/ outputs exactly (a few minutes: the metric study).

    python tests/test_release.py            # or: pytest tests
"""
import json
import lzma
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {name: json.loads((ROOT / "configs" / f"{name}.json").read_text()) for name in ("sets", "runs", "adapters", "pairs")}


def test_references():
    """Every source, partner, adapter, teacher-pair set, class list and prompt file a configuration names exists."""
    sets, runs, ads, pairs = (CONFIGS[k] for k in ("sets", "runs", "adapters", "pairs"))
    for g, r in runs.items():
        for m in r.get("pooled", []):
            assert m in runs, (g, m)
        for src in r.get("train", []):
            kind, name = src.split(":")[:2]
            assert kind == "real" or name in sets, (g, src)
    prompts = {p.name.removesuffix(".json.xz") for p in (ROOT / "data" / "prompts").glob("*.json.xz")}
    for s, c in sets.items():
        assert (ROOT / "data" / "classes" / f"{c['classes']}.json").exists(), s
        other = c["args"].get("other") or ""
        if "_ad:" in other:
            assert other.split(":")[1] in ads, (s, other)
        assert c["args"].get("captions") in (None, *prompts), s
    for x, c in ads.items():
        assert c["pairs"] in pairs and c["prompts"] in (None, *prompts), x
    for p, c in pairs.items():
        assert c["prompts"] in (None, *prompts), p


def test_prompt_files():
    """Per-class prompt files hold enough descriptions for every set that uses them."""
    sets = CONFIGS["sets"]
    for name in {c["args"]["captions"] for c in sets.values() if c["args"].get("captions")}:
        with lzma.open(ROOT / "data" / "prompts" / f"{name}.json.xz", "rt") as f:
            caps = json.load(f)["captions"]
        need = max(c["args"]["per_class"] for c in sets.values() if c["args"].get("captions") == name)
        classes = {c["classes"] for c in sets.values() if c["args"].get("captions") == name}
        assert len(classes) == 1 and len(caps) == len(json.loads((ROOT / "data" / "classes" / f"{classes.pop()}.json").read_text())["wnids"])
        assert min(len(c) for c in caps) >= need, name


def test_schedules():
    """klein's noise levels and every stage plan follow from the recorded settings, as fau.generate computes them."""
    from fau.generate import native_sigmas, parse_sigmas, parse_stages
    checked = 0
    for s, c in CONFIGS["sets"].items():
        a, sched = c["args"], c["schedule"]
        if a.get("model") in ("klein", "klein_base") and sched.get("sigmas"):
            levels = parse_sigmas(a["sigmas"], 512, a["model"]) if a.get("sigmas") else native_sigmas(len(sched["sigmas"]) - 1, 512, a["model"])
            assert [round(v, 5) for v in levels] + [0.0] == sched["sigmas"], s
            checked += 1
        if a.get("stages") and sched.get("plan"):
            plan = parse_stages(a["stages"], len(sched["plan"]))
            assert [[w, g] for w, g in plan] == sched["plan"], s
    assert checked > 100


def test_paper():
    """results/ rebuild paper/tables and paper/numbers.tex exactly."""
    from fau import metric_study
    from fau.paper import numbers, tables
    from fau.paper.__main__ import compare, write
    from fau.paper.data import Results
    R = Results(ROOT / "results")
    inputs = R.metric_study["sets"]
    study, weighting = metric_study.analyze(inputs, 4), metric_study.family_weighting(inputs)
    nums = numbers.build(R, study, weighting)
    tabs = tables.build(R, study, weighting)
    with tempfile.TemporaryDirectory() as out:
        write(Path(out), tabs, nums)
        assert compare(Path(out), ROOT / "paper", tabs)


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print(f"{name}: ok", flush=True)
