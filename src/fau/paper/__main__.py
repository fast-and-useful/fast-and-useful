"""Rebuild every table, number and figure of the paper from the recorded measurements, and compare them.

    python -m fau.paper                                  # results/ -> paper/ (tables/, numbers.tex, figures/)
    python -m fau.paper --results my-results --out my-paper --compare paper   # your reproduction against the paper

--compare lists every table cell and every number that differs between the two builds (a LaTeX source tree with a
main.tex works too: its numbers*.tex files are read in input order). The metric study's bootstrap takes a few minutes;
--workers runs its leave-one-family-out fits in parallel.
"""
import argparse
import json
import re
import time
from pathlib import Path

from fau import metric_study
from fau.paper import figures, numbers, tables
from fau.paper.data import Results

ROOT = Path(__file__).resolve().parents[3]
HEADER = r"""% Every number the paper's text quotes; written by `python -m fau.paper` from results/. Do not edit.
\makeatletter
\newcommand{\tp}[1]{\@ifundefined{tp@#1}{\textbf{??#1}}{\csname tp@#1\endcsname}}
\newcommand{\tpsd}[1]{\@ifundefined{tpsd@#1}{\textbf{??#1}}{\csname tpsd@#1\endcsname}}
\newcommand{\tpsec}[1]{\@ifundefined{tpsec@#1}{\textbf{??#1}}{\csname tpsec@#1\endcsname}}
\newcommand{\gp}[1]{\@ifundefined{gp@#1}{\textbf{??#1}}{\csname gp@#1\endcsname}}
\newcommand{\gpci}[1]{\@ifundefined{gpci@#1}{\textbf{??#1}}{\csname gpci@#1\endcsname}}
\newcommand{\gpa}[1]{\@ifundefined{gpa@#1}{\textbf{??#1}}{\csname gpa@#1\endcsname}}
\newcommand{\gpcia}[1]{\@ifundefined{gpcia@#1}{\textbf{??#1}}{\csname gpcia@#1\endcsname}}
\newcommand{\nm}[1]{\@ifundefined{nm@#1}{\textbf{??#1}}{\csname nm@#1\endcsname}}
"""
MACRO = re.compile(r"\\csname ([^\\]+)\\endcsname\{(.*)\}\s*$")
USE = re.compile(r"\\(tp|tpsd|tpsec|gp|gpci|gpa|gpcia|nm)\{([^}]+)\}")


def write(out, tabs, nums):
    (out / "tables").mkdir(parents=True, exist_ok=True)
    for name, body in tabs.items():
        (out / "tables" / f"{name}.tex").write_text(f"% Written by `python -m fau.paper` from results/. Do not edit.\n{body}")
    lines = [HEADER.rstrip("\n")] + [rf"\expandafter\def\csname {k}\endcsname{{{v}}}" for k, v in sorted(nums.items())]
    (out / "numbers.tex").write_text("\n".join(lines + [r"\makeatother"]) + "\n")


def read_macros(tree):
    """{kind@name: value} of a build (numbers.tex) or of a LaTeX tree (its numbers*.tex in main.tex's input order)."""
    tree = Path(tree)
    docs = [tree / f"{d}.tex" for d in ("main", "supp") if (tree / f"{d}.tex").exists()]
    files = list(dict.fromkeys(f for d in docs for f in re.findall(r"\\input\{(numbers[^}]*)\}", d.read_text()))) or ["numbers"]
    out = {}
    for f in files:
        for line in (tree / f"{f}.tex").read_text().splitlines():
            m = MACRO.search(line)
            if m:
                out[m[1]] = m[2]   # a later file overrides an earlier one, as in LaTeX
    return out


def table_rows(path):
    """A table's lines without comments, for a cell-by-cell comparison."""
    return [line for line in Path(path).read_text().splitlines() if line.strip() and not line.lstrip().startswith("%")]


QUALITATIVE = {"s-method-positioning"}   # tables without measurements


def paper_inputs(tree):
    """The number macros and tables the documents of a LaTeX tree use: main.tex (and supp.tex) and every file they
    input, comments ignored -> ({kind@name}, {table name})."""
    tree, out, tables, seen = Path(tree), set(), set(), set()

    def visit(name):
        f = tree / (name if name.endswith(".tex") else name + ".tex")
        if f in seen or not f.exists():
            return
        seen.add(f)
        text = re.sub(r"(?<!\\)%.*", "", f.read_text(errors="ignore"))
        if not f.name.startswith("numbers"):
            out.update(f"{k}@{n}" for k, n in USE.findall(text))
        for child in re.findall(r"\\input\{([^}]+)\}", text):
            if child.startswith("tables/"):
                tables.add(Path(child).stem)
            visit(child)
    for doc in ("main", "supp"):
        visit(doc)
    return out, tables - QUALITATIVE


def compare(build, other, names):
    build, other = Path(build), Path(other)
    differ = 0
    for name in sorted(names):
        a, b = table_rows(build / "tables" / f"{name}.tex"), table_rows(other / "tables" / f"{name}.tex")
        rows = [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
        if len(a) != len(b) or rows:
            differ += 1
            print(f"TABLE {name}: {len(rows)} rows differ" + (f", {len(a)} vs {len(b)} lines" if len(a) != len(b) else ""))
            for i, x, y in rows:
                print(f"  row {i}:\n    {x}\n    {y}")
    ours, theirs = read_macros(build), read_macros(other)
    wrong = {k: (v, theirs.get(k)) for k, v in ours.items() if theirs.get(k) != v}
    for k, (v, w) in sorted(wrong.items()):
        print(f"NUMBER {k}: {v} vs {w}")
    missing = []
    if (other / "main.tex").exists():
        macros, tables = paper_inputs(other)
        missing = sorted(macros - set(ours)) + sorted(f"table {t}" for t in tables - set(names))
    for k in missing:
        print(f"MISSING {k}: used by the paper, not built")
    print(f"COMPARE: {len(names) - differ}/{len(names)} tables identical, {len(ours) - len(wrong)}/{len(ours)} numbers "
          f"identical" + (f", {len(missing)} used by the paper but not built" if missing else ""))
    return differ == 0 and not wrong and not missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=ROOT / "results", type=Path)
    ap.add_argument("--configs", default=ROOT / "configs", type=Path)
    ap.add_argument("--out", default=ROOT / "paper", type=Path)
    ap.add_argument("--compare", type=Path, help="another build, or a LaTeX tree, to compare against")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--used", type=Path, help="write the run groups the build reads to this JSON file")
    a = ap.parse_args()
    t0 = time.time()
    R = Results(a.results, a.configs)
    inputs = R.metric_study["sets"]
    study = metric_study.analyze(inputs, a.workers)
    weighting = metric_study.family_weighting(inputs)
    nums = numbers.build(R, study, weighting)
    tabs = tables.build(R, study, weighting, nums)
    write(a.out, tabs, nums)
    if not a.no_figures:
        figures.build(R, a.out / "figures")
    print(f"PAPER: {len(tabs)} tables, {len(nums)} numbers{'' if a.no_figures else ', 7 figures'} -> {a.out} "
          f"({time.time() - t0:.0f} s)")
    if a.used:
        a.used.write_text(json.dumps(sorted(R.used), indent=1) + "\n")
    if a.compare:
        raise SystemExit(0 if compare(a.out, a.compare, tabs) else 1)


if __name__ == "__main__":
    main()
