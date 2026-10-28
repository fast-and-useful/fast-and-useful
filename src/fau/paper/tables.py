"""Every table of the paper and its supplement, as the LaTeX the paper inputs, from results/ (data.Results).

Each function returns one table body; TABLES maps the paper's file names to them. Mean +- SD cells print the stored
two-decimal group means and SDs at one decimal, as published; two late reference cells (`RUN_LEVEL`) are computed from
their individual runs instead, so their SD is rounded once.
"""
import statistics

from fau.paper.stats import gap, interval, pm, signed

END = r" \\"
RUN_LEVEL = {"tin200_kb_g1_s20", "in1k_sd21_g1_full"}


def top1(R, g):
    """(mean, SD) of a run group, as the paper prints it."""
    return R.runs_from_runs_csv(g) if g in RUN_LEVEL else (R.mean(g), R.sd(g))


def cell(R, g, bold=False):
    return "--" if g is None or not R.has(g) else pm(*top1(R, g), bold=bold)


def tabular(spec, header, body, rule_after_header=True):
    return "\n".join([rf"\begin{{tabular}}{{{spec}}}", r"\toprule", *header, *([r"\midrule"] if rule_after_header else []),
                      *body, r"\bottomrule", r"\end{tabular}"]) + "\n"


def block(title, ncols):
    return rf"\multicolumn{{{ncols}}}{{@{{}}l}}{{\emph{{{title}}}}}" + END


def shortstack_gap(mean, lo, hi):
    return r"\shortstack{" + signed(mean) + r"\\{\tableuncert" + interval(lo, hi) + "}}"


def gcell(mean, lo, hi):
    """A gap, bold when its interval lies above zero, with the interval in the uncertainty font."""
    m = signed(mean)
    return (rf"\textbf{{{m}}}" if lo > 0 else m) + r"\,{\tableuncert" + interval(lo, hi) + "}"


# Method ----------------------------------------------------------------------------------------------------------------
def t0_method_config(R):
    """Teacher targets and adapted sampling schedules (the adapters' first training runs)."""
    rows = []
    for student, teacher, g in [("FLUX.2-klein", "klein base", "kd_ad2"), ("SD-Turbo", "SD 2.1", "sdt_ad1"),
                                ("Z-Image-Turbo", "Z-Image", "zt_ad2")]:
        sched = R.set_config(f"in100/{g}")["schedule"]
        target = (f"$t={int(sched['timesteps'][1])}$" if g == "sdt_ad1" else f"$\\sigma={sched['sigmas'][1]:.4f}$")
        rows.append(f"{student} & {teacher} & {target} & {R.col(g, 'nfe'):g}" + END)
    return tabular("@{}llll@{}", ["Student & Base teacher & Jump endpoint & NFE" + END], rows)


# Main results ----------------------------------------------------------------------------------------------------------
T1 = [   # family, [(label, ImageNet-100 group, 200-class group, ImageNet-1k group at 1,281 images/class)]
    ("FLUX.2-klein", [("distilled, 4 steps", "klein", "tin200_klein", "in1k_klein_full"),
                      ("base-first", "kb_dd2_g1", "tin200_kb_dd2_g1", "in1k_kb_dd2_g1_full"),
                      ("+ adapter (ours)", "kd_ad2_x3", "tin200_kd_ad2", "in1k_kd_ad2_full"),
                      ("\\quad trained on COCO captions", "kd_coco", "tin200_kd_coco", None),
                      ("base, no guidance", "kb_g1", "tin200_kb_g1", None),
                      ("base, no guidance, 20 steps", "kb_g1_s20", "tin200_kb_g1_s20", "in1k_kb_g1_s20_full"),
                      ("base, default guidance", "klein_base", "tin200_klein_base", None)]),
    ("SD-Turbo", [("distilled, 4 steps", "sdt4", "tin200_sdt4", "in1k_sdt4_full"),
                  ("base-first", "sdt4_dd2_g1", "tin200_sdt4_dd2_g1", "in1k_sdt4_dd2_g1_full"),
                  ("+ adapter (ours)", "sdt_ad1_x3", "tin200_sdt_ad1", "in1k_sdt_ad1_full"),
                  ("\\quad trained on COCO captions", "sdt_coco", "tin200_sdt_coco", None),
                  ("base, no guidance", "sd21_g1", "tin200_sd21_g1", "in1k_sd21_g1_full"),
                  ("base, default guidance", "sd21", "tin200_sd21", None)]),
    ("Z-Image-Turbo", [("distilled, 9 steps", "zimage_turbo", "tin200_zimage_turbo", None),
                       ("base-first", "zt_dd3_g1", "tin200_zt_dd3_g1", None),
                       ("+ adapter (ours)", "zt_ad2_x3", "tin200_zt_ad2", None),
                       ("base, no guidance", "zimage_g1", None, None),
                       ("base, default guidance", "zimage", None, None)]),
]


def t1_main(R):
    """Complete primary results: NFE, cost and top-1 on three protocols; bold: the best fast sampler per column."""
    body = []
    for i, (family, arms) in enumerate(T1):
        if i:
            body.append(r"\midrule")
        fast = [groups for label, *groups in arms if label.startswith(("distilled", "base-first", "+ adapter", "\\quad"))]
        best = [max((top1(R, g[c])[0] for g in fast if g[c] and R.has(g[c])), default=None) for c in range(3)]
        for j, (label, *groups) in enumerate(arms):
            cells = [cell(R, g, g is not None and round(top1(R, g)[0], 1) == round(best[c], 1)) for c, g in enumerate(groups)]
            fam = rf"\multirow{{{len(arms)}}}{{*}}{{{family}}}" if j == 0 else ""
            body.append(f"{fam} & {label} & {R.col(groups[0], 'nfe'):g} & {R.col(groups[0], 'sec'):.2f} & "
                        + " & ".join(cells) + END)
    return tabular("@{}llrrrrr@{}", ["family & sampler & NFE & s/img & ImageNet-100 & 200 classes & ImageNet-1k" + END], body,
                   rule_after_header=True)


def t1_scale_transfer(R):
    """200 classes and full ImageNet-1k (Table 2 of the paper); bold: ours."""
    arms = [("FLUX.2-klein", [("Released student", "tin200_klein", "in1k_klein_full"),
                              ("Base-first", "tin200_kb_dd2_g1", "in1k_kb_dd2_g1_full"),
                              ("Ours", "tin200_kd_ad2", "in1k_kd_ad2_full"),
                              ("Unguided base, 20 steps", "tin200_kb_g1_s20", "in1k_kb_g1_s20_full")]),
            ("SD-Turbo", [("Released student", "tin200_sdt4", "in1k_sdt4_full"),
                          ("Base-first", "tin200_sdt4_dd2_g1", "in1k_sdt4_dd2_g1_full"),
                          ("Ours", "tin200_sdt_ad1", "in1k_sdt_ad1_full"),
                          ("Unguided base, 30 steps", "tin200_sd21_g1", "in1k_sd21_g1_full")]),
            ("Z-Image-Turbo", [("Released student", "tin200_zimage_turbo", None),
                               ("Base-first", "tin200_zt_dd3_g1", None),
                               ("Ours", "tin200_zt_ad2", None)])]
    body = []
    for i, (family, rows) in enumerate(arms):
        if i:
            body.append(r"\midrule")
        body.append(block(family, 3))
        body += [name + " & " + " & ".join(cell(R, g, name == "Ours") for g in groups) + END for name, *groups in rows]
    return tabular("@{}lrr@{}", ["Sampler & 200 classes & ImageNet-1k" + END], body)


T2 = [   # label, klein group, SD-Turbo group; klein's guided targets and endpoints use the finer teacher grid (kd_ad2x)
    ("ours: unguided, step 1, rank 32", "kd_ad2_x3", "sdt_ad1_x3"),
    ("\\quad same, finer teacher grid", "kd_ad2x", None),
    ("target at guidance 2", "kd_ad2_tg2", "sdt_ad1_tg2"),
    ("target at default guidance", "kd_ad2_tg4", "sdt_ad1_tg75"),
    ("adapter in every step", "kd_ad2_all", "sdt_ad1_all"),
    ("every weight fine-tuned", "kd_ad2_full_lr3e6", "sdt_ad1_full_lr1e5"),
    ("rank 8", "kd_ad2_r8", "sdt_ad1_r8"),
    ("rank 128", "kd_ad2_r128", "sdt_ad1_r128_x3"),
    ("jump to the first level only", "kd_ad0", "sdt_ad0"),
    ("jump to a deeper level", "kd_ad3", "sdt_ad2"),
    ("FSG", "kd_fsg", None),
    ("MoNO, 4 images per basis", "kd_mono4", None),
    ("MoNO, 32 images per basis", "kd_mono4_g32", None),
    ("skip first step (no training)", "kd_sft", "sdt_sft"),
    ("distilled model alone", "klein", "sdt4"),
]


def t2_ablation(R):
    """Adapter design ablations; bold: the best variant per column (every row that ties it at one decimal)."""
    best = [max(round(R.mean(row[c]), 1) for row in T2 if row[c]) for c in (1, 2)]
    body = []
    for i, (label, gk, gs) in enumerate(T2):
        cells = [cell(R, g, g is not None and round(R.mean(g), 1) == best[c]) for c, g in enumerate((gk, gs))]
        body.append(f"{label} & " + " & ".join(cells) + END)
        if i in (1, 3, 7, 9, 12):
            body.append(r"\addlinespace[2pt]")
    return tabular("@{}lrr@{}", ["variant & FLUX.2-klein & SD-Turbo" + END], body)


FINE = [("Food-101", "food101", 101), ("Cars", "cars", 196), ("Pets", "pets", 37)]


def t3_finegrained(R):
    """The ImageNet-100-trained adapters on fine-grained datasets; bold: the best shown per family and dataset."""
    families = [("FLUX.2-klein", [("Released", "klein"), ("Base-first", "kb_dd2_g1"), ("Ours", "kd_ad2")]),
                ("SD-Turbo", [("Released", "sdt4"), ("Base-first", "sdt4_dd2_g1"), ("Ours", "sdt_ad1")]),
                ("Z-Image-Turbo", [("Released", "zimage_turbo"), ("SDA", "zt_sda_all"), ("Ours", "zt_ad2")]),
                ("DP-DMD's student", [("Released", "dpdmd"), ("+ ours", "dpdmd_ad1")])]
    body = []
    for i, (family, arms) in enumerate(families):
        if i:
            body.append(r"\midrule")
        body.append(block(family, 5))
        best = {ds: max(R.mean(f"{ds}_{arm}") for _, arm in arms) for _, ds, _ in FINE}
        for label, arm in arms:
            groups = [f"{ds}_{arm}" for _, ds, _ in FINE]
            nfe = {R.col(g, "nfe") for g in groups}
            assert len(nfe) == 1, groups
            body.append(f"{label} & {nfe.pop():g} & "
                        + " & ".join(cell(R, g, R.mean(g) == best[ds]) for g, (_, ds, _) in zip(groups, FINE)) + END)
    return tabular("@{}lrrrr@{}", ["Sampler & NFE & Food-101 & Cars & Pets" + END], body)


LEARNERS = [("R50 + RA", "ra_"), ("Prior recipe", "wp_"), ("ConvNeXt-T", "cnx_"), ("ViT-S/16", "vits_")]


def t5_learners(R):
    """Released and adapted sets under other learners and recipes; bold: the adapted generator."""
    head = [r"learner & \multicolumn{2}{c}{FLUX.2-klein} & \multicolumn{2}{c}{SD-Turbo}" + END,
            r"\cmidrule(lr){2-3}\cmidrule(l){4-5}", " & Original & Adapted & Original & Adapted" + END]
    body = [label + " & " + " & ".join(cell(R, pre + g, adapted) for g, adapted in
                                       (("klein", False), ("kd_ad2", True), ("sdt4", False), ("sdt_ad1", True))) + END
            for label, pre in LEARNERS]
    return tabular(r"@{}lr@{\hspace{7pt}}rr@{\hspace{7pt}}r@{}", head, body)


def t6_prompts(R):
    """Adapted minus released under each prompt regime, and the complete procedures at ImageNet-1k scale."""
    body = [block("ImageNet-100: adapted $-$ released", 3)]
    for label, prefix in [("Templates / templates", ""), ("Captions / templates", "cap_")]:
        cells = []
        for a, b in (("kd_ad2", "klein"), ("sdt_ad1", "sdt4")):   # the same first trained adapter for every prompt
            mean, lo, hi = gap(R.per_class(prefix + a), R.per_class(prefix + b))
            cells.append(r"\textbf{" + signed(mean) + r"}\,{\tableuncert" + interval(lo, hi) + "}")
        body.append(label + " & " + " & ".join(cells) + END)
    for label, ka, sa in [("LLM / templates", "kd_ad2", "sdt_ad1"), ("LLM / LLM", "kd_llm", "sdt_llm")]:
        cells = [gcell(*gap(R.per_class("llmx_" + a), R.per_class("llmx_" + b))) for a, b in ((ka, "klein"), (sa, "sdt4"))]
        body.append(label + " & " + " & ".join(cells) + END)
    body += [r"\midrule", block("ImageNet-1k: vs. released with LLM prompts", 3)]
    for label, pairs in [(r"\shortstack[l]{Template-based\\procedure}", [("in1k_kd_ad2", "in1k_llm_klein"),
                                                                          ("in1k_sdt_ad1", "in1k_llm_sdt4")]),
                         (r"\shortstack[l]{LLM-prompt\\procedure}", [("in1k_llm_kd_llm", "in1k_llm_klein"),
                                                                     ("in1k_llm_sdt_ad1", "in1k_llm_sdt4")])]:
        body.append(label + " & " + " & ".join(gcell(*gap(R.per_class(a), R.per_class(b))) for a, b in pairs) + END)
    return tabular("@{}lrr@{}", ["Comparison & FLUX.2-klein & SD-Turbo" + END], body)


# Costs -----------------------------------------------------------------------------------------------------------------
def preparation(R, adapter):
    """One-time preparation of an adapter in H100-hours: (target pairs, fit, total including any prompt pool)."""
    jobs = R.costs["preparation"][adapter]
    return jobs["pairs"], jobs["fit"], sum(jobs.values())


def s14_deployment(R):
    """Preparation, generation cost, accuracy and model requirements, ImageNet-100 and full ImageNet-1k."""
    head = [r"Sampler & \multicolumn{2}{c}{One-time preparation (h)} & Generation & Total & Top-1 & Base at" + END,
            r"\cmidrule(lr){2-3}", r" & Targets & Fit & s/image & H100-hours & (\%) & inference" + END]
    panels = [
        ("FLUX.2-klein, ImageNet-100 (total for 50,000 images)", "kd_ad2", 50000,
         [("Released student", "klein", False), ("Base-first", "kb_dd2_g1", True), ("+ adapter (first run)", "kd_ad2", False),
          ("Unguided base, 20 steps", "kb_g1_s20", True)]),
        ("SD-Turbo / SD 2.1, ImageNet-100 (total for 50,000 images)", "sdt_ad1", 50000,
         [("Released student", "sdt4", False), ("Base-first", "sdt4_dd2_g1", True),
          ("+ adapter (first run)", "sdt_ad1", False), ("Unguided base, 30 steps", "sd21_g1", True)]),
        ("FLUX.2-klein, ImageNet-1k (total for 1,281,000 images)", "kd_ad2", None,
         [("Released student", "in1k_klein_full", False), ("Base-first", "in1k_kb_dd2_g1_full", True),
          ("+ adapter (first run)", "in1k_kd_ad2_full", False), ("Unguided base, 20 steps", "in1k_kb_g1_s20_full", True)]),
        ("SD-Turbo, ImageNet-1k (total for 1,281,000 images)", "sdt_ad1", None,
         [("Released student", "in1k_sdt4_full", False), ("Base-first", "in1k_sdt4_dd2_g1_full", True),
          ("+ adapter (first run)", "in1k_sdt_ad1_full", False)])]
    body = []
    for i, (title, adapter, images, rows) in enumerate(panels):
        if i:
            body.append(r"\midrule")
        body.append(block(title, 7))
        pairs, fit, _ = preparation(R, adapter)
        for name, g, base in rows:
            adapted = "adapter" in name
            n = images or int(R.col(g, "images"))
            sec = R.col(g, "sec")
            total = (pairs + fit if adapted else 0) + n * sec / 3600
            cells = [f"{pairs:.1f}" if adapted else "--", f"{fit:.1f}" if adapted else "--", f"{sec:.3f}", f"{total:.1f}",
                     pm(R.mean(g), R.sd(g), adapted), "Yes" if base else "No"]
            body.append(name + " & " + " & ".join(cells) + END)
    return tabular("@{}lrrrrrl@{}", head, body)


COST_IMAGES = [50000, 500000, 1281167]
COST_SAMPLERS = {   # family: [(row label, template-prompt group, LLM-prompt group, adapter whose preparation it carries)]
    "FLUX.2-klein": [("released", "klein", "llm_klein", None), ("base-first", "kb_dd2_g1", "llm_kb_dd2_g1", None),
                     ("template adapter", "kd_ad2", "llm_kd_ad2", "kd_ad2"), ("LLM adapter", "kd_llm", "llm_kd_llm", "kd_llm"),
                     ("Unguided base", "kb_g1_s20", "kb_g1_s20", None)],
    "SD-Turbo": [("released", "sdt4", "llm_sdt4", None), ("base-first", "sdt4_dd2_g1", "llm_sdt4_dd2_g1", None),
                 ("template adapter", "sdt_ad1", "llm_sdt_ad1", "sdt_ad1"), ("LLM adapter", "sdt_llm", "llm_sdt_llm", "sdt_llm"),
                 ("unguided base", "sd21_g1", "sd21_g1", None)],
}


def crossover(prep, sec_reference, sec_adapter):
    """Images after which the adapter's preparation plus generation costs less than the reference's generation."""
    return prep * 3600 / (sec_reference - sec_adapter) if sec_reference > sec_adapter else None


def s_e50_costs(R):
    """Preparation plus generation (H100-hours) at three dataset sizes, and the crossover image counts."""
    head = [r"Sampler & Prompts & Prep. h & 50k & 500k & 1.28M & $N^*_{\rm BF}$ & $N^*_{\rm base}$" + END]
    body = []
    for i, (family, rows) in enumerate(COST_SAMPLERS.items()):
        if i:
            body.append(r"\midrule")
        body.append(block(family, 8))
        sec = {label: {"template": R.col(t, "sec"), "llm": R.col(l, "sec")} for label, t, l, _ in rows}
        for label, _, _, adapter in rows:
            prep = preparation(R, adapter)[2] if adapter else 0.0
            for style, printed in (("template", "Templates"), ("llm", "LLM")):
                totals = [f"{round(prep + n * sec[label][style] / 3600, 2):.1f}" for n in COST_IMAGES]
                cross = ["--", "--"]
                if adapter:
                    cross = [f"{round(crossover(prep, sec[ref][style], sec[label][style])) / 1000:.0f}k"
                             for ref in ("base-first", rows[-1][0])]
                body.append(f"{label} & {printed} & {prep:.2f} & " + " & ".join(totals + cross) + END)
    return tabular("@{}llrrrrrr@{}", head, body)


# Target design and baselines -------------------------------------------------------------------------------------------
def s_dataset_regeneration(R):
    """Four headline sets regenerated with new noise (seed offset 300,000)."""
    rows = [("FLUX.2-klein, released", "klein", "klein_ns1"), ("FLUX.2-klein + adapter", "kd_ad2", "kd_ad2_ns1"),
            ("SD-Turbo, released", "sdt4", "sdt4_ns1"), ("SD-Turbo + adapter", "sdt_ad1", "sdt_ad1_ns1")]
    return tabular("@{}lrr@{}", ["Sampler & Original set & Regenerated set" + END],
                   [f"{label} & {cell(R, a)} & {cell(R, b)}" + END for label, a, b in rows])


ADDITIONAL = [   # family, [(group, setting, steps, pixels)]
    ("SD 1.5", [("sd15", "Base, guidance 7.5", 30, 512), ("sd15_g2", "Base, guidance 2", 30, 512)]),
    ("SDXL", [("sdxl_g5", "Base, guidance 5", 30, 1024), ("sdxl_g2", "Base, guidance 2", 30, 1024),
              ("sdxl_turbo", "Turbo", 1, 512), ("sdxl_dmd2", "DMD2", 4, 1024), ("sdxl_lightning", "Lightning", 4, 1024),
              ("sdxl_lcm", "LCM, guidance 8", 4, 1024)]),
    ("FLUX.1-dev", [("flux_dev", "Guidance 3.5", 28, 512), ("flux_dev_g1", "Guidance 1", 28, 512),
                    ("srpo", "SRPO, guidance 3.5", 28, 512)]),
]


def s_additional_generators(R):
    """Further generator families; bold: the best setting per family."""
    body = []
    for i, (family, rows) in enumerate(ADDITIONAL):
        if i:
            body.append(r"\midrule")
        best = max(float(f"{R.mean(g):.1f}") for g, *_ in rows)
        for j, (g, setting, steps, pixels) in enumerate(rows):
            body.append(f"{family if j == 0 else ''} & {setting} & {steps} & {pixels} & "
                        f"{pm(R.mean(g), R.sd(g), float(f'{R.mean(g):.1f}') == best)} " + r"\\")
    return tabular("@{}llrrr@{}", [r"Family & Setting & Steps & Pixels & Top-1 (\%)" + END], body)


def s10_dpdmd(R):
    """DP-DMD's released SD3.5-Medium student, its teacher, base-first and our adapter, at 512 and 1024 pixels."""
    arms = [("SD3.5-Medium, guidance 3.5", "sd35m", "tin200_sd35m"), ("SD3.5-Medium, no guidance", "sd35m_g1", "tin200_sd35m_g1"),
            ("DP-DMD's student", "dpdmd", "tin200_dpdmd"), (r"\quad base-first, first level", "dpdmd_dd1_g1", "tin200_dpdmd_dd1_g1"),
            (r"\quad + adapter, first level", "dpdmd_ad0_x3", "tin200_dpdmd_ad0"),
            (r"\quad + adapter, second level", "dpdmd_ad1", None),
            ("DP-DMD's student", "dpdmd_r1024", None), (r"\quad + adapter, second level", "dpdmd_ad1_r1024", None)]
    body = []
    for i, (label, g, g200) in enumerate(arms):
        if i == 6:
            body.append(r"\midrule")
        body.append(f"{label} & {R.col(g, 'resolution'):g} & {R.col(g, 'nfe'):g} & {R.col(g, 'sec'):.3f} & "
                    f"{cell(R, g, g in ('dpdmd_ad1', 'dpdmd_ad1_r1024'))} & {cell(R, g200, g200 == 'tin200_dpdmd_ad0')}" + END)
    return tabular("@{}lrrrrr@{}", ["Sampler & Pixels & NFE & s/img & ImageNet-100 & 200 classes" + END], body)


# Generalization --------------------------------------------------------------------------------------------------------
def unseen_classes(R):
    """ImageNet-1k label indices outside ImageNet-100 (the classes the adapters never saw)."""
    seen = set(R.classes("in100")["imagenet_index"])
    assert len(seen) == 100
    return [k for k in range(1000) if k not in seen]


def s12_unseen_fullscale(R):
    """Full-scale ImageNet-1k results on the 900 classes outside adapter training (1,000-way predictions kept)."""
    unseen = unseen_classes(R)
    head = [r"Generator & Released & Base-first & + adapter & Over released [95\% CI] & Over base-first [95\% CI]" + END]
    body = []
    for label, released, basefirst, adapter in [
            ("FLUX.2-klein", "in1k_klein_full", "in1k_kb_dd2_g1_full", "in1k_kd_ad2_full"),
            ("SD-Turbo", "in1k_sdt4_full", "in1k_sdt4_dd2_g1_full", "in1k_sdt_ad1_full")]:
        cells = []
        for g in (released, basefirst, adapter):
            seeds = [100 * statistics.mean(v) for _, v in sorted(R.per_class(g, unseen).items(), key=lambda kv: int(kv[0]))]
            cells.append(pm(statistics.mean(seeds), statistics.stdev(seeds), g == adapter))
        for reference in (released, basefirst):
            cells.append(shortstack_gap(*gap(R.per_class(adapter, unseen), R.per_class(reference, unseen))))
        body.append(label + " & " + " & ".join(cells) + END)
    return tabular("@{}lrrrrr@{}", head, body)


def s_crossdata_comparisons(R):
    """Paired gains on the fine-grained datasets, and their generation times."""
    pairs = [("SDA $-$ released Z-Image-Turbo", "zt_sda_all", "zimage_turbo"),
             ("Ours $-$ released Z-Image-Turbo", "zt_ad2", "zimage_turbo"), ("Ours $-$ SDA", "zt_ad2", "zt_sda_all"),
             ("DP-DMD + ours $-$ DP-DMD", "dpdmd_ad1", "dpdmd")]
    gains = [label + " & " + " & ".join(gcell(*gap(R.per_class(f"{ds}_{a}"), R.per_class(f"{ds}_{b}"), exact_mean=True))
                                        for _, ds, _ in FINE) + END for label, a, b in pairs]
    first = tabular("@{}lrrr@{}", [r"Mean-class accuracy gain [95\% CI] & Food-101 & Cars & Pets" + END], gains)
    times = []
    for label, arm in [("Z-Image-Turbo, released", "zimage_turbo"), ("+ SDA", "zt_sda_all"), ("+ ours", "zt_ad2"),
                       ("DP-DMD's student", "dpdmd"), ("+ ours", "dpdmd_ad1")]:
        groups = [f"{ds}_{arm}" for _, ds, _ in FINE]
        times.append(f"{label} & {R.col(groups[0], 'nfe'):g} & " + " & ".join(f"{R.col(g, 'sec'):.3f}" for g in groups) + END)
    second = tabular("@{}lrrrr@{}", ["H100 seconds/image & NFE & Food-101 & Cars & Pets" + END], times)
    return first + r"\par\medskip" + "\n" + second


def s_finegrained_references(R):
    """References on the fine-grained datasets: COCO-target adapters, the unguided bases, real training images."""
    rows = [("FLUX.2-klein, COCO targets", "kd_coco"), ("FLUX.2-klein, unguided base", "kb_g1_s20"),
            ("SD-Turbo, COCO targets", "sdt_coco"), ("SD 2.1, unguided base", "sd21_g1"), ("Real training images", "real")]
    real = {"food101": "food101_real500", "cars": "cars_real", "pets": "pets_real"}
    body = [label + " & " + " & ".join(cell(R, real[ds] if arm == "real" else f"{ds}_{arm}") for _, ds, _ in FINE) + END
            for label, arm in rows]
    return tabular("@{}lrrr@{}", ["Training images & Food-101 & Cars & Pets" + END], body)


TRANSFER = [("Food-101", "food101"), ("Cars", "cars"), ("Pets", "pets"), ("IN-900", "in900")]


def s9_transfer(R):
    """Linear probes of the ImageNet-100 classifiers' frozen features on four unseen datasets; bold: the adapters."""
    blocks = [("FLUX.2-klein", [("distilled", "klein"), ("+ adapter (ours)", "kd_ad2"), ("base, no guidance", "kb_g1")]),
              ("SD-Turbo", [("distilled", "sdt4"), ("+ adapter (ours)", "sdt_ad1"), ("SD 2.1, no guidance", "sd21_g1")]),
              (None, [("real, 500 per class", "real500")])]
    body = []
    for i, (title, rows) in enumerate(blocks):
        if i:
            body.append(r"\midrule")
        if title:
            body.append(block(title, 5))
        for label, g in rows:
            body.append(label + " & " + " & ".join(pm(*R.transfer_top1(g, t), bold="adapter" in label) for _, t in TRANSFER) + END)
    return tabular("@{}lrrrr@{}", ["training source & " + " & ".join(l for l, _ in TRANSFER) + END], body)


def s_learner_references(R):
    """The unguided bases and real images under the five learners and recipes."""
    head = [r"training set & R50 & \shortstack{R50 +\\RandAugment} & \shortstack{Prior\\recipe} & ConvNeXt-T & ViT-S" + END]
    rows = [("FLUX.2-klein base, no guidance", "kb_g1"), ("SD 2.1, no guidance", "sd21_g1"), ("real images, 500 per class", "real500")]
    body = [label + " & " + " & ".join(cell(R, pre + g) for pre in ("", "ra_", "wp_", "cnx_", "vits_")) + END
            for label, g in rows]
    return tabular("@{}lrrrrr@{}", head, body)


# Prompts and real data -------------------------------------------------------------------------------------------------
CAPTIONS = [   # (label, template-prompt set, caption-prompt set); the LLM column is "llm_" + the template set
    ("SD 1.5, guidance 7.5", "sd15", "cap_sd15"), ("SD 1.5, guidance 2", "sd15_g2", "cap_sd15_g2"),
    ("SD 2.1, guidance 7.5", "sd21", "cap_sd21"), ("SD 2.1, guidance 2", "sd21_g2", "cap_sd21_g2"),
    ("SD 2.1, no guidance", "sd21_g1", "cap_sd21_g1"), ("SD-Turbo, 1 step", "sd_turbo", "cap_sd_turbo"),
    ("SD-Turbo, 4 steps", "sdt4", "cap_sdt4"), ("SD-Turbo, base-first", "sdt4_dd2_g1", "cap_sdt4_dd2_g1"),
    ("SD-Turbo + adapter", "sdt_ad1", "cap_sdt_ad1"), ("SDXL, guidance 5", "sdxl_g5", "cap_sdxl_g5"),
    ("SDXL-Turbo", "sdxl_turbo", "cap_sdxl_turbo"), ("klein base, guidance 4", "klein_base", "cap_klein_base"),
    ("klein base, guidance 2", "klein_base_g2", "cap_klein_base_g2"), ("klein base, no guidance", "kb_g1", "cap_kb_g1"),
    ("FLUX.2-klein", "klein", "cap_klein"), ("FLUX.2-klein, base-first", "kb_dd2_g1", "cap_kb_dd2_g1"),
    ("FLUX.2-klein + adapter", "kd_ad2", "cap_kd_ad2"),
]


def s5_captions(R):
    """Class-name templates against real-image captions and LLM descriptions; bold: the best prompt per sampler."""
    body = []
    for label, tg, cg in CAPTIONS:
        if label in ("SD 2.1, guidance 7.5", "SD-Turbo, 1 step", "SDXL, guidance 5", "klein base, guidance 4"):
            body.append(r"\midrule")
        groups = [tg, cg, "llm_" + tg]
        best = max(round(R.mean(g), 1) for g in groups if R.has(g))
        body.append(f"{label} & " + " & ".join(cell(R, g, R.has(g) and round(R.mean(g), 1) == best) for g in groups) + END)
    return tabular("@{}lrrr@{}", ["sampler & templates & captions & LLM prompts" + END], body)


E50_ARMS = {"FLUX.2-klein": [("Released", "klein"), ("Base-first", "kb_dd2_g1"), ("Template-fitted", "kd_ad2"),
                             ("LLM-fitted", "kd_llm")],
            "SD-Turbo": [("Released", "sdt4"), ("Base-first", "sdt4_dd2_g1"), ("Template-fitted", "sdt_ad1"),
                         ("LLM-fitted", "sdt_llm")]}


def s_e50_replication(R):
    """Three independently generated LLM-prompt datasets per sampler: classifier and dataset variability."""
    head = ["Sampler & Mean & Classifier SD & Dataset-mean SD & Set 1 & Set 2 & Set 3" + END]
    body = []
    for i, (family, arms) in enumerate(E50_ARMS.items()):
        if i:
            body.append(r"\midrule")
        body.append(block(family, 7))
        for label, arm in arms:
            sets = [R.runs(prefix + arm) for prefix in ("llm_", "llm2_", "llm3_")]
            assert all(len(v) == 3 for v in sets), arm
            means = [statistics.mean(v) for v in sets]
            overall = statistics.mean(x for v in sets for x in v)
            classifier = statistics.mean(statistics.variance(v) for v in sets) ** .5
            body.append(f"{label} & {overall:.1f} & {classifier:.2f} & {statistics.stdev(means):.2f} & "
                        + " & ".join(f"{m:.1f}" for m in means) + END)
    return tabular("@{}lrrrrrr@{}", head, body)


def s_e50_prompt_fitting(R):
    """Adapters fitted on LLM descriptions against template-fitted adapters, released students and base-first."""
    body = [block("LLM generation prompts: three datasets, three classifier seeds each", 3)]
    for label, ka, kb, sa, sb in [("LLM-fitted $-$ template-fitted", "kd_llm", "kd_ad2", "sdt_llm", "sdt_ad1"),
                                  ("LLM-fitted $-$ released", "kd_llm", "klein", "sdt_llm", "sdt4"),
                                  ("LLM-fitted $-$ base-first", "kd_llm", "kb_dd2_g1", "sdt_llm", "sdt4_dd2_g1"),
                                  ("Template-fitted $-$ released", "kd_ad2", "klein", "sdt_ad1", "sdt4"),
                                  ("Template-fitted $-$ base-first", "kd_ad2", "kb_dd2_g1", "sdt_ad1", "sdt4_dd2_g1")]:
        body.append(label + " & " + " & ".join(gcell(*gap(R.per_class("llmx_" + a), R.per_class("llmx_" + b)))
                                               for a, b in ((ka, kb), (sa, sb))) + END)
    body += [r"\midrule", block("Template generation prompts: one dataset, three classifier seeds", 3)]
    body.append("LLM-fitted $-$ template-fitted & " + " & ".join(gcell(*gap(R.per_class(a), R.per_class(b)))
                                                               for a, b in (("kd_llm", "kd_ad2"), ("sdt_llm", "sdt_ad1"))) + END)
    return tabular("@{}lrr@{}", ["Adapter comparison & FLUX.2-klein & SD-Turbo" + END], body)


def s_e50_in1k(R):
    """ImageNet-1k at 100 images/class with LLM generation prompts, against the released student and base-first."""
    families = [("FLUX.2-klein", "klein", "kb_dd2_g1", [("Original adapter", "in1k_kd_ad2", "Templates"),
                                                        ("LLM-fitted adapter", "in1k_llm_kd_llm", "LLM"),
                                                        ("Template fit (post hoc)", "in1k_llm_kd_ad2", "LLM"),
                                                        ("Unguided klein base", "in1k_llm_kb_g1_s20", "LLM")]),
                ("SD-Turbo", "sdt4", "sdt4_dd2_g1", [("Original adapter", "in1k_sdt_ad1", "Templates"),
                                                     ("Template-fitted adapter", "in1k_llm_sdt_ad1", "LLM"),
                                                     ("Unguided SD 2.1", "in1k_llm_sd21_g1", "LLM")])]
    body = []
    for i, (family, released, basefirst, arms) in enumerate(families):
        if i:
            body.append(r"\midrule")
        body.append(block(family, 6))
        rel, bf = "in1k_llm_" + released, "in1k_llm_" + basefirst
        for label, g, prompts in [("Released", rel, "LLM"), ("Base-first", bf, "LLM")] + arms:
            vr = "--" if g == rel else gcell(*gap(R.per_class(g), R.per_class(rel)))
            vb = "--" if g in (rel, bf) else gcell(*gap(R.per_class(g), R.per_class(bf)))
            body.append(f"{label} & {prompts} & {R.col(g, 'nfe'):g} & {cell(R, g)} & {vr} & {vb}" + END)
    return tabular("@{}llrrrr@{}", ["Sampler & Generation prompts & NFE & Top-1 & vs. LLM released & vs. LLM base-first" + END],
                   body)


OOD = [("v2", "V2"), ("sketch", "Sketch"), ("r", "R"), ("a", "A")]
OOD_BLOCKS = [
    ("ImageNet-100, generated only", [("real images, 500 per class", "real500"), ("FLUX.2-klein, distilled", "klein"),
                                      ("\\quad base-first", "kb_dd2_g1"), ("\\quad + adapter (ours)", "kd_ad2_x3"),
                                      ("\\quad base, no guidance", "kb_g1"), ("SD-Turbo, distilled", "sdt4"),
                                      ("\\quad base-first", "sdt4_dd2_g1"), ("\\quad + adapter (ours)", "sdt_ad1_x3"),
                                      ("\\quad SD 2.1, no guidance", "sd21_g1"), ("Z-Image-Turbo, distilled", "zimage_turbo"),
                                      ("\\quad + adapter (ours)", "zt_ad2_x3")]),
    ("ImageNet-100, 500 real + 500 generated", [("real only, trained as long", "real500_e200"),
                                                ("+ FLUX.2-klein, distilled", "mix_klein"), ("+ FLUX.2-klein adapter", "mix_kd_ad2"),
                                                ("+ SD-Turbo, distilled", "mix_sdt4"), ("+ SD-Turbo adapter", "mix_sdt_ad1")]),
    ("ImageNet-1k, 1,281 per class", [("FLUX.2-klein, distilled", "in1k_klein_full"), ("\\quad + adapter (ours)", "in1k_kd_ad2_full"),
                                      ("SD-Turbo, distilled", "in1k_sdt4_full"), ("\\quad + adapter (ours)", "in1k_sdt_ad1_full")]),
    ("ImageNet-1k, all 1.28M real + 1.28M generated", [("real only, trained as long", "in1k_real_e200"),
                                                       ("+ FLUX.2-klein, distilled", "in1k_mix_klein"),
                                                       ("+ FLUX.2-klein adapter", "in1k_mix_kd_ad2")]),
]


def s6_ood(R):
    """Top-1 in distribution and on ImageNet-V2, -Sketch, -R and -A."""
    body = []
    for title, rows in OOD_BLOCKS:
        body += [r"\midrule", block(title, 6)]
        for label, g in rows:
            cells = [cell(R, g)]
            for test, _ in OOD:
                v = R.ood_top1(g, test)
                cells.append(pm(statistics.mean(v), statistics.stdev(v)) if len(v) > 1 else "--")
            body.append(f"{label} & " + " & ".join(cells) + END)
    return tabular("@{}lrrrrr@{}", ["training set & in distribution & " + " & ".join(l for _, l in OOD) + END], body,
                   rule_after_header=False)


# Diagnostics -----------------------------------------------------------------------------------------------------------
def s3_metrics(R, study):
    """Every set metric against top-1 over the metric study's sets (fau.metric_study)."""
    from fau.metric_study import METRICS
    fams = [("klein", "FLUX.2-klein"), ("sd21", "SD 2.1"), ("zimage", "Z-Image"), ("sdxl", "SDXL")]
    sig = lambda x: signed(x, 2)
    c3 = lambda c: f"{sig(c['r'])} {{\\tableuncert[{sig(c['lo'])}, {sig(c['hi'])}]}}"
    body = []
    for key, label in METRICS:
        c = study["metrics"][key]
        body.append(f"{label} & {c3(c['centered'])} & {sig(c['pooled']['r'])} & "
                    + " & ".join(sig(c["within"][f]["r"]) if f in c["within"] else "--" for f, _ in fams)
                    + f" & {sig(c['lofo']['pooled']['r'])}" + END)
    body.append(r"\midrule")
    for score, label in (("aesthetic", "aesthetic score given recall"), ("pickscore", "PickScore given recall")):
        body.append(f"{label} & {c3(study['partial'][f'{score}|dino_recall']['centered'])}" + " &" * (2 + len(fams)) + END)
    head = ["metric & within families & pooled & " + " & ".join(l for _, l in fams) + " & held-out family" + END]
    return tabular("@{}l" + "r" * (3 + len(fams)) + "@{}", head, body)


def s11_heldout(R, study):
    """Recall's held-out-family prediction: Spearman of out-of-fold isotonic predictions with top-1."""
    lofo = study["metrics"]["dino_recall"]["lofo"]
    body = []
    for key, label in [("klein", "FLUX.2-klein"), ("sd21", "SD 2.1 / SD-Turbo"), ("zimage", "Z-Image"), ("sdxl", "SDXL"),
                       ("flux", "FLUX.1-dev")]:
        v = lofo["held_out"][key]
        body.append(f"{label} & {v['n']} & {signed(v['r'], 2)} " + r"{\tableuncert" + interval(v["lo"], v["hi"], 2) + "}" + END)
    v = lofo["pooled"]
    body += [r"\midrule", f"Pooled predictions & {v['n']} & {signed(v['r'], 2)} " + r"{\tableuncert"
             + interval(v["lo"], v["hi"], 2) + "}" + END]
    return tabular("@{}lrr@{}", [r"Held-out family & sets & $\rho$ [95\% CI]" + END], body)


def s13_family_weighting(R, weighting):
    """Family-centered ranks against equally weighted within-family correlations; bold: the largest point estimate."""
    res = weighting["results"]
    body = []
    for metric, label in [("dino_recall", "DINOv3 recall"), ("clip_recall", "CLIP recall"),
                          ("clip_fd", r"$-$CLIP Fr\'echet"), ("fid", "$-$FID")]:
        cells = []
        for stat in ("centered", "equal_family"):
            v = res[metric][stat]
            m = signed(v["value"], 2)
            if metric == max(res, key=lambda k: res[k][stat]["value"]):
                m = rf"\textbf{{{m}}}"
            cells.append(m + r"\,{\tableuncert" + interval(*v["interval"], 2) + "}")
        body.append(label + " & " + " & ".join(cells) + END)
    return tabular("@{}lrr@{}", ["Metric & Family-centered & Equal-family" + END], body)


QUALITY = [   # (family, [(label, group)]): the adapters' first training runs, which carry the set metrics
    ("FLUX.2-klein", [("distilled", "klein"), ("\\quad + adapter (ours)", "kd_ad2"), ("base, no guidance", "kb_g1"),
                      ("base, default guidance", "klein_base")]),
    ("SD-Turbo", [("distilled", "sdt4"), ("\\quad + adapter (ours)", "sdt_ad1"), ("SD 2.1, no guidance", "sd21_g1"),
                  ("SD 2.1, default guidance", "sd21")]),
    ("Z-Image-Turbo", [("distilled", "zimage_turbo"), ("\\quad + adapter (ours)", "zt_ad2"), ("Z-Image, no guidance", "zimage_g1"),
                       ("Z-Image, default guidance", "zimage")]),
]
QUALITY_COLS = [("FID", "fid", "{:.1f}"), ("aesthetic", "aesthetic", "{:.2f}"), ("PickScore", "pickscore", "{:.2f}"),
                ("recognized", "recog", "{:.2f}"), ("recall", "recall", "{:.3f}")]


def s7_quality(R):
    """Image quality beside training value."""
    body = []
    for family, rows in QUALITY:
        body += [r"\midrule", block(family, 2 + len(QUALITY_COLS))]
        for label, g in rows:
            body.append(f"{label} & {cell(R, g)} & " + " & ".join(fmt.format(R.col(g, c)) for _, c, fmt in QUALITY_COLS) + END)
    head = ["sampler & top-1 & " + " & ".join(l for l, _, _ in QUALITY_COLS) + END]
    return tabular("@{}l" + "r" * (1 + len(QUALITY_COLS)) + "@{}", head, body, rule_after_header=False)


def s15_contrasts(R, numbers):
    """Supporting paired contrasts (numbers: the gap macros of fau.paper.numbers)."""
    body = [block("ImageNet-100: generator and dataset-size comparisons", 3)]
    for label, a, b in [("SRPO versus FLUX.1-dev", "srpo", "flux_dev"),
                        ("FLUX.2-klein: adapter versus base-first", "kd_ad2_x3", "kb_dd2_g1"),
                        ("SD-Turbo: adapter versus base-first", "sdt_ad1_x3", "sdt4_dd2_g1"),
                        ("Z-Image-Turbo: adapter versus base-first", "zt_ad2_x3", "zt_dd3_g1"),
                        ("FLUX.2-klein: 100 adapted versus 3,000 released images/class", "scale_kd_ad2_100", "scale_klein_3000")]:
        mean, lo, hi = gap(R.per_class(a), R.per_class(b))
        body.append(label + " & " + (rf"\textbf{{{signed(mean)}}}" if lo > 0 else signed(mean)) + " & " + interval(lo, hi) + END)
    blocks = [("ImageNet-100: 500 real images/class, synthetic augmentation", [
                  ("mixSDOne", "SD 1.5: guidance 2 versus 7.5"), ("mixKlein", "klein: base-first versus released"),
                  ("mixKleinOverReal", "klein: released augmentation versus real-only"),
                  ("mixBfOverReal", "klein: base-first augmentation versus real-only"),
                  ("mixAdKlein", "klein: adapted versus released augmentation"),
                  ("mixAdSD", "SD-Turbo: adapted versus released augmentation"),
                  ("mixAdKleinOverReal", "klein: adapted augmentation versus real-only"),
                  ("mixAdSDOverReal", "SD-Turbo: adapted augmentation versus real-only")]),
              ("ImageNet-100: 50 real images/class, synthetic augmentation", [
                  ("lowAdKlein", "klein: adapted versus released augmentation"),
                  ("lowAdSD", "SD-Turbo: adapted versus released augmentation"),
                  ("lowAdVsBaseKlein", "klein: adapted versus unguided-base augmentation"),
                  ("lowAdVsBaseSD", "SD-Turbo: adapted versus unguided-base augmentation")]),
              ("ImageNet-1k: full real-data training, klein augmentation", [
                  ("inkMixAdOverReal", "Adapted versus real-only: in distribution"),
                  ("inkMixDistOverReal", "Released versus real-only: in distribution"),
                  ("inkMixAdOverDist", "Adapted versus released: in distribution"),
                  ("inkMixOodSketchAdOverReal", "Adapted versus real-only: ImageNet-Sketch"),
                  ("inkMixOodRAdOverReal", "Adapted versus real-only: ImageNet-R"),
                  ("inkMixOodSketchDistOverReal", "Released versus real-only: ImageNet-Sketch"),
                  ("inkMixOodRDistOverReal", "Released versus real-only: ImageNet-R")])]
    for title, rows in blocks:
        body += [r"\midrule", block(title, 3)]
        for key, label in rows:
            g, ci = numbers["gp@" + key], numbers["gpci@" + key]
            low = float(ci[1:].split(",")[0].replace("$-$", "-"))
            body.append(label + " & " + (rf"\textbf{{{g}}}" if low > 0 else g) + " & " + ci + END)
    return tabular("@{}lrr@{}", [r"Comparison & Gain & 95\% CI" + END], body)


# The paper's table files -----------------------------------------------------------------------------------------------
def build(R, study, weighting, numbers):
    """{file name: LaTeX} for every table the paper and its supplement input."""
    plain = {"t0-method-config": t0_method_config, "t1-scale-transfer": t1_scale_transfer, "t2-ablation": t2_ablation,
             "t3-finegrained": t3_finegrained, "t5-learners": t5_learners, "t6-prompts": t6_prompts,
             "s14-deployment": s14_deployment, "s-e50-costs": s_e50_costs, "s-dataset-regeneration": s_dataset_regeneration,
             "s-additional-generators": s_additional_generators, "t1-main": t1_main, "s10-dpdmd": s10_dpdmd,
             "s12-unseen-fullscale": s12_unseen_fullscale, "s-crossdata-comparisons": s_crossdata_comparisons,
             "s-finegrained-references": s_finegrained_references, "s9-transfer": s9_transfer,
             "s-learner-references": s_learner_references, "s5-captions": s5_captions,
             "s-e50-replication": s_e50_replication, "s-e50-prompt-fitting": s_e50_prompt_fitting,
             "s-e50-in1k": s_e50_in1k, "s6-ood": s6_ood, "s7-quality": s7_quality}
    out = {name: f(R) for name, f in plain.items()}
    out["s3-metrics"] = s3_metrics(R, study)
    out["s11-heldout"] = s11_heldout(R, study)
    out["s13-family-weighting"] = s13_family_weighting(R, weighting)
    out["s15-contrasts"] = s15_contrasts(R, numbers)
    return out
