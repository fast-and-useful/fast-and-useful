"""The paper's figures from results/ (accuracies and costs), configs/sets.json (sampling schedules) and the method
figure's example images (results/figure-images/). The opening teaser is a designed illustration and is not rebuilt.

  f3-comparison                  ImageNet-100 top-1 of the released students, base-first, ours and the unguided bases
  f4-generation-cost             top-1 against H100 seconds per image
  f3a-guidance-window            guidance only before or only after a switch noise level
  f3b-handoff                    base-first and base-last hand-offs between klein's base and its student
  f5-method                      the adapted sampler on klein's noise levels, and the same noise through three samplers
  s-additional-operating-points  further training-free and guided operating points
  s-scaling                      the released student and the adapter from 100 to 3,000 images per class
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import FixedLocator, NullLocator
from PIL import Image

COL, TEXT = 3.25, 6.875   # CVPR column and text widths (inches): LaTeX includes the figures at 100%
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, VIOLET, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#707070"
ROLE = {"distilled": (ORANGE, "^"), "basefirst": (AQUA, "D"), "adapter": (VIOLET, "*"), "unguided": (BLUE, "o"),
        "other": (GRAY, "s")}
AREA = {"o": 17, "^": 21, "D": 13, "*": 50, "s": 13}   # marker areas (pt^2) of equal visual weight
GLYPH = {"o": "●", "^": "▲", "D": "◆", "*": "★"}


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5, "legend.frameon": False, "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": AXIS,
        "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.2,
        "pdf.fonttype": 42, "savefig.dpi": 300})


def save(fig, out, name):
    fig.savefig(Path(out) / f"{name}.pdf", metadata={"CreationDate": None, "Creator": None})
    plt.close(fig)


def mean_sd(R, g):
    return R.mean(g), R.sd(g) or 0.0


def label_pm(m, sd):
    """Two decimals when the SD rounds to zero at one."""
    return f"{m:.2f} ± {sd:.2f}" if round(sd, 1) == 0 else f"{m:.1f} ± {sd:.1f}"


# Main comparison -------------------------------------------------------------------------------------------------------
COMPARISON = [   # (panel title, [(label, group, role)])
    ("(a) FLUX.2-klein", [("Released student", "klein", "distilled"), ("Base-first", "kb_dd2_g1", "basefirst"),
                          ("Ours", "kd_ad2_x3", "adapter"), ("Unguided base", "kb_g1", "unguided")]),
    ("(b) SD-Turbo", [("Released student", "sdt4", "distilled"), ("Base-first", "sdt4_dd2_g1", "basefirst"),
                      ("Ours", "sdt_ad1_x3", "adapter"), ("Unguided base", "sd21_g1", "unguided")]),
    ("(c) Z-Image-Turbo: comparison with SDA", [("Released student", "zimage_turbo_s8", "distilled"), ("SDA", "zt_sda_all", "other"),
                                                ("Base-first", "zt_dd3_g1", "basefirst"), ("Ours", "zt_ad2_x3", "adapter"),
                                                ("Unguided base", "zimage_g1", "unguided")]),
    ("(d) Adapting DP-DMD's released student", [("DP-DMD", "dpdmd", "distilled"), ("DP-DMD + ours", "dpdmd_ad1", "adapter"),
                                                ("Unguided base", "sd35m_g1", "unguided")]),
]


def fig_comparison(R, out):
    fig, axs = plt.subplots(2, 2, figsize=(TEXT, 3.3), layout="constrained")
    for ax, (title, rows) in zip(axs.flat, COMPARISON):
        y = np.arange(len(rows))[::-1]
        for yi, (label, g, role) in zip(y, rows):
            m, sd = mean_sd(R, g)
            color = ROLE[role][0]
            ax.barh(yi, m, height=0.6, color=color, zorder=2)
            ax.errorbar(m, yi, xerr=sd, color=INK, lw=0, elinewidth=1.0, capsize=2, zorder=3)
            ax.text(55, yi, label_pm(m, sd), va="center", ha="left", color=color, fontsize=8,
                    fontweight="bold" if role == "adapter" else "normal", clip_on=False)
        ax.set_yticks(y, [r[0] for r in rows])
        for tick, (_, _, role) in zip(ax.get_yticklabels(), rows):
            if role == "adapter":
                tick.set_color(VIOLET)
                tick.set_fontweight("bold")
        ax.set_xlim(0, 52)
        ax.grid(axis="y", visible=False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(title, loc="left", fontweight="bold")
    fig.supxlabel("ImageNet-100 synthetic-to-real top-1 accuracy (%)", fontsize=8)
    save(fig, out, "f3-comparison")


COST = [   # (panel, [(role, group or [groups along the unguided base's steps])])
    ("FLUX.2-klein", [("distilled", "klein"), ("adapter", "kd_ad2_x3"), ("basefirst", "kb_dd2_g1"),
                      ("unguided", ["kb_g1_s10", "kb_g1_s20", "kb_g1"])]),
    ("SD-Turbo", [("distilled", "sdt4"), ("adapter", "sdt_ad1_x3"), ("basefirst", "sdt4_dd2_g1"),
                  ("unguided", ["sd21_g1_s10", "sd21_g1"])]),
    ("Z-Image-Turbo", [("distilled", "zimage_turbo"), ("adapter", "zt_ad2_x3"), ("basefirst", "zt_dd3_g1"),
                       ("unguided", ["zimage_g1"])]),
]
COST_LABELS = {"distilled": "Released student", "adapter": "Ours", "basefirst": "Base-first", "unguided": "Unguided base"}


def fig_cost(R, out):
    fig, axs = plt.subplots(1, 3, figsize=(TEXT, 2.0), sharey=True, layout="constrained")
    for ax, (title, points) in zip(axs, COST):
        for role, groups in points:
            groups = groups if isinstance(groups, list) else [groups]
            color, marker = ROLE[role]
            x = [R.col(g, "sec") for g in groups]
            m, sd = zip(*[mean_sd(R, g) for g in groups])
            ax.errorbar(x, m, yerr=sd, color=color, marker="o", ms=(AREA["o"] * (2.2 if role == "adapter" else 1.3)) ** 0.5,
                        mec="white", mew=0.5, lw=1.0 if len(groups) > 1 else 0, elinewidth=0.6, capsize=1.5, zorder=3)
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_title(title)
        ax.set_ylim(10, 53)
    for ax, ticks in zip(axs, ([0.1, 0.2, 0.5, 1, 2], [0.02, 0.03, 0.05, 0.1], [0.3, 0.5, 1, 2])):
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:g}")
    axs[0].set_ylabel("Top-1 accuracy (%)")
    fig.supxlabel("H100 seconds per image (log scale)", fontsize=8)
    handles = [Line2D([], [], ls="none", marker="o", color=ROLE[r][0], mec="white", ms=6, label=l) for r, l in COST_LABELS.items()]
    fig.legend(handles=handles, loc="outside upper center", ncol=4)
    save(fig, out, "f4-generation-cost")


OPERATING = [   # (family, [(label, group)]): further operating points; the last row is ours
    ("FLUX.2-klein", [("Default-guided base", "klein_base"), ("Interval-guided base", "kb_interval_g4"), ("FSG", "kd_fsg"),
                      ("MoNO, 4 / basis", "kd_mono4"), ("MoNO, 32 / basis", "kd_mono4_g32"), ("Skip first step", "kd_sft"),
                      ("Ours", "kd_ad2_x3")]),
    ("SD 2.1 / SD-Turbo", [("Default-guided base", "sd21"), ("Interval-guided base", "sd21_interval75"),
                           ("Skip first step", "sdt_sft"), ("Ours", "sdt_ad1_x3")]),
]


def fig_operating_points(R, out):
    heights = [len(rows) for _, rows in OPERATING]
    fig, axs = plt.subplots(2, 2, figsize=(TEXT, 3.9), layout="constrained", width_ratios=[2.2, 1],
                            height_ratios=heights)
    for (family, rows), (bars, dots) in zip(OPERATING, axs):
        y = np.arange(len(rows))[::-1]
        for yi, (label, g) in zip(y, rows):
            ours = label == "Ours"
            color = VIOLET if ours else "#a4a4ab"
            m, sd = mean_sd(R, g)
            bars.barh(yi, m, height=0.6, color=color, zorder=2)
            bars.errorbar(m, yi, xerr=sd, color=INK, lw=0, elinewidth=1.0, capsize=2, zorder=3)
            sec = R.col(g, "sec")
            dots.scatter(sec, yi, s=AREA["o"] * 2, color=color, zorder=3)
            dots.text(sec * 1.25, yi, f"{sec:.3f}", va="center", ha="left", fontsize=8)
        bars.set_yticks(y, [label for label, _ in rows])
        bars.set_xlim(0, 52)
        bars.set_xlabel("Top-1 accuracy (%)")
        bars.set_title(family, loc="left", fontsize=9)
        bars.grid(axis="y", visible=False)
        dots.set_xscale("log")
        dots.set_xlim(0.018, 6)
        dots.xaxis.set_major_locator(FixedLocator([0.03, 0.1, 1, 3]))
        dots.xaxis.set_major_formatter(lambda v, _: f"{v:g}")
        dots.xaxis.set_minor_locator(NullLocator())
        dots.set_yticks(y, [])
        dots.set_ylim(bars.get_ylim())
        dots.grid(axis="y", visible=False)
        dots.set_xlabel("H100 s/image (log)")
    save(fig, out, "s-additional-operating-points")


# Trajectory --------------------------------------------------------------------------------------------------------------
WINDOW = [   # (panel, schedule source, EDM sigmas, [(curve, color, marker, group at k = 0, two-stage groups, group at k = N)])
    ("klein: guidance 4 vs none", "kb_hi10", False, [
        ("guide before switch", ORANGE, "^", "kb_g1", ["kb_hi10", "kb_hi20", "kb_cfg_early", "kb_cfg_gt51"], "klein_base"),
        ("guide after switch", BLUE, "o", "klein_base", ["kb_lo10", "kb_lo20", "kb_cfg_late", "kb_cfg_lt51"], "kb_g1")]),
    ("SD 2.1: guidance 7.5 vs 2", "sd21_hi3", True, [
        ("guide before switch", ORANGE, "^", "sd21_g2",
         ["sd21_hi3", "sd21_hi6", "sd21_hi9", "sd21_hi12", "sd21_g75early", "sd21_hi18", "sd21_hi24"], "sd21"),
        ("guide after switch", BLUE, "o", "sd21",
         ["sd21_lo3", "sd21_lo6", "sd21_lo9", "sd21_lo12", "sd21_g75late", "sd21_lo18", "sd21_lo24"], "sd21_g2")]),
]
HANDOFF = [   # klein's base and its distilled model on the base's 50-step schedule
    ("klein: unguided base", "kb_b2d25_g1", False, [
        ("base first", AQUA, "D", "klein_all50", ["kb_b2d25_g1", "kb_b2d40_g1"], "kb_g1"),
        ("distilled first", ORANGE, "^", "kb_g1", ["kb_d2b25_g1", "kb_d2b40_g1"], "klein_all50")]),
    ("klein: base at guidance 2", "kb_b2d25_g2", False, [
        ("base first", AQUA, "D", "klein_all50", ["kb_b2d25_g2"], "klein_base_g2"),
        ("distilled first", ORANGE, "^", "klein_base_g2", ["kb_d2b10_g2", "kb_d2b25_g2", "kb_d2b40_g2", "kb_d2b45_g2"],
         "klein_all50")]),
]


def t_levels(R, group, edm):
    """Noise level t at each step boundary from the set's schedule: flow sigmas as they are; SD's EDM sigma mapped to
    the flow level of equal signal-to-noise ratio, t = sigma / (1 + sigma)."""
    s = R.set_config(f"in100/{group}")["schedule"]["sigmas"]
    return [x / (1 + x) for x in s] if edm else list(s)


def switch_step(R, group):
    return int(R.set_config(f"in100/{group}")["args"]["stages"].split(",")[0].split(":")[2].split("-")[1])   # end of the first stage


def fig_switch(R, out, spec, name, xlabel):
    fig, axs = plt.subplots(1, len(spec), figsize=(COL, 1.95), sharey=True, layout="constrained")
    handles = {}
    for ax, (title, src, edm, curves) in zip(axs, spec):
        t = t_levels(R, src, edm)
        for label, c, mk, first, mids, last in curves:
            pts = sorted([(0, first), (len(t) - 1, last)] + [(switch_step(R, g), g) for g in mids])
            m, sd = zip(*[mean_sd(R, g) for _, g in pts])
            ax.errorbar([t[k] for k, _ in pts], m, yerr=sd, color=c, marker=mk, ms=AREA[mk] ** 0.5, mec="white", mew=0.4,
                        lw=1.1, elinewidth=0.6, capsize=0)
            handles[label] = Line2D([], [], color=c, marker=mk, ms=AREA[mk] ** 0.5, mec="white", mew=0.4, lw=1.1, label=label)
        ax.set_xlim(1.03, -0.03)
        ax.xaxis.set_major_locator(FixedLocator([1, 0.5, 0]))
        ax.set_title(title, fontsize=7.5, color=INK2)
    axs[0].set_ylabel("top-1 (%)")
    fig.supxlabel(xlabel, fontsize=8)
    fig.legend(handles=list(handles.values()), loc="outside upper center", ncol=2)
    save(fig, out, name)


# Method ------------------------------------------------------------------------------------------------------------------
METHOD_ROWS = [("unguided", "kb_or34_d1", "base state at $\\sigma_a$\n+ last distilled step"),
               ("adapter", "kd_ad2", "+ adapter (ours)\n2 NFE"), ("distilled", "klein", "distilled (released)\n4 NFE")]
METHOD = (54, [0, 5, 9, 15])   # the hare (class 54) at four image indices, chosen by eye for image quality


def arrow(ax, a, b, color, rad=0.0, ls="-", lw=1.1):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=6, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=1.5, zorder=3))


def fig_method(R, images, out):
    cls, js = METHOD
    name = R.classes("in100")["names"][cls].split(",")[0]
    teacher, adapter, distilled = (R.set_config(f"in100/{g}") for g in ("kb_or34_d1", "kd_ad2", "klein"))
    n_base = sum(who == "other" for who, _ in teacher["schedule"]["plan"])   # "other": the base; "base": the student
    fine, sig_a = teacher["schedule"]["sigmas"][:n_base + 1], teacher["schedule"]["sigmas"][n_base]
    assert adapter["schedule"]["sigmas"][1] == sig_a, "the adapter's jump must end on the teacher's level"
    levels = distilled["schedule"]["sigmas"]
    H, head, foot, gap, px = 2.25, 0.40, 0.07, 0.03, 220
    tile = (H - head - foot - 2 * gap) / 3
    grid_w = 4 * tile + 3 * gap
    x_grid = TEXT - grid_w
    fig = plt.figure(figsize=(TEXT, H))
    for r, (role, g, label) in enumerate(METHOD_ROWS):
        y0 = H - head - (r + 1) * tile - r * gap
        for c, j in enumerate(js):
            a = fig.add_axes([(x_grid + c * (tile + gap)) / TEXT, y0 / H, tile / TEXT, tile / H])
            a.imshow(Image.open(Path(images) / f"{g}-{cls}-{j}.png").convert("RGB").resize((px, px)), interpolation="none")
            a.set_axis_off()
        col, mk = ROLE[role]
        yc = (y0 + tile / 2) / H
        fig.text((x_grid - 0.07) / TEXT, yc, GLYPH[mk], ha="center", va="center", fontsize=7, color=col)
        fig.text((x_grid - 0.15) / TEXT, yc, label, ha="right", va="center", fontsize=7.5, linespacing=1.2)
    fig.text((x_grid + grid_w / 2) / TEXT, 1 - 0.13 / H, f"same noise and prompt per column (“{name}”)", ha="center",
             va="center", fontsize=8)
    nx, nw = 0.04, 0.42
    yc1 = H - head - 1.5 * tile - gap
    a = fig.add_axes([nx / TEXT, (yc1 - nw / 2) / H, nw / TEXT, nw / H])
    a.imshow(np.random.default_rng(0).normal(0.5, 0.18, (40, 40, 3)).clip(0, 1), interpolation="nearest")
    a.set_axis_off()
    fig.text((nx + nw / 2) / TEXT, (yc1 - nw / 2 - 0.06) / H, "noise $x_1$\nprompt $c$", ha="center", va="top", fontsize=7.5,
             linespacing=1.2)
    left, right = nx + nw + 0.12, x_grid - 1.0
    ax = fig.add_axes([left / TEXT, (foot - gap / 2) / H, (right - left) / TEXT, 3 * (tile + gap) / H])
    ax.set_xlim(1.03, -0.03)
    ax.set_ylim(-0.5, 2.5)
    ax.grid(False)
    for sp in ("left", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.set_yticks([])
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xticks([1, levels[2], sig_a, 0], ["1", f"{levels[2]:.2f}", f"$\\sigma_a$ = {sig_a:.2f}", "0"])
    ax.set_xticks([levels[1]], minor=True)
    ax.set_xlabel("noise level $\\sigma$ (FLUX.2-klein)", labelpad=3)
    ax.axvline(sig_a, color=AXIS, lw=0.6, ls=":", zorder=1)
    ax.plot(fine, [2] * len(fine), color=BLUE, lw=0.8, zorder=2)
    ax.scatter(fine[:-1], [2] * (len(fine) - 1), s=3, color=BLUE, lw=0, zorder=3)
    ax.scatter([sig_a], [2], s=AREA["o"] * 1.6, marker="o", color=BLUE, edgecolors="white", linewidths=0.5, zorder=4)
    arrow(ax, (sig_a, 2), (0, 2), ORANGE, ls=(0, (2, 2)), lw=0.8)
    ax.text(sig_a / 2, 1.9, "student finish (display)", ha="center", va="top", fontsize=7, color=INK2)
    ax.text(1, 2.3, f"unguided base: {n_base} steps (training only)", ha="left", va="bottom", fontsize=7.5, color=BLUE)
    ax.text(sig_a + 0.02, 1.98, "$x^\\star_{\\sigma_a}$", ha="right", va="top", fontsize=7, color=BLUE)
    arrow(ax, (1, 1), (sig_a, 1), VIOLET, rad=-0.45, lw=1.3)
    ax.scatter([sig_a], [1], s=AREA["*"], marker="*", color=VIOLET, edgecolors="white", linewidths=0.4, zorder=4)
    arrow(ax, (sig_a, 1), (0, 1), ORANGE)
    ax.text(1, 0.86, "adapted first step", ha="left", va="top", fontsize=7.5, color=VIOLET)
    ax.text(sig_a / 2, 1.07, "student finish", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.annotate("", xy=(sig_a - 0.015, 1.12), xytext=(sig_a - 0.015, 1.88),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=0.7, shrinkA=0, shrinkB=0, ls=(0, (2, 1.5))))
    ax.text(sig_a - 0.035, 1.47, "regression\n$\\Vert \\hat{x}_{\\sigma_a} - x^\\star_{\\sigma_a} \\Vert^2$", ha="left",
            va="center", fontsize=7, linespacing=1.15)
    ax.text(sig_a - 0.05, 0.98, "$\\hat{x}_{\\sigma_a}$", ha="left", va="top", fontsize=7, color=VIOLET)
    for a_, b_ in zip(levels[:-1], levels[1:]):
        arrow(ax, (a_, 0), (b_, 0), ORANGE)
    ax.scatter(levels[:-1], [0] * (len(levels) - 1), s=AREA["^"] * 0.8, marker="^", color=ORANGE, edgecolors="white",
               linewidths=0.4, zorder=4)
    ax.text(1, 0.3, f"released student: {len(levels) - 1} steps", ha="left", va="bottom", fontsize=7.5, color=ORANGE)
    save(fig, out, "f5-method")


# Data scaling ------------------------------------------------------------------------------------------------------------
SCALING = [   # (role, label, [(images per class, group)]): nested subsets of one 3,000-per-class set per sampler
    ("distilled", "distilled", [(100, "scale_klein_100"), (300, "scale_klein_300"), (500, "scale_klein_500"),
                                (1300, "scale_klein_1300"), (3000, "scale_klein_3000")]),
    ("adapter", "+ adapter (ours)", [(100, "scale_kd_ad2_100"), (300, "scale_kd_ad2_300"), (500, "kd_ad2"),
                                     (1300, "scale_kd_ad2_1300"), (3000, "scale_kd_ad2_3000")]),
]


def fig_scaling(R, out):
    fig, ax = plt.subplots(figsize=(COL, 2.05), layout="constrained")
    for role, label, pts in SCALING:
        c, mk = ROLE[role]
        m, sd = zip(*[mean_sd(R, g) for _, g in pts])
        ax.errorbar([k for k, _ in pts], m, yerr=sd, color=c, marker=mk, ms=AREA[mk] ** 0.5, mec="white", mew=0.4, lw=1.0,
                    elinewidth=0.6, capsize=0, label=label, zorder=3)
    c, mk = ROLE["unguided"]
    m, sd = mean_sd(R, "kb_g1")
    ax.errorbar([500], [m], yerr=[sd], color=c, marker=mk, ms=AREA[mk] ** 0.5, mec="white", mew=0.4, ls="none",
                elinewidth=0.6, capsize=0, label="base, no guidance", zorder=3)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([k for k, _ in SCALING[0][2]]))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("generated images per class")
    ax.set_ylabel("top-1 (%)")
    ax.legend(loc="upper left")
    save(fig, out, "s-scaling")


def build(R, out):
    style()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig_comparison(R, out)
    fig_cost(R, out)
    fig_operating_points(R, out)
    fig_switch(R, out, WINDOW, "f3a-guidance-window", "switch noise level t (1 = pure noise)")
    fig_switch(R, out, HANDOFF, "f3b-handoff", "hand-off noise level t (1 = pure noise)")
    fig_method(R, R.root / "figure-images", out)
    fig_scaling(R, out)
