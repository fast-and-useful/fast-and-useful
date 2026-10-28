"""Every number the paper's text quotes, as the LaTeX macros it uses:

  \\tp{g}, \\tpsd{g}, \\tpsec{g}   top-1 mean, mean $\\pm$ SD, H100 seconds per image of run group g
  \\gp{n}, \\gpci{n}              a named paired gap (points) and its 95% class-bootstrap interval
  \\gpa{n}, \\gpcia{n}            the gap's magnitude and the interval of the magnitude
  \\nm{n}                        derived numbers: recovery shares, costs, crossovers, correlations, counts

`build` returns {"kind@name": value}; only the macros the text uses are computed.
"""
import re

import numpy as np

from fau.paper.stats import gap, gap_by_class, interval, share, signed
from fau.paper.tables import TRANSFER, crossover, preparation, unseen_classes

GAPS = {   # name -> (A, B): A - B on per-class top-1, quoted with its sign (\gp, \gpci)
    "capDistSDXL": ("cap_sdxl_turbo", "cap_sdxl_g5"), "distSDXL": ("sdxl_turbo", "sdxl_g5"),
    "loTwentyKlein": ("kb_lo20", "kb_g1"), "srpo": ("srpo", "flux_dev"),
    "inkAdVsBaseHundred": ("in1k_kd_ad2", "in1k_kb_g1_s20"), "inkSdtAdVsBase": ("in1k_sdt_ad1", "in1k_sd21_g1"),
    "inkAdVsBfKlein": ("in1k_kd_ad2_full", "in1k_kb_dd2_g1_full"), "inkAdVsBfSD": ("in1k_sdt_ad1_full", "in1k_sdt4_dd2_g1_full"),
    "dpdmdAd": ("dpdmd_ad0_x3", "dpdmd"), "dpdmdAdDeep": ("dpdmd_ad1", "dpdmd"), "dpdmdAdVsBf": ("dpdmd_ad0_x3", "dpdmd_dd1_g1"),
    "dpdmdAdVsBfTin": ("tin200_dpdmd_ad0", "tin200_dpdmd_dd1_g1"),
    "mixKlein": ("mix_kb_dd2_g1", "mix_klein"), "mixSDOne": ("mix_sd15_g2", "mix_sd15"),
    "mixKleinOverReal": ("mix_klein", "real500_e200"), "mixBfOverReal": ("mix_kb_dd2_g1", "real500_e200"),
    "mixAdKlein": ("mix_kd_ad2", "mix_klein"), "mixAdSD": ("mix_sdt_ad1", "mix_sdt4"),
    "mixAdKleinOverReal": ("mix_kd_ad2", "real500_e200"), "mixAdSDOverReal": ("mix_sdt_ad1", "real500_e200"),
    "lowAdKlein": ("low_kd_ad2", "low_klein"), "lowAdSD": ("low_sdt_ad1", "low_sdt4"),
    "lowAdVsBaseKlein": ("low_kd_ad2", "low_kb_g1"), "lowAdVsBaseSD": ("low_sdt_ad1", "low_sd21_g1"),
    "inkMixAdOverReal": ("in1k_mix_kd_ad2", "in1k_real_e200"), "inkMixDistOverReal": ("in1k_mix_klein", "in1k_real_e200"),
    "inkMixAdOverDist": ("in1k_mix_kd_ad2", "in1k_mix_klein"),
    "e52KleinLlm100": ("scale_llm_kd_llm_100", "scale_llm_klein_100"), "e52KleinLlm500": ("llm_kd_llm", "llm_klein"),
}
MAGNITUDES = {   # name -> (A, B): gaps quoted as magnitudes, "costs X points [a, b]" (\gpa, \gpcia)
    "capDistKlein": ("cap_kb_g1", "cap_klein"), "capGuideSD": ("cap_sd21_g2", "cap_sd21"),
    "distKleinUnguided": ("kb_g1", "klein"), "guideKlein": ("kb_g1", "klein_base"), "guideSDTwo": ("sd21_g2", "sd21"),
    "hiTenKlein": ("kb_hi10", "kb_g1"),
    "hiNineSD": ("sd21_hi9", "sd21_g2"),   # SD 2.1's window arms switch between guidance 7.5 and 2
    "tgFourKlein": ("kd_ad2_tg4", "kd_ad2x"), "tgSevenSD": ("sdt_ad1_tg75", "sdt_ad1_x3"), "tgZ": ("zt_ad2_tg5", "zt_ad2_x3"),
    "tgFourKleinTin": ("tin200_kd_ad2_tg4", "tin200_kd_ad2"), "tgSevenSDTin": ("tin200_sdt_ad1_tg75", "tin200_sdt_ad1"),
    "tgZTin": ("tin200_zt_ad2_tg5", "tin200_zt_ad2"), "llmGapKlein": ("llm_kb_g1", "llm_klein"),
    "dpdmdVsTeacher": ("dpdmd", "sd35m"), "dpdmdGuide": ("sd35m_g1", "sd35m"),
}
OOD_GAPS = {   # name -> (A, B, OOD test set): with all real ImageNet-1k images, generated images added
    f"inkMixOod{label}{arm}OverReal": (g, "in1k_real_e200", test)
    for arm, g in (("Ad", "in1k_mix_kd_ad2"), ("Dist", "in1k_mix_klein")) for test, label in (("sketch", "Sketch"), ("r", "R"))}
SHARES = {   # name -> (arm, reference, released, column): the arm's share (%) of the reference's gain over the released model
    "tgShareKlein": ("kd_ad2_tg4", "kd_ad2x", "klein", "top1_mean"), "tgShareSD": ("sdt_ad1_tg75", "sdt_ad1_x3", "sdt4", "top1_mean"),
    "tgShareZ": ("zt_ad2_tg5", "zt_ad2_x3", "zimage_turbo", "top1_mean"),
    "tgVendiKlein": ("kd_ad2_tg4", "kd_ad2x", "klein", "vendi"), "tgVendiSD": ("sdt_ad1_tg75", "sdt_ad1", "sdt4", "vendi"),
    "tgVendiZ": ("zt_ad2_tg5", "zt_ad2", "zimage_turbo", "vendi"),
    "tgRecallKlein": ("kd_ad2_tg4", "kd_ad2x", "klein", "recall"), "tgRecallSD": ("sdt_ad1_tg75", "sdt_ad1", "sdt4", "recall"),
    "tgRecallZ": ("zt_ad2_tg5", "zt_ad2", "zimage_turbo", "recall"),
    "anchorShareKlein": ("kd_ad0", "kd_ad2x", "klein", "top1_mean"), "anchorShareSD": ("sdt_ad0", "sdt_ad1_x3", "sdt4", "top1_mean"),
    "anchorVendiKlein": ("kd_ad0", "kd_ad2x", "klein", "vendi"), "anchorVendiSD": ("sdt_ad0", "sdt_ad1", "sdt4", "vendi"),
    "capShareKlein": ("cap_kd_ad2", "cap_kb_g1", "cap_klein", "top1_mean"),
    "capShareSD": ("cap_sdt_ad1", "cap_sd21_g1", "cap_sdt4", "top1_mean"),
    "llmShareKlein": ("llm_kd_ad2", "llm_kb_g1", "llm_klein", "top1_mean"),
    "llmShareSD": ("llm_sdt_ad1", "llm_sd21_g2", "llm_sdt4", "top1_mean"),
    "dpdmdAdShare": ("dpdmd_ad0_x3", "sd35m_g1", "dpdmd", "top1_mean"),
    "dpdmdAdDeepShare": ("dpdmd_ad1", "sd35m_g1", "dpdmd", "top1_mean"),
    "inkFullShareKlein": ("in1k_kd_ad2_full", "in1k_kb_g1_s20_full", "in1k_klein_full", "top1_mean"),
    "inkFullShareSD": ("in1k_sdt_ad1_full", "in1k_sd21_g1_full", "in1k_sdt4_full", "top1_mean"),
}
ADAPTER_RUNS = {"Klein": ["kd_ad2", "kd_ad2b", "kd_ad2_s2"], "SD": ["sdt_ad1", "sdt_ad1_s1", "sdt_ad1_s2"],
                "Z": ["zt_ad2", "zt_ad2_s1", "zt_ad2_s2"]}
TP = ["dpdmd", "dpdmd_ad1", "dpdmd_dd1_g1", "in1k_kb_g1_s20", "in1k_kd_ad2", "in1k_kd_ad2_full", "in1k_klein",
      "in1k_klein_full", "in1k_sd21_g1", "in1k_sdt4_full", "in1k_sdt_ad1", "in1k_sdt_ad1_full", "kb_b2d25_g1", "kb_d2b25_g1",
      "kb_dd2_g1", "kb_interval_g4", "kb_or10_d3", "kb_or21_d2", "kb_or35_d1_g4", "kd_ad0", "kd_ad2_tg4", "kd_ad2_x3", "kd_ad2x",
      "kd_ad3", "klein", "low_real50", "scale_kd_ad2_100", "scale_kd_ad2_3000", "scale_klein_100", "scale_klein_3000", "sd21",
      "sd21_g1", "sd21_g2", "sd21_interval75", "sdt4", "sdt4_dd2_g1", "sdt_ad0", "sdt_ad1_x3", "sdt_ad2", "sdt_or15_d3",
      "sdt_or30_d2", "sdt_or30_d2_g75", "zt_ad2_x3", "zt_sda_all"]
TPSD = ["kd_ad2_x3", "klein", "zimage_turbo_s8", "zt_sda_all", "zt_sda_first2"]
TPSEC = ["kb_g1_s20", "kd_ad2_x3"]
WORDS = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four"}


def nozero(v):
    """A number that rounds to zero carries no sign: "$-$0.0" and "+0.0" print as "0.0"."""
    return re.sub(r"(\$-\$|\+)(0\.0+)(?!\d)", r"\2", str(v))


def spearman_ci(x, y):
    """Spearman correlation of two per-class vectors with a 95% interval from 10,000 resamples of the classes."""
    from fau.metric_study import pearson, rank, summary
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    idx = np.random.default_rng(0).integers(0, len(x), size=(10000, len(x)))
    return summary(pearson(rank(x), rank(y)), pearson(rank(x[idx]), rank(y[idx])), len(x))


def build(R, study, weighting):
    out = {}

    def put(kind, name, value, clean=True):
        out[f"{kind}@{name}"] = nozero(value) if clean else str(value)

    for g in TP:
        put("tp", g, f"{R.mean(g):.1f}")
    for g in TPSD:
        put("tpsd", g, f"{R.mean(g):.1f}$\\pm${R.sd(g):.1f}")
    for g in TPSEC:
        put("tpsec", g, f"{R.col(g, 'sec'):.2f}")

    for name, (a, b) in GAPS.items():
        g, lo, hi = gap(R.per_class(a), R.per_class(b))
        put("gp", name, signed(g))
        put("gpci", name, interval(lo, hi))
    for name, (a, b) in MAGNITUDES.items():
        g, lo, hi = gap(R.per_class(a), R.per_class(b))
        put("gpa", name, f"{abs(g):.1f}")
        alo, ahi = (lo, hi) if g >= 0 else (-hi, -lo)   # the interval of |gap|
        put("gpcia", name, f"[{alo:.1f}, {ahi:.1f}]".replace("-", "$-$"))
    for name, (a, b, test) in OOD_GAPS.items():
        g, lo, hi = gap_by_class(R.ood_per_class(a, test), R.ood_per_class(b, test))
        put("gp", name, signed(g))
        put("gpci", name, interval(lo, hi))
    for name, (a, b) in [("native", ("dpdmd_ad1_r1024", "dpdmd_r1024")), ("resolutionReleased", ("dpdmd_r1024", "dpdmd")),
                         ("resolutionAdapted", ("dpdmd_ad1_r1024", "dpdmd_ad1"))]:   # DP-DMD at its native resolution
        g, lo, hi = gap(R.per_class(a), R.per_class(b), exact_mean=True)
        put("gp", "e49" + name, signed(g), clean=False)
        put("gpci", "e49" + name, interval(lo, hi), clean=False)

    shares = {}
    for name, (arm, ref, released, col) in SHARES.items():
        v = (lambda g: R.mean(g)) if col == "top1_mean" else (lambda g, c=col: R.col(g, c))
        shares[name] = share(v(arm), v(ref), v(released))
        put("nm", name, f"{shares[name]:.0f}".replace("-", "$-$"))
    for kind in ("Vendi", "Recall"):   # a default-guidance target's largest share over the three families
        put("nm", f"tg{kind}Max", f"{max(shares[f'tg{kind}{f}'] for f in ('Klein', 'SD', 'Z')):.0f}")
    for fam, groups in ADAPTER_RUNS.items():   # the adapter over its three training runs
        v = [R.mean(g) for g in groups]
        put("nm", f"ad{fam}Range", f"{min(v):.1f}--{max(v):.1f}")
    regen = [gap(R.per_class(a), R.per_class(b))[0] for a, b in
             [("klein_ns1", "scale_klein_500"), ("kd_ad2_ns1", "kd_ad2"), ("sdt4_ns1", "sdt4"), ("sdt_ad1_ns1", "sdt_ad1")]]
    put("nm", "regenMax", f"{max(abs(g) for g in regen):.1f}")
    pairs = [("kd_ad2", "klein"), ("sdt_ad1", "sdt4"), ("zt_ad2", "zimage_turbo")]   # adapter, released (first runs)
    for name, col in (("qRecogLower", "recog"), ("qFidBetter", "fid")):
        k = sum(R.col(a, col) < R.col(d, col) for a, d in pairs)
        put("nm", name, f"all {WORDS[k]}" if k == len(pairs) else WORDS[k])
    gains = [gap_by_class(R.transfer_per_class(a, t), R.transfer_per_class(d, t))[0] for a, d in pairs[:2] for _, t in TRANSFER]
    put("nm", "trGainMin", f"{min(gains):.1f}")
    put("nm", "trGainMax", f"{max(gains):.1f}")

    # Diagnostics: the seed relay (seed share of the similarity to the teacher's images) and the per-class analysis
    relay = R.diagnostics["relay"]
    for pair, name in (("kd_ad2:kb_or34_d1", "relayAdKlein"), ("klein:kb_or34_d1", "relayDistKlein"),
                       ("kb_dd2_g1:kb_or34_d1", "relayBfKlein"), ("sdt_ad1:sdt_or30_d2", "relayAdSD"),
                       ("sdt4:sdt_or30_d2", "relayDistSD"), ("sdt4_dd2_g1:sdt_or30_d2", "relayBfSD")):
        put("nm", name, f"{relay[pair]['seed_share']:.2f}")
        R.used.update(pair.split(":"))
    recall = R.diagnostics["per_class_recall"]   # per class, the share of its 500 real images a set recalls

    def seeds_by_class(g):
        pc = R.per_class(g)
        return 100 * np.array([pc[s] for s in sorted(pc, key=int)], dtype=np.float64)
    for src, tgt, fam in (("klein", "kd_ad2", "Klein"), ("sdt4", "sdt_ad1", "SD")):
        gain = seeds_by_class(tgt).mean(0) - seeds_by_class(src).mean(0)
        c = spearman_ci(np.array(recall[tgt]) - np.array(recall[src]), gain)
        put("nm", f"pcRecallChange{fam}", f"{signed(c['r'], 2)} {interval(c['lo'], c['hi'], 2)}")

    # The metric study
    put("nm", "eNSets", study["metrics"]["dino_recall"]["n"])
    paired = weighting["clip_fd_minus_dino_recall"]
    put("nm", "equalFamilyFdMinusRecall", signed(paired["value"], 2), clean=False)
    put("nm", "equalFamilyFdMinusRecallCI", interval(*paired["interval"], 2), clean=False)

    # Costs
    for fam, adapter, base in (("Klein", "kd_ad2", "kb_g1_s20"), ("SD", "sdt_ad1", "sd21_g1")):
        prep = preparation(R, adapter)[2]
        put("nm", f"prepHours{fam}", f"{prep:.1f}")
        put("nm", f"breakEven{fam}", f"{round(round(crossover(prep, R.col(base, 'sec'), R.col(adapter, 'sec'))) / 1000) * 1000:,}")
    for fam, adapter, basefirst in (("Klein", "kd_ad2", "kb_dd2_g1"), ("SD", "sdt_ad1", "sdt4_dd2_g1")):
        prep = preparation(R, adapter)[2]
        put("nm", f"breakEvenBaseFirst{fam}", f"{round(crossover(prep, R.col(basefirst, 'sec'), R.col(adapter, 'sec')) / 1000) * 1000:,}")
    for fam, adapter, ad_full, bf_full in (("Klein", "kd_ad2", "in1k_kd_ad2_full", "in1k_kb_dd2_g1_full"),
                                           ("SD", "sdt_ad1", "in1k_sdt_ad1_full", "in1k_sdt4_dd2_g1_full")):
        prep = preparation(R, adapter)[2]
        sa, sb = R.col(ad_full, "sec"), R.col(bf_full, "sec")
        put("nm", f"fullTotalAd{fam}", f"{prep + int(R.col(ad_full, 'images')) * sa / 3600:.1f}")
        put("nm", f"fullTotalBf{fam}", f"{int(R.col(bf_full, 'images')) * sb / 3600:.1f}")
        put("nm", f"e51CrossoverMillion{fam}", f"{round(crossover(prep, sb, sa)) / 1e6:.1f}")

    # Prompt regimes: ImageNet-1k at 100 images/class under LLM prompts, and the ImageNet-100 replications
    unseen = unseen_classes(R)
    e50 = {"OriginalKleinVsLlmReleased": ("in1k_kd_ad2", "in1k_llm_klein"),
           "OriginalSDVsLlmReleased": ("in1k_sdt_ad1", "in1k_llm_sdt4"),
           "LlmBaseFirstKlein": ("llmx_kd_llm", "llmx_kb_dd2_g1"),
           "LlmVsTemplateAtScale": ("in1k_llm_kd_llm", "in1k_llm_kd_ad2"),
           "PostHocKleinVsReleased": ("in1k_llm_kd_ad2", "in1k_llm_klein"),
           "PostHocKleinVsBaseFirst": ("in1k_llm_kd_ad2", "in1k_llm_kb_dd2_g1")}
    for name, (a, b) in e50.items():
        g, lo, hi = gap(R.per_class(a), R.per_class(b))
        put("nm", "e50" + name, signed(g), clean=False)
        put("nm", "e50" + name + "CI", interval(lo, hi), clean=False)
    for name, (a, b) in {"NineHundredkd_llm": ("in1k_llm_kd_llm", "in1k_llm_klein"),
                         "NineHundredsdt_ad1": ("in1k_llm_sdt_ad1", "in1k_llm_sdt4"),
                         "PostHocKleinNineHundred": ("in1k_llm_kd_ad2", "in1k_llm_klein")}.items():
        g, lo, hi = gap(R.per_class(a, unseen), R.per_class(b, unseen))
        put("nm", "e50" + name, signed(g), clean=False)
        put("nm", "e50" + name + "CI", interval(lo, hi), clean=False)
    for name, seed in (("PoorRun", 0), ("OtherRunOne", 1), ("OtherRunTwo", 2)):   # base-first's second LLM dataset
        put("nm", "e50" + name, f"{R.run('llm2_kb_dd2_g1', seed):.1f}", clean=False)
    # sensitivity: LLM-fitted against base-first with base-first's poor run left out
    pooled = {label: {prefix + seed: v for prefix in ("llm_", "llm2_", "llm3_") for seed, v in R.per_class(prefix + arm).items()
                      if not (arm == "kb_dd2_g1" and prefix == "llm2_" and seed == "0")}
              for label, arm in (("a", "kd_llm"), ("b", "kb_dd2_g1"))}
    g, lo, hi = gap(pooled["a"], pooled["b"])
    put("nm", "e50Sensitivity", signed(g), clean=False)
    put("nm", "e50SensitivityCI", interval(lo, hi), clean=False)
    put("nm", "e52KleinBaseLoss", f"{abs(gap(R.per_class('in1k_llm_kb_g1_s20'), R.per_class('in1k_llm_klein'))[0]):.1f}", clean=False)
    put("nm", "e52SDBaseGain", f"{gap(R.per_class('in1k_llm_sd21_g1'), R.per_class('in1k_llm_sdt4'))[0]:.1f}", clean=False)
    return out
