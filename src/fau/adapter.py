"""The first-step adapter: a low-rank update of a released distilled model whose first prediction lands, in one jump from
pure noise, on the state its unguided base reaches after its own fine early steps from the same noise. Generation then
needs no base model: the adapted first step, then the released model's remaining steps. Families (--family):
  klein   FLUX.2-klein from the klein base: noise 1 -> 0.958, 0.884, 0.7175 (the paper's target) or 0.511; flow-matching
          Euler steps on the base's fine grid (--grid v1: the grid of the primary adapters; fine: one level more, 0.958)
  sd      SD-Turbo from SD 2.1 (same text encoder and VAE): timestep 999 -> 749, 499 (the paper's target) or 249, on
          SD-Turbo's Euler pipeline (latents x0 + sigma eps); the teacher runs SD 2.1's U-Net on 30 fine Euler steps to 499
  zimage  Z-Image-Turbo from the Z-Image base: noise 1 -> 0.96, 0.913 or 0.857 (the paper's target), Turbo's levels after
          1, 2 and 3 of its 9 steps; flow matching, latents in float32
  sd35m   DP-DMD's released SD3.5-Medium student from unguided SD3.5-Medium: noise 1 -> 0.858 or 0.602, the student's
          levels after 1 and 2 of its 4 steps; flow matching

Modes:
  pairs  The teacher: the base on its fine grid at guidance --guidance (1: unguided; w in neg + w (pos - neg)), plus the
         base on the student's own coarse levels (the base-first state) from the same noise as a reference. Writes x_1,
         the state at every target and its noise level per (prompt group k, index j). Prompts are the class templates
         (j mod 8); seeds are 1_000_003 k + 600_000 + j, disjoint from the generated sets' 1_000_003 k + j (j < 500).
  train  The student: rank-r updates (B zero-initialized, so training starts from the released model exactly) of every
         linear layer of the distilled transformer's blocks (of the U-Net's attention blocks for SD), or --full (klein,
         SD): every weight, in float32; one Euler step from noise 1 to the target level; loss: mean squared error to the
         teacher state. 5% of the pairs (at least 64; split seed 0) are held out; their relative error
         ||x_pred - x_t||^2 / ||x_1 - x_t||^2 is reported for the unadapted model, the base-first state and, at the end,
         the update folded into the weights in the inference precision, as generation runs it.

generate.py loads a fitted adapter as the partner 'klein_ad:ID', 'sdt_ad:ID', 'zt_ad:ID' or 'sd35m_ad:ID' (merge()).

Usage: python -m fau.adapter pairs --config pairs_in100 [--shard i]       # teacher pairs of the paper (configs/pairs.json)
       python -m fau.adapter train --config ad2                           # an adapter of the paper (configs/adapters.json)
       python -m fau.adapter pairs --family klein --classes in100 --per_class 160 --out DIR [--guidance 1] [--shard i --nshards n]
       python -m fau.adapter train --family klein --pairs DIR --classes in100 --target 7175 --out ADAPTER.pt
"""
import argparse, glob, json, math, os, time

import numpy as np

import torch
import torch.nn as nn

from fau import ROOT, WORK
from fau.generate import NATIVE_OFFSET, TEMPLATES, class_name, load, native_sigmas, parse_sigmas, partner
from fau.hub import snapshot
from fau.prompts import class_list, load_prompts

RES = 512
# klein's teacher grids: its native 50-step levels with the distilled model's 4-step levels inserted. "v1", the grid of the
# primary adapters (35 levels, 1 -> 0.7175 in 33 steps); "fine" adds the level 0.958 and runs on to 0.511 (44 levels:
# 1 .. 0.96152, 0.95809, 0.95641 .. 0.89162, 0.88398, 0.87469 .. 0.73859, 0.7175, 0.70154 .. 0.51093)
GRIDS = {"v1": ("50[0:21],4[2:3],50[22:34],4[3:4]", {"884": 20, "7175": 33}),
         "fine": ("50[0:10],4[1:2],50[10:21],4[2:3],50[22:34],4[3:4],50[35:43]", {"958": 9, "884": 21, "7175": 34, "511": 42})}
STATE_AT = GRIDS["fine"][1]                 # step index after which the latent sits at that noise level
COARSE = "4[0:3]"                           # 1, 0.95809, 0.88398: the base-first fix's 2 base steps
# SD: 30 fine timesteps 999 -> 499 (749 the 15th), 15 more to 249 (SD-Turbo's own 4 steps are 999, 749, 499, 249)
SD_FINE = [int(round(x)) for x in np.linspace(999, 499, 31)] + [int(round(x)) for x in np.linspace(499, 249, 16)[1:]]
SD_STATE_AT = {"749": 14, "499": 29, "249": 44}
SD_COARSE = [999, 749, 499]                 # the base-first fix's 2 SD 2.1 steps on SD-Turbo's schedule
# The MMDiT families (flow matching, Euler steps): the teacher's grid (parse_sigmas; unprefixed parts are the student's
# own levels) and the anchors as indices into the student's own schedule; latents in the pipeline's precision.
# Z-Image: the base's native 50-step levels (static shift 6) with Turbo's (shift 3) put in, in place of the base's level
# where the two coincide (0.96, 0.857): 1 -> 0.857 in 26 steps. SD3.5-Medium: its native 55-step levels hold the
# student's 4 (indices 0, 18, 36, 54): 1 -> 0.602 in 36 steps.
MMDIT = {"zimage": dict(teacher="zimage", student="zimage_turbo", steps=9, anchors={"960": 1, "913": 2, "857": 3},
                        fine="zimage:50[0:10],9[1:2],zimage:50[11:19],9[2:3],zimage:50[19:25],9[3:4]", dtype=torch.float32),
         "sd35m": dict(teacher="sd35_medium", student="dpdmd_sd35m", steps=4, anchors={"858": 1, "602": 2},
                       fine="sd35_medium:55[0:18],4[1:2],sd35_medium:55[19:36],4[2:3]", dtype=torch.bfloat16)}


class LoRALinear(nn.Module):
    """y = W x + (alpha / r) B A x, the low-rank part in float32; B starts at zero."""

    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base, self.scale, self.enabled = base, alpha / rank, True
        self.A = nn.Parameter(torch.randn(rank, base.in_features, device=base.weight.device) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))

    def forward(self, x):
        y = self.base(x)
        if not self.enabled:
            return y
        return y + ((x.float() @ self.A.t()) @ self.B.t() * self.scale).to(y.dtype)


def targets(net):
    """Every linear layer of the transformer blocks (klein, SD3.5-Medium), of the U-Net's attention blocks (SD), or of
    Z-Image's blocks: its 30 main layers and the two refiners of the image tokens and of the text tokens (unmodulated)."""
    zimage = ("layers", "noise_refiner", "context_refiner")
    return [n for n, m in net.named_modules() if isinstance(m, nn.Linear) and
            ("transformer_blocks." in n or "attentions." in n or n.split(".")[0] in zimage)]


def attach(transformer, rank, alpha):
    names = targets(transformer)
    for n in names:
        parent, attr = transformer.get_submodule(n.rsplit(".", 1)[0]), n.rsplit(".", 1)[1]
        setattr(parent, attr, LoRALinear(getattr(parent, attr), rank, alpha))
    return names


def merge(transformer, path):
    """Folds a saved adapter into the weights: W <- W + (alpha / r) B A, or W <- W + delta for a full fine-tune."""
    ad = torch.load(path, map_location="cpu", weights_only=True)
    with torch.no_grad():
        if ad.get("full"):
            for n, d in ad["delta"].items():
                p = transformer.get_parameter(n)
                p.copy_((p.float() + d.to(p.device).float()).to(p.dtype))
            print(f"merged full fine-tune {path}: {len(ad['delta'])} tensors, target noise {ad['sigma_target']}", flush=True)
            return transformer
        scale = ad["alpha"] / ad["rank"]
        for n, (A, B) in ad["weights"].items():
            lin = transformer.get_submodule(n)
            lin.weight += (scale * B.float() @ A.float()).to(device=lin.weight.device, dtype=lin.weight.dtype)
    print(f"merged adapter {path}: {len(ad['weights'])} layers, rank {ad['rank']}, target noise {ad['sigma_target']}", flush=True)
    return transformer


def names_of(classes):
    meta = class_list(classes)
    return [class_name(w, n) for w, n in zip(meta["wnids"], meta["names"])]


def prompt_source(a):
    """-> (number of prompt groups k, prompt of (k, j), the list of distinct prompts, index of (k, j) in that list). By
    default k is a class and j cycles the 8 templates; with --prompts (a list of captions, e.g. COCO's) k is a caption and
    j only indexes its noises. Seeds are 1_000_003 k + 600_000 + j either way. A per-class description file
    ({"captions": [[...], ...]}, LLM descriptions) is taken class by class, each description as generate.py --captions
    prompts it: '<class name>, <description>'."""
    if getattr(a, "prompts", None):
        caps = load_prompts(a.prompts)
        if isinstance(caps, dict):
            names = names_of(a.classes)
            assert len(caps["captions"]) == len(names), f"{a.prompts}: {len(caps['captions'])} classes, {a.classes}: {len(names)}"
            caps = [f"{names[c]}, {t}" for c, per in enumerate(caps["captions"]) for t in per]
        return len(caps), (lambda k, j: caps[k]), caps, (lambda k, j: k)
    names = names_of(a.classes)
    nT = len(TEMPLATES)
    return (len(names), (lambda k, j: TEMPLATES[j % nT].format(names[k])),
            [TEMPLATES[t].format(names[c]) for c in range(len(names)) for t in range(nT)], (lambda k, j: k * nT + j % nT))


def restrict(stops, a, prefix="t"):
    """--anchors: keep only these anchors (the teacher then stops at the last of them)."""
    if not getattr(a, "anchors", None):
        return stops
    keep = {prefix + n for n in a.anchors.split(",")}
    assert keep <= set(stops), f"unknown anchors {keep - set(stops)}; known {sorted(stops)}"
    return {n: v for n, v in stops.items() if n in keep}


def grab(got, stops):
    """callback_on_step_end that keeps the latents after the given steps and stops the loop after the last of them."""
    def cb(p, i, t, kw):
        for name, at in stops.items():
            if i == at:
                got[name] = kw["latents"].detach().cpu()
        if i >= max(stops.values()):
            p._interrupt = True
        return kw
    return cb


def pairs_sd(a):
    from diffusers.utils.torch_utils import randn_tensor
    out = f"{a.out}/pairs{a.shard}.pt"
    if os.path.exists(out):
        print(f"{out} exists, nothing to do"); return
    npg, prompt_of, _, _ = prompt_source(a)
    jobs = [(k, j) for k in range(npg) for j in range(a.per_class)][a.shard::a.nshards]
    pipe, _, res, bs, _ = load("sd_turbo")
    assert type(pipe.scheduler).__name__ == "EulerDiscreteScheduler", type(pipe.scheduler).__name__
    pipe.unet = partner("sd21")   # the teacher: SD 2.1's UNet on SD-Turbo's pipeline (same text encoder and VAE)
    pipe.to("cuda"); pipe.set_progress_bar_config(disable=True)
    g = a.guidance if a.guidance > 1 else 0.0   # the pipeline runs classifier-free guidance for g > 1 only
    stops = restrict({"t" + n: at for n, at in SD_STATE_AT.items()}, a)
    rec, sig, t0 = dict(k=[], j=[], x1=[], c499=[], **{n: [] for n in stops}), None, time.time()
    for b in range(0, len(jobs), bs):
        chunk = jobs[b:b + bs]
        prompts = [prompt_of(k, j) for k, j in chunk]
        gens = [torch.Generator("cuda").manual_seed(1_000_003 * k + 600_000 + j) for k, j in chunk]
        noise = randn_tensor((len(chunk), 4, res // 8, res // 8), generator=gens, device=torch.device("cuda"), dtype=torch.float16)
        got = {}
        with torch.no_grad():
            pipe(prompt=prompts, latents=noise, timesteps=SD_FINE, guidance_scale=g, height=res, width=res, output_type="latent",
                 callback_on_step_end=grab(got, stops))
            assert pipe.do_classifier_free_guidance == (g > 1)
            x1 = (noise * pipe.scheduler.init_noise_sigma).cpu()   # the start the pipeline used (prepare_latents)
            if sig is None:
                s = pipe.scheduler.sigmas
                sig = dict(s_max=float(s[0]), sig={n: float(s[at + 1]) for n, at in SD_STATE_AT.items() if "t" + n in stops})
            pipe(prompt=prompts, latents=noise, timesteps=SD_COARSE, guidance_scale=g, height=res, width=res, output_type="latent",
                 callback_on_step_end=grab(got, {"c499": 1}))
        rec["x1"].append(x1)
        for name in got:
            rec[name].append(got[name])
        rec["k"] += [k for k, _ in chunk]; rec["j"] += [j for _, j in chunk]
        if (b // bs) % 25 == 0:
            print(f"pairs shard {a.shard}: {b + len(chunk)}/{len(jobs)}, {(time.time() - t0) / (b + len(chunk)):.3f} s each", flush=True)
    os.makedirs(a.out, exist_ok=True)
    torch.save(dict(family="sd", guidance=a.guidance, k=torch.tensor(rec["k"]), j=torch.tensor(rec["j"]), fine=SD_FINE, coarse=SD_COARSE,
                    **sig, **{n: torch.cat(v) for n, v in rec.items() if n not in ("k", "j")}), out + ".tmp")
    os.replace(out + ".tmp", out)
    print(f"PAIRS DONE shard {a.shard}: {len(jobs)} pairs at guidance {a.guidance} in {time.time() - t0:.0f} s; sigma max "
          f"{sig['s_max']:.4f}, targets {sig['sig']}", flush=True)


def pairs_mmdit(a):
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.utils.torch_utils import randn_tensor
    f = MMDIT[a.family]
    out = f"{a.out}/pairs{a.shard}.pt"
    if os.path.exists(out):
        print(f"{out} exists, nothing to do"); return
    names = names_of(a.classes)
    jobs = [(k, j) for k in range(len(names)) for j in range(a.per_class)][a.shard::a.nshards]
    own = native_sigmas(f["steps"], RES, f["student"])   # the distilled model's own levels
    fine = parse_sigmas(f["fine"], RES, f["student"])
    sig = {n: own[i] for n, i in f["anchors"].items()}
    assert all(s in fine for s in sig.values()), f"the teacher grid misses an anchor {sig}: {fine}"
    stops = {"t" + n: fine.index(s) - 1 for n, s in sig.items()}   # the step after which the teacher sits at the anchor
    coarse = own[:max(f["anchors"].values()) + 1]                    # base-first: the base on the student's own levels
    cstops = {"c" + n: i - 1 for n, i in f["anchors"].items()}
    pipe, _, _, bs, _ = load(f["teacher"])
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config, use_dynamic_shifting=False, shift=1.0)
    pipe.to("cuda"); pipe.set_progress_bar_config(disable=True)
    g = a.guidance - NATIVE_OFFSET.get(f["teacher"], 0.0)   # Z-Image's own scale: w - 1
    ch = pipe.transformer.config.in_channels
    rec = dict(k=[], j=[], x1=[], **{n: [] for n in [*stops, *cstops]})
    t0 = time.time()
    for b in range(0, len(jobs), bs):
        chunk = jobs[b:b + bs]
        prompts = [TEMPLATES[j % len(TEMPLATES)].format(names[k]) for k, j in chunk]
        gens = [torch.Generator("cuda").manual_seed(1_000_003 * k + 600_000 + j) for k, j in chunk]
        noise = randn_tensor((len(chunk), ch, RES // 8, RES // 8), generator=gens, device=torch.device("cuda"), dtype=f["dtype"])
        got = {}
        with torch.no_grad():   # prompt lists as copies: Z-Image's pipeline writes its chat template into them
            pipe(prompt=list(prompts), latents=noise, sigmas=fine, num_inference_steps=len(fine), guidance_scale=g, height=RES,
                 width=RES, output_type="latent", callback_on_step_end=grab(got, stops))
            assert pipe.do_classifier_free_guidance == (a.guidance > 1)
            pipe(prompt=list(prompts), latents=noise, sigmas=coarse, num_inference_steps=len(coarse), guidance_scale=g, height=RES,
                 width=RES, output_type="latent", callback_on_step_end=grab(got, cstops))
        rec["x1"].append(noise.cpu())
        for name in got:
            rec[name].append(got[name])
        rec["k"] += [k for k, _ in chunk]; rec["j"] += [j for _, j in chunk]
        if (b // bs) % 25 == 0:
            print(f"pairs shard {a.shard}: {b + len(chunk)}/{len(jobs)}, {(time.time() - t0) / (b + len(chunk)):.3f} s each", flush=True)
    os.makedirs(a.out, exist_ok=True)
    torch.save(dict(family=a.family, guidance=a.guidance, sig=sig, k=torch.tensor(rec["k"]), j=torch.tensor(rec["j"]), fine=fine,
                    coarse=coarse, **{n: torch.cat(v) for n, v in rec.items() if n not in ("k", "j")}), out + ".tmp")
    os.replace(out + ".tmp", out)
    print(f"PAIRS DONE shard {a.shard}: {len(jobs)} pairs at guidance {a.guidance} in {time.time() - t0:.0f} s "
          f"({(time.time() - t0) / len(jobs):.2f} s per pair, {max(stops.values()) + 1} teacher steps), "
          f"{os.path.getsize(out) / 1e9:.2f} GB; targets {sig}", flush=True)


def pairs(a):
    if a.family == "sd":
        return pairs_sd(a)
    if a.family in MMDIT:
        return pairs_mmdit(a)
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.utils.torch_utils import randn_tensor
    out = f"{a.out}/pairs{a.shard}.pt"
    if os.path.exists(out):
        print(f"{out} exists, nothing to do"); return
    npg, prompt_of, _, _ = prompt_source(a)
    jobs = [(k, j) for k in range(npg) for j in range(a.per_class)][a.shard::a.nshards]
    pipe, _, _, bs, _ = load("klein_base")
    assert not pipe.config.is_distilled   # else the pipeline skips classifier-free guidance
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config, use_dynamic_shifting=False, shift=1.0)
    pipe.to("cuda"); pipe.set_progress_bar_config(disable=True)
    spec, state_at = GRIDS[a.grid]
    fine, coarse = parse_sigmas(spec, RES), parse_sigmas(COARSE, RES)
    ch = pipe.transformer.config.in_channels   # packed channels (4 x the VAE's)
    stops = restrict({"t" + n: at for n, at in state_at.items()}, a)
    sig = {n: fine[at + 1] for n, at in state_at.items() if "t" + n in stops}
    rec = dict(k=[], j=[], x1=[], c884=[], **{n: [] for n in stops})
    t0 = time.time()
    for b in range(0, len(jobs), bs):
        chunk = jobs[b:b + bs]
        prompts = [prompt_of(k, j) for k, j in chunk]
        gens = [torch.Generator("cuda").manual_seed(1_000_003 * k + 600_000 + j) for k, j in chunk]
        noise = randn_tensor((len(chunk), ch, RES // 16, RES // 16), generator=gens, device=torch.device("cuda"), dtype=torch.bfloat16)
        got = {}
        with torch.no_grad():
            pipe(prompt=prompts, latents=noise, sigmas=fine, num_inference_steps=len(fine), guidance_scale=a.guidance, height=RES,
                 width=RES, output_type="latent", callback_on_step_end=grab(got, stops))
            assert pipe.do_classifier_free_guidance == (a.guidance > 1)
            pipe(prompt=prompts, latents=noise, sigmas=coarse, num_inference_steps=len(coarse), guidance_scale=a.guidance, height=RES,
                 width=RES, output_type="latent", callback_on_step_end=grab(got, {"c884": 1}))
        rec["x1"].append(pipe._pack_latents(noise).cpu())
        for name in got:
            rec[name].append(got[name])
        rec["k"] += [k for k, _ in chunk]; rec["j"] += [j for _, j in chunk]
        if (b // bs) % 25 == 0:
            print(f"pairs shard {a.shard}: {b + len(chunk)}/{len(jobs)}, {(time.time() - t0) / (b + len(chunk)):.3f} s each", flush=True)
    os.makedirs(a.out, exist_ok=True)
    torch.save(dict(family="klein", guidance=a.guidance, grid=a.grid, sig=sig, k=torch.tensor(rec["k"]), j=torch.tensor(rec["j"]), fine=fine,
                    coarse=coarse, **{n: torch.cat(v) for n, v in rec.items() if n not in ("k", "j")}), out + ".tmp")
    os.replace(out + ".tmp", out)
    print(f"PAIRS DONE shard {a.shard}: {len(jobs)} pairs at guidance {a.guidance} in {time.time() - t0:.0f} s; targets {sig}", flush=True)


def train(a):
    from fau.generate import WEIGHTS, warm
    torch.manual_seed(a.seed)
    files = sorted(glob.glob(f"{a.pairs}/pairs*.pt"))
    P = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
    cat = lambda n: torch.cat([p[n] for p in P])
    sd = a.family == "sd"
    assert P[0].get("family", "klein") == a.family, f"{a.pairs} holds {P[0].get('family', 'klein')} pairs"
    assert "t" + a.target in P[0], f"{a.pairs} has no state at {a.target}: {[n for n in P[0] if n.startswith('t')]}"
    k, j, x1, tgt = cat("k"), cat("j"), cat("x1"), cat("t" + a.target)
    if a.first:   # only the pairs of the first N prompt groups
        keep = k < a.first
        k, j, x1, tgt = k[keep], j[keep], x1[keep], tgt[keep]
    # the base-first fix's state, where the pairs hold it (klein 884, SD 499; Z-Image and SD3.5 every anchor)
    coarse = cat("c" + a.target) if "c" + a.target in P[0] else None
    if coarse is not None and a.first:
        coarse = coarse[keep]
    sig_0 = P[0]["s_max"] if sd else 1.0
    sig_t = P[0]["sig"][a.target] if "sig" in P[0] else P[0]["fine"][GRIDS["v1"][1][a.target] + 1]   # early pair files
    teacher_g = P[0].get("guidance", 1.0)
    _, _, prompts, pid_of = prompt_source(a)
    pid = torch.tensor([pid_of(int(c), int(i)) for c, i in zip(k, j)])   # index into prompts: (class, template) or caption
    n = len(k)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    held, tr = perm[: max(64, n // 20)], perm[max(64, n // 20):]
    print(f"{n} pairs from {len(files)} shards ({a.family}, teacher guidance {teacher_g}); noise {sig_0:.4f} -> {sig_t:.5f}; "
          f"{len(tr)} train, {len(held)} held out", flush=True)

    amp = [torch.bfloat16]   # SD's forward precision: bfloat16 to train; the merged check runs in float16, as generate.py does
    if sd:
        from diffusers import AutoPipelineForText2Image
        warm(WEIGHTS["sd_turbo"], "sd_turbo")
        pipe = AutoPipelineForText2Image.from_pretrained(snapshot("stabilityai/sd-turbo"), variant="fp16", torch_dtype=torch.float16)
        pipe.text_encoder.to("cuda")
        with torch.no_grad():
            emb = torch.cat([pipe.encode_prompt(prompts[i:i + 64], torch.device("cuda"), 1, False)[0].cpu() for i in range(0, len(prompts), 64)])
        net = pipe.unet.float().cuda()   # float32 weights, bfloat16 autocast in the forward (float16 for the merged check)
        t999 = torch.tensor(999.0, device="cuda")

        def forward(idx):
            xb = x1[idx].cuda()
            with torch.autocast("cuda", dtype=amp[0]):
                eps = net(xb.float() / (sig_0 ** 2 + 1) ** 0.5, t999, encoder_hidden_states=emb[pid[idx]].cuda()).sample
            return xb.float() + (sig_t - sig_0) * eps.float()   # one Euler step in sigma, as SD-Turbo's scheduler takes it
    elif a.family == "klein":
        from diffusers import Flux2KleinPipeline
        warm(WEIGHTS["klein"], "klein")
        pipe = Flux2KleinPipeline.from_pretrained(snapshot("black-forest-labs/FLUX.2-klein-4B"), torch_dtype=torch.bfloat16)
        pipe.text_encoder.to("cuda")
        if len(prompts) > 2000:   # a caption list: 512 x 7680 per prompt would not fit, so each batch is encoded
            emb = None            # when it is used, with the text encoder kept on the GPU

            def embed(ids):
                with torch.no_grad():
                    return pipe.encode_prompt([prompts[i] for i in ids], device=torch.device("cuda"))[0]
            first = embed([0])
        else:
            with torch.no_grad():
                emb = torch.cat([pipe.encode_prompt(prompts[i:i + 16], device=torch.device("cuda"))[0].cpu() for i in range(0, len(prompts), 16)])
            embed = lambda ids: emb[torch.as_tensor(ids)].cuda()
            first = emb[:1]
        txt_ids = pipe._prepare_text_ids(first).cuda()
        img_ids = pipe._prepare_latent_ids(torch.zeros(1, x1.shape[2], RES // 16, RES // 16)).cuda()
        net = pipe.transformer.cuda()
        if a.full:
            net.float()   # float32 master weights: a bfloat16 update at lr 1e-5 is below the weights' resolution

        def forward(idx):
            xb = x1[idx].cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.full):
                v = net(hidden_states=xb, timestep=torch.ones(len(idx), device="cuda", dtype=xb.dtype), guidance=None,
                        encoder_hidden_states=embed(pid[idx].tolist()), txt_ids=txt_ids.expand(len(idx), -1, -1),
                        img_ids=img_ids.expand(len(idx), -1, -1), return_dict=False)[0]
            return xb.float() + (sig_t - 1.0) * v.float()
    else:   # Z-Image, SD3.5: the student's own pipeline, its transformer in bfloat16, inputs as the pipelines pass them
        assert not a.full, "--full keeps float32 weights and two Adam moments (16 bytes per parameter): 6-8B models do not fit one H100"
        pipe = load(MMDIT[a.family]["student"])[0]
        net = pipe.transformer.cuda()
        if MMDIT[a.family]["teacher"] == "zimage":   # zimage and zimage_deep
            pipe.text_encoder.to("cuda")
            with torch.no_grad():   # one embedding per prompt, of its own length: padded here, cut back in the forward
                E = [e.cpu() for i in range(0, len(prompts), 16) for e in pipe.encode_prompt(prompts[i:i + 16], torch.device("cuda"), False)[0]]
            lens, emb = [len(e) for e in E], nn.utils.rnn.pad_sequence(E, batch_first=True)

            def forward(idx):   # a list of (C, 1, H, W) latents, timestep (1000 - t) / 1000 = 0, output minus the velocity
                xb = x1[idx].cuda()
                out = net(list(xb.to(torch.bfloat16).unsqueeze(2).unbind(0)), torch.zeros(len(idx), device="cuda"),
                          [emb[i, :lens[i]].cuda() for i in pid[idx].tolist()], return_dict=False)[0]
                return xb.float() - (sig_t - 1.0) * torch.stack(out).float().squeeze(2)
        else:
            tes = [pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3]   # CLIP-L, CLIP-G, T5-XXL
            for te in tes:
                te.to("cuda")
            with torch.no_grad():
                enc = [pipe.encode_prompt(prompts[i:i + 16], None, None, device=torch.device("cuda"), do_classifier_free_guidance=False)
                       for i in range(0, len(prompts), 16)]
            emb, pooled = torch.cat([e[0].cpu() for e in enc]), torch.cat([e[2].cpu() for e in enc])
            for te in tes[1:]:
                te.to("cpu")

            def forward(idx):   # timestep sigma * 1000
                xb = x1[idx].cuda()
                v = net(hidden_states=xb, timestep=torch.full((len(idx),), 1000.0, device="cuda"), encoder_hidden_states=emb[pid[idx]].cuda(),
                        pooled_projections=pooled[pid[idx]].cuda(), return_dict=False)[0]
                return xb.float() + (sig_t - 1.0) * v.float()
    if emb is not None:
        print(f"prompt embeddings {tuple(emb.shape)} ({emb.numel() * emb.element_size() / 1e9:.2f} GB)", flush=True)
        pipe.text_encoder.to("cpu"); torch.cuda.empty_cache()
    else:
        print(f"{len(prompts)} prompts, encoded batch by batch (the text encoder stays on the GPU)", flush=True)
    infer_dtype = torch.float16 if sd else torch.bfloat16   # the denoiser's precision in generate.py
    if a.full:
        w0 = {n: p.detach().to("cpu", infer_dtype, copy=True) for n, p in net.named_parameters()}   # exact: the weights ship in it
        params = [p.requires_grad_(True) for p in net.parameters()]
        names_lora = []
        print(f"full fine-tune: {sum(p.numel() for p in params) / 1e6:.1f}M parameters in float32", flush=True)
    else:
        for p in net.parameters():
            p.requires_grad_(False)
        names_lora = attach(net, a.rank, a.alpha)
        params = [p for m in net.modules() if isinstance(m, LoRALinear) for p in (m.A, m.B)]
        print(f"adapters on {len(names_lora)} linear layers, {sum(p.numel() for p in params) / 1e6:.1f}M parameters", flush=True)
    if not sd:
        net.enable_gradient_checkpointing()
    step_pred = forward

    def rel_err(pred, idx, content=False):
        t, x = tgt[idx].cuda().float(), x1[idx].cuda().float()
        ref = x * (sig_t / sig_0) if content else x   # content: against the empty image's state
        return ((pred - t).pow(2).flatten(1).sum(1) / (ref - t).pow(2).flatten(1).sum(1)).tolist()

    def evaluate(content=False):
        net.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(held), a.bs):
                idx = held[i:i + a.bs]
                out += rel_err(step_pred(idx), idx, content)
        net.train()
        return sum(out) / len(out)

    def lora(on):
        for m in net.modules():
            if isinstance(m, LoRALinear):
                m.enabled = on

    # the zero-initialized adapter must leave the distilled model exactly as it is
    if not a.full:
        with torch.no_grad():
            idx = held[:2]
            on = step_pred(idx)
            lora(False)
            off = step_pred(idx)
            lora(True)
        assert torch.equal(on, off), "zero-initialized adapters change the output"
    coarse_err, coarse_c = ((sum(rel_err(coarse[held].cuda().float(), held, c)) / len(held) for c in (False, True))
                            if coarse is not None else (float("nan"), float("nan")))
    log = dict(start=evaluate(), coarse_base=coarse_err, start_content=evaluate(True), coarse_base_content=coarse_c, curve=[])
    print(f"held-out relative error: unadapted distilled step {log['start']:.4f}, coarse base (base-first) {coarse_err:.4f}; "
          f"content error: unadapted {log['start_content']:.4f}, coarse base {coarse_c:.4f}", flush=True)

    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, (lambda s: 1.0) if a.schedule == "constant" else
                                              (lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * s / a.steps))))
    net.train(); t0 = time.time()
    for s in range(a.steps):
        idx = tr[torch.randint(len(tr), (a.bs,))]
        loss = (step_pred(idx) - tgt[idx].cuda().float()).pow(2).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        if a.clip:
            torch.nn.utils.clip_grad_norm_(params, a.clip)
        opt.step(); sched.step()
        if (s + 1) % 50 == 0:
            print(f"step {s + 1}/{a.steps}: loss {loss.item():.5f}, {(time.time() - t0) / (s + 1):.2f} s/step", flush=True)
        if (s + 1) % 500 == 0 or s + 1 == a.steps:
            e, ec = evaluate(), evaluate(True); log["curve"].append((s + 1, e, ec))
            print(f"step {s + 1}: held-out relative error {e:.4f}, content error {ec:.4f}", flush=True)
    minutes = (time.time() - t0) / 60
    # merged: the adapter folded into the weights in the inference precision, as generate.py's partner runs it
    with torch.no_grad():
        if a.full:
            delta = {n: p.detach().float().cpu() - w0[n].float() for n, p in net.named_parameters()}
            for n, p in net.named_parameters():
                p.copy_(p.to(infer_dtype).float())
        else:
            keep = {}
            for n in names_lora:
                m = net.get_submodule(n)
                w = m.base.weight
                keep[n] = w.detach().clone()
                d = (m.scale * m.B.float() @ m.A.float()).to(infer_dtype)
                w.copy_((w.to(infer_dtype) + d).to(w.dtype))   # as merge() folds it into the inference-precision weights
            lora(False)
        amp[0] = infer_dtype
        log["merged"], log["merged_content"] = evaluate(), evaluate(True)
        if not a.full:
            lora(True)
            for n in names_lora:
                net.get_submodule(n).base.weight.copy_(keep[n])
    ad = dict(rank=None if a.full else a.rank, alpha=None if a.full else a.alpha, full=a.full, sigma_target=sig_t, target=a.target,
              family=a.family, teacher_guidance=teacher_g, log=log, args=vars(a))
    if a.full:
        ad["delta"] = delta
    else:
        ad["weights"] = {n: (net.get_submodule(n).A.detach().cpu(), net.get_submodule(n).B.detach().cpu()) for n in names_lora}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save(ad, a.out + ".tmp"); os.replace(a.out + ".tmp", a.out)
    json.dump(dict(log=log, args=vars(a), sigma_target=sig_t, teacher_guidance=teacher_g, full=a.full), open(a.out + ".json", "w"), indent=1)
    print(f"ADAPTER DONE {a.out}: held-out relative error {log['start']:.4f} -> {log['curve'][-1][1]:.4f}, merged {log['merged']:.4f} "
          f"(coarse base {coarse_err:.4f}); content error {log['start_content']:.4f} -> {log['curve'][-1][2]:.4f}, merged "
          f"{log['merged_content']:.4f} (coarse {coarse_c:.4f}); teacher guidance {teacher_g}, {'full' if a.full else f'rank {a.rank}'}, "
          f"{minutes:.0f} min ({minutes * 60 / a.steps:.2f} s/step, peak {torch.cuda.max_memory_allocated() / 1e9:.0f} GB)", flush=True)


def from_config(a):
    """--config ID: the recorded settings of the paper's teacher pairs (configs/pairs.json) or adapter (adapters.json)."""
    if a.mode == "pairs":
        c = json.loads((ROOT / "configs" / "pairs.json").read_text())[a.config]
        a.family, a.guidance, a.classes, a.per_class, a.prompts = c["family"], c["guidance"], c["classes"], c["per_class"], c["prompts"]
        a.grid = c["grid"] if c["family"] == "klein" else a.grid
        a.nshards, a.out = a.nshards or c["shards"], str(WORK / "pairs" / a.config)
    else:
        c = json.loads((ROOT / "configs" / "adapters.json").read_text())[a.config]
        for k in ("family", "target", "rank", "alpha", "steps", "bs", "lr", "seed", "full", "prompts"):
            setattr(a, k, c[k])
        pairs = json.loads((ROOT / "configs" / "pairs.json").read_text())[c["pairs"]]
        a.classes, a.pairs, a.out = pairs["classes"], str(WORK / "pairs" / c["pairs"]), str(WORK / "adapters" / f"{a.config}.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pairs", "train"])
    ap.add_argument("--config", default=None, help="teacher pairs (pairs mode) or an adapter (train mode) of the paper, by id")
    ap.add_argument("--classes", default=None, help="a class list (in100, ...) or a JSON file")
    ap.add_argument("--out", default=None)
    ap.add_argument("--per_class", type=int, default=160)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=0, help="shards of the pairs (default 1; with --config, the recorded number)")
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--family", default="klein", choices=["klein", "sd", *MMDIT])
    ap.add_argument("--grid", default="v1", choices=list(GRIDS), help="pairs: klein's teacher grid (v1: the paper's adapters)")
    ap.add_argument("--target", default="7175", choices=[*STATE_AT, *SD_STATE_AT, *(n for f in MMDIT.values() for n in f["anchors"])])
    ap.add_argument("--guidance", type=float, default=1.0, help="pairs: the teacher's guidance w (1: unguided)")
    ap.add_argument("--full", action="store_true", help="train: every weight instead of low-rank updates (use --lr 1e-5; klein, sd)")
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=0.0, help="gradient-norm clipping (0: none)")
    ap.add_argument("--schedule", default="cosine", choices=["cosine", "constant"])
    ap.add_argument("--prompts", default=None, help="a prompt file (data/prompts) instead of the class templates (pairs and train)")
    ap.add_argument("--anchors", default=None, help="pairs: only these targets, e.g. 958 (the teacher stops at the last)")
    ap.add_argument("--first", type=int, default=0, help="train: only the pairs of the first N prompt groups (0: all)")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.config:
        from_config(a)
    assert a.classes and a.out, "give --config, or --classes and --out"
    a.nshards = a.nshards or 1
    (pairs if a.mode == "pairs" else train)(a)


if __name__ == "__main__":
    main()
