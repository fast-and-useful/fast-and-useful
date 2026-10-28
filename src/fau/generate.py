"""Class-conditional training sets from text-to-image generators, with per-step control of the denoiser and guidance.

Generators come in pairs that differ by one post-training step:
  distillation   sd21 -> sd_turbo, sdxl -> sdxl_turbo, klein_base -> klein, zimage -> zimage_turbo,
                 sd35_medium -> dpdmd_sd35m (DP-DMD's released student); SDXL also as sdxl_dmd2, sdxl_lightning, sdxl_lcm
  preference RL  flux_dev -> srpo
  (sd15 and flux_schnell complete the metric study)
Each runs at its standard settings (steps, guidance, resolution) unless --guidance / --steps override them.

Stage plans (--stages) run the base model's own schedule (its steps, its sampler) and choose, for every step, which
denoiser predicts and with what effective guidance. "other" is the partner model (--other), "base" the pipeline's own:
  --model klein_base --other klein --stages other:1:0-10,base:2:10-50
lets the distilled model draft the first 10 of 50 steps without guidance and the base finish at guidance 2;
  --model klein --other klein_base --stages other:1:0-2,base:1:2-4
runs the distilled model's 4-step schedule with the unguided base drawing the first two steps (base-first).
The adapted sampler is a stage plan too: the distilled model with its first-step adapter as the partner,
  --model klein --other klein_ad:ad2 --sigmas '4[0:1],4[3:]' --stages other:1:0-1,base:1:1-2
jumps from noise 1 to the target level 0.7175 with the adapter, then finishes with the released model's last step.
The pipeline runs with a nominal guidance g0 (the largest in the plan); a step at effective guidance g returns, as its
unconditional prediction, n' = (neg (1 - g) + pos (g - g0)) / (1 - g0), so the pipeline's own n' + g0 (pos - n') equals
neg + g (pos - neg); at g = 1 only the conditional pass runs.

Prompts are fixed photo templates over the class name, cycled per image (template j mod 8), or with --captions one
description per (class, index) from a prompt file (data/prompts); image (class k, index j) uses the seed
1,000,003 k + j + seed_offset in every generator, so sets pair up image by image. Images are stored as JPEG (quality
90) with the short side at 256, one pack per shard (packs.py), each record carrying the denoiser evaluations (nfe) and
H100 seconds it cost; <set>/gen.json records the settings and the schedule that ran.

Usage: python -m fau.generate --set in100/kd_ad2 [--shard i --nshards n]      # a set of the paper (configs/sets.json)
       python -m fau.generate --model klein --classes in100 --per_class 500 --out DIR [--shard i --nshards n]
"""
import argparse, contextlib, json, os, time

import numpy as np
import torch

from fau import ROOT, WORK
from fau.hub import snapshot
from fau.packs import PackWriter, encode
from fau.prompts import class_list, load_prompts

# SRPO's transformer in bfloat16, converted once by `python -m fau.srpo` (tencent/SRPO ships one 47.6 GB float32 file)
SRPO_DIR = os.environ.get("FAU_SRPO") or str(WORK / "models" / "srpo-flux1-dev-bf16")
TEMPLATES = ["a photo of a {}.", "a close-up photo of a {}.", "a photo of the {} in its natural setting.", "a bright photo of a {}.",
             "a photo of a small {}.", "a photo of a large {}.", "a cropped photo of a {}.", "a good photo of a {}."]
# ImageNet names whose first synonym is shared by another class or is vague as a prompt (in the spirit of CLIP's curated
# ImageNet names); every other class uses its first synonym
DISAMBIG = {"n02012849": "crane bird", "n03126707": "construction crane", "n03710637": "maillot swimsuit",
            "n03710721": "tank suit swimsuit", "n02113186": "Cardigan Welsh corgi dog", "n02963159": "cardigan sweater",
            "n02869837": "poke bonnet", "n04517823": "vacuum cleaner", "n13037406": "gyromitra mushroom", "n02105505": "komondor dog",
            "n02788148": "banister", "n13040303": "stinkhorn mushroom", "n03594734": "jeans", "n02877765": "bottle cap",
            "n02087046": "toy terrier dog", "n04592741": "airplane wing", "n03947888": "pirate ship", "n03017168": "chime bell",
            "n02113978": "Mexican hairless dog", "n02089867": "Walker hound dog", "n02231487": "walking stick insect",
            "n03032252": "movie theater", "n02104029": "kuvasz dog", "n03492542": "hard disk drive", "n04067472": "fishing reel",
            "n03930630": "pickup truck", "n03584829": "clothes iron", "n02123045": "tabby cat", "n04336792": "medical stretcher",
            "n02108089": "boxer dog", "n03424325": "gas mask", "n02086910": "papillon dog", "n02091831": "Saluki dog",
            "n02100583": "vizsla dog", "n07714571": "cabbage", "n03379051": "American football helmet",
            "n02116738": "African wild dog", "n04493381": "tub",
            # the 200-class protocol of Adamkiewicz et al. (2026), classes outside ImageNet-100
            "n02892201": "brass memorial plaque", "n02190166": "fly insect", "n03014705": "wooden chest", "n03854065": "pipe organ", "n03976657": "tall pole", "n03804744": "metal nail", "n07579787": "plate of food", "n04456115": "flaming torch", "n04399382": "teddy bear", "n04417672": "thatched roof", "n02814860": "lighthouse", "n02395406": "hog pig", "n01945685": "slug", "n02999410": "metal chain", "n02795169": "wooden barrel", "n09193705": "alpine mountain", "n07875152": "pot pie", "n02236044": "praying mantis", "n03250847": "drumstick for a drum", "n04008634": "missile", "n02509815": "red panda", "n03599486": "rickshaw", "n02814533": "station wagon car", "n02977058": "ATM cash machine", "n03089624": "candy store", "n07615774": "popsicle", "n03983396": "soda bottle", "n03902125": "payphone", "n03393912": "railway freight car", "n02793495": "barn", "n03160309": "dam", "n02403003": "ox", "n03617480": "kimono", "n04133789": "sandal", "n04596742": "wok", "n02699494": "altar",
            # the 1,000-class pilot: classes outside both lists above whose name collides with one of them
            "n03773504": "guided missile"}


# --guidance is always w in the standard convention neg + w (pos - neg). Z-Image's pipeline computes pos + g (pos - neg),
# so its native g = w - 1 (its default g = 4 is w = 5). FLUX.1's guidance is a distilled embedding, passed as is.
NATIVE_OFFSET = {"zimage": 1.0, "zimage_turbo": 1.0}


def class_name(wnid, full):
    return DISAMBIG.get(wnid, full.split(",")[0].strip())


# SDXL distillations run on SDXL base's pipeline: name -> (diffusers scheduler class, call kwargs at their recommended settings)
SDXL_DISTILLED = {"sdxl_dmd2": ("LCMScheduler", dict(num_inference_steps=4, guidance_scale=0.0, timesteps=[999, 749, 499, 249])),
                  "sdxl_lightning": ("EulerDiscreteScheduler", dict(num_inference_steps=4, guidance_scale=0.0)),
                  "sdxl_lcm": ("LCMScheduler", dict(num_inference_steps=4, guidance_scale=8.0))}

# The weight files each loader reads, as (hub repo or local directory, globs relative to it), for warm(). DENOISER: the
# files of a model's denoiser alone (a stage partner).
SD_FP16 = ["text_encoder/*.fp16.safetensors", "text_encoder_2/*.fp16.safetensors", "unet/*.fp16.safetensors", "vae/*.fp16.safetensors"]
FLOW = ["text_encoder/*.safetensors", "text_encoder_2/*.safetensors", "transformer/*.safetensors", "vae/*.safetensors"]
SD3 = ["text_encoder/model.safetensors", "text_encoder_2/model.safetensors", "text_encoder_3/model-*.safetensors",
       "transformer/*.safetensors", "vae/*.safetensors"]
SDXL_REPO = "stabilityai/stable-diffusion-xl-base-1.0"
SD35 = {"sd35_large": "stabilityai/stable-diffusion-3.5-large", "sd35_medium": "stabilityai/stable-diffusion-3.5-medium"}
# DP-DMD's released SD3.5-Medium student (a full transformer state dict; Wu et al., 2026), run on SD3.5-Medium's
# pipeline; SD3.5-Medium's text encoders are SD3.5-Large's files (same sha256), loaded from SD3.5-Large's snapshot
DPDMD = ("TianheWu/dpdmd", "DPDMD-SD35M-4NFE-natural.pt")
SCHED = {"sd35_medium": SD35["sd35_medium"], "dpdmd_sd35m": SD35["sd35_medium"]}   # the scheduler each SD3.5 model runs on
ZIMAGE = {"zimage": "Tongyi-MAI/Z-Image", "zimage_turbo": "Tongyi-MAI/Z-Image-Turbo"}
SDA = ("F16/z-image-turbo-sda", "zit_sda_v1.safetensors")   # the public SDA adapter for Z-Image-Turbo (a full-rank LoKr)
# partners with a first-step adapter (adapter.py) folded into the distilled denoiser it was trained on
AD_BASE = {"klein_ad": "klein", "sdt_ad": "sd_turbo", "zt_ad": "zimage_turbo", "sd35m_ad": "dpdmd_sd35m"}
SDXL_TE = ["text_encoder/model.safetensors", "text_encoder_2/model.safetensors"]
VAE_FIX = ("madebyollin/sdxl-vae-fp16-fix", ["diffusion_pytorch_model.safetensors"])
WEIGHTS = {"sd15": [("stable-diffusion-v1-5/stable-diffusion-v1-5", SD_FP16)],
           "sd21": [("sd2-community/stable-diffusion-2-1-base", SD_FP16)],
           "sd_turbo": [("stabilityai/sd-turbo", SD_FP16)],
           "sdxl": [(SDXL_REPO, SDXL_TE + ["unet/diffusion_pytorch_model.safetensors"]), VAE_FIX],
           "sdxl_turbo": [("stabilityai/sdxl-turbo", SD_FP16)],
           "sdxl_dmd2": [(SDXL_REPO, SDXL_TE), VAE_FIX, ("tianweiy/DMD2", ["dmd2_sdxl_4step_unet_fp16.bin"])],
           "sdxl_lightning": [(SDXL_REPO, SDXL_TE), VAE_FIX, ("ByteDance/SDXL-Lightning", ["sdxl_lightning_4step_unet.safetensors"])],
           "sdxl_lcm": [(SDXL_REPO, SDXL_TE), VAE_FIX, ("latent-consistency/lcm-sdxl", ["diffusion_pytorch_model.fp16.safetensors"])],
           "flux_schnell": [("black-forest-labs/FLUX.1-schnell", FLOW)],
           "flux_dev": [("black-forest-labs/FLUX.1-dev", FLOW)],
           "srpo": [("black-forest-labs/FLUX.1-dev", [g for g in FLOW if not g.startswith("transformer")]), (SRPO_DIR, ["*.safetensors"])],
           "klein": [("black-forest-labs/FLUX.2-klein-4B", FLOW)],
           "klein_base": [("black-forest-labs/FLUX.2-klein-base-4B", FLOW)],
           "zimage": [("Tongyi-MAI/Z-Image", FLOW)],
           "zimage_turbo": [("Tongyi-MAI/Z-Image-Turbo", FLOW)],
           "sd35_medium": [(SD35["sd35_medium"], ["transformer/*.safetensors", "vae/*.safetensors"]), (SD35["sd35_large"], SD3[:3])],
           "dpdmd_sd35m": [(SD35["sd35_medium"], ["vae/*.safetensors"]), (SD35["sd35_large"], SD3[:3]), (DPDMD[0], [DPDMD[1]])]}
DENOISER = {"klein": [("black-forest-labs/FLUX.2-klein-4B", ["transformer/*.safetensors"])],
            "klein_base": [("black-forest-labs/FLUX.2-klein-base-4B", ["transformer/*.safetensors"])],
            "sd_turbo": [("stabilityai/sd-turbo", ["unet/*.fp16.safetensors"])],
            "sd21": [("sd2-community/stable-diffusion-2-1-base", ["unet/*.fp16.safetensors"])],
            "sdxl": [(SDXL_REPO, ["unet/diffusion_pytorch_model.safetensors"])],
            "sdxl_turbo": [("stabilityai/sdxl-turbo", ["unet/*.fp16.safetensors"])],
            "sdxl_dmd2": WEIGHTS["sdxl_dmd2"][-1:], "sdxl_lightning": WEIGHTS["sdxl_lightning"][-1:],
            "zimage": [("Tongyi-MAI/Z-Image", ["transformer/*.safetensors"])],
            "zimage_turbo": [("Tongyi-MAI/Z-Image-Turbo", ["transformer/*.safetensors"])],
            "sd35_medium": [(SD35["sd35_medium"], ["transformer/*.safetensors"])], "dpdmd_sd35m": [(DPDMD[0], [DPDMD[1]])]}


def weight_files(sources):
    import glob
    return sorted({f for src, globs in sources for g in globs for f in glob.glob(os.path.join(snapshot(src), g))})


def warm(sources, label):
    """Reads the weight files a loader is about to map, once and sequentially, so the maps hit the page cache: on network
    file systems (Lustre) memory-mapped reads can page in at ~14 MB/s, a sequential read at hundreds of MB/s."""
    t0, n, files = time.time(), 0, weight_files(sources)
    for f in files:
        with open(f, "rb") as fh:
            while chunk := fh.read(1 << 26):
                n += len(chunk)
    print(f"warmed {label}: {len(files)} files, {n / 1e9:.1f} GB in {time.time() - t0:.0f} s", flush=True)


def sdxl_unet(name):
    """The UNet of an SDXL distillation: DMD2 (distribution matching + GAN, 4 steps), SDXL-Lightning (progressive
    adversarial, 4 steps), LCM-SDXL (consistency, 4 steps, guidance as an input), or SDXL-Turbo (ADD, 1 step)."""
    from diffusers import UNet2DConditionModel
    f16 = torch.float16
    if name == "sdxl_lcm":
        return UNet2DConditionModel.from_pretrained(snapshot("latent-consistency/lcm-sdxl"), variant="fp16", torch_dtype=f16)
    if name == "sdxl_turbo":
        return UNet2DConditionModel.from_pretrained(snapshot("stabilityai/sdxl-turbo"), subfolder="unet", variant="fp16", torch_dtype=f16)
    if name == "sdxl":
        return UNet2DConditionModel.from_pretrained(snapshot("stabilityai/stable-diffusion-xl-base-1.0"), subfolder="unet", torch_dtype=f16)
    unet = UNet2DConditionModel.from_config(UNet2DConditionModel.load_config(snapshot("stabilityai/stable-diffusion-xl-base-1.0"), subfolder="unet"))
    if name == "sdxl_dmd2":
        sd = torch.load(os.path.join(snapshot("tianweiy/DMD2"), "dmd2_sdxl_4step_unet_fp16.bin"), map_location="cpu", weights_only=True)
    else:
        from safetensors.torch import load_file
        sd = load_file(os.path.join(snapshot("ByteDance/SDXL-Lightning"), "sdxl_lightning_4step_unet.safetensors"))
    unet.load_state_dict(sd)
    return unet.to(f16)


def load(name):
    """name -> (pipeline, call kwargs, resolution, batch, family); family: 'cc' (separate conditional / unconditional calls
    in cache contexts), 'unet' (one batched call), or 'flux1' (guidance-distilled FLUX.1, no stage control)."""
    from diffusers import (DPMSolverMultistepScheduler, FluxPipeline, StableDiffusionPipeline, StableDiffusionXLPipeline,
                           AutoPipelineForText2Image)
    f16, bf16 = torch.float16, torch.bfloat16
    warm(WEIGHTS.get(name, []), name)
    if name == "sd15":
        p = StableDiffusionPipeline.from_pretrained(snapshot("stable-diffusion-v1-5/stable-diffusion-v1-5"), variant="fp16", torch_dtype=f16,
                                                    safety_checker=None, requires_safety_checker=False)
        p.scheduler = DPMSolverMultistepScheduler.from_config(p.scheduler.config)
        return p, dict(num_inference_steps=30, guidance_scale=7.5), 512, 16, "unet"
    if name == "sd21":
        p = StableDiffusionPipeline.from_pretrained(snapshot("sd2-community/stable-diffusion-2-1-base"), variant="fp16", torch_dtype=f16,
                                                    safety_checker=None, requires_safety_checker=False)
        p.scheduler = DPMSolverMultistepScheduler.from_config(p.scheduler.config)
        return p, dict(num_inference_steps=30, guidance_scale=7.5), 512, 16, "unet"
    if name == "sd_turbo":
        p = AutoPipelineForText2Image.from_pretrained(snapshot("stabilityai/sd-turbo"), variant="fp16", torch_dtype=f16)
        return p, dict(num_inference_steps=1, guidance_scale=0.0), 512, 32, "unet"
    if name == "sdxl":
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(snapshot("madebyollin/sdxl-vae-fp16-fix"), torch_dtype=f16)
        p = StableDiffusionXLPipeline.from_pretrained(snapshot("stabilityai/stable-diffusion-xl-base-1.0"), vae=vae, torch_dtype=f16)
        p.scheduler = DPMSolverMultistepScheduler.from_config(p.scheduler.config)
        return p, dict(num_inference_steps=30, guidance_scale=7.0), 1024, 4, "unet"
    if name == "sdxl_turbo":
        p = AutoPipelineForText2Image.from_pretrained(snapshot("stabilityai/sdxl-turbo"), variant="fp16", torch_dtype=f16)
        return p, dict(num_inference_steps=1, guidance_scale=0.0), 512, 32, "unet"
    if name in SDXL_DISTILLED:   # SDXL distilled other ways, on SDXL base's pipeline (text encoders, fp16-fix VAE)
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(snapshot("madebyollin/sdxl-vae-fp16-fix"), torch_dtype=f16)
        p = StableDiffusionXLPipeline.from_pretrained(snapshot("stabilityai/stable-diffusion-xl-base-1.0"), vae=vae, unet=sdxl_unet(name),
                                                      torch_dtype=f16)
        sched, kw = SDXL_DISTILLED[name]
        p.scheduler = getattr(__import__("diffusers"), sched).from_config(
            p.scheduler.config, **({"timestep_spacing": "trailing"} if name == "sdxl_lightning" else {}))
        return p, dict(kw), 1024, 8, "unet"
    if name in ("flux_schnell", "flux_dev", "srpo"):
        repo = snapshot("black-forest-labs/FLUX.1-schnell" if name == "flux_schnell" else "black-forest-labs/FLUX.1-dev")
        if name == "srpo":   # FLUX.1-dev after SRPO preference tuning, converted to bfloat16 once by `python -m fau.srpo`
            from diffusers import FluxTransformer2DModel
            p = FluxPipeline.from_pretrained(repo, transformer=FluxTransformer2DModel.from_pretrained(SRPO_DIR, torch_dtype=bf16),
                                             torch_dtype=bf16)
        else:
            p = FluxPipeline.from_pretrained(repo, torch_dtype=bf16)
        kw = dict(num_inference_steps=4, guidance_scale=0.0, max_sequence_length=256) if name == "flux_schnell" else \
            dict(num_inference_steps=28, guidance_scale=3.5, max_sequence_length=512)
        return p, kw, 512, 8, "flux1"
    if name in ("klein", "klein_base"):
        from diffusers import Flux2KleinPipeline
        p = Flux2KleinPipeline.from_pretrained(snapshot("black-forest-labs/FLUX.2-klein-4B" if name == "klein" else
                                                        "black-forest-labs/FLUX.2-klein-base-4B"), torch_dtype=bf16)
        kw = dict(num_inference_steps=4) if name == "klein" else dict(num_inference_steps=50, guidance_scale=4.0)
        return p, kw, 512, 8, "cc"
    if name in ("sd35_medium", "dpdmd_sd35m"):   # 28 steps at 3.5 (DP-DMD's teacher guidance) / DP-DMD's own setting
        from diffusers import StableDiffusion3Pipeline   # (4 steps, guidance 1), 512 px
        from transformers import CLIPTextModelWithProjection, T5EncoderModel
        large = snapshot(SD35["sd35_large"])
        te = dict(text_encoder=CLIPTextModelWithProjection.from_pretrained(large, subfolder="text_encoder", torch_dtype=bf16),
                  text_encoder_2=CLIPTextModelWithProjection.from_pretrained(large, subfolder="text_encoder_2", torch_dtype=bf16),
                  text_encoder_3=T5EncoderModel.from_pretrained(large, subfolder="text_encoder_3", torch_dtype=bf16))
        p = StableDiffusion3Pipeline.from_pretrained(snapshot(SD35["sd35_medium"]), torch_dtype=bf16, **te)
        if name == "dpdmd_sd35m":
            p.transformer.load_state_dict(torch.load(os.path.join(snapshot(DPDMD[0]), DPDMD[1]), map_location="cpu"), strict=True)
        kw = dict(num_inference_steps=28, guidance_scale=3.5) if name == "sd35_medium" else dict(num_inference_steps=4, guidance_scale=1.0)
        return p, kw, 512, 8, "unet"
    if name in ZIMAGE:
        from diffusers import ZImagePipeline
        p = ZImagePipeline.from_pretrained(snapshot(ZIMAGE[name]), torch_dtype=bf16)
        kw = dict(num_inference_steps=9, guidance_scale=0.0) if name == "zimage_turbo" else dict(num_inference_steps=50, guidance_scale=4.0)
        return p, kw, 512, 8, "zimage"
    raise ValueError(name)


class LoKrLinear(torch.nn.Module):
    """A linear layer plus a full-rank LoKr kept beside it (the public SDA adapter for Z-Image-Turbo, ai-toolkit's
    format): y = W x + kron(w1, w2) x, computed as w1 X w2^T on x reshaped to (b, d), in float32. Unmerged, because the
    update (mean |dW| 4e-4 against |W| 0.13) is close to bfloat16's resolution: merged, much of it would round away. The
    scale alpha / dim is 1 (full rank: alpha equals the dimension)."""

    def __init__(self, base, w1, w2):
        super().__init__()
        self.base = base
        self.register_buffer("w1", w1.float(), persistent=False)
        self.register_buffer("w2", w2.float(), persistent=False)

    def forward(self, x):
        (a, b), (c, d) = self.w1.shape, self.w2.shape
        y = torch.matmul(torch.matmul(self.w1, x.float().reshape(*x.shape[:-1], b, d)), self.w2.T)
        return self.base(x) + y.reshape(*x.shape[:-1], a * c).to(x.dtype)


def lokr(net, path):
    """Wraps every module a LoKr file names ('diffusion_model.<diffusers module>.lokr_w1/lokr_w2/alpha') in LoKrLinear."""
    from safetensors.torch import load_file
    sd = load_file(path)
    mods = sorted({k.rsplit(".", 1)[0] for k in sd})
    assert all(float(sd[f"{m}.alpha"]) > 1e9 for m in mods), "a LoKr of finite rank: its scale alpha / dim is not 1"
    named = dict(net.named_modules())
    for m in mods:
        name = m.removeprefix("diffusion_model.")
        parent, _, child = name.rpartition(".")
        lin, w1, w2 = named[name], sd[f"{m}.lokr_w1"], sd[f"{m}.lokr_w2"]
        assert (w1.shape[0] * w2.shape[0], w1.shape[1] * w2.shape[1]) == tuple(lin.weight.shape), (m, w1.shape, w2.shape)
        setattr(named[parent], child, LoKrLinear(lin, w1, w2).to(lin.weight.device))
    print(f"LoKr: {len(mods)} modules from {path}", flush=True)
    return net


def adapter_path(ref):
    """An adapter id (configs/adapters.json; $FAU_WORK/adapters/<id>.pt) or a path to an adapter file."""
    return ref if os.path.isfile(ref) else str(WORK / "adapters" / f"{ref}.pt")


def partner(name):
    """The denoiser of a model whose pipeline matches the base's (same latents, text encoder and VAE). 'klein_ad:ID',
    'sdt_ad:ID', 'zt_ad:ID', 'sd35m_ad:ID': distilled klein, SD-Turbo, Z-Image-Turbo or DP-DMD's student with a first-step
    adapter (adapter.py) folded into its weights; 'zt_lokr:sda': Z-Image-Turbo with the public SDA adapter."""
    kind, _, ref = name.partition(":")
    if kind == "zt_lokr":   # Z-Image-Turbo with a LoKr beside its linear layers (SDA)
        return lokr(partner("zimage_turbo"), os.path.join(snapshot(SDA[0]), SDA[1]) if ref == "sda" else ref)
    if kind in AD_BASE:
        from fau.adapter import merge
        return merge(partner(AD_BASE[kind]), adapter_path(ref))
    f16, bf16 = torch.float16, torch.bfloat16
    warm(DENOISER.get(name, []), f"{name} denoiser")
    if name in ("klein", "klein_base"):
        from diffusers import Flux2Transformer2DModel
        return Flux2Transformer2DModel.from_pretrained(snapshot("black-forest-labs/FLUX.2-klein-4B" if name == "klein" else
                                                                "black-forest-labs/FLUX.2-klein-base-4B"), subfolder="transformer",
                                                       torch_dtype=bf16)
    if name in ("sd_turbo", "sd21"):
        from diffusers import UNet2DConditionModel
        return UNet2DConditionModel.from_pretrained(snapshot("stabilityai/sd-turbo" if name == "sd_turbo" else
                                                             "sd2-community/stable-diffusion-2-1-base"),
                                                    subfolder="unet", variant="fp16", torch_dtype=f16)
    if name in ("sdxl", "sdxl_turbo", "sdxl_dmd2", "sdxl_lightning"):   # not LCM: its guidance input needs the LCM pipeline path
        return sdxl_unet(name)
    # SD3.5 and Z-Image from the cached snapshot's directory: by repo id the Z-Image loader queries the hub even in offline mode
    if name == "dpdmd_sd35m":   # SD3.5-Medium's transformer with DP-DMD's weights
        from diffusers import SD3Transformer2DModel
        net = SD3Transformer2DModel.from_pretrained(os.path.join(snapshot(SD35["sd35_medium"]), "transformer"), torch_dtype=bf16)
        net.load_state_dict(torch.load(os.path.join(snapshot(DPDMD[0]), DPDMD[1]), map_location="cpu"), strict=True)
        return net
    if name == "sd35_medium":
        from diffusers import SD3Transformer2DModel
        return SD3Transformer2DModel.from_pretrained(os.path.join(snapshot(SD35[name]), "transformer"), torch_dtype=bf16)
    if name in ZIMAGE:
        from diffusers import ZImageTransformer2DModel
        return ZImageTransformer2DModel.from_pretrained(os.path.join(snapshot(ZIMAGE[name]), "transformer"), torch_dtype=bf16)
    raise ValueError(f"no partner loader for {name}")


def parse_stages(spec, steps):
    """'other:1:0-10,base:2:10-50' -> [(model, guidance)] for every step; the ranges must tile 0..steps."""
    plan = [None] * steps
    for part in spec.split(","):
        who, g, rng = part.split(":")
        lo, hi = (int(v) for v in rng.split("-"))
        for i in range(lo, hi):
            assert plan[i] is None, f"step {i} given twice"
            plan[i] = (who, float(g))
    assert all(p is not None for p in plan), f"stages leave steps {[i for i, p in enumerate(plan) if p is None]} unassigned"
    return plan


def uncond_for(pos, neg, g, g0):
    """The unconditional prediction that makes the pipeline's neg' + g0 (pos - neg') equal neg + g (pos - neg). Only plans
    with two different guidance values above 1 need it; the pipeline's own arithmetic then amplifies rounding by |1 - g0|."""
    return ((neg.float() * (1 - g) + pos.float() * (g - g0)) / (1 - g0)).to(neg.dtype)


class Stager(torch.nn.Module):
    """Stands in for the pipeline's denoiser and runs the plan: step i uses models[plan[i][0]] at guidance plan[i][1]."""

    def __init__(self, models, plan, g0, family):
        super().__init__()
        self.models = torch.nn.ModuleDict(models)
        self.plan, self.g0, self.family = plan, g0, family
        self.i, self.branch, self.pos = 0, "cond", None
        self.repel = None   # (strength, first step, end step): see repelled()
        self.cads, self.seed = None, 0   # (noise scale, t1, t2): see cads_cond(); seed set per batch

    @property
    def dtype(self):
        return self.models["base"].dtype

    @property
    def config(self):
        return self.models["base"].config

    def __getattr__(self, name):   # device, attention processors and the like come from the base denoiser
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._modules["models"]["base"], name)

    @contextlib.contextmanager
    def cache_context(self, name):
        self.branch = name
        with self.models[self.plan[self.i][0]].cache_context(name):
            yield

    def repelled(self, v, x, sigma):
        """Within-batch repulsion of the predicted clean latents (a batch holds images of one class; generate.py orders its
        jobs that way): x0 = x - sigma v on FLUX.2's token grid, pooled to 4 x 4 cells (the layout), each image pushed
        away from the others along the RBF-kernel gradient (median bandwidth, as in particle guidance), the push scaled to
        `strength` times the batch's spread around its mean, then turned back into a velocity."""
        lam, B = self.repel[0], x.shape[0]
        if B < 2:
            return v
        n, C = x.shape[1], x.shape[2]
        g = int(round(n ** 0.5))
        s = sigma.view(-1, 1, 1).float()
        x0 = x.float() - s * v[:, :n].float()
        cells = torch.nn.functional.adaptive_avg_pool2d(x0.view(B, g, g, C).permute(0, 3, 1, 2), 4)   # B, C, 4, 4
        a = cells.flatten(1)
        d2 = torch.cdist(a, a) ** 2
        h = d2[d2 > 0].median() / np.log(B + 1)
        k = torch.exp(-d2 / h)
        r = (k.unsqueeze(-1) * (a.unsqueeze(1) - a.unsqueeze(0))).sum(1).view(B, C, 4, 4)
        r = torch.nn.functional.interpolate(r, size=(g, g), mode="nearest").permute(0, 2, 3, 1).reshape(B, n, C)
        spread = (x0 - x0.mean(0, keepdim=True)).pow(2).mean().sqrt()
        x0 = x0 + lam * spread * r / (r.pow(2).mean().sqrt() + 1e-8)
        out = v.clone()
        out[:, :n] = ((x.float() - x0) / s).to(v.dtype)
        return out

    def cads_cond(self, y, t):
        """CADS (Sadat et al., 2024): the condition y at noise level t in [0, 1] (1 = pure noise) becomes
        sqrt(gamma) y + s sqrt(1 - gamma) n, gamma = 1 below t1, 0 above t2, linear between, then is rescaled to y's mean
        and spread (psi = 1). Fresh noise per step, seeded per batch and step."""
        scale, t1, t2 = self.cads
        gamma = min(1.0, max(0.0, (t2 - t) / (t2 - t1)))
        if gamma >= 1.0:
            return y
        gen = torch.Generator(device=y.device).manual_seed(self.seed * 1000 + self.i)
        n = torch.randn(y.shape, generator=gen, device=y.device, dtype=torch.float32)
        yf = y.float()
        z = gamma ** 0.5 * yf + scale * (1 - gamma) ** 0.5 * n
        dims = tuple(range(1, y.ndim))
        z = (z - z.mean(dims, keepdim=True)) / (z.std(dims, keepdim=True) + 1e-8) * yf.std(dims, keepdim=True) + yf.mean(dims, keepdim=True)
        return z.to(y.dtype)

    def forward(self, *args, **kw):
        who, g = self.plan[self.i]
        m = self.models[who]
        if self.cads:   # perturb the conditional branch only
            if self.family == "cc" and self.branch == "cond":
                kw = dict(kw, encoder_hidden_states=self.cads_cond(kw["encoder_hidden_states"], float(kw["timestep"].flatten()[0])))
            elif self.family == "unet":
                t, y = float((args[1] if len(args) > 1 else kw["timestep"]).flatten()[0]) / 1000, kw["encoder_hidden_states"]
                half = y.shape[0] // 2
                kw = dict(kw, encoder_hidden_states=torch.cat([y[:half], self.cads_cond(y[half:], t)]))
        if self.family == "route":   # every step unguided: one call per step, sent to that step's model
            return m(*args, **kw)
        if self.family == "cc":
            if self.branch == "cond":
                out = m(*args, **kw)
                if self.repel and self.repel[1] <= self.i < self.repel[2]:
                    out = (self.repelled(out[0], kw["hidden_states"], kw["timestep"]),) + tuple(out[1:])
                self.pos = out[0]
                return out
            if g <= 1:
                return (self.pos,)
            neg = m(*args, **kw)[0]
            return (neg if g == self.g0 else uncond_for(self.pos, neg, g, self.g0),)
        # unet (and SD3's transformer): one batched call over [unconditional; conditional]
        B = (args[0] if args else kw["sample"] if "sample" in kw else kw["hidden_states"]).shape[0] // 2
        if g <= 1:   # the conditional half only
            half = lambda v: v[B:] if torch.is_tensor(v) and v.ndim and v.shape[0] == 2 * B else v
            pos = m(*[half(v) for v in args], **{k: {kk: half(vv) for kk, vv in v.items()} if isinstance(v, dict) else half(v)
                                                  for k, v in kw.items()})[0]
            return (torch.cat([pos, pos]),)
        neg, pos = m(*args, **kw)[0].chunk(2)
        return (torch.cat([neg if g == self.g0 else uncond_for(pos, neg, g, self.g0), pos]),)


def advance(pipe, i, t, kw):
    """callback_on_step_end: the next call belongs to step i + 1."""
    denoiser(pipe).i = i + 1
    return kw


def denoiser(pipe):
    return pipe.transformer if getattr(pipe, "transformer", None) is not None else pipe.unet


class Counter:
    """Counts denoiser evaluations: every forward call adds its batch size (a batched [unconditional; conditional] call
    counts twice per image)."""

    def __init__(self, *modules):
        self.n = 0
        for m in modules:
            m.register_forward_pre_hook(self.hook, with_kwargs=True)

    def hook(self, module, args, kwargs):
        x = args[0] if args else next(kwargs[k] for k in ("hidden_states", "sample", "x") if k in kwargs)
        self.n += len(x) if isinstance(x, (list, tuple)) else x.shape[0]


def native_sigmas(n, res, model="klein"):
    """A model's own n-step noise levels at resolution res. FLUX.2-klein (base or distilled): linear sigmas under its
    resolution- and step-dependent exponential shift (the pipeline's compute_empirical_mu; 16 px per latent token).
    Z-Image and SD3.5: the model's scheduler set as its pipeline sets it (Z-Image passes linear sigmas 1 .. 1/n, SD3 the
    step count); both shift statically (dynamic shifting is off in their configs, so res plays no part): Z-Image 6,
    Z-Image-Turbo 3, SD3.5-Medium 3."""
    import numpy as np
    if model in ("klein", "klein_base"):
        from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu
        e = np.exp(compute_empirical_mu((res // 16) ** 2, n))
        s = np.linspace(1.0, 1 / n, n)
        return [float(v) for v in e / (e + (1 / s - 1))]
    from diffusers import FlowMatchEulerDiscreteScheduler
    sch = FlowMatchEulerDiscreteScheduler.from_pretrained(snapshot({**ZIMAGE, **SCHED}[model]), subfolder="scheduler")
    assert not sch.config.use_dynamic_shifting, f"{model}: a resolution-dependent shift is not implemented"
    if model in ZIMAGE:
        from diffusers.pipelines.z_image.pipeline_z_image import get_default_z_image_sigmas
        sch.set_timesteps(sigmas=get_default_z_image_sigmas(n))
    else:
        sch.set_timesteps(n)
    return sch.sigmas[:-1].tolist()   # float32 values: used as given, they reproduce the pipeline's own run exactly


def parse_sigmas(spec, res, model="klein"):
    """'4[0:3],50[34:]' -> the native 4-step schedule's first 3 noise levels, then the native 50-step schedule's from index
    34 on (native_sigmas of `model`); a part can name another model's schedule ('zimage:50[0:19]', the Z-Image base's
    levels in a Z-Image-Turbo run); plain numbers are taken as they are. The scheduler appends the final 0."""
    out = []
    for part in spec.split(","):
        if "[" in part:
            head, rng = part.split("[")
            other, _, n = head.rpartition(":")
            a, b = rng.rstrip("]").split(":")
            out += native_sigmas(int(n), res, other or model)[int(a) if a else None:int(b) if b else None]
        else:
            out.append(float(part))
    assert out[0] <= 1 and all(x > y for x, y in zip(out, out[1:])) and out[-1] > 0, f"noise levels must fall from <= 1: {out}"
    return out


def build(model, other=None, stages=None, guidance=None, g0=None, steps=None, sigmas=None, repel=None, cads=None, timesteps=None):
    """The pipeline of `model` on the GPU, with a stage plan installed when given -> (pipe, call kwargs, resolution, batch,
    evaluation counter). g0 overrides the nominal guidance (tests of the rescaling). sigmas (parse_sigmas) replaces the
    schedule by explicit noise levels, for the flow-matching pipelines of FLUX.2-klein, Z-Image and SD3.5; timesteps
    ('999,499,249') by explicit timesteps, for the Euler pipelines of SD UNets (SD-Turbo): stage ranges then index those
    levels."""
    pipe, kw, res, bs, family = load(model)
    if guidance is not None:
        kw = dict(kw, guidance_scale=guidance - NATIVE_OFFSET.get(model, 0.0))
    if steps is not None:
        kw = dict(kw, num_inference_steps=steps)
    if sigmas:
        from diffusers import FlowMatchEulerDiscreteScheduler
        assert model in ("klein", "klein_base", *ZIMAGE, *SCHED), \
            f"explicit noise levels are implemented for FLUX.2-klein, Z-Image and SD3.5, not {model}"
        sig = parse_sigmas(sigmas, res, model)   # used as given: no shift on top
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config, use_dynamic_shifting=False, shift=1.0)
        kw = dict(kw, sigmas=sig, num_inference_steps=len(sig))
    if timesteps:
        assert family == "unet" and type(pipe.scheduler).__name__ == "EulerDiscreteScheduler", \
            f"explicit timesteps are implemented for Euler UNet pipelines (SD-Turbo), not {model}"
        ts = [int(v) for v in timesteps.split(",")]
        assert ts[0] <= 999 and all(x > y for x, y in zip(ts, ts[1:])) and ts[-1] > 0, f"timesteps must fall from <= 999: {ts}"
        kw = dict(kw, timesteps=ts, num_inference_steps=len(ts))
    counted = [denoiser(pipe)]
    if stages:
        plan = parse_stages(stages, kw["num_inference_steps"])
        if family not in ("cc", "unet"):   # any pipeline, as long as no step is guided: the calls are only routed
            assert all(g <= 1 for _, g in plan), f"guided stages are implemented for FLUX.2-klein and SD UNets, not {model}"
            assert (kw.get("guidance_scale") or 0) + NATIVE_OFFSET.get(model, 0.0) <= 1, f"run {model} unguided (--guidance 1)"
            family = "route"
        models = {"base": denoiser(pipe)}
        if any(w == "other" for w, _ in plan):
            models["other"] = partner(other)
            counted.append(models["other"])
        if family != "route":
            g0 = g0 or max(max(g for _, g in plan), 1.5)   # the pipeline must run its guidance branch; g <= 1 steps skip it
        st = Stager(models, plan, g0, family)
        if cads:   # 'scale:t1:t2'
            assert family in ("cc", "unet"), "CADS is implemented for FLUX.2-klein and SD UNets"
            st.cads = tuple(float(v) for v in cads.split(":"))
        if repel:   # 'strength:a-b'
            assert family == "cc", "repulsion is implemented for FLUX.2-klein's token grid"
            lam, rng = repel.split(":")
            lo, hi = (int(v) for v in rng.split("-"))
            st.repel = (float(lam), lo, hi)
        if family != "route" and getattr(pipe.config, "is_distilled", False):   # a distilled pipeline as the base (its
            pipe.register_to_config(is_distilled=False)   # schedule, e.g. 4 steps): run the guidance branch for guided steps
        if getattr(pipe, "transformer", None) is not None:
            pipe.transformer = st
        else:
            pipe.unet = st
        kw = dict(kw, callback_on_step_end=advance, **({} if family == "route" else dict(guidance_scale=g0)))
    count = Counter(*counted)
    pipe.to("cuda"); pipe.set_progress_bar_config(disable=True)
    return pipe, kw, res, bs, count


def from_config(a):
    """--set ID: the arguments recorded for a set of the paper (configs/sets.json); its schedule is checked after it runs."""
    cfg = json.loads((ROOT / "configs" / "sets.json").read_text())[a.set]
    assert cfg["generator"] == "generate", f"{a.set} is made by python -m fau.baselines --set {a.set}"
    for k, v in cfg["args"].items():
        setattr(a, k, v)
    a.classes = cfg["classes"]
    a.out, a.tag = str(WORK / "sets" / a.set.split("/")[0]), a.set.split("/")[1]
    a.nshards = a.nshards or cfg["shards"]
    return cfg


def check_schedule(run, cfg):
    """The schedule that ran must be the one the paper's set ran (noise levels, timesteps, stage plan)."""
    for k in ("sigmas", "timesteps", "plan"):
        want, got = cfg["schedule"].get(k), run.get(k)
        if want is not None and got is not None:
            assert json.loads(json.dumps(got)) == want, f"{k}: ran {got}, the recorded set ran {want}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=None, help="a set of the paper, e.g. in100/kd_ad2 (configs/sets.json): its recorded settings")
    ap.add_argument("--model", default=None)
    ap.add_argument("--classes", default=None, help="a class list: in100, tin200, in1k, food101, cars, pets, or a JSON file")
    ap.add_argument("--per_class", type=int, default=500)
    ap.add_argument("--start", type=int, default=0, help="first image index per class: images start..per_class-1 extend a set "
                                                          "of `start` images per class (same seeds as one full run)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=0, help="shards of the set (default 1; with --set, the recorded number)")
    ap.add_argument("--limit", type=int, default=0, help="first N jobs of the shard only (smoke tests)")
    ap.add_argument("--bs", type=int, default=0, help="override --model's batch size")
    ap.add_argument("--res", type=int, default=0, help="override --model's resolution")
    ap.add_argument("--guidance", type=float, default=None, help="override --model's guidance scale (w in neg + w (pos - neg))")
    ap.add_argument("--steps", type=int, default=None, help="override --model's number of steps (stage ranges refer to these)")
    ap.add_argument("--sigmas", default=None, help="explicit noise levels, e.g. '4[0:3],50[34:]' (parse_sigmas; klein, Z-Image, SD3.5)")
    ap.add_argument("--timesteps", default=None, help="explicit timesteps, e.g. '999,499,249' (Euler UNet pipelines: SD-Turbo)")
    ap.add_argument("--other", default=None, help="the partner model whose denoiser 'other' stages use")
    ap.add_argument("--stages", default=None, help="model:guidance:a-b,... over the base schedule's steps")
    ap.add_argument("--captions", default=None, help="a prompt file (data/prompts): prompt '<class name>, <description>'")
    ap.add_argument("--cads", default=None, help="'scale:t1:t2': CADS condition annealing (Stager.cads_cond; needs --stages)")
    ap.add_argument("--repel", default=None, help="'strength:a-b': within-class repulsion of the predicted layouts in steps a..b-1 "
                                                  "(Stager.repelled; needs --stages)")
    ap.add_argument("--seed_offset", type=int, default=0, help="added to every image's seed 1,000,003 k + j: an independent "
                                                                "draw of the same set (teacher pairs use 600,000 + j)")
    a = ap.parse_args()
    cfg = from_config(a) if a.set else None
    assert a.model and a.classes and a.out, "give --set, or --model, --classes and --out"
    a.nshards = a.nshards or 1
    meta = class_list(a.classes)
    names = [class_name(w, n) for w, n in zip(meta["wnids"], meta["names"])]
    dup = {n for n in names if names.count(n) > 1}
    assert not dup, f"ambiguous class names in this subset: {dup}"
    caps = load_prompts(a.captions)["captions"] if a.captions else None
    assert caps is None or min(len(c) for c in caps) >= a.per_class, "fewer captions than images per class"
    jobs = [(k, j) for k in range(len(names)) for j in range(a.start, a.per_class)][a.shard::a.nshards]
    if a.limit:
        jobs = jobs[:a.limit]
    tag = a.tag or a.model
    prefix = f"{a.out}/{tag}/part{a.shard}"
    if os.path.exists(prefix + ".json"):
        print(f"{prefix} exists, nothing to do"); return
    pipe, kw, res, bs, count = build(a.model, a.other, a.stages, a.guidance, steps=a.steps, sigmas=a.sigmas, repel=a.repel, cads=a.cads,
                                     timesteps=a.timesteps)
    bs, res = a.bs or bs, a.res or res
    steps, plan = kw["num_inference_steps"], parse_stages(a.stages, kw["num_inference_steps"]) if a.stages else None
    run = dict(tag=tag, set=a.set, model=a.model, other=a.other, stages=a.stages, steps=steps, sigma_spec=a.sigmas,
               timestep_spec=a.timesteps, repel=a.repel, cads=a.cads, resolution=res, batch=bs,
               guidance=None if a.stages or "guidance_scale" not in kw else kw["guidance_scale"] + NATIVE_OFFSET.get(a.model, 0.0),
               plan=plan, per_class=a.per_class, start=a.start, classes=a.classes, captions=a.captions, seed_offset=a.seed_offset,
               shards=a.nshards, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
               diffusers=__import__("diffusers").__version__)
    print(json.dumps({k: v for k, v in run.items() if k != "plan"}), flush=True)
    w, t0, nfes = PackWriter(prefix), time.time(), set()
    for b in range(0, len(jobs), bs):
        chunk = jobs[b:b + bs]
        prompts = [f"{names[k]}, {caps[k][j]}" if caps else TEMPLATES[j % len(TEMPLATES)].format(names[k]) for k, j in chunk]
        gens = [torch.Generator("cuda").manual_seed(1_000_003 * k + j + a.seed_offset) for k, j in chunk]
        if a.stages:
            denoiser(pipe).i = 0
            denoiser(pipe).seed = 1_000_003 * chunk[0][0] + chunk[0][1] + a.seed_offset
        count.n, t1 = 0, time.time()
        # a copy: Z-Image's pipeline writes its chat template into the prompt list it is given
        imgs = pipe(prompt=list(prompts), generator=gens, height=res, width=res, **kw).images
        torch.cuda.synchronize()
        nfe, sec = count.n / len(chunk), (time.time() - t1) / len(chunk)
        nfes.add(nfe)
        if "sigmas" not in run:   # the schedule actually run: stage boundaries read as noise levels, not step indices
            sch = pipe.scheduler
            run["timesteps"] = [round(float(v), 4) for v in sch.timesteps]
            run["sigmas"] = [round(float(v), 5) for v in sch.sigmas] if hasattr(sch, "sigmas") else None
            run["init_noise_sigma"] = round(float(getattr(sch, "init_noise_sigma", 1.0)), 5)   # the scale of the starting noise
            if cfg:
                check_schedule(run, cfg)
        for (k, j), p, im in zip(chunk, prompts, imgs):
            w.add(encode(im, short=256), k, j=j, prompt=p, nfe=nfe, sec=round(sec, 4))
        if (b // bs) % 50 == 0:
            print(f"{tag} shard {a.shard}: {b + len(chunk)}/{len(jobs)} images, {nfe:g} denoiser evaluations and "
                  f"{sec:.3f} s per image", flush=True)
    n = w.close()
    run["nfe"] = sorted(nfes)
    tmp = f"{a.out}/{tag}/gen.json.{a.shard}.tmp"
    json.dump(run, open(tmp, "w"), indent=1)
    os.replace(tmp, f"{a.out}/{tag}/gen.json")   # identical across shards
    print(f"GEN DONE {tag} shard {a.shard}: {n} images in {time.time() - t0:.0f} s ({(time.time() - t0) / max(n, 1):.3f} s per image, "
          f"{'/'.join(f'{v:g}' for v in sorted(nfes))} denoiser evaluations per image)", flush=True)


if __name__ == "__main__":
    main()
