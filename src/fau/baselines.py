"""Class-conditional training sets from distilled FLUX.2-klein-4B with two published training-free diversity methods,
the inference-time baselines for the commitment-step adapter:

  fsg   Feature Self-Guidance (Bhat et al., "Don't Settle at the Mode!", ECCV 2026, arXiv 2606.27371)
  mono  manifold-constrained noise optimization, MoNO (Shi et al., arXiv 2607.23937)
  none  plain klein, batched the same way (the control)

Neither paper released code, so both are reimplemented from the papers with their FLUX.2-klein settings.

Prompts, seeds, job order, pack format and gen.json follow generate.py, whose functions are imported: image (class k, index
j) uses template j mod 8 and the seed 1_000_003 k + j. Jobs are cut into batches of at most --bs (8) images of one
class in generate.py's job order. With 8 or 16 shards every image of a class in a shard has the same template, so a batch is
up to eight seeds of one prompt (FSG's setting).

fsg: in steps at noise level t >= 0.8 (klein's 4-step schedule at 512 px is 1.0, 0.958, 0.884, 0.718: steps 0-2), the
image-stream output of klein's first transformer block is dispersed across the batch by N_opt Adam steps of learning
rate w on G = log mean_ij exp(cos(h_i, h_j) / tau) (cosine of the flattened image tokens), passed through the same
block once more and blended, h' = beta B(h~) + (1 - beta) h~; the remaining blocks run on h'. Klein settings from the
paper: N = 8, N_opt = 10, w = 0.05, tau = 0.5, beta = 0.1, block 1, t in [1.0, 0.8].

mono: the images of a class are made one at a time. Each initial noise (the pipeline's own draw for its seed) is
rotated K times by phi on the sphere of its low-frequency part (per-channel 2D FFT, circular cutoff rho; the high
frequencies and the norm stay fixed) along the Riemannian gradient of the mean squared projection of its proxy
image's DINOv2-base patch features (256 px, no CLS, each patch L2-normalized) onto the span of the same patch position's
features in the class's previous final images. The proxy is klein's one-step estimate z - sigma_0 v(z, sigma_0),
VAE-decoded. The final image comes from the unchanged 4-step sampler; its patch features join the bases (Gram-Schmidt).
A group (one basis) is at most --mono_group images of a class, by default the paper's 4; its first image is not
optimized. Klein settings from the paper: K = 4, phi = 20 degrees (K = 1: 30 degrees), rho = 0.15, DINOv2-base.
--mono_dir random is the control: the same rotations in random tangent directions, no objective.

Every record carries its denoiser forward evaluations (nfe; mono: 4 + K once the basis is non-empty, random: 4) and GPU
seconds; gen.json adds the extra block passes (fsg) and backward passes (mono) per image and diagnostics of the
intervention (fsg_stats, mono_stats).

Usage: python -m fau.baselines --set in100/kd_fsg [--shard i]        # a set of the paper (configs/sets.json)
       python -m fau.baselines --method fsg|mono|none --classes in100 --per_class 500 --out DIR --tag NAME \
           [--shard i --nshards n] [--limit N] [--fsg_* / --mono_* overrides]
"""
import argparse, json, math, os, time

import numpy as np
import torch
import torch.nn.functional as F

from fau import ROOT, SETS
from fau.generate import TEMPLATES, Counter, class_name, load
from fau.hub import snapshot
from fau.packs import PackWriter, encode
from fau.prompts import class_list

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def batches(jobs, bs):
    """generate.py's job order cut into runs of at most bs consecutive jobs of one class."""
    out = []
    for k, j in jobs:
        if out and out[-1][0][0] == k and len(out[-1]) < bs:
            out[-1].append((k, j))
        else:
            out.append([(k, j)])
    return out


# ---------------------------------------------------------------- Feature Self-Guidance

def fsg_energy(h, tau):
    """FSG's Eq. 1: G = log mean_ij exp(cos(h_i, h_j) / tau) over all N^2 pairs of the batch (i = j included, as
    written); h_i is sample i's feature map, flattened."""
    f = F.normalize(h.flatten(1), dim=1)
    return torch.logsumexp((f @ f.T).flatten() / tau, 0) - 2 * math.log(len(f))


def mean_cos(h):
    """Mean cosine similarity between the flattened feature maps of two different samples of the batch."""
    f = F.normalize(h.flatten(1).float(), dim=1)
    n = len(f)
    return ((f @ f.T).sum().item() - n) / (n * (n - 1))


def disperse(h, steps, lr, tau, opt):
    """FSG's Eq. 2, h <- h - w grad G, for N_opt steps on a float32 copy of the features. opt 'adam' takes the steps with
    torch.optim.Adam at learning rate w; 'sgd' is the literal update, which moves block features by a vanishing fraction
    of their norm at the paper's w (the cosine gradient scales as 1 / |h|)."""
    x = h.detach().float().clone().requires_grad_(True)
    o = torch.optim.Adam([x], lr=lr) if opt == "adam" else torch.optim.SGD([x], lr=lr)
    info = {}
    with torch.enable_grad():
        for i in range(steps):
            o.zero_grad(set_to_none=True)
            fsg_energy(x, tau).backward()
            if i == 0:   # the gradient's scale, and Adam's first step as a fraction of lr (|g| / (|g| + eps))
                a = x.grad.abs()
                info = dict(grad_abs=a.mean().item(), adam_first_step=(a / (a + 1e-8)).mean().item())
            o.step()
    return x.detach(), info


class FSGBlock(torch.nn.Module):
    """Stands in for one double-stream block of klein's transformer (FSG's disperse-and-refine): runs the block, and at
    noise levels t >= tmin disperses its image-stream output across the batch, passes the dispersed features through the
    same block again (text input: the first pass's text output) and blends, h' = beta B(h~) + (1 - beta) h~. The text
    stream continues from the first pass. t is set before every transformer call by track_t."""

    def __init__(self, block, steps, lr, tau, beta, tmin, opt):
        super().__init__()
        self.block = block
        self.steps, self.lr, self.tau, self.beta, self.tmin, self.opt = steps, lr, tau, beta, tmin, opt
        self.t, self.extra, self.stats = None, 0, {}

    def forward(self, hidden_states, encoder_hidden_states, **kw):
        enc, h = self.block(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, **kw)
        if self.t is None or self.t < self.tmin - 1e-6 or len(h) < 2:
            return enc, h
        ht, info = disperse(h, self.steps, self.lr, self.tau, self.opt)
        _, hh = self.block(hidden_states=ht.to(h.dtype), encoder_hidden_states=enc, **kw)
        out = self.beta * hh.float() + (1 - self.beta) * ht
        self.extra += len(h)
        self.log(h.float(), ht, out, info)
        return enc, out.to(h.dtype)

    def log(self, h, ht, out, info):
        s = self.stats.setdefault(f"{self.t:.3f}", dict(calls=0, norm=0.0, cos=0.0, cos_dispersed=0.0, cos_out=0.0, change=0.0,
                                                         change_rms=0.0, grad_abs=0.0, adam_first_step=0.0))
        norm, d = h.flatten(1).norm(dim=1), (ht - h).flatten(1)
        for key, v in (("norm", norm.mean().item()), ("cos", mean_cos(h)), ("cos_dispersed", mean_cos(ht)), ("cos_out", mean_cos(out)),
                       ("change", (d.norm(dim=1) / norm).mean().item()), ("change_rms", d.pow(2).mean().sqrt().item()),
                       ("grad_abs", info.get("grad_abs", 0.0)), ("adam_first_step", info.get("adam_first_step", 0.0))):
            s[key] += v
        s["calls"] += 1

    def summary(self):
        """Per noise level, averaged over batches: mean feature norm; mean pairwise cosine before / after dispersion / after
        the blend; the dispersion's relative change |h~ - h| / |h| and its rms per element; the first step's mean |grad G|
        per element and Adam's first step as a fraction of the learning rate."""
        return {t: dict({k: float(f"{v / s['calls']:.5g}") for k, v in s.items() if k != "calls"}, calls=s["calls"])
                for t, s in self.stats.items()}


def track_t(fsg):
    """Transformer pre-hook: the noise level of the coming call (the pipeline passes timestep / 1000 = sigma)."""
    def hook(module, args, kwargs):
        fsg.t = float(kwargs["timestep"].flatten()[0])
    return hook


# ---------------------------------------------------------------- MoNO

def lowpass_mask(h, w, rho):
    """MoNO's binary circular low-pass mask over normalized 2D frequencies: |omega| / sqrt(1/2) <= rho."""
    fy, fx = torch.fft.fftfreq(h), torch.fft.fftfreq(w)
    return ((fy[:, None] ** 2 + fx[None, :] ** 2).sqrt() / math.sqrt(0.5) <= rho).float()


def lowpass(z, mask):
    """P_L: the spatial Fourier low-pass of every channel. An orthogonal projection: the mask is symmetric under
    omega -> -omega, so real inputs stay real."""
    return torch.fft.ifft2(torch.fft.fft2(z.float()) * mask).real


def rotate(u, g, r, mask, phi):
    """One Riemannian step on the low-frequency sphere {u in Im(P_L): |u| = r}: the Euclidean gradient g is projected
    onto the tangent space (P_L g minus its radial part) and u moves along the geodesic against it by the angle phi
    (MoNO sets the angle alpha |h| / r = phi rather than the step size alpha)."""
    gl = lowpass(g, mask)
    d = -(gl - (gl * u).sum() / (u * u).sum() * u)
    n = d.norm()
    if not torch.isfinite(n) or n <= 1e-12 * r:
        return u
    return u * math.cos(phi) + r * d / n * math.sin(phi)


def gs_append(B, v, eps=1e-6):
    """Adds v (P, D) to the per-patch orthonormal bases B (P, D, m), or starts them: classical Gram-Schmidt run twice; a
    vector already in the span adds a zero column."""
    r = v.float()
    if B is not None:
        for _ in range(2):
            r = r - torch.einsum("pdm,pm->pd", B, torch.einsum("pdm,pd->pm", B, r))
    n = r.norm(dim=-1, keepdim=True)
    q = torch.where(n > eps, r / n.clamp_min(eps), torch.zeros_like(r)).unsqueeze(-1)
    return q if B is None else torch.cat([B, q], -1)


def proj_loss(B, psi):
    """MoNO's loss: the squared length of each patch feature's projection onto its basis, averaged over patches."""
    return torch.einsum("pdm,pd->pm", B, psi).pow(2).sum(-1).mean()


class PatchFeatures:
    """MoNO's feature map, differentiable: an image in [0, 1] resized to res x res, ImageNet-normalized, the DINOv2 last
    hidden state without the CLS token, every patch L2-normalized -> (B, P, D)."""

    def __init__(self, name, res):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(snapshot(name), dtype=torch.float32, disable_mmap=True).to("cuda").eval().requires_grad_(False)
        self.res = res
        self.mean = torch.tensor(IMAGENET_MEAN, device="cuda").view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device="cuda").view(1, 3, 1, 1)

    def __call__(self, img):
        x = F.interpolate(img.float(), size=(self.res, self.res), mode="bilinear", antialias=True, align_corners=False)
        return F.normalize(self.model(pixel_values=(x - self.mean) / self.std).last_hidden_state[:, 1:].float(), dim=-1)


class MoNO:
    """Sequential noise optimization over the images of a class (one basis per group of at most `group` images).
    direction 'random' is the control: the same rotations of the low-frequency part, in random tangent directions drawn
    from the image's own generator (no proxy, no features)."""

    def __init__(self, pipe, res, steps, iters, angle, rho, feat, feat_res, group, direction="grad"):
        from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu
        self.pipe, self.iters, self.phi, self.group = pipe, iters, math.radians(angle), group
        self.random = direction == "random"
        self.feat = None if self.random else PatchFeatures(feat, feat_res)
        sch = pipe.scheduler   # the pipeline's own schedule, for the proxy's noise level and timestep
        sch.set_timesteps(sigmas=np.linspace(1.0, 1 / steps, steps), mu=compute_empirical_mu((res // 16) ** 2, steps), device="cuda")
        self.sigma0, self.t0 = float(sch.sigmas[0]), sch.timesteps[0].clone()
        self.shape = (1, pipe.transformer.config.in_channels, res // 16, res // 16)   # the pipeline's patchified noise
        self.mask = lowpass_mask(self.shape[2], self.shape[3], rho).to("cuda")
        vae = pipe.vae
        self.bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).float()
        self.bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).float()
        self.ids = pipe._prepare_latent_ids(torch.zeros(self.shape)).to("cuda")
        for m in (pipe.transformer, pipe.vae, pipe.text_encoder):
            m.requires_grad_(False)
        self.basis, self.key, self.count = None, None, 0
        self.stats = dict(images=0, optimized=0, backward=0, loss_iter=[0.0] * iters, final_proj=0.0, basis=0)

    def decode(self, packed):
        """Clean latent tokens (1, L, C) -> image in [0, 1], as the pipeline decodes (batch-norm stats, unpatchify, VAE)."""
        x = packed.permute(0, 2, 1).reshape(packed.shape[0], self.shape[1], self.shape[2], self.shape[3])
        x = self.pipe._unpatchify_latents(x * self.bn_std + self.bn_mean)
        return (self.pipe.vae.decode(x.to(self.pipe.vae.dtype), return_dict=False)[0].float() / 2 + 0.5).clamp(0, 1)

    def proxy_loss(self, z, emb, txt_ids):
        """The loss of noise z (1, C, H, W) through the one-step proxy z - sigma_0 v(z, sigma_0)."""
        packed = z.reshape(1, self.shape[1], -1).permute(0, 2, 1)
        t = self.t0.expand(1).to(torch.bfloat16) / 1000   # as the pipeline passes it
        v = self.pipe.transformer(hidden_states=packed.to(torch.bfloat16), timestep=t, guidance=None, encoder_hidden_states=emb,
                                  txt_ids=txt_ids, img_ids=self.ids, return_dict=False)[0]
        return proj_loss(self.basis, self.feat(self.decode(packed - self.sigma0 * v.float()))[0])

    def noise(self, gen, k, emb, txt_ids):
        """The initial noise for the next image of class k: the pipeline's draw from gen, optimized against the group's
        bases -> (noise in bf16, backward passes spent)."""
        from diffusers.utils.torch_utils import randn_tensor
        z0 = randn_tensor(self.shape, generator=gen, device=torch.device("cuda"), dtype=torch.bfloat16)
        if self.key != k or self.count >= self.group:   # a new group: empty bases
            self.basis, self.key, self.count = None, k, 0
        self.count += 1
        self.stats["images"] += 1
        if self.count == 1 or self.iters == 0:   # a group's first image: nothing to avoid yet
            return z0, 0
        z = z0.float()
        u = lowpass(z, self.mask)
        zh, r = z - u, u.norm()
        if self.random:
            for _ in range(self.iters):
                u = rotate(u, torch.randn(self.shape, generator=gen, device="cuda"), r, self.mask, self.phi)
            self.stats["optimized"] += 1
            return (zh + u).to(torch.bfloat16), 0
        losses = []
        for _ in range(self.iters):
            zz = (zh + u).requires_grad_(True)
            with torch.enable_grad():
                loss = self.proxy_loss(zz, emb, txt_ids)
                g, = torch.autograd.grad(loss, zz)
            losses.append(loss.item())
            u = rotate(u, g, r, self.mask, self.phi)
        s = self.stats
        s["optimized"] += 1; s["backward"] += self.iters
        s["loss_iter"] = [a + b for a, b in zip(s["loss_iter"], losses)]; s["basis"] += self.basis.shape[-1]
        return (zh + u).to(torch.bfloat16), self.iters

    def add(self, img):
        """Adds a final image's patch features to the group's bases."""
        if self.random:
            return
        psi = self.feat(img)[0]
        if self.basis is not None:
            self.stats["final_proj"] += proj_loss(self.basis, psi).item()
        self.basis = gs_append(self.basis, psi)

    def summary(self):
        s, n = self.stats, max(self.stats["optimized"], 1)
        if self.random:
            return dict(images=s["images"], rotated=s["optimized"], direction="random")
        return dict(images=s["images"], optimized=s["optimized"], backward_per_image=round(s["backward"] / max(s["images"], 1), 4),
                    proxy_loss_per_iter=[round(v / n, 5) for v in s["loss_iter"]],   # each before that iteration's rotation
                    final_image_proj=round(s["final_proj"] / n, 5), mean_basis_size=round(s["basis"] / n, 2))


# ---------------------------------------------------------------- generation

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=None, help="a set of the paper, e.g. in100/kd_fsg (configs/sets.json): its recorded settings")
    ap.add_argument("--method", default=None, choices=["fsg", "mono", "none"])
    ap.add_argument("--classes", default=None, help="a class list (in100, ...) or a JSON file")
    ap.add_argument("--per_class", type=int, default=500)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=0, help="shards (default 1; with --set, the recorded number, which sets the "
                                                            "batches FSG disperses and the order MoNO's bases grow in)")
    ap.add_argument("--limit", type=int, default=0, help="first N jobs of the shard only (smoke tests)")
    ap.add_argument("--bs", type=int, default=8, help="images per batch, all of one class (FSG's N)")
    ap.add_argument("--fsg_block", type=int, default=1, help="1-based double-stream block (paper: 1 for klein, 2 for FLUX.1)")
    ap.add_argument("--fsg_steps", type=int, default=10, help="N_opt")
    ap.add_argument("--fsg_lr", type=float, default=0.05, help="w")
    ap.add_argument("--fsg_tau", type=float, default=0.5)
    ap.add_argument("--fsg_beta", type=float, default=0.1)
    ap.add_argument("--fsg_tmin", type=float, default=0.8, help="act at noise levels t in [tmin, 1]")
    ap.add_argument("--fsg_opt", default="adam", choices=["adam", "sgd"], help="adam (default) or the literal update")
    ap.add_argument("--mono_iters", type=int, default=4, help="K")
    ap.add_argument("--mono_angle", type=float, default=20.0, help="rotation per iteration, degrees (paper: 20 at K = 4, 30 at K = 1)")
    ap.add_argument("--mono_rho", type=float, default=0.15, help="low-frequency cutoff")
    ap.add_argument("--mono_feat", default="facebook/dinov2-base")
    ap.add_argument("--mono_res", type=int, default=256, help="feature extractor input size")
    ap.add_argument("--mono_group", type=int, default=4,
                    help="images per basis: 4 is the paper's set size; 32 with 16 shards is all of a class in a shard")
    ap.add_argument("--mono_dir", default="grad", choices=["grad", "random"],
                    help="grad (MoNO) or random: the same rotations in random directions (the control)")
    a = ap.parse_args()
    if a.set:
        cfg = json.loads((ROOT / "configs" / "sets.json").read_text())[a.set]
        assert cfg["generator"] == "baselines", f"{a.set} is made by python -m fau.generate --set {a.set}"
        for k, v in cfg["args"].items():
            setattr(a, k, v)
        a.classes, a.out, a.tag = cfg["classes"], str(SETS / a.set.split("/")[0]), a.set.split("/")[1]
        a.nshards = a.nshards or cfg["shards"]
    assert a.method and a.classes and a.out, "give --set, or --method, --classes and --out"
    a.nshards = a.nshards or 1
    meta = class_list(a.classes)
    names = [class_name(w, n) for w, n in zip(meta["wnids"], meta["names"])]
    dup = {n for n in names if names.count(n) > 1}
    assert not dup, f"ambiguous class names in this subset: {dup}"
    jobs = [(k, j) for k in range(len(names)) for j in range(a.per_class)][a.shard::a.nshards]
    if a.limit:
        jobs = jobs[:a.limit]
    tag = a.tag or f"kd_{a.method}"
    prefix = f"{a.out}/{tag}/part{a.shard}"
    if os.path.exists(prefix + ".json"):
        print(f"{prefix} exists, nothing to do"); return
    pipe, kw, res, _, _ = load("klein")
    kw = dict(kw, guidance_scale=1.0)   # the distilled pipeline never runs CFG; 1.0 only stops its per-call warning
    settings, fsg, mono = {}, None, None
    if a.method == "fsg":
        blocks = pipe.transformer.transformer_blocks
        assert 1 <= a.fsg_block <= len(blocks), f"klein has {len(blocks)} double-stream blocks"
        fsg = FSGBlock(blocks[a.fsg_block - 1], a.fsg_steps, a.fsg_lr, a.fsg_tau, a.fsg_beta, a.fsg_tmin, a.fsg_opt)
        blocks[a.fsg_block - 1] = fsg
        pipe.transformer.register_forward_pre_hook(track_t(fsg), with_kwargs=True)
        settings = dict(block=a.fsg_block, n_opt=a.fsg_steps, w=a.fsg_lr, tau=a.fsg_tau, beta=a.fsg_beta, t_window=[a.fsg_tmin, 1.0],
                        optimizer=a.fsg_opt, features="image-stream output of the block, flattened",
                        refine_text_input="the first pass's text-stream output")
    count = Counter(pipe.transformer)
    pipe.to("cuda"); pipe.set_progress_bar_config(disable=True)
    if a.method == "mono":
        mono = MoNO(pipe, res, kw["num_inference_steps"], a.mono_iters, a.mono_angle, a.mono_rho, a.mono_feat, a.mono_res, a.mono_group,
                    a.mono_dir)
        settings = dict(K=a.mono_iters, angle_deg=a.mono_angle, rho=a.mono_rho, direction=a.mono_dir, group=a.mono_group,
                        features=None if a.mono_dir == "random" else a.mono_feat, feature_res=a.mono_res, proxy_sigma=mono.sigma0,
                        noise_shape=list(mono.shape[1:]), low_frequencies=int(mono.mask.sum().item()) * mono.shape[1])
    run = dict(tag=tag, model="klein", method=a.method, settings=settings, steps=kw["num_inference_steps"], resolution=res, batch=a.bs,
               guidance=None, per_class=a.per_class, classes=a.classes, shards=a.nshards, gpu=torch.cuda.get_device_name(),
               torch=torch.__version__, diffusers=__import__("diffusers").__version__, code="gen_divfix.py")
    print(json.dumps(run), flush=True)
    ip = pipe.image_processor
    w, t0, nfes, done = PackWriter(prefix), time.time(), set(), 0
    for bi, chunk in enumerate(batches(jobs, a.bs)):
        prompts = [TEMPLATES[j % len(TEMPLATES)].format(names[k]) for k, j in chunk]
        gens = [torch.Generator("cuda").manual_seed(1_000_003 * k + j) for k, j in chunk]
        if mono is None:   # none / fsg: one batched call, exactly as generate.py makes it
            count.n, t1 = 0, time.time()
            imgs = pipe(prompt=prompts, generator=gens, height=res, width=res, **kw).images
            torch.cuda.synchronize()
            costs = [(count.n / len(chunk), (time.time() - t1) / len(chunk))] * len(chunk)
        else:   # mono: one image at a time, each after the previous one joined the bases
            t1 = time.time()
            with torch.no_grad():
                emb, txt_ids = pipe.encode_prompt(prompts, device=torch.device("cuda"))
            torch.cuda.synchronize()
            share = (time.time() - t1) / len(chunk)   # the batch's text encoding, split evenly
            imgs, costs = [], []
            for i, ((k, j), g) in enumerate(zip(chunk, gens)):
                count.n, t1 = 0, time.time()
                z, _ = mono.noise(g, k, emb[i:i + 1], txt_ids[i:i + 1])
                out = pipe(prompt_embeds=emb[i:i + 1], latents=z, height=res, width=res, output_type="pt", **kw).images
                with torch.no_grad():
                    mono.add(out)
                torch.cuda.synchronize()
                imgs += ip.numpy_to_pil(ip.pt_to_numpy(out))   # what output_type="pil" returns
                costs.append((float(count.n), time.time() - t1 + share))
        if "sigmas" not in run:
            sch = pipe.scheduler
            run["timesteps"] = [round(float(v), 4) for v in sch.timesteps]
            run["sigmas"] = [round(float(v), 5) for v in sch.sigmas]
        for (k, j), p, im, (nfe, sec) in zip(chunk, prompts, imgs, costs):
            w.add(encode(im, short=256), k, j=j, prompt=p, nfe=nfe, sec=round(sec, 4))
            nfes.add(nfe)
        done += len(chunk)
        if bi % 50 == 0:
            print(f"{tag} shard {a.shard}: {done}/{len(jobs)} images, {costs[-1][0]:g} denoiser evaluations and "
                  f"{sum(c[1] for c in costs) / len(costs):.3f} s per image", flush=True)
    n = w.close()
    run["nfe"] = sorted(nfes)
    run["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 1)
    if fsg is not None:
        run["extra_block_passes_per_image"] = round(fsg.extra / max(n, 1), 4)
        run["fsg_stats"] = fsg.summary()
    if mono is not None:
        run["mono_stats"] = mono.summary()
    print(json.dumps({k: run[k] for k in ("nfe", "peak_gpu_gb", "fsg_stats", "mono_stats", "extra_block_passes_per_image") if k in run}),
          flush=True)
    tmp = f"{a.out}/{tag}/gen.json.{a.shard}.tmp"
    json.dump(run, open(tmp, "w"), indent=1)
    os.replace(tmp, f"{a.out}/{tag}/gen.json")   # settings identical across shards; the diagnostics are this shard's
    print(f"GEN DONE {tag} shard {a.shard}: {n} images in {time.time() - t0:.0f} s ({(time.time() - t0) / max(n, 1):.3f} s per image, "
          f"{'/'.join(f'{v:g}' for v in sorted(nfes))} denoiser evaluations per image)", flush=True)


if __name__ == "__main__":
    main()
