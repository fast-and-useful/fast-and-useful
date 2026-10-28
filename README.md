# Fast and Useful: Adapting Distilled Generators for Visual Learning

Code to reproduce the tables, figures and numbers of the paper.

Distilled text-to-image generators are fast, but classifiers trained on their images perform worse than classifiers
trained on images from their slower base models. We fit a low-rank update to the released student's first prediction
so that it jumps to an intermediate state of its unguided base, then disable the update and let the unchanged student
finish. The base supplies training targets and is absent at inference.

## Contents

| Path | What it holds |
|---|---|
| `src/fau/` | the method and the experiments (modules listed below) |
| `configs/` | the exact settings behind every number: 345 generated sets (`sets.json`), 43 adapters (`adapters.json`), 15 teacher-pair sets (`pairs.json`), 417 classifier run groups (`runs.json`), and the pinned revision of every model (`models.json`) |
| `results/` | the recorded measurements the paper reads: 1,203 classifier runs with per-class accuracies, the set metrics and OOD and transfer accuracies it quotes, the metric study's inputs, the adapters' preparation costs; and the measured generation and training times behind `fau.plan`'s costs |
| `paper/` | the paper's tables (`tables/*.tex`), the numbers quoted in its text (`numbers.tex`) and its figures, rebuilt from `results/` |
| `data/` | class lists and the prompt files of the paper's sets (real-image captions, LLM descriptions, COCO captions) |

| Module | Role |
|---|---|
| `fau.generate` | generators with per-step control of denoiser and guidance; the adapted sampler; `--set ID` runs a set of the paper |
| `fau.adapter` | teacher pairs from the unguided base (`pairs`) and the first-step low-rank update (`train`) |
| `fau.baselines` | the training-free baselines FSG and MoNO |
| `fau.train`, `fau.train_prior` | classifiers trained from scratch: the paper's recipe, and the recipe of Adamkiewicz et al. |
| `fau.ood`, `fau.transfer` | ImageNet-V2/-Sketch/-R/-A accuracy, and linear probes of frozen features on unseen datasets |
| `fau.metrics.*` | set metrics: DINOv3 and CLIP precision/recall/coverage/Vendi/Frechet distance, FID, recognizability, aesthetic score, PickScore, image statistics |
| `fau.metric_study` | correlations of set metrics with classifier accuracy |
| `fau.imagenet`, `fau.finegrained`, `fau.prompts.*` | real image packs and prompt construction |
| `fau.collect`, `fau.paper`, `fau.plan` | results from a reproduction, the paper's tables and figures, and the commands behind each table |

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # the exact environment of the paper (PyTorch 2.11, CUDA 12.8, diffusers 0.40)
pip install -e .
```

Working data (image packs, adapters, classifier runs) goes to `$FAU_WORK` (default `./work`). Models load from the
Hugging Face cache at the revisions in `configs/models.json`: `python -m fau.hub download [REPO ...]` fetches them and
`python -m fau.hub check` lists what is missing. FLUX.1-dev, SD3.5 and ImageNet are gated: accept their terms on the
Hub and log in (`hf auth login`) first.

## 1. Rebuild the paper's tables and numbers (CPU, about 5 minutes)

```bash
python -m fau.paper            # results/ -> paper/tables/*.tex, paper/numbers.tex, paper/figures/*.pdf
```

Every table is the LaTeX the paper inputs and every number of the text is a macro in `paper/numbers.tex`
(`\tp{group}`: a run group's top-1; `\gp{name}`: a paired gap; `\nm{name}`: a derived number). The paired gaps
average each class over the classifier runs and bootstrap the classes (10,000 resamples). The metric study's
bootstrap takes most of the time; `--workers 8` parallelizes it. The opening teaser is a designed illustration and is
not rebuilt; its numbers are `\tpsd{klein}` and `\tpsd{kd_ad2_x3}`. `python tests/test_release.py` checks that the
configs reference each other completely, that the recorded schedules follow from them, and that the rebuild matches
`paper/` exactly.

## 2. Fit and apply an adapter

The adapter of FLUX.2-klein in the paper (`kd_ad2`): 16,000 teacher pairs from the unguided klein base (160 noises per
ImageNet-100 class name, cycling eight photo templates), a rank-32 update fitted for 3,000 steps, and the adapted
sampler: the adapted first step jumps from noise 1 to the base's state at 0.7175, then the released student's last
step finishes (2 denoiser evaluations).

```bash
python -m fau.hub download black-forest-labs/FLUX.2-klein-4B black-forest-labs/FLUX.2-klein-base-4B
for i in 0 1 2 3; do python -m fau.adapter pairs --config pairs_in100 --shard $i; done   # 4.6 H100-hours in total
python -m fau.adapter train --config ad2                                                 # 1.1 H100-hours
for i in 0 1; do python -m fau.generate --set in100/kd_ad2 --shard $i; done               # 50,000 images, 0.10 s each
```

`--config` and `--set` read the recorded settings; every option can also be given directly, for example for your own
classes (a class list is a JSON file with `wnids` and `names`):

```bash
python -m fau.adapter pairs --family klein --classes my_classes.json --per_class 160 --out work/pairs/mine
python -m fau.adapter train --family klein --classes my_classes.json --pairs work/pairs/mine --target 7175 --out work/adapters/mine.pt
python -m fau.generate --model klein --other klein_ad:work/adapters/mine.pt --sigmas '4[0:1],4[3:]' \
    --stages other:1:0-1,base:1:1-2 --classes my_classes.json --per_class 500 --out work/sets/mine
```

| Student | `--family` | `--target` | Fit batch | Adapted sampler (`fau.generate`) | NFE |
|---|---|---|---|---|---|
| FLUX.2-klein | `klein` | `7175` | 8 | `--model klein --other klein_ad:ID --sigmas '4[0:1],4[3:]' --stages other:1:0-1,base:1:1-2` | 2 |
| SD-Turbo | `sd` | `499` | 16 | `--model sd_turbo --other sdt_ad:ID --timesteps 999,499,249 --stages other:1:0-1,base:1:1-3` | 3 |
| Z-Image-Turbo | `zimage` | `857` | 8 | `--model zimage_turbo --other zt_ad:ID --sigmas '9[0:1],9[3:]' --stages other:1:0-1,base:1:1-7` | 7 |
| DP-DMD's SD3.5-Medium student | `sd35m` | `602` | 8 | `--model dpdmd_sd35m --other sd35m_ad:ID --sigmas '4[0:1],4[2:]' --stages other:1:0-1,base:1:1-3` | 3 |

Teacher pairs need the base and the student; generation needs only the student and its adapter. klein's primary
adapters use the teacher grid `--grid v1` (the default); the ablation "same, finer teacher grid" and the guided and
deeper targets use `--grid fine` (`configs/pairs.json` records which).

## 3. Reproduce a table or figure from scratch

`python -m fau.plan TARGET` prints every command a table, figure or the text's numbers need, in order, with the
recorded H100 cost: models and datasets, teacher pairs, adapters, generated sets, classifiers, evaluations, and the
final comparison. Targets are table and figure files (`t2-ablation`, `f3-comparison`), the paper's labels
(`tab:ablation`), `numbers`, or `all`.

```bash
python -m fau.plan t2-ablation                     # the commands
python -m fau.plan all --summary                   # counts and H100-hours only
```

The pipeline ends with

```bash
python -m fau.collect --out results-repro          # $FAU_WORK -> a results directory laid out as results/
python -m fau.paper --results results-repro --out results-repro/paper --compare paper
```

which lists every table cell and every number that differs from the paper. Image generation and adapter fitting use
the recorded seeds, schedules, shard counts and batch sizes (a set checks the schedule it runs against the recorded
one); classifier training on GPUs is not bit-deterministic, so accuracies reproduce within the classifier-seed
variation the paper reports (mean and SD over three seeds).

Recorded cost per target (H100-hours: generation, classifiers and adapter preparation; model loading excluded):

| Target (`fau.plan` name) | In the paper | H100-hours |
|---|---|---|
| `t0-method-config` | teacher targets and adapted schedules | none: follows from `configs/` |
| `t1-scale-transfer` | 200 classes and full ImageNet-1k | 776 |
| `t2-ablation` | target guidance, endpoint, placement, rank, training-free baselines | 96 |
| `t3-finegrained` | Food-101, Cars, Pets | 152 |
| `t5-learners` | RandAugment, the prior recipe, ConvNeXt-T, ViT-S/16 | 34 |
| `t6-prompts` | prompt regimes, ImageNet-100 and ImageNet-1k | 107 |
| `f3-comparison`, `f4-generation-cost` | main comparison and generation cost | 134, 129 |
| `f5-method` | the method figure | none: recorded images |
| `t1-main` | complete primary results (supplement) | 1,132 |
| `s3-metrics`, `s11-heldout`, `s13-family-weighting` | the metric study (186 sets) | 1,741 |
| `s14-deployment`, `s-e50-costs` | preparation and generation costs | 640, 42 |
| `s6-ood`, `s15-contrasts` | shifted test sets, real-data mixtures | 526, 454 |
| `s5-captions` | templates, real-image captions, LLM descriptions | 369 |
| `s12-unseen-fullscale`, `s9-transfer` | unseen ImageNet classes, feature transfer | 376, 41 |
| `f3a-guidance-window`, `f3b-handoff` | guidance windows and hand-offs | 369, 289 |
| `numbers` | every number quoted in the text (the metric study's included) | 3,051 |
| `all` | everything: 1,203 classifier runs on 345 sets, 43 adapters | 3,690 |

Targets share most of their runs, so their costs do not add up. Adapter preparation is counted for the four adapters
whose cost the paper reports (`results/costs.json`: 5.7 H100-hours for klein's, 0.9 for SD-Turbo's, 7.9 and 1.2 for
their counterparts fitted on LLM descriptions); `fau.plan` lists the other adapters' pair and fit commands without a
cost. Eight sets have no recorded generation time and are not counted: seven early FLUX.2-klein sets of the metric
study (`klein_all50`, `klein_b2d*`, `klein_d2b*`) and `kd_ad2_ext`. The method figure composes twelve recorded images
(`results/figure-images`: images 0, 5, 9 and 15 of class 54 in the sets `kb_or34_d1`, `kd_ad2` and `klein`).

## Conventions

- **Sets** are `<protocol>/<name>`: `in100` (ImageNet-100, 500 images per class), `tin200` (the 200-class protocol),
  `in1k` (ImageNet-1k; `_full`: 1,281 images per class), `food101`, `cars`, `pets`. Image `j` of class `k` uses seed
  `1,000,003 k + j` plus the set's `seed_offset` and template `j mod 8`, so sets pair up image by image.
- **Run groups** name a classifier trained with three seeds; most are named after their training set. Prefixes give
  the protocol (`tin200_`, `in1k_`, `food101_`, `cars_`, `pets_`), the prompts (`cap_`: real-image captions, `llm_`,
  `llm2_`, `llm3_`: three independently generated LLM-description datasets), other learners (`ra_`: RandAugment,
  `wp_`: the prior recipe, `cnx_`: ConvNeXt-T, `vits_`: ViT-S/16), real data added (`mix_`: 500 real images per
  class, `low_`: 50) or subsets (`scale_`). `_x3` and `llmx_` pool three adapter trainings or three datasets.
- **Samplers**: `klein` and `sdt4` are the released FLUX.2-klein and SD-Turbo (4 steps); `kb_*` and `sd21*` their
  bases (`kb_g1`, `sd21_g1`: unguided); `*_dd<k>_g1` base-first (the unguided base draws the first k steps);
  `kd_ad2`, `sdt_ad1`, `zt_ad2` our adapters; `zt_sda_*`
  the public SDA adapter; `dpdmd*` DP-DMD's student. Every set and run group has a one-line `description` in
  `configs/`.

## Data and licenses

The code is released under the MIT license. The generators, encoders and datasets keep their own licenses and terms
of use (several generators restrict commercial use; ImageNet requires accepting its terms of access). The
prompt files contain captions and descriptions written by Qwen2.5-VL-7B-Instruct and 16,000 COCO-2017 training captions
(CC BY 4.0); `results/figure-images` holds the twelve FLUX.2-klein images of the method figure.
