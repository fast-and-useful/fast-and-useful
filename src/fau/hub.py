"""Model weights and ImageNet at the exact revisions the paper used (configs/models.json), from the Hugging Face cache.

    python -m fau.hub download [REPO ...]    # the pinned files of these repositories (default: all), at their revisions
    python -m fau.hub check                  # which pinned snapshots are present

Gated repositories (FLUX.1-dev, SD3.5, ImageNet) need their terms accepted on the Hub and a token (`hf auth login`).
The OOD test sets and fine-grained datasets are pinned in fau/ood.py and fau/finegrained.py.
"""
import argparse
import json
import os

from fau import ROOT

MODELS = json.loads((ROOT / "configs" / "models.json").read_text())
DATASETS = {"ILSVRC/imagenet-1k": dict(use="ImageNet-1k (train and validation parquet shards)",
                                      revision="49e2ee26f3810fb5a7536bbf732a7b07389a47b5",
                                      files=["classes.py", "data/train-*.parquet", "data/validation-*.parquet"])}


def cache_dir():
    return os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "hub")


def snapshot(repo):
    """The local directory of a repository at its pinned revision (a local directory is returned as it is)."""
    if os.path.isdir(repo):
        return repo
    kind, rev = ("datasets", DATASETS[repo]["revision"]) if repo in DATASETS else ("models", MODELS[repo]["revision"])
    d = os.path.join(cache_dir(), f"{kind}--" + repo.replace("/", "--"), "snapshots", rev)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"{repo}@{rev[:7]} is not in {cache_dir()}: run `python -m fau.hub download {repo}`")
    return d


def download(repos):
    from huggingface_hub import snapshot_download
    for repo in repos:
        m = DATASETS.get(repo) or MODELS[repo]
        path = snapshot_download(repo, revision=m["revision"], allow_patterns=m["files"],
                                 repo_type="dataset" if repo in DATASETS else "model")
        print(f"{repo}@{m['revision'][:7]} -> {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["download", "check"])
    ap.add_argument("repos", nargs="*")
    a = ap.parse_args()
    repos = a.repos or [*MODELS, *DATASETS]
    if a.mode == "download":
        download(repos)
        return
    for repo in repos:
        try:
            snapshot(repo)
            print(f"ok       {repo}")
        except FileNotFoundError:
            print(f"missing  {repo}  ({(DATASETS.get(repo) or MODELS[repo])['use']})")


if __name__ == "__main__":
    main()
