"""Model weights at the exact revisions the paper used (configs/models.json), read from the Hugging Face cache.

    python -m fau.hub download [REPO ...]    # the pinned files of these repositories (default: all), at their revisions
    python -m fau.hub check                  # which pinned snapshots are present

Gated repositories (FLUX.1-dev, SD3.5) need their license accepted on the Hub and a token (`hf auth login`).
"""
import argparse
import json
import os

from fau import ROOT

MODELS = json.loads((ROOT / "configs" / "models.json").read_text())


def cache_dir():
    return os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "hub")


def snapshot(repo):
    """The local directory of a repository at its pinned revision (a local directory is returned as it is)."""
    if os.path.isdir(repo):
        return repo
    rev = MODELS[repo]["revision"]
    d = os.path.join(cache_dir(), "models--" + repo.replace("/", "--"), "snapshots", rev)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"{repo}@{rev[:7]} is not in {cache_dir()}: run `python -m fau.hub download {repo}`")
    return d


def download(repos):
    from huggingface_hub import snapshot_download
    for repo in repos:
        m = MODELS[repo]
        path = snapshot_download(repo, revision=m["revision"], allow_patterns=m["files"])
        print(f"{repo}@{m['revision'][:7]}: {len(m['files'])} files -> {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["download", "check"])
    ap.add_argument("repos", nargs="*")
    a = ap.parse_args()
    repos = a.repos or list(MODELS)
    if a.mode == "download":
        download(repos)
        return
    for repo in repos:
        try:
            snapshot(repo)
            print(f"ok       {repo}")
        except FileNotFoundError:
            print(f"missing  {repo}  ({MODELS[repo]['use']})")


if __name__ == "__main__":
    main()
