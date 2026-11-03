#!/usr/bin/env python3
"""Apply the pinned A3 MegaMoE source patches without building or installing."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def git(repo, *args):
    return (
        subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.PIPE)
        .decode()
        .strip()
    )


def digest(path):
    if path.is_symlink():
        raise ValueError(f"Refusing symlink: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def inspect_component(bundle, component, repo, reverse):
    repo = repo.resolve(strict=True)
    if Path(git(repo, "rev-parse", "--show-toplevel")).resolve() != repo:
        raise ValueError(f"Pass the repository root: {repo}")
    base = component["base_revision"]
    head = git(repo, "rev-parse", "HEAD")
    if head != base:
        raise ValueError(f"{component['name']}: expected HEAD {base}, got {head}")
    patch = bundle / component["patch"]
    if digest(patch) != component["sha256"]:
        raise ValueError(f"Patch checksum mismatch: {patch}")
    files = component["files"]
    if any(Path(p).is_absolute() or ".." in Path(p).parts for p in files):
        raise ValueError("Invalid manifest path")
    # Unrelated changes must never be mistaken for part of this patch.
    changed = set(git(repo, "diff", "HEAD", "--name-only").splitlines())
    if changed - files.keys():
        raise ValueError(
            f"Unrelated tracked changes in {repo}: {sorted(changed - files.keys())}"
        )
    if git(repo, "diff", "--cached", "--name-only"):
        raise ValueError(f"Staged changes in {repo}; unstage before applying")
    states = []
    for name, hashes in files.items():
        path = repo / name
        # Reject symlinked parent directories as well as symlinked files.
        if any(p.is_symlink() for p in [path, *path.parents] if p != repo.parent):
            raise ValueError(f"Refusing symlinked path: {path}")
        actual = digest(path)
        states.append((actual == hashes["before"], actual == hashes["after"]))
    before = all(s[0] for s in states)
    after = all(s[1] for s in states)
    if not before and not after:
        raise ValueError(
            f"{component['name']}: partial patch or modified source; refusing"
        )
    done = before if reverse else after
    if not done:
        args = ["apply", "--check"]
        if reverse:
            args.append("--reverse")
        git(repo, *args, str(patch))
    state = "original" if before else "patched"
    print(
        f"{component['name']}: {head} {state}" + (" (no change needed)" if done else "")
    )
    return repo, patch, component, done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sgl-kernel-npu", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="Validate without writing")
    parser.add_argument(
        "--reverse", action="store_true", help="Restore original source files"
    )
    args = parser.parse_args()
    bundle = Path(__file__).resolve().parent
    manifest = json.loads((bundle / "manifest.json").read_text())
    roots = {
        "sgl-kernel-npu": args.sgl_kernel_npu,
    }
    # Validate source hashes before changing any files.
    plans = [
        inspect_component(bundle, c, roots[c["name"]], args.reverse)
        for c in manifest["components"]
    ]
    if args.check:
        print("Check passed; no files changed")
        return
    completed = []
    try:
        for repo, patch, component, done in plans:
            if done:
                continue
            git(repo, "apply", *(["--reverse"] if args.reverse else []), str(patch))
            completed.append((repo, patch))
            key = "before" if args.reverse else "after"
            if any(
                digest(repo / p) != hashes[key]
                for p, hashes in component["files"].items()
            ):
                raise ValueError(f"Post-apply checksum mismatch in {repo}")
    except Exception:
        for repo, patch in reversed(completed):
            git(repo, "apply", *([] if args.reverse else ["--reverse"]), str(patch))
        raise
    print("Source patches ready; no packages built or installed")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        detail = (
            exc.stderr.decode().strip()
            if isinstance(exc, subprocess.CalledProcessError)
            else str(exc)
        )
        sys.exit(f"error: {detail}")
