"""Fetch the pinned serving commits before starting the ten-hour session."""

import argparse
import subprocess

import session10


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--container", default="sglang-a3")
    parser.add_argument("--repo", default="/home/swx1199799/megamoe_test/sglang")
    parser.add_argument(
        "--fetch-url",
        default="https://ghfast.top/https://github.com/TamirBaydasov/sglang.git",
    )
    parser.add_argument("--dry-run", action="store_true")
    args, _ = parser.parse_known_args(argv)
    if args.dry_run:
        return 0
    prefix = ["docker", "exec", "-w", args.repo, args.container, "git"]
    subprocess.run(
        prefix
        + [
            "fetch",
            "--no-tags",
            args.fetch_url,
            "refs/heads/npu-w4a8-tp-bias-fix:refs/remotes/ttft10/npu-w4a8-tp-bias-fix",
            "refs/heads/npu-bde-combined-ttft10-f1:refs/remotes/ttft10/npu-bde-combined-ttft10-f1",
            "refs/heads/npu-bde-combined-ep-ttft10:refs/remotes/ttft10/npu-bde-combined-ep-ttft10",
        ],
        check=True,
        timeout=240,
    )
    for commit in (session10.BASE, session10.TP, session10.BUFFERED_TP, session10.EP):
        subprocess.run(
            prefix + ["cat-file", "-e", commit + "^{commit}"], check=True, timeout=30
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
