"""Three isolated indexer/SFA speed cases, with separately labelled short profiles."""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import run_pr_tests as h
from summarize_sfa_profile import summarize

ROOT = Path(__file__).resolve().parent
SOURCE = "39375d04d004db540f241603515b28a85a0b3a84"
PLAN = ("40-indexer-int8", "41-indexer-f1-buffered", "42-indexer-f2-original")
DISABLED = (
    "SGLANG_ENABLE_CHAT_ENCODE_POOL",
    "SGLANG_CHAT_ENCODE_MEMO",
    "SGLANG_CHAT_ENCODE_RAW",
    "SGLANG_NPU_ENABLE_DCP_PACKED_READ",
    "SGLANG_NPU_ENABLE_DCP_PACKED_GATHER_PREFETCH",
    "SGLANG_NPU_ENABLE_PG_PRIORITY",
    "SGLANG_NPU_DSA_CP_TOKEN_SHARDED",
    "SGLANG_NPU_DSA_CP_Q_FULL_WIDTH",
    "SGLANG_NPU_DSA_CP_Q_A2A_PREISSUE",
    "SGLANG_NPU_DCP_GATHER_BUF_ROUND",
    "SGLANG_NPU_ENABLE_DCP_EXTEND_GATHER_PREFETCH",
    "SGLANG_NPU_DCP_PAGE_INTERLEAVE",
    "SGLANG_NPU_DSA_CP_LOCAL_TOPK",
    "SGLANG_NPU_DCP_DECODE_Q_PREISSUE",
    "SGLANG_NPU_STREAM_O_PROJ",
    "SGLANG_NPU_STREAM_Q_B",
    "SGLANG_NPU_STREAM_W_KC",
    "SGLANG_NPU_STREAM_W_VC",
    "SGLANG_NPU_ENABLE_DCP_SFA_GATHER_OVERLAP",
)


def register():
    cases = []
    for name in PLAN:
        env = dict.fromkeys(DISABLED, "False")
        env.update(
            SGLANG_NPU_INDEXER_INT8="True",
            SGLANG_NPU_SFA_KV_8BIT="False" if name == PLAN[0] else "True",
            SGLANG_NPU_SFA_KV_QUANT_READ="True" if name == PLAN[2] else "False",
            SGLANG_NPU_SFA_F1_PREALLOC="True" if name == PLAN[1] else "False",
            SGLANG_DEBUG_NPU_DCP_EXTEND_MEMORY="False",
            HCCL_BUFFSIZE="256",
            SGLANG_NPU_DCP_GATHER_CHUNKS="1",
        )
        cases.append(
            (
                name,
                SOURCE,
                env,
                "SFA2H.md; speed only; quality unverified",
                (
                    "--max-running-requests",
                    "1",
                    "--prefill-max-requests",
                    "1",
                    "--max-total-tokens",
                    "63488",
                ),
            )
        )
        h.CASE_PROFILES[name] = dict(
            chunk=32768,
            max_prefill_tokens=16384,
            context=1048576,
            memfrac=0.80,
            decode_graph="off",
            allow_server_queueing=True,
        )
        h.MATCHED_BASELINES[name] = PLAN[0]
    h.CASES = cases


def install_hooks():
    original_create, original_ais = h.create_worktree, h.run_aisbench
    patch = (ROOT / "sfa2h-serving.diff").read_text()

    def create(args, record, registry, deadline):
        path = original_create(args, record, registry, deadline)
        h.git(
            args,
            "apply",
            "--check",
            "-",
            repo=path,
            deadline=deadline,
            input_text=patch,
        )
        h.git(args, "apply", "-", repo=path, deadline=deadline, input_text=patch)
        diff = h.git(args, "diff", "--binary", repo=path, deadline=deadline)
        record["source_diff_sha256"] = hashlib.sha256(diff.encode()).hexdigest()
        record["source_overlay_sha256"] = hashlib.sha256(patch.encode()).hexdigest()
        return path

    def ais(args, case_dir, deadline, server, state_path, remote_log, log, commit):
        measured = original_ais(
            args, case_dir, deadline, server, state_path, remote_log, log, commit
        )
        h.save(case_dir / "unprofiled-speed.json", measured)
        name = case_dir.parent.parent.name
        text = (case_dir / "server.log").read_text(errors="replace")
        evidence = {
            "f1_buffer_active": "SFA F1 prepared read ACTIVE" in text,
            "f2_active": "SFA F2 quant sparse attention ACTIVE" in text,
            "f2_fallback": "SFA F2 FALLBACK to F1" in text,
        }
        measured["activation"] = evidence
        if name == PLAN[1] and not evidence["f1_buffer_active"]:
            measured.update(
                status="failed",
                reason="buffered F1 activation unverified; diagnostic timing only",
            )
        if name == PLAN[2] and (not evidence["f2_active"] or evidence["f2_fallback"]):
            measured.update(
                status="failed",
                reason="F2 inactive or fell back to F1; diagnostic timing only",
            )
        profile_dir = case_dir / "profiling"
        profile_dir.mkdir()
        remote = args.remote_root + "/profiles/" + name
        profile = {
            "status": "failed",
            "reason": "profiling incomplete",
            "steps": 3,
            "remote_trace_dir": remote,
            "timings_comparable": False,
        }
        measured["profiling"] = profile
        try:
            if deadline - time.monotonic() < 180:
                raise RuntimeError(
                    "less than three minutes remain for requested profiling"
                )
            # A separate invocation keeps profiler overhead out of the speed sample.
            captured = original_ais(
                args,
                profile_dir,
                deadline,
                server,
                state_path,
                remote_log,
                log,
                commit,
                profile=dict(
                    PROFILE="1",
                    PROFILE_PHASE="2",
                    PROFILE_STEPS="3",
                    PROFILE_TRACE_DIR=remote,
                    WITH_STACK="0",
                    RECORD_SHAPES="0",
                ),
            )
            h.save(profile_dir / "profiled-diagnostic.json", captured)
            trace = profile_dir / "trace"
            trace.mkdir()
            copied = h.execute(
                ["docker", "cp", args.container + ":" + remote + "/.", str(trace)],
                deadline,
                120,
                log_path=profile_dir / "trace-export.log",
            )
            if copied.returncode:
                raise RuntimeError("profile trace export failed; see trace-export.log")
            summary = summarize(trace)
            h.save(profile_dir / "kernel-summary.json", summary)
            profile.update(summary=summary, status="ok", reason="")
            if captured["status"] != "ok":
                raise RuntimeError("profiled requests failed: " + captured["reason"])
            if not summary["indexer_int8_observed"]:
                raise RuntimeError("INT8 indexer kernel absent from exported profile")
            if name == PLAN[2] and not summary["f2_observed"]:
                raise RuntimeError("F2 kernel absent from exported profile")
        except Exception as exc:
            profile.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
            measured.update(
                status="failed",
                reason=(
                    measured.get("reason", "") + "; short profile: " + str(exc)
                ).strip("; "),
            )
        finally:
            h.save(case_dir / "speed-with-profile.json", measured)
        return measured

    h.create_worktree, h.run_aisbench = create, ais


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    register()
    install_hooks()
    # Fixed matrix prevents accidental accuracy runs or workload drift.
    forbidden = {
        "--cases",
        "--round",
        "--suite",
        "--reuse-results",
        "--flags-overrides",
        "--extra-server-args",
        "--hours",
        "--case-minutes",
        "--speed-minutes",
        "--moe-mode",
        "--memfrac",
        "--chunk",
        "--context",
        "--requests",
        "--length",
        "--max-prefill-tokens",
        "--decode-graph",
        "--fallback-memfracs",
        "--retry-oom",
        "--skip-baseline-bracket",
        "--diagnose-baseline",
        "--p4-reference",
    }
    if any(arg.split("=", 1)[0] in forbidden for arg in argv):
        print(
            "ERROR: fixed two-hour speed matrix; configuration overrides are disabled",
            file=sys.stderr,
        )
        return 2
    started = float(os.environ.get("SFA2H_STARTED_MONOTONIC", time.monotonic()))
    now = time.monotonic()
    if not 0 <= started <= now:
        raise ValueError("invalid monotonic launch time")
    seconds = 7200 - (now - started) - 180
    if seconds <= 0:
        print("ERROR: two-hour budget exhausted during preparation", file=sys.stderr)
        return 1
    if "--output" not in argv:
        parsed, _ = h.parser().parse_known_args(argv)
        argv += [
            "--output",
            str(
                Path(parsed.save_dir)
                / "glm-sfa2h-results"
                / (
                    time.strftime("%Y%m%d-%H%M%S", time.gmtime())
                    + "-"
                    + h.uuid.uuid4().hex[:6]
                )
            ),
        ]
    if "--dry-run" in argv:
        print(
            "Separate short profile after each unprofiled speed run: "
            "PROFILE=1 PROFILE_PHASE=2 PROFILE_STEPS=3 WITH_STACK=0 RECORD_SHAPES=0"
        )
        print(
            "Trace output: <owned-run-root>/profiles/<case>; exported to profiling/trace"
        )
        print(
            "Serving overlay SHA256: "
            + hashlib.sha256((ROOT / "sfa2h-serving.diff").read_bytes()).hexdigest()
        )
    code = h.main(
        argv
        + [
            "--cases",
            ",".join(PLAN),
            "--round",
            "speed",
            "--no-fetch",
            "--fetch-url",
            "",
            "--hours",
            str(seconds / 3600),
            "--case-minutes",
            "38",
            "--speed-minutes",
            "36",
            "--benchmark-timeout",
            "900",
            "--decode-graph",
            "off",
        ]
    )
    output = Path(h.parser().parse_known_args(argv)[0].output)
    if (output / "results.json").exists():
        lines = [
            "# Indexer / SFA speed comparison",
            "",
            "No accuracy testing in this round. All timings are single-run diagnostics.",
            "",
            "| Case | Unprofiled mean / P90 TTFT (s) | Activation | Short profile |",
            "|---|---|---|---|",
        ]
        for row in json.loads((output / "results.json").read_text()):
            speed = h.selected_attempt(row, "speed").get("speed", {})
            metrics = speed.get("metrics") or {}
            timing = (
                f"{metrics['TTFT avg'] / 1000:.3f} / {metrics['TTFT P90'] / 1000:.3f}"
                if metrics
                else "unavailable"
            )
            lines.append(
                f"| {row['case']} | {timing} | {speed.get('activation', {})} | "
                f"{speed.get('profiling', {}).get('status', 'incomplete')} |"
            )
        (output / "SFA2H_RESULTS.md").write_text("\n".join(lines) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
