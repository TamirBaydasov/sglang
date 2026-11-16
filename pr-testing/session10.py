"""Ten-hour combined TTFT experiment; exact sources and fresh speed samples."""

import copy
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import run_pr_tests as h
import session16 as previous

ORIGINAL_CASES = copy.deepcopy(h.CASES)
ROOT = Path(__file__).resolve().parent
TP = "4eb600357bf7f8ae98406c0fca6d206ae715408c"
EP = "8e219cba5f16ef00c1cc67beab3a1ef309315fc5"
BUFFERED_TP = "39375d04d004db540f241603515b28a85a0b3a84"
BASE = "aa32b34a4aec8b6bea740d20d520e3f6a4ff03d6"
CONTROLS = {"24-historical-control", "33-best-ttft"}
CANDIDATES = (
    "33-best-ttft",
    "35-current-f1",
    "36-buffered-f1",
    "27-combined-ep",
    "28-combined-megamoe",
)
PLAN = ("24-historical-control", *CANDIDATES)


def register_cases():
    h.CASES = copy.deepcopy(ORIGINAL_CASES)
    table = previous.register_cases()
    best = dict(table["33-best-ttft"][2])
    extra = table["33-best-ttft"][4]
    cases = []
    for name in PLAN:
        env = dict(table["24-historical-control"][2] if name == PLAN[0] else best)
        ref = (
            BASE if name == PLAN[0] else BUFFERED_TP if name == "36-buffered-f1" else TP
        )
        server = extra
        if name in ("35-current-f1", "36-buffered-f1"):
            env["SGLANG_NPU_SFA_KV_8BIT"] = "True"
            env["SGLANG_NPU_SFA_KV_QUANT_READ"] = "False"
            if name == "36-buffered-f1":
                env["SGLANG_NPU_SFA_F1_PREALLOC"] = "True"
        if name in ("27-combined-ep", "28-combined-megamoe"):
            ref = EP
            server += (
                "--ep-size",
                "16",
                "--moe-a2a-backend",
                "deepep" if name == "27-combined-ep" else "ascend_fuseep",
            )
            if name == "28-combined-megamoe":
                server += ("--fuseep-mode", "3")
                env["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK"] = "2048"
            for flag in (
                "SGLANG_NPU_DSA_CP_TOKEN_SHARDED",
                "SGLANG_NPU_DSA_CP_Q_FULL_WIDTH",
                "SGLANG_NPU_STREAM_O_PROJ",
                "SGLANG_NPU_STREAM_Q_B",
                "SGLANG_NPU_STREAM_W_KC",
                "SGLANG_NPU_STREAM_W_VC",
            ):
                env[flag] = "False"
        cases.append(
            (name, ref, env, "SESSION10.md; all request limits are one", server)
        )
        h.CASE_PROFILES[name] = dict(
            chunk=32768,
            max_prefill_tokens=16384,
            context=1048576,
            memfrac=0.8,
            decode_graph="off",
            allow_server_queueing=True,
        )
        h.MATCHED_BASELINES[name] = "24-historical-control"
    h.CASES = cases
    return {c[0]: c for c in cases}


def install_hooks():
    original_create = h.create_worktree
    original_reuse = h.try_reuse_phase
    original_compatible = h.reusable_phase

    def reuse(args, record, phase, *rest):
        if phase != "accuracy" or record["case"] not in CONTROLS:
            return False
        return original_reuse(args, record, phase, *rest)

    def compatible(candidate, record, *rest):
        if not candidate.get("source_diff_sha256") or candidate[
            "source_diff_sha256"
        ] != record.get("source_diff_sha256"):
            return False
        return original_compatible(candidate, record, *rest)

    h.try_reuse_phase, h.reusable_phase = reuse, compatible
    return original_create, original_reuse, original_compatible


def winner(rows):
    qualified = [
        r
        for r in rows
        if r["case"] in CANDIDATES
        and previous.quality_ok(r)
        and math.isfinite(previous.mean_ttft(r))
    ]
    return min(qualified, key=previous.mean_ttft) if qualified else None


def matching_speed_sources(results):
    signatures = [
        (
            row.get("commit"),
            row.get("source_diff_sha256"),
            h.selected_attempt(row, "speed").get("comparison_config"),
        )
        for row in results
    ]
    return bool(
        signatures
        and all(signatures[0])
        and all(signature == signatures[0] for signature in signatures)
    )


def report(output, rows, attempts):
    h.save(output / "results.json", rows)
    h.save(output / "attempts.json", attempts)
    h.summarize(rows, output, None)
    lines = [
        "# Ten-hour TTFT samples",
        "",
        "64-question five-shot reproduction; five examples overlap. Timing samples are fresh.",
        "",
        "| Case | Accuracy qualified | Fresh mean / P90 samples (s) | Median mean (s) | Evidence |",
        "|---|---|---|---|---|",
    ]
    for name in PLAN:
        row = next((r for r in rows if r["case"] == name), {})
        samples = []
        configs = []
        for result in attempts:
            if result["case"] != name or result.get("phases", {}).get("speed", {}).get(
                "reused_from"
            ):
                continue
            attempt = h.selected_attempt(result, "speed")
            speed = attempt.get("speed", {})
            if speed.get("status") == "ok":
                metrics = speed["metrics"]
                samples.append((metrics["TTFT avg"] / 1000, metrics["TTFT P90"] / 1000))
                configs.append(
                    (
                        result.get("commit"),
                        result.get("source_diff_sha256"),
                        attempt.get("comparison_config"),
                    )
                )
        matching = (
            bool(configs)
            and all(x == configs[0] for x in configs)
            and configs[0][2] is not None
        )
        evidence = (
            "three fresh samples" if matching and len(samples) >= 3 else "preliminary"
        )
        med = (
            f"{statistics.median(x[0] for x in samples):.3f}"
            if samples
            else "unavailable"
        )
        values = (
            "; ".join(f"{mean:.3f} / {p90:.3f}" for mean, p90 in samples)
            or "unavailable"
        )
        lines.append(
            f"| {name} | {previous.quality_ok(row)} | {values} | {med} | {evidence} |"
        )
    controls = [
        previous.mean_ttft(a) / 1000
        for a in attempts
        if a["case"] == "24-historical-control" and math.isfinite(previous.mean_ttft(a))
    ]
    control_rows = [
        a
        for a in attempts
        if a["case"] == "24-historical-control" and math.isfinite(previous.mean_ttft(a))
    ]
    if len(controls) >= 2 and matching_speed_sources(control_rows):
        lines += [
            "",
            f"Opening/closing TP control: {controls[0]:.3f} / {controls[-1]:.3f} s; drift {(controls[-1] / controls[0] - 1) * 100:+.2f}%.",
        ]
    else:
        lines += [
            "",
            "Baseline drift unavailable: matching opening and closing samples are required.",
        ]
    lines += [
        "",
        "Raw phase evidence includes cold-prefix preparation separately, commands, source hashes, runtime, failures and capacity checks.",
    ]
    (output / "SAMPLES.md").write_text("\n".join(lines) + "\n")


def run(argv=None):
    command = list(sys.argv[1:] if argv is None else argv)
    table = register_cases()
    args = h.parser().parse_args(command)
    if (
        args.cases
        or args.suite
        or args.flags_overrides
        or args.stop_only
        or args.regenerate
    ):
        raise ValueError(
            "SESSION10 owns the case matrix; use run_pr_tests.py for custom cases"
        )
    args.cases = ",".join(PLAN)
    h.validate_args(args)
    if args.hours is None or not 0 < args.hours <= 10:
        raise ValueError("--hours must be in (0, 10]")
    if (
        args.extra_server_args
        or args.retry_oom
        or args.skip_baseline_bracket
        or args.accuracy_requests
        or args.moe_mode != "tp"
    ):
        raise ValueError(
            "this matched round forbids server overrides, fallback and skipped controls"
        )
    if (
        args.gsm8k_samples,
        args.gsm8k_parallel,
        args.gsm8k_max_tokens,
        args.requests,
        args.length,
    ) != (64, 16, 1024, 16, 1000000):
        raise ValueError("SESSION10 fixed workload: GSM8K 64/16/1024; AISBench 16 x 1M")
    if args.gsm8k_replay or args.gsm8k_shots_data:
        raise ValueError("SESSION10 uses the unchanged reference GSM8K protocol")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "hours": args.hours,
                    "cleanup_minutes": 20,
                    "plan": PLAN,
                    "accuracy_minutes": 90,
                    "speed_minutes": 20,
                    "reuse": "matching control accuracy only",
                },
                indent=2,
            )
        )
        for name in PLAN:
            effective = h.case_settings(args, name)
            effective.round = "all"
            h.dry_run(effective, [table[name]])
            print(json.dumps({"case": name, "source_commit": table[name][1]}))
        return 0
    if not args.gsm8k:
        raise ValueError("--gsm8k is required")
    h.load_gsm8k(args.gsm8k, 64, args.gsm8k_seed)
    now = time.monotonic()
    started = float(os.environ.get("TTFT10_STARTED_MONOTONIC", now))
    if not math.isfinite(started) or started < 0 or started > now:
        raise ValueError("invalid launcher monotonic start")
    deadline = started + args.hours * 3600
    work_deadline = deadline - min(20 * 60, args.hours * 3600 / 10)
    output = (
        Path(args.output)
        if args.output
        else Path(args.save_dir)
        / "glm-ttft10-results"
        / (time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + h.uuid.uuid4().hex[:6])
    )
    output.mkdir(parents=True, exist_ok=False)
    rows, attempts = [], []
    interrupted = False
    confirmations_complete = False
    originals = install_hooks()
    h.save(
        output / "session-plan.json",
        {
            "settings": vars(args),
            "plan": PLAN,
            "total_seconds": args.hours * 3600,
            "cleanup_minutes": 20,
            "confirmation_minutes": 100,
            "initial_control_minutes_if_reused": 40,
        },
    )

    def checkpoint():
        report(output, rows, attempts)

    def execute(name, phase="all", repeat=None):
        left = work_deadline - time.monotonic()
        if left <= 0:
            return None
        cap = min(110 if phase == "all" else 20, left / 60)
        target = (
            output
            / "runs"
            / (f"{len(attempts):02d}-" + name + (f"-repeat-{repeat}" if repeat else ""))
        )
        child = command + [
            "--cases",
            name,
            "--round",
            phase,
            "--hours",
            str(cap / 60),
            "--case-minutes",
            str(cap),
            "--accuracy-minutes",
            "90",
            "--speed-minutes",
            "20",
            "--baseline-ref",
            BASE,
            "--no-fetch",
            "--fetch-url",
            "",
            "--output",
            str(target),
        ]
        if repeat:
            child = without_reuse(child)
        h.save(output / "next-command.json", {"argv": child})
        code = h.main(child)
        result_file = target / "results.json"
        result = (
            json.loads(result_file.read_text())[0]
            if result_file.exists()
            else {
                "case": name,
                "status": "failed",
                "reason": f"runner exited {code} before results",
                "phases": {},
            }
        )
        result["comparison_baseline"] = "24-historical-control"
        result["comparison_reference"] = name == "24-historical-control"
        result["session_command"] = child
        attempts.append(copy.deepcopy(result))
        existing = next((r for r in rows if r["case"] == name), None)
        if existing and not repeat:
            existing.setdefault("phases", {}).update(result.get("phases", {}))
        elif not existing:
            rows.append(copy.deepcopy(result))
        checkpoint()
        if result.get("stop_failed") or any(
            a.get("cleanup") not in (None, "ok")
            for p in result.get("phases", {}).values()
            for a in p.get("attempts", [])
        ):
            raise RuntimeError("owned cleanup failed; session halted")
        if result.get("status") == "interrupted":
            raise KeyboardInterrupt
        return result

    try:
        pins = h.pin_refs(
            list(table.values()),
            args,
            deadline,
            lambda x: h.save(output / "pinned-refs.json", x),
        )
        if any("error" in x for x in pins.values()):
            raise RuntimeError("source pinning failed; see pinned-refs.json")
        for name in PLAN:
            execute(name)
        selected = winner(rows)
        # Always attempt the closing control; reserve its speed cap before repeats.
        if selected:
            for repeat in (2, 3):
                if work_deadline - time.monotonic() < 40 * 60:
                    break
                execute(selected["case"], "speed", repeat)
        closing = None
        if work_deadline - time.monotonic() > 0:
            closing = execute("24-historical-control", "speed", "closing")
        fresh_count = sum(
            1
            for a in attempts
            if selected
            and a["case"] == selected["case"]
            and math.isfinite(previous.mean_ttft(a))
        )
        winner_samples = [
            a
            for a in attempts
            if selected
            and a["case"] == selected["case"]
            and math.isfinite(previous.mean_ttft(a))
        ]
        confirmations_complete = bool(
            matching_speed_sources(winner_samples)
            and selected
            and fresh_count >= 3
            and closing
            and math.isfinite(previous.mean_ttft(closing))
        )
        h.save(
            output / "confirmation-status.json",
            {
                "winner": selected["case"] if selected else None,
                "fresh_samples": fresh_count,
                "required_fresh_samples": 3,
                "matching_winner_samples": matching_speed_sources(winner_samples),
                "closing_control_complete": bool(
                    closing and math.isfinite(previous.mean_ttft(closing))
                ),
                "status": "ok" if confirmations_complete else "incomplete",
                "reason": ""
                if confirmations_complete
                else "budget exhausted or no qualified candidate; missing repetitions remain unexecuted",
            },
        )
    except (KeyboardInterrupt, SystemExit):
        interrupted = True
    except (ValueError, OSError, RuntimeError) as exc:
        h.save(output / "session-error.json", {"error": f"{type(exc).__name__}: {exc}"})
    finally:
        h.create_worktree, h.try_reuse_phase, h.reusable_phase = originals
        names = {r["case"] for r in rows}
        for name in PLAN:
            if name not in names:
                rows.append(
                    {
                        "case": name,
                        "status": "skipped",
                        "reason": "session interrupted, failed or budget exhausted",
                        "phases": {},
                    }
                )
        h.save(
            output / "run-status.json",
            {
                "interrupted": interrupted,
                "elapsed_s": time.monotonic() - started,
                "budget_seconds": args.hours * 3600,
                "device_validation": "inspect A3 phase evidence",
            },
        )
        checkpoint()
        print("Session results:", output)
    return int(
        interrupted
        or not confirmations_complete
        or (output / "session-error.json").exists()
        or any(r.get("status") != "ok" for r in rows)
    )


def without_reuse(command):
    result = []
    i = 0
    while i < len(command):
        if command[i] == "--reuse-results":
            i += 2
        else:
            result.append(command[i])
            i += 1
    return result


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
