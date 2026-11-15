"""Bounded GLM follow-up: pin once, reuse validated phases, and retain every outcome."""

import argparse
import copy
import json
import math
import statistics
import sys
import time
from pathlib import Path

import run_pr_tests as h

BLOCKS = [
    ("diagnostics", 120),
    ("individuals", 240),
    ("historical", 100),
    ("combined", 320),
    ("confirmation", 120),
    ("reserve", 60),
]


def register_cases():
    table = {row[0]: row for row in h.CASES}
    full = dict(table["07-combined"][2])
    common = {"HCCL_BUFFSIZE": "256", "SGLANG_DEBUG_NPU_DCP_EXTEND_MEMORY": "False"}
    historical = (
        "--max-running-requests",
        "1",
        "--prefill-max-requests",
        "1",
        "--max-total-tokens",
        "63488",
    )
    definitions = [
        ("24-historical-control", "npu-w4a8-tp-bias-fix", common, historical),
        ("25-combined-full", "npu-bde-combined-v2", dict(full, **common), historical),
        (
            "26-combined-no-sfa",
            "npu-bde-combined-v2",
            dict(
                full,
                **common,
                SGLANG_NPU_SFA_KV_8BIT="False",
                SGLANG_NPU_SFA_KV_QUANT_READ="False",
            ),
            historical,
        ),
        (
            "27-combined-ep",
            "npu-bde-combined-ep-v2",
            dict(full, **common),
            historical + ("--ep-size", "16", "--moe-a2a-backend", "deepep"),
        ),
        (
            "28-combined-megamoe",
            "npu-bde-combined-ep-v2",
            dict(full, **common, SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK="1024"),
            historical
            + (
                "--ep-size",
                "16",
                "--moe-a2a-backend",
                "ascend_fuseep",
                "--fuseep-mode",
                "3",
            ),
        ),
        (
            "29-tp-breakable",
            "npu-w4a8-tp-bias-fix",
            common,
            historical + ("--cuda-graph-backend-prefill", "breakable"),
        ),
        (
            "30-sfa-f1-probe",
            "npu-sfa-kv-8bit",
            {"SGLANG_NPU_SFA_KV_8BIT": "True", "SGLANG_NPU_SFA_KV_QUANT_READ": "False"},
            (),
        ),
        (
            "31-packed-single-chunk",
            "npu-dcp-packed-read",
            dict(table["03-pr-prefetch-packed"][2], SGLANG_NPU_DCP_GATHER_CHUNKS="1"),
            ("--prefill-max-requests", "1"),
        ),
    ]
    for name, ref, env, extra in definitions:
        table[name] = (
            name,
            ref,
            env,
            "session16 explicit diagnostic; see SESSION16.md",
            extra,
        )
        h.CASE_PROFILES[name] = {"decode_graph": "off"}
        if name[:2] in ("24", "25", "26", "27", "28", "29"):
            h.CASE_PROFILES[name].update(
                chunk=32768, context=1048576, memfrac=0.8, allow_server_queueing=True
            )
        if name in ("30-sfa-f1-probe", "31-packed-single-chunk"):
            h.CASE_PROFILES[name].update(
                round="accuracy",
                gsm8k_samples=4,
                gsm8k_min_accuracy=1.0,
                accuracy_minutes=15,
                speed_minutes=None,
                case_minutes=15,
            )
    # Explicit memory diagnostic; a changed recipe never inherits old accuracy.
    h.CASE_PROFILES["05-pr-ts-token-sharded"] = {
        "chunk": 8192,
        "max_prefill_tokens": 8192,
    }
    row = table["05-pr-ts-token-sharded"]
    table[row[0]] = (*row[:4], ("--prefill-max-requests", "1"))
    row = table["06-pr-moe-int8"]
    table[row[0]] = (*row[:4], ("--prefill-max-requests", "1"))
    h.MATCHED_BASELINES["06-pr-moe-int8"] = "18-baseline-prefill1"
    h.CASES = list(table.values())
    h.MATCHED_BASELINES.update(
        {
            name: "24-historical-control"
            for name in (
                "25-combined-full",
                "26-combined-no-sfa",
                "27-combined-ep",
                "28-combined-megamoe",
                "29-tp-breakable",
            )
        }
    )
    return table


def quality_ok(row):
    attempt = h.selected_attempt(row, "accuracy")
    gsm = attempt.get("gsm8k", {})
    details = gsm.get("details", [])
    return (
        row.get("phases", {}).get("accuracy", {}).get("status") == "ok"
        and gsm.get("requested") == 64
        and gsm.get("completed") == 64
        and len(details) == 64
        and not any(x.get("error") or x.get("truncated") for x in details)
        and sum(bool(x.get("correct")) for x in details) >= 58
    )


def mean_ttft(row):
    attempt = h.selected_attempt(row, "speed")
    speed = attempt.get("speed", {})
    return (
        float((speed.get("metrics") or {}).get("TTFT avg", math.inf))
        if speed.get("status") == "ok"
        else math.inf
    )


def select_tp(rows):
    choices = [
        r
        for r in rows
        if r["case"] in ("25-combined-full", "26-combined-no-sfa") and quality_ok(r)
    ]
    return min(choices, key=mean_ttft) if choices else None


def target_report(output, rows, attempts):
    lines = [
        "# Target qualification",
        "",
        "Fresh-server samples only; historical timings are not accuracy qualification.",
        "",
        "| Case | Accuracy qualified | Fresh samples (s) | Median mean TTFT (s) | Target (s) | Result |",
        "|---|---|---|---|---|---|",
    ]
    for name, target in [
        ("25-combined-full", 24),
        ("26-combined-no-sfa", 24),
        ("27-combined-ep", 20),
        ("28-combined-megamoe", 20),
    ]:
        row = next((r for r in rows if r["case"] == name), {})
        samples, configs, commits = [], [], []
        for attempt in attempts:
            if attempt["case"] != name or attempt.get("phases", {}).get(
                "speed", {}
            ).get("reused_from"):
                continue
            value = mean_ttft(attempt)
            if math.isfinite(value):
                samples.append(value / 1000)
                configs.append(
                    h.selected_attempt(attempt, "speed").get("comparison_config")
                )
                commits.append(attempt.get("commit"))
        qualified = quality_ok(row)
        matching = (
            bool(configs)
            and configs[0] is not None
            and all(c == configs[0] for c in configs)
            and all(c == commits[0] for c in commits)
        )
        median = statistics.median(samples) if samples else None
        status = "pending/unqualified"
        if qualified and matching and len(samples) >= 3:
            status = "met" if median <= target else "not met"
        lines.append(
            f"| {name} | {qualified} | {', '.join(f'{x:.3f}' for x in samples)} | {f'{median:.3f}' if median is not None else 'unavailable'} | {target} | {status} |"
        )
    lines += [
        "",
        "EP/MegaMoE candidates disable TP-only residual sharding and streamed projection features. Full all-2–7 EP is not claimed.",
    ]
    (output / "TARGETS.md").write_text("\n".join(lines) + "\n")


def task_plan():
    return {
        "diagnostics": [
            ("30-sfa-f1-probe", "accuracy", 15),
            ("31-packed-single-chunk", "accuracy", 15),
            ("24-historical-control", "speed", 20),
            ("29-tp-breakable", "all", 60),
        ],
        "individuals": [
            (name, "all", 80)
            for name in [
                "00-baseline",
                "14-baseline-ep",
                "09-pr1-megamoe",
                "01-pr-e-chat-encode",
                "02-pr-b-indexer-int8",
                "18-baseline-prefill1",
                "03-pr-prefetch-packed",
                "04-pr-f-sfa-kv-8bit",
                "05-pr-ts-token-sharded",
                "06-pr-moe-int8",
            ]
        ],
        "historical": [("24-historical-control", "all", 80)],
        "combined": [
            (name, "all", 80)
            for name in [
                "25-combined-full",
                "26-combined-no-sfa",
                "27-combined-ep",
                "28-combined-megamoe",
            ]
        ],
        "confirmation": [],
        "reserve": [],
    }


def run(argv=None):
    outer = argparse.ArgumentParser(add_help=False)
    outer.add_argument("--only", default="")
    outer.add_argument(
        "--resume", help="prior session directory, copied into a fresh output"
    )
    special, common = outer.parse_known_args(argv)
    table = register_cases()
    args = h.parser().parse_args(common)
    h.validate_args(args)
    h.CASES = [
        (
            name,
            args.baseline_ref
            if ref in ("dcp_ascend_testing", "npu-w4a8-tp-bias-fix")
            else ref,
            env,
            note,
            extra,
        )
        for name, ref, env, note, extra in h.CASES
    ]
    table = {case[0]: case for case in h.CASES}
    if args.hours > 16:
        raise ValueError("session16 has a 16-hour maximum")
    only = set(special.only.split(",")) if special.only else set()
    if only - set(table):
        raise ValueError("unknown --only case")
    if special.resume:
        args.reuse_results.append(str(Path(special.resume) / "results.json"))
    plan = task_plan()
    if only:
        plan = {"combined": [(name, "all", 80) for name in sorted(only)]}
    if args.dry_run:
        print(
            json.dumps(
                {
                    "block_minutes": dict(BLOCKS),
                    "total_minutes": sum(x[1] for x in BLOCKS),
                    "tasks": plan,
                    "accuracy_minutes": args.accuracy_minutes,
                    "speed_minutes": args.speed_minutes,
                    "gates": "EP/MegaMoE require qualified combined TP; TP-only residual features disabled explicitly",
                },
                indent=2,
            )
        )
        for tasks in plan.values():
            for name, phase, minutes in tasks:
                effective = h.case_settings(args, name)
                effective.round = h.CASE_PROFILES.get(name, {}).get("round", phase)
                h.dry_run(effective, [table[name]])
        return 0
    if not args.gsm8k:
        raise ValueError("--gsm8k is required")
    h.load_gsm8k(args.gsm8k, 64, args.gsm8k_seed)
    output = (
        Path(args.output)
        if args.output
        else Path(args.save_dir)
        / "glm-session-results"
        / time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    )
    output.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + args.hours * 3600
    rows, attempts, interrupted = [], [], False
    settings = vars(args).copy()
    h.save(
        output / "session-plan.json",
        {"settings": settings, "blocks": BLOCKS, "tasks": plan},
    )
    # Resolve every branch once, before any server starts. Every child uses SHAs.
    pins = h.pin_refs(
        list(table.values()),
        args,
        deadline,
        lambda data: h.save(output / "pinned-refs.json", data),
    )
    if any("error" in p for p in pins.values()):
        h.save(output / "pin-errors.json", pins)
        for tasks in plan.values():
            for name, _, _ in tasks:
                if not any(row["case"] == name for row in rows):
                    rows.append(
                        {
                            "case": name,
                            "status": "blocked",
                            "reason": "source pinning failed; see pin-errors.json",
                            "phases": {},
                        }
                    )
        h.save(output / "results.json", rows)
        h.summarize(rows, output, args)
        return 1
    h.CASES = [
        (n, pins[r]["commit"], e, note, extra) for n, r, e, note, extra in h.CASES
    ]
    table = {c[0]: c for c in h.CASES}
    base_ref = args.baseline_ref
    if base_ref in pins:
        base_ref = pins[base_ref]["commit"]
    reused = list(args.reuse_results)

    def checkpoint():
        h.save(output / "results.json", rows)
        h.save(output / "attempts.json", attempts)
        h.summarize(rows, output, args)
        target_report(output, rows, attempts)

    def execute(name, phase, cap, block_deadline, *, fresh=False, suffix=""):
        left = min(deadline, block_deadline) - time.monotonic()
        if left <= 0:
            rows.append(
                {
                    "case": name + suffix,
                    "status": "skipped",
                    "reason": "session/block budget exhausted",
                    "phases": {},
                }
            )
            checkpoint()
            return None
        cap = min(cap, left / 60)
        child_dir = output / "runs" / (f"{len(attempts):02d}-" + name + suffix)
        command = list(common)
        # Later arguments replace launcher defaults through argparse.
        command += [
            "--cases",
            name,
            "--round",
            phase,
            "--hours",
            str(cap / 60),
            "--case-minutes",
            str(cap),
            "--accuracy-minutes",
            "60",
            "--speed-minutes",
            "20",
            "--no-fetch",
            "--baseline-ref",
            base_ref,
            "--output",
            str(child_dir),
        ]
        if not fresh:
            for source in reused:
                command += ["--reuse-results", source]
        else:
            # Fresh confirmations must never reuse previously measured timings.
            index = 0
            while index < len(command):
                if command[index] == "--reuse-results":
                    del command[index : index + 2]
                else:
                    index += 1
        h.save(
            output / "next-command.json",
            {"python_api": "run_pr_tests.main", "argv": command},
        )
        code = h.main(command)
        result_path = child_dir / "results.json"
        if not result_path.exists():
            result = {
                "case": name,
                "status": "failed",
                "reason": f"runner exited {code} before results",
                "phases": {},
            }
        else:
            result = json.loads(result_path.read_text())[0]
            reused.append(str(result_path))
        result["session_name"] = name + suffix
        result["session_command"] = command
        attempts.append(copy.deepcopy(result))
        existing = next((r for r in rows if r["case"] == name), None)
        if suffix:
            confirmation = copy.deepcopy(result)
            confirmation["case"] = name + suffix
            rows.append(confirmation)
        elif existing:
            existing.setdefault("phases", {}).update(result.get("phases", {}))
            existing["status"] = (
                "ok"
                if all(p.get("status") == "ok" for p in existing["phases"].values())
                else "failed"
            )
        else:
            rows.append(result)
        checkpoint()
        if result.get("stop_failed") or any(
            a.get("cleanup") not in (None, "ok")
            for p in result.get("phases", {}).values()
            for a in p.get("attempts", [])
        ):
            raise RuntimeError("cleanup failed; session halted")
        return result

    carry = 0.0
    try:
        for block, minutes in BLOCKS:
            if only and block != "combined":
                continue
            start = time.monotonic()
            block_deadline = min(deadline, start + minutes * 60 + carry)
            tasks = list(plan.get(block, []))
            if block == "confirmation":
                best_tp = select_tp(rows)
                ep_choices = [
                    r
                    for r in rows
                    if r["case"] in ("27-combined-ep", "28-combined-megamoe")
                    and quality_ok(r)
                    and math.isfinite(mean_ttft(r))
                ]
                selected = [r for r in rows if r["case"] == "24-historical-control"]
                if best_tp:
                    selected.append(best_tp)
                if ep_choices:
                    selected.append(min(ep_choices, key=mean_ttft))
                for row in selected:
                    if quality_ok(row) and math.isfinite(mean_ttft(row)):
                        tasks.extend([(row["case"], "speed", 20)] * 2)
            for index, (name, phase, cap) in enumerate(tasks):
                if name in ("27-combined-ep", "28-combined-megamoe"):
                    best = select_tp(rows)
                    if not best:
                        rows.append(
                            {
                                "case": name,
                                "status": "blocked",
                                "reason": "no accuracy-qualified combined TP configuration",
                                "phases": {},
                            }
                        )
                        checkpoint()
                        continue
                    env = table[name][2]
                    env["SGLANG_NPU_SFA_KV_8BIT"] = table[best["case"]][2].get(
                        "SGLANG_NPU_SFA_KV_8BIT", "False"
                    )
                    env["SGLANG_NPU_SFA_KV_QUANT_READ"] = table[best["case"]][2].get(
                        "SGLANG_NPU_SFA_KV_QUANT_READ", "False"
                    )
                    # The existing TP residual implementation rejects EP by design.
                    for flag in (
                        "SGLANG_NPU_DSA_CP_TOKEN_SHARDED",
                        "SGLANG_NPU_DSA_CP_Q_FULL_WIDTH",
                        "SGLANG_NPU_STREAM_O_PROJ",
                        "SGLANG_NPU_STREAM_Q_B",
                        "SGLANG_NPU_STREAM_W_KC",
                        "SGLANG_NPU_STREAM_W_VC",
                    ):
                        env[flag] = "False"
                if name == "04-pr-f-sfa-kv-8bit":
                    probe = next(
                        (r for r in rows if r["case"] == "30-sfa-f1-probe"), {}
                    )
                    if (
                        h.selected_attempt(probe, "accuracy")
                        .get("gsm8k", {})
                        .get("correct")
                        != 4
                    ):
                        rows.append(
                            {
                                "case": name,
                                "status": "blocked",
                                "reason": "four-prompt F1 correctness probe failed",
                                "phases": {},
                            }
                        )
                        checkpoint()
                        continue
                execute(
                    name,
                    phase,
                    cap,
                    block_deadline,
                    fresh=block == "confirmation",
                    suffix=f"-repeat-{index + 1}" if block == "confirmation" else "",
                )
                if any(r.get("status") == "interrupted" for r in rows):
                    interrupted = True
                    break
            carry = max(0.0, block_deadline - time.monotonic())
            if interrupted:
                break
    except (KeyboardInterrupt, SystemExit):
        interrupted = True
    finally:
        existing_names = {r["case"] for r in rows}
        for tasks in plan.values():
            for name, _, _ in tasks:
                if name not in existing_names:
                    rows.append(
                        {
                            "case": name,
                            "status": "skipped",
                            "reason": "session interrupted or budget exhausted",
                            "phases": {},
                        }
                    )
                    existing_names.add(name)
        h.save(
            output / "run-status.json",
            {
                "interrupted": interrupted,
                "device_validation": "inspect per-case evidence; no target guaranteed",
            },
        )
        checkpoint()
        print("Session results:", output)
    return 1 if interrupted or any(r.get("status") != "ok" for r in rows) else 0


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
