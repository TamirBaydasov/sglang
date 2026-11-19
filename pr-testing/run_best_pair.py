"""Pinned TP/MegaMoE best recipes; four-hour fresh quality and speed round."""

import argparse, copy, json, math, os, signal, sys, time
from pathlib import Path
import run_pr_tests as h

ROOT = Path(__file__).resolve().parent


def run(argv=None):
    raw = list(sys.argv[1:] if argv is None else argv)
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--tp-ref")
    selector.add_argument("--ep-ref")
    refs, command = selector.parse_known_args(raw)
    recipes = json.loads((ROOT / "best-pair-recipes.json").read_text())
    table = []
    for name, override in [
        ("33-best-ttft", refs.tp_ref),
        ("28-combined-megamoe", refs.ep_ref),
    ]:
        r = recipes[name]
        table.append(
            (
                name,
                override or r["source"],
                r["env"],
                r["provenance"],
                tuple(r["server_args"]),
            )
        )
        h.CASE_PROFILES[name] = dict(
            chunk=32768,
            max_prefill_tokens=16384,
            context=1048576,
            memfrac=0.8,
            decode_graph="off",
            allow_server_queueing=True,
        )
    h.CASES = table
    h.MATCHED_BASELINES["28-combined-megamoe"] = "33-best-ttft"
    parser = h.parser()
    parser.set_defaults(hours=4, gsm8k_timeout=1800)
    args = parser.parse_args(command)
    if (
        args.cases
        or args.suite
        or args.flags_overrides
        or args.extra_server_args
        or args.reuse_results
        or args.retry_oom
        or args.gsm8k_replay
        or args.moe_mode != "tp"
        or args.stop_only
        or args.regenerate
        or args.diagnose_baseline
    ):
        raise ValueError("pair runner owns cases, source flags and fresh phases")
    if not 0 < args.hours <= 4:
        raise ValueError("--hours must be in (0,4]")
    if (
        args.gsm8k_samples,
        args.gsm8k_parallel,
        args.gsm8k_max_tokens,
        args.requests,
        args.length,
        args.server_seed,
        args.gsm8k_seed,
    ) != (64, 16, 1024, 16, 1000000, 4, 4):
        raise ValueError("fixed workload required")
    args.cases = ",".join(x[0] for x in table)
    h.validate_args(args)
    args.cases = None
    if args.dry_run:
        for case in table:
            for phase in ("accuracy", "speed"):
                effective = h.case_settings(args, case[0])
                effective.round = phase
                effective.case_minutes = (
                    (75 if case[0] == "33-best-ttft" else 120)
                    if phase == "accuracy"
                    else 15
                )
                h.dry_run(effective, [case])
        return 0
    if not args.gsm8k:
        raise ValueError("--gsm8k required")
    h.load_gsm8k(args.gsm8k, 64, 4)
    now = time.monotonic()
    started = float(os.environ.get("BESTPAIR_STARTED_MONOTONIC", now))
    if not math.isfinite(started) or started < 0 or started > now:
        raise ValueError("invalid start clock")
    end = started + args.hours * 3600
    work_end = end - 900
    output = (
        Path(args.output)
        if args.output
        else Path(args.save_dir)
        / "glm-best-pair-results"
        / (time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + h.uuid.uuid4().hex[:6])
    )
    output.mkdir(parents=True, exist_ok=False)
    rows = {
        c[0]: dict(
            case=c[0],
            ref=c[1],
            case_env=c[2],
            case_server_args=list(c[4]),
            phases={},
            status="pending",
        )
        for c in table
    }
    queue = []
    attempts = []
    errors = []
    original_probe = h.probe_runtime
    identity = []

    def probe(*a, **kw):
        info = original_probe(*a, **kw)
        signature = {
            k: info.get(k) for k in ("versions", "model_files_sha256", "model_config")
        }
        if identity and signature != identity[0]:
            raise RuntimeError("runtime/checkpoint changed")
        if not identity:
            identity.append(signature)
        return info

    h.probe_runtime = probe

    def checkpoint():
        h.save(output / "results.json", list(rows.values()))
        h.save(output / "queue.json", queue)
        h.save(output / "attempts.json", attempts)
        try:
            h.summarize(list(rows.values()), output, args)
        except Exception as exc:
            h.save(output / "report-error.json", dict(error=repr(exc)))

    try:
        pins = h.pin_refs(
            table,
            args,
            min(work_end, now + 240),
            lambda p: h.save(output / "pins.json", p),
        )
        if any("error" in p for p in pins.values()):
            raise RuntimeError("pinning failed")
        h.CASES = [
            (n, pins[ref]["commit"], env, note, extra)
            for n, ref, env, note, extra in table
        ]
        for name, phase, minutes in [
            ("33-best-ttft", "accuracy", 75),
            ("33-best-ttft", "speed", 15),
            ("28-combined-megamoe", "accuracy", 120),
            ("28-combined-megamoe", "speed", 15),
        ]:
            if work_end - time.monotonic() < minutes * 60:
                queue.append(
                    dict(
                        case=name,
                        phase=phase,
                        status="skipped",
                        reason="insufficient full phase allowance plus cleanup reserve",
                    )
                )
                rows[name]["status"] = "incomplete"
                checkpoint()
                continue
            target = output / "runs" / (name + "-" + phase)
            child = command + [
                "--cases",
                name,
                "--round",
                phase,
                "--hours",
                str(minutes / 60),
                "--case-minutes",
                str(minutes),
                "--accuracy-minutes",
                str(minutes),
                "--speed-minutes",
                str(minutes),
                "--gsm8k-timeout",
                "1800",
                "--gsm8k-min-accuracy",
                str(58 / 64),
                "--no-fetch",
                "--fetch-url",
                "",
                "--deadline-monotonic",
                str(work_end),
                "--output",
                str(target),
            ]
            item = dict(case=name, phase=phase, status="running", argv=child)
            queue.append(item)
            checkpoint()
            code = h.main(child)
            p = target / "results.json"
            result = (
                json.loads(p.read_text())[0]
                if p.exists()
                else dict(
                    case=name,
                    status="failed",
                    reason=f"exit {code} without results",
                    phases={},
                )
            )
            if code and result.get("status") == "ok":
                result.update(status="failed", reason=f"nonzero harness exit {code}")
            if phase not in result.get("phases", {}):
                result.setdefault("phases", {})[phase] = dict(
                    status="failed", reason=result.get("reason", "phase result missing")
                )
            attempts.append(copy.deepcopy(result))
            rows[name]["phases"].update(result.get("phases", {}))
            for key in (
                "commit",
                "ref",
                "env",
                "case_env",
                "case_server_args",
                "comparison_baseline",
                "quality_threshold",
                "note",
            ):
                if key in result:
                    rows[name][key] = result[key]
            failures = [
                k + ": " + v.get("reason", "")
                for k, v in rows[name]["phases"].items()
                if v.get("status") != "ok"
            ]
            rows[name].update(
                status="failed" if failures else "ok", reason="; ".join(failures)
            )
            item.update(
                status="failed" if code else result.get("status", "failed"),
                reason=rows[name]["reason"],
            )
            checkpoint()
            if result.get("stop_failed") or any(
                a.get("cleanup") not in (None, "ok")
                for p in result.get("phases", {}).values()
                for a in p.get("attempts", [])
            ):
                raise RuntimeError("owned cleanup failed")
            if result.get("status") == "interrupted":
                raise KeyboardInterrupt
    except (KeyboardInterrupt, SystemExit):
        errors.append("interrupted")
    except Exception as exc:
        errors.append(repr(exc))
    finally:
        h.probe_runtime = original_probe
        for name in rows:
            for phase in ("accuracy", "speed"):
                if not any(x["case"] == name and x["phase"] == phase for x in queue):
                    queue.append(
                        dict(
                            case=name,
                            phase=phase,
                            status="skipped",
                            reason="session ended before scheduling",
                        )
                    )
        for x in queue:
            if x["status"] == "running":
                x.update(status="failed", reason="session stopped")
            if x["status"] != "ok":
                rows[x["case"]].update(
                    status="incomplete", reason=x.get("reason", "phase failed")
                )
        checkpoint()
        failed = bool(errors or any(x["status"] != "ok" for x in queue))
        h.save(
            output / "run-status.json",
            dict(
                status="failed/incomplete" if failed else "completed",
                errors=errors,
                elapsed_s=time.monotonic() - started,
                budget_s=args.hours * 3600,
            ),
        )
        print("Results:", output, flush=True)
    return int(failed)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, h._interrupt)
    try:
        sys.exit(run())
    except (ValueError, OSError) as exc:
        print("ERROR:", exc, file=sys.stderr)
        sys.exit(2)
