import json, sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import run_best_pair as pair
import run_pr_tests as h


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    monkeypatch.setattr(h, "CASE_PROFILES", {})
    monkeypatch.setattr(h, "MATCHED_BASELINES", {})
    monkeypatch.delenv("BESTPAIR_STARTED_MONOTONIC", raising=False)


def test_recipes_and_actual_commands(monkeypatch):
    calls = []
    monkeypatch.setattr(h, "dry_run", lambda a, c: calls.append((a, c)))
    assert pair.run(["--dry-run"]) == 0
    assert len(calls) == 4
    for a, c in calls:
        case = c[0]
        cmd = h.server_argv(a, 0.8, False, case[4], phase=a.round)

        def value(flag):
            return cmd[len(cmd) - 1 - cmd[::-1].index(flag) + 1]

        assert value("--max-running-requests") == "1"
        assert value("--prefill-max-requests") == "1"
        assert value("--tp-size") == "16"
        assert value("--max-total-tokens") == "63488"
        if case[0].startswith("28"):
            assert value("--moe-a2a-backend") == "ascend_fuseep"
            assert value("--fuseep-mode") == "3"
            assert case[2]["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK"] == "2048"


def setup_run(monkeypatch, tmp_path, failed=False):
    monkeypatch.setattr(h, "load_gsm8k", lambda *a: [])
    monkeypatch.setattr(
        h, "pin_refs", lambda cases, *a: {r: {"commit": r} for n, r, e, t, x in cases}
    )
    monkeypatch.setattr(h, "summarize", lambda *a: None)
    seen = []

    def child(argv):
        args = h.parser().parse_args(argv)
        seen.append(args)
        Path(args.output).mkdir(parents=True)
        h.save(
            Path(args.output) / "results.json",
            [
                dict(
                    case=args.cases,
                    status="failed" if failed else "ok",
                    phases={
                        args.round: dict(
                            status="failed" if failed else "ok",
                            reason="fixture failure" if failed else "",
                        )
                    },
                )
            ],
        )
        return int(failed)

    monkeypatch.setattr(h, "main", child)
    return seen, [
        "--gsm8k",
        "/fake",
        "--output",
        str(tmp_path / "output"),
        "--no-fetch",
        "--fetch-url",
        "",
    ]


def test_fresh_phases_and_ref_pins(monkeypatch, tmp_path):
    seen, args = setup_run(monkeypatch, tmp_path)
    assert pair.run(args + ["--tp-ref", "a" * 40, "--ep-ref", "b" * 40]) == 0
    assert len(seen) == 4
    assert [a.case_minutes for a in seen] == [75, 15, 120, 15]
    assert len({a.output for a in seen}) == 4
    assert all(a.gsm8k_timeout == 1800 for a in seen)
    assert [c[1] for c in h.CASES] == ["a" * 40, "b" * 40]
    assert (
        json.loads((tmp_path / "output/run-status.json").read_text())["status"]
        == "completed"
    )


def test_deadline_records_all_skips(monkeypatch, tmp_path):
    seen, args = setup_run(monkeypatch, tmp_path)
    assert pair.run(args + ["--hours", ".1"]) == 1
    assert not seen
    q = json.loads((tmp_path / "output/queue.json").read_text())
    assert len(q) == 4 and all(x["status"] == "skipped" for x in q)
    assert all(
        r["status"] == "incomplete"
        for r in json.loads((tmp_path / "output/results.json").read_text())
    )


def test_failure_retains_speed(monkeypatch, tmp_path):
    seen, args = setup_run(monkeypatch, tmp_path, True)
    assert pair.run(args) == 1
    assert len(seen) == 4 and [a.round for a in seen] == [
        "accuracy",
        "speed",
        "accuracy",
        "speed",
    ]
    assert len(json.loads((tmp_path / "output/attempts.json").read_text())) == 4


def test_report_failure_retains_measurements(monkeypatch, tmp_path):
    seen, args = setup_run(monkeypatch, tmp_path)

    def bad(*a):
        raise RuntimeError("report failure")

    monkeypatch.setattr(h, "summarize", bad)
    assert pair.run(args) == 0
    assert (tmp_path / "output/report-error.json").exists()
    assert len(json.loads((tmp_path / "output/attempts.json").read_text())) == 4


def test_owned_cleanup_failure_stops(monkeypatch, tmp_path):
    seen, args = setup_run(monkeypatch, tmp_path)

    def child(argv):
        a = h.parser().parse_args(argv)
        seen.append(a)
        Path(a.output).mkdir(parents=True)
        h.save(
            Path(a.output) / "results.json",
            [dict(case=a.cases, status="failed", stop_failed=True, phases={})],
        )
        return 1

    monkeypatch.setattr(h, "main", child)
    assert pair.run(args) == 1
    assert len(seen) == 1
    assert len(json.loads((tmp_path / "output/queue.json").read_text())) == 4
