"""Regression checks for the ten-hour schedule and published command builders."""

import copy
import json
from pathlib import Path

import pytest
import session10 as s


@pytest.fixture(autouse=True)
def isolate_case_tables(monkeypatch):
    monkeypatch.delenv("TTFT10_STARTED_MONOTONIC", raising=False)
    cases = copy.deepcopy(s.h.CASES)
    profiles = copy.deepcopy(s.h.CASE_PROFILES)
    baselines = copy.deepcopy(s.h.MATCHED_BASELINES)
    yield
    s.h.CASES = cases
    s.h.CASE_PROFILES = profiles
    s.h.MATCHED_BASELINES = baselines


def test_fixed_phase_commands_and_ep_subset():
    table = s.register_cases()
    args = s.h.parser().parse_args(["--hours", "10", "--moe-mode", "tp"])
    for name, case in table.items():
        effective = s.h.case_settings(args, name)
        for phase in ("accuracy", "speed"):
            argv = s.h.server_argv(
                effective, effective.memfrac, False, case[4], phase=phase
            )
            for key, value in {
                "--max-running-requests": "1",
                "--prefill-max-requests": "1",
                "--chunked-prefill-size": "32768",
                "--max-prefill-tokens": "16384",
                "--context-length": "1048576",
                "--max-total-tokens": "63488",
                "--mem-fraction-static": "0.8",
                "--cuda-graph-backend-decode": "disabled",
            }.items():
                assert argv[argv.index(key) + 1] == value
                assert argv.count(key) == 1
        if name in ("27-combined-ep", "28-combined-megamoe"):
            assert case[2]["SGLANG_NPU_DSA_CP_TOKEN_SHARDED"] == "False"
            assert argv[argv.index("--ep-size") + 1] == "16"
    assert (
        table["28-combined-megamoe"][2][
            "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK"
        ]
        == "2048"
    )
    assert table["35-current-f1"][2]["SGLANG_NPU_SFA_KV_QUANT_READ"] == "False"
    assert table["36-buffered-f1"][2]["SGLANG_NPU_SFA_F1_PREALLOC"] == "True"


def test_only_matching_control_accuracy_can_be_reused(monkeypatch):
    monkeypatch.setattr(s.h, "try_reuse_phase", lambda *a: True)
    monkeypatch.setattr(s.h, "reusable_phase", lambda *a: True)
    originals = s.install_hooks()
    try:
        assert s.h.try_reuse_phase(None, {"case": "33-best-ttft"}, "accuracy")
        assert not s.h.try_reuse_phase(None, {"case": "33-best-ttft"}, "speed")
        assert not s.h.try_reuse_phase(None, {"case": "36-buffered-f1"}, "accuracy")
        assert not s.h.reusable_phase({}, {})
        assert not s.h.reusable_phase(
            {"source_diff_sha256": "a"}, {"source_diff_sha256": "b"}
        )
    finally:
        s.h.create_worktree, s.h.try_reuse_phase, s.h.reusable_phase = originals


def test_repetitions_remove_all_reuse_options():
    assert s.without_reuse(
        ["--reuse-results", "old", "--hours", "10", "--reuse-results", "new"]
    ) == ["--hours", "10"]


def test_reject_budget_and_unmatched_overrides():
    for args in (
        ["--hours", "11"],
        ["--hours", "10", "--retry-oom"],
        ["--hours", "10", "--requests", "8"],
    ):
        with pytest.raises(ValueError):
            s.run(args)


def test_monotonic_budget_skips_cases_without_launch(monkeypatch, tmp_path):
    monkeypatch.setattr(s.h, "load_gsm8k", lambda *a: [])
    monkeypatch.setattr(
        s.h, "pin_refs", lambda cases, *a: {c[1]: {"commit": c[1]} for c in cases}
    )
    clock = iter([0, 601, 602, 603, 604, 605, 606, 607, 608])
    monkeypatch.setattr(s.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(s.h, "main", lambda *a: pytest.fail("expired case launched"))
    output = tmp_path / "run"
    assert s.run(["--hours", "0.1", "--gsm8k", "unused", "--output", str(output)]) == 1
    assert all(
        r["status"] == "skipped"
        for r in json.loads((output / "results.json").read_text())
    )


def test_full_session_uses_fresh_speed_and_qualified_winner(monkeypatch, tmp_path):
    monkeypatch.setattr(s.h, "load_gsm8k", lambda *a: [{}] * 64)
    monkeypatch.setattr(
        s.h, "pin_refs", lambda cases, *a: {c[1]: {"commit": c[1]} for c in cases}
    )
    calls = []

    def main(argv):
        args = s.h.parser().parse_args(argv)
        calls.append(args)
        phase = {
            "status": "ok",
            "attempts": [
                {
                    "status": "ok",
                    "comparison_config": {"fixed": True},
                    "gsm8k": {
                        "requested": 64,
                        "completed": 64,
                        "details": [{"correct": True}] * 60 + [{"correct": False}] * 4,
                    },
                    "speed": {
                        "status": "ok",
                        "metrics": {
                            "TTFT avg": 20000
                            if args.cases == "36-buffered-f1"
                            else 30000,
                            "TTFT P90": 40000,
                        },
                    },
                }
            ],
        }
        phases = {
            k: copy.deepcopy(phase)
            for k in (["accuracy", "speed"] if args.round == "all" else [args.round])
        }
        out = Path(args.output)
        out.mkdir(parents=True)
        s.h.save(
            out / "results.json",
            [
                {
                    "case": args.cases,
                    "status": "ok",
                    "commit": "a" * 40,
                    "source_diff_sha256": "test-source",
                    "phases": phases,
                }
            ],
        )
        return 0

    monkeypatch.setattr(s.h, "main", main)
    out = tmp_path / "session"
    assert s.run(["--hours", "10", "--gsm8k", "mock", "--output", str(out)]) == 0
    assert len(calls) == 9
    assert [c.cases for c in calls[-3:]] == [
        "36-buffered-f1",
        "36-buffered-f1",
        "24-historical-control",
    ]
    assert all(c.accuracy_minutes == 90 and c.speed_minutes == 20 for c in calls)
    assert all(c.round == "speed" and not c.reuse_results for c in calls[-3:])
    assert (
        json.loads((out / "confirmation-status.json").read_text())["fresh_samples"] == 3
    )
    assert "drift +0.00%" in (out / "SAMPLES.md").read_text()


def test_published_serving_commits_are_pinned():
    table = s.register_cases()
    assert table["36-buffered-f1"][1] == s.BUFFERED_TP
    assert table["27-combined-ep"][1] == s.EP
    assert table["28-combined-megamoe"][1] == s.EP
    assert table["33-best-ttft"][1] == s.TP
    assert table["35-current-f1"][1] == s.TP
