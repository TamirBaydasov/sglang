"""Offline scheduler checks; no Docker, network, or NPU calls."""

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import session16 as session  # noqa: E402


def test_session_budget_and_breakable_command(monkeypatch):
    h = session.h
    monkeypatch.setattr(h, "CASES", copy.deepcopy(h.CASES))
    monkeypatch.setattr(h, "CASE_PROFILES", {})
    table = session.register_cases()
    args = h.parser().parse_args(["--decode-graph", "off"])
    h.validate_args(args)
    effective = h.case_settings(args, "29-tp-breakable")
    argv = h.server_argv(
        effective, effective.memfrac, False, table["29-tp-breakable"][4], phase="speed"
    )
    assert argv[argv.index("--cuda-graph-backend-prefill") + 1] == "breakable"
    assert argv[argv.index("--cuda-graph-backend-decode") + 1] == "disabled"
    assert argv[argv.index("--max-running-requests") + 1] == "1"
    assert effective.allow_server_queueing
    assert sum(cap for _, cap in session.BLOCKS) == 16 * 60


def test_offline_session_pins_once_and_uses_qualified_tp_for_ep(tmp_path, monkeypatch):
    h = session.h
    monkeypatch.setattr(h, "CASES", copy.deepcopy(h.CASES))
    monkeypatch.setattr(h, "CASE_PROFILES", {})
    pins = []

    def pin(cases, *args):
        pins.append(cases)
        return {c[1]: {"commit": "a" * 40} for c in cases}

    monkeypatch.setattr(h, "pin_refs", pin)
    monkeypatch.setattr(h, "load_gsm8k", lambda *args: [{}] * 64)
    calls = []

    def main(argv):
        args = h.parser().parse_args(argv)
        calls.append(args)
        n = 4 if args.cases in ("30-sfa-f1-probe", "31-packed-single-chunk") else 64
        phase = {
            "status": "ok",
            "attempts": [
                {
                    "status": "ok",
                    "comparison_config": {"recipe": "same"},
                    "gsm8k": {
                        "requested": n,
                        "completed": n,
                        "correct": n,
                        "details": [{"correct": True} for _ in range(n)],
                    },
                    "speed": {
                        "status": "ok",
                        "metrics": {"TTFT avg": 20000, "TTFT P90": 30000},
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
        (out / "results.json").write_text(
            json.dumps(
                [
                    {
                        "case": args.cases,
                        "commit": "a" * 40,
                        "status": "ok",
                        "phases": phases,
                    }
                ]
            )
        )
        return 0

    monkeypatch.setattr(h, "main", main)
    out = tmp_path / "session"
    assert (
        session.run(
            [
                "--gsm8k",
                str(tmp_path / "dataset"),
                "--output",
                str(out),
                "--hours",
                "16",
                "--decode-graph",
                "off",
                "--baseline-ref",
                "npu-w4a8-tp-bias-fix",
            ]
        )
        == 0
    )
    assert len(pins) == 1
    assert all(a.no_fetch for a in calls)
    assert len({a.output for a in calls}) == len(calls)
    assert any(a.cases == "29-tp-breakable" for a in calls)
    table = {c[0]: c for c in h.CASES}
    for name in ("27-combined-ep", "28-combined-megamoe"):
        assert table[name][2]["SGLANG_NPU_DSA_CP_TOKEN_SHARDED"] == "False"
    results = json.loads((out / "results.json").read_text())
    assert any(r["case"].startswith("25-combined-full-repeat-") for r in results)
