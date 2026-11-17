"""Regression checks for isolated cases, profile failures and F1-only reservation."""

import ast
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import sfa2h as s
from summarize_sfa_profile import summarize


def test_case_commands_match_requested_matrix(monkeypatch):
    monkeypatch.setattr(s.h, "CASES", [])
    monkeypatch.setattr(s.h, "CASE_PROFILES", {})
    monkeypatch.setattr(s.h, "MATCHED_BASELINES", {})
    s.register()
    args = s.h.parser().parse_args(["--round", "speed", "--decode-graph", "off"])
    for name, ref, env, _, extra in s.h.CASES:
        effective = s.h.case_settings(args, name)
        command = s.h.server_argv(
            effective, effective.memfrac, False, extra, phase="speed"
        )
        for flag, value in {
            "--max-running-requests": "1",
            "--prefill-max-requests": "1",
            "--max-total-tokens": "63488",
            "--chunked-prefill-size": "32768",
            "--max-prefill-tokens": "16384",
            "--ep-size": "1",
        }.items():
            assert command[command.index(flag) + 1] == value
            assert command.count(flag) == 1
        assert env["SGLANG_NPU_INDEXER_INT8"] == "True"
        assert all(env[flag] == "False" for flag in s.DISABLED)
        assert ref == s.SOURCE
    assert s.h.CASES[1][2]["SGLANG_NPU_SFA_F1_PREALLOC"] == "True"
    assert s.h.CASES[2][2]["SGLANG_NPU_SFA_KV_QUANT_READ"] == "True"


def test_profile_export_failure_retains_unprofiled_measurement(tmp_path, monkeypatch):
    case = tmp_path / s.PLAN[0] / "speed" / "attempt-1"
    case.mkdir(parents=True)
    (case / "server.log").write_text("")
    calls = []

    def run(*args, **kwargs):
        calls.append(kwargs)
        return {
            "status": "ok",
            "reason": "",
            "metrics": {"TTFT avg": 32000, "TTFT P90": 54000},
        }

    monkeypatch.setattr(s.h, "run_aisbench", run)
    monkeypatch.setattr(s.h, "create_worktree", lambda *a: "unused")
    monkeypatch.setattr(s.h, "execute", lambda *a, **kw: SimpleNamespace(returncode=1))
    s.install_hooks()
    result = s.h.run_aisbench(
        SimpleNamespace(remote_root="/tmp/owned", container="test"),
        case,
        time.monotonic() + 400,
        {},
        "owned",
        "log",
        print,
        s.SOURCE,
    )
    assert result["metrics"]["TTFT avg"] == 32000
    assert result["status"] == "failed"
    assert result["profiling"]["status"] == "failed"
    assert json.loads((case / "unprofiled-speed.json").read_text())["status"] == "ok"
    assert calls[0] == {}
    assert calls[1]["profile"]["PROFILE_STEPS"] == "3"
    assert calls[1]["profile"]["PROFILE_PHASE"] == "2"


def test_f2_fallback_is_failed_even_with_valid_timings(tmp_path, monkeypatch):
    case = tmp_path / s.PLAN[2] / "speed" / "attempt-1"
    case.mkdir(parents=True)
    (case / "server.log").write_text(
        "SFA F2 quant sparse attention ACTIVE\nSFA F2 FALLBACK to F1"
    )
    monkeypatch.setattr(s.h, "create_worktree", lambda *a: "unused")
    monkeypatch.setattr(
        s.h,
        "run_aisbench",
        lambda *a, **kw: {"status": "ok", "reason": "", "metrics": {"TTFT avg": 1}},
    )
    s.install_hooks()
    result = s.h.run_aisbench(
        SimpleNamespace(remote_root="/tmp/owned"),
        case,
        time.monotonic() + 1,
        {},
        "owned",
        "log",
        print,
        s.SOURCE,
    )
    assert result["status"] == "failed"
    assert result["activation"]["f2_fallback"]
    assert result["metrics"]["TTFT avg"] == 1


def test_kernel_summary_and_missing_exports(tmp_path):
    with pytest.raises(ValueError, match="no exported"):
        summarize(tmp_path)
    path = tmp_path / "rank0" / "kernel_details.csv"
    path.parent.mkdir()
    path.write_text(
        "Name,Type,Duration(us)\na,QuantLightningIndexer,12\nb,KvQuantSparseFlashAttention,30\n"
    )
    result = summarize(tmp_path)
    assert result["indexer_int8_observed"] and result["f2_observed"]
    assert result["exports"][0]["device_time_us"] == 42


def test_prepared_f1_reservation_without_other_features():
    # Execute the production function without importing an NPU runtime.
    patch = (s.ROOT / "sfa2h-serving.diff").read_text()
    source = (
        Path(__file__).resolve().parents[2]
        / "sfa2h-source/python/sglang/srt/hardware_backend/npu/attention/dcp_memory.py"
    )
    if not source.exists():
        pytest.skip("serving worktree absent in deployment archive")
    module = ast.parse(source.read_text())
    fn = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "runtime_gather_reserve_bytes"
    )
    flags = {"SGLANG_NPU_SFA_KV_8BIT", "SGLANG_NPU_SFA_F1_PREALLOC"}
    scope = {
        "_flag": lambda name, default=False: name in flags,
        "get_parallel": lambda: SimpleNamespace(attn_tp_size=16),
    }
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), scope)
    model = SimpleNamespace(
        modules=lambda: [
            SimpleNamespace(use_dsa=True, kv_lora_rank=512, qk_rope_head_dim=64)
        ]
    )
    reserve = scope[fn.name](model, 32768, context_tokens=1048576)
    expected_rows = (1048576 + 32768 + 15) // 16 + 256
    assert reserve == expected_rows * 2 * (656 + 8 + 2048 + 1152)
    flags.clear()
    assert scope[fn.name](model, 32768, context_tokens=1048576) == 0
    assert "SFA F2 FALLBACK to F1" in patch


def test_fixed_matrix_rejects_accuracy_override():
    assert s.main(["--round", "accuracy"]) == 2
