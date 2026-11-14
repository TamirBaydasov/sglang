"""Regression cases from the November 14 A3 archive."""

import importlib.util
import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


h = load("night_harness", "run_pr_tests.py")
c = load("night_client", "prtest_gsm8k.py")


def args_for(*argv):
    args = h.parser().parse_args(list(argv))
    h.validate_args(args)
    return args


@pytest.mark.parametrize(
    "variant,backend,ep",
    [
        ("baseline", "none", 1),
        ("deepep", "deepep", 16),
        ("megamoe", "ascend_fuseep", 16),
    ],
)
def test_backend_validation_and_exact_replay(
    tmp_path, monkeypatch, variant, backend, ep
):
    seen = []
    config = dict(
        tp_size=16, dcp_size=16, ep_size=ep, moe_a2a_backend=backend, fuseep_mode=3
    )

    class Server(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(config).encode())

        def do_POST(self):
            seen.append(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            self.send_response(200)
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    dict(text="42", meta_info={"finish_reason": {"type": "stop"}})
                ).encode()
            )

        def log_message(self, *args):
            pass

    source = tmp_path / "raw.jsonl"
    prompt = "Saved failing prompt, including its original five shots."
    source.write_text(
        json.dumps(
            dict(
                prompt_id=7,
                prompt=prompt,
                prompt_sha256=c.digest(prompt.encode()),
                label=42,
            )
        )
        + "\n"
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), Server)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(h, "SERVER_PORT", server.server_port)
    args = args_for("--gsm8k-replay", str(source), "--gsm8k-samples", "1")
    args.dataset_sha256 = c.digest(source.read_bytes())
    try:
        result = h.run_gsm8k(
            [],
            args,
            tmp_path,
            time.monotonic() + 15,
            lambda message: None,
            server=config,
        )
    finally:
        server.shutdown()
        server.server_close()
    assert result["status"] == "ok"
    assert result["completed"] == result["answered"] == result["correct"] == 1
    assert result["manifest"]["variant"] == variant
    assert result["details"][0]["id"] == 7
    assert seen[0]["text"] == prompt
    assert seen[0]["sampling_params"]["max_new_tokens"] == 1024
    broken = dict(config, ep_size=2)
    with pytest.raises(ValueError, match="ep_size"):
        c.validate_server(broken, variant)


def test_replay_rejects_changed_prompt_and_duplicate_ids(tmp_path):
    source = tmp_path / "raw.jsonl"
    row = dict(prompt_id=0, prompt="p", prompt_sha256=c.digest(b"p"), label=1)
    source.write_text(json.dumps(dict(row, prompt="changed")) + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        c.replay_rows(source, 1)
    source.write_text((json.dumps(row) + "\n") * 2)
    with pytest.raises(ValueError, match="unique"):
        c.replay_rows(source, 1)


def test_unevaluated_is_not_zero_accuracy(tmp_path):
    args = args_for()
    args.dataset_sha256 = "hash"
    result = h.gsm8k_snapshot(tmp_path, args, complete=True, returncode=2)
    assert result["status"] == "not-evaluated"
    assert result["accuracy"] is None
    assert result["completed"] == result["answered"] == 0
    record = {
        "case": "14-baseline-ep",
        "phases": {"accuracy": {"attempts": [{"gsm8k": result}]}},
    }
    h.summarize([record], tmp_path, args)
    report = (tmp_path / "SUMMARY.md").read_text()
    assert "not evaluated; requested=64, completed=0, answered=0" in report
    assert "0/64" not in report


def test_diagnostic_commands_keep_workloads_and_isolate_flags():
    args = args_for(
        "--cases",
        "15-baseline-8k,16-megamoe-8k,17-baseline-8k-repeat,18-baseline-prefill1,19-baseline-prefill1-repeat,20-moe-ts-off-eager,21-moe-ts-on-eager,22-moe-ts-off-graph,23-moe-ts-on-graph",
    )
    for name, ref, env, _, extra in h.plan_cases(args):
        effective = h.case_settings(args, name)
        argv = h.server_argv(
            effective,
            effective.memfrac,
            effective.decode_graph == "on",
            extra,
            phase="accuracy"
            if name.startswith(("20-", "21-", "22-", "23-"))
            else "speed",
        )
        assert effective.length == 1000000 and effective.requests == 16
        if name == "16-megamoe-8k":
            assert env["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK"] == "1024"
        if name.startswith(("15-", "16-", "17-")):
            assert argv[argv.index("--chunked-prefill-size") + 1] == "8192"
            assert argv[argv.index("--max-prefill-tokens") + 1] == "8192"
            assert argv[argv.index("--max-running-requests") + 1] == "16"
        elif name.startswith(("18-", "19-")):
            assert argv[argv.index("--prefill-max-requests") + 1] == "1"
        else:
            assert effective.round == "accuracy"
            assert env["SGLANG_NPU_DSA_CP_TOKEN_SHARDED"] == (
                "False" if "-off-" in name else "True"
            )
            assert argv[argv.index("--cuda-graph-backend-decode") + 1] == (
                "disabled" if name.endswith("eager") else "full"
            )


@pytest.mark.parametrize("state", ["supported", "missing", "incompatible"])
def test_probe_imports_provider_before_checking_registration(tmp_path, state):
    backend = (
        tmp_path / "python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py"
    )
    backend.parent.mkdir(parents=True)
    backend.write_text(
        "from sgl_kernel_npu.attention.sinks_attention import register\ndef attention(query):\n return torch.ops.npu.npu_sparse_flash_attention_lse(query=query)\n"
    )
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "torch_npu.py").write_text("")
    (tmp_path / "torch.py").write_text(
        "from types import SimpleNamespace\nops=SimpleNamespace(npu=SimpleNamespace(),loaded_libraries=set())\n"
    )
    provider = tmp_path / "sgl_kernel_npu/attention/sinks_attention.py"
    provider.parent.mkdir(parents=True)
    argument = "query" if state == "supported" else "different"
    provider.write_text(
        "import torch\nfrom types import SimpleNamespace\n"
        "class Schema:\n"
        f" arguments=[SimpleNamespace(name={argument!r})]\n"
        " def __str__(self): return 'LSE(Tensor query) -> Tensor'\n"
        "class Packet:\n def overloads(self): return ['default']\n"
        " default=SimpleNamespace(_schema=Schema())\n"
        + (
            "torch.ops.npu.npu_sparse_flash_attention_lse=Packet()\n"
            if state != "missing"
            else ""
        )
        + "torch.ops.loaded_libraries.add('/fake/libattention.so')\nregister=True\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", h.PROBE, str(tmp_path)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    data = json.loads(result.stdout.split("PRTEST_PROBE=")[1])
    assert data["ops"]["npu_sparse_flash_attention_lse"] == (state != "missing")
    assert bool(data["operator_schema_errors"]) == (state == "incompatible")
    if state != "missing":
        assert data["operator_schemas"]["npu_sparse_flash_attention_lse"][
            "default"
        ].startswith("LSE(")
    assert data["provider_modules"]["sgl_kernel_npu.attention.sinks_attention"] == str(
        provider
    )
    assert "/fake/libattention.so" in data["loaded_libraries"]


def test_matched_control_still_rejects_different_prefill_limits():
    base = {
        "chunk": 8192,
        "resolved_server": {
            "moe_a2a_backend": "none",
            "ep_size": 1,
            "prefill_max_requests": 16,
        },
    }
    variant = {
        "chunk": 8192,
        "resolved_server": dict(
            base["resolved_server"],
            moe_a2a_backend="ascend_fuseep",
            ep_size=16,
            fuseep_mode=3,
        ),
    }
    assert h.backend_comparison_matches("16-megamoe-8k", variant, base)
    changed = dict(
        variant,
        resolved_server=dict(variant["resolved_server"], prefill_max_requests=1),
    )
    assert not h.backend_comparison_matches("16-megamoe-8k", changed, base)


def test_client_rejection_reason_is_checkpointed(tmp_path, monkeypatch):
    args = args_for()
    args.gsm8k = "unused.jsonl"
    args.dataset_sha256 = "hash"

    def reject(command, deadline, timeout, **kwargs):
        kwargs["log_path"].write_text("ERROR: Wrong server: ep_size=16, expected 1\n")
        return type("Result", (), {"returncode": 2})()

    monkeypatch.setattr(h, "execute", reject)
    result = h.run_gsm8k(
        [], args, tmp_path, time.monotonic() + 10, lambda message: None
    )
    assert result["status"] == "not-evaluated"
    assert result["reason"] == "ERROR: Wrong server: ep_size=16, expected 1"
    assert (
        json.loads((tmp_path / "gsm8k.json").read_text())["reason"] == result["reason"]
    )


def test_overlay_checksums_paths_and_pinned_base(tmp_path, monkeypatch):
    sha = "a" * 40
    path = tmp_path / sha / "python/fix.py"
    path.parent.mkdir(parents=True)
    path.write_text("fixed = True\n")
    entry = dict(
        ref="feature",
        base_commit=sha,
        files={"python/fix.py": c.digest(path.read_bytes())},
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([entry]))
    assert h.load_serving_overlay(tmp_path)[1] == [entry]
    args = args_for("--serving-overlay", str(tmp_path))
    args.remote_root = "/tmp/owned"
    args.registry_path = tmp_path / "registry.json"
    calls = []
    monkeypatch.setattr(h, "git", lambda *a, **kw: "")
    monkeypatch.setattr(
        h, "run_capture", lambda *a, **kw: calls.append(kw["input_text"]) or ""
    )
    record = dict(case="test", ref="feature", commit=sha)
    h.create_worktree(args, record, {"worktrees": []}, time.monotonic() + 10)
    assert json.loads(calls[0]) == {"python/fix.py": path.read_text()}
    assert record["serving_overlay"] == entry
    with pytest.raises(ValueError, match="base differs"):
        h.create_worktree(
            args,
            dict(record, commit="b" * 40),
            {"worktrees": []},
            time.monotonic() + 10,
        )
    path.write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        h.load_serving_overlay(tmp_path)
    entry["files"] = {"../escape": "x"}
    manifest.write_text(json.dumps([entry]))
    with pytest.raises(ValueError, match="path"):
        h.load_serving_overlay(tmp_path)
