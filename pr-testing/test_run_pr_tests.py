"""Regression checks using archived logs, local HTTP, and owned subprocesses."""

import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("pr_harness", ROOT / "run_pr_tests.py")
h = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(h)
SPEC = importlib.util.spec_from_file_location("owned_remote", ROOT / "prtest_remote.py")
remote = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(remote)
ARCHIVE = ROOT / "prtest-results" / "20261112-232436"
if not ARCHIVE.exists():
    ARCHIVE = ROOT / "fixtures" / "20261112-232436"


def args_for(*argv):
    args = h.parser().parse_args(list(argv))
    h.validate_args(args)
    return args


def test_archived_baseline_and_failure_placeholders():
    text = (ARCHIVE / "00-baseline/ais.log").read_text()
    data = h.validate_ais(text, 0, 16, 1000000)
    assert data["status"] == "ok"
    assert data["preparation"]["total_req"] == 1
    assert data["metrics"]["TTFT avg"] == 36104.9553
    assert data["metrics"]["TTFT P90"] == 60187.489
    failed = h.validate_ais(
        (ARCHIVE / "01-pr-e-chat-encode/ais.log").read_text(), 0, 16
    )
    assert failed["status"] == "failed"
    assert failed["metrics"] is None
    assert "request/tool failure" in failed["reason"]
    assert h.validate_ais(text, 1, 16)["status"] == "failed"
    assert h.validate_ais(text, 0, 8)["status"] == "failed"
    assert h.validate_ais(text, 0, 16, 10000)["status"] == "failed"


def test_safe_metric_parsing_and_partial_logs():
    assert h.parse_ttft_lines("INFO {\"TTFT avg\": 12.3}\nINFO {'TTFT avg': 14.5}") == [
        {"TTFT avg": 12.3},
        {"TTFT avg": 14.5},
    ]
    assert h.parse_ttft_lines("INFO {'TTFT avg': __import__('os').getpid()}") == []
    text = (ARCHIVE / "00-baseline/ais.log").read_text()
    assert h.validate_ais(text[: text.index("'TTFT avg'")], 0, 16)["status"] == "failed"


def test_summary_legacy_sha_and_partial_records(tmp_path):
    records = [
        {
            "case": "00-baseline",
            "ref": "base",
            "status": "failed",
            "accuracy": {"sha": {"a": "same"}},
        },
        {
            "case": "01-pr-e-chat-encode",
            "ref": "test",
            "status": "failed",
            "accuracy": {"sha": {"a": "same"}},
        },
        {"case": "02-pr-b-indexer-int8", "status": "pending"},
    ]
    h.summarize(records, tmp_path, SimpleNamespace())
    assert "| 01-pr-e-chat-encode | 1/1 |" in (tmp_path / "SUMMARY.md").read_text()
    assert "unavailable" in (tmp_path / "SUMMARY.md").read_text()


def test_report_failure_keeps_results(tmp_path, monkeypatch):
    monkeypatch.setattr(
        h,
        "summarize",
        lambda *args: (_ for _ in ()).throw(ValueError("bad presentation")),
    )
    args = SimpleNamespace(report_failed=False)
    h.checkpoint_results([{"case": "saved"}], tmp_path, args, lambda message: None)
    assert json.loads((tmp_path / "results.json").read_text()) == [{"case": "saved"}]
    assert args.report_failed


def test_regenerate_preserves_original_evidence(tmp_path):
    before = {path: path.read_bytes() for path in ARCHIVE.rglob("*") if path.is_file()}
    assert (
        h.main(["--regenerate", str(ARCHIVE), "--output", str(tmp_path / "report")])
        == 0
    )
    summary = (tmp_path / "report/SUMMARY.md").read_text()
    assert "36.105 / 60.187" in summary
    assert "| 01-pr-e-chat-encode | failed" in summary
    assert "99.999" not in summary
    assert all(path.read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize(
    "argv",
    [
        ["--cases", "typo"],
        ["--cases", "00-baseline,00-baseline"],
        ["--hours", "nan"],
        ["--case-minutes", "0"],
        ["--requests", "-2"],
        ["--memfrac", "nan"],
        ["--flags-overrides", "00-baseline:X;bad=1"],
        ["--extra-server-args=--model-path /wrong/model"],
    ],
)
def test_invalid_arguments_fail_before_execution(argv):
    assert h.main(argv + ["--dry-run"]) == 2


def test_missing_gsm8k_fails_without_docker():
    assert h.main(["--round", "accuracy"]) == 2


def test_dry_run_uses_real_builders(capsys):
    args = args_for(
        "--round", "all", "--cases", "00-baseline", "--ais-conda-env", "custom"
    )
    h.dry_run(args, h.plan_cases(args))
    output = capsys.readouterr().out
    for phase, width in (("accuracy", "1"), ("speed", "16")):
        argv = h.server_argv(args, args.memfrac, True, phase=phase)
        assert argv[argv.index("--max-running-requests") + 1] == width
        assert (
            h.server_shell(args, "<run-root>/worktrees/00-baseline", {}, argv) in output
        )
    assert h.shlex.join(h.ais_command(args)) in output
    assert "accuracy: fresh server" in output and "speed: fresh server" in output


def test_combined_phases_launch_and_stop_separate_servers(tmp_path, monkeypatch):
    args = args_for("--round", "all")
    args.registry_path = tmp_path / "registry.json"
    args.ais_preflight_record = None
    args.ais_preflight_error = None
    args.gsm_rows = []
    record = {"case": "00-baseline", "case_env": {}}
    registry = {"processes": []}
    runtime = {"model_files_sha256": {}, "versions": {}, "environment": {}, "ops": {}}
    events = []

    def launch(args, command, directory, registry, deadline):
        phase = directory.parent.name
        state = phase + "-owned"
        registry["processes"].append({"state": state})
        events.append(("launch", phase))
        return state, None

    monkeypatch.setattr(h, "require_free_port", lambda: None)
    monkeypatch.setattr(h, "probe_runtime", lambda *a: runtime)
    monkeypatch.setattr(h, "launch_server", launch)
    monkeypatch.setattr(h, "wait_health", lambda *a: None)
    monkeypatch.setattr(h, "smoke", lambda *a: {"server": {}})
    monkeypatch.setattr(h, "resolved_comparison", lambda config, smoke: config)
    monkeypatch.setattr(h, "run_gsm8k", lambda *a: {"status": "ok", "accuracy": 1})
    monkeypatch.setattr(h, "run_accuracy_prompts", lambda *a: {"errors": 0})
    monkeypatch.setattr(h, "run_aisbench", lambda *a: {"status": "ok", "reason": ""})
    monkeypatch.setattr(
        h, "stop_server", lambda args, state: events.append(("stop", state))
    )
    for phase, width in (("accuracy", "1"), ("speed", "16")):
        h.run_phase(
            args,
            record,
            phase,
            "/tree",
            tmp_path,
            registry,
            time.monotonic() + 10,
            lambda message: None,
            lambda: None,
        )
        attempt = record["phases"][phase]["attempts"][0]
        assert attempt["status"] == "ok"
        saved = json.loads((tmp_path / phase / "attempt-1" / "launch.json").read_text())
        argv = saved["argv"]
        assert argv[argv.index("--max-running-requests") + 1] == width
        assert saved["command"] == h.server_shell(args, "/tree", {}, argv)
    assert events == [
        ("launch", "accuracy"),
        ("stop", "accuracy-owned"),
        ("launch", "speed"),
        ("stop", "speed-owned"),
    ]


@pytest.mark.parametrize("discovery", ["exe", "path", "inherited_stub"])
def test_ais_preflight_and_runtime_share_custom_conda(tmp_path, monkeypatch, discovery):
    model = tmp_path / "model"
    model.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    tool = tmp_path / "tool"
    tool.mkdir()
    (tool / "config.py").write_text(
        f'HOST_PORT=30000\nHOST_IP="127.0.0.1"\nMODEL_PATH={str(model)!r}\nWORK_PATH={str(work)!r}\n'
    )
    (tool / "aisbench_test.py").write_text(
        'import os\nprint("ENV="+os.environ["ACTIVE_TEST_ENV"])\n'
    )
    binary = tmp_path / "ais_bench"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    env = tmp_path / "env.sh"
    conda = tmp_path / "conda"
    conda.write_text(
        "#!/bin/bash\n"
        'if [[ "$1" == shell.bash && "$2" == hook ]]; then\n'
        '  echo \'conda() { test "$1" = activate && export ACTIVE_TEST_ENV="$2"; };\'\n'
        "else\n"
        '  echo "CondaError: Run conda init before conda activate" >&2; exit 1\n'
        "fi\n"
    )
    conda.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.delenv("CONDA_EXE", raising=False)
    monkeypatch.delenv("BASH_FUNC_conda%%", raising=False)
    if discovery in ("exe", "inherited_stub"):
        monkeypatch.setenv("CONDA_EXE", str(conda))
    if discovery == "inherited_stub":
        monkeypatch.setenv(
            "BASH_FUNC_conda%%",
            '() { echo "CondaError: uninitialized" >&2; return 1; }',
        )
    env.write_text("conda activate fixture\n")
    wrapper = tmp_path / "profile-wrapper.sh"
    wrapper.write_text('test "$1" = --\nshift\nexec "$@"\n')
    monkeypatch.setattr(h, "REFERENCE_PROFILE0", wrapper)
    args = args_for(
        "--model",
        str(model),
        "--ais-dir",
        str(tool),
        "--ais-env",
        str(env),
        "--ais-conda-env",
        "custom",
    )
    record = h.ais_preflight(args, time.monotonic() + 10)
    assert record["ais_bench"] == str(binary)
    result = h.execute(
        h.ais_command(args), time.monotonic() + 10, env=h.ais_environment(args)
    )
    assert result.stdout.strip() == "ENV=custom"
    env.write_text("conda activate fixture\nfalse\n")
    with pytest.raises(RuntimeError, match="AIS preflight failed"):
        h.ais_preflight(args, time.monotonic() + 10)
    env.write_text("conda activate fixture\n")
    (tool / "config.py").write_text('HOST_PORT=1234\nHOST_IP="127.0.0.1"\n')
    with pytest.raises(RuntimeError, match="port mismatch"):
        h.ais_preflight(args, time.monotonic() + 10)


def test_gsm8k_reference_protocol_and_truncation(tmp_path, monkeypatch):
    rows = [
        {"sample_id": n, "question": f"question {n}", "answer": "reason #### 42"}
        for n in range(5)
    ]
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(
        "".join(
            json.dumps({k: v for k, v in row.items() if k != "sample_id"}) + "\n"
            for row in rows
        )
    )
    requests = []

    class ReferenceServer(BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.path == "/server_info"
            body = json.dumps(
                dict(
                    tp_size=16,
                    dcp_size=16,
                    ep_size=1,
                    moe_a2a_backend="none",
                    load_format="auto",
                    max_running_requests=16,
                )
            ).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, body))
            if body["text"].endswith("Question: question 2\nAnswer:"):
                self.send_response(500)
                self.end_headers()
                return
            finish = (
                "length"
                if body["text"].endswith("Question: question 1\nAnswer:")
                else "stop"
            )
            self.send_response(200)
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "text": "The answer is 42",
                        "meta_info": {
                            "finish_reason": {"type": finish},
                            "completion_tokens": 12,
                        },
                    }
                ).encode()
            )

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), ReferenceServer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(h, "SERVER_PORT", server.server_port)
    args = args_for("--gsm8k", str(dataset), "--gsm8k-samples", "3")
    args.dataset_sha256 = h.hashlib.sha256(dataset.read_bytes()).hexdigest()
    try:
        result = h.run_gsm8k(
            rows[:3], args, tmp_path, time.monotonic() + 15, lambda message: None
        )
    finally:
        server.shutdown()
        server.server_close()
    assert (result["correct"], result["errors"], result["truncated"]) == (2, 1, 1)
    assert result["status"] == "failed"
    assert len(requests) == 3 and {path for path, _ in requests} == {"/generate"}
    prefix = "".join(
        f"Question: question {n}\nAnswer: reason #### 42\n\n" for n in range(5)
    )
    assert {body["text"] for _, body in requests} == {
        prefix + f"Question: question {n}\nAnswer:" for n in range(3)
    }
    for _, body in requests:
        assert body["stream"] is False
        assert body["sampling_params"] == {
            "temperature": 0,
            "top_p": 1,
            "max_new_tokens": 1024,
            "stop": ["Question", "Assistant:", "<|separator|>"],
        }
    assert result["manifest"]["shot_overlap_question_ids"] == [0, 1, 2]
    assert result["manifest"]["parallel"] == 16
    assert result["manifest"]["scorer"] == "legacy-last-integer-v1"
    assert result["details"][1]["truncated"] and result["details"][1]["ok"]
    assert (tmp_path / "gsm8k-reference/raw.jsonl").exists()
    assert (tmp_path / "gsm8k-reference/result.json").exists()


def test_error_strings_excluded_from_output_comparisons(tmp_path, monkeypatch):
    args = args_for("--accuracy-requests", "2")
    monkeypatch.setattr(
        h, "chat", lambda *a: (_ for _ in ()).throw(RuntimeError("identical error"))
    )
    result = h.run_accuracy_prompts(
        args, tmp_path, time.monotonic() + 10, lambda message: None
    )
    assert result["sha"] == {}
    assert result["errors"] == 2


def test_global_deadline_prevents_new_commands():
    with pytest.raises(h.BudgetExpired):
        h.execute(
            [sys.executable, "-c", 'raise RuntimeError("must not execute")'],
            time.monotonic() - 1,
        )


def test_http_deadline_with_trickled_body(monkeypatch):
    class Slow(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            try:
                for _ in range(100):
                    self.wfile.write(b"x")
                    self.wfile.flush()
                    time.sleep(0.02)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Slow)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(h, "SERVER_PORT", server.server_port)
    started = time.monotonic()
    try:
        with pytest.raises((h.BudgetExpired, TimeoutError)):
            h.http_raw("/trickle", None, started + 0.15, 30)
        assert time.monotonic() - started < 0.7
    finally:
        server.shutdown()
        server.server_close()


def owned_launch(tmp_path, command):
    state = tmp_path / "owner.json"
    spec = {
        "command": command,
        "token": os.urandom(16).hex(),
        "log": str(tmp_path / "server.log"),
    }
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "prtest_remote.py"),
            "launch",
            str(state),
            json.dumps(spec),
        ],
        check=True,
        capture_output=True,
    )
    return state


def owned_stop(state):
    return subprocess.run(
        [sys.executable, str(ROOT / "prtest_remote.py"), "stop", str(state)],
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("clear_worker_env", [False, True])
def test_owned_cleanup_kills_children_but_preserves_unrelated_process(
    tmp_path, clear_worker_env
):
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    state = owned_launch(
        tmp_path,
        h.shlex.join(
            [
                sys.executable,
                "-c",
                "import os,subprocess,time; env=dict(os.environ); "
                + ('env.pop("PRTEST_RUN_TOKEN",None); ' if clear_worker_env else "")
                + 'subprocess.Popen(["sleep","60"],env=env); time.sleep(60)',
            ]
        ),
    )
    try:
        identity = json.loads(state.read_text())
        deadline = time.monotonic() + 5
        while len(remote.members(identity)) < 2 and time.monotonic() < deadline:
            time.sleep(0.02)
        processes = remote.members(identity)
        assert len(processes) >= 2
        if clear_worker_env:
            assert any(
                info["token"] is None and info["session"] == identity["pid"]
                for info in processes
            )
        assert owned_stop(state).returncode == 0
        assert remote.members(identity) == []
        assert unrelated.poll() is None
    finally:
        owned_stop(state)
        unrelated.terminate()
        unrelated.wait()


def test_reused_pid_is_never_signaled(tmp_path):
    state = owned_launch(tmp_path, "sleep 60")
    original = state.read_text()
    try:
        identity = json.loads(original)
        identity["start"] = "wrong-start-time"
        state.write_text(json.dumps(identity))
        result = owned_stop(state)
        assert result.returncode != 0 and "PID reused" in result.stderr
        assert remote.proc_info(identity["pid"]) is not None
    finally:
        state.write_text(original)
        owned_stop(state)


def test_benchmark_timeout_kills_descendants(tmp_path):
    pidfile = tmp_path / "child.pid"
    script = (
        "import subprocess,time,pathlib; "
        'p=subprocess.Popen(["sleep","60"]); '
        f"pathlib.Path({str(pidfile)!r}).write_text(str(p.pid)); time.sleep(60)"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        h.execute([sys.executable, "-c", script], time.monotonic() + 0.4)
    info = remote.proc_info(pidfile.read_text())
    assert info is None or info["state"] == "Z"


def test_stop_only_reports_docker_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(
        h,
        "run_capture",
        lambda *args: (_ for _ in ()).throw(RuntimeError("Docker offline")),
    )
    assert h.main(["--stop-only", "--state-dir", str(tmp_path)]) == 1


def test_refs_are_pinned_once_with_same_baseline_bracket(monkeypatch):
    args = args_for("--no-fetch")
    calls = []

    def fake_git(args, *argv, **kwargs):
        calls.append(argv)
        return "a" * 40

    monkeypatch.setattr(h, "git", fake_git)
    result = h.pin_refs(
        [h.CASES[0], next(case for case in h.CASES if case[0] == "08-baseline-repeat")],
        args,
        time.monotonic() + 10,
        lambda pins: None,
    )
    assert len(calls) == 1
    assert result["dcp_ascend_testing"]["commit"] == "a" * 40


def test_ratio_refused_for_mismatched_configuration(tmp_path):
    def record(name, config):
        return {
            "case": name,
            "status": "ok",
            "phases": {
                "speed": {
                    "attempts": [
                        {
                            "comparison_config": config,
                            "speed": {
                                "status": "ok",
                                "metrics": {"TTFT avg": 1000, "TTFT P90": 1200},
                            },
                        }
                    ]
                }
            },
        }

    h.summarize(
        [
            record("00-baseline", {"graph": True}),
            record("01-pr-e-chat-encode", {"graph": False}),
        ],
        tmp_path,
        SimpleNamespace(),
    )
    assert "configuration mismatch" in (tmp_path / "SUMMARY.md").read_text()


def test_startup_exception_and_interrupt_always_cleanup(tmp_path, monkeypatch):
    args = args_for("--round", "smoke")
    args.remote_root = "/tmp/prtest-unit"
    args.registry_path = tmp_path / "registry.json"
    args.ais_preflight_record = None
    record = {"case": "00-baseline", "case_env": {}, "phases": {}}
    registry = {"processes": [{"state": "owned-id"}], "worktrees": []}
    runtime = {"model_files_sha256": {}, "versions": {}, "environment": {}, "ops": {}}
    monkeypatch.setattr(h, "require_free_port", lambda: None)
    monkeypatch.setattr(h, "probe_runtime", lambda *a: runtime)
    monkeypatch.setattr(h, "launch_server", lambda *a: ("owned-id", "server.log"))
    stopped = []
    monkeypatch.setattr(h, "stop_server", lambda *a: stopped.append(a[1]))
    monkeypatch.setattr(h, "capture_server_log", lambda *a: "startup failed")
    monkeypatch.setattr(
        h,
        "wait_health",
        lambda *a: (_ for _ in ()).throw(RuntimeError("startup failed")),
    )
    h.run_phase(
        args,
        record,
        "smoke",
        "/tree",
        tmp_path,
        registry,
        time.monotonic() + 10,
        lambda message: None,
        lambda: None,
    )
    assert stopped == ["owned-id"]
    assert record["phases"]["smoke"]["status"] == "failed"
    assert record["phases"]["smoke"]["attempts"][0]["cleanup"] == "ok"
    registry["processes"][0]["cleaned"] = False
    monkeypatch.setattr(
        h, "wait_health", lambda *a: (_ for _ in ()).throw(KeyboardInterrupt())
    )
    with pytest.raises(KeyboardInterrupt):
        h.run_phase(
            args,
            record,
            "accuracy",
            "/tree",
            tmp_path,
            registry,
            time.monotonic() + 10,
            lambda message: None,
            lambda: None,
        )
    assert stopped == ["owned-id", "owned-id"]
    assert record["phases"]["accuracy"]["status"] == "interrupted"


def test_failed_accuracy_does_not_suppress_speed_or_later_cases(tmp_path, monkeypatch):
    dataset = tmp_path / "gsm8k.jsonl"
    dataset.write_text(
        (json.dumps({"question": "What is 2+2?", "answer": "#### 4"}) + "\n") * 5
    )
    args = args_for(
        "--round",
        "all",
        "--cases",
        "00-baseline,01-pr-e-chat-encode",
        "--gsm8k",
        str(dataset),
        "--gsm8k-samples",
        "1",
        "--no-fetch",
        "--container-cann-env",
        "",
        "--output",
        str(tmp_path / "out"),
        "--state-dir",
        str(tmp_path / "state"),
    )
    monkeypatch.setattr(h, "require_free_port", lambda: None)
    monkeypatch.setattr(h, "ais_preflight", lambda *a: {"python": "/test/python"})
    monkeypatch.setattr(h, "git", lambda *a, **kw: "a" * 40)
    monkeypatch.setattr(h, "create_worktree", lambda *a: "/isolated-tree")
    monkeypatch.setattr(h, "cleanup_registry", lambda *a: None)
    calls = []

    def run_phase(
        args, record, phase, worktree, case_dir, registry, deadline, log, checkpoint
    ):
        calls.append((record["case"], phase))
        record["phases"][phase] = {
            "status": "failed" if phase == "accuracy" else "ok",
            "reason": "request failure" if phase == "accuracy" else "",
            "attempts": [],
        }
        checkpoint()

    monkeypatch.setattr(h, "run_phase", run_phase)
    assert (
        h.main(
            [
                "--round",
                "all",
                "--cases",
                args.cases,
                "--gsm8k",
                str(dataset),
                "--gsm8k-samples",
                "1",
                "--no-fetch",
                "--container-cann-env",
                "",
                "--output",
                str(tmp_path / "out"),
                "--state-dir",
                str(tmp_path / "state"),
            ]
        )
        == 1
    )
    assert calls == [
        ("00-baseline", "accuracy"),
        ("00-baseline", "speed"),
        ("01-pr-e-chat-encode", "accuracy"),
        ("01-pr-e-chat-encode", "speed"),
    ]
    rows = json.loads((tmp_path / "out/results.json").read_text())
    assert len(rows) == 2 and all(row["status"] == "failed" for row in rows)


def test_runtime_operator_absence_keeps_probe_evidence(tmp_path, monkeypatch):
    args = args_for("--round", "smoke")
    args.registry_path = tmp_path / "registry.json"
    record = {
        "case": "04-pr-f-sfa-kv-8bit",
        "case_env": {"SGLANG_NPU_SFA_KV_QUANT_READ": "True"},
        "phases": {},
    }
    registry = {"processes": [], "worktrees": []}
    monkeypatch.setattr(h, "require_free_port", lambda: None)
    monkeypatch.setattr(
        h,
        "probe_runtime",
        lambda *a: {"ops": {"npu_sparse_flash_attention_lse": False}},
    )
    h.run_phase(
        args,
        record,
        "smoke",
        "/tree",
        tmp_path,
        registry,
        time.monotonic() + 10,
        lambda message: None,
        lambda: None,
    )
    attempt = record["phases"]["smoke"]["attempts"][0]
    assert "required torch.ops" in attempt["reason"]
    assert attempt["runtime"]["ops"]["npu_sparse_flash_attention_lse"] is False


def test_timeout_kills_child_that_ignores_term(tmp_path):
    pidfile = tmp_path / "resistant-child.pid"
    child = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
    script = (
        "import subprocess,time,pathlib; "
        f'p=subprocess.Popen([{sys.executable!r},"-c",{child!r}]); '
        f"pathlib.Path({str(pidfile)!r}).write_text(str(p.pid)); time.sleep(60)"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        h.execute([sys.executable, "-c", script], time.monotonic() + 0.5)
    pid = pidfile.read_text()
    deadline = time.monotonic() + 2
    while (
        (info := remote.proc_info(pid))
        and info["state"] != "Z"
        and time.monotonic() < deadline
    ):
        time.sleep(0.01)
    assert info is None or info["state"] == "Z"


def test_shared_case_deadline_skips_speed_and_records_every_remaining_case(
    tmp_path, monkeypatch
):
    dataset = tmp_path / "gsm8k.jsonl"
    dataset.write_text(
        (json.dumps({"question": "What is 2+2?", "answer": "#### 4"}) + "\n") * 5
    )
    clock = [100.0]
    monkeypatch.setattr(h, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(h, "require_free_port", lambda: None)
    monkeypatch.setattr(h, "ais_preflight", lambda *a: {})
    monkeypatch.setattr(h, "git", lambda *a, **kw: "a" * 40)
    monkeypatch.setattr(h, "create_worktree", lambda *a: "/isolated-tree")
    monkeypatch.setattr(h, "cleanup_registry", lambda *a: None)
    calls = []

    def run_phase(
        args, record, phase, worktree, case_dir, registry, deadline, log, checkpoint
    ):
        calls.append(phase)
        assert deadline == 160.0
        record["phases"][phase] = {"status": "ok", "attempts": []}
        clock[0] = 200.0

    monkeypatch.setattr(h, "run_phase", run_phase)
    assert (
        h.main(
            [
                "--round",
                "all",
                "--cases",
                "00-baseline,01-pr-e-chat-encode",
                "--gsm8k",
                str(dataset),
                "--gsm8k-samples",
                "1",
                "--no-fetch",
                "--container-cann-env",
                "",
                "--case-minutes",
                "1",
                "--hours",
                ".02",
                "--output",
                str(tmp_path / "out"),
                "--state-dir",
                str(tmp_path / "state"),
            ]
        )
        == 1
    )
    assert calls == ["accuracy"]
    records = json.loads((tmp_path / "out/results.json").read_text())
    assert len(records) == 2
    assert records[0]["phases"]["speed"]["status"] == "skipped"
    assert records[1]["status"] == "skipped"
    assert all(phase["status"] == "skipped" for phase in records[1]["phases"].values())


def test_resolved_runtime_changes_prevent_ratios(tmp_path):
    common = dict(
        model_path="/model",
        load_format="auto",
        dtype="bfloat16",
        tp_size=16,
        dcp_size=16,
        attention_backend="ascend",
        quantization="modelslim",
    )
    base = h.resolved_comparison({"requested": "same"}, {"server": common})
    changed = h.resolved_comparison(
        {"requested": "same"}, {"server": dict(common, dtype="float16")}
    )
    assert base != changed
    assert h.resolved_comparison({"requested": "same"}, {"server": {}}) is None
    records = []
    for name, config in [("00-baseline", base), ("01-pr-e-chat-encode", changed)]:
        records.append(
            {
                "case": name,
                "status": "ok",
                "phases": {
                    "speed": {
                        "attempts": [
                            {
                                "comparison_config": config,
                                "speed": {
                                    "status": "ok",
                                    "metrics": {"TTFT avg": 1000, "TTFT P90": 1200},
                                },
                            }
                        ]
                    }
                },
            }
        )
    h.summarize(records, tmp_path, SimpleNamespace())
    assert "configuration mismatch" in (tmp_path / "SUMMARY.md").read_text()


def test_legacy_registry_cleans_envless_worker_after_leader_exit(tmp_path):
    script = (
        "import os,subprocess,time; env=dict(os.environ); "
        'env.pop("PRTEST_RUN_TOKEN",None); '
        'subprocess.Popen(["sleep","60"],env=env); time.sleep(.3)'
    )
    state = owned_launch(tmp_path, h.shlex.join([sys.executable, "-c", script]))
    identity = json.loads(state.read_text())
    identity.pop("boot_id")
    state.write_text(json.dumps(identity))
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            leader = remote.proc_info(identity["pid"])
            if not leader or leader["state"] == "Z":
                break
            time.sleep(0.02)
        workers = remote.members(identity)
        assert workers and all(info["token"] is None for info in workers)
        assert owned_stop(state).returncode == 0
        assert remote.members(identity) == []
    finally:
        owned_stop(state)


def test_previous_boot_identity_is_rejected(tmp_path):
    state = owned_launch(tmp_path, "sleep 60")
    original = state.read_text()
    try:
        identity = json.loads(original)
        identity["boot_id"] = "different-boot"
        state.write_text(json.dumps(identity))
        result = owned_stop(state)
        assert result.returncode != 0 and "different boot" in result.stderr
        assert remote.proc_info(identity["pid"]) is not None
    finally:
        state.write_text(original)
        owned_stop(state)


def test_e2e_capacity_and_cache_guards():
    args = args_for("--round", "speed")
    server = dict(
        max_running_requests=16, disable_radix_cache=False, page_size=128, dcp_size=16
    )
    result = h.speed_capacity(server, "max_total_num_tokens=1196032", args)
    assert result["required"] == 1189888 and result["status"] == "ok"
    assert (
        h.speed_capacity(server, "max_total_num_tokens=1000000", args)["status"]
        == "capacity-blocked"
    )
    assert (
        h.speed_capacity(server, "no capacity evidence", args)["status"]
        == "capacity-blocked"
    )
    with pytest.raises(RuntimeError, match="radix"):
        h.speed_capacity(dict(server, disable_radix_cache=True), "", args)
    with pytest.raises(RuntimeError, match="below concurrency"):
        h.speed_capacity(dict(server, max_running_requests=1), "", args)
    assert (
        h.prefix_evidence(
            "Prefill batch, #new-seq: 1, #new-token: 100, #cached-token: 0"
        )["status"]
        == "cache-hit-unverified"
    )


def test_monitored_client_failure_stops_its_process(tmp_path):
    pidfile = tmp_path / "pid"
    script = f"import os,pathlib,time; pathlib.Path({str(pidfile)!r}).write_text(str(os.getpid())); time.sleep(60)"

    def monitor():
        if pidfile.exists():
            raise RuntimeError("server exited")

    with pytest.raises(RuntimeError, match="server exited"):
        h.execute(
            [sys.executable, "-c", script],
            time.monotonic() + 5,
            log_path=tmp_path / "client.log",
            monitor=monitor,
            monitor_interval=0.02,
        )
    info = remote.proc_info(pidfile.read_text())
    assert info is None or info["state"] == "Z"


def test_reference_e2e_wrapper_and_artifact_capture(tmp_path, monkeypatch):
    tool = tmp_path / "tool"
    tool.mkdir()
    model = tmp_path / "model"
    model.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    fixture = ARCHIVE / "00-baseline/ais.log"
    (tool / "aisbench_test.py").write_text(
        "import json,os,sys\nfrom pathlib import Path\n"
        f'Path("arguments.json").write_text(json.dumps(sys.argv[1:]))\n'
        'Path("artifact").mkdir(exist_ok=True)\n'
        'Path("artifact/result.json").write_text("{\\"requests\\":16}")\n'
        'Path("aisbench.log").write_text("Current exp folder: artifact\\n")\n'
        'Path("aisbench_result.csv").write_text("requests,ttft\\n16,36104.9553\\n")\n'
        f"print(Path({str(fixture)!r}).read_text(),flush=True)\n"
    )
    binary = tmp_path / "ais_bench"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    conda = tmp_path / "conda"
    conda.write_text("#!/bin/bash\necho 'conda() { return 0; };'\n")
    conda.chmod(0o755)
    monkeypatch.setenv("CONDA_EXE", str(conda))
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    env = tmp_path / "env.sh"
    env.write_text("conda activate fixture\n")
    server_info = dict(
        max_running_requests=16,
        disable_radix_cache=False,
        page_size=128,
        dcp_size=16,
        context_length=1024000,
        chunked_prefill_size=16384,
    )
    routes = []

    class Server(BaseHTTPRequestHandler):
        def do_GET(self):
            routes.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(
                json.dumps(server_info).encode() if self.path == "/server_info" else b""
            )

        def do_POST(self):
            routes.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Cache flushed")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Server)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(h, "SERVER_PORT", server.server_port)
    (tool / "config.py").write_text(
        f'HOST_IP="127.0.0.1"\nHOST_PORT={server.server_port}\nMODEL_PATH={str(model)!r}\nWORK_PATH={str(work)!r}\n'
    )
    args = args_for("--round", "speed", "--ais-dir", str(tool), "--ais-env", str(env))
    case = tmp_path / "output"
    case.mkdir()
    captures = [0]

    def capture(*a):
        captures[0] += 1
        text = "max_total_num_tokens=1196032\n"
        if captures[0] > 1:
            text += "[ TP0] Prefill batch, #new-seq: 1, #new-token: 10000, #cached-token: 989184\n"
        (case / "server.log").write_text(text)
        return text

    monkeypatch.setattr(h, "capture_server_log", capture)
    monkeypatch.setattr(
        h, "remote_action", lambda *a: {"identity": {"pid": 1}, "members": [{"pid": 1}]}
    )
    try:
        result = h.run_aisbench(
            args,
            case,
            time.monotonic() + 20,
            server_info,
            "owned",
            "remote.log",
            lambda message: None,
            "commit",
        )
    finally:
        server.shutdown()
        server.server_close()
    assert result["status"] == "ok", result
    assert result["metrics"]["TTFT avg"] == 36104.9553
    assert result["metrics"]["TPOT avg"] is None
    assert result["preparation"]["total_req"] == 1
    assert result["prefix_cache"]["cached_tokens"] == 989184
    profile = next((case / "profile0").iterdir())
    assert (profile / "aisbench_result.csv").exists()
    assert (profile / "aisbench_outputs/artifact/result.json").exists()
    assert (profile / "config.py").exists() and (profile / "server_info.json").exists()
    assert json.loads((case / "ais-environment.json").read_text())["PROFILE"] == "0"
    assert len([route for route in routes if route.startswith("/flush_cache")]) == 1
    assert not any("profile" in route for route in routes)
    command = json.loads((tool / "arguments.json").read_text())
    assert command[command.index("--data_num") + 1] == "16"
    assert command[command.index("--concurrency") + 1] == "16"


def test_monitored_timeout_kills_detached_client(tmp_path):
    pidfile = tmp_path / "detached.pid"
    child = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)"
    script = (
        "import subprocess,pathlib,time; "
        f'p=subprocess.Popen([{sys.executable!r},"-c",{child!r}],start_new_session=True); '
        f"pathlib.Path({str(pidfile)!r}).write_text(str(p.pid)); time.sleep(60)"
    )
    with pytest.raises(h.BudgetExpired):
        h.execute(
            [sys.executable, "-c", script],
            time.monotonic() + 0.4,
            log_path=tmp_path / "out.log",
            monitor=lambda: None,
            monitor_interval=0.05,
        )
    deadline = time.monotonic() + 2
    while (
        (info := remote.proc_info(pidfile.read_text()))
        and info["state"] != "Z"
        and time.monotonic() < deadline
    ):
        time.sleep(0.01)
    assert info is None or info["state"] == "Z"


@pytest.mark.parametrize("selected", [True, False])
def test_server_environment_scrubs_toolkit_and_feature_leaks(tmp_path, selected):
    old = tmp_path / "old-cann"
    alias = tmp_path / "old-alias"
    old.mkdir()
    alias.symlink_to(old, target_is_directory=True)
    unrelated = tmp_path / "old-cann-extra"
    worktree = tmp_path / "work tree"
    worktree.mkdir()
    current = tmp_path / "selected cann"
    binding = current / "runtime/python/site-packages"
    binding.mkdir(parents=True)
    script = current / "set_env.sh"
    script.write_text(
        'test -z "${ASCEND_OPP_PATH:-}" || return 21\n'
        f"export ASCEND_HOME_PATH={h.shlex.quote(str(current))}\n"
        f'export LD_LIBRARY_PATH={h.shlex.quote(str(current / "lib"))}:"${{LD_LIBRARY_PATH:-}}"\n'
        # Exercise binding discovery when the setup script does not set PYTHONPATH.
        "export SGLANG_FROM_SETUP=unexpected\n"
    )
    env = dict(os.environ)
    env.update(
        ASCEND_HOME_PATH=str(old),
        CANN_HOME=str(alias),
        ASCEND_TOOLKIT_HOME=str(old),
        ASCEND_OPP_PATH=str(old / "opp"),
        TBE_IMPL_PATH=str(old / "impl"),
        PATH=str(alias / "bin") + os.pathsep + os.defpath,
        PYTHONPATH=os.pathsep.join([str(alias / "python"), str(unrelated)]),
        LD_LIBRARY_PATH=os.pathsep.join([str(old / "lib"), str(unrelated)]),
        SGLANG_UNKNOWN_INHERITED="unexpected",
        SGLANG_NPU_INDEXER_INT8="inherited",
        HTTP_PROXY="http://inherited.invalid:8080",
        https_proxy="http://inherited.invalid:8080",
    )
    env.pop("LD_PRELOAD", None)
    env.pop("CONDA_PREFIX", None)
    args = SimpleNamespace(
        container_python=sys.executable,
        cann_env_resolved=str(script) if selected else None,
    )
    argv = [
        sys.executable,
        "-c",
        "import json,os,sys; print(json.dumps(dict(env=dict(os.environ), argv=sys.argv[1:], cwd=os.getcwd())))",
        "literal;$(false)",
    ]
    flags = {"SGLANG_NPU_INDEXER_INT8": "case-value"}
    result = subprocess.run(
        ["bash", "-c", h.server_shell(args, str(worktree), flags, argv)],
        env=env,
        text=True,
        capture_output=True,
        timeout=15,
        check=True,
    )
    captured = json.loads(result.stdout)
    actual = captured["env"]
    assert captured["argv"] == ["literal;$(false)"]
    assert captured["cwd"] == str(worktree)
    assert actual["SGLANG_NPU_INDEXER_INT8"] == "case-value"
    assert "SGLANG_UNKNOWN_INHERITED" not in actual
    assert "SGLANG_FROM_SETUP" not in actual
    assert "HTTP_PROXY" not in actual and "https_proxy" not in actual
    assert actual["NO_PROXY"] == h.BASE_ENV["NO_PROXY"]
    assert actual["PYTHONPATH"].split(os.pathsep)[0] == str(worktree / "python")
    assert str(unrelated) in actual["PYTHONPATH"].split(os.pathsep)
    if selected:
        for key in ("PATH", "PYTHONPATH", "LD_LIBRARY_PATH"):
            assert str(old) + "/" not in actual[key]
            assert str(alias) not in actual[key]
        assert actual["ASCEND_HOME_PATH"] == str(current)
        assert "CANN_HOME" not in actual and "ASCEND_TOOLKIT_HOME" not in actual
        assert "ASCEND_OPP_PATH" not in actual and "TBE_IMPL_PATH" not in actual
        assert str(binding) in actual["PYTHONPATH"].split(os.pathsep)
    else:
        assert actual["ASCEND_HOME_PATH"] == str(old)
        assert str(alias / "python") in actual["PYTHONPATH"].split(os.pathsep)


def test_server_environment_preload_parity(tmp_path):
    library = next(
        (
            Path(path)
            for path in (
                "/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
                "/usr/lib/aarch64-linux-gnu/libstdc++.so.6",
                "/usr/lib64/libstdc++.so.6",
            )
            if Path(path).is_file()
        ),
        None,
    )
    if library is None:
        pytest.skip("a loadable libstdc++ is required for the exec regression")
    old = tmp_path / "old"
    old.mkdir()
    old_library = old / "libstdc++.so.6"
    # Keep this link inside the obsolete root so resolve() still sees it there.
    old_library.write_bytes(library.read_bytes())
    conda = tmp_path / "conda"
    (conda / "lib").mkdir(parents=True)
    conda_library = conda / "lib/libstdc++.so.6"
    conda_library.symlink_to(library)
    current = tmp_path / "current"
    current.mkdir()
    setup = current / "set_env.sh"
    setup.write_text(f"export CANN_HOME={h.shlex.quote(str(current))}\n")
    env = dict(
        os.environ,
        ASCEND_HOME_PATH=str(old),
        CONDA_PREFIX=str(conda),
        LD_PRELOAD=f"{old_library} {library}",
    )
    args = SimpleNamespace(
        container_python=sys.executable, cann_env_resolved=str(setup)
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            h.server_shell(
                args,
                str(tmp_path),
                {},
                [sys.executable, "-c", "import os; print(os.environ['LD_PRELOAD'])"],
            ),
        ],
        env=env,
        text=True,
        capture_output=True,
        check=True,
        timeout=15,
    )
    assert result.stdout.strip().split(":") == [str(library), str(conda_library)]


@pytest.mark.parametrize("script", ["return 17\n", "export UNRELATED=value\n"])
def test_selected_cann_environment_failure_prevents_server_exec(tmp_path, script):
    setup = tmp_path / "set_env.sh"
    setup.write_text(script)
    marker = tmp_path / "server-started"
    args = SimpleNamespace(
        container_python=sys.executable, cann_env_resolved=str(setup)
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            h.server_shell(
                args,
                str(tmp_path),
                {},
                [
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(marker)!r}).touch()",
                ],
            ),
        ],
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode != 0
    assert not marker.exists()


def test_ep_default_and_explicit_tp_diagnostic():
    args = args_for("--round", "speed")
    argv = h.server_argv(args, args.memfrac, True, phase="speed")
    assert argv[argv.index("--moe-a2a-backend") + 1] == "deepep"
    assert argv[argv.index("--ep-size") + 1] == "16"
    tp = args_for("--round", "accuracy", "--moe-mode", "tp")
    argv = h.server_argv(tp, tp.memfrac, True, phase="accuracy")
    assert argv[argv.index("--moe-a2a-backend") + 1] == "none"
    assert argv[argv.index("--ep-size") + 1] == "1"
    mega = next(case for case in h.CASES if case[0] == "09-pr1-megamoe")
    argv = h.server_argv(args, args.memfrac, True, mega[4], phase="speed")
    assert argv.count("--moe-a2a-backend") == 1
    assert argv[argv.index("--moe-a2a-backend") + 1] == "ascend_fuseep"


def test_overnight_budget_brackets_and_isolated_diagnostics(capsys):
    args = args_for("--suite", "overnight12")
    cases = h.plan_cases(args)
    assert args.hours == 12
    assert cases[0][0] == "00-baseline"
    assert cases[-1][0] == "08-baseline-repeat"
    assert sum(h.case_settings(args, case[0]).case_minutes for case in cases) == 675
    assert len(cases) == 13
    h.dry_run(args, cases)
    output = capsys.readouterr().out
    for name, _, env, _, case_args in cases:
        effective = h.case_settings(args, name)
        if "tp-" in name:
            assert effective.round == "accuracy"
            assert effective.gsm8k_samples == 8
            assert effective.case_minutes == 15
        if name == "10-ep-p4-eager":
            assert effective.round == "accuracy"
            assert effective.chunk == -1
            assert effective.context == 32768
            assert effective.gsm8k_parallel == 64
            assert effective.accuracy_server_requests == 64
        for phase in h.phase_names(effective):
            graph = effective.decode_graph == "on"
            argv = h.server_argv(
                effective, effective.memfrac, graph, case_args, phase=phase
            )
            assert argv.count("--moe-a2a-backend") == 1
            assert argv.count("--cuda-graph-backend-decode") == 1
            assert (
                h.server_shell(args, "<run-root>/worktrees/" + name, env, argv)
                in output
            )
    assert (
        "speed: fresh server"
        not in output.split("== 10-ep-p4-eager")[1].split("== 08-baseline-repeat")[0]
    )
    assert "8 samples in file order, parallel=16" in output
    assert "64 samples in file order, parallel=64" in output


@pytest.mark.parametrize(
    "extra",
    [
        ["--round", "accuracy"],
        ["--moe-mode", "tp"],
        ["--cases", "00-baseline"],
        ["--hours", "13"],
    ],
)
def test_overnight_rejects_incompatible_overrides(extra):
    with pytest.raises(ValueError):
        args_for("--suite", "overnight12", *extra)


def test_tp_fix_is_an_explicit_accuracy_only_case():
    args = args_for("--cases", "13-tp-bias-fix")
    (case,) = h.plan_cases(args)
    assert case[1] == "npu-w4a8-tp-bias-fix"
    effective = h.case_settings(args, case[0])
    assert h.phase_names(effective) == ["accuracy"]
    assert effective.gsm8k_samples == 64
    argv = h.server_argv(effective, effective.memfrac, True, case[4], phase="accuracy")
    assert argv[argv.index("--ep-size") + 1] == "1"
    assert argv[argv.index("--moe-a2a-backend") + 1] == "none"
    assert "13-tp-bias-fix" not in [case[0] for case in h.plan_cases(args_for())]
    assert "13-tp-bias-fix" not in h.OVERNIGHT12_ORDER
