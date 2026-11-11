#!/usr/bin/env python3
"""Speed + accuracy test harness for the six Ascend NPU PR series (PRs #2-#7).

Follows the established box workflow (run_a3_nightly.py / run_bde_overnight.py
lineage): one fresh server per case, one measured AISBench invocation per
server, paired greedy accuracy vs the baseline branch, host-side orchestration
over `docker exec`, live per-case logs, and an honest SUMMARY.md.

Cases (in order; baseline bracket first + last):

  00-baseline              dcp_ascend_testing           all new flags off
  01-pr-e-chat-encode      npu-chat-encode-pool         PR #2  (E)
  02-pr-b-indexer-int8     npu-dsa-indexer-int8         PR #3  (B)
  03-pr-prefetch-packed    npu-dcp-packed-read          PR #4  (C1/D2p/D2v2)
  04-pr-f-sfa-kv-8bit      npu-sfa-kv-8bit              PR #5  (F1/F2)
  05-pr-ts-token-sharded   npu-dsa-cp-token-sharded     PR #6  (D1v3/D2v3/D2v4)
  06-pr-moe-int8           npu-dsa-cp-moe-int8          PR #7  (M, on top of ts)
  07-combined              npu-bde-combined             everything on
  08-baseline-repeat       dcp_ascend_testing           bracket drift check

Rounds:
  smoke     server starts, /health + one greedy generation per case
  accuracy  paired greedy prompt set (+ optional GSM8K subset) per case
  speed     AISBench prefix-cache 1M TTFT per case
  all       accuracy then speed on the same fresh server (default)

Usage (on the A3 HOST):
  python3 run_pr_tests.py --round all --hours 10 --case-minutes 60
  python3 run_pr_tests.py --round speed
  python3 run_pr_tests.py --stop-only
  python3 run_pr_tests.py --dry-run

Nothing is pushed by this script; branches are fetched read-only from the
fork (default https://github.com/TamirBaydasov/sglang.git) or used from the
local checkout with --no-fetch.
"""

import argparse
import datetime
import fcntl
import json
import os
import re
import shlex
import socket
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# --------------------------------------------------------------------------
# Box constants (defaults follow the GLM A3 nightly lineage; override via CLI)
# --------------------------------------------------------------------------
AIS_ENV_CANDIDATES = ("/home/tbaydasov/set_env.sh", "/home/tbaydasovset_env.sh")
AIS_DIR_DEFAULT = "/home/tbaydasov/aisbench/aisbench_auto_tools_prefix"
FORK_URL_DEFAULT = "https://github.com/TamirBaydasov/sglang.git"

AIS_SHELL = """\
set -e
if ! declare -F conda >/dev/null; then
    conda_command=${CONDA_EXE:-conda}
    if ! command -v "$conda_command" >/dev/null; then
        conda_command=conda
    fi
    if command -v "$conda_command" >/dev/null; then
        conda_hook=$("$conda_command" shell.bash hook)
        eval "$conda_hook"
    fi
fi
source "$1"
cd "$2"
unset HTTP_PROXY http_proxy HTTPS_PROXY https_proxy ALL_PROXY all_proxy
export NO_PROXY=localhost,127.0.0.1,::1
exec python3 aisbench_test.py "${@:3}"
"""

# BDE serve recipe (archived p4_exit1_serve_dcp.sh via launch_bde_v2.py).
BASE_ENV = {
    "SGLANG_NPU_ENABLE_DSA_CP": "True",
    "SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP": "True",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_BUFFSIZE": "1000",
    "DEEPEP_HCCL_BUFFSIZE": "2500",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_NPU_USE_MULTI_STREAM": "1",
}

SERVER_PORT = 30000
BASE_SERVER_ARGS = [
    "--device", "npu",
    "--attention-backend", "ascend",
    "--tp-size", "16",
    "--dcp-size", "16",
    "--cuda-graph-backend-prefill", "disabled",
    "--cuda-graph-backend-decode", "disabled",
    "--context-length", "1048576",
    "--chunked-prefill-size", "32768",
    "--mem-fraction-static", "0.88",
    "--port", str(SERVER_PORT),
    # Packed DCP reads require one request per prefill batch.
    "--prefill-max-requests", "1",
    "--enable-metrics",
]

# --------------------------------------------------------------------------
# Case table: (name, ref, env, note).  Baseline bracket = first + last.
# --------------------------------------------------------------------------
CASES = [
    ("00-baseline", "dcp_ascend_testing", {}, "control, bracket start"),
    ("01-pr-e-chat-encode", "npu-chat-encode-pool", {
        "SGLANG_ENABLE_CHAT_ENCODE_POOL": "True",
        "SGLANG_CHAT_ENCODE_MEMO": "True",
    }, "PR #2 (E): pool on, memo on; RAW stays off (its documented default)"),
    ("02-pr-b-indexer-int8", "npu-dsa-indexer-int8", {
        "SGLANG_NPU_INDEXER_INT8": "True",
    }, "PR #3 (B): int8 index-K + per-token fp16 scale"),
    ("03-pr-prefetch-packed", "npu-dcp-packed-read", {
        "SGLANG_NPU_ENABLE_DCP_PACKED_READ": "True",
        "SGLANG_NPU_ENABLE_DCP_GATHER_PREFETCH": "True",
        "SGLANG_NPU_ENABLE_PG_PRIORITY": "True",
    }, "PR #4 (C1/D2p/D2v2): packed read + prime + chunked gather"),
    ("04-pr-f-sfa-kv-8bit", "npu-sfa-kv-8bit", {
        "SGLANG_NPU_SFA_KV_8BIT": "True",
        "SGLANG_NPU_SFA_KV_QUANT_READ": "True",
    }, "PR #5 (F): 656 B/row packed KV; F2 in-kernel dequant on"),
    ("05-pr-ts-token-sharded", "npu-dsa-cp-token-sharded", {
        "SGLANG_NPU_DSA_CP_TOKEN_SHARDED": "True",
        "SGLANG_NPU_DSA_CP_Q_FULL_WIDTH": "True",
        "SGLANG_NPU_DSA_CP_LOCAL_TOPK": "True",
        "SGLANG_NPU_DCP_DECODE_Q_PREISSUE": "True",
        "SGLANG_NPU_STREAM_O_PROJ": "True",
    }, "PR #6 (D1v3): bf16 self-sufficient MoE input"),
    ("06-pr-moe-int8", "npu-dsa-cp-moe-int8", {
        "SGLANG_NPU_DSA_CP_TOKEN_SHARDED": "True",
    }, "PR #7 (M): int8 MoE input (unconditional under the ts flag)"),
    ("07-combined", "npu-bde-combined", {
        "SGLANG_ENABLE_CHAT_ENCODE_POOL": "True",
        "SGLANG_CHAT_ENCODE_MEMO": "True",
        "SGLANG_NPU_INDEXER_INT8": "True",
        "SGLANG_NPU_ENABLE_DCP_PACKED_READ": "True",
        "SGLANG_NPU_ENABLE_DCP_GATHER_PREFETCH": "True",
        "SGLANG_NPU_ENABLE_PG_PRIORITY": "True",
        "SGLANG_NPU_SFA_KV_8BIT": "True",
        "SGLANG_NPU_SFA_KV_QUANT_READ": "True",
        "SGLANG_NPU_DSA_CP_TOKEN_SHARDED": "True",
        "SGLANG_NPU_DSA_CP_Q_FULL_WIDTH": "True",
        "SGLANG_NPU_DSA_CP_LOCAL_TOPK": "True",
        "SGLANG_NPU_DCP_DECODE_Q_PREISSUE": "True",
        "SGLANG_NPU_STREAM_O_PROJ": "True",
    }, "full deployment configuration (local-only branch; see README)"),
    ("08-baseline-repeat", "dcp_ascend_testing", {}, "control, bracket end"),
]

# Attribution overrides: --flags-overrides 03:SGLANG_NPU_DCP_GATHER_CHUNKS=1,...
# applied AFTER the case env so any case can be re-pointed without editing.

# --------------------------------------------------------------------------
# Fixed greedy accuracy set.  The long-prefix prompts exercise the DCP gather
# / packed read / token-shard / SFA rows paths; short ones pin general parity.
# --------------------------------------------------------------------------
LONG_PARAGRAPH = (
    "The assembly plant runs three shifts a day and tracks every pallet "
    "with a barcode scanner at the loading dock. Maintenance windows are "
    "scheduled on Sunday nights, and the quality team samples one unit per "
    "hundred for a full teardown inspection. "
)

ACCURACY_PROMPTS = [
    ("short-math", "What is 17 * 23 - 91? Give the number only."),
    ("short-reason", "A train leaves at 14:05 and travels for 2h 50m. At what time does it arrive? Answer with HH:MM."),
    ("short-code", "In Python, write a one-line function that reverses a string."),
    ("short-translate", "Translate to English: 今天下午三点开会，请准时参加。"),
    ("short-fact", "Name the largest planet in the solar system with one word."),
]

GSM8K_EXEMPLARS = [
    ("Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
     "Weng earns 12/60 = $0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $10. #### 10"),
    ("Betty is saving money to buy a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
     "Betty has 100/2 = $50. Her grandparents gave 15*2 = $30. Betty now has 50+15+30 = $95. She needs 100-95 = $5 more. #### 5"),
    ("James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
     "He writes 3*2=6 pages a week (per friend-set) times 2 friends = 12 pages per week. Per year 12*52 = 624. #### 624"),
    ("A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
     "It takes 2/2 = 1 bolt of white fiber. So the total is 2+1 = 3 bolts. #### 3"),
    ("Ken had fifty apples. He gave half to James and sold the rest for $0.60 each. How much did he earn?",
     "He gave away 50/2 = 25. He sold 25 apples at 0.60 = $15. #### 15"),
]


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")


# --------------------------------------------------------------------------
# HTTP client (host side; container uses host networking)
# --------------------------------------------------------------------------
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def http(path, payload=None, timeout=10):
    request = urllib.request.Request(
        f"http://127.0.0.1:{SERVER_PORT}" + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with _OPENER.open(request, timeout=timeout) as response:
        return json.loads(response.read())


def health_ok():
    try:
        http("/health", timeout=10)
        return True
    except Exception:
        return False


def generate(text, max_new_tokens=64):
    data = http(
        "/generate",
        {"text": text, "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens}},
        timeout=7200,
    )
    return data.get("text", ""), data.get("meta_info", {})


# --------------------------------------------------------------------------
# Long prompt builder (deterministic; ~4 chars/token heuristic)
# --------------------------------------------------------------------------
def build_long_prompt(target_tokens, suffix):
    chars = max(1024, target_tokens * 4)
    body = (LONG_PARAGRAPH * (chars // len(LONG_PARAGRAPH) + 1))[:chars]
    return body + "\n\n" + suffix


# --------------------------------------------------------------------------
# GSM8K
# --------------------------------------------------------------------------
_NUM_RE = re.compile(r"-?\$?[\d,]*\.?\d+")


def last_number(text):
    nums = _NUM_RE.findall(text.replace(",", "").replace("$", ""))
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None


def gold_number(answer_field):
    part = answer_field.split("####")[-1].strip().replace(",", "").replace("$", "")
    try:
        return float(part)
    except ValueError:
        return None


def load_gsm8k(path, samples, seed):
    import random

    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"no rows in {path}")
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:samples]


def run_gsm8k(rows, log):
    shots = "\n\n".join(f"Question: {q}\nAnswer: {a}" for q, a in GSM8K_EXEMPLARS)
    correct = 0
    details = []
    for i, row in enumerate(rows):
        question = "Question: " + row["question"] + "\nAnswer:"
        try:
            text, _ = generate(shots + "\n\n" + question, max_new_tokens=512)
        except Exception as exc:  # noqa: BLE001 - record and continue
            details.append({"id": i, "error": str(exc)[:200]})
            log(f"    gsm8k[{i}] request error: {exc}")
            continue
        got, want = last_number(text), gold_number(row["answer"])
        ok = got is not None and want is not None and abs(got - want) < 1e-6
        correct += int(ok)
        details.append({"id": i, "ok": ok, "got": got, "want": want})
        if (i + 1) % 25 == 0:
            log(f"    gsm8k {i + 1}/{len(rows)}: {correct}/{i + 1} correct")
    return {"n": len(rows), "correct": correct,
            "accuracy": correct / len(rows) if rows else None, "details": details}


# --------------------------------------------------------------------------
# AISBench invocation + TTFT parsing
# --------------------------------------------------------------------------
def parse_ttft_lines(text):
    """Metrics are logged as JSON objects containing the key 'TTFT avg' (ms)."""
    metrics = []
    for line in text.splitlines():
        if "TTFT avg" in line and "{" in line:
            candidate = line[line.index("{"):]
            try:
                item = json.loads(candidate)
            except ValueError:
                continue
            if isinstance(item, dict) and "TTFT avg" in item:
                metrics.append(item)
    return metrics


def run_aisbench(ais_dir, ais_env, args, log_path):
    command = [
        "bash", "-c", AIS_SHELL,
        ais_env, ais_dir,
        "--input_len", str(args.length),
        "--output_len", "1",
        "--request_rate", "0",
        "--dataset_type", "prefix_cache",
        "--prefix_test",
        "--seed", "4",
        "--repeat_rate", "0.99",
        "--data_num", str(args.requests),
        "--concurrency", str(args.requests),
    ]
    env = dict(os.environ)
    env["NO_PROXY"] = env["no_proxy"] = "localhost,127.0.0.1,::1"
    with open(log_path, "w") as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                 env=env, timeout=args.benchmark_timeout)
    text = Path(log_path).read_text(errors="replace")
    metrics = parse_ttft_lines(text)
    return process.returncode, metrics, text


# --------------------------------------------------------------------------
# Docker helpers
# --------------------------------------------------------------------------
def docker(container):
    return ["docker", "exec", container]


def run_capture(container, cmd, timeout=120):
    try:
        return subprocess.run([*docker(container), "bash", "-c", cmd],
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            args=[], returncode=124, stdout="",
            stderr=f"timeout after {timeout}s: {cmd[:120]}")


# Set from --git-proxy in main(); the box networks reach GitHub only through
# the ghfast.top proxy the manual launch commands apply via insteadOf.
GIT_PROXY = []


def git_in_repo(container, repo, *git_args, timeout=600):
    quoted = " ".join("'" + a.replace("'", "'\\''") + "'" for a in git_args)
    prefix = f"-c {shlex.quote(GIT_PROXY[0])} " if GIT_PROXY else ""
    return run_capture(container, f"git {prefix}-C '{repo}' {quoted}", timeout=timeout)


# --------------------------------------------------------------------------
# Server lifecycle
# --------------------------------------------------------------------------
def launch_server(container, repo, env, args, case_dir, log):
    remote_log = f"/tmp/prtest-server-{case_dir.name}.log"
    env_str = " ".join(f"{k}={v}" for k, v in {**BASE_ENV, **env}.items())
    python = args.container_python
    model = args.model
    server_args = list(BASE_SERVER_ARGS)
    server_args[server_args.index("--mem-fraction-static") + 1] = str(args.memfrac)
    server_args[server_args.index("--chunked-prefill-size") + 1] = str(args.chunk)
    if args.context:
        server_args[server_args.index("--context-length") + 1] = str(args.context)
    server_args += shlex_split(args.extra_server_args)
    command = (
        f"cd '{repo}' && setsid env {env_str} {python} -m sglang.launch_server "
        f"--load-format auto --model-path '{model}' "
        + " ".join(server_args)
        + f" > '{remote_log}' 2>&1 & echo $! > '{remote_log}.pid'"
    )
    log(f"    launching server (pid file {remote_log}.pid)")
    run_capture(container, command, timeout=60)
    return remote_log


def wait_health(container, remote_log, case_dir, args, log, deadline):
    started = time.time()
    while time.time() - started < args.startup_timeout:
        if time.time() > deadline:
            return "time-budget"
        if health_ok():
            return "ok"
        time.sleep(10)
    dump_server_tail(container, remote_log, case_dir)
    return "startup-timeout"


def dump_server_tail(container, remote_log, case_dir):
    result = run_capture(container, f"tail -c 16384 '{remote_log}' 2>/dev/null")
    (case_dir / "server-tail.log").write_text(result.stdout, encoding="utf-8")


def pid_alive(container, pid):
    return run_capture(container, f"test -d /proc/{pid} && echo alive || echo dead",
                       timeout=30).stdout.strip() == "alive"


def stop_server(container, remote_log, case_dir, log):
    pid_file = remote_log + ".pid"
    pid = run_capture(container, f"cat '{pid_file}' 2>/dev/null").stdout.strip()
    if not pid:
        log("    no pid file; nothing to stop")
        return True
    run_capture(container, f"kill -TERM -{pid} 2>/dev/null; kill -TERM {pid} 2>/dev/null")
    for _ in range(45):
        if not pid_alive(container, pid):
            log(f"    server pid {pid} stopped")
            return True
        time.sleep(1)
    run_capture(container, f"kill -KILL -{pid} 2>/dev/null; kill -KILL {pid} 2>/dev/null")
    time.sleep(3)
    alive = pid_alive(container, pid)
    log(f"    server pid {pid} {'SURVIVED SIGKILL' if alive else 'stopped after SIGKILL'}")
    return not alive


def require_free_port():
    with socket.socket() as connection:
        connection.settimeout(1)
        if connection.connect_ex(("127.0.0.1", SERVER_PORT)) == 0:
            raise RuntimeError(
                f"Port {SERVER_PORT} is occupied; stop your existing server first"
            )


def shlex_split(text):
    return shlex.split(text) if text else []


# --------------------------------------------------------------------------
# Case runner
# --------------------------------------------------------------------------
def resolve_ref(container, repo, ref, fetch_url, log):
    if fetch_url:
        result = git_in_repo(container, repo, "fetch", "--no-tags", fetch_url, ref)
        if result.returncode != 0:
            return None, f"fetch failed: {result.stderr.strip()[:200]}"
        result = git_in_repo(container, repo, "rev-parse", "FETCH_HEAD")
        sha = result.stdout.strip()
        if result.returncode != 0 or not sha:
            return None, "FETCH_HEAD unresolved"
        result = git_in_repo(container, repo, "checkout", "-q", "-B", "prtest/current", sha)
        if result.returncode != 0:
            return None, f"checkout failed: {result.stderr.strip()[:200]}"
        return sha, None
    result = git_in_repo(container, repo, "rev-parse", "--verify", ref + "^{commit}")
    if result.returncode != 0:
        return None, f"ref not found locally: {ref}"
    sha = result.stdout.strip()
    result = git_in_repo(container, repo, "checkout", "-q", "-B", "prtest/current", sha)
    if result.returncode != 0:
        return None, f"checkout failed: {result.stderr.strip()[:200]}"
    return sha, None


def run_case(case, container, repo, args, case_dir, log, deadline):
    name, ref, env, note = case
    record = {"case": name, "ref": ref, "note": note, "status": "pending",
              "env": {**BASE_ENV, **env}}

    log(f"  resolving {ref}")
    sha, error = resolve_ref(container, repo, ref, args.fetch_url, log)
    if error:
        record.update(status="skipped", reason=error)
        return record
    record["commit"] = sha
    log(f"    at {sha[:10]}")

    remote_log = launch_server(container, repo, env, args, case_dir, log)
    record["server_log"] = remote_log

    state = wait_health(container, remote_log, case_dir, args, log, deadline)
    if state != "ok":
        record.update(status="failed", reason=f"server {state}")
        stop_server(container, remote_log, case_dir, log)
        return record

    try:
        if args.round in ("smoke", "all"):
            info = http("/get_model_info", timeout=30)
            text, meta = generate("Say OK.", max_new_tokens=8)
            smoke = {"model": info, "output": text.strip()[:120]}
            (case_dir / "smoke.json").write_text(json.dumps(smoke, indent=2))
            record["smoke"] = "ok"
            log(f"    smoke ok: {text.strip()[:40]!r}")

        if args.round in ("accuracy", "all") and state == "ok":
            accuracy = run_accuracy_prompts(args, case_dir, log)
            record["accuracy"] = accuracy

        if args.round in ("speed", "all") and state == "ok":
            log(f"    AISBench: len={args.length} req={args.requests} "
                f"chunk={args.chunk} (one invocation, fresh server)")
            code, metrics, _ = run_aisbench(
                args.ais_dir, args.ais_env, args, case_dir / "ais.log")
            record["ais_returncode"] = code
            if metrics:
                record["metrics"] = metrics[-1]
                record["metrics_all"] = metrics
                ttft = metrics[-1].get("TTFT avg")
                p90 = metrics[-1].get("TTFT P90")
                log(f"    TTFT mean {ttft} ms / P90 {p90} ms")
            else:
                record["status"] = "failed"
                record["reason"] = "no TTFT metrics parsed from ais.log"
    except Exception as exc:  # noqa: BLE001 - keep the server-stop path alive
        record.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
    finally:
        if not stop_server(container, remote_log, case_dir, log):
            record["stop_failed"] = True

    if record["status"] == "pending":
        record["status"] = "ok" if record.get("metrics") or record.get("accuracy") or record.get("smoke") == "ok" else "no-data"
    return record


def run_accuracy_prompts(args, case_dir, log):
    results = {}
    for tag, prompt in ACCURACY_PROMPTS:
        text, _ = generate(prompt, max_new_tokens=64)
        results[tag] = text
    long_suffix = "Summarize the text above in one short sentence."
    long_prompt = build_long_prompt(args.accuracy_long_tokens, long_suffix)
    text_first, _ = generate(long_prompt, max_new_tokens=64)
    text_cached, _ = generate(long_prompt, max_new_tokens=64)  # cached-prefix pass
    results["long-cold"] = text_first
    results["long-cached"] = text_cached
    import hashlib

    digest = {tag: hashlib.sha256(t.encode()).hexdigest()[:12]
              for tag, t in results.items()}
    out = {"outputs": results, "sha": digest}
    if args.gsm8k:
        rows = load_gsm8k(args.gsm8k, args.gsm8k_samples, args.gsm8k_seed)
        log(f"    gsm8k: {len(rows)} samples")
        out["gsm8k"] = run_gsm8k(rows, log)
    save(case_dir / "accuracy.json", out)
    return {"sha": digest, "gsm8k": out.get("gsm8k", {}).get("accuracy")}


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def fmt_ttft(record):
    m = record.get("metrics")
    if not m:
        return ""
    return f"{m.get('TTFT avg', 0) / 1000:.2f} / {m.get('TTFT P90', 0) / 1000:.2f}"


def summarize(records, output, args):
    rows = ["# PR-series box results", "",
            f"Generated {now()} | round={args.round} | length={args.length} "
            f"requests={args.requests} chunk={args.chunk} memfrac={args.memfrac}",
            "",
            "## Speed (AISBench prefix-cache TTFT s mean/P90; single sample, no speedup claims)",
            "",
            "| Case | Ref | Status | TTFT mean/P90 (s) | E2E (s) | Reason | Commit |",
            "|---|---|---|---|---|---|---|"]
    for r in records:
        reason = str(r.get("reason", ""))[:70]
        rows.append(f"| {r['case']} | {r['ref']} | {r['status']} | {fmt_ttft(r)} | "
                    f"{r.get('metrics', {}).get('E2E_time', '')} | {reason} | {r.get('commit', '')[:10]} |")

    base = next((r for r in records if r["case"] == "00-baseline"), None)
    if base and base.get("accuracy"):
        rows += ["", "## Accuracy (paired greedy outputs vs baseline)", "",
                 "Exact-output match is reported raw; NPU serving is not bit-deterministic, "
                 "so 'differ' is information, not automatically a failure.", "",
                 "| Case | exact-match | GSM8K acc |", "|---|---|---|"]
        for r in records:
            acc = r.get("accuracy")
            if not acc:
                continue
            if r is base:
                rows.append(f"| {r['case']} | (reference) | "
                            f"{acc.get('gsm8k') if acc.get('gsm8k') is not None else 'n/a'} |")
                continue
            shared = set(acc["sha"]) & set(base["sha"])
            same = sum(acc["sha"][t] == base["sha"][t] for t in shared)
            gsm = acc.get("gsm8k")
            rows.append(f"| {r['case']} | {same}/{len(shared)} | "
                        f"{gsm if gsm is not None else 'n/a'} |")

    rows += ["", "Statuses: ok=measured; failed=see case dir; skipped=ref/fetch problem; "
             "no-data=server ran but nothing measured. Diagnostic timings only - numerical "
             "quality comes from the accuracy section, not from AISBench.", ""]
    (output / "SUMMARY.md").write_text("\n".join(rows), encoding="utf-8")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def plan_cases(args):
    wanted = args.cases.split(",") if args.cases else None
    cases = [c for c in CASES if not wanted or c[0] in wanted]
    if args.skip_baseline_bracket:
        cases = [c for c in cases if "baseline" not in c[0]]
    return cases


def apply_overrides(cases, overrides):
    if not overrides:
        return cases
    table = {c[0]: [c[0], c[1], dict(c[2]), c[3]] for c in cases}
    for item in overrides.split(","):
        case, flag = item.split(":", 1)
        key, value = flag.split("=", 1)
        if case not in table:
            raise SystemExit(f"--flags-overrides: unknown case {case}")
        table[case][2][key] = value
    return [(n, r, e, note) for n, r, e, note in table.values()]


def describe_commands(case, container, repo, args):
    """Dry-run preview of the per-case command sequence."""
    name, ref, env, _ = case
    env_str = " ".join(f"{k}={v}" for k, v in {**BASE_ENV, **env}.items())
    return [
        f"git -c '{args.git_proxy}' -C {repo} fetch --no-tags {args.fetch_url or '<local>'} {ref}",
        f"git -C {repo} checkout -B prtest/current <sha>",
        f"docker exec {container} bash -c \"cd {repo} && setsid env {env_str} "
        f"{args.container_python} -m sglang.launch_server ... > /tmp/prtest-server-{name}.log 2>&1 &\"",
        "poll http://127.0.0.1:30000/health",
        f"accuracy: {len(ACCURACY_PROMPTS)} short + 2 long ({args.accuracy_long_tokens} tok) greedy prompts"
        + (f" + gsm8k {args.gsm8k_samples}" if args.gsm8k else ""),
        f"bash AIS_SHELL {args.ais_env} {args.ais_dir} --input_len {args.length} "
        f"--output_len 1 --dataset_type prefix_cache --prefix_test --seed 4 "
        f"--repeat_rate 0.99 --data_num {args.requests} --concurrency {args.requests}",
        f"kill server group; verify /proc gone",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--round", choices=("smoke", "accuracy", "speed", "all"), default="all")
    parser.add_argument("--cases", default="", help="comma list of case names (default all)")
    parser.add_argument("--flags-overrides", default="",
                        help="case:ENV=VAL[,case:ENV=VAL...] applied after case env")
    parser.add_argument("--container", default="sglang-a3")
    parser.add_argument("--container-python", default="python3")
    parser.add_argument("--repo", default="/home/swx1199799/megamoe_test/sglang")
    parser.add_argument("--model", default="/home/weights/GLM5.2-w4a8-int")
    parser.add_argument("--fetch-url", default=FORK_URL_DEFAULT,
                        help="fork to fetch branches from ('' or --no-fetch to skip)")
    parser.add_argument("--git-proxy",
                        default="url.https://ghfast.top/https://github.com/.insteadOf=https://github.com/",
                        help="git -c insteadOf rewrite for reaching GitHub from the box ('' = direct)")
    parser.add_argument("--no-fetch", action="store_true", help="use local refs only")
    parser.add_argument("--ais-dir", default=AIS_DIR_DEFAULT)
    parser.add_argument("--ais-env", default=None)
    parser.add_argument("--gsm8k", default=None, help="path to gsm8k test jsonl (enables subset)")
    parser.add_argument("--gsm8k-samples", type=int, default=200)
    parser.add_argument("--gsm8k-seed", type=int, default=4)
    parser.add_argument("--length", type=int, default=1000000)
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--chunk", type=int, default=32768)
    parser.add_argument("--context", type=int, default=1048576)
    parser.add_argument("--memfrac", type=float, default=0.88)
    parser.add_argument("--accuracy-long-tokens", type=int, default=32768)
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--benchmark-timeout", type=int, default=14400)
    parser.add_argument("--hours", type=float, default=10.0, help="total time budget")
    parser.add_argument("--case-minutes", type=int, default=60, help="per-case wall cap")
    parser.add_argument("--extra-server-args", default="")
    parser.add_argument("--skip-baseline-bracket", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--stop-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.no_fetch:
        args.fetch_url = ""
    GIT_PROXY[:] = [args.git_proxy] if args.git_proxy else []
    if args.ais_env is None:
        args.ais_env = next((p for p in AIS_ENV_CANDIDATES if Path(p).is_file()),
                            AIS_ENV_CANDIDATES[0])
    cases = apply_overrides(plan_cases(args), args.flags_overrides)

    if args.dry_run:
        print(f"DRY RUN round={args.round} cases={len(cases)} budget={args.hours}h")
        for case in cases:
            print(f"\n== {case[0]} -> {case[1]}  ({case[3]})")
            for line in describe_commands(case, args.container, args.repo, args):
                print(f"   {line}")
        return 0

    if args.stop_only:
        run_capture(args.container, "pkill -TERM -f sglang.launch_server || true")
        for _ in range(45):
            if not health_ok():
                break
            time.sleep(1)
        run_capture(args.container, "pkill -KILL -f sglang.launch_server || true")
        print("stopped; port free" if not health_ok() else "PORT STILL BUSY")
        return 0

    require_free_port()
    lock = open("/tmp/prtest.lock", "w")  # noqa: SIM115 - held for the driver lifetime
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit("another prtest run is active (/tmp/prtest.lock)")

    stamp = now()
    output = Path(args.output or f"prtest-results/{stamp}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    save(output / "settings.json", {k: v for k, v in vars(args).items()})
    deadline = time.time() + args.hours * 3600

    def log(message):
        line = f"[{datetime.datetime.now():%H:%M:%S}] {message}"
        print(line, flush=True)
        with open(output / "runner.log", "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log(f"round={args.round} cases={len(cases)} output={output} budget={args.hours}h")
    records = []
    for case in cases:
        name, ref, env, note = case
        case_dir = output / name
        case_dir.mkdir()
        remaining = deadline - time.time()
        if remaining < args.case_minutes * 60:
            log(f"== {name}: SKIPPED (time budget exhausted)")
            records.append({"case": name, "ref": ref, "status": "skipped",
                            "reason": "time budget"})
            break
        log(f"== {name} -> {ref} ({note})")

        def case_log(message):
            log("  " + message)

        record = run_case(case, args.container, args.repo, args, case_dir, case_log, deadline)
        records.append(record)
        save(output / "results.json", records)
        summarize(records, output, args)
        if record.get("stop_failed"):
            log("STOP: a server survived cleanup; halting to avoid stacking experiments.")
            break

    summarize(records, output, args)
    log(f"done: {output / 'SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
