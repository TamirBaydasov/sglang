#!/usr/bin/env python3
"""Speed + accuracy test harness for the six Ascend NPU PR series (PRs #2-#7).

Follows the established box workflow (run_a3_nightly.py / run_bde_overnight.py
lineage): separate fresh servers for accuracy and speed, one measured AISBench
invocation per speed server, scored GSM8K and paired-output diagnostics, host-side orchestration
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
  all       accuracy and speed on separate fresh servers (default)

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
import ast
import datetime
import fcntl
import hashlib
import json
import math
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

# --------------------------------------------------------------------------
# Box constants (defaults follow the GLM A3 nightly lineage; override via CLI)
# --------------------------------------------------------------------------
AIS_ENV_CANDIDATES = ("/home/tbaydasov/set_env.sh", "/home/tbaydasovset_env.sh")
AIS_DIR_DEFAULT = "/home/tbaydasov/aisbench/aisbench_auto_tools_prefix"
FORK_URL_DEFAULT = "https://github.com/TamirBaydasov/sglang.git"

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
# Baseline serving recipe = the archived 1M pairing of p4_exit1_serve_dcp.sh
# and profile_aisbench.sh (PROFILE=0):
#   CTX=1024000 CHUNK=16384 MAXREQ=16 CAPTURE=1 MEMFRAC=0.70 \
#     EXTRA_ARGS="--watchdog-timeout 7200" bash p4_exit1_serve_dcp.sh 16
# Decode graph capture ON at max-bs 16 (= MAXREQ, a multiple of TP=16; the
# token-sharded PR refuses only EXTEND capture). Prefill capture stays
# disabled (NPU forces it off anyway). memfrac is 0.70, NOT 0.76/0.88: the
# first live run OOM'd at pool init at 0.88 (12.83 GiB index_k_buffer, 47.40
# GiB already allocated of a 61.28 GiB die), and the archived 1M recipe pairs
# this exact workload with 0.70. moe-a2a stays 'none' deliberately: the ts PR
# refuses a2a backends and a paired A/B needs one config across all cases
# (TP-MoE is what the series optimizes; the checkpoint loads fine on 'none').
BASE_SERVER_ARGS = [
    "--device",
    "npu",
    "--attention-backend",
    "ascend",
    "--tp-size",
    "16",
    "--dcp-size",
    "16",
    "--trust-remote-code",
    "--cuda-graph-backend-prefill",
    "disabled",
    "--context-length",
    "1024000",
    "--chunked-prefill-size",
    "16384",
    "--max-running-requests",
    "16",
    "--cuda-graph-max-bs-decode",
    "16",
    "--watchdog-timeout",
    "7200",
    "--mem-fraction-static",
    "0.70",
    "--port",
    str(SERVER_PORT),
    # Packed DCP reads require one request per prefill batch.
    "--prefill-max-requests",
    "1",
    "--enable-metrics",
]

# --------------------------------------------------------------------------
# Case table: (name, ref, env, note).  Baseline bracket = first + last.
# --------------------------------------------------------------------------
CASES = [
    ("00-baseline", "dcp_ascend_testing", {}, "control, bracket start"),
    (
        "01-pr-e-chat-encode",
        "npu-chat-encode-pool",
        {
            "SGLANG_ENABLE_CHAT_ENCODE_POOL": "True",
            "SGLANG_CHAT_ENCODE_MEMO": "True",
            "SGLANG_CHAT_ENCODE_RAW": "True",
        },
        "PR #2 (E): all three features - pool + memo + raw Rust encode",
    ),
    (
        "02-pr-b-indexer-int8",
        "npu-dsa-indexer-int8",
        {
            "SGLANG_NPU_INDEXER_INT8": "True",
        },
        "PR #3 (B): int8 index-K + per-token fp16 scale",
    ),
    (
        "03-pr-prefetch-packed",
        "npu-dcp-packed-read",
        {
            "SGLANG_NPU_ENABLE_DCP_PACKED_READ": "True",
            "SGLANG_NPU_ENABLE_DCP_PACKED_GATHER_PREFETCH": "True",
            "SGLANG_NPU_ENABLE_PG_PRIORITY": "True",
        },
        "PR #4 (C1/D2p/D2v2): packed read + prime/prefetch + chunked gather "
        "(CHUNKS=8 and stream priority are the branch defaults)",
    ),
    (
        "04-pr-f-sfa-kv-8bit",
        "npu-sfa-kv-8bit",
        {
            "SGLANG_NPU_SFA_KV_8BIT": "True",
            "SGLANG_NPU_SFA_KV_QUANT_READ": "True",
        },
        "PR #5 (F): 656 B/row packed KV; F2 in-kernel dequant on",
    ),
    (
        "05-pr-ts-token-sharded",
        "npu-dsa-cp-token-sharded",
        {
            "SGLANG_NPU_DSA_CP_TOKEN_SHARDED": "True",
            "SGLANG_NPU_DSA_CP_Q_FULL_WIDTH": "True",
            "SGLANG_NPU_DSA_CP_LOCAL_TOPK": "True",
            "SGLANG_NPU_DCP_DECODE_Q_PREISSUE": "True",
            "SGLANG_NPU_STREAM_O_PROJ": "True",
            "SGLANG_NPU_STREAM_Q_B": "True",
            "SGLANG_NPU_STREAM_W_KC": "True",
            "SGLANG_NPU_STREAM_W_VC": "True",
            "SGLANG_NPU_ENABLE_DCP_SFA_GATHER_OVERLAP": "True",
        },
        "PR #6 (D1v3) every feature: rebuild + all four streamed weights + "
        "SFA gather overlap (permuting-path flag; this branch has no packed read)",
    ),
    (
        "06-pr-moe-int8",
        "npu-dsa-cp-moe-int8",
        {
            "SGLANG_NPU_DSA_CP_TOKEN_SHARDED": "True",
        },
        "PR #7 (M): int8 MoE input (unconditional under the ts flag)",
    ),
    (
        "07-combined",
        "npu-bde-combined",
        {
            "SGLANG_ENABLE_CHAT_ENCODE_POOL": "True",
            "SGLANG_CHAT_ENCODE_MEMO": "True",
            "SGLANG_CHAT_ENCODE_RAW": "True",
            "SGLANG_NPU_INDEXER_INT8": "True",
            "SGLANG_NPU_ENABLE_DCP_PACKED_READ": "True",
            "SGLANG_NPU_ENABLE_DCP_PACKED_GATHER_PREFETCH": "True",
            "SGLANG_NPU_ENABLE_PG_PRIORITY": "True",
            "SGLANG_NPU_SFA_KV_8BIT": "True",
            "SGLANG_NPU_SFA_KV_QUANT_READ": "True",
            "SGLANG_NPU_DSA_CP_TOKEN_SHARDED": "True",
            "SGLANG_NPU_DSA_CP_Q_FULL_WIDTH": "True",
            "SGLANG_NPU_DSA_CP_LOCAL_TOPK": "True",
            "SGLANG_NPU_DCP_DECODE_Q_PREISSUE": "True",
            "SGLANG_NPU_STREAM_O_PROJ": "True",
            "SGLANG_NPU_STREAM_Q_B": "True",
            "SGLANG_NPU_STREAM_W_KC": "True",
            "SGLANG_NPU_STREAM_W_VC": "True",
        },
        "full deployment configuration: every branch feature on (SFA overlap "
        "stays off - superseded by the packed-read prime in this tree)",
    ),
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
    (
        "short-reason",
        "A train leaves at 14:05 and travels for 2h 50m. At what time does it arrive? Answer with HH:MM.",
    ),
    ("short-code", "In Python, write a one-line function that reverses a string."),
    ("short-translate", "Translate to English: 今天下午三点开会，请准时参加。"),
    ("short-fact", "Name the largest planet in the solar system with one word."),
    (
        "short-logic",
        "If all bloops are razzies and all razzies are lazzies, are all bloops definitely lazzies? Answer yes or no.",
    ),
    (
        "short-order",
        "Sort these words alphabetically: mango, apple, cherry, banana. Answer with the sorted list only.",
    ),
    (
        "short-count",
        "How many days are in February during a leap year? Answer with the number only.",
    ),
]

# Deterministic template bank for the remaining accuracy requests.
ACCURACY_TEMPLATES = [
    "Compute {a} * {b} - {c}. Answer with just the number.",
    "A shop buys {a} items at ${b} each and sells them at ${c} each. What is the total profit in dollars? Answer with just the number.",
    "A train travels {a} km in {b} hours. What is its average speed in km/h? Answer with just the number.",
    "Sum the integers from 1 to {a}. Answer with just the number.",
    "Is {a} a prime number? Answer yes or no.",
    "What is the capital of {c_word}? Answer with the city name only.",
    "Write a Python one-liner that squares each element of a list called xs.",
    "Translate to English: 会议将于明天上午十点在三层会议室开始。",
]
TEMPLATE_COUNTRIES = [
    "France",
    "Japan",
    "Egypt",
    "Brazil",
    "Canada",
    "Australia",
    "India",
    "Kenya",
    "Chile",
    "Norway",
    "Peru",
    "Spain",
]


def build_accuracy_prompts(total, long_tokens):
    """Deterministic greedy set of exactly `total` (tag, prompt) pairs.

    8 fixed + template-generated short/medium prompts + a long-prompt pair
    (same body, different tails, so the second request rides the cached
    prefix). Outputs are SHA-compared per tag against the baseline case.
    """
    # ACCURACY_PROMPTS entries are (tag, prompt) pairs; take the prompt, not
    # the pair. (The tuple-as-prompt bug sent JSON lists to /generate, which
    # answered in batch mode with a list -- run 20261112-212404, exactly the
    # 8 fixed prompts erroring 'list' object has no attribute get.)
    prompts = [(f"fixed-{i}", prompt) for i, (_, prompt) in enumerate(ACCURACY_PROMPTS)]
    i = 0
    while len(prompts) < total - 2:
        tpl = ACCURACY_TEMPLATES[i % len(ACCURACY_TEMPLATES)]
        subs = {
            "a": 12 + (i * 7) % 89,
            "b": 3 + (i * 5) % 17,
            "c": 2 + (i * 11) % 29,
            "c_word": TEMPLATE_COUNTRIES[i % len(TEMPLATE_COUNTRIES)],
        }
        try:
            text = tpl.format(**subs)
        except KeyError:
            text = tpl
        prompts.append((f"gen-{i:03d}", text))
        i += 1
    prompts.append(
        (
            "long-cold",
            build_long_prompt(
                long_tokens, "Summarize the text above in one short sentence."
            ),
        )
    )
    prompts.append(
        (
            "long-cached",
            build_long_prompt(
                long_tokens,
                "How many shifts per day are described above? Answer with just the number.",
            ),
        )
    )
    return prompts[:total]


GSM8K_EXEMPLARS = [
    (
        "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "Weng earns 12/60 = $0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $10. #### 10",
    ),
    (
        "Betty is saving money to buy a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
        "Betty has 100/2 = $50. Her grandparents gave 15*2 = $30. Betty now has 50+15+30 = $95. She needs 100-95 = $5 more. #### 5",
    ),
    (
        "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
        "He writes 3*2*2 = 12 pages per week. Per year 12*52 = 624. #### 624",
    ),
    (
        "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
        "It takes 2/2 = 1 bolt of white fiber. So the total is 2+1 = 3 bolts. #### 3",
    ),
    (
        "Ken had fifty apples. He gave half to James and sold the rest for $0.60 each. How much did he earn?",
        "He gave away 50/2 = 25. He sold 25 apples at 0.60 = $15. #### 15",
    ),
]


# All host and container work shares a monotonic deadline.
class BudgetExpired(TimeoutError):
    pass


def remaining(deadline, cap=None):
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise BudgetExpired("time budget exhausted")
    return min(seconds, cap) if cap is not None else seconds


def save(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")


def execute(command, deadline, timeout=120, env=None, input_text=None, log_path=None):
    """Kill the entire host command group on timeout or interruption."""
    seconds = remaining(deadline, timeout)
    output = open(log_path, "w") if log_path else subprocess.PIPE
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            start_new_session=True,
        )
    except BaseException:
        if log_path:
            output.close()
        raise
    try:
        stdout, _ = process.communicate(input_text, timeout=seconds)
        return subprocess.CompletedProcess(
            command, process.returncode, stdout or "", ""
        )
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        # Children can outlive a reaped parent and ignore SIGTERM.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise
    finally:
        if log_path:
            output.close()


def run_capture(args, command, deadline, timeout=120, input_text=None):
    result = execute(
        ["docker", "exec", "-i", args.container, "bash", "-c", command],
        deadline,
        timeout,
        input_text=input_text,
    )
    if result.returncode:
        raise RuntimeError(
            f"docker command failed ({result.returncode}): {result.stdout[-1500:]}"
        )
    return result.stdout


def git(args, *git_args, deadline, repo=None, input_text=None):
    command = ["git"]
    if args.git_proxy:
        command += ["-c", args.git_proxy]
    command += ["-C", repo or args.repo, *git_args]
    return run_capture(
        args, shlex.join(command), deadline, timeout=240, input_text=input_text
    ).strip()


def require_free_port(port=SERVER_PORT):
    with socket.socket() as connection:
        connection.settimeout(1)
        if connection.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"port {port} is occupied")


_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def http_raw(path, payload, deadline, timeout=120):
    request = urllib.request.Request(
        f"http://127.0.0.1:{SERVER_PORT}{path}",
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with _OPENER.open(request, timeout=remaining(deadline, timeout)) as response:
        chunks = []
        while response.fp is not None:
            seconds = remaining(deadline, timeout)
            response.fp.raw._sock.settimeout(seconds)
            chunk = response.read1(65536)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks).decode("utf-8")


def http(path, payload, deadline, timeout=120):
    body = http_raw(path, payload, deadline, timeout)
    return json.loads(body) if body else None


def generate(text, deadline, max_new_tokens=64):
    data = http(
        "/generate",
        {
            "text": text,
            "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens},
        },
        deadline,
        7200,
    )
    if not isinstance(data, dict) or not isinstance(data.get("text"), str):
        raise ValueError("invalid /generate response")
    return data["text"], data.get("meta_info", {})


def chat(prompt, model, deadline, max_new_tokens=64, stream=False):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_new_tokens,
        "stream": stream,
    }
    if stream:
        body = http_raw("/v1/chat/completions", payload, deadline, 7200)
        chunks = [
            json.loads(line[6:])
            for line in body.splitlines()
            if line.startswith("data: ") and line[6:] != "[DONE]"
        ]
        if not chunks or any(
            "error" in chunk or "choices" not in chunk for chunk in chunks
        ):
            raise ValueError(f"invalid streaming chat response: {body[:300]}")
        if "[DONE]" not in body:
            raise ValueError("chat stream ended without [DONE]")
        return "".join(
            choice.get("delta", {}).get("content") or ""
            for chunk in chunks
            for choice in chunk["choices"]
        ), {"chunks": chunks}
    data = http("/v1/chat/completions", payload, deadline, 7200)
    choice = data["choices"][0]
    message = choice["message"]
    text = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    if not isinstance(text, str) or not isinstance(reasoning, str):
        raise ValueError("invalid chat content")
    return text, {
        "finish_reason": choice.get("finish_reason"),
        "usage": data.get("usage"),
        "reasoning_content": reasoning,
        "response": data,
    }


def build_long_prompt(target_tokens, suffix):
    chars = target_tokens * 4
    return (LONG_PARAGRAPH * (chars // len(LONG_PARAGRAPH) + 1))[:chars] + "\n" + suffix


def last_number(text):
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text.replace(",", "").replace("$", ""))
    return float(numbers[-1]) if numbers else None


def gold_number(answer):
    if not isinstance(answer, str) or "####" not in answer:
        raise ValueError("GSM8K answer has no #### final answer")
    value = float(answer.split("####")[-1].strip().replace(",", "").replace("$", ""))
    if not math.isfinite(value):
        raise ValueError("non-finite GSM8K answer")
    return value


def load_gsm8k(path, samples, seed):
    import random

    rows = []
    with open(path, encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"invalid GSM8K record at line {index + 1}")
            if not isinstance(row.get("question"), str) or not row["question"].strip():
                raise ValueError(f"invalid GSM8K question at line {index + 1}")
            gold_number(row["answer"])
            rows.append(dict(row, sample_id=index))
    if len(rows) < samples:
        raise ValueError(f"GSM8K has {len(rows)} samples; requested {samples}")
    random.Random(seed).shuffle(rows)
    return rows[:samples]


def run_gsm8k(rows, args, case_dir, deadline, log):
    shots = "\n\n".join(f"Question: {q}\nAnswer: {a}" for q, a in GSM8K_EXEMPLARS)
    details = []
    out = {
        "dataset_sha256": args.dataset_sha256,
        "seed": args.gsm8k_seed,
        "requested": len(rows),
        "details": details,
        "generation": {"temperature": 0, "max_tokens": args.gsm8k_max_tokens},
        "status": "running",
    }
    for row in rows:
        remaining(deadline)
        prompt = shots + "\n\nQuestion: " + row["question"] + "\nAnswer:"
        detail = {"id": row["sample_id"], "prompt": prompt, "gold": row["answer"]}
        try:
            text, meta = chat(prompt, args.model, deadline, args.gsm8k_max_tokens)
            detail.update(
                output=text,
                metadata=meta,
                got=last_number(text),
                want=gold_number(row["answer"]),
                truncated=meta.get("finish_reason") == "length",
            )
            detail["ok"] = (
                not detail["truncated"]
                and detail["got"] is not None
                and abs(detail["got"] - detail["want"]) < 1e-6
            )
        except (BudgetExpired, subprocess.TimeoutExpired):
            raise
        except Exception as exc:
            detail.update(error=f"{type(exc).__name__}: {exc}", ok=False)
        details.append(detail)
        out.update(
            completed=len(details),
            correct=sum(d["ok"] for d in details),
            errors=sum("error" in d for d in details),
            truncated=sum(d.get("truncated", False) for d in details),
        )
        out["accuracy"] = out["correct"] / len(rows)
        save(case_dir / "gsm8k.json", out)
        if len(details) % 25 == 0:
            log(
                f"GSM8K {len(details)}/{len(rows)} correct={out['correct']} errors={out['errors']}"
            )
    out["status"] = "failed" if out["errors"] or out["truncated"] else "ok"
    save(case_dir / "gsm8k.json", out)
    return out


def run_accuracy_prompts(args, case_dir, deadline, log):
    results = {}
    digest = {}
    for tag, prompt in build_accuracy_prompts(
        args.accuracy_requests, args.accuracy_long_tokens
    ):
        remaining(deadline)
        try:
            text, meta = chat(prompt, args.model, deadline)
            results[tag] = {"prompt": prompt, "output": text, "metadata": meta}
            if text and meta.get("finish_reason") != "length":
                digest[tag] = hashlib.sha256(text.encode()).hexdigest()
        except (BudgetExpired, subprocess.TimeoutExpired):
            raise
        except Exception as exc:
            results[tag] = {"prompt": prompt, "error": str(exc)}
        save(
            case_dir / "accuracy.json",
            {
                "outputs": results,
                "sha": digest,
                "errors": sum("error" in r for r in results.values()),
            },
        )
    log(f"paired-output diagnostics: {len(digest)}/{len(results)} complete outputs")
    return {"sha": digest, "errors": sum("error" in r for r in results.values())}


# AISBench's wrapper logs Python repr dictionaries and may exit zero on failure.
def parse_ttft_lines(text):
    metrics = []
    for line in text.splitlines():
        if "TTFT avg" not in line or "{" not in line:
            continue
        candidate = line[line.index("{") :].strip()
        try:
            item = json.loads(candidate)
        except (ValueError, TypeError):
            try:
                item = ast.literal_eval(candidate)
            except (ValueError, SyntaxError):
                continue
        if isinstance(item, dict) and "TTFT avg" in item:
            metrics.append(item)
    return metrics


def validate_ais(text, returncode, requests, length=None):
    metrics = parse_ttft_lines(text)
    errors = []
    if returncode != 0:
        errors.append(f"AISBench rc={returncode}")
    if re.search(
        r"All requests failed|Request failed:|Traceback \(most recent call last\)|AisBenchClientException",
        text,
    ):
        errors.append("AISBench request/tool failure in log")
    failed = re.findall(r"Failed Requests\s*[│|]\s*total\s*[│|]\s*([\d.]+)", text)
    if not failed:
        errors.append("missing AISBench failed-request counts")
    elif len(failed) != 2 or any(float(value) != 0 for value in failed):
        errors.append("AISBench failed requests")
    if len(metrics) != 2:
        errors.append(
            f"expected prefix preparation + measurement, found {len(metrics)} records"
        )
    for index, item in enumerate(metrics):
        expected = 1 if index == 0 else requests
        try:
            if float(item["total_req"]) != expected:
                errors.append(f"phase {index} request count differs from {expected}")
            for key in ("TTFT avg", "TTFT P90", "E2E_time", "input_len", "output_len"):
                value = float(item[key])
                if not math.isfinite(value) or value <= 0 or value == 99999:
                    errors.append(f"phase {index} invalid {key}: {value}")
            if length is not None:
                target_length = length * (0.99 if index == 0 else 1)
                if abs(float(item["input_len"]) - target_length) > max(
                    1024, length * 0.02
                ):
                    errors.append(f"phase {index} input length differs from workload")
            if float(item["output_len"]) != 1:
                errors.append("AISBench output length differs from 1")
        except (KeyError, TypeError, ValueError):
            errors.append(f"phase {index} missing/non-numeric metrics")
    return {
        "status": "failed" if errors else "ok",
        "reason": "; ".join(dict.fromkeys(errors)),
        "returncode": returncode,
        "preparation": metrics[0] if metrics else None,
        "metrics": metrics[1] if len(metrics) == 2 and not errors else None,
        "metrics_all": metrics,
        "units": {"TTFT avg": "ms", "TTFT P90": "ms", "E2E_time": "s"},
    }


AIS_SETUP = r"""
set -e
# set_env.sh may itself activate Conda; initialize before sourcing it.
conda_command=${CONDA_EXE:-}
if [ ! -x "$conda_command" ]; then
    conda_command=$(type -P conda || true)
fi
if [ -n "$conda_command" ]; then
    conda_hook=$("$conda_command" shell.bash hook)
    eval "$conda_hook"
elif [ -f /home/miniconda3/etc/profile.d/conda.sh ]; then
    source /home/miniconda3/etc/profile.d/conda.sh
elif ! declare -F conda >/dev/null; then
    echo "prtest FATAL: cannot initialize Conda (CONDA_EXE, PATH, conda.sh unavailable)" >&2
    exit 78
fi
source "$1"
conda activate "$AIS_CONDA_ENV"
command -v python3 >/dev/null
command -v ais_bench >/dev/null
ais_bench --help >/dev/null
cd "$2"
unset HTTP_PROXY http_proxy HTTPS_PROXY https_proxy ALL_PROXY all_proxy
export NO_PROXY=localhost,127.0.0.1,::1
"""

AIS_PREFLIGHT = r"""
python3 - "$3" "$4" <<'PYCHECK'
import importlib.metadata, json, os, runpy, shutil, sys
from pathlib import Path
assert Path('aisbench_test.py').is_file(), 'aisbench_test.py missing'
c = runpy.run_path('config.py')
assert int(c['HOST_PORT']) == int(sys.argv[2]), 'AISBench port mismatch'
assert c['HOST_IP'] in ('127.0.0.1', 'localhost'), 'AISBench must target this host'
assert os.path.realpath(c['MODEL_PATH']) == os.path.realpath(sys.argv[1]), 'AISBench model mismatch'
assert Path(c['MODEL_PATH']).is_dir(), 'AISBench tokenizer model directory missing on host'
assert Path(c['WORK_PATH']).is_dir(), 'AISBench WORK_PATH missing'
print(json.dumps({'python': sys.executable, 'ais_bench': shutil.which('ais_bench'),
                  'conda_prefix': os.environ.get('CONDA_PREFIX'),
                  'config': {k: str(c.get(k)) for k in ('HOST_PORT','HOST_IP','MODEL_PATH','WORK_PATH')},
                  'versions': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()
                               if (d.metadata.get('Name') or '').lower().replace('_','-') in ('ais-bench-benchmark','transformers','tokenizers')}}))
PYCHECK
"""


def ais_command(args, preflight=False):
    script = AIS_SETUP + (
        AIS_PREFLIGHT if preflight else 'exec python3 aisbench_test.py "${@:3}"\n'
    )
    command = ["bash", "-c", script, "prtest-ais", args.ais_env, args.ais_dir]
    if preflight:
        return command + [args.model, str(SERVER_PORT)]
    return command + [
        "--input_len",
        str(args.length),
        "--output_len",
        "1",
        "--request_rate",
        "0",
        "--dataset_type",
        "prefix_cache",
        "--prefix_test",
        "--seed",
        "4",
        "--repeat_rate",
        "0.99",
        "--data_num",
        str(args.requests),
        "--concurrency",
        str(args.requests),
    ]


def ais_environment(args):
    return dict(
        os.environ,
        AIS_CONDA_ENV=args.ais_conda_env,
        NO_PROXY="localhost,127.0.0.1,::1",
        no_proxy="localhost,127.0.0.1,::1",
    )


def ais_preflight(args, deadline):
    result = execute(ais_command(args, True), deadline, 180, env=ais_environment(args))
    if result.returncode:
        raise RuntimeError(f"AIS preflight failed: {result.stdout[-2000:]}")
    for line in reversed(result.stdout.splitlines()):
        try:
            return json.loads(line)
        except ValueError:
            continue
    raise ValueError("AIS preflight produced no environment record")


def run_aisbench(args, case_dir, deadline):
    # A failed flush invalidates the cold-prefix contract.
    http_raw("/flush_cache?timeout=60", {}, deadline, 120)
    command = ais_command(args)
    save(case_dir / "ais-command.json", command)
    path = case_dir / "ais.log"
    result = execute(
        command,
        deadline,
        args.benchmark_timeout,
        env=ais_environment(args),
        log_path=path,
    )
    return validate_ais(
        path.read_text(errors="replace"), result.returncode, args.requests, args.length
    )


REMOTE = Path(__file__).with_name("prtest_remote.py").read_text()
CANN_ENV_FALLBACKS = (
    "/usr/local/Ascend/cann-9.2.0-beta.2/set_env.sh",
    "/usr/local/Ascend/cann-9.2/set_env.sh",
    "/usr/local/Ascend/cann-9.2.0/set_env.sh",
)

PROBE = r"""
import hashlib, importlib.metadata, json, os, sys
from pathlib import Path
import torch, torch_npu
model = Path(sys.argv[1])
config = json.loads((model/'config.json').read_text())
files = {}
for name in ('config.json','tokenizer_config.json','tokenizer.json','quant_model_description.json','model.safetensors.index.json'):
 p = model/name
 if p.is_file(): files[name] = hashlib.sha256(p.read_bytes()).hexdigest()
versions = {}
for name in ('torch','torch-npu','transformers','tokenizers','sgl-kernel-npu'):
 try: versions[name] = importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError: versions[name] = None
ops = {name: hasattr(torch.ops.npu, name) for name in ('npu_sparse_flash_attention_lse',)}
backend = Path('python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py').read_text()
requires_lse = 'torch.ops.npu.npu_sparse_flash_attention_lse(' in backend
print('PRTEST_PROBE='+json.dumps({'versions': versions, 'ops':ops, 'model_config':config,
 'requires_sparse_fa_lse': requires_lse,
 'model_files_sha256':files, 'python':sys.executable, 'torch_path':torch.__file__,
 'environment':{k:v for k,v in os.environ.items() if k.startswith(('SGLANG_', 'ASCEND_', 'HCCL_', 'DEEPEP_')) or k in ('PYTHONPATH','LD_LIBRARY_PATH','PATH')}}))
"""


def server_argv(args, memfrac, graph):
    values = list(BASE_SERVER_ARGS)
    for key, value in [
        ("--mem-fraction-static", memfrac),
        ("--chunked-prefill-size", args.chunk),
        ("--context-length", args.context),
    ]:
        values[values.index(key) + 1] = str(value)
    if not graph:
        index = values.index("--cuda-graph-max-bs-decode")
        values[index : index + 2] = ["--cuda-graph-backend-decode", "disabled"]
    return [
        args.container_python,
        "-m",
        "sglang.launch_server",
        "--load-format",
        "auto",
        "--model-path",
        args.model,
        "--random-seed",
        str(args.server_seed),
        *values,
        *shlex.split(args.extra_server_args),
    ]


def server_shell(args, worktree, env, argv):
    # Clear inherited feature toggles so the baseline is a real control.
    keys = sorted({key for _, _, flags, _ in CASES for key in flags} | set(BASE_ENV))
    setup = "set -e; "
    if args.cann_env_resolved:
        setup += "source " + shlex.quote(args.cann_env_resolved) + "; "
    setup += "cd " + shlex.quote(worktree) + "; "
    setup += "unset " + " ".join(keys) + "; "
    assignments = dict(BASE_ENV, **env)
    setup += (
        "export "
        + shlex.join([f"{key}={value}" for key, value in assignments.items()])
        + "; "
    )
    setup += (
        "export PYTHONPATH="
        + shlex.quote(worktree + "/python")
        + ':"${PYTHONPATH:-}"; '
    )
    return setup + "exec " + shlex.join(argv)


def resolve_cann_env(args, deadline, log):
    if not args.container_cann_env:
        return None
    candidates = list(dict.fromkeys([args.container_cann_env, *CANN_ENV_FALLBACKS]))
    for candidate in candidates:
        exists = run_capture(
            args,
            "if test -f " + shlex.quote(candidate) + "; then echo yes; fi",
            deadline,
        ).strip()
        if exists == "yes":
            log(
                "CANN environment: "
                + candidate
                + " (operator availability checked separately)"
            )
            return candidate
    raise RuntimeError(
        "no requested CANN environment file exists: " + ", ".join(candidates)
    )


def probe_runtime(args, worktree, env, deadline):
    command = server_shell(
        args, worktree, env, [args.container_python, "-c", PROBE, args.model]
    )
    output = run_capture(args, command, deadline, timeout=180)
    for line in output.splitlines():
        if line.startswith("PRTEST_PROBE="):
            info = json.loads(line.split("=", 1)[1])
            return info
    raise RuntimeError("runtime probe produced no record: " + output[-1000:])


def remote_command(args, action, state_path, spec=None):
    command = [args.container_python, "-c", REMOTE, action, state_path]
    if spec is not None:
        command.append(json.dumps(spec))
    return command


def remote_action(args, action, state_path, deadline):
    output = run_capture(
        args, shlex.join(remote_command(args, action, state_path)), deadline, timeout=40
    )
    return json.loads(output)


def launch_server(args, command, case_dir, registry, deadline):
    token = uuid.uuid4().hex
    state_path = args.remote_root + "/processes/" + token + ".json"
    remote_log = args.remote_root + "/logs/" + token + ".log"
    run_capture(args, shlex.join(["mkdir", "-p", args.remote_root + "/logs"]), deadline)
    spec = {"token": token, "command": command, "log": remote_log}
    registry["processes"].append(
        {"state": state_path, "log": remote_log, "token": token}
    )
    save(args.registry_path, registry)
    result = execute(
        [
            "docker",
            "exec",
            "-d",
            args.container,
            *remote_command(args, "launch", state_path, spec),
        ],
        deadline,
        30,
    )
    if result.returncode:
        raise RuntimeError("detached launch failed: " + result.stdout)
    # docker exec -d returns before the ownership helper has written its file.
    for _ in range(50):
        output = run_capture(
            args,
            "if test -s "
            + shlex.quote(state_path)
            + "; then cat "
            + shlex.quote(state_path)
            + "; fi",
            deadline,
        )
        if output.strip():
            identity = json.loads(output)
            save(case_dir / "process.json", identity)
            return state_path, remote_log
        time.sleep(min(0.1, remaining(deadline)))
    raise RuntimeError("detached launch did not publish its ownership record")


def stop_server(args, state_path):
    # Cleanup has a small independent budget even if the case deadline expired.
    result = remote_action(args, "stop", state_path, time.monotonic() + 45)
    if not result.get("stopped"):
        raise RuntimeError("owned server cleanup did not confirm termination")


def capture_server_log(args, remote_log, case_dir):
    output = run_capture(
        args, shlex.join(["cat", remote_log]), time.monotonic() + 20, timeout=15
    )
    (case_dir / "server.log").write_text(output)
    return output


def wait_health(args, state_path, deadline, log):
    startup_deadline = min(deadline, time.monotonic() + args.startup_timeout)
    last_note = time.monotonic() - 60
    while True:
        remaining(startup_deadline)
        status = remote_action(args, "check", state_path, startup_deadline)
        if not status["members"]:
            raise RuntimeError("owned server exited before readiness")
        try:
            http_raw("/health", None, startup_deadline, 5)
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            remaining(startup_deadline)
        if time.monotonic() - last_note >= 60:
            log("waiting for owned server /health")
            last_note = time.monotonic()
        time.sleep(min(2, remaining(startup_deadline)))


def pin_refs(cases, args, deadline, checkpoint):
    pinned = {}
    for _, ref, _, _ in cases:
        if ref in pinned:
            continue
        try:
            if args.fetch_url:
                git(args, "fetch", "--no-tags", args.fetch_url, ref, deadline=deadline)
                sha = git(
                    args,
                    "rev-parse",
                    "--verify",
                    "FETCH_HEAD^{commit}",
                    deadline=deadline,
                )
            else:
                sha = git(
                    args, "rev-parse", "--verify", ref + "^{commit}", deadline=deadline
                )
            if not re.fullmatch(r"[0-9a-f]{40}", sha):
                raise ValueError("invalid resolved commit")
            pinned[ref] = {"commit": sha}
        except Exception as exc:
            pinned[ref] = {"error": f"{type(exc).__name__}: {exc}"}
        checkpoint(pinned)
    return pinned


def create_worktree(args, record, registry, deadline):
    path = args.remote_root + "/worktrees/" + record["case"]
    registry["worktrees"].append(path)
    save(args.registry_path, registry)
    git(args, "worktree", "add", "--detach", path, record["commit"], deadline=deadline)
    record["source_diff_sha256"] = hashlib.sha256(
        git(args, "diff", "--binary", deadline=deadline, repo=path).encode()
    ).hexdigest()
    return path


def comparison_config(args, runtime, memfrac, graph):
    return {
        "model": args.model,
        "model_files_sha256": runtime["model_files_sha256"],
        "versions": runtime["versions"],
        "length": args.length,
        "requests": args.requests,
        "chunk": args.chunk,
        "context": args.context,
        "memfrac": memfrac,
        "decode_graph": graph,
        "server_seed": args.server_seed,
        "extra_server_args": shlex.split(args.extra_server_args),
        "cann_env": args.cann_env_resolved,
        "runtime_environment": {
            key: value
            for key, value in runtime["environment"].items()
            if key in ("LD_LIBRARY_PATH", "ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME")
        },
        "ais_environment": getattr(args, "ais_preflight_record", None),
    }


def resolved_comparison(config, smoke_record):
    server = smoke_record.get("server") or {}
    server = server.get("server_args", server)
    essential = (
        "model_path",
        "load_format",
        "dtype",
        "tp_size",
        "dcp_size",
        "attention_backend",
    )
    if not all(key in server for key in essential):
        return None
    keys = (
        *essential,
        "tokenizer_path",
        "quantization",
        "kv_cache_dtype",
        "moe_runner_backend",
        "moe_a2a_backend",
        "mem_fraction_static",
        "context_length",
        "chunked_prefill_size",
        "max_running_requests",
        "prefill_max_requests",
        "cuda_graph_backend_prefill",
        "cuda_graph_backend_decode",
        "cuda_graph_max_bs_decode",
        "json_model_override_args",
    )
    return dict(config, resolved_server={key: server.get(key) for key in keys})


def smoke(args, case_dir, deadline):
    model = http("/get_model_info", None, deadline, 30)
    server = http("/get_server_info", None, deadline, 30)
    text, meta = chat(
        "What is 2 + 2? Answer with the number only.",
        args.model,
        deadline,
        32,
        stream=True,
    )
    info = {
        "model": model,
        "server": server,
        "output": text,
        "metadata": meta,
        "endpoint": "streaming /v1/chat/completions",
        "transport": "ok",
        "quality": "expected answer observed" if text.strip() == "4" else "unresolved",
    }
    save(case_dir / "smoke.json", info)
    return info


RENDER = r"""
import json, sys
from transformers import AutoTokenizer
p = json.loads(sys.argv[2])
t = AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=True)
r = []
for prompt in p:
 messages = [{'role':'user','content':prompt}]
 rendered = t.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
 r.append({'user_prompt':prompt,'rendered_prompt':rendered,
           'input_ids':t.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)})
print('PRTEST_RENDER='+json.dumps({'tokenizer_class':type(t).__name__, 'tokenizer_module':type(t).__module__,
 'special_tokens':t.special_tokens_map, 'chat_template':t.chat_template, 'prompts':r}))
"""


def diagnose_baseline(args, worktree, env, case_dir, deadline):
    prompts = [prompt for _, prompt in ACCURACY_PROMPTS[:3]]
    command = server_shell(
        args,
        worktree,
        env,
        [args.container_python, "-c", RENDER, args.model, json.dumps(prompts)],
    )
    output = run_capture(args, command, deadline, timeout=180)
    rendered = next(
        json.loads(line.split("=", 1)[1])
        for line in output.splitlines()
        if line.startswith("PRTEST_RENDER=")
    )
    details = []
    save(case_dir / "tokenizer.json", rendered)
    for item in rendered["prompts"]:
        result = dict(item)
        for mode in ("raw", "rendered", "ids", "chat"):
            try:
                if mode == "chat":
                    text, meta = chat(item["user_prompt"], args.model, deadline)
                elif mode == "ids":
                    response = http(
                        "/generate",
                        {
                            "input_ids": item["input_ids"],
                            "sampling_params": {"temperature": 0, "max_new_tokens": 64},
                        },
                        deadline,
                        7200,
                    )
                    text, meta = response["text"], response.get("meta_info")
                else:
                    text, meta = generate(
                        item["user_prompt"]
                        if mode == "raw"
                        else item["rendered_prompt"],
                        deadline,
                    )
                result[mode] = {"output": text, "metadata": meta}
            except (BudgetExpired, subprocess.TimeoutExpired):
                raise
            except Exception as exc:
                result[mode] = {"error": str(exc)}
        details.append(result)
        save(case_dir / "diagnosis.json", details)
    return {
        "status": "ok",
        "details": details,
        "interpretation": "device evidence; inspect saved outputs before accepting quality",
    }


def phase_names(args):
    if args.diagnose_baseline:
        return ["diagnostic-graph", "diagnostic-eager"]
    return ["accuracy", "speed"] if args.round == "all" else [args.round]


def run_phase(
    args, record, phase, worktree, case_dir, registry, deadline, log, checkpoint
):
    phase_dir = case_dir / phase
    phase_dir.mkdir()
    result = {"status": "running", "attempts": []}
    record.setdefault("phases", {})[phase] = result
    checkpoint()
    memfrac = args.memfrac
    graph = args.decode_graph == "on" and phase != "diagnostic-eager"
    for attempt_index in range(2 if args.retry_oom else 1):
        attempt_dir = phase_dir / f"attempt-{attempt_index + 1}"
        attempt_dir.mkdir()
        attempt = {"status": "running", "memfrac": memfrac, "decode_graph": graph}
        result["attempts"].append(attempt)
        state_path = remote_log = None
        server_text = ""
        try:
            require_free_port()
            runtime = probe_runtime(args, worktree, record["case_env"], deadline)
            attempt["runtime"] = runtime
            checkpoint()
            if (
                runtime.get(
                    "requires_sparse_fa_lse",
                    record["case"] in ("04-pr-f-sfa-kv-8bit", "07-combined"),
                )
                and not runtime["ops"]["npu_sparse_flash_attention_lse"]
            ):
                raise RuntimeError(
                    "required torch.ops.npu.npu_sparse_flash_attention_lse is unavailable"
                )
            argv = server_argv(args, memfrac, graph)
            command = server_shell(args, worktree, record["case_env"], argv)
            attempt.update(
                argv=argv,
                command=command,
                comparison_config=comparison_config(args, runtime, memfrac, graph),
            )
            save(attempt_dir / "launch.json", attempt)
            checkpoint()
            state_path, remote_log = launch_server(
                args, command, attempt_dir, registry, deadline
            )
            attempt["ownership"] = state_path
            checkpoint()
            wait_health(args, state_path, deadline, log)
            attempt["smoke"] = smoke(args, attempt_dir, deadline)
            attempt["comparison_config"] = resolved_comparison(
                attempt["comparison_config"], attempt["smoke"]
            )
            if attempt["comparison_config"] is None:
                attempt["comparison_unavailable"] = (
                    "resolved server configuration incomplete"
                )
            checkpoint()
            if phase == "accuracy":
                gsm = run_gsm8k(args.gsm_rows, args, attempt_dir, deadline, log)
                attempt["gsm8k"] = gsm
                checkpoint()
                attempt["accuracy"] = run_accuracy_prompts(
                    args, attempt_dir, deadline, log
                )
                attempt["status"] = gsm["status"]
                if attempt["accuracy"]["errors"]:
                    attempt.update(
                        status="failed", reason="paired-output requests failed"
                    )
                if gsm["status"] != "ok":
                    attempt["reason"] = "GSM8K requests failed or responses truncated"
                elif (
                    args.gsm8k_min_accuracy is not None
                    and gsm["accuracy"] < args.gsm8k_min_accuracy
                ):
                    attempt.update(
                        status="failed",
                        reason="GSM8K below configured accuracy threshold",
                    )
            elif phase == "speed":
                if args.ais_preflight_error:
                    raise RuntimeError(args.ais_preflight_error)
                attempt["speed"] = run_aisbench(args, attempt_dir, deadline)
                attempt["status"] = attempt["speed"]["status"]
                attempt["reason"] = attempt["speed"]["reason"]
            elif phase.startswith("diagnostic-"):
                attempt["diagnosis"] = diagnose_baseline(
                    args, worktree, record["case_env"], attempt_dir, deadline
                )
                attempt["status"] = "ok"
            else:
                attempt["status"] = "ok"
        except (KeyboardInterrupt, SystemExit):
            attempt.update(status="interrupted", reason="runner interrupted")
            result.update(status="interrupted", reason="runner interrupted")
            raise
        except Exception as exc:
            attempt.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
            log(f"{phase}: {attempt['reason']}")
        finally:
            if state_path:
                try:
                    stop_server(args, state_path)
                    require_free_port()
                    attempt["cleanup"] = "ok"
                except Exception as exc:
                    record["stop_failed"] = True
                    attempt.update(
                        status="failed", cleanup=f"{type(exc).__name__}: {exc}"
                    )
            if remote_log:
                try:
                    server_text = capture_server_log(args, remote_log, attempt_dir)
                except Exception as exc:
                    attempt["log_capture_error"] = str(exc)
            # Keep incomplete GSM8K/paired outputs after any exception.
            for filename, key in [
                ("gsm8k.json", "gsm8k"),
                ("accuracy.json", "accuracy"),
            ]:
                path = attempt_dir / filename
                if key not in attempt and path.exists():
                    attempt[key] = json.loads(path.read_text())
                    if attempt[key].get("status") == "running":
                        attempt[key]["status"] = "incomplete"
                        save(path, attempt[key])
            if not state_path and result["status"] == "running":
                # A detached launch can fail before returning its identity.
                # Registry cleanup still owns it and must finish before continuing.
                for process in registry["processes"]:
                    if not process.get("cleaned"):
                        try:
                            stop_server(args, process["state"])
                            process["cleaned"] = True
                        except Exception as exc:
                            record["stop_failed"] = True
                            attempt["cleanup"] = str(exc)
            if state_path and attempt.get("cleanup") == "ok":
                next(p for p in registry["processes"] if p["state"] == state_path)[
                    "cleaned"
                ] = True
            save(args.registry_path, registry)
            checkpoint()
        if result["status"] == "interrupted":
            break
        if attempt["status"] == "ok" or record.get("stop_failed"):
            break
        if (
            attempt_index == 0
            and args.retry_oom
            and re.search(r"OutOfMemoryError|NPU out of memory", server_text)
        ):
            if graph and re.search(
                r'File "[^"\n]*(?:cuda_graph_runner|npu_graph_runner)\.py"|in capture_one_batch',
                server_text[-12000:],
            ):
                graph = False
            elif memfrac - args.memfrac_step >= 0.55:
                memfrac = round(memfrac - args.memfrac_step, 3)
            else:
                break
            log(
                f"{phase}: OOM fallback is a separate attempt (memfrac={memfrac}, graph={graph})"
            )
            continue
        break
    chosen = result["attempts"][-1]
    result.update(status=chosen["status"], reason=chosen.get("reason", ""))
    checkpoint()
    return result


def selected_attempt(record, phase):
    attempts = record.get("phases", {}).get(phase, {}).get("attempts", [])
    return attempts[-1] if attempts else {}


def speed_data(record):
    attempt = selected_attempt(record, "speed")
    return attempt.get("speed", {}), attempt


def paired_data(record):
    attempt = selected_attempt(record, "accuracy")
    return attempt.get("accuracy", record.get("accuracy", {})) or {}


def quality(record):
    attempt = selected_attempt(record, "accuracy")
    gsm = attempt.get("gsm8k")
    if not gsm:
        return "unverified"
    if (
        gsm.get("status") != "ok"
        or not gsm.get("correct")
        or attempt.get("status") != "ok"
    ):
        return "unresolved"
    return (
        "scored; acceptance threshold not set"
        if record.get("quality_threshold") is None
        else "threshold met"
    )


def markdown(value):
    return str(value).replace("|", "\\|").replace("\n", " ")[:500]


def summarize(records, output, args):
    base = next((r for r in records if r["case"] == "00-baseline"), {})
    base_speed, base_attempt = speed_data(base)
    base_metrics = (
        base_speed.get("metrics") if base_speed.get("status") == "ok" else None
    )
    rows = [
        "# GLM PR test results",
        "",
        "TTFT uses milliseconds in JSON and seconds below. Ratios are single-run diagnostic samples.",
        "GSM8K quality and performance status are independent; unresolved quality does not stop timing collection.",
        "",
        "| Case | Status | GSM8K correct/total | Quality | TTFT mean/P90 (s) | Baseline/variant mean | Reason |",
        "|---|---|---|---|---|---|---|",
    ]
    for record in records:
        accuracy = selected_attempt(record, "accuracy")
        gsm = accuracy.get("gsm8k", {})
        gsm_text = (
            f"{gsm.get('correct', 0)}/{gsm['requested']} ({gsm.get('errors', 0)} errors, {gsm.get('truncated', 0)} truncated)"
            if gsm.get("requested")
            else "unavailable"
        )
        speed, attempt = speed_data(record)
        metrics = speed.get("metrics") if speed.get("status") == "ok" else None
        timing = (
            f"{float(metrics['TTFT avg']) / 1000:.3f} / {float(metrics['TTFT P90']) / 1000:.3f}"
            if metrics
            else "unavailable"
        )
        ratio = "unavailable"
        if record is base and metrics:
            ratio = "reference"
        elif (
            metrics
            and base_metrics
            and attempt.get("comparison_config") is not None
            and attempt.get("comparison_config")
            == base_attempt.get("comparison_config")
        ):
            ratio = (
                f"{float(base_metrics['TTFT avg']) / float(metrics['TTFT avg']):.3f}x"
            )
            if quality(base) not in ("threshold met",) or quality(record) not in (
                "threshold met",
            ):
                ratio += " (quality unqualified)"
        elif metrics and base_metrics:
            ratio = "configuration mismatch"
        reasons = [record.get("reason", "")]
        reasons += [
            f"{name}: {phase.get('reason', phase.get('status'))}"
            for name, phase in record.get("phases", {}).items()
            if phase.get("status") != "ok"
        ]
        rows.append(
            "| "
            + " | ".join(
                markdown(v)
                for v in [
                    record["case"],
                    record.get("status", "pending"),
                    gsm_text,
                    quality(record),
                    timing,
                    ratio,
                    "; ".join(filter(None, reasons)),
                ]
            )
            + " |"
        )
    repeat = next((r for r in records if r["case"] == "08-baseline-repeat"), {})
    repeat_speed, repeat_attempt = speed_data(repeat)
    if (
        base_metrics
        and repeat_speed.get("status") == "ok"
        and repeat_attempt.get("comparison_config") is not None
        and repeat_attempt.get("comparison_config")
        == base_attempt.get("comparison_config")
        and repeat.get("commit") == base.get("commit")
    ):
        drift = (
            float(repeat_speed["metrics"]["TTFT avg"]) / float(base_metrics["TTFT avg"])
            - 1
        ) * 100
        rows += ["", f"Baseline-repeat mean TTFT drift: {drift:+.2f}%."]
    else:
        rows += [
            "",
            "Baseline drift unavailable: both matching baseline measurements are required.",
        ]
    rows += [
        "",
        "## Paired-output diagnostics",
        "",
        "| Case | Exact matches / shared complete outputs |",
        "|---|---|",
    ]
    base_digest = paired_data(base).get("sha", {})
    for record in records:
        digest = paired_data(record).get("sha", {})
        shared = set(digest) & set(base_digest)
        count = sum(digest[tag] == base_digest[tag] for tag in shared)
        rows.append(
            f"| {record['case']} | {f'{count}/{len(shared)}' if shared else 'unavailable'} |"
        )
    rows += [
        "",
        "Request failures and truncated responses are excluded from exact-output comparisons.",
        "Missing/failed metrics are never passes. Historical replays lack complete runtime provenance.",
        "See results.json and phase attempt directories for commands, SHAs, deviations, and raw evidence.",
        "",
    ]
    temporary = output / "SUMMARY.md.tmp"
    temporary.write_text("\n".join(rows))
    temporary.replace(output / "SUMMARY.md")


def checkpoint_results(records, output, args, log):
    save(output / "results.json", records)
    try:
        summarize(records, output, args)
    except Exception as exc:
        # Results were already persisted; a presentation failure cannot erase them.
        log(f"REPORT ERROR: {type(exc).__name__}: {exc}")
        save(output / "report-error.json", {"error": f"{type(exc).__name__}: {exc}"})
        args.report_failed = True


def regenerate(args):
    source = Path(args.regenerate).resolve()
    output = Path(args.output).resolve()
    if output == source or source in output.parents:
        raise ValueError(
            "regeneration output must be outside the source evidence directory"
        )
    output.mkdir(parents=True, exist_ok=False)
    settings = json.loads((source / "settings.json").read_text())
    records_path = source / "results.json"
    records = json.loads(records_path.read_text()) if records_path.exists() else []
    by_name = {record["case"]: record for record in records}
    for directory in sorted(source.iterdir()):
        if not directory.is_dir():
            continue
        if directory.name not in {case[0] for case in CASES}:
            continue
        record = by_name.setdefault(
            directory.name, {"case": directory.name, "ref": "", "status": "incomplete"}
        )
        current_format = "phases" in record
        phases = record.setdefault("phases", {})
        for phase_name, phase in phases.items():
            for index, attempt in enumerate(phase.get("attempts", []), 1):
                attempt_dir = directory / phase_name / f"attempt-{index}"
                log_path = attempt_dir / "ais.log"
                if phase_name == "speed" and log_path.exists():
                    old_speed = attempt.get("speed", {})
                    speed = validate_ais(
                        log_path.read_text(errors="replace"),
                        old_speed.get("returncode"),
                        settings.get("requests", 16),
                        settings.get("length"),
                    )
                    attempt["speed"] = speed
                    attempt["source_log"] = str(log_path)
                    if speed["status"] != "ok":
                        phase.update(status="failed", reason=speed["reason"])
                for filename, key in [
                    ("gsm8k.json", "gsm8k"),
                    ("accuracy.json", "accuracy"),
                ]:
                    saved = attempt_dir / filename
                    if saved.exists():
                        attempt[key] = json.loads(saved.read_text())
        path = directory / "ais.log"
        if path.exists():
            code = record.get("ais_returncode")
            speed = validate_ais(
                path.read_text(errors="replace"),
                code,
                settings.get("requests", 16),
                settings.get("length"),
            )
            phases["speed"] = {
                "status": speed["status"],
                "reason": speed["reason"],
                "attempts": [
                    {
                        "status": speed["status"],
                        "speed": speed,
                        "source_log": str(path),
                        "comparison_config": None,
                        "provenance": "historical; effective runtime not fully recorded",
                    }
                ],
            }
        path = directory / "accuracy.json"
        if path.exists():
            acc = json.loads(path.read_text())
            # Old SHA dictionaries may include hashes of error strings.
            outputs = acc.get("outputs", {})
            valid = {
                tag: value
                for tag, value in acc.get("sha", {}).items()
                if tag in outputs
                and isinstance(outputs[tag], str)
                and outputs[tag]
                and not outputs[tag].startswith("<request error:")
            }
            phases["accuracy"] = {
                "status": "diagnostic",
                "reason": "historical exact-output comparison only",
                "attempts": [
                    {"accuracy": {"sha": valid, "errors": acc.get("errors", 0)}}
                ],
            }
        record["original_status"] = record.get("status")
        record["original_reason"] = record.pop("reason", None)
        if any(p.get("status") == "failed" for p in phases.values()):
            record["status"] = "failed"
        elif current_format or phases.get("speed", {}).get("status") != "ok":
            record["status"] = record["original_status"]
            if record["original_reason"]:
                record["reason"] = record["original_reason"]
        else:
            record["status"] = "diagnostic"
        if record["status"] in ("pending", "running"):
            record["status"] = "incomplete"
    ordered = sorted(by_name.values(), key=lambda record: record["case"])
    save(output / "settings.json", {"source": str(source), "source_settings": settings})
    save(output / "results.json", ordered)
    summarize(ordered, output, args)
    print(output / "SUMMARY.md")
    return 0


def plan_cases(args):
    wanted = args.cases.split(",") if args.cases else [case[0] for case in CASES]
    unknown = set(wanted) - {case[0] for case in CASES}
    if unknown or len(set(wanted)) != len(wanted):
        raise ValueError(
            "unknown or duplicate cases: " + ",".join(sorted(unknown or set(wanted)))
        )
    cases = [case for case in CASES if case[0] in wanted]
    if args.skip_baseline_bracket:
        cases = [case for case in cases if "baseline" not in case[0]]
    if args.diagnose_baseline:
        cases = [CASES[0]]
    if not cases:
        raise ValueError("no cases selected")
    return cases


def apply_overrides(cases, overrides):
    table = {name: [name, ref, dict(env), note] for name, ref, env, note in cases}
    for item in overrides.split(",") if overrides else []:
        try:
            name, flag = item.split(":", 1)
            key, value = flag.split("=", 1)
        except ValueError as exc:
            raise ValueError("override must be case:ENV=value") from exc
        if name not in table or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(
                "unknown case or invalid environment key in override: " + item
            )
        table[name][2][key] = value
    return list(table.values())


def parser():
    p = argparse.ArgumentParser(
        description="Reliable GLM PR GSM8K and AISBench benchmarks"
    )
    p.add_argument(
        "--round", choices=("smoke", "accuracy", "speed", "all"), default="all"
    )
    p.add_argument("--cases", default="")
    p.add_argument("--flags-overrides", default="")
    p.add_argument("--container", default="sglang-a3")
    p.add_argument("--container-python", default="python3")
    p.add_argument("--repo", default="/home/swx1199799/megamoe_test/sglang")
    p.add_argument("--model", default="/home/weights/GLM5.2-w4a8-int")
    p.add_argument("--fetch-url", default=FORK_URL_DEFAULT)
    p.add_argument(
        "--git-proxy",
        default="url.https://ghfast.top/https://github.com/.insteadOf=https://github.com/",
    )
    p.add_argument(
        "--container-cann-env",
        default="/home/tbaydasov/Ascend/cann-9.2.0-beta.2/set_env.sh",
    )
    p.add_argument("--no-fetch", action="store_true")
    p.add_argument("--ais-dir", default=AIS_DIR_DEFAULT)
    p.add_argument("--ais-env", default=None)
    p.add_argument("--ais-conda-env", default="tbay_sglang")
    p.add_argument(
        "--gsm8k", help="host path to GSM8K test JSONL; required for accuracy/all"
    )
    p.add_argument("--gsm8k-samples", type=int, default=200)
    p.add_argument("--gsm8k-seed", type=int, default=4)
    p.add_argument("--gsm8k-max-tokens", type=int, default=512)
    p.add_argument("--gsm8k-min-accuracy", type=float, default=None)
    p.add_argument("--length", type=int, default=1000000)
    p.add_argument("--requests", type=int, default=16)
    p.add_argument("--chunk", type=int, default=16384)
    p.add_argument("--context", type=int, default=1024000)
    p.add_argument("--memfrac", type=float, default=0.70)
    p.add_argument("--memfrac-step", type=float, default=0.06)
    p.add_argument(
        "--retry-oom",
        action="store_true",
        help="opt in to a separately recorded graph/memory fallback",
    )
    p.add_argument("--decode-graph", choices=("on", "off"), default="on")
    p.add_argument("--server-seed", type=int, default=4)
    p.add_argument("--accuracy-requests", type=int, default=64)
    p.add_argument("--accuracy-long-tokens", type=int, default=32768)
    p.add_argument("--startup-timeout", type=float, default=1800)
    p.add_argument("--benchmark-timeout", type=float, default=14400)
    p.add_argument("--hours", type=float, default=10)
    p.add_argument("--case-minutes", type=float, default=60)
    p.add_argument("--extra-server-args", default="")
    p.add_argument("--skip-baseline-bracket", action="store_true")
    p.add_argument("--state-dir", default=f"/tmp/prtest-owned-{os.getuid()}")
    p.add_argument(
        "--save-dir",
        default="/home/swx1199799/",
        help="parent directory for prtest-results/<run-id> (default: /home/swx1199799/)",
    )
    p.add_argument("--output", help="exact run directory; overrides --save-dir")
    p.add_argument(
        "--regenerate",
        metavar="SOURCE",
        help="regenerate historical reports offline into --output",
    )
    p.add_argument(
        "--diagnose-baseline",
        action="store_true",
        help="save graph/eager chat/raw/token-ID diagnostics",
    )
    p.add_argument(
        "--stop-only",
        action="store_true",
        help="stop only processes recorded in this runner state directory",
    )
    p.add_argument("--dry-run", action="store_true")
    return p


def validate_args(args):
    for key in (
        "gsm8k_samples",
        "gsm8k_max_tokens",
        "length",
        "requests",
        "chunk",
        "context",
        "accuracy_long_tokens",
        "startup_timeout",
        "benchmark_timeout",
        "hours",
        "case_minutes",
    ):
        value = getattr(args, key)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{key.replace('_', '-')} must be positive and finite")
    if (
        not 0 <= args.accuracy_requests
        or not 0.0 < args.memfrac < 1
        or not 0 < args.memfrac_step < 1
    ):
        raise ValueError("invalid accuracy request count or memory fraction")
    if args.gsm8k_min_accuracy is not None and not 0 <= args.gsm8k_min_accuracy <= 1:
        raise ValueError("--gsm8k-min-accuracy must be within [0,1]")
    if args.length + 1 >= args.context:
        raise ValueError("--context must exceed --length + output token")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.container):
        raise ValueError("invalid container name")
    # Prevent hidden overrides of the serving recipe from invalidating attribution.
    extras = shlex.split(args.extra_server_args)
    controlled = {value for value in BASE_SERVER_ARGS if value.startswith("--")}
    controlled |= {
        "--model-path",
        "--load-format",
        "--random-seed",
        "--host",
        "--cuda-graph-backend-decode",
        "--disable-cuda-graph",
    }
    if any(value.split("=", 1)[0] in controlled for value in extras):
        raise ValueError(
            "--extra-server-args overrides a controlled option; use its dedicated flag"
        )
    if args.no_fetch:
        args.fetch_url = ""
    args.ais_env = args.ais_env or next(
        (path for path in AIS_ENV_CANDIDATES if Path(path).is_file()),
        AIS_ENV_CANDIDATES[0],
    )
    args.cann_env_resolved = args.container_cann_env
    return apply_overrides(plan_cases(args), args.flags_overrides)


def dry_run(args, cases):
    for name, ref, env, _ in cases:
        worktree = "<run-root>/worktrees/" + name
        print(f"== {name}: pin {ref} once; isolated worktree {worktree}")
        for phase in phase_names(args):
            graph = args.decode_graph == "on" and phase != "diagnostic-eager"
            print(
                f"  {phase}: fresh server, shared case cap={args.case_minutes} minutes"
            )
            print(
                "  "
                + server_shell(
                    args, worktree, env, server_argv(args, args.memfrac, graph)
                )
            )
            if phase == "speed":
                print("  AIS preflight: " + shlex.join(ais_command(args, True)))
                print(
                    "  require successful cache flush; " + shlex.join(ais_command(args))
                )
            if phase == "accuracy":
                print(
                    f"  GSM8K: {args.gsm8k_samples} samples, seed={args.gsm8k_seed}, dataset={args.gsm8k or '<required at execution>'}"
                )
    return 0


def cleanup_registry(args, registry, log):
    errors = []
    for process in registry.get("processes", []):
        if process.get("cleaned"):
            continue
        try:
            stop_server(args, process["state"])
            process["cleaned"] = True
            save(args.registry_path, registry)
        except Exception as exc:
            errors.append(str(exc))
    if errors:
        raise RuntimeError("ownership cleanup failed: " + "; ".join(errors))
    registered = None
    for path in list(registry.get("worktrees", [])):
        if registered is None:
            listing = git(
                args, "worktree", "list", "--porcelain", deadline=time.monotonic() + 30
            )
            registered = {
                line[9:]
                for line in listing.splitlines()
                if line.startswith("worktree ")
            }
        if path in registered:
            git(
                args,
                "worktree",
                "remove",
                "--force",
                path,
                deadline=time.monotonic() + 30,
            )
        else:
            # A timed-out add may not have created a registered worktree.
            run_capture(args, "test ! -e " + shlex.quote(path), time.monotonic() + 15)
        registry["worktrees"].remove(path)
        save(args.registry_path, registry)
    require_free_port()
    log("owned processes stopped; port free")


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        cases = validate_args(args)
        if args.regenerate:
            if not args.output:
                raise ValueError("--regenerate requires a new --output directory")
            return regenerate(args)
        if args.dry_run:
            return dry_run(args, cases)
        if (
            not args.stop_only
            and not args.diagnose_baseline
            and args.round in ("accuracy", "all")
        ):
            if not args.gsm8k:
                raise ValueError(
                    "--gsm8k is required for accuracy/all; speed runs independently with --round speed"
                )
            args.gsm_rows = load_gsm8k(args.gsm8k, args.gsm8k_samples, args.gsm8k_seed)
            args.dataset_sha256 = hashlib.sha256(
                Path(args.gsm8k).read_bytes()
            ).hexdigest()
    except (ValueError, OSError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    state_dir = Path(args.state_dir) / args.container
    state_dir.mkdir(parents=True, exist_ok=True)
    args.registry_path = state_dir / "registry.json"
    with open(Path(args.state_dir) / f"port-{SERVER_PORT}.lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(
                "ERROR: runner is active; interrupt it to trigger owned cleanup",
                file=sys.stderr,
            )
            return 2
        if args.registry_path.exists():
            registry = json.loads(args.registry_path.read_text())
            if (
                registry.get("container") != args.container
                or registry.get("repo") != args.repo
            ):
                print(
                    "ERROR: ownership registry container/repo mismatch", file=sys.stderr
                )
                return 2
            try:
                cleanup_registry(args, registry, print)
            except Exception as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 1
        if args.stop_only:
            try:
                # Probe Docker even when no registry exists.
                run_capture(args, "true", time.monotonic() + 15, 10)
                require_free_port()
            except Exception as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 1
            print("owned processes stopped; port free")
            return 0
        return run(args, cases)


def results_directory(args):
    if args.output:
        return Path(args.output).expanduser().resolve()
    return (
        Path(args.save_dir).expanduser().resolve()
        / "prtest-results"
        / f"{now()}-{uuid.uuid4().hex[:6]}"
    )


def run(args, cases):
    output = results_directory(args)
    output.mkdir(parents=True, exist_ok=False)
    args.remote_root = "/tmp/prtest-owned/" + uuid.uuid4().hex
    registry = {
        "container": args.container,
        "repo": args.repo,
        "root": args.remote_root,
        "processes": [],
        "worktrees": [],
    }
    save(args.registry_path, registry)
    deadline = time.monotonic() + args.hours * 3600
    args.report_failed = False
    args.ais_preflight_record = None
    args.ais_preflight_error = None
    records = [
        {
            "case": name,
            "ref": ref,
            "case_env": env,
            "env": dict(BASE_ENV, **env),
            "note": note,
            "status": "pending",
            "phases": {},
            "quality_threshold": args.gsm8k_min_accuracy,
        }
        for name, ref, env, note in cases
    ]

    def log(message):
        line = (
            f"[{datetime.datetime.now(datetime.timezone.utc):%H:%M:%S} UTC] {message}"
        )
        print(line, flush=True)
        with open(output / "runner.log", "a") as handle:
            handle.write(line + "\n")

    def checkpoint():
        checkpoint_results(records, output, args, log)

    interrupted = False
    fatal = None
    checkpoint()
    save(
        output / "settings.json",
        {
            key: value
            for key, value in vars(args).items()
            if key not in ("gsm_rows", "registry_path")
        },
    )
    try:
        require_free_port()
        args.cann_env_resolved = resolve_cann_env(args, deadline, log)
        if "speed" in phase_names(args):
            try:
                args.ais_preflight_record = ais_preflight(args, deadline)
            except Exception as exc:
                args.ais_preflight_error = f"{type(exc).__name__}: {exc}"
                log(args.ais_preflight_error)
        save(
            output / "settings.json",
            {
                key: value
                for key, value in vars(args).items()
                if key not in ("gsm_rows", "registry_path")
            },
        )
        pinned = pin_refs(
            cases,
            args,
            deadline,
            lambda values: save(output / "pinned-refs.json", values),
        )
        for record in records:
            if time.monotonic() >= deadline:
                record.update(status="skipped", reason="global time budget exhausted")
                checkpoint()
                continue
            pin = pinned[record["ref"]]
            if "error" in pin:
                record.update(status="skipped", reason=pin["error"])
                checkpoint()
                continue
            record["commit"] = pin["commit"]
            case_deadline = min(deadline, time.monotonic() + args.case_minutes * 60)
            try:
                remaining(case_deadline)
                log(f"== {record['case']} {record['commit'][:12]}")
                case_dir = output / record["case"]
                case_dir.mkdir()
                worktree = create_worktree(args, record, registry, case_deadline)
                record["worktree"] = worktree
                for phase in phase_names(args):
                    if time.monotonic() >= case_deadline:
                        record["phases"][phase] = {
                            "status": "skipped",
                            "reason": "case/global time budget exhausted",
                            "attempts": [],
                        }
                        checkpoint()
                        continue
                    if phase == "speed" and args.ais_preflight_error:
                        record["phases"][phase] = {
                            "status": "failed",
                            "reason": args.ais_preflight_error,
                            "attempts": [],
                        }
                        checkpoint()
                        continue
                    run_phase(
                        args,
                        record,
                        phase,
                        worktree,
                        case_dir,
                        registry,
                        case_deadline,
                        log,
                        checkpoint,
                    )
                    if record.get("stop_failed"):
                        raise RuntimeError("server cleanup failed; halting sweep")
                record["status"] = (
                    "ok"
                    if all(p["status"] == "ok" for p in record["phases"].values())
                    else "failed"
                )
            except (KeyboardInterrupt, SystemExit):
                record.update(status="interrupted", reason="runner interrupted")
                raise
            except Exception as exc:
                record.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
                log(record["reason"])
                if record.get("stop_failed"):
                    raise
            finally:
                checkpoint()
    except (KeyboardInterrupt, SystemExit):
        interrupted = True
        fatal = "runner interrupted"
    except Exception as exc:
        fatal = f"{type(exc).__name__}: {exc}"
        log("STOP: " + fatal)
    finally:
        for record in records:
            if record["status"] == "pending":
                record.update(status="skipped", reason=fatal or "time budget exhausted")
            for phase in phase_names(args):
                record["phases"].setdefault(
                    phase,
                    {
                        "status": "skipped",
                        "reason": record.get("reason", "not executed"),
                        "attempts": [],
                    },
                )
        try:
            cleanup_registry(args, registry, log)
        except Exception as exc:
            fatal = f"cleanup failed: {exc}"
            log("STOP: " + fatal)
        checkpoint()
        save(
            output / "run-status.json",
            {
                "fatal": fatal,
                "interrupted": interrupted,
                "report_failed": args.report_failed,
            },
        )
    log("results: " + str(output / "SUMMARY.md"))
    return (
        130
        if interrupted
        else int(
            bool(
                fatal
                or args.report_failed
                or any(record["status"] != "ok" for record in records)
            )
        )
    )


def _interrupt(signum, frame):
    raise KeyboardInterrupt(f"signal {signum}")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _interrupt)
    sys.exit(main())
