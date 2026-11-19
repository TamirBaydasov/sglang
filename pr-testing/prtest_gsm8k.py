#!/usr/bin/env python3
"""Five-shot GSM8K accuracy against the server launched by the current A3 script."""

import argparse
import ast
import datetime
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

STOPS = ["Question", "Assistant:", "<|separator|>"]
DATA_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
SCORER = "legacy-last-integer-v1"


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def request(base, route, payload=None, timeout=30):
    body = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        base + route, data=body, headers={"Content-Type": "application/json"}
    )
    # The local server must not be routed through a host HTTP proxy.
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
        req, timeout=timeout
    ) as response:
        return json.load(response)


def answer_value(text):
    # Preserve the attached harness's scorer for comparable historical results.
    numbers = re.findall(r"\d+", text.replace(",", ""))
    if not numbers:
        return None
    try:
        return ast.literal_eval(numbers[-1])
    except (ValueError, SyntaxError):
        return None


def example(row, with_answer=False):
    text = "Question: " + row["question"] + "\nAnswer:"
    return text + (" " + row["answer"] if with_answer else "")


def evaluate(base, prompt_id, prompt, label, sampling, timeout):
    row = dict(
        prompt_id=prompt_id,
        prompt=prompt,
        prompt_sha256=digest(prompt.encode()),
        label=label,
        output="",
        prediction=None,
        correct=False,
        invalid=True,
    )
    start = time.monotonic()
    try:
        result = request(
            base,
            "/generate",
            dict(text=prompt, sampling_params=sampling, stream=False),
            timeout,
        )
        output = result["text"]
        if not isinstance(output, str):
            raise ValueError("Server did not return one text completion")
        meta = result.get("meta_info", {})
        row.update(output=output, prediction=answer_value(output), meta_info=meta)
        row["invalid"] = row["prediction"] is None
        row["correct"] = not row["invalid"] and row["prediction"] == label
        reason = meta.get("finish_reason", {})
        row["truncated"] = (
            reason.get("type") == "length"
            if isinstance(reason, dict)
            else reason == "length"
        )
    except Exception as exc:
        row["error"] = str(exc)
    row["elapsed_s"] = time.monotonic() - start
    return row


def pair(reference, current, output):
    previous = json.loads((reference / "manifest.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    for key in (
        "dataset_sha256",
        "shots_sha256",
        "sampling",
        "num_questions",
        "parallel",
        "scorer",
        "prompt_mode",
    ):
        if previous.get(key) != manifest.get(key):
            raise ValueError(f"Reference differs in {key}; refusing paired comparison")
    if previous["variant"] == manifest["variant"]:
        raise ValueError("Reference must use a different backend variant")
    for key in (
        "model_path",
        "quantization",
        "dtype",
        "tp_size",
        "dcp_size",
        "chunked_prefill_size",
        "max_prefill_tokens",
        "max_running_requests",
        "prefill_max_requests",
        "max_total_tokens",
        "context_length",
        "cuda_graph_backend_decode",
        "cuda_graph_backend_prefill",
        "mem_fraction_static",
    ):
        if previous["server_args"].get(key) != manifest["server_args"].get(key):
            raise ValueError(
                f"Reference server differs in {key}; refusing paired comparison"
            )
    records = [
        json.loads(line)
        for line in (reference / "raw.jsonl").read_text().splitlines()
        if line.strip()
    ]
    before = {r["prompt_id"]: r for r in records}
    after = {r["prompt_id"]: r for r in current}
    if len(before) != len(records) or set(before) != set(after):
        raise ValueError("Reference contains duplicate or different prompt IDs")
    for key in before:
        if (
            before[key]["prompt_sha256"] != after[key]["prompt_sha256"]
            or before[key]["label"] != after[key]["label"]
        ):
            raise ValueError(f"Prompt/label mismatch at {key}")
    if any(
        r.get("error") or r.get("truncated")
        for r in [*before.values(), *after.values()]
    ):
        raise ValueError(
            "Request failures or truncated completions prevent paired qualification"
        )
    lost = [i for i in before if before[i]["correct"] and not after[i]["correct"]]
    gained = [i for i in before if after[i]["correct"] and not before[i]["correct"]]
    discordant = len(lost) + len(gained)
    tail = sum(math.comb(discordant, j) for j in range(min(len(lost), len(gained)) + 1))
    pvalue = min(1.0, 2 * (tail / (1 << discordant)))
    result = dict(
        reference=str(reference),
        reference_variant=previous["variant"],
        current_variant=manifest["variant"],
        questions=len(current),
        reference_only_correct=lost,
        current_only_correct=gained,
        discordant=discordant,
        mcnemar_exact_two_sided_p=pvalue,
        conclusion="Identical correctness outcomes"
        if not discordant
        else "Significant difference; inspect direction and raw answers"
        if pvalue <= 0.05
        else "No significant difference detected; equivalence is not established",
    )
    save(output / "paired.json", result)
    return result


def validate_server(server, variant):
    backend = {"baseline": "none", "deepep": "deepep", "megamoe": "ascend_fuseep"}[
        variant
    ]
    expected = dict(
        tp_size=16,
        dcp_size=16,
        ep_size=1 if variant == "baseline" else 16,
        moe_a2a_backend=backend,
    )
    if variant == "megamoe":
        expected["fuseep_mode"] = 3
    for key, value in expected.items():
        if server.get(key) != value:
            raise ValueError(f"Wrong server: {key}={server.get(key)}, expected {value}")
    if server.get("load_format") == "dummy":
        raise ValueError("Accuracy requires real weights")


def replay_rows(path, count):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows.sort(key=lambda row: row["prompt_id"])
    if len(rows) < count or len({row["prompt_id"] for row in rows}) != len(rows):
        raise ValueError("Replay needs enough unique prompt IDs")
    for row in rows:
        if (
            not isinstance(row.get("prompt"), str)
            or not row["prompt"]
            or not isinstance(row.get("label"), int)
        ):
            raise ValueError("Replay requires saved prompt text and integer labels")
        if digest(row["prompt"].encode()) != row.get("prompt_sha256"):
            raise ValueError("Replay prompt hash mismatch")
    return rows[:count]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 30000)))
    parser.add_argument(
        "--num-questions", type=int, default=int(os.environ.get("NUM", 1319))
    )
    parser.add_argument(
        "--parallel", type=int, default=int(os.environ.get("PARALLEL", 16))
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=int(os.environ.get("MAXTOK", 1024))
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(os.environ.get("GSM8K_DATA", "gsm8k_test.jsonl")),
    )
    parser.add_argument(
        "--replay-jsonl",
        type=Path,
        help="Replay exact saved prompts and labels; diagnostic accuracy",
    )
    parser.add_argument(
        "--shots-data",
        type=Path,
        help="Optional separate shot dataset; default matches attachment's test-set shots",
    )
    parser.add_argument(
        "--variant", choices=("megamoe", "baseline", "deepep"), default="megamoe"
    )
    parser.add_argument(
        "--timeout", type=float, default=900, help="Per-request timeout in seconds"
    )
    parser.add_argument(
        "--reference", type=Path, help="Other variant's completed output directory"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.num_questions, args.parallel, args.max_new_tokens, args.timeout) <= 0:
        parser.error(
            "Question count, parallelism, output limit and timeout must be positive"
        )
    base = f"http://{args.host}:{args.port}"
    try:
        info = request(base, "/server_info")
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        info = request(base, "/get_server_info")
    server = info.get("server_args", info)
    validate_server(server, args.variant)
    width = server.get("max_running_requests")
    print(
        json.dumps(
            dict(
                server=base,
                variant=args.variant,
                resolved_max_running_requests=width,
                parallel=args.parallel,
                graphs=server.get("cuda_graph_config"),
            ),
            default=str,
        ),
        flush=True,
    )
    if width and args.parallel > width:
        print(
            f"Client parallelism {args.parallel} exceeds server width {width}; surplus requests queue.",
            flush=True,
        )
    if args.replay_jsonl:
        saved = replay_rows(args.replay_jsonl, args.num_questions)
        dataset = args.replay_jsonl.read_bytes()
        prefix = ""
        overlap = []
        source_manifest = args.replay_jsonl.with_name("manifest.json")
        if source_manifest.exists():
            source = json.loads(source_manifest.read_text())
            overlap = [
                i
                for i in source.get("shot_overlap_question_ids", [])
                if i in {r["prompt_id"] for r in saved}
            ]
        tasks = [(r["prompt_id"], r["prompt"], r["label"]) for r in saved]
    else:
        if not args.data_path.exists():
            args.data_path.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(DATA_URL, timeout=120) as response:
                content = response.read()
            # Parse before saving to avoid retaining an incomplete download.
            for line in content.decode().splitlines():
                json.loads(line)
            args.data_path.write_bytes(content)
        dataset = args.data_path.read_bytes()
        rows = [
            json.loads(line) for line in dataset.decode().splitlines() if line.strip()
        ]
        shots_bytes = args.shots_data.read_bytes() if args.shots_data else dataset
        shots = [
            json.loads(line)
            for line in shots_bytes.decode().splitlines()
            if line.strip()
        ]
        if len(rows) < args.num_questions or len(shots) < 5:
            raise ValueError("Dataset cannot supply requested questions and five shots")
        prefix = "".join(example(row, True) + "\n\n" for row in shots[:5])
        shot_questions = {row["question"] for row in shots[:5]}
        overlap = [
            i
            for i, row in enumerate(rows[: args.num_questions])
            if row["question"] in shot_questions
        ]
        tasks = [
            (i, prefix + example(row), answer_value(row["answer"]))
            for i, row in enumerate(rows[: args.num_questions])
        ]
    sampling = dict(
        temperature=0, top_p=1, max_new_tokens=args.max_new_tokens, stop=STOPS
    )
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    output = args.output or Path(f"gsm8k-{args.variant}-{stamp}")
    output.mkdir(parents=True, exist_ok=False)
    save(output / "server_info.json", info)
    manifest = dict(
        variant=args.variant,
        server=base,
        server_args=server,
        num_questions=args.num_questions,
        parallel=args.parallel,
        sampling=sampling,
        num_shots=5,
        scorer=SCORER,
        dataset_sha256=digest(dataset),
        shots_sha256=digest(prefix.encode()),
        prompt_mode="saved-prompt-replay"
        if args.replay_jsonl
        else "raw-completion-legacy-five-shot",
        replay_source=str(args.replay_jsonl) if args.replay_jsonl else None,
        script_sha256=digest(Path(__file__).read_bytes()),
        legacy_test_shots=args.shots_data is None,
        shot_overlap_question_ids=overlap,
        runtime_identity="HTTP server configuration captured; CANN identity belongs to the server launch artifacts",
    )
    save(output / "manifest.json", manifest)
    print(f"Results: {output.resolve()}", flush=True)
    if overlap:
        print(
            f"Shot/test overlap: {len(overlap)} evaluated questions are also examples; this is not a held-out GSM8K score.",
            flush=True,
        )
    results = []
    started = time.monotonic()
    with (
        (output / "raw.jsonl").open("w") as raw,
        ThreadPoolExecutor(max_workers=args.parallel) as pool,
    ):
        futures = []
        for index, prompt, label in tasks:
            if label is None:
                raise ValueError(f"Unparseable dataset answer at {index}")
            futures.append(
                pool.submit(
                    evaluate, base, index, prompt, label, sampling, args.timeout
                )
            )
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            raw.write(json.dumps(result) + "\n")
            raw.flush()
            if len(results) % 8 == 0 or len(results) == args.num_questions:
                print(f"Completed {len(results)}/{args.num_questions}", flush=True)
    wall = time.monotonic() - started
    failures = sum(bool(r.get("error")) for r in results)
    truncated = sum(bool(r.get("truncated")) for r in results)
    summary = dict(
        questions=len(results),
        correct=sum(r["correct"] for r in results),
        accuracy=sum(r["correct"] for r in results) / len(results),
        invalid=sum(r["invalid"] for r in results),
        invalid_rate=sum(r["invalid"] for r in results) / len(results),
        request_failures=failures,
        truncated=truncated,
        wall_s=wall,
        output_tokens=sum(
            (r.get("meta_info", {}).get("completion_tokens") or 0) for r in results
        ),
        status="request-failed"
        if failures
        else "truncated-diagnostic"
        if truncated
        else "scored-unqualified",
        held_out=False if args.replay_jsonl else not bool(overlap),
    )
    save(output / "result.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if args.reference:
        print(json.dumps(pair(args.reference, results, output), indent=2), flush=True)
    return int(failures > 0)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
