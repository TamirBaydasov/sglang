# GLM PR accuracy and performance harness

Run GSM8K accuracy, AISBench prefix-cache performance, or both for the six PRs,
the combined branch, and the two baseline brackets. The GSM8K client and
PROFILE=0 wrapper are extracted verbatim from the two PR #1 runbooks;
`REFERENCE_SOURCES.json` records their source and extracted-content hashes. `all` starts separate fresh
servers for accuracy and speed. The driver runs on the A3 **host**; serving runs
inside `sglang-a3`. Python 3.9+ is required on both sides.

The serving recipe stays TP16/DCP16, context 1024000, chunk 16384, maximum 16
running requests, one request per prefill batch, decode graphs on, static memory
fraction 0.70, and TP-MoE. AISBench prepares a 99% shared prefix and measures
16 concurrent 1M-token requests with one output token. A fixed server seed of 4
is recorded. Each unique branch is resolved once; both baseline brackets use
the same SHA. The driver's original checkout is never switched or reset.

| Case | Branch | Enabled feature |
|---|---|---|
| 00-baseline | dcp_ascend_testing | baseline |
| 01-pr-e-chat-encode | npu-chat-encode-pool | chat pool, memo, raw encoding |
| 02-pr-b-indexer-int8 | npu-dsa-indexer-int8 | INT8 indexer |
| 03-pr-prefetch-packed | npu-dcp-packed-read | packed read, gather prefetch, PG priority |
| 04-pr-f-sfa-kv-8bit | npu-sfa-kv-8bit | 8-bit SFA KV and quantized read |
| 05-pr-ts-token-sharded | npu-dsa-cp-token-sharded | token sharding and streamed projections |
| 06-pr-moe-int8 | npu-dsa-cp-moe-int8 | token sharding with INT8 MoE input |
| 07-combined | npu-bde-combined | combined deployment flags |
| 08-baseline-repeat | dcp_ascend_testing | drift control |

## Deploy the repairs

Copy `glm-pr-harness-repair.tar.gz` to the A3 host. Extract into a fresh directory:

```bash
mkdir -p /tmp/glm-pr-harness-repair
tar -xzf /tmp/glm-pr-harness-repair.tar.gz -C /tmp/glm-pr-harness-repair
cd /tmp/glm-pr-harness-repair
sha256sum --check CHECKSUMS.sha256
python3 run_pr_tests.py --round speed --cases 00-baseline --dry-run
```

The archive contains the driver, process helper, reference clients, provenance
manifest, and this runbook. Serving
repairs are committed directly to `npu-chat-encode-pool` and `npu-bde-combined`.
The driver fetches and pins those branch heads, then tests pristine isolated
worktrees. No runtime patch files or overlay arguments are used.

Before the first run, stop any old server using its original launch/session and
verify that port 30000 is free. The old harness did not record trustworthy
ownership. This runner will not adopt or sweep those processes. Inspect with
`docker exec sglang-a3 ps -eo pid,ppid,pgid,args` if needed.

## A3 commands

Run these in Bash from the extracted directory. Set `GSM8K` to an existing
**host-side** GSM8K test JSONL (`question` and `answer` with `####` final answers).
The confirmed host path is shown below. At least 64 valid rows and five shot examples
are required by default; validation happens before any server launch.

```bash
export GSM8K=/home/tbaydasov/grade-school-math/grade_school_math/data/test.jsonl
COMMON=(
  --container sglang-a3
  --repo /home/swx1199799/megamoe_test/sglang
  --model /home/weights/GLM5.2-w4a8-int
  --container-cann-env /home/tbaydasov/Ascend/cann-9.2.0-beta.2/set_env.sh
  --ais-env /home/tbaydasov/set_env.sh
  --ais-dir /home/tbaydasov/aisbench/aisbench_auto_tools_prefix
  --ais-conda-env tbay_sglang
)

# Baseline GSM8K: reference raw five-shot client, first 64 questions, 16 workers.
python3 -u run_pr_tests.py "${COMMON[@]}" \
  --round accuracy --cases 00-baseline --gsm8k "$GSM8K" \
  --gsm8k-samples 64 --gsm8k-parallel 16 --gsm8k-max-tokens 1024 \
  --hours 3 --case-minutes 150

# Baseline AISBench plus the matching end bracket; no GSM8K required.
python3 -u run_pr_tests.py "${COMMON[@]}" \
  --round speed --cases 00-baseline,08-baseline-repeat \
  --length 1000000 --requests 16 --hours 3 --case-minutes 75

# PR #2 repair: independent accuracy/speed servers and baseline brackets.
python3 -u run_pr_tests.py "${COMMON[@]}" \
  --round all --cases 00-baseline,01-pr-e-chat-encode,08-baseline-repeat \
  --gsm8k "$GSM8K" --hours 8 --case-minutes 150

# Full nine-case sweep using the repaired branch heads.
python3 -u run_pr_tests.py "${COMMON[@]}" \
  --round all --gsm8k "$GSM8K" --hours 24 --case-minutes 150
```

To retain the full sweep inside screen, replace its `python3 -u` prefix with
`screen -L -Logfile "$PWD/prtest-fixed-screen.log" -dmS prtest-fixed python3 -u`.
The driver writes its own logs and exit status, so no `tee` pipeline is needed.
The budgets above are caps, not duration predictions. The case cap includes
both phases. Bounded process cleanup may extend a work deadline to safely stop
owned processes. A smaller `--hours` budget records every remaining case as
skipped and exits nonzero.

Use `--round smoke` for streaming-chat transport checks. `--no-fetch` resolves
local refs only. `--container-cann-env ''` retains the container's configured
runtime; required operators are still probed in that exact environment. Use
`--retry-oom` to opt into one separately recorded memory/graph fallback. These
deviations suppress ratios against a differently configured baseline.

When selecting a CANN setup script, the launcher removes inherited toolkit paths
and root variables before sourcing it. It then retains the selected CANN Python
bindings, conditionally preloads the active conda `libstdc++`, clears inherited
`SGLANG_*` flags and proxy variables, and applies the explicit baseline/case
settings. Runtime probes and servers use the same setup; their environment
records include `LD_PRELOAD`. This matches the reference accuracy launcher's
environment preparation; A3 accuracy must still be rerun.

## Baseline gibberish diagnosis

```bash
python3 -u run_pr_tests.py "${COMMON[@]}" \
  --diagnose-baseline --hours 4 --case-minutes 210
```

This uses the same baseline SHA and seed on two fresh servers: decode graph on
and eager. It saves tokenizer class, chat template, special tokens, rendered
prompts, and token IDs. Each prompt is sent as raw text, rendered chat text,
explicit token IDs, and a chat request. Inspect `diagnosis.json`, `tokenizer.json`,
`smoke.json` (including resolved server configuration), and `server.log`.
Compare rendered-text and token-ID generation with chat output; raw unformatted
text is also retained to expose prompt-format effects. Confirm checkpoint,
resolved quantization, full-model geometry, and loaded weights before assigning
a cause. No baseline model-code change is claimed without this device evidence.

## Accuracy and timing interpretation

GSM8K uses the unmodified client extracted from `test_pr1_megamoe_accuracy.py`:
raw `/generate`, the first 64 dataset questions in file order, the first five
examples from that same dataset, 16 client workers, temperature 0, top-p 1,
1,024 output tokens, and stops `Question`, `Assistant:`, `<|separator|>`.
`--gsm8k-samples`, `--gsm8k-parallel`, `--gsm8k-max-tokens`, and
`--gsm8k-timeout` override client settings. `--gsm8k-seed` is retained for old
commands but does not shuffle the reference protocol's question order.

The legacy scorer takes the last integer, matching the earlier reported score;
it does not preserve signs or decimals. A truncated answer can still score as
correct under that reference rule. Truncation, invalid responses, and request
failures are reported separately. Truncated results remain quality-unqualified.
The summary displays shot/test overlap; the default examples overlap the test
set, so this is a reproduction metric rather than a held-out score. Use
`--gsm8k-shots-data /path/to/train.jsonl` for a separate five-shot dataset and
compare only runs with matching prompt/scorer manifests.

Each attempt retains `gsm8k-reference/manifest.json`, `server_info.json`,
`raw.jsonl`, and `result.json`, plus the driver's normalized `gsm8k.json` and
live `gsm8k-client.log`. Completed responses are flushed by the client and
checkpointed by the driver while it monitors server liveness. Request failures
or incomplete execution make the phase fail. `--gsm8k-min-accuracy` remains an
optional threshold for the exact chosen protocol.

Additional chat-output comparisons are optional and off by default. Enable
`--accuracy-requests 64` to run them separately; they never replace the raw
GSM8K score. This keeps the default accuracy workload at 64 GSM8K requests.
Poor GSM8K quality does not suppress diagnostic speed runs.

AISBench preflight and execution initialize the Conda Bash hook before sourcing
`--ais-env`, because that file may itself call `conda activate`. The hook is
resolved from `CONDA_EXE` or an executable on PATH, with the known
`/home/miniconda3/etc/profile.d/conda.sh` as a fallback. Both phases then activate
the chosen `--ais-conda-env`. The preflight requires a matching model, localhost endpoint,
port, tool, and working directory. The model/tokenizer directory must also be
accessible to the host client. Preflight failure prevents expensive speed-server
launches; an independent accuracy phase can still run.

Speed invokes the unmodified `PROFILE0_SCRIPT` extracted from
`test_pr1_megamoe.py`, with profiling disabled. The driver performs a strict
cache flush once, then lets the wrapper prepare the 99% prefix and measure
16 requests at concurrency 16, 1M input tokens and one output token. The
wrapper's own optional flush is disabled to avoid a duplicate flush.

The driver verifies enabled radix caching, sufficient resolved running-request
capacity, and page/DCP-rounded KV capacity using the reference formula. A missing
or insufficient capacity is reported as capacity-blocked before the expensive
prefix preparation. It monitors the owned server while the wrapper runs and
terminates owned client processes, including detached AISBench children, on
failure, interruption or timeout.

The reference wrapper captures `config.py`, `server_info.json`, tool logs,
`aisbench_result.csv` and experiment directories under `profile0/<run>/`.
`capacity.json`, `cache-flush.json`, `timed-server.log` and `prefix-cache.json`
record the driver's validation evidence. Cached-prefix hits must be visible in
the timed server log; missing evidence leaves performance unverified.

Successful tool exit, preparation/measurement records, expected request counts,
zero failed requests, and valid metrics are required. `99999` failure records
are rejected. TPOT is null for one-token output. JSON preserves TTFT in
milliseconds; the summary displays seconds. Ratios require matching effective
configurations and remain diagnostic when quality is unresolved.

Server launch arguments and feature flags are preserved from the existing
harness. The reference server launchers are not imported: memory fraction stays
0.70, running-request capacity stays 16, and the single-request prefill cap stays
1 for packed DCP reads. The client protocol repair does not claim to reproduce
the PR #1 server configuration or its historical performance.

A smoke response confirms endpoint availability. Empty, nonsensical, or wrong
answers remain unresolved quality. Successful HTTP requests and exact-output
matches do not establish full-model correctness.

## Artifacts, stopping, and offline reports

```
/home/swx1199799/prtest-results/<UTC timestamp>-<id>/
  settings.json, pinned-refs.json, results.json, SUMMARY.md
  runner.log, run-status.json
  <case>/<accuracy|speed|smoke|diagnostic-graph|diagnostic-eager>/attempt-N/
    launch.json, process.json, smoke.json, server.log
    gsm8k.json, gsm8k-client.log, gsm8k-reference/ # accuracy phase
    accuracy.json                                # optional chat diagnostics
    ais-command.json, ais.log, profile0/          # speed phase
    capacity.json, prefix-cache.json, timed-server.log
    tokenizer.json, diagnosis.json              # diagnostic phase
```

Results default to `/home/swx1199799/prtest-results/<run-id>/`. Choose another
parent with `--save-dir /path/to/save`; `--output /exact/run/directory` takes
precedence when provided. This changes saved results and captured logs; ownership
state and temporary container worktrees retain their separate locations.

`results.json` is atomically checkpointed after each phase transition; the reference client flushes each completed response to `raw.jsonl`, and
the driver checkpoints its normalized GSM8K record while monitoring. Optional
paired diagnostics checkpoint each request. A summary-rendering error keeps
those records and makes the run fail. Every case and requested phase has a final
status. Exit 0 means the requested benchmark execution completed; it does not
mean quality acceptance. Failures/skips return 1, argument/prerequisite validation
returns 2, and interruption returns 130.

Interrupt an active runner with Ctrl-C (or SIGTERM) to trigger cleanup. The same
host user can recover a dead runner using its state registry:

```bash
python3 run_pr_tests.py "${COMMON[@]}" --stop-only
```

The runner locks port 30000 across containers and stores ownership per container. `--stop-only` refuses to race an active
runner, refuses PID reuse or missing ownership, checks Docker errors, and verifies
that the port is free. It verifies the recorded session and PID start time; workers in that session
remain owned even if they clear their environment. The unique token also
identifies descendants that start a separate session. A boot identifier guards
new ownership records against reuse after a reboot. Existing records from the
previous harness version remain usable for cleanup. No process-name sweeps are used. Temporary execution worktrees are removed
after verified process cleanup; original worktrees and archived logs are retained.

Regenerate a historical or current report without Docker, NPU, or network access.
The output directory must be new and outside the source evidence directory:

```bash
python3 run_pr_tests.py \
  --regenerate prtest-results/20261112-232436 \
  --output /tmp/prtest-recovered-20261112-232436
```

The archived baseline is recoverable at **36.105/60.187 seconds mean/P90**.
The PR #2 run remains failed because the chat endpoint rejected `Encoding` objects;
its placeholder metrics are not performance evidence. Historical recovery does
not establish baseline numerical quality or fill missing runtime provenance.

Host regression checks, from the development workspace:

```bash
python3 -m pytest -q test_run_pr_tests.py test_chat_encode_regressions.py
```

The tokenizer regressions require `transformers` and `tokenizers` and inspect both
local serving worktrees. The HTTP/process regressions need local loopback and
subprocess access. Host checks do not establish A3 numerical quality, graph
correctness, HCCL stability, or performance; the A3 commands above remain required.
