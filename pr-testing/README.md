# PR-series box tests (PRs #2–#7 on `dcp_ascend_testing`)

Automatic **speed** (AISBench 1M prefix-cache TTFT) and **accuracy** (paired
greedy outputs + optional GSM8K) tests for the six draft PRs, following the
established A3 nightly workflow (`run_a3_nightly.py` lineage):

- one **fresh server per case**, one **measured AISBench invocation per server**
- baseline bracket (first + last case) to measure drift
- host-side orchestration over `docker exec`; live per-case logs
- honest `SUMMARY.md`: raw single samples, no speedup claims

## Cases

| Case | Branch (fork) | Flags on | PR |
|---|---|---|---|
| 00-baseline | `dcp_ascend_testing` (from the fork) | none (control) | — |
| 01-pr-e-chat-encode | `npu-chat-encode-pool` | pool + memo + raw encode | #2 |
| 02-pr-b-indexer-int8 | `npu-dsa-indexer-int8` | `INDEXER_INT8` | #3 |
| 03-pr-prefetch-packed | `npu-dcp-packed-read` | packed read + packed gather prefetch + PG priority | #4 |
| 04-pr-f-sfa-kv-8bit | `npu-sfa-kv-8bit` | `SFA_KV_8BIT` + `SFA_KV_QUANT_READ` | #5 |
| 05-pr-ts-token-sharded | `npu-dsa-cp-token-sharded` | ts + Q_FULL_WIDTH + LOCAL_TOPK + DECODE_Q_PREISSUE + all four streamed weights + SFA gather overlap | #6 |
| 06-pr-moe-int8 | `npu-dsa-cp-moe-int8` | ts (int8 MoE input is unconditional under it) | #7 |
| 07-combined | `npu-bde-combined` | everything (SFA overlap off — superseded by the packed-read prime) | — |
| 08-baseline-repeat | `dcp_ascend_testing` (from the fork) | none (control) | — |

Defaults follow the archived 1M baseline pairing
(`CTX=1024000 CHUNK=16384 MAXREQ=16 CAPTURE=1 MEMFRAC=0.70 EXTRA_ARGS="--watchdog-timeout 7200"
of `p4_exit1_serve_dcp.sh`): TP16 / DCP16, decode-graph capture ON at bs 16,
prefill capture off, `--prefill-max-requests 1`, watchdog 7200 s, HCCL
buffsize 1000/2500. **memfrac 0.70, not 0.76/0.88** — the first live run at
0.88 OOM'd at KV-pool init (12.83 GiB `index_k_buffer` short); the archived
recipe pairs the 1M workload with 0.70. If a case still OOMs at pool init,
the runner retries **once** at memfrac − `--memfrac-step` (0.06, floor 0.55)
and records both values. `--moe-a2a-backend` stays unset (TP-MoE): the
token-sharded PR refuses a2a backends, and a paired A/B needs one config.
Every per-case env is recorded in `results.json`.

## Prerequisites (A3 host, one-time)

1. The box repo can reach the fork. The branches were pushed to
   `https://github.com/TamirBaydasov/sglang` (six PR branches). **Nothing is
   pushed by the harness.**
2. AISBench host client: `--ais-dir` must contain `aisbench_test.py`
   (default `/home/tbaydasov/aisbench/aisbench_auto_tools_prefix`) and
   `--ais-env` is the conda env file (default `/home/tbaydasov/set_env.sh`).
3. The container (`sglang-a3`) is running with host networking.
4. Port 30000 free — the runner refuses to start otherwise.
5. The runner launches servers with `docker exec -d`, `PYTHONPATH=<repo>/python`
   prepended and the pid captured for clean group stops — nothing to set up
   by hand. If a case fails during startup, the full server log lands in
   `<case>/server.log` and its last lines are echoed to the runner log.
6. GitHub reachability: the runner applies the same `ghfast.top` insteadOf
   rewrite your manual box commands use (`--git-proxy`, on by default; fetch
   retries once on failure). Pass `--git-proxy ''` on networks with direct
   GitHub access.

### The `07-combined` branch

`npu-bde-combined` is the clean squash of the verified six-PR series
(d850514275); it is pushed to the fork so case 07 can fetch it. The six
reviewable PRs remain the history of record.

## Run

Copy this directory to the A3 host (or run it from any host that can
`docker exec` into the box container). Then:

```bash
# stop anything already holding the port / a previous run
python3 run_pr_tests.py --stop-only

# full sweep (accuracy + speed per case), ~10 h budget
python3 run_pr_tests.py --round all --hours 10 --case-minutes 60

# speed only (AISBench per case, no accuracy prompts)
python3 run_pr_tests.py --round speed --hours 8 --case-minutes 45

# accuracy only
python3 run_pr_tests.py --round accuracy --hours 6

# smoke: server starts + one greedy generation per case (~fastest sanity loop)
python3 run_pr_tests.py --round smoke
```

Useful variants:

```bash
# subset of cases
python3 run_pr_tests.py --cases 00-baseline,05-pr-ts-token-sharded,07-combined

# attribution A/B without editing the table
python3 run_pr_tests.py --cases 03-pr-prefetch-packed \
  --flags-overrides 03-pr-prefetch-packed:SGLANG_NPU_DCP_GATHER_CHUNKS=1

# GSM8K subset for each case (needs the test jsonl on the host)
python3 run_pr_tests.py --round accuracy --gsm8k /path/to/gsm8k_test.jsonl --gsm8k-samples 200

# point at a different model / container / repo
python3 run_pr_tests.py --model /home/ckpt/GLM-5.2-Provide-0610-W4A8-crop-tiny \
  --container sglang-a3 --repo /home/swx1199799/megamoe_test/sglang

# preview the full command plan without touching the box
python3 run_pr_tests.py --dry-run
```

## What each round does

- **smoke** — checkout case branch (fetched from the fork, `prtest/current`),
  launch the server with the case env, wait for `/health`
  (`--startup-timeout`, default 1800 s), one greedy generation, stop the
  server (SIGTERM to the process group, SIGKILL fallback, survival halts the
  run like the nightly does).
- **accuracy** — `--accuracy-requests` greedy requests per case (default
  **64**): 8 fixed + template-generated short/medium prompts + a long-prompt
  pair (same body, different tails — the second request rides the cached
  prefix; `--accuracy-long-tokens`, default 32 k tokens). SHA-256 of each
  output is compared per tag against case `00-baseline`: exact matches are
  counted raw — NPU serving is not bit-deterministic, so "differ" is
  information, not an automatic failure. With `--gsm8k`, a seeded subset
  (default 200 samples, 5-shot, greedy) gives an accuracy number per case.
- **speed** — one AISBench `prefix_cache` invocation per fresh server
  (`profile_aisbench.sh` with `PROFILE=0`): the prefix cache is flushed
  (`/flush_cache`) first so the warm-up really prefills, then
  `--input_len 1000000 --output_len 1 --data_num 16 --concurrency 16
  --repeat_rate 0.99 --seed 4`. TTFT mean/P90 (ms→s), E2E, qps and prefill
  throughput are parsed from the `'TTFT avg'` metric lines into
  `results.json`. Queueing dominates per-request TTFT at concurrency 16
  (prefill is serialised); read the mean/P90 with that in mind.

## Output

```
prtest-results/<stamp>/
  SUMMARY.md          # speed table + paired-accuracy table (with skip reasons)
  results.json        # full records incl. per-case env + commit
  runner.log
  <case>/server.log   # full server log (captured on startup failure)
  <case>/ais.log  accuracy.json  smoke.json
```

`SUMMARY.md` statuses: `ok` measured · `failed` see case dir (server log tail
is captured) · `skipped` ref missing / fetch failed · `no-data` server ran but
nothing measured. AISBench timings are diagnostic samples — numerical quality
comes from the accuracy section, never from AISBench.
