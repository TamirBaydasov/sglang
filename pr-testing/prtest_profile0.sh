#!/usr/bin/env bash
# Profile one aisbench_auto_tools_prefix run against a live sglang server, and keep
# everything about the run in one directory.
#
#   bash profile_aisbench_profile0.sh              # the 1M prefix-cache run without profiling
#   PROFILE=1 bash profile_aisbench_profile0.sh    # explicit profiling opt-in
#   bash profile_aisbench.sh -- python3 aisbench_test.py --input_len 131072 ...
#
# THE SERVER, launched first in another shell:
#
#   LOG=serve_1m_dcp16.log CTX=1024000 CHUNK=16384 MAXREQ=16 CAPTURE=1 \
#   MEMFRAC=0.70 EXTRA_ARGS="--watchdog-timeout 7200" bash p4_exit1_serve_dcp.sh 16
#
# Indexer query sharding and DSA-CP default to 1 in the code (a0eb86a958); QSHARD is
# gone. That script turns the per-rank DCP extend
# memory log on (DCPMEM=1) by default; neither needs typing on the launch line.
#
# MEMFRAC=0.70, NOT the 0.76 the p10 tail runs use. This run OOM'd at 0.76 on
# 2026-09-17 (50.05 GiB allocated of a die's 61.27 GiB usable, ~9.6 GiB of which the
# torch allocator never sees -- HCCL, DeepEP, the graph pool). The budget:
#   static (weights + KV pool)         45.69 GiB, measured, steady between forwards
#   transient per prefill forward      ~2.3 GiB, of which ~1.05 GiB is the DCP gather
#   ceiling on allocated               ~50.3 GiB
# so 0.76 leaves room for about two forwards' transients and the scheduler pipelines
# with event_loop_overlap. 0.70 frees ~3.7 GiB of KV pool for roughly 3.5 forwards,
# and the pool still holds ~1.47M tokens against the ~1.16M this workload needs
# (989k shared trunk + 16 tails). Do NOT cap concurrency to buy the same headroom:
# the point of the run is 16 requests in flight.
#
# max_running_requests must be >= the benchmark's --concurrency, or the extra
# requests queue in the scheduler and their TTFT measures the queue, not the server.
#
# then:  SERVER_LOG=serve_1m_dcp16.log bash profile_aisbench.sh
#
# --watchdog-timeout: the scheduler kills the server when one forward batch runs
# longer than 300 s. An unprofiled 1M prefill chunk is well under that, but this run
# records every operator with its stack and shapes, and losing a 40-minute trace to
# the watchdog is not worth the risk.
#
# WHAT THE BENCHMARK DOES
# -----------------------
# aisbench_test.py with --dataset_type prefix_cache --prefix_test runs ais_bench twice:
#   1. warm-up: ONE request that is only the shared prefix, int(1000000 * 0.99) =
#      990,000 tokens, output 1 token. This is the 1M-class prefill.
#   2. measured: data_num=16 requests, each that same prefix + 3 random tokens +
#      9,997 unique tokens, concurrency 16, output 1 token, ignore_eos, streamed chat.
#      These should hit the cached prefix and prefill only ~10k tokens each.
# Before either it generates the dataset by tokenizing ~1M tokens on the CPU, which
# takes a while; the profiler records an idle server during that part.
#
# The counts above follow the default BENCH below; pass `-- python3 aisbench_test.py ...`
# to change them and this comment stops describing the run.
#
# READING THE RESULT (measured on this box 2026-09-16, dcp16, chunk 16384, 0.76)
#   - Page rounding: the radix match is cut to whole 2048-token allocator pages, so of
#     990,000 shared tokens only 483*2048 = 989,184 are served from cache and each
#     request computes ~816 of the prefix again plus its ~10,000 unique -> ~10.8k.
#   - Cost per request: ~4.2 s of linear work (0.387 ms/token) + ~3.5 s of per-chunk
#     context cost, so ~7.7 s each, about 125 s for all 16.
#   - Prefill is serialised on the device, so with concurrency 16 and request_rate 0
#     TTFT is mostly QUEUEING: request i finishes near i*7.7 s and the mean TTFT lands
#     near 65 s. Read total wall time and throughput; read a per-request TTFT only
#     against its position in the queue.
#   - Capacity: one shared trunk (stored once) + 16 tails ~= 1.15M of the 1.83M
#     servable at dcp16. If the trunk were stored per request it would need ~16M and
#     the run would fail -- so this run is also the test of that claim.
#
# Its own "prefix cache hit" table reads vLLM's /metrics names, which sglang does
# not publish, so it will report no metrics. This script prints the equivalent from
# sglang's "Prefill batch ... #cached-token" log lines instead (needs SERVER_LOG).
#
# The warm-up request's first token arrives after the whole 990k prefill. That was
# 33 minutes when this was written; since indexer query sharding landed
# (SGLANG_NPU_ENABLE_DSA_INDEXER_QUERY_SHARDING=1) a 960k prefill measures 456 s, so
# expect about 8 minutes. If it takes 30+ again, indexer query sharding did not engage
# (SGLANG_NPU_ENABLE_DSA_INDEXER_QUERY_SHARDING=0 in the environment, or it refused)
# or predates that default -- grep its log for the engagement line. The tool
# configures ais_bench with retry=2, so if ais_bench's
# HTTP client gives up first it re-sends the request, and the run and its trace no
# longer mean what they appear to. Check its timeout before trusting a run.
#
# aisbench_test.py must run from its own directory (it opens default_api.py and
# GSM8K.jsonl by relative path), and its config.py must point at this server:
#   HOST_IP     127.0.0.1                         HOST_PORT  "30000" (PORT below)
#   MODEL_PATH  /home/tbaydasov/GLM5.2-w4a8-int   (tokenizer for dataset generation)
#   WORK_PATH   the ais_bench benchmark checkout  DATASET_PATH  any writable directory
# MODEL_NAME can stay "ds": sglang reads the request's model field only for a
# "base:adapter" LoRA suffix.
#
# WHERE THE RESULTS GO
# --------------------
#   profiles/<date>-<time>_<LABEL>/
#     meta.txt           command, git rev, profiler settings, timings, exit codes
#     server_info.json   /server_info as it was when the run started
#     config.py          the benchmark tool's config for this run
#     bench.log          everything aisbench_test.py printed
#     aisbench.log, aisbench_result.csv, aisbench_outputs/   the tool's own results
#     server_log.txt     the server-log lines written during the run (needs SERVER_LOG)
#     trace/             the profiler output, one torch_npu directory per rank
#
# trace/ is written by the server processes, so the server and this script must share
# a filesystem: run both on the same box.
#
# PROFILING
# ---------
# The whole run, with stacks and record_shapes, by default. Trace size follows the
# number of operators, not wall time: a 1M prefill is ~62 long chunks and the decode
# is ~1024 graph replays, so the event count is moderate even though the run is
# long. /stop_profile writes the traces before it returns, which can take a while
# (STOP_TIMEOUT). WITH_STACK=0 RECORD_SHAPES=0 make it lighter, and
# PROFILE_DELAY=N / PROFILE_STEPS=K profile a window instead of the whole run.
#
# A profiled run is not a benchmark: quote TTFT and throughput from PROFILE=0.
#
# The prefix cache is flushed first (FLUSH=0 to skip), so the warm-up really prefills.

set -uo pipefail

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-30000}
URL="http://${HOST}:${PORT}"
LABEL=${LABEL:-aisbench_1m}
BENCH_DIR=${BENCH_DIR:-/home/tbaydasov/aisbench/aisbench_auto_tools_prefix}
PROFILE=${PROFILE:-0}
PROFILE_DELAY=${PROFILE_DELAY:-0}
# PROFILE_PHASE=2 starts the profiler when phase 2 (the 16 cached tails) begins, and
# not before: no 990k warm-up in the trace. Triggered by the benchmark's own log line,
# not a timer, because the gap between the phases is only ~25 s (2026-09-22: phase 1
# done 00:37:49, phase 2 requests sent 00:38:14) and the warm-up's length varies.
PROFILE_PHASE=${PROFILE_PHASE:-}
# aisbench_test.py prints this ("full dataset test") the moment phase 1 has finished,
# ~21 s before phase 2 sends its first request.
PHASE2_MARKER="全量数据集测试"
PROFILE_STEPS=${PROFILE_STEPS:-}
WITH_STACK=${WITH_STACK:-1}
RECORD_SHAPES=${RECORD_SHAPES:-1}
FLUSH=${FLUSH:-1}
SERVER_LOG=${SERVER_LOG:-}
SGLANG_DIR=${SGLANG_DIR:-/home/tbaydasov/sglang}
STOP_TIMEOUT=${STOP_TIMEOUT:-14400}
OUT_ROOT=${OUT_ROOT:-$(cd "$(dirname "$0")" && pwd)/profiles}

if [ $# -gt 0 ]; then
  if [ "$1" != "--" ] || [ $# -lt 2 ]; then
    echo "usage: bash $0 [-- <benchmark command ...>]" >&2
    exit 2
  fi
  shift
  BENCH=("$@")
else
  BENCH=(python3 aisbench_test.py --input_len 1000000 --output_len 1 --data_num 16
         --concurrency 16 --request_rate 0 --dataset_type prefix_cache --repeat_rate 0.99
         --prefix_test)
fi

# ---- preflight: everything that would waste a 40-minute run --------------------
if [ ! -f "${BENCH_DIR}/aisbench_test.py" ]; then
  echo "aisbench_test.py not found in BENCH_DIR=${BENCH_DIR}" >&2
  exit 2
fi
if ! command -v ais_bench >/dev/null 2>&1; then
  echo "ais_bench is not on PATH -- aisbench_test.py shells out to it." >&2
  echo "Check: pip show ais-bench-benchmark" >&2
  exit 2
fi
if ! curl -sf "${URL}/v1/models" >/dev/null; then
  echo "no server answering on ${URL}" >&2
  exit 2
fi

read -r CFG_IP CFG_PORT CFG_MODEL CFG_WORK <<EOF
$(cd "${BENCH_DIR}" && python3 -c 'import runpy
c = runpy.run_path("config.py")
print(c.get("HOST_IP", "?"), c.get("HOST_PORT", "?"), c.get("MODEL_PATH", "?"), c.get("WORK_PATH", "?"))' 2>/dev/null)
EOF
if [ "${CFG_PORT}" != "${PORT}" ]; then
  echo "config.py sends requests to ${CFG_IP}:${CFG_PORT}, but the server being profiled is ${URL}." >&2
  echo "Set HOST_IP/HOST_PORT in ${BENCH_DIR}/config.py, or PORT/HOST here." >&2
  exit 2
fi
case "${CFG_IP}" in
  127.0.0.1|localhost|"${HOST}") ;;
  *) echo "ERROR: config.py HOST_IP=${CFG_IP} does not match ${HOST}." >&2; exit 2 ;;
esac
[ -d "${CFG_MODEL}" ] || echo "WARNING: config.py MODEL_PATH=${CFG_MODEL} does not exist (tokenizer)" >&2
[ -d "${CFG_WORK}/ais_bench" ] || echo "WARNING: config.py WORK_PATH=${CFG_WORK} has no ais_bench/ directory" >&2

RUN="${OUT_ROOT}/$(date +%Y%m%d-%H%M%S)_${LABEL}"
mkdir -p "${RUN}" || exit 2
if [ "${PROFILE}" = "1" ]; then mkdir -p "${RUN}/trace" || exit 2; fi
META="${RUN}/meta.txt"
cp "${BENCH_DIR}/config.py" "${RUN}/config.py"

curl -sf "${URL}/server_info" > "${RUN}/server_info.json" \
  || echo "WARNING: could not read /server_info" >&2

read -r SDCP SGRAPH SCTX SCHUNK SMAX SRADIX <<EOF
$(python3 -c 'import json,sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    d = {}
d = d.get("server_args", d)
g = d.get("cuda_graph_config")
graph = g.get("decode", {}).get("backend", "?") if isinstance(g, dict) else "?"
print(d.get("dcp_size", "?"), graph, d.get("context_length", "?"),
      d.get("chunked_prefill_size", "?"), d.get("max_running_requests", "?"),
      d.get("disable_radix_cache", "?"))' "${RUN}/server_info.json" 2>/dev/null)
EOF

echo "server  : ${URL}  dcp_size=${SDCP}  decode_graph=${SGRAPH}  context_length=${SCTX}"
echo "          chunked_prefill_size=${SCHUNK}  max_running_requests=${SMAX}  disable_radix_cache=${SRADIX}"
BENCH_CC=16
for ((i=0; i<${#BENCH[@]}; i++)); do
  if [ "${BENCH[$i]}" = "--concurrency" ]; then BENCH_CC="${BENCH[$((i+1))]}"; fi
done
if [[ "${SMAX}" =~ ^[0-9]+$ ]] && [[ "${BENCH_CC}" =~ ^[0-9]+$ ]] && (( SMAX < BENCH_CC )); then
  echo "ERROR: max_running_requests=${SMAX} is below client concurrency=${BENCH_CC}" >&2
  exit 2
fi
if [ "${SRADIX}" = "True" ]; then
  echo "ERROR: radix cache is disabled for this prefix-cache benchmark." >&2
  exit 2
fi
echo "bench   : (cd ${BENCH_DIR} && $(printf '%q ' "${BENCH[@]}"))"
if [ "${PROFILE}" = "1" ]; then
  echo "profile : delay=${PROFILE_DELAY}s  phase=${PROFILE_PHASE:-all}  steps=${PROFILE_STEPS:-whole run}  with_stack=${WITH_STACK}  record_shapes=${RECORD_SHAPES}"
else
  echo "profile : OFF"
fi
echo "results : ${RUN}"
df -h "${RUN}" | tail -n 1 | awk '{print "disk    : " $4 " free on " $6}'
echo

{
  echo "label          ${LABEL}"
  echo "started        $(date -Is)"
  echo "server         ${URL}"
  echo "sglang rev     ${SGLANG_REV:-$(git -C "${SGLANG_DIR}" rev-parse --short HEAD 2>/dev/null || echo unknown)}"
  echo "bench dir      ${BENCH_DIR}"
  echo "bench          $(printf '%q ' "${BENCH[@]}")"
  echo "profile        ${PROFILE}  delay=${PROFILE_DELAY}  steps=${PROFILE_STEPS:-whole}  with_stack=${WITH_STACK}  record_shapes=${RECORD_SHAPES}"
  echo "flush_cache    ${FLUSH}"
} > "${META}"

SERVER_LOG_START=0
if [ -n "${SERVER_LOG}" ]; then
  if [ -f "${SERVER_LOG}" ]; then
    SERVER_LOG_START=$(wc -l < "${SERVER_LOG}")
  else
    echo "WARNING: SERVER_LOG=${SERVER_LOG} does not exist; server_log.txt will be skipped" >&2
    SERVER_LOG=""
  fi
fi

if [ "${FLUSH}" = "1" ]; then
  if curl -sf -X POST "${URL}/flush_cache?timeout=60" -m 120 >/dev/null; then
    echo "flushed the prefix cache"
  else
    echo "WARNING: /flush_cache failed; the warm-up may hit a previous run's prefix" >&2
    echo "flush_cache    FAILED" >> "${META}"
    exit 1
  fi
fi

PROFILING=0
BENCH_PID=""

profile_start() {
  local body code
  body=$(OUT="${RUN}/trace" STEPS="${PROFILE_STEPS}" STACK="${WITH_STACK}" \
         SHAPES="${RECORD_SHAPES}" LABEL="${LABEL}" python3 -c 'import json, os
req = {
    "output_dir": os.environ["OUT"],
    "activities": ["CPU", "GPU"],
    "with_stack": os.environ["STACK"] == "1",
    "record_shapes": os.environ["SHAPES"] == "1",
    "profile_id": os.environ["LABEL"],
}
if os.environ["STEPS"]:
    req["num_steps"] = int(os.environ["STEPS"])
print(json.dumps(req))')
  code=$(curl -sS -m 600 -o "${RUN}/start_profile.resp" -w '%{http_code}' \
         -H 'Content-Type: application/json' -d "${body}" "${URL}/start_profile")
  if [ "${code}" != "200" ]; then
    echo "ERROR: /start_profile returned HTTP ${code}:" >&2
    cat "${RUN}/start_profile.resp" >&2
    echo "If it says profiling is already in progress: curl -X POST ${URL}/stop_profile" >&2
    echo "profile start  FAILED http=${code}" >> "${META}"
    return 1
  fi
  PROFILING=1
  echo "profiler started ($(date +%H:%M:%S))"
  echo "profile start  $(date -Is)" >> "${META}"
}

profile_stop() {
  [ "${PROFILING}" = "1" ] || return 0
  PROFILING=0
  echo "stopping the profiler -- every rank writes its trace before this returns ($(date +%H:%M:%S))..."
  local code
  code=$(curl -sS -m "${STOP_TIMEOUT}" -o "${RUN}/stop_profile.resp" -w '%{http_code}' \
         -X POST "${URL}/stop_profile")
  if [ "${code}" = "200" ]; then
    echo "profile stop   $(date -Is)" >> "${META}"
  elif [ -n "${PROFILE_STEPS}" ] && grep -q "not in progress" "${RUN}/stop_profile.resp" 2>/dev/null; then
    # PROFILE_STEPS makes the server stop by itself after that many steps.
    echo "profiler had already stopped after ${PROFILE_STEPS} steps"
    echo "profile stop   automatic after ${PROFILE_STEPS} steps" >> "${META}"
  else
    echo "WARNING: /stop_profile returned HTTP ${code}:" >&2
    cat "${RUN}/stop_profile.resp" >&2
    echo "profile stop   FAILED http=${code}" >> "${META}"
  fi
}

collect_tool_outputs() {
  cp "${BENCH_DIR}/aisbench.log" "${RUN}/" 2>/dev/null
  cp "${BENCH_DIR}/aisbench_result.csv" "${RUN}/" 2>/dev/null
  # One ais_bench experiment folder per phase; paths are relative to BENCH_DIR.
  local dir
  grep -ho "Current exp folder:.*" "${BENCH_DIR}/aisbench.log" 2>/dev/null \
    | sed 's/^Current exp folder:[[:space:]]*//' | sort -u \
    | while IFS= read -r dir; do
        [ -n "${dir}" ] || continue
        case "${dir}" in /*) ;; *) dir="${BENCH_DIR}/${dir}" ;; esac
        if [ -d "${dir}" ]; then
          mkdir -p "${RUN}/aisbench_outputs"
          cp -r "${dir}" "${RUN}/aisbench_outputs/"
        fi
      done
}

summarize_prefix_hits() {
  [ -f "${RUN}/server_log.txt" ] || return 0
  python3 - "${RUN}/server_log.txt" <<'PY'
import re, sys
pat = re.compile(r"Prefill batch.*?#new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+)")
lines = [l for l in open(sys.argv[1], errors="replace") if "Prefill batch" in l]
tp0 = [l for l in lines if " TP0" in l]
rows = [tuple(map(int, m.groups())) for m in map(pat.search, tp0 or lines) if m]
if not rows:
    print("prefix cache : no 'Prefill batch' lines in server_log.txt")
    sys.exit(0)
new = sum(r[1] for r in rows)
cached = sum(r[2] for r in rows)
print(f"prefix cache : {len(rows)} prefill batches, {new:,} tokens computed, {cached:,} served "
      f"from cache ({cached / max(new + cached, 1):.1%} of prefill tokens)")
print("               per-batch detail: grep 'Prefill batch' server_log.txt")
# Batches by how many requests they carry. Phase 1 is one request, so the multi-request
# rows are phase 2 -- the batches the DSA-CP lift exists for (84.7% of phase 2's tokens
# declined DSA-CP on 2026-09-18, before the lift).
by = {}
for seqs, computed, _ in rows:
    b = by.setdefault(seqs, [0, 0])
    b[0] += 1
    b[1] += computed
for seqs in sorted(by):
    print(f"               #new-seq {seqs}: {by[seqs][0]} batches, {by[seqs][1]:,} tokens")
# Both lines are print_info_once, so after a fresh server start they appear here only if
# they fired during this run: one line per rank.
text = open(sys.argv[1], errors="replace").read()
refused = text.count("DSA-CP is off for multi-request")
print(f"DSA-CP       : engagement line {'present' if 'DSA-CP is ON' in text else 'ABSENT'};"
      f" multi-request refusals {refused} (0 = every multi-request batch kept DSA-CP)")
PY
}

finish() {
  local rc=$?
  if [ -n "${BENCH_PID}" ] && kill -0 "${BENCH_PID}" 2>/dev/null; then
    echo "stopping the benchmark (process group ${BENCH_PID})"
    # aisbench_test.py shells out to ais_bench; stop the whole group, not just python.
    kill -TERM -- "-${BENCH_PID}" 2>/dev/null || kill -TERM "${BENCH_PID}" 2>/dev/null
    wait "${BENCH_PID}" 2>/dev/null
  fi
  # Never leave the server profiling: it would keep recording into trace/.
  profile_stop
  collect_tool_outputs
  if [ -n "${SERVER_LOG}" ]; then
    tail -n +"$((SERVER_LOG_START + 1))" "${SERVER_LOG}" > "${RUN}/server_log.txt"
  fi
  echo "finished       $(date -Is)  exit=${rc}" >> "${META}"
  echo
  summarize_prefix_hits | tee -a "${META}"
  echo "results in ${RUN}"
  ls -la "${RUN}"
  if [ -n "$(ls -A "${RUN}/trace" 2>/dev/null)" ]; then
    echo "trace/: $(du -sh "${RUN}/trace" | cut -f1)"
    ls "${RUN}/trace"
  fi
}
trap finish EXIT
trap 'echo; echo "interrupted"; exit 130' INT TERM

if [ -n "${PROFILE_PHASE}" ] && [ "${PROFILE_PHASE}" != "2" ]; then
  echo "PROFILE_PHASE must be 2 or unset (got '${PROFILE_PHASE}')" >&2
  exit 2
fi
if [ "${PROFILE}" = "1" ] && [ "${PROFILE_DELAY}" = "0" ] && [ -z "${PROFILE_PHASE}" ]; then
  profile_start || exit 1
fi

echo "bench start    $(date -Is)" >> "${META}"
# setsid: its own process group, so an interrupt can stop ais_bench as well.
setsid bash -c 'cd "$1" && shift && exec "$@"' _ "${BENCH_DIR}" "${BENCH[@]}" \
  > >(tee "${RUN}/bench.log") 2>&1 &
BENCH_PID=$!

if [ "${PROFILE}" = "1" ] && [ "${PROFILE_PHASE}" = "2" ]; then
  echo "profiler armed: starts when phase 2 begins ($(date +%H:%M:%S))"
  while kill -0 "${BENCH_PID}" 2>/dev/null && ! grep -qF "${PHASE2_MARKER}" "${RUN}/bench.log" 2>/dev/null; do
    sleep 1
  done
  if kill -0 "${BENCH_PID}" 2>/dev/null; then
    profile_start
  else
    echo "WARNING: the benchmark ended without reaching phase 2; nothing was profiled" >&2
  fi
elif [ "${PROFILE}" = "1" ] && [ "${PROFILE_DELAY}" != "0" ]; then
  waited=0
  while [ "${waited}" -lt "${PROFILE_DELAY}" ] && kill -0 "${BENCH_PID}" 2>/dev/null; do
    sleep 5
    waited=$((waited + 5))
  done
  if kill -0 "${BENCH_PID}" 2>/dev/null; then
    profile_start
  else
    echo "WARNING: the benchmark finished before PROFILE_DELAY=${PROFILE_DELAY}s; nothing was profiled" >&2
  fi
fi

wait "${BENCH_PID}"
BENCH_RC=$?
BENCH_PID=""
# Let tee flush the last lines before the summary prints.
sleep 1
echo "bench end      $(date -Is)  exit=${BENCH_RC}" >> "${META}"
exit "${BENCH_RC}"

