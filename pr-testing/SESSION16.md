# GLM A3 follow-up: 16-hour maximum

Use `bash run_16h_session.sh --gsm8k /path/to/test.jsonl`. The host launches
servers in `sglang-a3`; results default to `/home/swx1199799/glm-session-results/`.
The session pins every source ref once and never launches experiments in the
source checkout. It may finish early. A deadline does not imply completion.

## Sources and ports

Source: `OrangeRedeng/sglang:glm52-a3-optimizations`, verified at
`f204b928a3e7b8332611bb9d5ea34906e0cf9440` on November 15, 2026.
The f204 runtime gather/activation/native headroom calculation is adapted into
`dcp_memory.py`; streamed projections also participate, and optional flags absent
from standalone PR branches default off. The KV profiler reserves this memory
before sizing the pool. This may expose an insufficient-capacity configuration
rather than promising that every 1M workload fits.

The source's pending-gather ownership pattern is adapted to packed prefetch:
keep source buffers and Work handles alive; wait at the attention consumer,
while inline gathers still finish before returning. Native HCCL correctness is
pending A3 reproduction. The source's `tc_piecewise` implementation is not
copied into the newer runtime, which uses `breakable`.

PR #5 packed decode now translates request-relative logical positions through
request block tables before reading physical cache rows. It uses speculative
frontiers/tables during verification. The stock TND LSE API and empty-rank rules
are retained. This corrects a reproduced addressing contract violation;
full-model quality must still pass on A3.

`npu-bde-combined-v2` contains current #2–#7 code and these ports.
`npu-bde-combined-ep-v2` also merges #1, preserving TP compensation sharding and
MegaMoE's missing-bias checks. TP-only residual sharding and streamed projection
flags are explicitly disabled for EP/MegaMoE. These are compatible subsets,
not a claim that all #2–#7 feature paths support EP.

## Cases and budgets

| Block | Maximum minutes | Work |
|---|---:|---|
| diagnostics | 120 | four-prompt SFA F1, packed single-chunk check, eager speed control, TP breakable |
| individuals | 240 | TP/EP baseline and PR #1–#7; reuse qualified matching phases |
| historical | 100 | accuracy + speed for matched historical serving controls |
| combined | 320 | full TP, #5-disabled TP, selected compatible EP and MegaMoE |
| confirmation | 120 | two fresh speed repeats of control, best qualified TP, and best qualified EP/MegaMoE |
| reserve | 60 | remaining global allowance; no automatic environment changes |
| total | 960 | 16 hours including pinning, startup and cleanup |

Unused block time carries forward, bounded by the global deadline. Cases outside
the remaining budget are recorded as skipped, not passed. Individual accuracy
has a 60-minute cap and speed a separate 20-minute cap. Request timeout is 1800 s;
all benchmark/startup calls remain bounded by case and global deadlines.

`29-tp-breakable` uses TP16/DCP16, `--cuda-graph-backend-prefill breakable`,
`--cuda-graph-backend-decode disabled`, server admission 1, chunk 32768,
prefill limit 1, context 1048576, KV cap 63488, memfrac 0.80. It is a diagnostic;
NPU/DSA breakable capture support is not established by host checks. Its eager
control is `24-historical-control` with the same serving settings.

The historical 23.775/23.840 s samples were performance-only, at `5ba09eae3e`.
This session uses the recovered serving controls on current fixed sources and
CANN beta.2/TorchNPU post7. It is not a bit-identical replay of the old CANN 9.1
runtime or its full flag set. New results must qualify independently.

AISBench remains 16 clients, 1M input tokens, prefix99, one output token, seed 4,
one measured invocation per fresh server. Historical server admission is 1,
so queued TTFT is explicit and compared only with matching controls. PR #6 uses
an 8192-token prefill diagnostic with prefill limit 1. PR #7 also uses prefill
limit 1 to bound retained context storage; its matching control is case 18.
A changed recipe/source invalidates inherited accuracy evidence.

## Evidence reuse, acceptance, and output

`--reuse-results /path/to/results.json` can be repeated. Reuse requires matching
source SHA, launch argv, recorded model-file hashes, runtime versions/operator
schemas, and environment (apart from transient executable/Python search paths).
Accuracy additionally requires the unchanged client hash, dataset hash,
generation settings and all 64 raw answers, at least 58 correct, no errors and
no truncation. Runtime provenance hashes cover recorded model/config/index
files; they do not hash every checkpoint weight shard. Speed reuse also retains
the original diagnostic/quality status. New source heads require fresh checks.
`--resume /path/to/session` imports evidence into a fresh output directory.
Fresh confirmations never reuse timing measurements.

Five questions overlap shots: these are reproduction scores, not held-out
accuracy. `TARGETS.md` qualifies <=24 s TP and <=20 s EP/MegaMoE only with
passing 64-question accuracy and at least three matching fresh timing samples;
it lists all samples and median mean TTFT. `SUMMARY.md` retains independent
quality/performance failures, and `attempts.json` retains repeated measurements.
No target is guaranteed. Missing operators, graph failures, OOM and bad outputs
remain failures; CANN is not upgraded automatically.

For a targeted breakable run, retain the normal deployment arguments and add:
`bash run_16h_session.sh --only 29-tp-breakable --gsm8k /path/to/test.jsonl`.
For offline command validation, add `--dry-run` to either command.

## Offline deployment

The delivery archive includes a Git bundle of the required serving and harness
commits, excluding the existing base history at `3c94728b7c25`. Verify the bundle
inside the container, then fetch it into a unique `refs/remotes/<session>/`
namespace. Run with `--no-fetch --source-prefix refs/remotes/<session>/` so the
original repository branches are not switched or overwritten. The normal
baseline-ref name is resolved in the same isolated namespace. Bundle verification
fails explicitly if the container repository does not contain the prerequisite.
