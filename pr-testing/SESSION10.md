# Ten-hour combined TTFT round — November 16, 2026

Published on dedicated TTFT10 branches. No CANN upgrade or new A3 measurement is included.

## Matched configuration

All cases: `max-running-requests=1`, `prefill-max-requests=1`, TP16/DCP16, eager prefill/decode, chunk 32768, maximum prefill tokens 16384, context 1048576, memory fraction .80 and local KV capacity 63488. Checkpoint and CANN 9.2 beta.2 are unchanged. EP cases additionally use EP16/MoE TP1.

GSM8K uses the reference raw-completion five-shot protocol: 64 questions, client parallelism 16, output limit 1024, seed 4, temperature 0 and the legacy-last-integer scorer. Five evaluated questions overlap examples; >=58 correct, all 64 completed, no failures/truncation qualifies this reproduction score.

AISBench submits 16 requests concurrently: 1M-token inputs, 99% shared prefix, output length one. Servers admit one request at a time. Cache clearing must succeed. Cold-prefix preparation is recorded separately. Each speed phase uses a fresh server and one measured AISBench invocation, requiring 16 successful requests. Accuracy and speed servers are separate.

## Matrix and budget

| Order | Case | Source / change | Planned minutes |
|---|---|---|---:|
| 1 | 24-historical-control | Fixed TP baseline, aa32b34a4aec | 20 if accuracy reusable |
| 2 | 33-best-ttft | Existing combined TP, SFA off, 4eb600357bf7 | 20 if accuracy reusable |
| 3 | 35-current-f1 | Same best TP + current allocating F1; F2 off | 110 |
| 4 | 36-buffered-f1 | Same best TP + historical buffered F1; F2 off | 110 |
| 5 | 27-combined-ep | Existing best-compatible DeepEP, with EP source adaptation | 110 |
| 6 | 28-combined-megamoe | Compatible MegaMoE mode 3; dispatch capacity 2048 | 110 |
| 7 | Winner + closing baseline | Two fresh winner repeats and closing plain TP | Up to 100 |
| 8 | Cleanup/report reserve | No benchmark started in final 20 minutes | 20 |

Global wall-time limit is ten hours including preparation, imports, cleanup and reporting. Phase ceilings: accuracy 90 minutes, speed 20 minutes, including server startup. Planning allocations do not clip phases; time saved carries forward. If an unchanged control lacks matching qualified accuracy, it receives fresh accuracy and consumes repetition time. Cases/confirmations that cannot execute are explicitly incomplete; return code is nonzero.

Accuracy reuse is restricted to cases 24 and 33 and requires matching source diff, commit, runtime versions, model hashes, operator schemas, environment, command and protocol. All timing is fresh. F1/EP/MegaMoE quality is rerun. Candidates remain timed diagnostically when quality fails, but only complete qualified candidates enter winner selection. The existing best TP is the fallback when no new case qualifies.

The winner target is three fresh mean/P90 samples plus opening/closing baseline drift. Fewer samples remain preliminary. `confirmation-status.json` records missing confirmations. New configurations have no promised TTFT improvement.

## Historical F1 and EP adaptation

`SGLANG_NPU_SFA_F1_PREALLOC=True` selects the port from `glm52-a3-optimizations` 5ba09eae3eeda8420285905f4de4fcad0c8908fd. It allocates two raw/index/FP32/BF16 output slots at pool initialization, applies FP32 scales in place, and includes the output buffers in pre-KV-sizing reservation. The current physical-row addressing, contiguous BF16 output, operator API and gather completion fixes remain.

The buffered path supports one prefill request. Excess scratch demand, different output dtype or a different owning stream uses the allocating path. Slot parity preserves the two-layer prefetch pipeline; existing collective completion waits protect send lifetimes. Startup and first use log prepared capacity and activation. These CPU-tested contracts still require A3 stream/lifetime verification.

The combined EP source already contains reusable gather outputs/scratch and a 1 GiB native-memory allowance; those implementations are retained. The missing F1 preparation is ported there too, available behind the same selector. The timed EP cases keep SFA off and disable TP-only residual/projection sharding, matching the previously fastest compatible subset.

MegaMoE uses BF16 inputs, separate shared experts and capacity 2048 for the recorded padded 2048-row input. Its existing guard rejects larger inputs before buffer creation. The installed DeepEP low-latency communication-size hint is added to the existing 1 GiB native-workspace allowance during KV budgeting. The allowance is conservative, not a measured workspace bound; original runtime failures and capacity evidence remain authoritative. Insufficient capacity fails explicitly without reducing context/chunk or changing memory fraction.

## Roadmap review

Reviewed [MoE roadmap #50795](https://github.com/sgl-project/sglang/issues/50795) and [native TP optimizations #44525](https://github.com/sgl-project/sglang/pull/44525), head b9bb06c92509734c9d8fc6d032f7f752df9e092b. Relevant ideas are input-quantization reuse, compatible fusion, shared-expert overlap and communication-buffer sizing. This round implements the bounded buffer/memory work. New quantization, shared fusion, offloading, EPLB and breakable graphs are excluded. The roadmap explicitly places CP/DCP outside the referenced breakable graph scope; A5 results cannot establish A3 performance.

## Git deployment

The serving changes are committed on dedicated branches:

- `npu-bde-combined-ttft10-f1`: `39375d04d004db540f241603515b28a85a0b3a84` (buffered F1).
- `npu-bde-combined-ep-ttft10`: `8e219cba5f16ef00c1cc67beab3a1ef309315fc5` (F1 port and MegaMoE memory reservation).
- `pr-test-harness-ttft10`: this launcher and matrix, based on ba9c4f5bfb5b473e54fe8d43008d7b0c233cc967.

Unchanged controls retain their original SHAs so qualified accuracy can be reused. The current F1 control and SFA-off best TP remain at 4eb600357bf7. The new F1 case uses 39375d04d004; both EP cases use 8e219cba5f16. Source commits are pinned, worktrees are detached and clean, and results record the effective source diff hash. No source-file overlays are applied.

On the A3 host, fetch the harness branch into the existing container repository, archive the published harness SHA into a fresh host directory, check CHECKSUMS.sha256, and invoke `launch_ttft_10h.sh`. The launch command supplied with publication pins the exact harness SHA. The launcher fetches the dedicated serving branches and baseline before starting screen, and verifies all required commit objects. The default fetch URL is `https://ghfast.top/https://github.com/TamirBaydasov/sglang.git`; override it with `--fetch-url` if needed.

Pass `--gsm8k /home/tbaydasov/grade-school-math/grade_school_math/data/test.jsonl` and existing matching `--reuse-results` paths. Default container/repo: sglang-a3 and /home/swx1199799/megamoe_test/sglang. Save directory defaults to /home/swx1199799. Missing reuse files must be omitted. Append --dry-run to inspect actual commands without Docker or NPU.

Results: `/home/swx1199799/glm-ttft10-results/<UTC timestamp>-<id>/`. Screen attachment and console path are printed. The older offline archive remains historical; use the published Git launcher for this revision.

## Acceptance and artifacts

Inspect `SUMMARY.md`, `SAMPLES.md`, `confirmation-status.json`, `results.json`, `attempts.json`, and raw phase directories. Record cold prefix, all measured mean/P90 samples, quality and truncation, baseline drift, actual feature activation, memory reservation, runtime versions and source identity. A request error, missing metric, failed cache clear, unexecuted accuracy or insufficient KV capacity cannot pass. Host checks validate harness and CPU arithmetic only; A3 performance/correctness remain pending.
