# Two-hour isolated INT8 indexer / SFA comparison

Only these three TP16/DCP16/EP1 cases run, each on a fresh server:

| Case | Enabled features |
|---|---|
| 40-indexer-int8 | W4A8 TP compensation fix + INT8 QuantLightningIndexer; SFA off |
| 41-indexer-f1-buffered | Same + packed INT8 SFA KV, BF16 attention, historical prepared F1 buffers |
| 42-indexer-f2-original | Same + original PR F2 quantized-read attention; prepared F1 buffers off |

Common source: `39375d04d004db540f241603515b28a85a0b3a84`, plus checksum-pinned `sfa2h-serving.diff`. The overlay adds F2 activation/fallback markers and accounts for prepared F1 buffers when optional packed/token-sharded paths are off. F2 operator arguments are unchanged from original PR source `c089fc27e525c2ec3b7d7f36a20ba87147cd8693`. Source diff and overlay hashes are recorded per case. All other combined optimization flags are explicitly disabled; upstream baseline DSA flags remain as in the established harness. No online source fetch occurs.

All cases: running limit 1, prefill limit 1, chunk 32,768, max prefill tokens 16,384, context 1,048,576, memfrac 0.80, local KV cap 63,488, eager prefill/decode. Checkpoint and CANN beta.2 retain harness defaults. AISBench uses 16 clients, 1M tokens, 99% shared prefix and one output token. Cache clearing must succeed. No OOM fallback or timing reuse.

Each case runs one unprofiled AISBench invocation, then a separate PROFILE=1 invocation. The second invocation prepares a fresh cold prefix and captures three scheduler steps during cached-tail processing using the server's multi-rank TorchNPU profiler. Stack and shape recording are disabled. Profiling uses CPU/NPU activities, Level1 and PipeUtilization, as configured by the serving backend. It does not use `msprof op`. Profiled timing is diagnostic and never replaces the primary sample. Exported traces go to `speed/attempt-1/profiling/trace`; `kernel-summary.json` lists top kernels per export. Three steps may cover only part of the 16 requests. Per-export summed device duration is not wall time or cross-rank latency.

F1 must show prepared-buffer activation. F2 must show successful quant-attention execution without fallback in the unprofiled run, and its short profile must include KvQuantSparseFlashAttention. Every profile must contain QuantLightningIndexer. Missing activation, fallback, failed trace export or profiling timeout is explicit failure; completed unprofiled measurements remain in `unprofiled-speed.json`. Unsupported F2 is not relabelled as a successful F2 result.

Budget: at most 38 minutes per case, speed phase ceiling 36 minutes, each AISBench invocation ceiling 15 minutes, global work deadline 117 minutes after launcher startup. Three minutes are reserved for cleanup. Preparation consumes the same monotonic budget. The runner records unexecuted cases. Cleanup uses existing owned-process groups. The remote trace directory is under the run-owned root and remains there if further device-side inspection is needed; its path is recorded.

No GSM8K is requested. Quality remains unverified for this isolated matrix; no accuracy-qualified winner claim is possible. TTFT samples are preliminary single runs. Cold-prefix preparation remains separate in AISBench records. Existing evidence is preserved. No commits or pushes are part of this delivery.

Run `bash run_sfa_2h.sh --dry-run` for actual server command builders. Offline deployment additionally verifies `SHA256SUMS` and imports the included Git bundle. The destination repository must already have bundle base `3c94728b7c254fc56fcb29b012a3ac7698c2c163`.
