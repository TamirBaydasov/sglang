# GLM A3 best TP measured recipe

Serving source retained exactly from `4eb600357bf7f8ae98406c0fca6d206ae715408c`. Publication adds reproducibility assets only; no serving code changes.

20261115-222859: TP 22.906s, 60/64 clean. TP has 60/64 clean accuracy. MegaMoE accuracy is incomplete and remains unqualified. Five evaluated questions overlap five-shot examples; these are reproduction scores, not held-out GSM8K.

Recipe: INT8 indexer, packed read/prefetch, local top-k and decode-Q preissue. TP adds token/residual sharding, O-projection streaming and conditional MoE INT8 gather. MegaMoE uses the compatible subset with TP-only sharding/projection paths off, BF16 input, separate shared experts and dispatch capacity 2048. SFA, chat encoding, full-width Q and Q/K/V projection streaming are disabled.

Exact environment and server arguments: `pr-testing/best-pair-recipes.json`. TP16/DCP16; EP1/native TP or EP16/ascend_fuseep mode3; running/prefill 1/1; chunk 32768, max prefill 16384, context 1048576, local KV 63488, memfrac 0.80, eager decode/prefill. AISBench:16 clients/1M tokens/99% shared prefix/one output token. Preserve existing CANN9.2 beta.2/operator/DeepEP runtime.

Four-hour fresh round: TP accuracy75min, TP speed15min, MegaMoE accuracy120min, MegaMoE speed15min, cleanup15min. Global deadline includes preparation; insufficient full phase allowance is explicitly skipped. All four phases use fresh servers. Request timeout1800s; GSM8K64 questions/parallel16/max1024/seed4; acceptance >=58 correct, all64 completed, zero errors/truncations. Speed remains diagnostic until quality passes. No timing reuse.

Run `python3 pr-testing/run_best_pair.py --dry-run` to inspect actual commands. Deployment passes `--tp-ref` and `--ep-ref` with final published SHAs; otherwise measured source parents are used. Results default to `/home/swx1199799/glm-best-pair-results/`.
