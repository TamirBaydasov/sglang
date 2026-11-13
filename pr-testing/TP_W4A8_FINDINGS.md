# W4A8 TP investigation

Evidence: baseline `3c94728b7c25` with TP/EP1 produced 0/64 correct and 64/64
length-limited responses. MegaMoE `8983de99ec4f` with EP16 produced 62/64 correct
and no truncations on the user A3 host. DeepEP EP16 is the supplied P4 serving
recipe; its quality under the repaired harness still requires the new run.
These observations do not identify the failing operator.

## Sources reviewed

- Official vLLM-Ascend, revision `775899991f9614052e4836a62e4cb6541897faee`,
  [W4A8 implementation](https://github.com/vllm-project/vllm-ascend/blob/775899991f9614052e4836a62e4cb6541897faee/vllm_ascend/quantization/methods/w4a8/w4a8.py).
- [CANN GroupedMatmulV4 documentation](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850/API/aolapi/context/ops-transformer/aclnnGroupedMatmulV4.md)
  and [GroupedMatmulV2 documentation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/API/aolapi/context/ops-transformer/aclnnGroupedMatmulV2.md).
  These describe scale/per-token scale, bias, layout and dtype contracts;
  they do not establish compatibility of the installed 9.2 beta operators.
  The English 9.1 page returned navigation only. No local ASC_DEVKIT_DIR was set.
- Supplied P4 launcher and GSM8K client; the existing raw five-shot client uses
  the same prompt construction, stop strings and last-integer scorer.

## Findings and candidate repair boundary

Both ordinary SGLang W4A8 and vLLM-Ascend ModelSlim processing transpose the
weights, encode FP32 scales as bit patterns in INT64 storage, and reduce loaded
compensation bias along its partial-sum axis. Packed checkpoints encode two
INT4 values per INT8 byte and eight per INT32 word. This agreement does not
prove the TP path's sharding or operator selection is correct.

Investigate the down projection first: row-parallel compensation must cover
only the local K shard before TP reduction. Trace the checkpoint's 16 bias
partials through the actual loader, the TP rank slice, post-load bias reduction,
and grouped matmul. Check bias is applied once and that rank-local packing,
scale layout and per-token scale match the installed kernel contract.
A replicated full compensation followed by all-reduce would be wrong, but
this has not been reproduced in these runs. Do not change the bias formula
or copy MegaMoE packing into the ordinary path on that hypothesis alone.

The sweep preserves the graph failure configuration and changes only decode
graph mode for its eager partner. If eager succeeds, isolate graph buffer and
collective lifetime before editing quantization. If both fail and DeepEP works,
compare one expert's dequantized w13/w2 outputs with a BF16 reference using
saved inputs and loaded scales/biases. Any serving fix needs that reproducer,
a focused regression, and a fresh A3 GSM8K run before propagation to PR branches.

The original EP-default harness change applied no TP serving fix; the isolated follow-up below now tests one. DeepEP is the
explicit default; TP failures remain recorded rather than hidden as EP results.


## Isolated TP fix for first A3 accuracy validation

Branch `npu-w4a8-tp-bias-fix` starts directly from baseline `3c94728b7c25`.
The scheme factory instantiates `ModelSlimW4A8Int8MoE` without `tp_size`, leaving
its value at 1. Consequently the old code allocates all 16 down-projection
compensation partials on every rank, labels them per-channel, copies the full
checkpoint bias, and sums all partials before the TP output reduction.

The fix derives compensation width from `layer.moe_tp_size` and labels
compensation as grouped metadata so the existing loader slices the K/partial
axis by TP rank. TP1/EP retains all 16 partials. Per-channel weight scales stay
replicated. The new case `13-tp-bias-fix` explicitly uses TP/EP1, 64 questions,
and accuracy only. It is excluded from default and overnight case lists until
the standalone accuracy test is reviewed.

Related upstream evidence:
- [vLLM-Ascend #7088](https://github.com/vllm-project/vllm-ascend/issues/7088)
  reports the unsliced W4A8 compensation axis.
- [vLLM-Ascend #7090](https://github.com/vllm-project/vllm-ascend/pull/7090)
  makes scale_bias part of grouped metadata even for per-channel weights.
- [vLLM-Ascend #14378](https://github.com/vllm-project/vllm-ascend/issues/14378)
  concerns C8 KV / PD accuracy fluctuations; it does not establish this TP cause.
- Official MindIE issue search found no matching TP-bias report. Its
  [W4A8 mixed precision guide](https://github.com/Ascend/MindIE-LLM/blob/master/docs/zh/user_guide/feature/w4a8_mixed_precision_quantization.md)
  describes routed-expert quantization and model/feature restrictions, not a
  fix for this loader. No MindIE issue is represented as a verified match.

CPU regressions exercise production allocation, rank slicing, and post-load
bias reduction for TP1/2/4/8/16, checking that rank sums reconstruct the original
compensation. This proves the loading defect; the A3 run must still establish
whether it explains the observed 0/64 score.
