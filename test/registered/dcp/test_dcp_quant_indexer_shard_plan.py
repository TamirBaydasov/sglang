"""CPU unit test for the quant indexer's plan under a DSA query shard.

``quant_lightning_indexer`` (v2, PR #40591) does not take per-call lengths the
way v1 did. It takes a *pre-planned* task list -- ``cu_seqlens_q``,
``seqused_k`` and an opaque ``metadata`` blob built by
``quant_lightning_indexer_metadata`` -- and ``AscendAttnBackend`` plans it once
per forward from the whole batch, so all 78 layers can share one plan.

``plan_indexer_query_shard`` (``SGLANG_NPU_ENABLE_DSA_INDEXER_QUERY_SHARDING``,
on by default) hands each attention-TP rank ``1/attn_tp_size`` of the prefill
query rows.

**The regression this file exists for.** The two features are correct alone and
wrong together. The quant branch read the batch-wide plan while passing the
shard's query tensor, so the operator was told to walk ``sum(extend_lens)``
rows through a tensor holding ``ceil(total / tp_size)`` of them. Nothing on the
host catches it: the lengths are device tensors, so the operator's tiling
function cannot check them against the query shape, and the kernel surfaces the
out-of-bounds read as an AICore trap ("timeout or trap error, subErrType 0x4")
from inside the operator -- with no Python frame naming either feature.

Neither feature's own CI can reach this. The query shard needs no quant pool
and the quant indexer needs no shard, so both test suites pass while main
carries an out-of-bounds device read. That is what makes a CPU test worth
having: the plan is integer arithmetic, and the invariant it must hold is one
line -- **``cu_seqlens_q[-1]`` is how many rows of ``query`` the kernel reads,
so it must never exceed the rows the caller actually passed.**

``test_the_batch_wide_plan_would_read_past_the_query_tensor`` is the control:
it builds the plan the pre-fix code built and asserts it violates exactly that.

Usage:
    python -m pytest test_dcp_quant_indexer_shard_plan.py -v
    python test_dcp_quant_indexer_shard_plan.py
"""

import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsa.dsa_npu_indexer import (
    DSANPUIndexerMixin,
    _IndexerQueryShard,
    plan_indexer_query_shard,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

TP_SIZES = [2, 4, 8, 16]

# (prefix_lens, extend_lens). The tails are the prefix-cached shapes the A3 box
# actually produced; 16384 is the chunked-prefill size and the only extend here
# that is a multiple of every tp_size, which is why a shard bug hides behind it.
CASES = [
    ([0], [16384]),
    ([1_000_000], [13846]),
    ([989_000], [10828]),
    ([65536, 65536], [8192, 8192]),
    ([120_000, 4096, 77_000], [4096, 1557, 3001]),
    ([32768] * 4, [4096, 100, 4095, 7]),
]

# GLM-5.2's indexer: index_n_heads 32, index_head_dim 128, top-k 2048.
N_HEADS, HEAD_DIM, INDEX_TOPK = 32, 128, 2048


class _MetadataSpy:
    """Stands in for ``torch.ops.cann_ops_transformer``.

    The real metadata op is a CANN kernel. Only its arguments matter here --
    they are the whole contract between the planner and the operator -- so the
    stub records them and hands back a sentinel.
    """

    def __init__(self):
        self.calls = []

    def quant_lightning_indexer_metadata(self, *args, **kwargs):
        self.calls.append(kwargs)
        return ("metadata", len(self.calls))


def _indexer():
    """A stand-in ``self`` for the mixin method: it reads three ints."""
    return types.SimpleNamespace(
        n_heads=N_HEADS, head_dim=HEAD_DIM, index_topk=INDEX_TOPK
    )


def _shard(prefix_lens, extend_lens, tp_size, tp_rank):
    start, rows, num_real, cum_query_lens, key_lens = plan_indexer_query_shard(
        prefix_lens, extend_lens, tp_size, tp_rank
    )
    return _IndexerQueryShard(
        start=start,
        rows=rows,
        num_real=num_real,
        total=sum(extend_lens),
        tp_size=tp_size,
        actual_seq_lengths_q=torch.tensor(cum_query_lens, dtype=torch.int32),
        actual_seq_lengths_kv=torch.tensor(key_lens, dtype=torch.int32),
    )


def _plan(shard, spy):
    with patch.object(torch.ops, "cann_ops_transformer", spy, create=True):
        return DSANPUIndexerMixin._plan_quant_lightning_indexer(
            _indexer(),
            shard.actual_seq_lengths_q,
            shard.actual_seq_lengths_kv,
            torch.device("cpu"),
        )


def _all_shards():
    for prefix_lens, extend_lens in CASES:
        for tp_size in TP_SIZES:
            for tp_rank in range(tp_size):
                yield prefix_lens, extend_lens, tp_size, tp_rank


class TestQuantIndexerShardPlan(CustomTestCase):
    def test_the_plan_describes_the_shard_not_the_batch(self):
        """The invariant the AICore trap violated, over every rank of every case."""
        spy = _MetadataSpy()
        for prefix_lens, extend_lens, tp_size, tp_rank in _all_shards():
            with self.subTest(case=extend_lens, tp=tp_size, rank=tp_rank):
                shard = _shard(prefix_lens, extend_lens, tp_size, tp_rank)
                cu_seqlens_q, seqused_k, _ = _plan(shard, spy)

                # This is the whole point: the last value is how many rows of
                # ``query`` the kernel walks, and ``take()`` hands it exactly
                # ``rows``, of which the first ``num_real`` are real tokens.
                self.assertEqual(int(cu_seqlens_q[-1]), shard.num_real)
                self.assertLessEqual(int(cu_seqlens_q[-1]), shard.rows)

                # v2 wants a leading zero; v1's actual_seq_lengths_query did not
                # have one, and this is where that conversion happens.
                self.assertEqual(int(cu_seqlens_q[0]), 0)
                self.assertEqual(cu_seqlens_q.numel(), len(extend_lens) + 1)

                # NPU cumsum upcasts int32 -> int64 and the operator rejects it.
                self.assertEqual(cu_seqlens_q.dtype, torch.int32)
                self.assertEqual(seqused_k.dtype, torch.int32)

                # Monotone non-decreasing, or the operator reads a request's
                # rows backwards.
                diffs = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
                self.assertTrue(bool((diffs >= 0).all()))

    def test_the_batch_wide_plan_would_read_past_the_query_tensor(self):
        """The control: the plan the pre-fix code used, and why it trapped.

        Built from the batch's own lengths -- which is what
        ``AscendAttnBackend._plan_quant_indexer_metadata`` produces and what the
        quant branch used to read -- the task list tells the operator to walk
        every token in the batch, while the shard passes it ``rows``. Asserting
        the overrun keeps this file honest: if a later change makes the two
        agree, the test above stops proving anything and this one will say so.
        """
        for prefix_lens, extend_lens, tp_size, tp_rank in _all_shards():
            total = sum(extend_lens)
            if total <= tp_size:  # degenerate: every rank holds a whole row
                continue
            with self.subTest(case=extend_lens, tp=tp_size, rank=tp_rank):
                shard = _shard(prefix_lens, extend_lens, tp_size, tp_rank)
                batch_wide_last = total  # cumsum(extend_lens)[-1]
                self.assertGreater(batch_wide_last, shard.rows)
                self.assertGreater(batch_wide_last, shard.num_real)

    def test_key_lengths_are_per_request_not_cumulative(self):
        """``seqused_k`` is per-request; ``cu_seqlens_q`` is cumulative.

        The two arrive next to each other and the operator reads them by
        position, so passing a cumsum for either is a silent corruption rather
        than an error. A request with no rows on this rank gets key length 0.
        """
        spy = _MetadataSpy()
        for prefix_lens, extend_lens, tp_size, tp_rank in _all_shards():
            with self.subTest(case=extend_lens, tp=tp_size, rank=tp_rank):
                shard = _shard(prefix_lens, extend_lens, tp_size, tp_rank)
                _, seqused_k, _ = _plan(shard, spy)
                self.assertEqual(seqused_k.numel(), len(extend_lens))
                self.assertTrue(
                    bool((seqused_k == shard.actual_seq_lengths_kv).all()),
                    "seqused_k must be the planner's per-request key lengths",
                )
                for i, (prefix_len, extend_len) in enumerate(
                    zip(prefix_lens, extend_lens)
                ):
                    key_len = int(seqused_k[i])
                    if key_len:
                        # Prefix plus this rank's last token of the request.
                        self.assertGreater(key_len, prefix_len)
                        self.assertLessEqual(key_len, prefix_len + extend_len)
                    # A request with no local rows contributes no query rows.
                    local_q = int(shard.actual_seq_lengths_q[i]) - (
                        int(shard.actual_seq_lengths_q[i - 1]) if i else 0
                    )
                    self.assertEqual(key_len == 0, local_q == 0)

    def test_the_metadata_op_runs_once_per_forward_not_once_per_layer(self):
        """GLM-5.2 has 78 layers and the lengths do not vary by layer.

        The backend's per-batch pre-plan is skipped when a shard is active, so
        the cache on the shard is what keeps this off the per-layer path. This
        replays the call site's caching rule.
        """
        spy = _MetadataSpy()
        shard = _shard([989_000], [10828], tp_size=8, tp_rank=3)
        plans = []
        for _layer in range(78):
            if shard.quant_indexer_plan is None:
                shard.quant_indexer_plan = _plan(shard, spy)
            plans.append(shard.quant_indexer_plan)

        self.assertEqual(len(spy.calls), 1, "planned more than once per forward")
        self.assertTrue(all(p is plans[0] for p in plans))

    def test_the_operator_is_told_the_shard_s_batch_size_and_shapes(self):
        """The rest of the metadata contract, which no other test covers."""
        spy = _MetadataSpy()
        prefix_lens, extend_lens = CASES[-1]
        shard = _shard(prefix_lens, extend_lens, tp_size=4, tp_rank=1)
        _plan(shard, spy)

        (kwargs,) = spy.calls
        self.assertEqual(kwargs["batch_size"], len(extend_lens))
        self.assertEqual(kwargs["layout_q"], "TND")
        self.assertEqual(kwargs["layout_k"], "PA_BBND")
        self.assertIs(kwargs["cu_seqlens_q"].dtype, torch.int32)
        self.assertIs(kwargs["seqused_k"].dtype, torch.int32)
        # -1 means "derive from the task list"; a stale positive value here
        # would override the shard's own lengths.
        self.assertEqual(kwargs["max_seqlen_q"], -1)
        self.assertEqual(kwargs["max_seqlen_k"], -1)

    def test_a_rank_past_the_last_real_token_plans_an_empty_call(self):
        """Padding ranks exist whenever total is not a multiple of tp_size."""
        spy = _MetadataSpy()
        # 7 tokens over 8 ranks: rows == 1, so rank 7 owns only padding.
        shard = _shard([4096], [7], tp_size=8, tp_rank=7)
        self.assertEqual(shard.num_real, 0)
        cu_seqlens_q, seqused_k, _ = _plan(shard, spy)
        self.assertEqual(int(cu_seqlens_q[-1]), 0)
        self.assertTrue(bool((seqused_k == 0).all()))


if __name__ == "__main__":
    unittest.main()
