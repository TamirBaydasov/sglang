"""CPU unit test for the D1v3 token-sharded rebuild's pure-tensor math.

No collectives, no NPU: construct ``DsaTokenShardPlan`` records directly and
pin

1. the rank slice layout (``local_start = rank * rows``, contiguous blocks in
   sequence order, so a rank-major all-gather of the slices reconstructs the
   batch and the pad is at the tail -- the premise of BUG 4's full-width
   restore and BUG 6/10's trims),
2. ``dsa_token_shard_slice`` padding (BUG 16's premise: the first-layer
   residual is [num_tokens], which equals ``num_tokens_pad`` only when the
   batch divides by attn_tp),
3. the D2v3 o_proj reassembly (rank-major flat fill is NOT the [out, in]
   matrix; ``movedim(0, 1)`` over the gathered [tp, out, in_r] buffer
   reconstructs exactly the dim-1 concatenation the static replica uses),
4. the plan's per-request ``key_lens`` being the LOCAL length a straddling
   request's last local token sees (BUG 17's premise).

Usage:
    python -m pytest test_dsa_cp_token_shard_d1v3.py -v
    python test_dsa_cp_token_shard_d1v3.py
"""

import unittest

import torch

from sglang.srt.layers.attention.dsa.dsa_token_shard import dsa_token_shard_slice
from sglang.srt.layers.attention.dsa.dsa_token_shard_layout import DsaTokenShardPlan
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def make_plans(num_tokens, tp, seq_lens):
    """The same layout plan_dsa_token_shard builds: rank blocks of ceil rows."""
    rows = -(-num_tokens // tp)
    pad = rows * tp
    plans = []
    for r in range(tp):
        start = r * rows
        end = min(num_tokens, start + rows)
        end_pad = min(pad, start + rows)
        key_lens = [min(end, s) for s in seq_lens]
        plans.append(
            DsaTokenShardPlan(
                num_tokens=num_tokens,
                num_tokens_pad=pad,
                rows=rows,
                local_start=start,
                local_end=end,
                local_end_with_pad=end_pad,
                query_lens=[max(0, min(end, s) - start) for s in seq_lens],
                key_lens=key_lens,
            )
        )
    return plans


class TestRankSliceLayout(unittest.TestCase):
    def test_slices_reconstruct_the_batch_in_order(self):
        num_tokens, tp, hidden = 10, 4, 8
        plans = make_plans(num_tokens, tp, seq_lens=[10])
        full = torch.arange(num_tokens * hidden, dtype=torch.float32).reshape(
            num_tokens, hidden
        )
        slices = [dsa_token_shard_slice(full, p) for p in plans]
        self.assertEqual(slices[0].shape[0], plans[0].rows)
        # Unpadded values: the concatenated real rows rebuild the batch.
        rebuilt = torch.cat(
            [s[: p.local_end - p.local_start] for s, p in zip(slices, plans)]
        )
        self.assertTrue(torch.equal(rebuilt, full))
        # Padded slices concatenate to num_tokens_pad with zero pad at the tail.
        padded = torch.cat(slices)
        self.assertEqual(padded.shape[0], plans[0].num_tokens_pad)
        self.assertTrue(torch.equal(padded[:num_tokens], full))
        self.assertTrue(
            torch.equal(padded[num_tokens:], torch.zeros_like(padded[num_tokens:]))
        )

    def test_non_divisible_batch_has_pad_rows_bug16_premise(self):
        plans = make_plans(10, 4, [10])
        p0 = plans[0]
        self.assertEqual(p0.rows, 3)
        self.assertEqual(p0.num_tokens_pad, 12)
        self.assertNotEqual(p0.num_tokens, p0.num_tokens_pad)
        # A divisible batch has no gap: the == num_tokens_pad test happens to
        # hold there, which is why the BUG-16 form was masked on power-of-two
        # chunks.
        plans2 = make_plans(12, 4, [12])
        self.assertEqual(plans2[0].num_tokens, plans2[0].num_tokens_pad)

    def test_key_lens_are_local_for_straddling_requests_bug17_premise(self):
        # One request of 24 tokens across 4 ranks of 6 rows: rank 0's last
        # local token is token 5, so it sees 6 keys -- not the full 24.
        plans = make_plans(24, 4, [24])
        self.assertEqual(plans[0].key_lens, [6])
        self.assertEqual(plans[3].key_lens, [24])  # the last rank sees all 24
        self.assertTrue(all(k < 24 for p in plans[:-1] for k in p.key_lens))


class TestOProjReassembly(unittest.TestCase):
    def test_rank_major_gather_plus_movedim_equals_dim1_concat(self):
        tp, out, in_r = 4, 6, 5
        shards = [torch.randn(out, in_r) for _ in range(tp)]
        # The static replica's form: concat along the sharded input dim.
        expected = torch.cat(shards, dim=1)  # [out, tp * in_r]
        # The streamed form: all_gather_into_tensor fills [tp, out, in_r]
        # rank-major (flat), and the reassembly movesims rank to dim 1.
        gathered = torch.stack(shards, dim=0)  # what the flat fill produces
        buf = gathered.movedim(0, 1)  # [out, tp, in_r]
        assembled = buf.reshape(out, tp * in_r)
        self.assertTrue(torch.equal(assembled, expected))

    def test_movedim_formula_matches_per_row_reference(self):
        tp, out, in_r = 3, 4, 2
        shards = [torch.randn(out, in_r) for _ in range(tp)]
        gathered = torch.stack(shards, dim=0)
        assembled = gathered.movedim(0, 1).reshape(out, tp * in_r)
        for o in range(out):
            for r in range(tp):
                for j in range(in_r):
                    self.assertEqual(
                        float(assembled[o, r * in_r + j]), float(shards[r][o, j])
                    )


if __name__ == "__main__":
    unittest.main()
