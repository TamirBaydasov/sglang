"""CPU contract checks for stock sparse attention LSE and paged-to-TND reads."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "python/sglang/srt/hardware_backend/npu/attention/dsa_dcp.py"


def stock_sparse(
    query,
    key,
    value,
    sparse_indices,
    scale_value,
    *,
    query_rope,
    key_rope,
    actual_seq_lengths_query,
    actual_seq_lengths_kv,
    sparse_block_size,
    layout_query,
    layout_kv,
    sparse_mode,
    attention_mode,
    return_softmax_lse,
):
    assert layout_query == layout_kv == "TND"
    assert return_softmax_lse and sparse_block_size == 1 and sparse_mode == 0
    output, maxima, sums = [], [], []
    start = 0
    for i, end in enumerate(actual_seq_lengths_kv.tolist()):
        selected = sparse_indices[i, 0]
        selected = selected[selected >= 0].long() + start
        assert selected.numel()
        logits = (
            query[i] @ key[selected, 0].T + query_rope[i] @ key_rope[selected, 0].T
        ) * scale_value
        maximum = logits.max(-1).values
        weights = (logits - maximum[:, None]).exp()
        output.append(weights @ value[selected, 0] / weights.sum(-1)[:, None])
        maxima.append(maximum)
        sums.append(weights.sum(-1))
        start = end
    assert actual_seq_lengths_query.tolist() == list(range(1, len(output) + 1))
    return torch.stack(output), torch.stack(maxima)[None], torch.stack(sums)[None]


class TestSparseLSEAPI(unittest.TestCase):
    def setUp(self):
        self.capturing = False
        self.nz = False
        self.calls = 0
        self.torch = SimpleNamespace(
            **{name: getattr(torch, name) for name in dir(torch)}
        )
        self.torch.npu = SimpleNamespace(
            is_current_stream_capturing=lambda: self.capturing
        )

        def call(*args, **kwargs):
            self.calls += 1
            return stock_sparse(*args, **kwargs)

        self.namespace = dict(
            torch=self.torch,
            torch_npu=SimpleNamespace(npu_sparse_flash_attention=call),
            is_fia_nz=lambda: self.nz,
        )
        functions = [
            node
            for node in ast.parse(SOURCE.read_text()).body
            if isinstance(node, ast.FunctionDef)
        ]
        future = ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=[future, *functions], type_ignores=[])
                ),
                str(SOURCE),
                "exec",
            ),
            self.namespace,
        )

    def test_masked_rows_lse_and_empty_rank(self):
        torch.manual_seed(3)
        q, qr = torch.randn(3, 2, 4), torch.randn(3, 2, 2)
        k, kr = torch.randn(12, 1, 4), torch.randn(12, 1, 2)
        valid = torch.tensor(
            [[True, False, True, False], [False] * 4, [False, True, True, True]]
        )
        out, lse = self.namespace["forward_tnd_sparse_attention"](
            q_nope=q, q_rope=qr, k_nope=k, k_rope=kr, valid=valid, scaling=0.5
        )
        for i in (0, 2):
            keys = k.reshape(3, 4, 4)[i, valid[i]]
            ropes = kr.reshape(3, 4, 2)[i, valid[i]]
            logits = (q[i] @ keys.T + qr[i] @ ropes.T) * 0.5
            torch.testing.assert_close(out[i], logits.softmax(-1) @ keys)
            torch.testing.assert_close(lse[i], logits.logsumexp(-1))
        self.assertTrue(torch.equal(out[1], torch.zeros_like(out[1])))
        self.assertTrue(torch.isneginf(lse[1]).all())
        out, lse = self.namespace["forward_tnd_sparse_attention"](
            q_nope=q,
            q_rope=qr,
            k_nope=k,
            k_rope=kr,
            valid=torch.zeros_like(valid),
            scaling=0.5,
        )
        self.assertEqual(self.calls, 1)
        self.assertTrue(torch.isneginf(lse).all())
        self.assertFalse(out.any())

    def test_graph_rejected_before_operator(self):
        self.capturing = True
        with self.assertRaisesRegex(RuntimeError, "eager"):
            self.namespace["forward_tnd_sparse_attention"](
                q_nope=None,
                q_rope=None,
                k_nope=None,
                k_rope=None,
                valid=None,
                scaling=1,
            )
        self.assertEqual(self.calls, 0)

    def test_paged_and_nz_verify_reads(self):
        torch.manual_seed(5)
        k, kr = torch.randn(4, 16, 1, 32), torch.randn(4, 16, 1, 16)
        q, qr = torch.randn(2, 2, 32), torch.randn(2, 2, 16)
        tables = torch.tensor([[2, 0], [1, 3]])
        lengths = torch.tensor([20, 32])
        indices = torch.tensor([[[17, -1, 1]], [[18, 0, -1]]], dtype=torch.int32)
        metadata = SimpleNamespace(
            dcp_block_tables=tables,
            dcp_seq_lens=lengths,
            dcp_spec_block_tables=tables,
            dcp_spec_seq_lens=lengths,
        )
        outputs = []
        for nz in (False, True):
            self.nz = nz

            def encode(cache):
                return (
                    cache.reshape(4, 16, -1, 16)
                    .permute(0, 2, 1, 3)
                    .contiguous()
                    .view_as(cache)
                    if nz
                    else cache
                )

            for verify in (False, True):
                outputs.append(
                    self.namespace["forward_dcp_sparse_attention"](
                        q_nope=q,
                        q_rope=qr,
                        k_nope=encode(k),
                        k_rope=encode(kr),
                        topk_indices=indices,
                        actual_seq_lengths_query=torch.tensor([1, 2]),
                        forward_metadata=metadata,
                        forward_batch=SimpleNamespace(
                            forward_mode=SimpleNamespace(
                                is_target_verify=lambda: verify
                            )
                        ),
                        speculative_num_draft_tokens=2,
                        scaling=0.2,
                    )
                )
        for out, lse in outputs[1:]:
            torch.testing.assert_close(out, outputs[0][0])
            torch.testing.assert_close(lse, outputs[0][1])
        for i, logical in enumerate(([17, 1], [18, 0])):
            rows = torch.tensor(logical)
            keys = k[tables[i, rows // 16], rows % 16, 0]
            ropes = kr[tables[i, rows // 16], rows % 16, 0]
            logits = (q[i] @ keys.T + qr[i] @ ropes.T) * 0.2
            torch.testing.assert_close(outputs[0][0][i], logits.softmax(-1) @ keys)
            torch.testing.assert_close(outputs[0][1][i], logits.logsumexp(-1))


if __name__ == "__main__":
    unittest.main()
