"""CPU regressions for SFA schema, FP8 routing, and packed F2 prefetch."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[3]
BACKEND = ROOT / "python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py"
MODULE = (
    ROOT
    / "python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py"
)


def compile_nodes(nodes, namespace):
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, *nodes], type_ignores=[])
            ),
            "sfa-review",
            "exec",
        ),
        namespace,
    )


class TestSFAReview(unittest.TestCase):
    def test_schema_uses_real_torch_argument_names(self):
        method = next(
            n
            for n in ast.walk(ast.parse(BACKEND.read_text()))
            if isinstance(n, ast.FunctionDef) and n.name == "_f2_quant_sfa_api_ok"
        )
        schema = torch._C.parse_schema(
            "sfa_test(Tensor query, Tensor key_rope, Tensor key_dequant_scale, Tensor value_dequant_scale) -> Tensor"
        )
        operator = SimpleNamespace(default=SimpleNamespace(_schema=schema))
        namespace = dict(
            torch=SimpleNamespace(
                ops=SimpleNamespace(
                    npu=SimpleNamespace(npu_kv_quant_sparse_flash_attention=operator)
                )
            )
        )
        compile_nodes([method], namespace)
        self.assertEqual(namespace[method.name](SimpleNamespace()), (True, ""))

    def test_paged_guard_preserves_existing_fp8(self):
        guard = next(
            n
            for n in ast.walk(ast.parse(BACKEND.read_text()))
            if isinstance(n, ast.Assert)
            and n.msg is not None
            and "packed 8-bit SFA KV reached the unsharded paged read"
            in ast.unparse(n.msg)
        )
        for int8 in (False, True):
            namespace = dict(
                self=SimpleNamespace(
                    token_to_kv_pool=SimpleNamespace(sfa_kv_store_8bit=int8),
                    _sfa_packed_kv=lambda: True,
                )
            )
            if int8:
                with self.assertRaises(AssertionError):
                    compile_nodes([guard], namespace)
            else:
                compile_nodes([guard], namespace)

    def test_prefetch_forwards_f2_and_sends_opaque_rows(self):
        tree = ast.parse(MODULE.read_text())
        send = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "_dcp_extend_send_rows"
        )
        prefetch = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "_issue_dcp_gather_prefetch"
        )
        call = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == prefetch.name
        )
        namespace = {}
        compile_nodes([send], namespace)
        rows = torch.arange(3 * 656, dtype=torch.int32).to(torch.uint8).reshape(3, 656)
        pool = SimpleNamespace(
            gather_packed_sfa_rows_raw=lambda *args: rows,
            get_mla_kv_buffer=lambda *args, **kw: (
                torch.zeros(3, 1, 512, dtype=torch.bfloat16),
                torch.zeros(3, 1, 64, dtype=torch.bfloat16),
            ),
        )
        plan = SimpleNamespace(send_rows=3, local_lens=[3], padded_lens=[3])
        md = SimpleNamespace(dcp_local_prefix_kv_indices=torch.arange(3))
        sent = []

        def issue(*args, **kwargs):
            bound = dict(zip([a.arg for a in prefetch.args.args], args))
            bound.update(kwargs)
            sent.extend(
                namespace[send.name](
                    pool, None, md, plan, 0, True, bound.get("f2", False)
                )
            )

        namespace.update(
            state=None,
            pool=pool,
            m=None,
            md=md,
            plan=plan,
            parallel=None,
            layer_id=0,
            k_nope=None,
            k_pe=None,
            packed_kv=True,
            f2=True,
            _issue_dcp_gather_prefetch=issue,
        )
        compile_nodes([ast.Expr(value=call)], namespace)
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0].dtype, torch.uint8)
        self.assertEqual(tuple(sent[0].shape), (3, 1, 656))
        self.assertTrue(torch.equal(sent[0].squeeze(1), rows))


if __name__ == "__main__":
    unittest.main()
