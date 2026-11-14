"""CPU regression for fresh/reused indexer overlap scheduling."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace


class TestIndexerOverlap(unittest.TestCase):
    def test_fresh_reuse_and_nextn(self):
        source = (
            Path(__file__).resolve().parents[3]
            / "python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py"
        )
        module = ast.parse(source.read_text())
        function = next(
            n
            for n in module.body
            if isinstance(n, ast.FunctionDef) and n.name == "forward_dsa_prepare_npu"
        )
        predicate = next(
            n
            for n in function.body
            if isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "indexer_runs" for t in n.targets
            )
        )
        gather = next(
            n
            for n in function.body
            if isinstance(n, ast.If)
            and ast.unparse(n.test)
            == "_enable_dcp_sfa_gather_overlap and _dcp_extend_active(forward_batch)"
        )
        indexer = next(
            n
            for n in function.body
            if isinstance(n, ast.If)
            and isinstance(n.test, ast.Name)
            and n.test.id == "indexer_runs"
        )
        # Retain the production indexer call/else; DCP remapping is orthogonal.
        indexer.body = indexer.body[:1]
        code = compile(
            ast.fix_missing_locations(
                ast.Module(body=[predicate, gather, indexer], type_ignores=[])
            ),
            str(source),
            "exec",
        )
        for skip, nextn, previous in [
            (False, False, None),
            (True, False, "old"),
            (True, True, None),
            (True, True, "old"),
        ]:
            for overlap in [False, True]:
                with self.subTest(
                    skip=skip, nextn=nextn, previous=previous, overlap=overlap
                ):
                    calls = []
                    m = SimpleNamespace(
                        skip_topk=skip,
                        is_nextn=nextn,
                        layer_id=0,
                        indexer=lambda *args: calls.append("indexer") or "fresh",
                    )
                    batch = SimpleNamespace()
                    scope = dict(
                        m=m,
                        prev_topk_indices=previous,
                        _enable_dcp_sfa_gather_overlap=overlap,
                        _dcp_extend_active=lambda batch: True,
                        forward_batch=batch,
                        _dcp_start_extend_gather_npu=lambda *args: (
                            calls.append("gather") or "pending"
                        ),
                        _dcp_sfa_gather_topk_width=lambda *args: 1,
                        hidden_states=None,
                        q_lora=None,
                        positions=None,
                        input_on_attn_tp_slices=False,
                        dynamic_scale=None,
                    )
                    exec(code, scope)
                    fresh = not skip or (nextn and previous is None)
                    self.assertEqual(
                        calls, (["indexer"] if fresh else ["gather"] if overlap else [])
                    )
                    self.assertEqual(
                        scope["topk_indices"], "fresh" if fresh else previous
                    )

    def test_gather_width_for_fresh_and_reused_topk(self):
        source = (
            Path(__file__).resolve().parents[3]
            / "python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py"
        )
        module = ast.parse(source.read_text())
        function = next(
            n
            for n in module.body
            if isinstance(n, ast.FunctionDef) and n.name == "_dcp_sfa_gather_topk_width"
        )
        future = ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
        code = compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, function], type_ignores=[])
            ),
            str(source),
            "exec",
        )
        scope = {}
        exec(code, scope)
        width = scope["_dcp_sfa_gather_topk_width"]
        previous = SimpleNamespace(shape=(2, 7))
        for skip, nextn, prior, expected in [
            (False, False, None, 11),
            (True, False, previous, 7),
            (True, True, None, 11),
            (True, True, previous, 7),
        ]:
            with self.subTest(skip=skip, nextn=nextn, prior=prior):
                m = SimpleNamespace(
                    skip_topk=skip,
                    is_nextn=nextn,
                    indexer=SimpleNamespace(index_topk=11),
                )
                self.assertEqual(width(m, prior), expected)


if __name__ == "__main__":
    unittest.main()
