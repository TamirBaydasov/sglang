"""Exercise the real dummy-batch hook without importing NPU initialization."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "python/sglang/srt/model_executor/model_runner.py"


class TestGraphOriginSlots(unittest.TestCase):
    def hook(self, npu=True, dcp=True, draft=False):
        tree = ast.parse(SOURCE.read_text())
        method = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef)
            and n.name == "prepare_dummy_forward_batch"
        )
        future = ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
        ns = dict(_is_npu=npu, get_parallel=lambda: SimpleNamespace(dcp_enabled=dcp))
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=[future, method], type_ignores=[])
                ),
                str(SOURCE),
                "exec",
            ),
            ns,
        )
        runner = SimpleNamespace(
            is_draft_worker=draft, attn_tp_sequence_sharded=lambda _: False
        )
        return lambda batch: ns[method.name](runner, batch)

    def batch(self, slots):
        return SimpleNamespace(
            out_cache_loc=slots,
            origin_out_cache_loc=None,
            _forward_num_tokens=lambda: slots.numel(),
        )

    def test_capture_and_replay_share_stable_allocator_global_slots(self):
        slots = torch.tensor([17, 145, 0])
        batch = self.hook()(self.batch(slots))
        self.assertIs(batch.origin_out_cache_loc, slots)
        slots.copy_(torch.tensor([33, 81, 0]))
        self.assertEqual(batch.origin_out_cache_loc.tolist(), [33, 81, 0])

    def test_preserve_explicit_origin_and_other_platform_roles(self):
        slots = torch.tensor([1, 2])
        origin = torch.tensor([41, 42])
        batch = self.batch(slots)
        batch.origin_out_cache_loc = origin
        self.assertIs(self.hook()(batch).origin_out_cache_loc, origin)
        for kwargs in [dict(npu=False), dict(dcp=False), dict(draft=True)]:
            self.assertIsNone(
                self.hook(**kwargs)(self.batch(slots)).origin_out_cache_loc
            )


if __name__ == "__main__":
    unittest.main()
