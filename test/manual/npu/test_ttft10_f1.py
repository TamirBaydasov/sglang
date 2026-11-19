"""CPU checks for the historical F1 port and actual MegaMoE dispatch guard."""

import ast
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[3]
POOL = ROOT / "python/sglang/srt/hardware_backend/npu/memory_pool_npu.py"


def methods(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, *nodes], type_ignores=[])
            ),
            str(path),
            "exec",
        ),
        namespace,
    )
    return namespace


def pool(buffered=True, requests=1):
    flags = SimpleNamespace(
        SGLANG_NPU_SFA_F1_PREALLOC=SimpleNamespace(get=lambda: buffered),
        SGLANG_NPU_SFA_KV_QUANT_READ=SimpleNamespace(get=lambda: False),
    )
    ns = methods(
        POOL,
        {
            "_prepare_dcp_packed_read_buffers",
            "_gather_packed_sfa_rows",
            "dequant_packed_sfa_rows",
        },
        dict(torch=torch, envs=flags, print_info_once=lambda *a: None),
    )
    rank, rope, rows = 256, 4, 8
    codes = torch.randint(-100, 101, (2, rows, rank), dtype=torch.int8)
    ropes = torch.randn(2, rows, rope).to(torch.bfloat16)
    scales = (
        torch.tensor([0.03125, 0.0078125], dtype=torch.float32)
        .expand(2, rows, 2)
        .clone()
    )
    storage = torch.cat(
        [codes.view(torch.uint8), ropes.view(torch.uint8), scales.view(torch.uint8)],
        dim=-1,
    ).view(torch.int8)
    obj = SimpleNamespace(
        sfa_kv_store_8bit=True,
        dcp_size=16,
        k_buffer=storage,
        kv_cache_dim=storage.shape[-1],
        kv_lora_rank=rank,
        qk_rope_head_dim=rope,
        store_dtype=torch.int8,
        dtype=torch.bfloat16,
        device="cpu",
        start_layer=0,
        layer_transfer_counter=None,
    )
    runtime = types.ModuleType("sglang.srt.runtime_context")
    runtime.get_schedule = lambda: SimpleNamespace(prefill_max_requests=requests)
    with patch.dict(sys.modules, {"sglang.srt.runtime_context": runtime}):
        ns["_prepare_dcp_packed_read_buffers"](obj)
    obj.dequant_packed_sfa_rows = types.MethodType(ns["dequant_packed_sfa_rows"], obj)
    return obj, ns["_gather_packed_sfa_rows"], codes, ropes, scales


class TestBufferedF1(unittest.TestCase):
    def test_physical_rows_and_fp32_rounding_match_current_f1(self):
        buffered, read, codes, ropes, scales = pool()
        current, old_read, *_ = pool(False)
        current.k_buffer = buffered.k_buffer
        for layer, loc in [(0, [6, 0, 3, 6]), (1, [7, 2]), (0, [1])]:
            ids = torch.tensor(loc, dtype=torch.int64)
            actual = read(buffered, layer, ids, None)
            expected = old_read(current, layer, ids, None)
            f2_fallback = buffered.dequant_packed_sfa_rows(
                buffered.k_buffer[layer, ids], None
            )
            for a, b in zip(expected, f2_fallback):
                self.assertTrue(torch.equal(a, b))
            for a, b in zip(actual, expected):
                self.assertEqual(a.dtype, b.dtype)
                self.assertTrue(torch.equal(a, b))
                self.assertTrue(a.is_contiguous())
            independent = (
                (
                    codes[layer, ids].float().reshape(-1, 2, 128)
                    * scales[layer, ids].unsqueeze(-1)
                )
                .reshape(-1, 256)
                .bfloat16()
            )
            self.assertTrue(torch.equal(actual[0].squeeze(1), independent))
            self.assertTrue(torch.equal(actual[1].squeeze(1), ropes[layer, ids]))

    def test_slots_are_reused_and_other_slot_remains_unchanged(self):
        obj, read, *_ = pool()
        pointers = [
            [x.data_ptr() for x in slot] for slot in obj._dcp_packed_read_buffers
        ]
        first = read(obj, 0, torch.tensor([1, 2]), None)
        saved = tuple(x.clone() for x in first)
        read(obj, 1, torch.tensor([3, 4, 5]), None)
        for a, b in zip(first, saved):
            self.assertEqual(a.dtype, b.dtype)
            self.assertTrue(torch.equal(a, b))
        read(obj, 0, torch.tensor([0, 6]), None)
        self.assertEqual(
            pointers,
            [[x.data_ptr() for x in slot] for slot in obj._dcp_packed_read_buffers],
        )

    def test_capacity_dtype_and_stream_refusals_keep_live_output(self):
        obj, read, *_ = pool()
        first = read(obj, 0, torch.tensor([1, 2]), None)
        saved = first[0].clone()
        # Capacity bounds the scratch, not valid repeated physical indices.
        large = read(obj, 0, torch.zeros(9, dtype=torch.int64), None)
        self.assertEqual(large[0].shape[0], 9)
        self.assertNotEqual(large[0].data_ptr(), first[0].data_ptr())
        obj._dcp_packed_read_owners[0] = "other-stream"
        other = read(obj, 0, torch.tensor([3, 4]), None)
        self.assertNotEqual(other[0].data_ptr(), first[0].data_ptr())
        self.assertTrue(torch.equal(first[0], saved))
        fp32 = read(obj, 0, torch.tensor([0]), torch.float32)
        self.assertEqual(fp32[0].dtype, torch.float32)
        disabled, *_ = pool(requests=2)
        self.assertIsNone(disabled._dcp_packed_read_buffers)

    def test_memory_reservation_includes_outputs_and_mega_communication(self):
        file = ROOT / "python/sglang/srt/hardware_backend/npu/attention/dcp_memory.py"
        enabled = {
            "SGLANG_NPU_ENABLE_DCP_PACKED_READ",
            "SGLANG_NPU_ENABLE_DCP_PACKED_GATHER_PREFETCH",
            "SGLANG_NPU_SFA_KV_8BIT",
        }
        parallel = SimpleNamespace(attn_tp_size=16, moe_ep_size=1)
        projection = SimpleNamespace(element_size=lambda: 2)
        layer = SimpleNamespace(
            w_vc=projection,
            use_dsa=True,
            o_proj=SimpleNamespace(output_size=4096),
            kv_lora_rank=256,
            qk_rope_head_dim=4,
            moe_ep_size=16,
            hidden_size=4096,
            num_experts=256,
        )
        ns = methods(
            file,
            {"runtime_gather_reserve_bytes"},
            dict(
                _flag=lambda key, default=False: (
                    key in enabled if isinstance(default, bool) else default
                ),
                get_parallel=lambda: parallel,
                get_exec=lambda: SimpleNamespace(moe=SimpleNamespace(fuseep_mode=3)),
                envs=SimpleNamespace(
                    SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=SimpleNamespace(
                        get=lambda: 2048
                    )
                ),
            ),
        )
        model = SimpleNamespace(modules=lambda: [layer])
        reserve = ns["runtime_gather_reserve_bytes"]
        plain = reserve(model, 32768, context_tokens=1048576)
        enabled.add("SGLANG_NPU_SFA_F1_PREALLOC")
        buffered = reserve(model, 32768, context_tokens=1048576)
        self.assertGreaterEqual(buffered - plain, 2 * 63488 * (256 + 4) * 2)
        if "get_low_latency_rdma_size_hint" in file.read_text():
            deep_ep = types.ModuleType("deep_ep")
            hints = []
            deep_ep.Buffer = SimpleNamespace(
                get_low_latency_rdma_size_hint=lambda *args: (
                    hints.append(args) or 123456
                )
            )
            parallel.moe_ep_size = 16
            with patch.dict(sys.modules, {"deep_ep": deep_ep}):
                mega = reserve(model, 32768, context_tokens=1048576)
            self.assertEqual(mega - buffered, (1 << 30) + 123456)
            self.assertEqual(hints, [(2048, 4096, 16, 256)])

    def test_mega_moe_guard_checks_actual_padded_rows_before_buffer_creation(self):
        file = ROOT / "python/sglang/srt/hardware_backend/npu/moe/fuseep.py"
        if not file.exists() or "def _mega_moe_capacity" not in file.read_text():
            self.skipTest("TP branch does not contain MegaMoE")
        calls = []
        ns = methods(
            file,
            {"forward_fuseep"},
            dict(
                torch=torch,
                get_exec=lambda: SimpleNamespace(moe=SimpleNamespace(fuseep_mode=3)),
                _mega_moe_capacity=lambda: 2048,
                _log_mega_moe_geometry=lambda *a: None,
                _get_fuseep_buffer=lambda layer: calls.append("buffer") or None,
                _prepare_mega_moe_buffer=lambda value: SimpleNamespace(
                    fused_deep_moe=lambda **kw: (kw["x"], None)
                ),
            ),
        )
        layer = SimpleNamespace(
            moe_ep_size=16,
            moe_tp_size=1,
            _mega_moe_weights_processed=True,
            w13_weight=None,
            w13_weight_scale=None,
            w2_weight=None,
            w2_weight_scale=None,
            num_experts=256,
        )
        for rows in (2047, 2048):
            x = torch.zeros(rows, 4, dtype=torch.bfloat16)
            topk = SimpleNamespace(
                topk_ids=torch.zeros(rows, 1, dtype=torch.int64),
                topk_weights=torch.ones(rows, 1),
            )
            self.assertEqual(ns["forward_fuseep"](layer, x, topk).shape[0], rows)
        self.assertEqual(len(calls), 2)
        with self.assertRaisesRegex(ValueError, "2049 tokens"):
            ns["forward_fuseep"](
                layer, torch.zeros(2049, 4, dtype=torch.bfloat16), topk
            )
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
