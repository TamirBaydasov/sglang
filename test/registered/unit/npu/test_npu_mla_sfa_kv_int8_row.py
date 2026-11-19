"""CPU unit test for the packed 8-bit SFA latent KV row (branch F).

Pins the 656 B/token row layout and its inverse against hand-built packed
rows (no NPU op needed):

    row = latent int8 x kv_lora_rank | rope BF16 bytes x qk_rope_head_dim*2
        | per-128-tile FP32 scales x kv_lora_rank // 128

``view_packed_sfa_rows`` must return the (latent, rope, scale) views of that
record, ``dequant_packed_sfa_rows`` must invert the pack byte-for-byte on the
rope and reproduce ``latent_int8 * scale`` per 128-tile on the latent (the
latent reinterpreted as SIGNED int8, not the unsigned byte values), and
``gather_packed_sfa_rows_raw`` must return the raw uint8 rows the F2 gather
ships. The pool class is bound over a minimal fake ``self`` so the test
exercises exactly these methods without torch_npu.

Usage:
    python -m pytest test_npu_mla_sfa_kv_int8_row.py -v
    python test_npu_mla_sfa_kv_int8_row.py
"""

import struct
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool

KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
TILES = KV_LORA_RANK // 128
KV_CACHE_DIM = (
    KV_LORA_RANK + QK_ROPE_HEAD_DIM * 2 + TILES * 4
)  # 512 + 128 + 16 = 656
ROWS = 9


def _make_pool(**overrides):
    gen = torch.Generator().manual_seed(7)
    storage = torch.randint(
        0, 256, (ROWS, KV_CACHE_DIM), dtype=torch.uint8, generator=gen
    )
    pool = SimpleNamespace(
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        kv_cache_dim=KV_CACHE_DIM,
        dtype=torch.bfloat16,
        layer_transfer_counter=None,
        start_layer=0,
        k_buffer=[storage],
    )
    for name, value in overrides.items():
        setattr(pool, name, value)
    for method in (
        "_gather_packed_sfa_rows",
        "gather_packed_sfa_rows_raw",
        "view_packed_sfa_rows",
        "dequant_packed_sfa_rows",
    ):
        setattr(pool, method, getattr(NPUMLATokenToKVPool, method).__get__(pool))
    return pool


def _build_expected_rows():
    """Hand-built packed rows plus the values they must decode to."""
    gen = torch.Generator().manual_seed(3)
    latent_f32 = torch.randn(ROWS, KV_LORA_RANK, generator=gen)
    rope_bf16 = torch.randn(ROWS, QK_ROPE_HEAD_DIM, generator=gen).to(torch.bfloat16)
    scale_f32 = torch.rand(ROWS, TILES, generator=gen) * 0.01 + 0.001

    latent_i8 = torch.empty(ROWS, KV_LORA_RANK, dtype=torch.int8)
    for r in range(ROWS):
        for t in range(TILES):
            tile = latent_f32[r, t * 128 : (t + 1) * 128]
            s = float(scale_f32[r, t])
            q = torch.clamp(torch.round(tile / s), -128, 127)
            latent_i8[r, t * 128 : (t + 1) * 128] = q.to(torch.int8)

    rows = torch.empty(ROWS, KV_CACHE_DIM, dtype=torch.uint8)
    rows[:, :KV_LORA_RANK] = latent_i8.view(torch.uint8)
    rows[:, KV_LORA_RANK : KV_LORA_RANK + QK_ROPE_HEAD_DIM * 2] = rope_bf16.view(
        torch.uint8
    )
    scales_flat = scale_f32.reshape(-1)  # row-major: row0 tiles, row1 tiles, ...
    scale_bytes = b"".join(struct.pack("<f", s) for s in scales_flat.tolist())
    scale_u8 = torch.frombuffer(bytearray(scale_bytes), dtype=torch.uint8).clone()
    rows[:, KV_LORA_RANK + QK_ROPE_HEAD_DIM * 2 :] = scale_u8.view(ROWS, TILES * 4)
    return rows, latent_f32, latent_i8, rope_bf16, scale_f32


class TestSfaPackedRowLayout(unittest.TestCase):
    def test_row_width_is_656_for_512_64(self):
        self.assertEqual(KV_CACHE_DIM, 656)

    def test_view_packed_sfa_rows_shapes_and_values(self):
        rows, _, latent_i8, rope_bf16, scale_f32 = _build_expected_rows()
        pool = _make_pool()
        pool.k_buffer = [rows]
        latent, rope, scale = pool.view_packed_sfa_rows(rows)
        self.assertEqual(latent.dtype, torch.int8)
        self.assertEqual(rope.dtype, torch.bfloat16)
        self.assertEqual(scale.dtype, torch.float32)
        self.assertTrue(torch.equal(latent, latent_i8))
        self.assertTrue(torch.equal(rope, rope_bf16))
        self.assertTrue(torch.equal(scale, scale_f32))

    def test_dequant_roundtrip(self):
        rows, latent_f32, latent_i8, rope_bf16, scale_f32 = _build_expected_rows()
        pool = _make_pool()
        pool.k_buffer = [rows]
        nope, rope = pool.dequant_packed_sfa_rows(rows)
        # rope is a straight bf16 reinterpretation: bit-exact.
        self.assertTrue(torch.equal(rope.squeeze(1), rope_bf16))
        # latent is int8 * fp32 scale, computed in fp32 and rounded to bf16.
        expected = (
            (latent_i8.to(torch.float32) * scale_f32.repeat_interleave(128, dim=1))
            .to(torch.bfloat16)
        )
        self.assertTrue(torch.equal(nope.squeeze(1), expected))
        # And it is the quantized values, not the originals (lossy by design).
        self.assertFalse(torch.allclose(nope.squeeze(1).float(), latent_f32, atol=1e-3))

    def test_latent_is_signed_int8_not_unsigned_bytes(self):
        rows = torch.zeros(2, KV_CACHE_DIM, dtype=torch.uint8)
        rows[0, 0] = 255  # int8 -1, not 255 (row 1 keeps the untouched 0 byte)
        pool = _make_pool()
        nope, _ = pool.dequant_packed_sfa_rows(rows)
        # scale row of zeros -> dequant -1 * 0 == 0, but the REINTERPRETATION
        # is what we pin: view path reports -1.
        latent, _, _ = pool.view_packed_sfa_rows(rows)
        self.assertEqual(int(latent[0, 0]), -1)
        self.assertEqual(int(latent[1, 0]), 0)

    def test_gather_raw_returns_uint8_rows(self):
        rows, *_ = _build_expected_rows()
        pool = _make_pool()
        pool.k_buffer = [rows]
        loc = torch.tensor([7, 0, 4])
        raw = pool.gather_packed_sfa_rows_raw(0, loc)
        self.assertEqual(raw.dtype, torch.uint8)
        self.assertEqual(tuple(raw.shape), (3, KV_CACHE_DIM))
        self.assertTrue(torch.equal(raw, rows[[7, 0, 4]]))

    def test_get_mla_kv_buffer_dispatch_dequantizes(self):
        rows, _, _, rope_bf16, _ = _build_expected_rows()
        pool = _make_pool()
        pool.k_buffer = [rows]
        nope, rope = pool._gather_packed_sfa_rows(0, torch.tensor([2]), None)
        self.assertEqual(nope.dtype, torch.bfloat16)
        self.assertEqual(tuple(nope.shape), (1, 1, KV_LORA_RANK))
        self.assertTrue(torch.equal(rope.squeeze(1), rope_bf16[[2]]))


if __name__ == "__main__":
    unittest.main()
