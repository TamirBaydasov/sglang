from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_token_shard import get_dsa_token_shard_plan
from sglang.srt.layers.attention.dsa.dsa_token_shard_layout import (
    cumulative,
    plan_dsa_token_shard,
)
from sglang.srt.layers.cp.utils import cp_gather_full_sequence_states
from sglang.srt.layers.dp_attention import attn_tp_all_gather_into_tensor
from sglang.srt.layers.layer_boundary import get_attn_tp_context
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_token_to_kv_pool,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_npu, print_info_once

if is_npu():
    # Registers the CANN ops-transformer kernels under
    # torch.ops.cann_ops_transformer; without this import the
    # torch.ops namespace is empty and op calls raise AttributeError.
    # Optional: only the 950 MXFP8 quant indexer needs it, and a failed import
    # here would drop every DSA model to the Transformers fallback on A3.
    try:
        import cann_ops_transformer  # noqa: F401
    except ImportError:
        pass
    import torch_npu

    from sglang.srt.hardware_backend.npu.utils import get_indexer_weight_stream

_use_ag_after_qlora = envs.SGLANG_USE_AG_AFTER_QLORA.get()


@lru_cache(maxsize=1)
def _shard_indexer_queries() -> bool:
    """Cached, not read at import, so tests can set it; see
    ``reset_indexer_shard_flag``."""
    return envs.SGLANG_NPU_ENABLE_DSA_INDEXER_QUERY_SHARDING.get()


def reset_indexer_shard_flag() -> None:
    """Re-read the flag. For tests; never call this while serving."""
    _shard_indexer_queries.cache_clear()


@lru_cache(maxsize=1)
def _create_hadamard_128_cpu() -> torch.Tensor:
    matrix = [[1.0]]
    while len(matrix) < 128:
        matrix = [row + row for row in matrix] + [
            row + [-value for value in row] for row in matrix
        ]
    return torch.tensor(matrix, dtype=torch.bfloat16)


def create_npu_hadamard_128(head_dim: int, device) -> torch.Tensor:
    assert head_dim == 128
    # Match vllm-ascend SFA: BF16 matrix, normalized once on the pool's device.
    return (_create_hadamard_128_cpu().to(device=device) / (128**0.5)).contiguous()


def _quantize_npu_indexer_activation(x, hadamard, dst_type):
    # Hadamard-rotate x and MX-quantize its 128-dim vectors.
    # Returns (quantized, scale): quantized has x's shape in dst_type (fp8)
    # and scale holds one E8M0 byte per 32-element block, shaped
    # x.shape[:-1] + (d/64, 2) == x.shape[:-1] + (2, 2) — the descale layout
    # quant_lightning_indexer (v2) expects for quant_mode 3 (MXFP8).
    assert x.dtype == torch.bfloat16 and x.shape[-1] == 128
    if x.numel() == 0:
        return (
            torch.empty_like(x, dtype=dst_type),
            torch.zeros(x.shape[:-1] + (2, 2), dtype=torch.uint8, device=x.device).view(
                torch.float8_e8m0fnu
            ),
        )
    rotated = x @ hadamard
    quantized, scale = torch.ops.npu.npu_dynamic_mx_quant(
        rotated.reshape(-1, 128), dst_type=dst_type, axis=-1
    )
    # npu_dynamic_mx_quant may return the block scales as [N, 4] or [N, 2, 2];
    # normalize to the kernel's (d/64, 2) == (2, 2) layout.
    scale = scale.reshape(x.shape[:-1] + (4,)).view(x.shape[:-1] + (2, 2))
    if scale.dtype != torch.float8_e8m0fnu:
        scale = scale.view(torch.float8_e8m0fnu)

    return quantized.reshape(x.shape), scale


def _check_quant_lightning_indexer_constraints(pool) -> None:
    # quant_lightning_indexer (v2) PA_BBND layout: block_size == pool
    # page_size must lie in [16, 1024] and be a multiple of 16.
    page_size = pool.page_size
    assert 16 <= page_size <= 1024 and page_size % 16 == 0, (
        "quant_lightning_indexer (v2) PA_BBND layout requires the index-k "
        f"pool page_size in [16, 1024] and a multiple of 16, got {page_size}. "
        "Relaunch with page_size=64 (engine kwarg / --page-size 64)."
    )


def plan_indexer_query_shard(
    prefix_lens: List[int], extend_lens: List[int], tp_size: int, tp_rank: int
):
    """One attention-TP rank's share of an extend batch's indexer queries.

    The token shard's partition, derived rather than recomputed: the token
    shard skips this top-k's all-gather on the strength of both picking the
    same rows (W2), so they must not be two implementations.

    Returns ``(start, rows, num_real, cum_query_lens, key_lens)``.
    """
    plan = plan_dsa_token_shard(
        extend_lens,
        [p + e for p, e in zip(prefix_lens, extend_lens)],
        tp_size,
        tp_rank,
    )
    return (
        plan.local_start,
        plan.rows,
        max(0, plan.local_end - plan.local_start),
        cumulative(plan.query_lens),
        plan.key_lens,
    )


@dataclass
class _IndexerQueryShard:
    """This rank's rows of an extend batch, and how to reassemble the top-k."""

    start: int
    rows: int
    num_real: int
    total: int
    tp_size: int
    actual_seq_lengths_q: torch.Tensor
    actual_seq_lengths_kv: torch.Tensor
    # ``(cu_seqlens_q, seqused_k, metadata)`` for the quantized indexer, or
    # None until this forward's first indexer layer plans it. The backend's
    # per-batch pre-plan cannot be used on a sharded call -- see the quant
    # branch of ``forward_npu`` -- and these lengths are the same for all 78
    # layers of one forward, so the plan is cached here rather than redone.
    quant_indexer_plan: Optional[tuple] = None

    def take(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_real == self.rows:
            return x[self.start : self.start + self.rows]
        # Padding rows lie past actual_seq_lengths_q[-1]; gather() drops them.
        out = x.new_zeros((self.rows, *x.shape[1:]))
        out[: self.num_real] = x[self.start : self.start + self.num_real]
        return out

    def gather(self, topk_indices: torch.Tensor, num_tokens: int) -> torch.Tensor:
        """All-gather the per-rank top-k back to ``num_tokens`` rows."""
        out = topk_indices.new_empty((self.rows * self.tp_size, topk_indices.shape[-1]))
        attn_tp_all_gather_into_tensor(out, topk_indices.contiguous())
        return out[:num_tokens]

    def resolve(
        self, topk_indices: torch.Tensor, num_tokens: int, forward_batch
    ) -> torch.Tensor:
        """Full-width top-k, or this rank's own rows when the token shard will
        slice the gathered tensor straight back to them anyway (W2).

        Sets ``npu_indexer_topk_is_local`` so attention knows which it got. Rows
        past ``num_real`` are zeroed to match ``dsa_token_shard_slice``'s
        padding, keeping the result bitwise identical.
        """
        plan = get_dsa_token_shard_plan(forward_batch)
        local = plan is not None
        if local and (plan.rows != self.rows or plan.local_start != self.start):
            # Same arithmetic, but the two sites read their inputs separately;
            # if those ever drift, the gather stays correct where this would not.
            print_info_once(
                "DSA token-shard and indexer sharding disagree on the row range "
                f"(plan {plan.local_start}+{plan.rows} against shard "
                f"{self.start}+{self.rows}); keeping the top-k all-gather"
            )
            local = False
        forward_batch.npu_indexer_topk_is_local = local
        if not local:
            return self.gather(topk_indices, num_tokens)
        # Every rank logs, so a grep counts ranks: the output is identical either
        # way, which makes this line the only evidence it ran.
        print_info_once(
            "DSA token-shard W2: skipping the indexer top-k all-gather; each rank "
            "keeps the rows it scored"
        )
        if self.num_real < self.rows:
            topk_indices[self.num_real :] = 0
        return topk_indices


def _build_indexer_query_shard(
    forward_batch: ForwardBatch,
) -> Optional[_IndexerQueryShard]:
    parallel = get_parallel()
    prefix_lens = forward_batch.extend_prefix_lens_cpu
    extend_lens = forward_batch.extend_seq_lens_cpu
    if (
        parallel.attn_tp_size == 1
        or forward_batch.forward_mode not in (ForwardMode.EXTEND, ForwardMode.MIXED)
        or prefix_lens is None
        or extend_lens is None
        or sum(extend_lens) == 0
    ):
        return None
    start, rows, num_real, cum_query_lens, key_lens = plan_indexer_query_shard(
        prefix_lens, extend_lens, parallel.attn_tp_size, parallel.attn_tp_rank
    )
    # Every rank logs, so a grep counts ranks; /server_info cannot show it.
    print_info_once(
        "DSA indexer query sharding is active: prefill indexer queries split "
        f"across attn_tp_size={parallel.attn_tp_size}"
    )
    device = forward_batch.seq_lens.device
    return _IndexerQueryShard(
        start=start,
        rows=rows,
        num_real=num_real,
        total=sum(extend_lens),
        tp_size=parallel.attn_tp_size,
        actual_seq_lengths_q=torch.tensor(
            cum_query_lens, dtype=torch.int32, device=device
        ),
        actual_seq_lengths_kv=torch.tensor(key_lens, dtype=torch.int32, device=device),
    )


def _get_indexer_query_shard(
    forward_batch: ForwardBatch, num_tokens: int
) -> Optional[_IndexerQueryShard]:
    """The query shard for a prefill indexer call, or None to score every row.

    Every rank already holds the full index-K, so each scores 1/attn_tp_size of
    the queries and all-gathers the top-k: 15.97x less indexer work at 1M and
    tp16. Planned once per forward from inputs identical across the group, so
    all ranks take the collective or none do.
    """
    if get_attn_tp_context().input_scattered:
        # A scattered input is already sliced.
        print_info_once(
            "DSA indexer query sharding is off: the attention input "
            "is scattered across the TP group"
        )
        return None
    if not hasattr(forward_batch, "npu_indexer_query_shard"):
        forward_batch.npu_indexer_query_shard = _build_indexer_query_shard(
            forward_batch
        )
    shard = forward_batch.npu_indexer_query_shard
    if shard is None:
        return None
    # SGLang pads the width to a multiple of attn_tp_size; comparing against
    # total alone would disable sharding for 15 token counts in 16.
    padded = shard.rows * shard.tp_size
    if not shard.total <= num_tokens <= padded:
        print_info_once(
            "DSA indexer query sharding is off for this forward: "
            f"num_tokens={num_tokens} is outside [{shard.total}, {padded}]"
        )
        return None
    return shard


class DSANPUIndexerMixin:
    def _plan_quant_lightning_indexer(
        self,
        cum_query_lens: torch.Tensor,
        key_lens: torch.Tensor,
        device,
    ):
        """``(cu_seqlens_q, seqused_k, metadata)`` for one quant indexer call.

        ``cum_query_lens`` is per-request *cumulative* query rows, as the v1
        operator's ``actual_seq_lengths_query`` was; the v2 operator instead
        wants it prefixed with a 0, and its last value is how many rows of
        ``query`` the kernel will read. ``key_lens`` is per-request and not
        cumulative. Both must describe the tensors this call actually passes.
        """
        cum_q = cum_query_lens.to(device=device, dtype=torch.int32)
        cu_seqlens_q = torch.cat([cum_q.new_zeros(1), cum_q])
        seqused_k = key_lens.to(device=device, dtype=torch.int32)
        metadata = torch.ops.cann_ops_transformer.quant_lightning_indexer_metadata(
            self.n_heads,
            1,
            self.head_dim,
            self.index_topk,
            3,  # QUANT_MODE_MXFP8 (5 == QUANT_MODE_MXFP4)
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            batch_size=int(cum_q.numel()),
            max_seqlen_q=-1,
            max_seqlen_k=-1,
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=3,
            cmp_ratio=1,
        )
        return cu_seqlens_q, seqused_k, metadata

    def forward_npu(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        input_on_attn_tp_slices: bool = False,
        dynamic_scale: torch.Tensor = None,
    ) -> torch.Tensor:
        if get_attn_backend().forward_metadata.seq_lens_cpu_int is None:
            actual_seq_lengths_kv = get_attn_backend().forward_metadata.seq_lens
        else:
            actual_seq_lengths_kv = get_attn_backend().forward_metadata.seq_lens_cpu_int
        is_prefill = (
            forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_draft_extend_v2()
            and not forward_batch.forward_mode.is_target_verify()
        )

        bs = q_lora.shape[0]

        # A rank that scores only its own rows projects only those (W3): wq_b is
        # 25.2 MFLOP per token. k stays full width -- every rank writes the whole
        # index-K, which is what lets any of them score a subset. Neox only: the
        # other branch rotates q and k in one call. Prefill CP is excluded: its
        # indexer below knows nothing of this shard.
        uses_prefill_cp = (
            self.dsa_enable_prefill_cp and forward_batch.attn_cp_metadata is not None
        )
        shard = (
            _get_indexer_query_shard(forward_batch, bs)
            if is_prefill and _shard_indexer_queries() and not uses_prefill_cp
            else None
        )
        q_sliced_early = shard is not None and self.rotary_emb.is_neox_style
        if q_sliced_early:
            if dynamic_scale is not None:
                dynamic_scale = shard.take(dynamic_scale)
            q_lora = shard.take(q_lora)
            bs_q = shard.rows
        else:
            bs_q = bs

        if self.rotary_emb.is_neox_style:
            if not hasattr(forward_batch, "npu_indexer_sin_cos_cache"):
                cos_sin = self.rotary_emb.cos_sin_cache[positions]
                cos, sin = cos_sin.chunk(2, dim=-1)
                cos = cos.repeat(1, 2).view(-1, 1, 1, self.rope_head_dim)
                sin = sin.repeat(1, 2).view(-1, 1, 1, self.rope_head_dim)
                forward_batch.npu_indexer_sin_cos_cache = (sin, cos)
            else:
                sin, cos = forward_batch.npu_indexer_sin_cos_cache

            # q rotates on this rank's rows; k still needs every one of them.
            sin_q, cos_q = (
                (shard.take(sin), shard.take(cos)) if q_sliced_early else (sin, cos)
            )

            if self.alt_stream is not None:
                self.alt_stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(self.alt_stream):
                    q_lora = (
                        (q_lora, dynamic_scale) if dynamic_scale is not None else q_lora
                    )
                    q = self.wq_b(q_lora)[
                        0
                    ]  # [rows, 1536] @ [1536, 64 * 128] = [rows, 64 * 128]
                    q = q.view(bs_q, self.n_heads, self.head_dim)  # [rows, 64, 128]
                    q_pe, q_nope = torch.split(
                        q,
                        [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                        dim=-1,
                    )  # [rows, 64, 64 + 64]
                    q_pe = q_pe.view(bs_q, self.n_heads, 1, self.rope_head_dim)
                    q_pe = torch_npu.npu_rotary_mul(q_pe, cos_q, sin_q).view(
                        bs_q, self.n_heads, self.rope_head_dim
                    )  # [rows, n, d]
                    q = torch.cat([q_pe, q_nope], dim=-1)
                    q.record_stream(self.alt_stream)
                    q_rope_event = self.alt_stream.record_event()
            else:
                q_lora = (
                    (q_lora, dynamic_scale) if dynamic_scale is not None else q_lora
                )
                q = self.wq_b(q_lora)[
                    0
                ]  # [rows, 1536] @ [1536, 64 * 128] = [rows, 64 * 128]
                q = q.view(bs_q, self.n_heads, self.head_dim)  # [rows, 64, 128]
                q_pe, q_nope = torch.split(
                    q,
                    [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                    dim=-1,
                )  # [rows, 64, 64 + 64]
                q_pe = q_pe.view(bs_q, self.n_heads, 1, self.rope_head_dim)
                q_pe = torch_npu.npu_rotary_mul(q_pe, cos_q, sin_q).view(
                    bs_q, self.n_heads, self.rope_head_dim
                )  # [rows, n, d]
                q = torch.cat([q_pe, q_nope], dim=-1)

            if envs.SGLANG_NPU_USE_MULTI_STREAM.get():
                indexer_weight_stream = get_indexer_weight_stream()
                indexer_weight_stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(indexer_weight_stream):
                    x = x.view(-1, self.hidden_size)
                    weights = self.weights_proj(x.float())[0].to(torch.bfloat16)
                    weights.record_stream(indexer_weight_stream)
                    weights_event = indexer_weight_stream.record_event()
            else:
                x = x.view(-1, self.hidden_size)
                weights = self.weights_proj(x.float())[0].to(torch.bfloat16)

            k_proj = self.wk(x)[0]  # [b, s, 7168] @ [7168, 128] = [b, s, 128]
            k = self.k_norm(k_proj)
            if _use_ag_after_qlora and input_on_attn_tp_slices:
                k = scattered_to_tp_attn_full(k, forward_batch)
            k_pe, k_nope = torch.split(
                k,
                [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                dim=-1,
            )  # [bs, 64 + 64]

            k_pe = k_pe.view(-1, 1, 1, self.rope_head_dim)
            k_pe = torch.ops.npu.npu_rotary_mul(k_pe, cos, sin).view(
                bs, 1, self.rope_head_dim
            )  # [bs, 1, d]
            k = torch.cat([k_pe, k_nope.unsqueeze(1)], dim=-1)  # [bs, 1, 128]

        else:
            if envs.SGLANG_NPU_USE_MULTI_STREAM.get():
                indexer_weight_stream = get_indexer_weight_stream()
                indexer_weight_stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(indexer_weight_stream):
                    x = x.view(-1, self.hidden_size)
                    weights = self.weights_proj(x.float())[0].to(torch.bfloat16)
                    weights.record_stream(indexer_weight_stream)
                    weights_event = indexer_weight_stream.record_event()
            else:
                x = x.view(-1, self.hidden_size)
                weights = self.weights_proj(x.float())[0].to(torch.bfloat16)

            q_lora = (q_lora, dynamic_scale) if dynamic_scale is not None else q_lora
            q = self.wq_b(q_lora)[0]  # [bs, 1536] @ [1536, 64 * 128] = [bs, 64 * 128]
            q = q.view(bs, self.n_heads, self.head_dim)  # [bs, 64, 128]
            q_pe, q_nope = torch.split(
                q,
                [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                dim=-1,
            )  # [bs, 64, 64 + 64]

            k_proj = self.wk(x)[0]  # [b, s, 7168] @ [7168, 128] = [b, s, 128]
            k = self.k_norm(k_proj)
            k_pe, k_nope = torch.split(
                k,
                [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                dim=-1,
            )  # [bs, 64 + 64]

            k_pe = k_pe.unsqueeze(1)

            if layer_id == get_token_to_kv_pool().start_layer:
                self.rotary_emb.sin_cos_cache = (
                    self.rotary_emb.cos_sin_cache.index_select(0, positions)
                )

            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
            k_pe = k_pe.squeeze(1)
            q = torch.cat([q_pe, q_nope], dim=-1)
            k = torch.cat([k_pe, k_nope], dim=-1)

        if (
            is_prefill
            and self.dsa_enable_prefill_cp
            and forward_batch.attn_cp_metadata is not None
        ):
            k = cp_gather_full_sequence_states(
                k.contiguous().view(-1, self.head_dim),
                forward_batch,
                torch.npu.current_stream(),
            )

        indexer_cache_loc = forward_batch.out_cache_loc
        parallel = get_parallel()
        if (
            parallel.dcp_enabled
            and parallel.attn_dcp_size > 1
            and not get_attn_backend().is_draft_worker
        ):
            indexer_cache_loc = (
                get_attn_backend().forward_metadata.dcp_origin_out_cache_loc
            )
            assert indexer_cache_loc is not None, (
                "NPU DSA+DCP requires allocator-global origin_out_cache_loc metadata"
            )
            assert indexer_cache_loc.shape[0] == positions.shape[0], (
                "NPU DSA+DCP origin_out_cache_loc metadata has an incompatible "
                f"length: {indexer_cache_loc.shape[0]} != {positions.shape[0]}"
            )
        pool = get_token_to_kv_pool()
        use_quant_indexer = pool.index_k_scale_buffer is not None
        if use_quant_indexer:
            _check_quant_lightning_indexer_constraints(pool)
            k, k_scale = _quantize_npu_indexer_activation(
                k, pool.indexer_hadamard_128, pool.dtype
            )
            pool.set_index_k_scale_buffer(layer_id, indexer_cache_loc, k_scale)
        pool.set_index_k_buffer(layer_id, indexer_cache_loc, k)
        if is_prefill:
            if (
                self.dsa_enable_prefill_cp
                and forward_batch.attn_cp_metadata is not None
            ):
                get_attn_backend().forward_metadata.actual_seq_lengths_q = (
                    forward_batch.attn_cp_metadata.actual_seq_q_prev_tensor,
                    forward_batch.attn_cp_metadata.actual_seq_q_next_tensor,
                )
                if sum(forward_batch.extend_prefix_lens_cpu) > 0:
                    total_kv_len_prev_tensor = (
                        forward_batch.attn_cp_metadata.kv_len_prev_tensor
                        + forward_batch.extend_prefix_lens.squeeze()
                    )
                    total_kv_len_next_tensor = (
                        forward_batch.attn_cp_metadata.kv_len_next_tensor
                        + forward_batch.extend_prefix_lens.squeeze()
                    )
                    get_attn_backend().forward_metadata.actual_seq_lengths_kv = (
                        total_kv_len_prev_tensor,
                        total_kv_len_next_tensor,
                    )
                else:
                    get_attn_backend().forward_metadata.actual_seq_lengths_kv = (
                        forward_batch.attn_cp_metadata.kv_len_prev_tensor,
                        forward_batch.attn_cp_metadata.kv_len_next_tensor,
                    )
                actual_seq_lengths_q = (
                    get_attn_backend().forward_metadata.actual_seq_lengths_q
                )
                actual_seq_lengths_kv = (
                    get_attn_backend().forward_metadata.actual_seq_lengths_kv
                )
            else:
                actual_seq_lengths_kv = forward_batch.seq_lens
                actual_seq_lengths_q = forward_batch.extend_seq_lens.cumsum(dim=0)
        else:
            if get_attn_backend().forward_metadata.actual_seq_lengths_q is None:
                if (
                    forward_batch.forward_mode.is_draft_extend_v2()
                    or forward_batch.forward_mode.is_target_verify()
                ):
                    num_draft_tokens = get_attn_backend().speculative_num_draft_tokens
                    actual_seq_lengths_q = torch.arange(
                        num_draft_tokens,
                        num_draft_tokens + bs,
                        num_draft_tokens,
                        dtype=torch.int32,
                        device=k.device,
                    )
                else:
                    actual_seq_lengths_q = torch.tensor(
                        [1 + i * 1 for i in range(bs)],
                        dtype=torch.int32,
                        device=k.device,
                    )
            else:
                actual_seq_lengths_q = (
                    get_attn_backend().forward_metadata.actual_seq_lengths_q
                )

        past_key_states = get_token_to_kv_pool().get_index_k_buffer(layer_id)

        if self.rotary_emb.is_neox_style and self.alt_stream is not None:
            torch.npu.current_stream().wait_event(q_rope_event)
        if envs.SGLANG_NPU_USE_MULTI_STREAM.get():
            torch.npu.current_stream().wait_event(weights_event)
        if _use_ag_after_qlora and input_on_attn_tp_slices:
            weights = scattered_to_tp_attn_full(weights, forward_batch)
        block_table = get_attn_backend().forward_metadata.block_tables
        if (
            is_prefill
            and self.dsa_enable_prefill_cp
            and forward_batch.attn_cp_metadata is not None
        ):
            block_table = block_table[: actual_seq_lengths_q[0].numel()]
            topk_indices = self.do_npu_cp_balance_indexer(
                q.view(-1, self.n_heads, self.head_dim),
                past_key_states,
                weights,
                actual_seq_lengths_q,
                actual_seq_lengths_kv,
                block_table,
            )
            return topk_indices
        else:
            block_table = (
                block_table[: actual_seq_lengths_q.size()[0]]
                if is_prefill
                else block_table
            )
            query = q.view(-1, self.n_heads, self.head_dim)
            # The full width for the gather, NOT query.shape[0] (already cut).
            num_query_tokens = bs
            if shard is not None:
                if not q_sliced_early:
                    query = shard.take(query)
                # A late take: under _use_ag_after_qlora weights is gathered
                # to full width above, so an earlier slice would cut twice.
                weights = shard.take(weights)
                actual_seq_lengths_q = shard.actual_seq_lengths_q
                actual_seq_lengths_kv = shard.actual_seq_lengths_kv

            if use_quant_indexer:
                query, query_scale = _quantize_npu_indexer_activation(
                    query,
                    pool.indexer_hadamard_128,
                    pool.dtype,
                )
                # quant_lightning_indexer (v2) contract, quant_mode 3 (MXFP8):
                #  - layout_q TND: q (q_t, q_n, d) fp8, cu_seqlens_q (b+1,) int32
                #    required (first value 0, last value q_t)
                #  - layout_k PA_BBND: k (block_num, block_size, k_n, d) fp8 with
                #    block_table (b, max_blocks) and seqused_k (b,) both required
                #  - descales are E8M0: q (q_t, q_n, d/64, 2),
                #    k (block_num, block_size, k_n, d/64, 2)
                #  - w is float32 (q_t, q_n)
                # The metadata op (task list / load balancing) is pre-planned
                # once per batch by AscendAttnBackend.init_forward_metadata;
                # fall back to computing it inline when the backend did not
                # pre-plan it (cuda-graph capture, other backends, CP, spec
                # draft model), and ALWAYS re-plan when this call was sharded.
                #
                # The pre-plan describes the whole batch. A sharded call was
                # handed 1/attn_tp_size of the query rows, so the batch-wide
                # cu_seqlens_q ends at the batch's token total and the operator
                # would walk attn_tp_size times past the end of `query`,
                # `weights` and `query_scale`. Nothing on the host catches it:
                # the lengths are device tensors, so the tiling cannot check
                # them against the query shape, and the kernel reports the
                # out-of-bounds read as an AICore trap from inside
                # quant_lightning_indexer. It is a write too: v2's infershape
                # sizes sparse_indices from queryShape.GetDim(0), not from
                # cu_seqlens_q.
                #
                # That same rule is what lets shard.gather() work here: the
                # output has one row per row of `query`, so a sharded call
                # returns shard.rows rows and the all-gather widths line up.
                # Rows past cu_seqlens_q[-1] are untouched, and gather() slices
                # them off.
                _fm = get_attn_backend().forward_metadata
                if shard is None:
                    cu_seqlens_q = getattr(_fm, "quant_indexer_cu_seqlens_q", None)
                    seqused_k = getattr(_fm, "quant_indexer_seqused_k", None)
                    metadata = getattr(_fm, "quant_indexer_metadata", None)
                    if metadata is None:
                        cu_seqlens_q, seqused_k, metadata = (
                            self._plan_quant_lightning_indexer(
                                actual_seq_lengths_q, actual_seq_lengths_kv, k.device
                            )
                        )
                else:
                    # Planned on this forward's first indexer layer and reused
                    # by the rest, which is what the backend's pre-plan buys.
                    if shard.quant_indexer_plan is None:
                        shard.quant_indexer_plan = self._plan_quant_lightning_indexer(
                            shard.actual_seq_lengths_q,
                            shard.actual_seq_lengths_kv,
                            k.device,
                        )
                    cu_seqlens_q, seqused_k, metadata = shard.quant_indexer_plan
                block_table = block_table.to(torch.int32)
                topk_indices, _ = (
                    torch.ops.cann_ops_transformer.quant_lightning_indexer(
                        query,
                        past_key_states,
                        weights.to(torch.float32),
                        query_scale,
                        pool.get_index_k_scale_buffer(layer_id),
                        self.index_topk,
                        3,  # QUANT_MODE_MXFP8   5, # QUANT_MODE_MXFP4,
                        cu_seqlens_q=cu_seqlens_q,
                        seqused_k=seqused_k,
                        block_table=block_table,
                        metadata=metadata,
                        max_seqlen_q=-1,
                        layout_q="TND",
                        layout_k="PA_BBND",
                        mask_mode=3,
                        cmp_ratio=1,
                    )
                )
                topk_indices = topk_indices.squeeze(1)
            else:
                topk_indices = torch_npu.npu_lightning_indexer(
                    query=query,
                    key=past_key_states,
                    weights=weights,
                    actual_seq_lengths_query=actual_seq_lengths_q.to(torch.int32),
                    actual_seq_lengths_key=actual_seq_lengths_kv.to(k.device).to(
                        torch.int32
                    ),
                    block_table=block_table,
                    layout_query="TND",
                    layout_key="PA_BSND",
                    sparse_count=self.index_topk,
                    sparse_mode=3,
                )[0].squeeze(1)
            if shard is not None:
                topk_indices = shard.resolve(
                    topk_indices, num_query_tokens, forward_batch
                )
            else:
                forward_batch.npu_indexer_topk_is_local = False
            # Keep DSA top-k as [T, K]; NPU attention expands it when needed.
            return topk_indices

    def do_npu_cp_balance_indexer(
        self,
        q,
        past_key_states,
        indexer_weights,
        actual_seq_lengths_q,
        actual_seq_lengths_kv,
        block_table,
    ):
        q_prev, q_next = torch.split(q, (q.size(0) + 1) // 2, dim=0)
        weights_prev, weights_next = None, None
        if indexer_weights is not None:
            weights_prev, weights_next = torch.split(
                indexer_weights, (indexer_weights.size(0) + 1) // 2, dim=0
            )
            weights_prev = weights_prev.contiguous().view(-1, weights_prev.shape[-1])
            weights_next = weights_next.contiguous().view(-1, weights_next.shape[-1])

        actual_seq_lengths_q_prev, actual_seq_lengths_q_next = actual_seq_lengths_q
        actual_seq_lengths_kv_prev, actual_seq_lengths_kv_next = actual_seq_lengths_kv

        topk_indices_prev = torch_npu.npu_lightning_indexer(
            query=q_prev,
            key=past_key_states,
            weights=weights_prev,
            actual_seq_lengths_query=actual_seq_lengths_q_prev.to(
                device=q.device, dtype=torch.int32
            ),
            actual_seq_lengths_key=actual_seq_lengths_kv_prev.to(
                device=q.device, dtype=torch.int32
            ),
            block_table=block_table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=self.index_topk,
            sparse_mode=3,
        )
        topk_indices_next = torch_npu.npu_lightning_indexer(
            query=q_next,
            key=past_key_states,
            weights=weights_next,
            actual_seq_lengths_query=actual_seq_lengths_q_next.to(
                device=q.device, dtype=torch.int32
            ),
            actual_seq_lengths_key=actual_seq_lengths_kv_next.to(
                device=q.device, dtype=torch.int32
            ),
            block_table=block_table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=self.index_topk,
            sparse_mode=3,
        )
        return torch.cat([topk_indices_prev[0], topk_indices_next[0]], dim=0).squeeze(1)


def scattered_to_tp_attn_full(
    hidden_states: torch.Tensor,
    forward_batch,
) -> torch.Tensor:
    hidden_states, local_hidden_states = (
        torch.empty(
            (forward_batch.input_ids.shape[0], hidden_states.shape[1]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        ),
        hidden_states,
    )
    attn_tp_all_gather_into_tensor(hidden_states, local_hidden_states.contiguous())
    return hidden_states
