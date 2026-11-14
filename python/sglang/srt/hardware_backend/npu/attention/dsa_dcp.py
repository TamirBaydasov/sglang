"""NPU DSA adaptation for token-sharded decode context parallelism."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch_npu

from sglang.srt.hardware_backend.npu.attention.mla_preprocess import is_fia_nz

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def forward_dcp_sparse_attention(
    *,
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    topk_indices: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    forward_metadata,
    forward_batch: ForwardBatch,
    speculative_num_draft_tokens: int | None,
    scaling: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute one rank's sparse partial attention and natural-log LSE."""

    is_target_verify = forward_batch.forward_mode.is_target_verify()
    if is_target_verify:
        block_tables = forward_metadata.dcp_spec_block_tables
        local_lens = forward_metadata.dcp_spec_seq_lens
    else:
        block_tables = forward_metadata.dcp_block_tables
        local_lens = forward_metadata.dcp_seq_lens

    assert block_tables is not None, (
        "NPU DSA+DCP sparse attention requires rank-local paged-KV metadata"
    )
    assert block_tables.shape[0] == q_nope.shape[0], (
        "NPU DSA+DCP block-table rows must match query rows"
    )
    assert local_lens.shape[0] == q_nope.shape[0], (
        "NPU DSA+DCP KV-length rows must match query rows"
    )
    indices = topk_indices.reshape(q_nope.shape[0], -1).long()
    valid = (indices >= 0) & (indices < local_lens[:, None])
    safe_indices = torch.where(valid, indices, 0)
    page_size = k_nope.shape[1]
    pages = block_tables.long().gather(1, safe_indices // page_size)
    offsets = safe_indices % page_size

    def gather(cache):
        if is_fia_nz():
            # NZ's public shape hides a physical [block, D/16, page, 16] layout.
            physical = cache.view(cache.shape[0], cache.shape[-1] // 16, page_size, 16)
            rows = physical[pages, :, offsets, :]
        else:
            rows = cache[pages, offsets]
        return rows.reshape(-1, 1, cache.shape[-1]).contiguous()

    return forward_tnd_sparse_attention(
        q_nope=q_nope,
        q_rope=q_rope,
        k_nope=gather(k_nope),
        k_rope=gather(k_rope),
        valid=valid,
        scaling=scaling,
    )


def forward_tnd_sparse_attention(*, q_nope, q_rope, k_nope, k_rope, valid, scaling):
    """One TND sequence per query; empty ranks contribute zero and -inf LSE."""
    if torch.npu.is_current_stream_capturing():
        raise RuntimeError(
            "Sparse attention LSE requires eager execution; disable decode graphs"
        )
    active = valid.any(dim=-1)
    output = torch.zeros_like(q_nope)
    lse = torch.full(
        q_nope.shape[:2], -float("inf"), device=q_nope.device, dtype=torch.float32
    )
    if not active.any():
        return output, lse
    width = valid.shape[-1]
    # CANN requires valid sparse indices before invalid entries, including after DCP masking.
    valid = valid[active]
    order = torch.argsort((~valid).to(torch.int32), dim=-1, stable=True)
    indices = (
        torch.where(valid.gather(1, order), order, -1).to(torch.int32).unsqueeze(1)
    )
    keys = (
        k_nope.reshape(-1, width, 1, k_nope.shape[-1])[active]
        .flatten(0, 1)
        .contiguous()
    )
    ropes = (
        k_rope.reshape(-1, width, 1, k_rope.shape[-1])[active]
        .flatten(0, 1)
        .contiguous()
    )
    query_lens = torch.arange(
        1, keys.shape[0] // width + 1, device=q_nope.device, dtype=torch.int32
    )
    result, softmax_max, softmax_sum = torch_npu.npu_sparse_flash_attention(
        query=q_nope[active].contiguous(),
        key=keys,
        value=keys,
        query_rope=q_rope[active].contiguous(),
        key_rope=ropes,
        sparse_indices=indices,
        scale_value=scaling,
        actual_seq_lengths_query=query_lens,
        actual_seq_lengths_kv=query_lens * width,
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="TND",
        sparse_mode=0,
        attention_mode=2,
        return_softmax_lse=True,
    )
    output[active] = result
    partial_lse = softmax_max.float() + torch.log(softmax_sum.float())
    lse[active] = partial_lse.permute(1, 0, 2).reshape(result.shape[:2])
    return output, lse
