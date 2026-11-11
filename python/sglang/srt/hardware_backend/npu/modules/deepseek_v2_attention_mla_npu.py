import logging
from typing import TYPE_CHECKING, Dict, NamedTuple, Optional, Tuple

import torch
import torch_npu
from sgl_kernel_npu.norm.fused_split_qk_norm import fused_split_qk_norm

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.attention.mla_preprocess import (
    NPUFusedMLAPreprocess,
    is_fia_nz,
    is_mla_preprocess_enabled,
)
from sglang.srt.hardware_backend.npu.utils import is_npu_arch35
from sglang.srt.layers.attention.dsa.dsa_npu_indexer import scattered_to_tp_attn_full
from sglang.srt.layers.attention.dsa.dsa_token_shard import (
    dsa_cp_redistribute_heads_async,
    dsa_cp_redistribute_heads_recv,
    dsa_cp_set_token_sharded,
    dsa_cp_token_sharded,
    dsa_cp_token_sharded_enabled,
    dsa_token_shard_narrow_a2a_enabled,
    dsa_token_shard_redistribute_heads,
    dsa_token_shard_restore_tokens,
    dsa_token_shard_slice,
    get_dsa_token_shard_plan,
    get_o_proj_stream_shard,
)
from sglang.srt.layers.attention.dsa.utils import (
    dsa_use_prefill_cp,
)
from sglang.srt.layers.dcp import (
    all_gather_q_for_mla_decode,
    cp_lse_ag_out_rs_mla_npu,
)
from sglang.srt.layers.dcp.layout import (
    DcpPackedReadPlan,
    dcp_extend_gather_buffer,
    dcp_interleave_size,
    dcp_packed_read_plan,
    plan_dcp_extend_gather,
    remap_dcp_sparse_indices,
    remap_topk_to_packed,
)
from sglang.srt.layers.layer_boundary import get_attn_tp_context
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_token_to_kv_pool,
)
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla import (
    is_dcp_mla_decode_phase,
)
from sglang.srt.runtime_context import get_disagg, get_parallel
from sglang.srt.state_capturer.indexer_topk import maybe_capture_indexer_topk
from sglang.srt.utils.common import print_info_once

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
    from sglang.srt.utils import BumpAllocator

logger = logging.getLogger(__name__)

_use_ag_after_qlora = envs.SGLANG_USE_AG_AFTER_QLORA.get()
_is_npu_arch35 = is_npu_arch35()
_debug_dcp_extend_memory = envs.SGLANG_DEBUG_NPU_DCP_EXTEND_MEMORY.get()
_prefetch_dcp_extend_gather = envs.SGLANG_NPU_ENABLE_DCP_EXTEND_GATHER_PREFETCH.get()
_dcp_gather_prefetch_stream = None


def _get_dcp_gather_prefetch_stream():
    """One side stream per process, built lazily: the module is imported
    before the device is selected, and a stream made on the wrong device
    would run the gathers where no rank is looking."""
    global _dcp_gather_prefetch_stream
    if _dcp_gather_prefetch_stream is None:
        _dcp_gather_prefetch_stream = torch.npu.Stream()
    return _dcp_gather_prefetch_stream


# region MHA
def forward_mha_prepare_npu(
    m: "DeepseekV2AttentionMLA",
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch: "ForwardBatch",
    zero_allocator: "BumpAllocator",
    input_on_attn_tp_slices: bool,
):
    if m.q_lora_rank is not None:
        q, latent_cache = (
            get_attn_tp_context()
            .fetch_qkv_latent()
            .split(
                [m.q_lora_rank, m.kv_lora_rank + m.qk_rope_head_dim],
                dim=-1,
            )
        )

        # DSA Indexer: cache quantized keys, auto-skip topk for sequences <= dsa_index_topk

        if m.use_dsa:
            q_lora = m.q_a_layernorm(q)
            q = m.q_b_proj(q_lora)[0].view(-1, m.num_local_heads, m.qk_head_dim)
            _ = m.indexer(
                x=hidden_states,
                q_lora=q_lora,
                positions=positions,
                forward_batch=forward_batch,
                layer_id=m.layer_id,
                return_indices=False,
            )

        else:
            q = m.q_a_layernorm(q)
            if _use_ag_after_qlora and input_on_attn_tp_slices:
                q = scattered_to_tp_attn_full(q, forward_batch)
                latent_cache = scattered_to_tp_attn_full(latent_cache, forward_batch)
            q = m.q_b_proj(q)[0].view(-1, m.num_local_heads, m.qk_head_dim)

    else:
        q = m.q_proj(hidden_states)[0].view(-1, m.num_local_heads, m.qk_head_dim)
        latent_cache = m.kv_a_proj_with_mqa(hidden_states)[0]

    _, q_pe = q.split([m.qk_nope_head_dim, m.qk_rope_head_dim], dim=-1)
    kv_a, _ = latent_cache.split([m.kv_lora_rank, m.qk_rope_head_dim], dim=-1)
    latent_cache = latent_cache.unsqueeze(1)

    if m.use_deepseek_yarn_rope:
        B, S = q.shape[0], 1
        cos, sin = m.rotary_emb.get_cos_sin_cache(
            positions, hidden_states.dtype, offsets=None
        )
        q_pe = torch_npu.npu_interleave_rope(
            q_pe.reshape(B, -1, S, m.qk_rope_head_dim),
            cos,
            sin,
        )
        q_pe = q_pe.reshape(B, -1, m.qk_rope_head_dim)

        ckv_cache, k_rope_cache = get_token_to_kv_pool().get_kv_buffer(m.layer_id)
        _, _, k_pe, kv_a = torch_npu.npu_kv_rmsnorm_rope_cache(
            latent_cache.view(-1, 1, 1, m.kv_lora_rank + m.qk_rope_head_dim),  # bnsd
            m.kv_a_layernorm.weight,
            cos,
            sin,
            forward_batch.out_cache_loc.to(torch.int64),
            k_rope_cache,
            ckv_cache,
            k_rope_scale=None,
            c_kv_scale=None,
            k_rope_offset=None,
            c_kv_offset=None,
            epsilon=m.kv_a_layernorm.variance_epsilon,
            cache_mode="PA_NZ" if is_fia_nz() else "PA_BNSD",
            is_output_kv=True,
        )  # adapter NZ

        k_pe = k_pe.reshape(B, -1, m.qk_rope_head_dim)
    else:
        kv_a = m.kv_a_layernorm(kv_a)
        k_pe = latent_cache[:, :, m.kv_lora_rank :]
        if m.rotary_emb is not None:
            q_pe, k_pe = m.rotary_emb(positions, q_pe, k_pe)
        # this is for model kimi-vl-a3B-instruct
        get_token_to_kv_pool().set_kv_buffer(
            m, forward_batch.out_cache_loc, kv_a.unsqueeze(1), k_pe
        )

    q[..., m.qk_nope_head_dim :] = q_pe

    kv = m.kv_b_proj(kv_a)[0]
    kv = kv.view(-1, m.num_local_heads, m.qk_nope_head_dim + m.v_head_dim)
    k_nope = kv[..., : m.qk_nope_head_dim]
    v = kv[..., m.qk_nope_head_dim :]

    k = m._concat_and_cast_mha_k(k_nope, k_pe, forward_batch)
    return q, k, v, forward_batch


def forward_mha_core_npu(
    m: "DeepseekV2AttentionMLA",
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    forward_batch: "ForwardBatch",
    # Gated attention (Ling-V3 / BailingMoeV3): the subclass appends its gate
    # to inner_state, so every *_core dispatched from forward_core takes it as
    # a trailing arg. None everywhere else.
    gate: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    attn_output = m.attn_mha(q, k, v, forward_batch, save_kv_cache=False)
    attn_output = attn_output.reshape(-1, m.num_local_heads * m.v_head_dim)
    if gate is not None:
        attn_output = m._apply_gated(attn_output, gate)
    output, _ = m.o_proj(attn_output)
    return output


# endregion


# region MLA
def forward_mla_prepare_npu(
    m: "DeepseekV2AttentionMLA",
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch: "ForwardBatch",
    zero_allocator: "BumpAllocator",
    input_on_attn_tp_slices: bool,
):
    if is_mla_preprocess_enabled():
        if not hasattr(m, "mla_preprocess"):
            m.mla_preprocess = NPUFusedMLAPreprocess(
                m.fused_qkv_a_proj_with_mqa,
                m.q_a_layernorm,
                m.kv_a_layernorm,
                m.q_b_proj,
                m.w_kc,
                m.rotary_emb,
                m.layer_id,
                m.num_local_heads,
                m.qk_nope_head_dim,
                m.qk_rope_head_dim,
                m.v_head_dim,
            )
        outputs = m.mla_preprocess.forward(
            positions, hidden_states, forward_batch, zero_allocator
        )
        q_pe, k_pe, q_nope_out, k_nope = outputs[:4]
        topk_indices = None
    else:
        q_lora = None
        if m.q_lora_rank is not None:
            qkv_latent = get_attn_tp_context().fetch_qkv_latent()
            if _use_ag_after_qlora and input_on_attn_tp_slices:
                q, latent_cache = qkv_latent.split(
                    [m.q_lora_rank, m.kv_lora_rank + m.qk_rope_head_dim],
                    dim=-1,
                )
                k_nope = latent_cache[..., : m.kv_lora_rank]

                q = m.q_a_layernorm(q)
                q = scattered_to_tp_attn_full(q, forward_batch)
                latent_cache = scattered_to_tp_attn_full(latent_cache, forward_batch)

                k_nope = m.kv_a_layernorm(k_nope).unsqueeze(1)
                k_pe = latent_cache[..., m.kv_lora_rank :].unsqueeze(1)
            else:
                if (
                    qkv_latent.shape[0] < 65536
                    and not dsa_use_prefill_cp(forward_batch)
                    and not getattr(m, "_disable_npu_fused_split_qk_norm", False)
                ):
                    q, k_nope, k_pe = fused_split_qk_norm(
                        qkv_latent,
                        m.q_a_layernorm,
                        m.kv_a_layernorm,
                        m.q_lora_rank,
                        m.kv_lora_rank,
                        m.qk_rope_head_dim,
                        eps=m.q_a_layernorm.variance_epsilon,
                    )
                else:
                    # The fused split+RMSNorm kernel is not numerically equivalent
                    # on Ascend. Keep the unfused path for models that opt out.
                    q, latent_cache = qkv_latent.split(
                        [m.q_lora_rank, m.kv_lora_rank + m.qk_rope_head_dim],
                        dim=-1,
                    )
                    k_nope = latent_cache[..., : m.kv_lora_rank]

                    q = m.q_a_layernorm(q)

                    k_nope = m.kv_a_layernorm(k_nope).unsqueeze(1)
                    k_pe = latent_cache[..., m.kv_lora_rank :].unsqueeze(1)

            # q_lora needed by indexer
            if m.use_dsa:
                q_lora = q

            q = m.q_b_proj(q)[0].view(-1, m.num_local_heads, m.qk_head_dim)
        else:
            q = m.q_proj(hidden_states)[0].view(-1, m.num_local_heads, m.qk_head_dim)
            latent_cache = m.kv_a_proj_with_mqa(hidden_states)[0]
            k_nope = latent_cache[..., : m.kv_lora_rank]
            k_nope = m.kv_a_layernorm(k_nope).unsqueeze(1)
            k_pe = latent_cache[..., m.kv_lora_rank :].unsqueeze(1)

        q_nope, q_pe = q.split([m.qk_nope_head_dim, m.qk_rope_head_dim], dim=-1)

        q_nope_out = torch.bmm(q_nope.transpose(0, 1), m.w_kc)

        q_nope_out = q_nope_out.transpose(0, 1)

        if m.rotary_emb is not None:
            q_pe, k_pe = m.rotary_emb(positions, q_pe, k_pe)

        if dsa_use_prefill_cp(forward_batch):
            # support allgather+rerrange
            k_nope, k_pe = m.rebuild_cp_kv_cache(
                latent_cache, forward_batch, k_nope, k_pe
            )
        topk_indices = None
        if q_lora is not None:
            topk_indices = m.indexer(
                x=hidden_states,
                q_lora=q_lora,
                positions=positions,
                forward_batch=forward_batch,
                layer_id=m.layer_id,
            )

    return (
        q_pe,
        k_pe,
        q_nope_out,
        k_nope,
        forward_batch,
        zero_allocator,
        positions,
        topk_indices,
    )


def forward_mla_core_npu(
    m: "DeepseekV2AttentionMLA",
    q_pe: torch.Tensor,
    k_pe: torch.Tensor,
    q_nope_out: torch.Tensor,
    k_nope: torch.Tensor,
    forward_batch: "ForwardBatch",
    zero_allocator: "BumpAllocator",
    positions: torch.Tensor,
    topk_indices: torch.Tensor,
    # Gated attention (Ling-V3 / BailingMoeV3): the subclass appends its gate
    # to inner_state, so every *_core dispatched from forward_core takes it as
    # a trailing arg. None everywhere else.
    gate: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    attn_output = m.attn_mqa(
        q_nope_out,
        k_nope,
        k_nope,
        forward_batch,
        q_rope=q_pe,
        k_rope=k_pe,
        **(dict(topk_indices=topk_indices) if topk_indices is not None else {}),
    )

    attn_output = attn_output.view(-1, m.num_local_heads, m.kv_lora_rank)

    attn_output = attn_output.contiguous()
    if (
        attn_output.shape[0] >= 65536
        or attn_output.shape[-1] * attn_output.shape[-2] >= 65536
        or m.w_vc.shape[-1] >= 65536
    ):
        # npu_transpose_batchmatmul does not support dimensions >= 65536.
        attn_bmm_output = torch.empty(
            (attn_output.shape[0], m.num_local_heads, m.v_head_dim),
            dtype=attn_output.dtype,
            device=attn_output.device,
        )
        torch.ops.npu.batch_matmul_transpose(attn_output, m.w_vc, attn_bmm_output)
    else:
        # Use the numerically validated torch_npu implementation when supported.
        attn_bmm_output = torch_npu.npu_transpose_batchmatmul(
            attn_output,
            m.w_vc,
            perm_x1=(1, 0, 2),
            perm_x2=(0, 1, 2),
            perm_y=(1, 0, 2),
        )

    attn_bmm_output = attn_bmm_output.reshape(-1, m.num_local_heads * m.v_head_dim)
    if gate is not None:
        attn_bmm_output = m._apply_gated(attn_bmm_output, gate)
    output, _ = m.o_proj(attn_bmm_output)

    return output


# endregion


# region DSA
def _apply_interleaved_rope_with_half_output(rotary_emb, positions, q_pe, k_pe):
    """Apply RoPE to interleaved Q/K and return half-layout outputs."""
    rotary_emb.get_cos_sin_with_position(positions)
    cos = rotary_emb.position_cos.to(device=q_pe.device, dtype=q_pe.dtype).view(
        -1, 1, 1, q_pe.shape[-1]
    )
    sin = rotary_emb.position_sin.to(device=q_pe.device, dtype=q_pe.dtype).view(
        -1, 1, 1, q_pe.shape[-1]
    )
    q_pe = torch_npu.npu_interleave_rope(q_pe.unsqueeze(2), cos, sin).squeeze(2)
    k_pe = torch_npu.npu_interleave_rope(k_pe.unsqueeze(2), cos, sin).squeeze(2)
    return q_pe, k_pe


def _dsa_token_shard_narrow_plan(m: "DeepseekV2AttentionMLA", forward_batch):
    """The token-shard plan when the narrow exchange applies, else None. Absent
    full weights (the loader declined) put the layer back on the wide path."""
    if getattr(forward_batch, "npu_dsa_cp_token_sharded", False):
        # Token sharded: the q is already full-head for this rank's rows, so the
        # narrow head-redistribution has nothing to redistribute (the same
        # reason the version-A pre-issue below is gated on not token_sharded).
        return None
    if not dsa_token_shard_narrow_a2a_enabled():
        return None
    if getattr(m, "w_kc_full", None) is None or getattr(m, "w_vc_full", None) is None:
        return None
    return get_dsa_token_shard_plan(forward_batch)


def forward_dsa_prepare_npu(
    m: "DeepseekV2AttentionMLA",
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch: "ForwardBatch",
    zero_allocator: "BumpAllocator",
    input_on_attn_tp_slices: bool,
    prev_topk_indices: torch.Tensor = None,
):
    dynamic_scale = None
    # Resolved here so the core can read the cached plan back. D1v3 captures the
    # plan (the base dropped it): the token-sharded rebuild and the version-B
    # full-width Q branch below all branch on it. The base's plan builder is the
    # 2-arg form (no layer_scatter_modes), which is sufficient -- the D1v3
    # conditions branch on the plan's presence, not on a scatter-mode object.
    dsa_cp_plan = get_dsa_token_shard_plan(
        forward_batch,
        m.indexer.index_topk if m.indexer is not None else None,
    )
    # D2p: prime the first DSA layer's prefix gather on the side stream, before
    # the QKV, so it runs under the QKV and the indexer instead of starting with
    # them. prime() is a no-op past the forward's first DSA layer, so this runs
    # on every prepare and only the first layer actually gathers. It needs the
    # packed-read plan, so it is a no-op (the permuting path takes over) for
    # multi-request / sub-index_topk extends. (The old ``skip_topk and
    # prev_topk_indices is not None and indexer is None`` guard here was dead:
    # prime acts only on the first DSA layer, where prev_topk_indices is None,
    # so the guard could never fire -- dropped.)
    if _enable_dcp_gather_prefetch and _dcp_extend_active(forward_batch):
        packed_plan = dcp_packed_read_plan(
            forward_batch, _dcp_sfa_gather_topk_width(m, prev_topk_indices)
        )
        if packed_plan is not None:
            _dcp_gather_prefetcher_get().prime(
                m,
                forward_batch.attn_dcp_metadata,
                packed_plan,
                hidden_states,
                m.layer_id,
            )
    # Whether this layer runs the indexer. The gather placement below and the
    # indexer call both branch on this, so the expression lives in one place.
    indexer_runs = not m.skip_topk or (m.is_nextn and prev_topk_indices is None)
    # D2v4: the streamed matrices carry their shard, not the static qrep, so
    # the "present" check follows which one the preparation stood.
    q_b_present = (
        getattr(m, "q_b_stream_shard", None) is not None
        if _stream_q_b
        else getattr(m, "q_b_proj_qrep_weight", None) is not None
    )
    w_kc_present = (
        getattr(m, "w_kc_stream_shard", None) is not None
        if _stream_w_kc
        else getattr(m, "w_kc_qrep", None) is not None
    )
    w_vc_present = (
        getattr(m, "w_vc_stream_shard", None) is not None
        if _stream_w_vc
        else getattr(m, "w_vc_qrep", None) is not None
    )
    # Resolved before the full-width / token-sharded gates below (they read it);
    # it depends only on the forward mode, so it can sit here.
    mla_preprocess_used = (
        is_mla_preprocess_enabled()
        and not forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed()
    )
    # Version B of the DSA-CP Q all-to-all removal: the gathered full-head
    # weights let this layer compute every head for this rank's tokens, so the
    # QKV and the w_kc matmul below take the full-width branch and the core
    # skips the head redistribution.
    q_full_width = (
        _enable_dsa_cp_q_full_width
        and dsa_cp_plan is not None
        and not mla_preprocess_used
        and q_b_present
    )
    # D1v3: the token-sharded rebuild. It needs the version B full-head Q
    # weights (which it reuses) plus the full w_vc / o_proj prepared by the
    # model runner. The residual input is sliced to this rank's rows here
    # (layer 0: right after the embedding allReduce; layers 1+: it already
    # arrived sliced), the per-token projections run on the slice, and the KV
    # latent and the index-K are gathered back to full width for the pool
    # writes (seam S1). The output side (restore a2a, O-proj allReduce, MoE
    # allReduce) is replaced in the core.
    if dsa_cp_token_sharded_enabled() and dsa_cp_plan is not None:
        from sglang.srt.model_executor.runner import get_is_capture_mode

        if get_is_capture_mode():
            raise RuntimeError(
                "SGLANG_NPU_DSA_CP_TOKEN_SHARDED is on and a graph capture is "
                "in progress: the rebuild's collectives depend on the batch's "
                "token grid and cannot be captured. Disable graph capture for "
                "extend (the box runs with it off) or turn the flag off. "
                "Decode capture does not reach this check."
            )
    # A forward that asks for INPUT logprobs needs the hidden state at every
    # new-token position, but the rebuild's logits tail only gathers the
    # last-token rows (that is what makes it cheap). So an input-logprob
    # forward must stay unsharded, while an OUTPUT-logprob-only forward is
    # fine: its next_token_logits come from exactly the gathered rows (the
    # scheduler turns them into the output top-logprobs). Mirror
    # LogitsMetadata.from_forward_batch so the two never drift; the flag is
    # per-batch, so every rank declines at the same layer.
    wants_input_logprobs = (
        forward_batch.forward_mode.is_extend()
        and forward_batch.return_logprob
        and not forward_batch.forward_mode.is_target_verify()
        and forward_batch.extend_seq_lens_cpu is not None
        and forward_batch.extend_logprob_start_lens_cpu is not None
        and any(
            extend_len - start_len > 0
            for extend_len, start_len in zip(
                forward_batch.extend_seq_lens_cpu,
                forward_batch.extend_logprob_start_lens_cpu,
            )
        )
    )
    # D2v3: the o_proj full matrix comes from the stream when
    # SGLANG_NPU_STREAM_O_PROJ is on (this rank's dequantized shard, registered
    # by layer id by the model runner) and from the static replica otherwise.
    o_proj_present = (
        get_o_proj_stream_shard(m.layer_id) is not None
        if _stream_o_proj
        else getattr(m, "o_proj_qrep_weight", None) is not None
    )
    token_sharded = (
        dsa_cp_token_sharded_enabled()
        and dsa_cp_plan is not None
        and not mla_preprocess_used
        and not wants_input_logprobs
        and q_b_present
        and w_kc_present
        and w_vc_present
        and o_proj_present
    )
    if dsa_cp_token_sharded_enabled() and dsa_cp_plan is not None:
        # The startup gate raised on an incomplete preparation, so reaching
        # here with the flag on, the plan built, and a weight missing is a
        # wiring bug, not a configuration.
        if (
            not q_b_present
            or not w_kc_present
            or not w_vc_present
            or not o_proj_present
        ):
            # D2v4: each matrix is independently streamed or static, so name
            # the specific missing attribute (its shard when streamed, its qrep
            # otherwise) rather than one suffix that assumed all-or-nothing.
            missing = [
                name + (" shard" if streamed else " qrep")
                for name, streamed, present in (
                    ("q_b", _stream_q_b, q_b_present),
                    ("w_kc", _stream_w_kc, w_kc_present),
                    ("w_vc", _stream_w_vc, w_vc_present),
                    ("o_proj", _stream_o_proj, o_proj_present),
                )
                if not present
            ]
            raise RuntimeError(
                "SGLANG_NPU_DSA_CP_TOKEN_SHARDED is on and the DSA-CP plan is "
                "built, but this layer's full-width weights are incomplete "
                f"({', '.join(missing)}); the model-runner preparation must "
                "have run before any forward"
            )
    dsa_cp_set_token_sharded(forward_batch, token_sharded)

    if token_sharded:
        print_info_once(
            "D1v3 token-sharded rebuild ENGAGED for this extend: the residual "
            "is sharded across the "
            f"{get_parallel().attn_tp_size} attention-TP ranks and every "
            "per-token projection runs on this rank's rows."
        )
        # The residual stream is token-sharded from here on: this rank owns
        # rows [local_start, local_end). The attention input is FULL width only
        # on layer 0 (the embedding); from layer 1 on the residual already
        # carries this rank's rows. Guard on the width, not the layer id.
        if hidden_states.shape[0] != dsa_cp_plan.rows:
            hidden_states = dsa_token_shard_slice(hidden_states, dsa_cp_plan)
        # positions is the per-forward full token-position vector for every
        # layer (rope and the indexer's cos_sin lookup index by it), so it
        # still needs the row slice on every layer, not just the first.
        if positions.shape[0] != dsa_cp_plan.rows:
            positions = dsa_token_shard_slice(positions, dsa_cp_plan)

    def _run_q_b(q_in: torch.Tensor) -> torch.Tensor:
        # The three ways to reach the head set. Token sharded (D1v3) runs the
        # full weight on the rows prepare already sliced (rows x 64 heads, the
        # same FLOPs as the head-sharded GEMM on all rows). Version B (D1v2_3b)
        # runs the full weight on every row and slices after the split. The
        # baseline runs this rank's head shard.
        if token_sharded or q_full_width:
            # D2v4: gather the full q_b on the fly when streamed; otherwise the
            # standing qrep (built by the version-B / token-sharded prep).
            w_qb = _dcp_q_b_weight(m) if _stream_q_b else m.q_b_proj_qrep_weight
            return torch.nn.functional.linear(q_in, w_qb).view(
                -1,
                m.num_local_heads * get_parallel().attn_tp_size,
                m.qk_head_dim,
            )
        return m.q_b_proj(q_in)[0].view(-1, m.num_local_heads, m.qk_head_dim)

    if _enable_dcp_sfa_gather_overlap and _dcp_extend_active(forward_batch):
        if not indexer_runs:
            # Reuse layer: no indexer to hide behind, so the QKV is the window
            # (the vLLM timing, one step earlier). The all-gather moves only
            # this layer's pool prefix rows, which were written by earlier
            # forwards, so the collective depends on nothing this layer
            # computes -- not even the QKV. Issue it here, before the QKV, and
            # the QKV projection runs underneath it. This chunk's own KV is
            # attached once the QKV has produced it, below.
            forward_batch.npu_dcp_sfa_gather_pending = _dcp_start_extend_gather_npu(
                m,
                forward_batch,
                hidden_states,
                _dcp_sfa_gather_topk_width(m, prev_topk_indices),
            )
        # Indexer layer: issued right before the indexer, below, where this
        # chunk's KV already exists and the gather's window is the indexer.
    if mla_preprocess_used:
        (
            q_pe,
            k_pe,
            q_nope_out,
            k_nope,
            q_lora,
            forward_batch,
            zero_allocator,
            positions,
            dynamic_scale,
        ) = npu_mla_preprocess(
            m,
            hidden_states,
            positions,
            forward_batch,
            zero_allocator,
        )
    else:
        fused_qkv_a_proj_out = m.fused_qkv_a_proj_with_mqa(hidden_states)[0]
        if m.rotary_emb.is_neox_style:
            q, latent_cache = fused_qkv_a_proj_out.split(
                [m.q_lora_rank, m.kv_lora_rank + m.qk_rope_head_dim], dim=-1
            )
            # overlap qk norm
            q = m.q_a_layernorm(q)
            if _use_ag_after_qlora and input_on_attn_tp_slices:
                q = scattered_to_tp_attn_full(q, forward_batch)
                latent_cache = scattered_to_tp_attn_full(latent_cache, forward_batch)
            q_lora = q.clone()  # required for topk_indices

            q_event = None
            if m.alt_stream is not None:
                m.alt_stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(m.alt_stream):
                    q = _run_q_b(q_lora)
                    # record q to ensure memory space will not be released
                    q.record_stream(m.alt_stream)
                    q_event = m.alt_stream.record_event()
            else:
                q = _run_q_b(q_lora)

            k_nope, k_pe = latent_cache.unsqueeze(1).split(
                [m.kv_lora_rank, m.qk_rope_head_dim], dim=-1
            )
            k_nope = m.kv_a_layernorm(k_nope)
            # main stream waits for the completion of the event on the alt stream to ensure data dependency is complete
            if q_event is not None:
                torch.npu.current_stream().wait_event(q_event)
        else:
            if (
                fused_qkv_a_proj_out.shape[0] < 65535
                and not dsa_use_prefill_cp(forward_batch)
                and not getattr(m, "_disable_npu_fused_split_qk_norm", False)
            ):
                q_lora, k_nope, k_pe = fused_split_qk_norm(
                    fused_qkv_a_proj_out,
                    m.q_a_layernorm,
                    m.kv_a_layernorm,
                    m.q_lora_rank,
                    m.kv_lora_rank,
                    m.qk_rope_head_dim,
                    eps=m.q_a_layernorm.variance_epsilon,
                )
            else:
                # Keep the numerically validated unfused path for models that
                # explicitly opt out of the fused split and RMSNorm kernel.
                q, latent_cache = fused_qkv_a_proj_out.split(
                    [m.q_lora_rank, m.kv_lora_rank + m.qk_rope_head_dim], dim=-1
                )
                # overlap qk norm
                q = m.q_a_layernorm(q)

                q_lora = q.clone()  # required for topk_indices
                k_nope, k_pe = latent_cache.unsqueeze(1).split(
                    [m.kv_lora_rank, m.qk_rope_head_dim], dim=-1
                )
                k_nope = m.kv_a_layernorm(k_nope)
            q = _run_q_b(q_lora)

        q_nope, q_pe = q.split([m.qk_nope_head_dim, m.qk_rope_head_dim], dim=-1)

        if q_full_width and not token_sharded:
            # Slice to this rank's rows BEFORE the w_kc batch-matmul: the tmm
            # then costs rows x 64 heads, the same FLOPs as the sharded path's
            # total x 8, and the write it would otherwise make drops sharply at
            # long chunks. dsa_cp_slice pads the tail to plan.rows, the same
            # width the head redistribution would produce. (Token sharded
            # already sliced q_lora before q_b_proj, so its q is at rows x all
            # heads here.)
            q_nope = dsa_token_shard_slice(q_nope, dsa_cp_plan)

        if q_full_width or token_sharded:
            # D2v4: gather the full w_kc on the fly when streamed; otherwise the
            # standing w_kc_qrep.
            w_kc = _dcp_w_kc_weight(m) if _stream_w_kc else m.w_kc_qrep
        else:
            w_kc = m.w_kc
        # Narrow exchange: the absorb is DEFERRED to after the all-to-all below,
        # so q_nope crosses the wire at 192 wide rather than 512.
        narrow_plan = _dsa_token_shard_narrow_plan(m, forward_batch)
        if narrow_plan is None:
            q_nope_out = torch_npu.npu_transpose_batchmatmul(
                q_nope,
                w_kc,
                perm_x1=(1, 0, 2),
                perm_x2=(0, 1, 2),
                perm_y=(1, 0, 2),
            )

        if is_mla_preprocess_enabled() and not m.rotary_emb.is_neox_style:
            # Match the half-layout RoPE outputs used by MLA preprocessing.
            q_pe, k_pe = _apply_interleaved_rope_with_half_output(
                m.rotary_emb, positions, q_pe, k_pe
            )
        else:
            if m.layer_id == get_token_to_kv_pool().start_layer:
                m.rotary_emb.sin_cos_cache = m.rotary_emb.cos_sin_cache.index_select(
                    0, positions
                )
            q_pe, k_pe = m.rotary_emb(positions, q_pe, k_pe)

        if q_full_width and not token_sharded:
            # RoPE ran on the full head set (it is per-head, so the extra
            # heads cost the rope nothing); take this rank's rows now, the same
            # slice q_nope got before the w_kc matmul.
            q_pe = dsa_token_shard_slice(q_pe, dsa_cp_plan)

        if token_sharded:
            # The chunk-KV allGather (seam S1). The KV latent was computed on
            # this rank's [rows] tokens, but the pool write (the sparse backend
            # scatters k at the FULL out_cache_loc) and the SFA read address
            # every token, so the chunk KV has to go back to full width here.
            # Issued async BEFORE the indexer so the collective overlaps the
            # top-k run (the vLLM-Ascend pattern); the core waits on the
            # handles before anything reads the gathered rows. Chunk-sized, not
            # context-sized: ~19 MB per layer at a 16.4k chunk.
            group = get_parallel().attn_tp_group
            k_nope_send = k_nope.contiguous()
            k_pe_send = k_pe.contiguous()
            k_nope = k_nope_send.new_empty(
                (dsa_cp_plan.num_tokens_pad, *k_nope_send.shape[1:])
            )
            k_pe = k_pe_send.new_empty(
                (dsa_cp_plan.num_tokens_pad, *k_pe_send.shape[1:])
            )
            forward_batch.npu_dsa_cp_chunk_kv_works = (
                group.all_gather_into_tensor_async(k_nope, k_nope_send),
                group.all_gather_into_tensor_async(k_pe, k_pe_send),
                k_nope_send,
                k_pe_send,
            )

        if narrow_plan is not None:
            # Record the width to restore to BEFORE the swap: the original row
            # count, possibly padded past num_tokens, is not recoverable after.
            forward_batch.npu_dsa_token_shard_input_rows = q_nope.shape[0]
            q_swapped = dsa_token_shard_redistribute_heads(
                torch.cat([q_nope, q_pe], dim=-1), narrow_plan
            )
            q_nope, q_pe = q_swapped.split(
                [m.qk_nope_head_dim, m.qk_rope_head_dim], dim=-1
            )
            q_nope_out = torch_npu.npu_transpose_batchmatmul(
                q_nope.contiguous(),
                m.w_kc_full,
                perm_x1=(1, 0, 2),
                perm_x2=(0, 1, 2),
                perm_y=(1, 0, 2),
            )
            q_pe = q_pe.contiguous()

        if dsa_use_prefill_cp(forward_batch):
            # support allgather+rerrange
            k_nope, k_pe = m.rebuild_cp_kv_cache(
                latent_cache, forward_batch, k_nope, k_pe
            )

    if _enable_dcp_decode_q_preissue and is_dcp_mla_decode_phase(forward_batch):
        # The decode Q all-gather depends only on this QKV's output, so issue
        # it now, in prepare: the layer's indexer (and the rest of the layer)
        # runs under it instead of after it. Skipped while a graph is being
        # captured: the async work.wait() is a host synchronisation capture
        # forbids, so the captured graph keeps the synchronous gather.
        from sglang.srt.model_executor.runner import get_is_capture_mode

        if not get_is_capture_mode():
            dcp_group = get_parallel().dcp_group
            combined = torch.cat(
                [q_pe.transpose(0, 1), q_nope_out.transpose(0, 1)], dim=-1
            )
            gathered = combined.new_empty(
                (dcp_group.world_size * combined.shape[0], *combined.shape[1:])
            )
            forward_batch.npu_dcp_decode_q_pre = _DcpDecodeQPre(
                work=dcp_group.all_gather_into_tensor_async(gathered, combined),
                gathered=gathered,
                send=combined,
                d_pe=q_pe.size(-1),
                d_nope=q_nope_out.size(-1),
            )
            print_info_once(
                "DCP decode Q pre-issue is ON: the decode Q all-gather is "
                "issued in prepare, under the layer's indexer, instead of "
                "after it (eager decode only; captured graphs keep the "
                "synchronous gather)"
            )

    if (
        _enable_dsa_cp_q_a2a_preissue
        and dsa_cp_plan is not None
        and not mla_preprocess_used
        and not token_sharded
    ):
        # Version A: both head redistributions depend only on this QKV's output
        # (suppressed under token sharding: its q is already rows x all heads,
        # there is nothing to redistribute), so issue them now, under the
        # indexer, instead of in the core between the indexer and the sparse
        # operator. The core waits on the handles right before the operator.
        q_nope_pre, q_nope_work, q_nope_send = dsa_cp_redistribute_heads_async(
            q_nope_out, dsa_cp_plan
        )
        q_pe_pre, q_pe_work, q_pe_send = dsa_cp_redistribute_heads_async(
            q_pe, dsa_cp_plan
        )
        forward_batch.npu_dsa_cp_q_pre = (
            q_nope_pre,
            q_pe_pre,
            q_nope_work,
            q_pe_work,
            q_nope_send,
            q_pe_send,
        )
        print_info_once(
            "DSA-CP Q pre-issue is ON: the Q all-to-alls run under the "
            "indexer instead of between the indexer and the sparse operator"
        )

    if _enable_dcp_sfa_gather_overlap and _dcp_extend_active(forward_batch):
        # The gather was issued before the QKV (reuse layer); attach this
        # chunk's own KV now that the QKV has produced it -- the one part a
        # pre-QKV issue cannot carry. The core waits on the handle right before
        # the sparse operator reads the buffer.
        pending = getattr(forward_batch, "npu_dcp_sfa_gather_pending", None)
        if pending is not None and pending.k_nope is None:
            if pending.out_nope.dtype != k_nope.dtype:
                raise AssertionError(
                    "DCP SFA gather overlap: the gather buffer was typed from "
                    f"hidden_states ({pending.out_nope.dtype}) but this layer's "
                    f"KV is {k_nope.dtype}; the buffer and the KV must share a "
                    "dtype or the sparse operator reads the wrong type"
                )
            forward_batch.npu_dcp_sfa_gather_pending = pending._replace(
                k_nope=k_nope, k_pe=k_pe
            )

    if (
        _enable_dcp_sfa_gather_overlap
        and _dcp_extend_active(forward_batch)
        and indexer_runs
        and getattr(forward_batch, "npu_dcp_sfa_gather_pending", None) is None
    ):
        # Indexer layer, the vLLM placement: issue right before the indexer.
        # This chunk's KV already exists, so it goes in with the collective
        # instead of a second step, and the gather's window is the indexer.
        forward_batch.npu_dcp_sfa_gather_pending = _dcp_start_extend_gather_npu(
            m,
            forward_batch,
            hidden_states,
            _dcp_sfa_gather_topk_width(m, prev_topk_indices),
            k_nope=k_nope,
            k_pe=k_pe,
        )

    if indexer_runs:
        topk_indices = m.indexer(
            hidden_states,
            q_lora,
            positions,
            forward_batch,
            m.layer_id,
            input_on_attn_tp_slices,
            dynamic_scale,
        )
        # Remap only a fresh top-k; layers that skip the indexer reuse the
        # remapped one. Decode only: extend reads the GATHERED context, whose
        # coordinates are global.
        if is_dcp_mla_decode_phase(forward_batch):
            parallel = get_parallel()
            topk_indices = remap_dcp_sparse_indices(
                topk_indices,
                parallel.attn_dcp_size,
                parallel.attn_dcp_rank,
                interleave_size=dcp_interleave_size(),
            )
        else:
            # C1: extend packed read -- remap the fresh top-k to the rank-major
            # packed rows. Guard on DCP-active (the same predicate the prime
            # site and the gather use): without DCP (dcp_size==1) the plan would
            # still build, log "packed read ON", and identity-remap -- harmless
            # but misleading, and a wasted plan-build on every non-decode
            # forward. dcp_packed_read_plan is None for the refusal cases
            # (multi-request, no top-k, causal crop needed, FP8 KV), so this is
            # a no-op except a one-request crop-free DCP extend with the flag on.
            # Remapped HERE (where the indexer produces it) so it runs once per
            # indexer-layer, not at the operator on all 78 layers.
            if _dcp_extend_active(forward_batch):
                packed_plan = dcp_packed_read_plan(
                    forward_batch, topk_indices.shape[-1]
                )
                if packed_plan is not None:
                    topk_indices = remap_topk_to_packed(topk_indices, packed_plan)
    else:
        topk_indices = prev_topk_indices

    topk_indices = maybe_capture_indexer_topk(m.layer_id, topk_indices)

    return (
        q_pe,
        k_pe,
        q_nope_out,
        k_nope,
        topk_indices,
        forward_batch,
        zero_allocator,
        positions,
        mla_preprocess_used,
    )


# Rows per extend-gather collective; caps the scratch, not the bytes moved.
_dcp_extend_gather_piece_rows = envs.SGLANG_NPU_DCP_EXTEND_GATHER_PIECE_ROWS.get()
if _dcp_extend_gather_piece_rows <= 0:
    _dcp_extend_gather_piece_rows = 1 << 62

_last_dcp_extend_rows: Optional[Tuple[int, int]] = None


def _log_dcp_extend_memory(prefix_rows: int, extend_rows: int) -> None:
    """Log the previous DCP extend's peak memory, then reset. The peak is what a
    chunk size has to fit under; npu-smi shows only the reservation."""
    global _last_dcp_extend_rows
    if _last_dcp_extend_rows is not None:
        gib = 1 << 30
        logger.info(
            "DCP extend memory: prefix=%d extend=%d peak_allocated=%.2f GiB "
            "allocated=%.2f GiB reserved=%.2f GiB",
            *_last_dcp_extend_rows,
            torch.npu.max_memory_allocated() / gib,
            torch.npu.memory_allocated() / gib,
            torch.npu.memory_reserved() / gib,
        )
    torch.npu.reset_peak_memory_stats()
    _last_dcp_extend_rows = (prefix_rows, extend_rows)


def _log_dcp_shared_prefix(forward_batch, row_bytes: int) -> None:
    """Report how much of this gather is one prefix fetched once per request.

    ``walked_ok`` is the design question: sharing read off the radix nodes has
    to agree with the scheduler's own tree-owned count, or a dedup cannot use
    it. The gather sends ``prefix``, which also carries each request's private
    partial page, so the saving is over the tree-owned part only.
    """
    shared = getattr(forward_batch, "npu_dcp_shared_prefix", None)
    if shared is None:
        return
    prefix_lens = list(forward_batch.extend_prefix_lens_cpu)
    sent = sum(prefix_lens)
    saved = sum(shared.protected) - shared.union_rows
    gib = 1 << 30
    logger.info(
        "DCP shared prefix: bs=%d prefix=%s protected=%s walked=%s walked_ok=%s "
        "sent_rows=%d union_rows=%d private_rows=%d saved=%.1f%% (%.2f GiB/layer)",
        len(prefix_lens),
        prefix_lens,
        shared.protected,
        shared.walked,
        shared.walked == shared.protected,
        sent,
        shared.union_rows,
        sent - sum(shared.protected),
        100.0 * saved / sent if sent else 0.0,
        saved * row_bytes / gib,
    )


def _pad_dcp_extend_send(shards: torch.Tensor, plan) -> torch.Tensor:
    """Lay this rank's per-request shards out at their padded send offsets."""
    send = shards.new_empty((plan.send_rows, *shards.shape[1:]))
    src = dst = 0
    for local_len, padded_len in zip(plan.local_lens, plan.padded_lens):
        send[dst : dst + local_len] = shards[src : src + local_len]
        src += local_len
        dst += padded_len
    return send


class _DcpGatherPrefetch:
    """Per-forward state for running each layer's prefix gather a layer ahead.

    Two scratch slots by layer parity. Both waits are unconditional, because
    getting either wrong corrupts silently: the side stream waits the main
    stream's ``release`` before refilling a slot (the index_select two layers
    back must be done), and the main stream waits the side stream's ``ready``
    before reading one.

    The __init__ syncs are once per forward, never per layer: ``wait_stream``
    depends on everything already queued, which at layer L includes layer L's
    compute and would serialise exactly what this overlaps.
    """

    __slots__ = ("stream", "pending", "release")

    def __init__(self, stream: "torch.npu.Stream"):
        self.stream = stream
        # layer_id -> (slot, ready_event, sends). ``sends`` is held only so the
        # caching allocator cannot reissue those buffers mid-collective.
        self.pending: Dict[int, tuple] = {}
        self.release: Dict[int, torch.npu.Event] = {}
        self.stream.wait_stream(torch.npu.current_stream())
        torch.npu.current_stream().wait_stream(self.stream)

    def slot_of(self, layer_id: int, start_layer: int) -> int:
        return (layer_id - start_layer) & 1


_logged_dcp_gather_prefetch = False


def _log_dcp_gather_prefetch_once(prefetching: bool, plan, packed_kv: bool) -> None:
    """Say once whether the prefetch engaged; the fallback is otherwise silent."""
    global _logged_dcp_gather_prefetch
    if _logged_dcp_gather_prefetch:
        return
    _logged_dcp_gather_prefetch = True
    if prefetching:
        logger.info(
            "DCP extend gather prefetch is ACTIVE: %d scratch rows per slot, "
            "%d keys, 2 slots",
            plan.scratch_rows,
            1 if packed_kv else 2,
        )
    elif _prefetch_dcp_extend_gather:
        logger.info(
            "DCP extend gather prefetch was REQUESTED but is OFF: the plan has "
            "%d pieces and it needs exactly 1. Set "
            "SGLANG_NPU_DCP_EXTEND_GATHER_PIECE_ROWS=0.",
            len(plan.pieces),
        )
    else:
        logger.info("DCP extend gather prefetch is off (inline gathers)")


def _dcp_extend_gather_scratches(slot: int, k_nope, k_pe, rows: int, packed_kv: bool):
    """``[(scratch, own), ...]``: one key under FP8, two under bf16. Slot 1
    exists only under the prefetch, so slot 0 keeps the historical names."""
    suffix = "" if slot == 0 else "_b"
    keys = [(dcp_extend_gather_buffer("latent_scratch" + suffix, k_nope, rows), k_nope)]
    if not packed_kv:
        keys.append(
            (dcp_extend_gather_buffer("rope_scratch" + suffix, k_pe, rows), k_pe)
        )
    return keys


def _dcp_extend_send_rows(
    pool, layer, md, plan, layer_id: int, packed_kv: bool, f2: bool = False
):
    """This rank's padded prefix shard for ``layer_id`` -- the gather's input.

    Prefix rows only, so it reads nothing the current forward wrote. That is
    what lets the prefetch call it a layer early. One send per gathered key,
    in ``_dcp_extend_gather_scratches`` order. Under F2 the send is the raw 656
    B int8 rows (one opaque uint8 key), read with the pool's raw gather instead
    of get_mla_kv_buffer's dequant-on-read; the operator dequantises in-kernel.
    """
    if not plan.send_rows:
        return [None] if packed_kv else [None, None]
    if f2:
        send = pool.gather_packed_sfa_rows_raw(
            layer_id, md.dcp_local_prefix_kv_indices
        ).unsqueeze(1)
        if plan.local_lens != plan.padded_lens:
            send = _pad_dcp_extend_send(send, plan)
        return [send]
    send_nope, send_rope = pool.get_mla_kv_buffer(
        layer, md.dcp_local_prefix_kv_indices, layer_id=layer_id
    )
    if plan.local_lens != plan.padded_lens:
        # Served prefixes are cycle-aligned and need no padding; this is the
        # general case.
        send_nope = _pad_dcp_extend_send(send_nope, plan)
        if send_rope is not None:
            send_rope = _pad_dcp_extend_send(send_rope, plan)
    return [send_nope] if packed_kv else [send_nope, send_rope]


def _dcp_extend_all_gather(parallel, piece, pairs) -> None:
    """The collectives for one piece; ``pairs`` is [(scratch, send), ...]."""
    gathered = (piece.send_end - piece.send_start) * parallel.dcp_size
    if not gathered:
        return
    for scratch, send in pairs:
        parallel.dcp_group.all_gather_into_tensor(
            scratch[:gathered], send[piece.send_start : piece.send_end]
        )


_enable_dcp_gather_prefetch = (
    envs.SGLANG_NPU_ENABLE_DCP_PACKED_GATHER_PREFETCH.get()
)

_enable_dcp_sfa_gather_overlap = envs.SGLANG_NPU_ENABLE_DCP_SFA_GATHER_OVERLAP.get()
_enable_dsa_cp_local_topk = envs.SGLANG_NPU_DSA_CP_LOCAL_TOPK.get()
_enable_dsa_cp_q_a2a_preissue = envs.SGLANG_NPU_DSA_CP_Q_A2A_PREISSUE.get()
_enable_dsa_cp_q_full_width = envs.SGLANG_NPU_DSA_CP_Q_FULL_WIDTH.get()
_enable_dcp_decode_q_preissue = envs.SGLANG_NPU_DCP_DECODE_Q_PREISSUE.get()
# D2v3: stream the o_proj full matrix on the DCP gather side stream instead of
# holding a static per-rank replica. OFF restores the D1v3 static
# o_proj_qrep_weight, byte for byte. It only takes effect under the D1v3
# token-sharded rebuild, which is what makes the full matrix necessary.
_stream_o_proj = (
    envs.SGLANG_NPU_STREAM_O_PROJ.get()
    and envs.SGLANG_NPU_DSA_CP_TOKEN_SHARDED.get()
)
_dcp_o_proj_fallback_logged = False
_eager_o_proj_slots: list = [None, None]
# The rank-major scratch the eager o_proj gather fills before reassembling the
# [out, in] matrix (see _dcp_o_proj_weight); all_gather_into_tensor fills its
# output flat, rank-major, which is NOT the RowParallel [out, in] matrix.
_eager_o_proj_gather_slots: list = [None, None]
# D2v4: stream the remaining token-sharded replicated weights the same way --
# keep only this rank's TP shard and all-gather the full matrix on the fly at
# the GEMM instead of standing it per rank. The gather is the EXACT operation
# that builds the static qrep (all_gather over the same shard, same dim), so
# the assembled weight is bit-identical to it. Gated on the token-sharded
# rebuild like o_proj: that is what makes the full matrix stand in the first
# place. The gather runs synchronously on the compute stream.
_token_sharded_flag = envs.SGLANG_NPU_DSA_CP_TOKEN_SHARDED.get()
_stream_q_b = envs.SGLANG_NPU_STREAM_Q_B.get() and _token_sharded_flag
_stream_w_kc = envs.SGLANG_NPU_STREAM_W_KC.get() and _token_sharded_flag
_stream_w_vc = envs.SGLANG_NPU_STREAM_W_VC.get() and _token_sharded_flag


def _dcp_q_b_weight(m) -> torch.Tensor:
    # D2v4: full q_b_proj, all-gathered on the fly. The same all_gather over the
    # same shard/dim as the static q_b_proj_qrep_weight, so bit-identical.
    shard = getattr(m, "q_b_stream_shard", None)
    if shard is None:
        raise RuntimeError(
            f"q_b streaming: layer {getattr(m, 'layer_id', '?')}'s shard was "
            "never registered; the model-runner preparation must have run "
            "before any forward"
        )
    return get_parallel().attn_tp_group.all_gather(shard, dim=0)


def _dcp_w_kc_weight(m) -> torch.Tensor:
    shard = getattr(m, "w_kc_stream_shard", None)
    if shard is None:
        raise RuntimeError(
            f"w_kc streaming: layer {getattr(m, 'layer_id', '?')}'s shard was "
            "never registered; the model-runner preparation must have run "
            "before any forward"
        )
    return get_parallel().attn_tp_group.all_gather(shard, dim=0)


def _dcp_w_vc_weight(m) -> torch.Tensor:
    shard = getattr(m, "w_vc_stream_shard", None)
    if shard is None:
        raise RuntimeError(
            f"w_vc streaming: layer {getattr(m, 'layer_id', '?')}'s shard was "
            "never registered; the model-runner preparation must have run "
            "before any forward"
        )
    return get_parallel().attn_tp_group.all_gather(shard, dim=0)


if _enable_dsa_cp_q_a2a_preissue and _enable_dsa_cp_q_full_width:
    # Both are versions of the same removal (the per-layer Q all-to-all).
    # Running either alone is the A/B this pair exists for; running both is a
    # configuration error, not a fallback order.
    raise ValueError(
        "SGLANG_NPU_DSA_CP_Q_A2A_PREISSUE and SGLANG_NPU_DSA_CP_Q_FULL_WIDTH "
        "are mutually exclusive: both replace the DSA-CP Q all-to-all, and the "
        "point of having both is to profile one against the other. Enable "
        "exactly one (or neither)."
    )

if _enable_dcp_gather_prefetch and not envs.SGLANG_NPU_ENABLE_DCP_PACKED_READ.get():
    # The prefetch fills the buffer the operator reads directly. On the
    # permuting path that buffer is produced by an index_select from a scratch,
    # which cannot be prefetched a layer ahead without also double-buffering the
    # scratch and splitting the plan's index. Refuse rather than be silently
    # inert -- the whole point of the flag is that it does something.
    raise ValueError(
        "SGLANG_NPU_ENABLE_DCP_PACKED_GATHER_PREFETCH requires "
        "SGLANG_NPU_ENABLE_DCP_PACKED_READ: there is nothing to prefetch into "
        "until the operator reads the gathered buffer directly."
    )


class _DcpGatherPrefetcher:
    """C3 (packed path): gather layer l+1's prefix on a side stream while layer l computes.

    The all-gather is ~0.52 s of a chunk at a ~990k prefix. It depends on
    nothing the current layer computes: the prefix KV was written by earlier
    forwards and sits in the pool, so it can run underneath the layer before it.

    Two slots, indexed ``layer_id % 2``. Every dcp-group gather goes on the side
    stream (including the first) so one HCCL communicator stays bound to one
    stream for the life of the process; the compute stream only waits on events.
    Only this chunk's own KV is written from the compute stream, because its
    k_nope/k_pe are computed during this layer.
    """

    def __init__(self) -> None:
        # D2v2: created with an explicit priority, the vLLM-CUDA "KV I/O yields
        # to compute" pattern (vLLM makes its offload transfer streams the
        # LOWEST-priority streams it can). The lowest torch_npu exposes is 0 --
        # positive values clamp to the same pool, -1 is the HIGH pool, and a
        # stream that must yield never takes it.
        self.stream = torch.npu.Stream(
            priority=envs.SGLANG_NPU_DCP_GATHER_STREAM_PRIORITY.get()
        )
        self.ready: list = [None, None]
        self.filled: list = [None, None]
        self.prev_layer: Optional[int] = None
        self._max_layer: Optional[int] = None
        # The layer whose prefix prime() issued in prepare, before any take().
        # take() must not let _note_layer reset that slot: the prime ran before
        # the first take, so the structural "layer id stopped advancing" test
        # cannot tell the primed first layer apart from a fresh forward.
        self.primed: Optional[int] = None
        # D2v3: the full o_proj matrix for a layer, assembled on THIS side
        # stream immediately after that layer's context-KV gather (one layer
        # ahead, so it hides under the layer's own SFA and absorb). Double
        # buffered by layer parity, like the KV slots: the gather for layer L+1
        # must not overwrite the slot layer L-1's o_proj GEMM is still
        # consuming, and the wait_stream at the top of _gather_prefix orders it.
        self.o_proj_slots: list = [None, None]
        # The rank-major scratch the side-stream o_proj gather fills before
        # reassembling the [out, in] matrix into o_proj_slots (see
        # _issue_o_proj_gather): all_gather_into_tensor fills its output flat,
        # rank-major, which is NOT the RowParallel [out, in] matrix, so gather
        # into [tp, out, in_r] and movedim(0, 1) into the slot.
        self.o_proj_gather_slots: list = [None, None]
        self.o_proj_ready: list = [None, None]
        self.o_proj_filled: list = [None, None]
        self._o_proj_ranks_checked = False

    def _note_layer(self, layer_id: int) -> None:
        """Drop both slots when a new forward starts.

        Detected structurally -- layers run 0, 1, 2, ... and then the next
        forward starts over, so a layer id that does not advance means a new
        batch -- rather than by the identity of the ForwardBatch.

        The one exception is a primed forward: prime() runs before the first
        take(), and fills the first layer's slot. The structural test cannot
        see that the first take() belongs to the primed forward (the layer id
        has not advanced past the prime's), so the marker is consumed here and
        the primed slot is kept.
        """
        if self.primed is not None:
            if layer_id == self.primed:
                # The take of the very layer prime() filled: keep its slot.
                self.primed = None
                self.prev_layer = layer_id
                return
            # A take that is not the primed layer: the prime is stale (its
            # forward never consumed it) or this is a newer forward. Drop the
            # marker and fall through to the normal reset below.
            self.primed = None
        if self.prev_layer is None or layer_id <= self.prev_layer:
            self.filled = [None, None]
        self.prev_layer = layer_id

    def slots(self, plan, k_nope, k_pe, slot: int):
        return (
            dcp_extend_gather_buffer(f"packed_latent_{slot}", k_nope, plan.rows),
            dcp_extend_gather_buffer(f"packed_rope_{slot}", k_pe, plan.rows),
        )

    def _gather_prefix(self, m, md, plan, ref_nope, ref_rope, layer_id: int) -> None:
        """Issue one layer's prefix all-gather on the side stream.

        ``ref_nope``/``ref_rope`` only type the slot buffers (dtype, device, row
        shape); they are the chunk's own KV on a take() and a (1, 1, dim) slice
        of hidden_states on a prime(), before the QKV has produced them.
        """
        slot = layer_id % 2
        pool = get_token_to_kv_pool()
        # BUG 15: under F2 use a single uint8 key buffer, no rope.
        _f2 = (
            getattr(pool, "sfa_kv_store_8bit", False)
            and envs.SGLANG_NPU_SFA_KV_QUANT_READ.get()
        )
        if _f2:
            _ref = ref_nope.new_empty(
                (1, 1, pool.kv_cache_dim), dtype=torch.uint8
            )
            out_nope = dcp_extend_gather_buffer(
                f"packed_latent_{slot}", _ref, plan.rows
            )
            out_rope = None
        else:
            out_nope, out_rope = self.slots(plan, ref_nope, ref_rope, slot)
        # record_stream stops the allocator handing the compute-allocated slot
        # to something else while the side stream is still writing it.
        out_nope.record_stream(self.stream)
        if out_rope is not None:
            out_rope.record_stream(self.stream)
        # Wait for everything already issued on compute, so the slot's previous
        # reader -- layer_id - 2's attention -- is done before it is overwritten.
        self.stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(self.stream):
            if plan.gathered_rows:
                parallel = get_parallel()
                if _f2:
                    send = pool.gather_packed_sfa_rows_raw(
                        layer_id, md.dcp_local_prefix_kv_indices
                    ).unsqueeze(1)
                    send = _pad_dcp_packed_send(send, plan, parallel.dcp_rank)
                    _dcp_issue_gather_chunks(
                        parallel.dcp_group, out_nope, send, plan
                    )
                else:
                    send_nope, send_rope = pool.get_mla_kv_buffer(
                        m.attn_mqa,
                        md.dcp_local_prefix_kv_indices,
                        layer_id=layer_id,
                    )
                    send_nope = _pad_dcp_packed_send(
                        send_nope, plan, parallel.dcp_rank
                    )
                    send_rope = _pad_dcp_packed_send(
                        send_rope, plan, parallel.dcp_rank
                    )
                    # The work handles are dropped on purpose: every collective
                    # is enqueued on this side stream, so the ready event below
                    # orders the compute stream after all of them.
                    _dcp_issue_gather_chunks(
                        parallel.dcp_group, out_nope, send_nope, plan
                    )
                    _dcp_issue_gather_chunks(
                        parallel.dcp_group, out_rope, send_rope, plan
                    )
            # The KV-only ready event is recorded here, before the o_proj
            # gather, so the SFA read does not wait for the weight assembly.
            self.ready[slot] = self.stream.record_event()
            # D2v3: the full o_proj matrix for this layer, on the same side
            # stream and immediately after the KV gather -- the ~2 ms
            # collective then hides under this layer's SFA and absorb instead
            # of 14.1 GiB of it standing on every device.
            if _stream_o_proj:
                o_shard = get_o_proj_stream_shard(layer_id)
                if o_shard is not None:
                    self._issue_o_proj_gather(layer_id, o_shard)
        self.filled[slot] = layer_id

    def _issue_o_proj_gather(self, layer_id: int, o_shard: torch.Tensor) -> None:
        """All-gather the full o_proj matrix for ``layer_id`` on the side
        stream (see SGLANG_NPU_STREAM_O_PROJ).

        The shards are RowParallel over the attention-TP group, but the
        collect runs on the DCP group -- the same communicator as the KV
        gather this rides on, one communicator per stream. That is correct
        only while both groups are the same ranks in the same order (true at
        TP==DCP, the supported configuration), which is checked once, loudly.
        """
        parallel = get_parallel()
        if not self._o_proj_ranks_checked:
            if list(parallel.dcp_group.ranks) != list(
                parallel.attn_tp_group.ranks
            ):
                raise RuntimeError(
                    "SGLANG_NPU_STREAM_O_PROJ assembles the attention-TP "
                    "o_proj shards on the DCP group, which requires both "
                    "groups to hold the same ranks in the same order; got "
                    f"dcp={list(parallel.dcp_group.ranks)} vs "
                    f"attn_tp={list(parallel.attn_tp_group.ranks)}"
                )
            self._o_proj_ranks_checked = True
            print_info_once(
                "D2v3 o_proj streaming ON: the full o_proj matrix is "
                "all-gathered on the DCP gather side stream, right after the "
                "same layer's context-KV gather, instead of standing "
                "replicated on every device"
            )
        slot = layer_id % 2
        full_shape = (
            o_shard.shape[0],
            o_shard.shape[1] * parallel.attn_tp_size,
        )
        slot_buf = self.o_proj_slots[slot]
        if (
            slot_buf is None
            or slot_buf.shape != full_shape
            or slot_buf.dtype != o_shard.dtype
            or slot_buf.device != o_shard.device
        ):
            slot_buf = o_shard.new_empty(full_shape)
            self.o_proj_slots[slot] = slot_buf
        # Same allocator bookkeeping as the KV slots above: written on the
        # side stream, read on the compute stream, cached for good.
        slot_buf.record_stream(self.stream)
        gather_shape = (
            parallel.attn_tp_size,
            o_shard.shape[0],
            o_shard.shape[1],
        )
        gather_buf = self.o_proj_gather_slots[slot]
        if (
            gather_buf is None
            or gather_buf.shape != gather_shape
            or gather_buf.dtype != o_shard.dtype
            or gather_buf.device != o_shard.device
        ):
            gather_buf = o_shard.new_empty(gather_shape)
            self.o_proj_gather_slots[slot] = gather_buf
        gather_buf.record_stream(self.stream)
        # The shards are RowParallel -- rank r holds W[:, r*in_r:(r+1)*in_r]
        # -- so the flat rank-major fill is NOT the [out, in] matrix: row o
        # of that is rank 0's row o, then rank 1's, and so on. Gather into
        # the rank-major [tp, out, in_r] shape and reassemble the rows, the
        # same math as the static D1v3 path's dim-1 all_gather. Gathered into
        # [out, in] directly, every o_proj GEMM would multiply by a permuted
        # weight and the output would be fluent but wrong.
        parallel.dcp_group.all_gather_into_tensor(gather_buf, o_shard)
        slot_buf.view(
            full_shape[0], parallel.attn_tp_size, o_shard.shape[1]
        ).copy_(gather_buf.movedim(0, 1))
        # A separate event: the o_proj GEMM waits on this one, and it is
        # ordered after the KV ready event by the side stream itself.
        self.o_proj_ready[slot] = self.stream.record_event()
        self.o_proj_filled[slot] = layer_id

    def take_o_proj(self, layer_id: int) -> torch.Tensor:
        """This layer's full o_proj matrix: wait for its side-stream gather
        and take the double-buffered slot. The wait is the last moment before
        the GEMM reads the buffer -- the same placement as the KV ready wait
        in take(). The filled-layer check, not just the event's presence: a
        parity slot can carry a live event from layer L-2 while layer L's
        gather was never issued (a forward the prefetch declined), and waiting
        on the stale event would read L-2's matrix for L, silently wrong."""
        slot = layer_id % 2
        if self.o_proj_ready[slot] is None:
            raise RuntimeError(
                f"o_proj streaming: layer {layer_id}'s full matrix was never "
                "gathered; the side stream must issue it one layer ahead "
                "(prime/take), the same as the KV gather"
            )
        if self.o_proj_filled[slot] != layer_id:
            raise RuntimeError(
                f"o_proj streaming: the slot for layer {layer_id} was filled "
                f"for layer {self.o_proj_filled[slot]}; the parity buffer "
                "holds a stale matrix"
            )
        torch.npu.current_stream().wait_event(self.o_proj_ready[slot])
        return self.o_proj_slots[slot]

    def prime(self, m, md, plan, hidden_states, layer_id: int) -> None:
        """Issue the first layer's prefix gather at the start of the forward.

        take() issues a layer's gather from the core, AFTER that layer's indexer
        has run; with no previous layer to prefetch from, layer 0's gather then
        starts in the indexer's window, idle. prime() runs before the QKV, so
        the first gather moves bytes under the QKV and the indexer. Only the
        first DSA layer of a forward primes; the rest no-op here and are
        prefetched by the previous layer's take() as usual. The slot it fills is
        the same one take() would fill, typed from hidden_states (the QKV output
        does not exist yet) rather than from the chunk's own KV, so the two share
        one grow-only buffer.
        """
        if self.prev_layer is not None and layer_id > self.prev_layer:
            # A layer earlier in THIS forward already ran, so this is not the
            # first layer: nothing to prime, take() will handle it.
            return
        # First DSA layer of a (possibly new) forward: drop the previous
        # forward's slots, then fill this layer's.
        self.filled = [None, None]
        self.prev_layer = layer_id
        ref_nope = hidden_states.new_empty((1, 1, m.kv_lora_rank))
        ref_rope = hidden_states.new_empty((1, 1, m.qk_rope_head_dim))
        self._gather_prefix(m, md, plan, ref_nope, ref_rope, layer_id)
        self.primed = layer_id
        print_info_once(
            "DCP gather first-layer prime is ON: layer "
            f"{layer_id}'s prefix all-gather is issued before the QKV, so it "
            "runs under the QKV and the indexer instead of starting with them"
        )

    def _max_servable_layer(self) -> int:
        """Highest layer id the KV pool can serve, from the list it indexes.

        NOT from ``pool.end_layer`` -- that field is inconsistent (the pool's
        constructor defaults it inclusive while model_runner passes it
        exclusive). ``get_key_buffer`` reads ``k_buffer[layer_id - start_layer]``
        (after start_layer is applied), so that buffer's length is the only
        statement of the bound that cannot disagree with the read it guards.
        ``-1`` disables prefetching entirely (every layer gathers inline).
        """
        if self._max_layer is None:
            pool = get_token_to_kv_pool()
            k = getattr(pool, "k_buffer", None)
            start = getattr(pool, "start_layer", 0)
            self._max_layer = -1 if k is None else start + len(k) - 1
        return self._max_layer

    def take(self, m, md, plan, k_nope, k_pe, layer_id: int):
        """This layer's KV buffer, with next layer's gather already in flight."""
        self._note_layer(layer_id)
        slot = layer_id % 2
        if self.filled[slot] != layer_id:
            # Layer 0 of a forward, or a layer whose prefetch was skipped. Issue
            # it now and pay for it, exactly as the unprefetched path would.
            self._gather_prefix(m, md, plan, k_nope, k_pe, layer_id)
        else:
            # Logged on a HIT, not on construction: a prefetcher built and then
            # missing on every layer does no overlapping at all.
            print_info_once(
                "DCP gather prefetch is ON: layer l+1's prefix all-gather runs "
                "on a side stream underneath layer l's compute"
            )
        pool = get_token_to_kv_pool()
        _f2 = (
            getattr(pool, "sfa_kv_store_8bit", False)
            and envs.SGLANG_NPU_SFA_KV_QUANT_READ.get()
        )
        if _f2:
            _ref = k_nope.new_empty(
                (1, 1, pool.kv_cache_dim), dtype=torch.uint8
            )
            out_nope = dcp_extend_gather_buffer(
                f"packed_latent_{slot}", _ref, plan.rows
            )
            out_rope = None
        else:
            out_nope, out_rope = self.slots(plan, k_nope, k_pe, slot)
        torch.npu.current_stream().wait_event(self.ready[slot])
        # The one part that could not be prefetched: this chunk's own KV.
        if _f2:
            n = plan.extend_len
            packed_tail = pool.pack_sfa_kv_cache(
                k_nope[:n], k_pe[:n]
            ).view(torch.uint8).unsqueeze(1)
            out_nope[plan.gathered_rows :] = packed_tail
        else:
            _write_packed_extend_tail(out_nope, out_rope, k_nope, k_pe, plan)
        if layer_id < self._max_servable_layer():
            self._gather_prefix(m, md, plan, k_nope, k_pe, layer_id + 1)
        return out_nope, out_rope


_dcp_gather_prefetcher: Optional["_DcpGatherPrefetcher"] = None


def _dcp_gather_prefetcher_get() -> "_DcpGatherPrefetcher":
    """The process-wide prefetcher, created on first use.

    Two call sites need it -- the core (to prime the first layer's gather and
    to take a layer's buffer) -- and either can run first, so the creation
    lives here rather than in one of them.
    """
    global _dcp_gather_prefetcher
    if _dcp_gather_prefetcher is None:
        _dcp_gather_prefetcher = _DcpGatherPrefetcher()
    return _dcp_gather_prefetcher


def _dcp_o_proj_weight(m) -> torch.Tensor:
    """This layer's full o_proj matrix for the D2v3 streamed path: the side
    stream assembles it right after the same layer's KV gather -- wait on its
    ready event and take the double-buffered slot. Fallback (the prefetch off,
    DCP disabled, or a forward whose packed plan declined the prefetch): gather
    synchronously on the compute stream over the attention-TP group -- the
    shards' own group, so no rank-order assumption across groups -- into the
    parity slot. Correct, and exposed on the critical path, which is why the
    prefetch path exists; the slot is safe across layers because the collect is
    enqueued after the previous consumer's GEMM in stream order."""
    global _dcp_o_proj_fallback_logged
    parallel = get_parallel()
    if _enable_dcp_gather_prefetch and parallel.dcp_enabled:
        prefetcher = _dcp_gather_prefetcher_get()
        if prefetcher.o_proj_filled[m.layer_id % 2] == m.layer_id:
            return prefetcher.take_o_proj(m.layer_id)
        if not _dcp_o_proj_fallback_logged:
            _dcp_o_proj_fallback_logged = True
            print_info_once(
                "D2v3 o_proj: the side stream did not gather layer "
                f"{m.layer_id}'s matrix this forward (DCP disabled, or the "
                "packed plan declined the prefetch); assembling it "
                "synchronously on the compute stream instead -- correct, but "
                "exposed on the critical path."
            )
    shard = get_o_proj_stream_shard(m.layer_id)
    if shard is None:
        raise RuntimeError(
            f"o_proj streaming: layer {m.layer_id}'s shard was never "
            "registered; the model-runner token-sharded preparation must "
            "have run before any forward"
        )
    slot = m.layer_id % 2
    full_shape = (
        shard.shape[0],
        shard.shape[1] * parallel.attn_tp_size,
    )
    buf = _eager_o_proj_slots[slot]
    if (
        buf is None
        or buf.shape != full_shape
        or buf.dtype != shard.dtype
        or buf.device != shard.device
    ):
        buf = shard.new_empty(full_shape)
        _eager_o_proj_slots[slot] = buf
    # RowParallel shards: the flat rank-major fill is not the [out, in]
    # matrix (row o is rank 0's row o, then rank 1's, ...), so gather into the
    # rank-major shape and reassemble the rows -- the same math as the
    # side-stream path and the static D1v3 dim-1 all_gather.
    gather_shape = (
        parallel.attn_tp_size,
        shard.shape[0],
        shard.shape[1],
    )
    gather_buf = _eager_o_proj_gather_slots[slot]
    if (
        gather_buf is None
        or gather_buf.shape != gather_shape
        or gather_buf.dtype != shard.dtype
        or gather_buf.device != shard.device
    ):
        gather_buf = shard.new_empty(gather_shape)
        _eager_o_proj_gather_slots[slot] = gather_buf
    parallel.attn_tp_group.all_gather_into_tensor(gather_buf, shard)
    buf.view(full_shape[0], parallel.attn_tp_size, shard.shape[1]).copy_(
        gather_buf.movedim(0, 1)
    )
    return buf


def _dcp_extend_active(forward_batch: "ForwardBatch") -> bool:
    """Whether this forward's DCP extend gather runs, per layer.

    prepare and core must agree on exactly this, so the expression lives in one
    place: prepare decides whether to issue the async gather, core whether to
    wait for it.
    """
    return (
        get_parallel().dcp_enabled
        and forward_batch.forward_mode.is_extend()
        and not is_dcp_mla_decode_phase(forward_batch)
        and forward_batch.attn_dcp_metadata is not None
    )


def _dcp_sfa_gather_topk_width(
    m: "DeepseekV2AttentionMLA", prev_topk_indices: Optional[torch.Tensor]
) -> Optional[int]:
    """This layer's eventual top-k width, known before the indexer runs.

    The gather planner takes the width the sparse operator will be handed
    (``topk_indices.shape[-1]`` in the core): the indexer's ``index_topk`` on
    the layers that run it, and the previous layer's on the 57 that reuse it.
    The path decision (permuting vs packed) has to be made in prepare, before
    the indexer has produced the tensor, so derive the same width from the same
    condition the indexer call is gated on.
    """
    if not m.skip_topk or (m.is_nextn and prev_topk_indices is None):
        return m.indexer.index_topk if m.indexer is not None else None
    return prev_topk_indices.shape[-1] if prev_topk_indices is not None else None


def _dcp_gather_extend_kv_packed_npu(
    m: "DeepseekV2AttentionMLA",
    forward_batch: "ForwardBatch",
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    plan: "DcpPackedReadPlan",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """C1: leave the gathered prefix in the order the collective produced it.

    The permuting path writes the all-gather into a scratch and then
    ``index_select``s it into a position-ordered output. This does neither: the
    collective writes straight into the buffer the operator will read, and this
    chunk's own KV is appended after it. What pays for that is
    ``remap_topk_to_packed``, arithmetic on the ``[tokens, k]`` index tensor.

    **No pieces here, deliberately.** Pieces cap the scratch held beside the
    output, and there is no scratch now -- the buffer IS the output. Piecing
    would also break the remap: one collective, one layout, one formula.
    """
    parallel = get_parallel()
    md = forward_batch.attn_dcp_metadata
    if not getattr(forward_batch, "npu_dcp_packed_ready", False):
        # Once per forward, on the first layer: carry the same once-per-forward
        # write-plan duty the permuting path does in its plan-building branch.
        forward_batch.npu_dcp_packed_ready = True
        md.dcp_kv_buffer = None
        plan_write = getattr(get_token_to_kv_pool(), "plan_dcp_extend_write", None)
        if plan_write is not None:
            plan_write(forward_batch.out_cache_loc)
        if _debug_dcp_extend_memory:
            _log_dcp_extend_memory(plan.prefix_len, plan.extend_len)

    if _enable_dcp_gather_prefetch:
        # The prefetcher is shared with the prepare-side prime, so it is created
        # (on first use) in one place: _dcp_gather_prefetcher_get().
        return _dcp_gather_prefetcher_get().take(
            m, md, plan, k_nope, k_pe, m.layer_id
        )

    pool = get_token_to_kv_pool()
    # BUG 15: under F2 the operator expects raw 656-byte uint8 rows, not
    # dequantized bf16 halves. Ship one opaque key like the permuting path.
    _f2 = (
        getattr(pool, "sfa_kv_store_8bit", False)
        and envs.SGLANG_NPU_SFA_KV_QUANT_READ.get()
    )
    if _f2:
        _ref = k_nope.new_empty((1, 1, pool.kv_cache_dim), dtype=torch.uint8)
        out_nope = dcp_extend_gather_buffer("packed_latent", _ref, plan.rows)
        out_rope = None
        if plan.gathered_rows:
            send = pool.gather_packed_sfa_rows_raw(
                m.layer_id, md.dcp_local_prefix_kv_indices
            ).unsqueeze(1)
            send = _pad_dcp_packed_send(send, plan, parallel.dcp_rank)
            _dcp_issue_gather_chunks(parallel.dcp_group, out_nope, send, plan)
        # Own-chunk tail: pack the bf16 KV into raw 656-byte rows.
        n = plan.extend_len
        packed_tail = pool.pack_sfa_kv_cache(k_nope[:n], k_pe[:n]).view(
            torch.uint8
        ).unsqueeze(1)
        out_nope[plan.gathered_rows :] = packed_tail
        return out_nope, None

    out_nope = dcp_extend_gather_buffer("packed_latent", k_nope, plan.rows)
    out_rope = dcp_extend_gather_buffer("packed_rope", k_pe, plan.rows)

    if plan.gathered_rows:
        send_nope, send_rope = pool.get_mla_kv_buffer(
            m.attn_mqa, md.dcp_local_prefix_kv_indices
        )
        send_nope = _pad_dcp_packed_send(send_nope, plan, parallel.dcp_rank)
        send_rope = _pad_dcp_packed_send(send_rope, plan, parallel.dcp_rank)
        # Sync issue, on the compute stream: stream ordering alone makes the
        # tail write and the return below see the filled prefix, no handles.
        _dcp_issue_gather_chunks(parallel.dcp_group, out_nope, send_nope, plan)
        _dcp_issue_gather_chunks(parallel.dcp_group, out_rope, send_rope, plan)
    _write_packed_extend_tail(out_nope, out_rope, k_nope, k_pe, plan)
    return out_nope, out_rope


def _write_packed_extend_tail(
    out_nope: torch.Tensor,
    out_rope: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    plan: "DcpPackedReadPlan",
) -> None:
    """This chunk's own KV, after the gathered prefix. Not gathered: identical
    on every rank.

    ``k_nope`` can be WIDER than the request's token count (the batch is padded
    to a multiple of attn_tp_size), so take only the first plan.extend_len rows.
    The operator is given true query lengths, so it never reads a padded query
    row, and a real query's top-k only names positions below prefix+extend.
    """
    n = plan.extend_len
    assert k_nope.shape[0] >= n, (
        f"DCP packed read: this chunk's KV has {k_nope.shape[0]} rows but the "
        f"plan expects at least {n}. The plan comes from extend_seq_lens_cpu; "
        "if the batch is now NARROWER than that, the two no longer describe "
        "the same forward."
    )
    out_nope[plan.gathered_rows :] = k_nope[:n]
    out_rope[plan.gathered_rows :] = k_pe[:n]


def _dcp_issue_gather_chunks(
    group, out: torch.Tensor, send: torch.Tensor, plan: "DcpPackedReadPlan"
):
    """Fill ``out``'s prefix region with the all-gather of ``send``.

    D2v2: ``plan.chunks > 1`` issues one sub-all-gather per chunk instead of
    one monolithic collective. Chunk k moves local rows [st, st + sz) of every
    rank -- the first chunks-1 hold q = send_rows // chunks rows, the last
    q + send_rows % chunks -- and each lands in the CONTIGUOUS slice
    ``out[C*st : C*(st + sz)]`` of the very buffer the packed read expects
    (chunk-major, see DcpPackedReadPlan), so no extra buffer and no copy. With
    ``plan.chunks == 1`` this is exactly the old single collective.

    The point is not the bytes (they are the same) but the AIV episode: the
    monolithic gather was one 9-16 ms AIV kernel (93% MTE2) that starved the
    compute stream's own AIV work for its whole flight; N sub-collectives of
    roughly 1-2 ms leave the inter-collective windows in which that work runs.
    See SGLANG_NPU_DCP_GATHER_CHUNKS for the measurement.

    Returns the async work handles (empty when issued synchronously); the
    prefetch and eager paths rely on stream ordering and drop them.
    """
    c = plan.dcp_size
    if plan.chunks <= 1:
        group.all_gather_into_tensor(out[: plan.gathered_rows], send)
        return ()
    q, rem = divmod(plan.send_rows, plan.chunks)
    n = plan.chunks
    works = []
    st = 0
    for k in range(n):
        sz = q + (rem if k == n - 1 else 0)
        works.append(
            group.all_gather_into_tensor_async(
                out[c * st : c * (st + sz)], send[st : st + sz]
            )
        )
        st += sz
    return tuple(works)


def _pad_dcp_packed_send(
    shards: torch.Tensor, plan: "DcpPackedReadPlan", dcp_rank: int
) -> torch.Tensor:
    """This rank's send, padded to ``send_rows`` so every rank sends alike.

    A served prefix is a whole number of ``page_size * dcp_size`` allocator
    pages, so it divides by ``dcp_size`` and no padding is needed at all -- the
    shards go to the collective untouched. Only ranks below ``prefix_len %
    dcp_size`` hold the extra row when a prefix is ragged.
    """
    local_len = plan.prefix_len // plan.dcp_size + int(
        dcp_rank < plan.prefix_len % plan.dcp_size
    )
    assert shards.shape[0] == local_len, (
        f"DCP packed read: rank {dcp_rank} holds {shards.shape[0]} prefix rows "
        f"but the plan computes {local_len} from prefix_len={plan.prefix_len} "
        f"over {plan.dcp_size} ranks"
    )
    if local_len == plan.send_rows:
        return shards.contiguous()
    send = shards.new_zeros((plan.send_rows, *shards.shape[1:]))
    send[:local_len] = shards[:local_len]
    return send


def _dcp_gather_extend_kv_npu(
    m: "DeepseekV2AttentionMLA",
    forward_batch: "ForwardBatch",
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    index_topk: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Materialise each request's FULL prefix+extend KV, for one layer.

    Extend gathers the context rather than the query: above a tail of
    prefix/178 it moves fewer bytes (p20_dcp_extend_cost_model.py). The result
    is one contiguous run per request, which the operator needs under the
    non-paged layout -- so not ``all_gather_kv_cache_for_mla_extend``, which
    groups all prefixes then all extends.

    Output and scratch are reused buffers: a moving context-sized allocation
    makes ranks free and refill out of step and stalls the collectives for
    minutes. The keys travel separately because the operator takes them apart.
    """
    parallel = get_parallel()
    md = forward_batch.attn_dcp_metadata
    packed_plan = dcp_packed_read_plan(forward_batch, index_topk)
    if packed_plan is not None:
        return _dcp_gather_extend_kv_packed_npu(
            m, forward_batch, k_nope, k_pe, packed_plan
        )
    plan = getattr(forward_batch, "npu_dcp_extend_gather", None)
    if plan is None:
        plan = plan_dcp_extend_gather(
            forward_batch.extend_prefix_lens_cpu,
            forward_batch.extend_seq_lens_cpu,
            parallel.dcp_size,
            parallel.dcp_rank,
            _dcp_extend_gather_piece_rows,
            dcp_interleave_size(),
        )
        plan = plan._replace(
            pieces=[
                piece._replace(index=piece.index.to(k_nope.device))
                for piece in plan.pieces
            ]
        )
        forward_batch.npu_dcp_extend_gather = plan
        # Wrapper pools (SWA, hybrid) may not offer this.
        plan_write = getattr(get_token_to_kv_pool(), "plan_dcp_extend_write", None)
        if plan_write is not None:
            plan_write(forward_batch.out_cache_loc)
        if _debug_dcp_extend_memory:
            _log_dcp_extend_memory(
                sum(forward_batch.extend_prefix_lens_cpu),
                sum(forward_batch.extend_seq_lens_cpu),
            )
        _log_dcp_shared_prefix(
            forward_batch, (k_nope.shape[-1] + k_pe.shape[-1]) * k_nope.element_size()
        )

    pool = get_token_to_kv_pool()
    # An FP8 DSA cache stores one packed 656-byte record per token (nope, rope,
    # dequant scales), so there is one key to move, in exactly the layout the
    # sparse operator reads at layout_kv TND, and 43% less wire than bf16. F2
    # (the int8 SFA flag + SGLANG_NPU_SFA_KV_QUANT_READ) moves the SAME raw 656
    # B rows: the operator dequantises in-kernel, so the gather ships one opaque
    # uint8 key instead of the F1 dequant-on-read's two bf16 halves.
    f2 = (
        getattr(pool, "sfa_kv_store_8bit", False)
        and envs.SGLANG_NPU_SFA_KV_QUANT_READ.get()
    )
    packed_kv = getattr(pool, "dsa_kv_cache_store_fp8", False) or f2
    if packed_kv:
        # uint8, not float8_e4m3fn: aclnnIndexSelect has no FP8, and the record
        # is opaque until the backend views it back at the operator call. The
        # pack is store-dtype-aware (int8 under F, f8 under the FP8 mode).
        # (rows, 1, kv_cache_dim) matches get_mla_kv_buffer's prefix rows.
        k_nope = pool._pack_dsa_fp8_kv_cache(k_nope, k_pe).view(torch.uint8)
        k_nope = k_nope.unsqueeze(1)
        k_pe = None
    total_rows = plan.pieces[-1].out_end if plan.pieces else 0
    out_nope = dcp_extend_gather_buffer("latent", k_nope, total_rows)
    out_rope = None if packed_kv else dcp_extend_gather_buffer("rope", k_pe, total_rows)

    # Prefetching needs every piece resident at once: one piece only.
    layer_id = m.attn_mqa.layer_id
    prefetching = _prefetch_dcp_extend_gather and len(plan.pieces) == 1
    _log_dcp_gather_prefetch_once(prefetching, plan, packed_kv)
    slot = 0
    state = None
    if prefetching:
        state = getattr(forward_batch, "npu_dcp_gather_prefetch", None)
        if state is None:
            state = _DcpGatherPrefetch(_get_dcp_gather_prefetch_stream())
            forward_batch.npu_dcp_gather_prefetch = state
        slot = state.slot_of(layer_id, pool.start_layer)

    # The prefetch must carry every key this cache uses, or a dropped one is
    # read as whatever the previous layer left in its scratch.
    scratches = _dcp_extend_gather_scratches(
        slot, k_nope, k_pe, plan.scratch_rows, packed_kv
    )

    # Every rank runs (or skips) the same collectives in the same order. Gather
    # and index_select stay INTERLEAVED per piece: pieces share one scratch, so
    # hoisting the gathers would let piece n+1 overwrite piece n unread.
    done = state.pending.pop(layer_id, None) if prefetching else None
    sends = None
    if done is not None:
        torch.npu.current_stream().wait_event(done[1])
    else:
        sends = _dcp_extend_send_rows(
            pool, m.attn_mqa, md, plan, layer_id, packed_kv, f2
        )

    outs = [out_nope] if packed_kv else [out_nope, out_rope]
    for piece in plan.pieces:
        gathered = (piece.send_end - piece.send_start) * parallel.dcp_size
        rows = gathered + piece.extend_end - piece.extend_start
        if sends is not None:
            _dcp_extend_all_gather(
                parallel, piece, [(buf, s) for (buf, _), s in zip(scratches, sends)]
            )
        for out, (buf, own) in zip(outs, scratches):
            scratch = buf[:rows]
            scratch[gathered:] = own[piece.extend_start : piece.extend_end]
            torch.index_select(
                scratch, 0, piece.index, out=out[piece.out_start : piece.out_end]
            )

    if prefetching:
        # The slot is free only after the index_select above.
        release = torch.npu.Event()
        release.record()
        state.release[slot] = release
        _issue_dcp_gather_prefetch(
            state,
            pool,
            m,
            md,
            plan,
            parallel,
            layer_id,
            k_nope,
            k_pe,
            packed_kv,
            f2,
        )
    return out_nope, out_rope


class _DcpDecodeQPre(NamedTuple):
    """The DCP decode Q all-gather, in flight: issued in prepare (right after
    the QKV, so the layer's indexer runs under it), finished in the core's
    decode branch where the synchronous gather used to run.

    ``gathered`` is the all-gather's output, ``[H * dcp_size, B, d_pe +
    d_nope]`` -- the same rank-major layout ``all_gather_q_for_mla_decode``
    builds, so the split below it is the same split. ``send`` is the
    all-gather's input: the communication stream reads it at execution time,
    so it must survive until the core's wait, not just prepare's frame.
    """

    work: Optional[object]  # None when dcp_size is 1 (a copy, nothing to wait on)
    gathered: torch.Tensor
    send: torch.Tensor
    d_pe: int
    d_nope: int


class _DcpSfaGatherPending(NamedTuple):
    """The DCP extend gather, in flight: issued in prepare, finished in core.

    ``works`` are the async all-gather handles (None entries are world-1 no-ops
    and stay skipped, as the collectives they stand in for are). The scratch
    lists are permuting-path only: one gather scratch per piece, because with
    every collective in flight before the elementwise pass, piece i's scratch
    can no longer be recycled into piece i+1 while piece i is still waiting.

    ``k_nope``/``k_pe`` are this chunk's own KV, the one part the collective
    cannot carry: None when the gather was issued before the QKV (a reuse
    layer) and attached by prepare once the QKV has produced them, or the real
    tensors straight away when the gather is issued after the QKV (an indexer
    layer).
    """

    kind: str  # "permuting" or "packed"
    works: list
    plan: object  # DcpExtendGatherPlan or DcpPackedReadPlan
    out_nope: torch.Tensor
    out_rope: torch.Tensor
    k_nope: Optional[torch.Tensor] = None
    k_pe: Optional[torch.Tensor] = None
    scratch_nope: Optional[list] = None
    scratch_rope: Optional[list] = None


def _dcp_start_extend_gather_permuting_npu(
    m: "DeepseekV2AttentionMLA",
    forward_batch: "ForwardBatch",
    hidden_states: torch.Tensor,
    k_nope: Optional[torch.Tensor] = None,
    k_pe: Optional[torch.Tensor] = None,
) -> _DcpSfaGatherPending:
    """Issue the permuting-path gather now (before the QKV on a reuse layer,
    before the indexer on an indexer layer); its elementwise tail runs later.

    Everything the sync path did before and between its collectives happens
    here: the once-per-forward plan (and its write-plan and buffer-release
    duties), the send tensors, the reserved output and scratch. What moves to
    ``_dcp_finish_extend_gather_npu`` is the per-piece copy-plus-index_select,
    which is what waits on the gathered bytes. The buffers are typed from this
    chunk's KV when it already exists and from ``hidden_states`` otherwise.
    """
    parallel = get_parallel()
    md = forward_batch.attn_dcp_metadata
    plan = getattr(forward_batch, "npu_dcp_extend_gather", None)
    if plan is None:
        md.dcp_kv_buffer = None
        plan = plan_dcp_extend_gather(
            forward_batch.extend_prefix_lens_cpu,
            forward_batch.extend_seq_lens_cpu,
            parallel.dcp_size,
            parallel.dcp_rank,
            _dcp_extend_gather_piece_rows,
        )
        plan = plan._replace(
            pieces=[
                piece._replace(index=piece.index.to(hidden_states.device))
                for piece in plan.pieces
            ]
        )
        forward_batch.npu_dcp_extend_gather = plan
        plan_write = getattr(get_token_to_kv_pool(), "plan_dcp_extend_write", None)
        if plan_write is not None:
            plan_write(forward_batch.out_cache_loc)
        if _debug_dcp_extend_memory:
            _log_dcp_extend_memory(
                sum(forward_batch.extend_prefix_lens_cpu),
                sum(forward_batch.extend_seq_lens_cpu),
            )

    ref_nope = (k_nope if k_nope is not None else hidden_states).new_empty(
        (1, 1, m.kv_lora_rank)
    )
    ref_rope = (k_pe if k_pe is not None else hidden_states).new_empty(
        (1, 1, m.qk_rope_head_dim)
    )
    total_rows = plan.pieces[-1].out_end if plan.pieces else 0
    out_nope = dcp_extend_gather_buffer("latent", ref_nope, total_rows)
    out_rope = dcp_extend_gather_buffer("rope", ref_rope, total_rows)

    scratch_nope: list = []
    scratch_rope: list = []
    for i, piece in enumerate(plan.pieces):
        rows = (
            (piece.send_end - piece.send_start) * parallel.dcp_size
            + piece.extend_end
            - piece.extend_start
        )
        scratch_nope.append(
            dcp_extend_gather_buffer(f"latent_scratch_{i}", ref_nope, rows)
        )
        scratch_rope.append(
            dcp_extend_gather_buffer(f"rope_scratch_{i}", ref_rope, rows)
        )

    works: list = []
    if plan.send_rows:
        send_nope, send_rope = get_token_to_kv_pool().get_mla_kv_buffer(
            m.attn_mqa,
            md.dcp_local_prefix_kv_indices,
        )
        if plan.local_lens != plan.padded_lens:
            send_nope = _pad_dcp_extend_send(send_nope, plan)
            send_rope = _pad_dcp_extend_send(send_rope, plan)
        for i, piece in enumerate(plan.pieces):
            gathered = (piece.send_end - piece.send_start) * parallel.dcp_size
            if gathered:
                works.append(
                    parallel.dcp_group.all_gather_into_tensor_async(
                        scratch_nope[i][:gathered],
                        send_nope[piece.send_start : piece.send_end],
                    )
                )
                works.append(
                    parallel.dcp_group.all_gather_into_tensor_async(
                        scratch_rope[i][:gathered],
                        send_rope[piece.send_start : piece.send_end],
                    )
                )
    return _DcpSfaGatherPending(
        "permuting", works, plan, out_nope, out_rope, k_nope, k_pe,
        scratch_nope, scratch_rope,
    )


def _dcp_start_extend_gather_packed_npu(
    m: "DeepseekV2AttentionMLA",
    forward_batch: "ForwardBatch",
    hidden_states: torch.Tensor,
    plan: "DcpPackedReadPlan",
    k_nope: Optional[torch.Tensor] = None,
    k_pe: Optional[torch.Tensor] = None,
) -> _DcpSfaGatherPending:
    """Issue the packed-path gather now (before the QKV on a reuse layer,
    before the indexer on an indexer layer); the extend tail is written later.

    The packed path needs no scratch at all: the collective lands directly in
    the buffer the operator reads, so the only thing that moves to the finish
    is the local copy of this chunk's own KV after the gathered prefix.
    """
    parallel = get_parallel()
    md = forward_batch.attn_dcp_metadata
    if not getattr(forward_batch, "npu_dcp_packed_ready", False):
        forward_batch.npu_dcp_packed_ready = True
        md.dcp_kv_buffer = None
        plan_write = getattr(get_token_to_kv_pool(), "plan_dcp_extend_write", None)
        if plan_write is not None:
            plan_write(forward_batch.out_cache_loc)
        if _debug_dcp_extend_memory:
            _log_dcp_extend_memory(plan.prefix_len, plan.extend_len)

    pool = get_token_to_kv_pool()
    # BUG 15: under F2 ship raw 656-byte uint8 rows (one key, no rope).
    _f2 = (
        getattr(pool, "sfa_kv_store_8bit", False)
        and envs.SGLANG_NPU_SFA_KV_QUANT_READ.get()
    )
    works: list = []
    if _f2:
        _ref = hidden_states.new_empty((1, 1, pool.kv_cache_dim), dtype=torch.uint8)
        out_nope = dcp_extend_gather_buffer("packed_latent", _ref, plan.rows)
        out_rope = None
        if plan.gathered_rows:
            send = pool.gather_packed_sfa_rows_raw(
                m.layer_id, md.dcp_local_prefix_kv_indices
            ).unsqueeze(1)
            send = _pad_dcp_packed_send(send, plan, parallel.dcp_rank)
            works.append(
                parallel.dcp_group.all_gather_into_tensor_async(
                    out_nope[: plan.gathered_rows], send
                )
            )
        return _DcpSfaGatherPending(
            "packed", works, plan, out_nope, out_rope, k_nope, k_pe
        )

    ref_nope = (k_nope if k_nope is not None else hidden_states).new_empty(
        (1, 1, m.kv_lora_rank)
    )
    ref_rope = (k_pe if k_pe is not None else hidden_states).new_empty(
        (1, 1, m.qk_rope_head_dim)
    )
    out_nope = dcp_extend_gather_buffer("packed_latent", ref_nope, plan.rows)
    out_rope = dcp_extend_gather_buffer("packed_rope", ref_rope, plan.rows)

    if plan.gathered_rows:
        send_nope, send_rope = pool.get_mla_kv_buffer(
            m.attn_mqa, md.dcp_local_prefix_kv_indices
        )
        send_nope = _pad_dcp_packed_send(send_nope, plan, parallel.dcp_rank)
        send_rope = _pad_dcp_packed_send(send_rope, plan, parallel.dcp_rank)
        works.append(
            parallel.dcp_group.all_gather_into_tensor_async(
                out_nope[: plan.gathered_rows], send_nope
            )
        )
        works.append(
            parallel.dcp_group.all_gather_into_tensor_async(
                out_rope[: plan.gathered_rows], send_rope
            )
        )
    return _DcpSfaGatherPending(
        "packed", works, plan, out_nope, out_rope, k_nope, k_pe
    )


def _dcp_start_extend_gather_npu(
    m: "DeepseekV2AttentionMLA",
    forward_batch: "ForwardBatch",
    hidden_states: torch.Tensor,
    index_topk: Optional[int],
    k_nope: Optional[torch.Tensor] = None,
    k_pe: Optional[torch.Tensor] = None,
) -> Optional[_DcpSfaGatherPending]:
    """Start this layer's SFA KV gather, in flight under the QKV (reuse layer)
    or under the indexer (indexer layer, the vLLM placement).

    Returns None when the layer should take the sync path instead: the existing
    side-stream prefetch (SGLANG_NPU_ENABLE_DCP_GATHER_PREFETCH) owns the packed
    path when it is on (it starts the same gather a layer earlier).
    """
    print_info_once(
        "DCP SFA gather overlap ACTIVE (async in-layer): dcp_size="
        f"{get_parallel().dcp_size}"
    )
    packed_plan = dcp_packed_read_plan(forward_batch, index_topk)
    if packed_plan is not None:
        if _enable_dcp_gather_prefetch:
            print_info_once(
                "DCP SFA gather overlap defers to the side-stream prefetch "
                "on the packed path (SGLANG_NPU_ENABLE_DCP_GATHER_PREFETCH), "
                "which starts the same gather a layer earlier"
            )
            return None
        return _dcp_start_extend_gather_packed_npu(
            m, forward_batch, hidden_states, packed_plan, k_nope, k_pe
        )
    return _dcp_start_extend_gather_permuting_npu(
        m, forward_batch, hidden_states, k_nope, k_pe
    )


def _dcp_finish_extend_gather_npu(
    pending: _DcpSfaGatherPending,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Wait for the in-flight gather and run the elementwise tail (the vLLM
    wait point, right before the sparse operator reads the buffer)."""
    for work in pending.works:
        if work is not None:
            work.wait()
    if pending.k_nope is None or pending.k_pe is None:
        raise AssertionError(
            "DCP SFA gather overlap: the gather was finished before prepare "
            "attached this chunk's KV; the QKV output must be attached in "
            "prepare, after the QKV, before the core waits"
        )
    if pending.kind == "packed":
        if pending.out_rope is None:
            # BUG 15: F2 packed path -- pack the bf16 tail into raw uint8.
            pool = get_token_to_kv_pool()
            n = pending.plan.extend_len
            packed_tail = pool.pack_sfa_kv_cache(
                pending.k_nope[:n], pending.k_pe[:n]
            ).view(torch.uint8).unsqueeze(1)
            pending.out_nope[pending.plan.gathered_rows :] = packed_tail
        else:
            _write_packed_extend_tail(
                pending.out_nope,
                pending.out_rope,
                pending.k_nope,
                pending.k_pe,
                pending.plan,
            )
        return pending.out_nope, pending.out_rope

    parallel = get_parallel()
    plan = pending.plan
    for i, piece in enumerate(plan.pieces):
        gathered = (piece.send_end - piece.send_start) * parallel.dcp_size
        for out, scratch, own in (
            (pending.out_nope, pending.scratch_nope[i], pending.k_nope),
            (pending.out_rope, pending.scratch_rope[i], pending.k_pe),
        ):
            scratch = scratch[
                : gathered + piece.extend_end - piece.extend_start
            ]
            scratch[gathered:] = own[piece.extend_start : piece.extend_end]
            torch.index_select(
                scratch, 0, piece.index, out=out[piece.out_start : piece.out_end]
            )
    return pending.out_nope, pending.out_rope


def _issue_dcp_gather_prefetch(
    state: "_DcpGatherPrefetch",
    pool,
    m: "DeepseekV2AttentionMLA",
    md,
    plan,
    parallel,
    layer_id: int,
    k_nope: torch.Tensor,
    k_pe: Optional[torch.Tensor],
    packed_kv: bool,
    f2: bool = False,
) -> None:
    """Start the next layer's prefix all-gather(s) on the side stream."""
    nxt = layer_id + 1
    if nxt > pool.start_layer + pool.layer_num - 1 or nxt in state.pending:
        return
    slot = state.slot_of(nxt, pool.start_layer)
    scratches = _dcp_extend_gather_scratches(
        slot, k_nope, k_pe, plan.scratch_rows, packed_kv
    )
    release = state.release.get(slot)
    with torch.npu.stream(state.stream):
        if release is not None:
            state.stream.wait_event(release)
        sends = _dcp_extend_send_rows(
            pool, m.attn_mqa, md, plan, nxt, packed_kv, f2
        )
        pairs = [(scratch, s) for (scratch, _), s in zip(scratches, sends)]
        for piece in plan.pieces:
            _dcp_extend_all_gather(parallel, piece, pairs)
        for send in sends:
            if send is not None:
                send.record_stream(state.stream)
        ready = state.stream.record_event()
    state.pending[nxt] = (slot, ready, sends)


def forward_dsa_core_npu(
    m: "DeepseekV2AttentionMLA",
    q_pe: torch.Tensor,
    k_pe: torch.Tensor,
    q_nope_out: torch.Tensor,
    k_nope: torch.Tensor,
    topk_indices: torch.Tensor,
    forward_batch: "ForwardBatch",
    zero_allocator: "BumpAllocator",
    positions: torch.Tensor,
    mla_preprocess_used: bool,
    # Gated attention (Ling-V3 / BailingMoeV3): the subclass appends its gate
    # to inner_state, so every *_core dispatched from forward_core takes it as
    # a trailing arg. None everywhere else.
    gate: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Only the narrow token-shard path sets this; DCP never does.
    w_vc_applied = False
    # GLM-5.2 dispatches DSA_NPU here, not to forward_absorb_core, so
    # forward_mla.py's DCP block never runs for it.
    if dsa_cp_token_sharded(forward_batch):
        # Wait on the chunk-KV allGather issued in prepare, the last moment
        # before anything reads the gathered rows: the DCP gather's local chunk
        # copy, the DCP SFA-gather finish, and the pool write inside the sparse
        # operator all run from here on. The stash also holds the send buffers,
        # which must survive until the communication engine has read them.
        works = getattr(forward_batch, "npu_dsa_cp_chunk_kv_works", None)
        if works is not None:
            for work in works[:2]:
                if work is not None:
                    work.wait()
            print_info_once(
                "D1v3: the chunk-KV allGather (seam S1) issued in prepare is "
                "done; the gathered KV rows are now readable on every rank."
            )
            forward_batch.npu_dsa_cp_chunk_kv_works = None

    dcp_extend = _dcp_extend_active(forward_batch)
    if (
        not dcp_extend
        and get_parallel().dcp_enabled
        and forward_batch.forward_mode.is_extend()
        and not is_dcp_mla_decode_phase(forward_batch)
        and not get_attn_backend().is_draft_worker
    ):
        # Without the metadata the backend reads this rank's shard through a
        # full-span page table and returns plausible output. The merge of
        # #37787 did exactly that, silently, by returning None for all NPU.
        raise RuntimeError(
            "DCP extend on the NPU DSA path has no attn_dcp_metadata; "
            "prepare_context_parallel_metadata_for_dcp must build it for DSA "
            "extend on NPU"
        )
    if dcp_extend:
        pending = getattr(forward_batch, "npu_dcp_sfa_gather_pending", None)
        forward_batch.npu_dcp_sfa_gather_pending = None
        if pending is not None:
            # The gather was issued at the top of prepare, before the QKV, and
            # has been moving bytes underneath the QKV and (where the layer
            # runs one) the indexer: wait for it NOW, the last moment before
            # the sparse operator reads the buffer (the vLLM wait point).
            forward_batch.npu_dcp_extend_kv = _dcp_finish_extend_gather_npu(pending)
        else:
            forward_batch.npu_dcp_extend_kv = _dcp_gather_extend_kv_npu(
                m,
                forward_batch,
                k_nope,
                k_pe,
                topk_indices.shape[-1] if topk_indices is not None else None,
            )

    if is_dcp_mla_decode_phase(forward_batch):
        q_pre = getattr(forward_batch, "npu_dcp_decode_q_pre", None)
        if q_pre is not None:
            # Issued in prepare, under the layer's indexer: wait for it now,
            # where the synchronous gather used to run, and split the gathered
            # buffer the same way the synchronous path does.
            forward_batch.npu_dcp_decode_q_pre = None
            if q_pre.work is not None:
                q_pre.work.wait()
            gathered_pe, gathered_nope = q_pre.gathered.split(
                [q_pre.d_pe, q_pre.d_nope], dim=-1
            )
            q_pe = gathered_pe.transpose(0, 1)
            q_nope_out = gathered_nope.transpose(0, 1)
        else:
            q_nope_out, q_pe = all_gather_q_for_mla_decode(
                q_nope_out=q_nope_out, q_pe=q_pe
            )
        # save_kv_cache stays True here: the DCP write is idempotent.
        attn_output, lse = m.attn_mqa_for_dcp_decode(
            q_nope_out.contiguous(),
            k_nope.contiguous(),
            k_nope.contiguous(),
            forward_batch,
            save_kv_cache=True,
            q_rope=q_pe.contiguous(),
            k_rope=k_pe.contiguous(),
            topk_indices=topk_indices,
        )
        attn_output = attn_output.view(
            -1, m.num_local_heads * get_parallel().attn_dcp_size, m.kv_lora_rank
        )
        # Returns the local head slice already, in natural log on both sides.
        attn_output = cp_lse_ag_out_rs_mla_npu(
            attn_output, lse, get_parallel().dcp_group
        )
    else:
        attn_mqa = m.attn_mqa
        dsa_token_shard_plan = get_dsa_token_shard_plan(forward_batch)
        # D1v3: the token-sharded rebuild + the version B full-width Q. Defined
        # here so the redistribute, the restore, and the o_proj all see them.
        token_sharded = dsa_cp_token_sharded(forward_batch)
        # BUG 5: a plan can be cached for a forward the gate refused (e.g.
        # logprob prefill). The core must branch on the per-forward decision,
        # not plan presence: demote the plan when this forward is NOT
        # token-sharded so every branch below falls through to base DCP.
        if dsa_token_shard_plan is not None and not token_sharded:
            dsa_token_shard_plan = None
        # D2v4: the streamed q_b carries its shard, not the static qrep, so the
        # full-width presence check follows which one the preparation stood --
        # the same stream-aware expression the prepare's q_b_present uses, or
        # the core would fall into the head-sharded redistribute on an already
        # full-head token-sharded q.
        q_full_width = (
            (_enable_dsa_cp_q_full_width or token_sharded)
            and not mla_preprocess_used
            and (
                getattr(m, "q_b_stream_shard", None) is not None
                if _stream_q_b
                else getattr(m, "q_b_proj_qrep_weight", None) is not None
            )
        )
        if dsa_token_shard_plan is not None and (
            topk_indices is None or m.attn_mqa_for_dsa_token_shard is None
        ):
            raise RuntimeError(
                "DSA token-shard planned this forward but the layer is not set up for "
                f"it: attn_mqa_for_dsa_token_shard={m.attn_mqa_for_dsa_token_shard is not None}, "
                f"topk_indices={topk_indices is not None}"
            )
        narrow_a2a = (
            dsa_token_shard_plan is not None
            and _dsa_token_shard_narrow_plan(m, forward_batch) is not None
        )
        if dsa_token_shard_plan is not None:
            if q_full_width:
                # Version B / token sharded: prepare computed every head for
                # this rank's rows with the gathered full-head weights, so there
                # is nothing to redistribute. The width the restore hands back
                # is the batch's padded width, not this rank's row count.
                dsa_token_shard_rows = dsa_token_shard_plan.num_tokens_pad
            elif narrow_a2a:
                # prepare already swapped, before the absorb.
                dsa_token_shard_rows = forward_batch.npu_dsa_token_shard_input_rows
            else:
                dsa_token_shard_rows = q_nope_out.shape[0]
                q_pre = getattr(forward_batch, "npu_dsa_cp_q_pre", None)
                if q_pre is not None:
                    # Version A: both all-to-alls ran under the indexer; wait on
                    # them now, and only then materialise the recv buffers (their
                    # permute+reshape is a copy that must not run while the
                    # collectives are still writing).
                    forward_batch.npu_dsa_cp_q_pre = None
                    for work in q_pre[2:4]:
                        if work is not None:
                            work.wait()
                    q_nope_out = dsa_cp_redistribute_heads_recv(
                        q_pre[0], dsa_token_shard_plan
                    )
                    q_pe = dsa_cp_redistribute_heads_recv(
                        q_pre[1], dsa_token_shard_plan
                    )
                else:
                    q_nope_out = dsa_token_shard_redistribute_heads(
                        q_nope_out, dsa_token_shard_plan
                    )
                    q_pe = dsa_token_shard_redistribute_heads(
                        q_pe, dsa_token_shard_plan
                    )
            # A separate name is load-bearing: topk_indices is returned for the
            # next layer, and rebinding it would hand that a slice of a slice.
            if (
                token_sharded
                or (_enable_dsa_cp_local_topk
                    and getattr(forward_batch, "npu_dsa_topk_local", False))
                or getattr(forward_batch, "npu_indexer_topk_is_local", False)
            ):
                # The indexer kept its top-k in this rank's query rows (the plan's
                # own split), so there is nothing to slice. Checked, not assumed:
                # a full-width tensor here gives every rank but 0 the wrong rows
                # with no shape error.
                if topk_indices.shape[0] != dsa_token_shard_plan.rows:
                    raise RuntimeError(
                        "top-k is local to this rank but carries "
                        f"{topk_indices.shape[0]} rows, not the plan's "
                        f"{dsa_token_shard_plan.rows}"
                    )
                attn_topk_indices = topk_indices
            else:
                attn_topk_indices = dsa_token_shard_slice(
                    topk_indices, dsa_token_shard_plan
                )
            attn_mqa = m.attn_mqa_for_dsa_token_shard
        else:
            attn_topk_indices = topk_indices
        attn_output = attn_mqa(
            q_nope_out.contiguous(),
            k_nope.contiguous(),
            k_nope.contiguous(),
            forward_batch,
            save_kv_cache=not mla_preprocess_used,
            q_rope=q_pe.contiguous(),
            k_rope=k_pe.contiguous(),
            topk_indices=attn_topk_indices,
        )
        if dsa_token_shard_plan is not None and not token_sharded:
            # Undo the swap before anything else sees it. w_vc, o_proj and the
            # layer communicator all expect this rank's own heads for the whole
            # batch, which is what they get without DSA-CP, so the sharding ends
            # here. Token sharded has no swap to undo: the full w_vc and the
            # full o_proj below consume rows x all heads and emit the COMPLETE
            # [rows, hidden] result, so the sharding propagates through the rest
            # of the layer instead (the MoE gathers it back to full width).
            attn_output = attn_output.reshape(
                dsa_token_shard_plan.rows, -1, m.kv_lora_rank
            )
            if narrow_a2a:
                # w_vc_full: this rank holds every head right now.
                attn_output = torch_npu.npu_transpose_batchmatmul(
                    attn_output.contiguous(),
                    m.w_vc_full,
                    perm_x1=(1, 0, 2),
                    perm_x2=(0, 1, 2),
                    perm_y=(1, 0, 2),
                )
                w_vc_applied = True
            attn_output = dsa_token_shard_restore_tokens(
                attn_output,
                dsa_token_shard_plan,
                dsa_token_shard_rows,
            )
    if dcp_extend:
        # So a later forward cannot read a stale gather.
        forward_batch.npu_dcp_extend_kv = None
    if w_vc_applied:
        attn_bmm_output = attn_output
    else:
        # Token sharded keeps rows x ALL heads here (no restore a2a ran), so the
        # absorb and the o_proj take the full replicated weights and emit the
        # complete [rows, hidden] result; everything else keeps this rank's head
        # shard.
        if dsa_cp_token_sharded(forward_batch):
            attn_output = attn_output.view(-1, m.num_heads, m.kv_lora_rank)
            # D2v4: gather the full w_vc on the fly when streamed; otherwise
            # the standing w_vc_qrep.
            w_vc = _dcp_w_vc_weight(m) if _stream_w_vc else m.w_vc_qrep
        else:
            attn_output = attn_output.view(-1, m.num_local_heads, m.kv_lora_rank)
            w_vc = m.w_vc

        if _is_npu_arch35 or (
            forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_draft_extend_v2()
            and not forward_batch.forward_mode.is_target_verify()
        ):
            attn_bmm_output = torch_npu.npu_transpose_batchmatmul(
                attn_output,
                w_vc,
                perm_x1=(1, 0, 2),
                perm_x2=(0, 1, 2),
                perm_y=(1, 0, 2),
            )
        else:
            attn_bmm_output = torch.empty(
                (attn_output.shape[0], attn_output.shape[1], m.v_head_dim),
                dtype=attn_output.dtype,
                device=attn_output.device,
            )
            attn_output = attn_output.contiguous()
            torch.ops.npu.batch_matmul_transpose(attn_output, w_vc, attn_bmm_output)

    attn_bmm_output = attn_bmm_output.reshape(-1, attn_output.shape[1] * m.v_head_dim)

    if gate is not None:
        attn_bmm_output = m._apply_gated(attn_bmm_output, gate)
    if dsa_cp_token_sharded(forward_batch):
        # Full W_O: the complete result for this rank's rows, no allReduce.
        if _stream_o_proj:
            # D2v3: the full matrix was assembled on the gather side stream
            # one layer ahead; take_o_proj waits on its ready event here, the
            # last moment before this GEMM reads the slot.
            print_info_once(
                "D2v3: o_proj runs the STREAMED full W_O (all-gathered on the "
                "DCP side stream after this layer's KV gather), so the "
                "attention output is complete and the o_proj allReduce is "
                "skipped, with no standing per-device replica."
            )
            output = torch.nn.functional.linear(
                attn_bmm_output, _dcp_o_proj_weight(m)
            )
        else:
            print_info_once(
                "D1v3: o_proj runs the full replicated W_O (o_proj_qrep_weight) "
                "on this rank's rows, so the attention output is complete and the "
                "o_proj allReduce is skipped."
            )
            output = torch.nn.functional.linear(attn_bmm_output, m.o_proj_qrep_weight)
    else:
        output, _ = m.o_proj(attn_bmm_output)
    if not m.next_skip_topk:
        return output, None
    else:
        return output, topk_indices


def npu_mla_preprocess(
    m: "DeepseekV2AttentionMLA",
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: "ForwardBatch",
    zero_allocator: "BumpAllocator",
):
    dynamic_scale = None
    if not hasattr(m, "mla_preprocess"):
        m.mla_preprocess = NPUFusedMLAPreprocess(
            m.fused_qkv_a_proj_with_mqa,
            m.q_a_layernorm,
            m.kv_a_layernorm,
            m.q_b_proj,
            m.w_kc,
            m.rotary_emb,
            m.layer_id,
            m.num_local_heads,
            m.qk_nope_head_dim,
            m.qk_rope_head_dim,
            m.v_head_dim,
            m.quant_config,
        )
        if (
            get_disagg().disaggregation_mode == "decode"
            and m.mla_preprocess.uses_mlaprolog()
            and m.mla_preprocess.weight_quant_mode != 2
            and m.w_kc is not None
        ):
            m.w_kc.untyped_storage().resize_(0)
    # MLAProlog returns q_lora; BF16 indexers need it recomputed below.
    if m.mla_preprocess.uses_mlaprolog():
        (
            q_pe,
            k_pe,
            q_nope_out,
            k_nope,
            q_lora,
            forward_batch,
            positions,
            dynamic_scale,
        ) = m.mla_preprocess.forward(
            positions, hidden_states, forward_batch, zero_allocator
        )
        if m.mla_preprocess.weight_quant_mode == 2 and m.indexer is not None:
            from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
                NPUW8A8Int8DynamicLinearMethod,
            )

            if not isinstance(
                getattr(m.indexer.wq_b.scheme, "kernel", None),
                NPUW8A8Int8DynamicLinearMethod,
            ):
                # BF16 indexers need q_lora before INT8 quantization.
                q = m.fused_qkv_a_proj_with_mqa(hidden_states)[0][..., : m.q_lora_rank]
                q_lora = m.q_a_layernorm(q)
                dynamic_scale = None
    else:
        if m.alt_stream is not None:
            mla_event = torch.npu.Event()
            mla_event.record()
            with torch.npu.stream(m.alt_stream):
                # alt stream waits for the completion of the event on the main stream to ensure data dependency is complete
                torch.npu.current_stream().wait_event(mla_event)
                (
                    q_pe,
                    k_pe,
                    q_nope_out,
                    k_nope,
                    forward_batch,
                    zero_allocator,
                    positions,
                ) = m.mla_preprocess.forward(
                    positions, hidden_states, forward_batch, zero_allocator
                )

            fused_qkv_a_proj_out = m.fused_qkv_a_proj_with_mqa(hidden_states)[0]
            q, _ = fused_qkv_a_proj_out.split(
                [m.q_lora_rank, m.kv_lora_rank + m.qk_rope_head_dim], dim=-1
            )
            q_lora = m.q_a_layernorm(q)
            torch.npu.current_stream().wait_event(m.alt_stream)
        else:
            (
                q_pe,
                k_pe,
                q_nope_out,
                k_nope,
                forward_batch,
                zero_allocator,
                positions,
            ) = m.mla_preprocess.forward(
                positions, hidden_states, forward_batch, zero_allocator
            )
            fused_qkv_a_proj_out = m.fused_qkv_a_proj_with_mqa(hidden_states)[0]
            q, _ = fused_qkv_a_proj_out.split(
                [m.q_lora_rank, m.kv_lora_rank + m.qk_rope_head_dim], dim=-1
            )
            q_lora = m.q_a_layernorm(q)

    return (
        q_pe,
        k_pe,
        q_nope_out,
        k_nope,
        q_lora,
        forward_batch,
        zero_allocator,
        positions,
        dynamic_scale,
    )


# endregion
