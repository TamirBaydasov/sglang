from typing import TYPE_CHECKING, Optional, Sequence

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.environ import envs
from sglang.srt.layers.dcp.layout import (
    dcp_interleave_size,
    dcp_owner_and_row,
    localize_dcp_indices,
    plan_dcp_owner_write,
)
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKOnlyPool,
    MHATokenToKVPool,
    MiniMaxSparseKVPool,
    MLATokenToKVPool,
    get_tensor_size_bytes,
    unwrap_write_loc,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import get_bool_env_var
from sglang.srt.utils.async_probe import maybe_detect_oob
from sglang.srt.utils.common import is_npu, print_info_once

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention

if is_npu():
    import torch_npu


def _mla_fia_nz_scatter_indices(
    loc: torch.Tensor, head_dim: int, page_size: int
) -> torch.Tensor:
    """Return physical rows for token-wise writes into an MLA NZ cache.

    The storage allocation remains page-major ``[page, slot, 1, D]`` for
    transfer and bookkeeping compatibility. FIA reads that storage as
    ``[page, 1, D / 16, page_size, 16]``. A token-major scatter would therefore
    write the wrong physical rows, so every logical token expands to its
    ``D / 16`` NZ tiles.
    """
    if head_dim % 16:
        raise ValueError(
            "FIA NZ MLA cache requires a head dimension divisible by 16, "
            f"got {head_dim}."
        )
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}.")

    num_tiles = head_dim // 16
    page = torch.div(loc, page_size, rounding_mode="floor")
    slot = torch.remainder(loc, page_size)
    tiles = torch.arange(num_tiles, dtype=loc.dtype, device=loc.device)
    # Flatten [token, tile] in the same order as source.view(T, tiles, 16).
    rows = ((page[:, None] * num_tiles + tiles) * page_size) + slot[:, None]
    return rows.reshape(-1, 1)


def _init_npu_conv_state(
    conv_state_in,
    conv_state_shape,
    speculative_num_draft_tokens: Optional[int] = None,
    is_kda: bool = False,
):
    extra_conv_len = 0
    if speculative_num_draft_tokens is not None:
        extra_conv_len = speculative_num_draft_tokens - 1

    # Both KDA and Mamba/GDN NPU conv states use the unified
    # [layers, pool, window, channels] layout. KDA shapes arrive as
    # (window, channels) while Mamba/GDN shapes arrive as (channels, window);
    # resolve the correct axis ordering and extend the window by
    # speculative_num_draft_tokens - 1 so that verify can write all draft
    # token conv states directly into conv_states (GDN rollback scheme).
    conv_state = [
        torch.zeros(
            size=(
                conv_state_in.shape[0],
                conv_state_in.shape[1],
                (conv_shape[0] if is_kda else conv_shape[1]) + extra_conv_len,
                conv_shape[1] if is_kda else conv_shape[0],
            ),
            dtype=conv_state_in.dtype,
            device=conv_state_in.device,
        )
        for conv_shape in conv_state_shape
    ]
    return conv_state


class NPUMHATokenToKVPool(MHATokenToKVPool):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        v_head_dim: Optional[int] = None,
        swa_head_num: Optional[int] = None,
        swa_head_dim: Optional[int] = None,
        swa_v_head_dim: Optional[int] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_alt_stream: bool = True,
        enable_kv_cache_copy: bool = False,
        **kwargs,
    ):
        self.use_fia = get_bool_env_var("ASCEND_USE_FIA", "False")
        self.use_scatter_pa_kv_cache = envs.SGLANG_NPU_USE_SCATTER_PA_KV_CACHE.get()
        self.use_triton_prefix_kv_cache_store = (
            envs.SGLANG_NPU_USE_TRITON_PREFIX_KV_CACHE_STORE.get()
        )
        super().__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            v_head_dim=v_head_dim,
            swa_head_num=swa_head_num,
            swa_head_dim=swa_head_dim,
            swa_v_head_dim=swa_v_head_dim,
            start_layer=start_layer,
            end_layer=end_layer,
            enable_alt_stream=enable_alt_stream,
            enable_kv_cache_copy=enable_kv_cache_copy,
            **kwargs,
        )

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            # [size, head_num, head_dim] for each layer
            # The padded slot 0 is used for writing dummy outputs from padded tokens.
            # Continuous memory improves the efficiency of Ascend`s transmission backend,
            # while other backends remain unchanged.
            # FIA exposes the KV cache as per-layer Python views so graph
            # capture does not retain the full multi-layer tensor. HiCache's
            # NPU exchange operator still requires the original contiguous
            # [layer, page, token, head, dim] allocation.
            self._hicache_k_buffer = torch.zeros(
                (
                    self.layer_num,
                    self.size // self.page_size + 1,
                    self.page_size,
                    self.head_num,
                    self.head_dim,
                ),
                dtype=self.store_dtype,
                device=self.device,
            )
            self._hicache_v_buffer = torch.zeros(
                (
                    self.layer_num,
                    self.size // self.page_size + 1,
                    self.page_size,
                    self.head_num,
                    self.v_head_dim,
                ),
                dtype=self.store_dtype,
                device=self.device,
            )
            # Keep a reference to the contiguous tensor for HiCache
            # D2H/H2D transfers (transfer_kv_dim_exchange expects a
            # tensor, not the per-layer list used in FIA mode below).
            self.k_buffer_tensor = self._hicache_k_buffer
            self.v_buffer_tensor = self._hicache_v_buffer

            self.k_buffer = self._hicache_k_buffer
            self.v_buffer = self._hicache_v_buffer

            if self.use_fia:
                # Use per-layer Python lists to avoid torch.compile capturing
                # the entire multi-layer tensor (OOM during graph capture).
                # Each layer view: [P*ps, 1, H, D], sharing the contiguous
                # storage allocated above.
                self.k_buffer = [
                    self._hicache_k_buffer[i].view(-1, 1, self.head_num, self.head_dim)
                    for i in range(self.layer_num)
                ]
                self.v_buffer = [
                    self._hicache_v_buffer[i].view(
                        -1, 1, self.head_num, self.v_head_dim
                    )
                    for i in range(self.layer_num)
                ]

    def get_hicache_transfer_buffers(self):
        """Return contiguous all-layer KV tensors for NPU HiCache IO."""
        return self._hicache_k_buffer, self._hicache_v_buffer

    def _mha_buffer_info(self):
        from sglang.srt.mem_cache.device_pool_info import (
            KVBufferLayout,
            KVBuffers,
            MHABufferInfo,
        )

        return MHABufferInfo(
            buffers=KVBuffers(
                k=tuple(self.k_buffer),
                v=tuple(self.v_buffer),
                k_data_ptrs=None,
                v_data_ptrs=None,
                layout=KVBufferLayout.PAGED_NHD,
            ),
            page_size=self.page_size,
            head_num=self.row_dim // self.head_dim
            if self.head_dim == self.v_head_dim
            else self.head_num,
            head_dim=self.head_dim,
            v_head_dim=self.v_head_dim,
            dtype=self.store_dtype,
            # self.device is the bare backend name; tensors carry the rank index.
            device=self._hicache_k_buffer.device,
            transfer_buffers=self.get_hicache_transfer_buffers,
        )

    def _init_kv_copy_and_warmup(self):
        # implementation relies on self.data_strides / self.data_ptrs, which the
        # NPU paged buffer layout never builds.
        self._kv_copy_config = None

    # for disagg
    def get_contiguous_buf_infos(self):
        # layer_num x [seq_len, head_num, head_dim]
        # layer_num x [page_num, page_size, head_num, head_dim]
        kv_data_ptrs = [
            self.get_key_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self.get_value_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_data_lens = [
            self.get_key_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self.get_value_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        if self.use_fia:
            kv_item_lens = [
                self.get_key_buffer(i)[0].nbytes * self.page_size
                for i in range(self.start_layer, self.start_layer + self.layer_num)
            ] + [
                self.get_value_buffer(i)[0].nbytes * self.page_size
                for i in range(self.start_layer, self.start_layer + self.layer_num)
            ]
        else:
            kv_item_lens = [
                self.get_key_buffer(i)[0].nbytes
                for i in range(self.start_layer, self.start_layer + self.layer_num)
            ] + [
                self.get_value_buffer(i)[0].nbytes
                for i in range(self.start_layer, self.start_layer + self.layer_num)
            ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def set_kv_buffer(
        self,
        layer: "RadixAttention",
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
        dcp_kv_mask: Optional[torch.Tensor] = None,
    ):
        loc, _, _ = unwrap_write_loc(loc_info)
        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

        # Both FIA and paged-attention buffers share the same physical layout.
        # Opt in at pool initialization, keeping environment reads outside graph
        # capture. Otherwise retain the original writers and their memory usage.
        use_scatter_pa = (
            self.use_scatter_pa_kv_cache
            and hasattr(torch_npu, "npu_scatter_pa_kv_cache")
            and self.store_dtype in (torch.float16, torch.bfloat16, torch.int8)
        )
        if self.use_fia or use_scatter_pa:
            k_buffer_layer = self.k_buffer[layer_id - self.start_layer]
            v_buffer_layer = self.v_buffer[layer_id - self.start_layer]
            num_rows = loc.numel()
            expected_k_numel = num_rows * self.head_num * self.head_dim
            expected_v_numel = num_rows * self.head_num * self.v_head_dim
            if (
                cache_k.numel() != expected_k_numel
                or cache_v.numel() != expected_v_numel
            ):
                raise ValueError(
                    "NPU KV scatter row mismatch: "
                    f"loc_rows={num_rows}, cache_k_shape={tuple(cache_k.shape)}, "
                    f"cache_v_shape={tuple(cache_v.shape)}, "
                    f"head_num={self.head_num}, head_dim={self.head_dim}, "
                    f"v_head_dim={self.v_head_dim}."
                )

            if use_scatter_pa:
                torch_npu.npu_scatter_pa_kv_cache(
                    cache_k.contiguous().view(num_rows, self.head_num, self.head_dim),
                    cache_v.contiguous().view(num_rows, self.head_num, self.v_head_dim),
                    k_buffer_layer.view(
                        -1, self.page_size, self.head_num, self.head_dim
                    ),
                    v_buffer_layer.view(
                        -1, self.page_size, self.head_num, self.v_head_dim
                    ),
                    loc.reshape(-1).to(torch.int32).contiguous(),
                    cache_mode="Norm",
                )
            else:
                # aclnnScatterNdUpdate on the deployed CANN rejects the otherwise
                # valid 4-D [slot, 1, head, dim] update during tiling. Flatten only
                # the singleton FIA layout axis and scatter through an equivalent
                # 3-D view; the underlying KV storage and attention layout stay
                # unchanged.
                loc_indices = loc.contiguous().view(-1, 1)
                torch_npu.npu_scatter_nd_update_(
                    k_buffer_layer.view(-1, self.head_num, self.head_dim),
                    loc_indices,
                    cache_k.contiguous().view(num_rows, self.head_num, self.head_dim),
                )
                torch_npu.npu_scatter_nd_update_(
                    v_buffer_layer.view(-1, self.head_num, self.v_head_dim),
                    loc_indices,
                    cache_v.contiguous().view(num_rows, self.head_num, self.v_head_dim),
                )
        else:
            loc = loc.to(torch.int32)
            torch_npu._npu_reshape_and_cache(
                key=cache_k,
                value=cache_v,
                key_cache=self.k_buffer[layer_id - self.start_layer].view(
                    -1, self.page_size, self.head_num, self.head_dim
                ),
                value_cache=self.v_buffer[layer_id - self.start_layer].view(
                    -1, self.page_size, self.head_num, self.v_head_dim
                ),
                slot_indices=loc,
            )

    def set_kv_buffer_prefix_valid(
        self,
        layer: "RadixAttention",
        loc_2d: torch.Tensor,
        commit_lens: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        if not self.use_triton_prefix_kv_cache_store:
            return super().set_kv_buffer_prefix_valid(
                layer,
                loc_2d,
                commit_lens,
                cache_k,
                cache_v,
                k_scale,
                v_scale,
                layer_id_override,
            )

        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id
        if loc_2d.ndim != 2:
            raise ValueError(f"loc_2d must be rank-2, got {tuple(loc_2d.shape)}")

        num_rows = loc_2d.numel()
        if (
            cache_k.numel() != num_rows * self.head_num * self.head_dim
            or cache_v.numel() != num_rows * self.head_num * self.v_head_dim
        ):
            raise ValueError(
                "dense NPU KV rows must match loc_2d size: "
                f"cache_k={tuple(cache_k.shape)}, cache_v={tuple(cache_v.shape)}, "
                f"loc_2d={tuple(loc_2d.shape)}"
            )

        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)
        if self.store_dtype != self.dtype:
            cache_k = cache_k.contiguous().view(self.store_dtype)
            cache_v = cache_v.contiguous().view(self.store_dtype)

        k_buffer_layer = self.k_buffer[layer_id - self.start_layer]
        v_buffer_layer = self.v_buffer[layer_id - self.start_layer]
        if loc_2d.device != k_buffer_layer.device:
            loc_2d = loc_2d.to(device=k_buffer_layer.device, non_blocking=True)
        if commit_lens.device != k_buffer_layer.device:
            commit_lens = commit_lens.to(
                device=k_buffer_layer.device, non_blocking=True
            )
        self._debug_prefix_valid_backend = "npu_triton"
        from sgl_kernel_npu.mem_cache.kv_cache_store import (
            store_kv_cache_prefix_valid_npu_triton,
        )

        store_kv_cache_prefix_valid_npu_triton(
            k_buffer_layer.view(-1, self.head_num, self.head_dim),
            v_buffer_layer.view(-1, self.head_num, self.v_head_dim),
            cache_k.reshape(num_rows, self.head_num, self.head_dim),
            cache_v.reshape(num_rows, self.head_num, self.v_head_dim),
            loc_2d,
            commit_lens,
        )

    def _chunk_copy_npu_to_cpu(self, buf_of_layers, indices):
        chunk_size = self.cpu_offloading_chunk_size
        out = []
        for tensors_per_layer in buf_of_layers:  # [k_buf, v_buf]
            layer_chunks = []
            for i in range(0, len(indices), chunk_size):
                ci = indices[i : i + chunk_size]
                layer_chunks.append(
                    [
                        t[ci].to("cpu", non_blocking=True)
                        for t in tensors_per_layer
                        if t is not None
                    ]
                )
            out.append(layer_chunks)
        return out

    # Parent MHATokenToKVPool.get_cpu_copy / load_cpu_copy use
    # `self.k_buffer[layer_id][chunk_indices]` which indexes the first dim.
    # NPUMHATokenToKVPool stores buffers as
    #   (num_pages, page_size, head_num, head_dim)            # use_fia=False
    #   (num_pages*page_size, 1, head_num, head_dim)          # use_fia=True
    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        torch.npu.synchronize()
        buf_of_layers = []
        for local_layer_id in range(self.layer_num):
            k_layer = self.k_buffer[local_layer_id].view(
                -1, self.head_num, self.head_dim
            )
            v_layer = self.v_buffer[local_layer_id].view(
                -1, self.head_num, self.head_dim
            )
            buf_of_layers.append([k_layer, v_layer])
        kv_cache_cpu = self._chunk_copy_npu_to_cpu(buf_of_layers, indices)
        torch.npu.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(
        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
        torch.npu.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for local_layer_id in range(self.layer_num):
            k_layer = self.k_buffer[local_layer_id].view(
                -1, self.head_num, self.head_dim
            )
            v_layer = self.v_buffer[local_layer_id].view(
                -1, self.head_num, self.head_dim
            )
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_cpu, v_cpu = (
                    kv_cache_cpu[local_layer_id][i // chunk_size][0],
                    kv_cache_cpu[local_layer_id][i // chunk_size][1],
                )
                assert k_cpu.shape[0] == v_cpu.shape[0] == len(chunk_indices)
                k_layer[chunk_indices] = k_cpu.to(k_layer.device, non_blocking=True)
                v_layer[chunk_indices] = v_cpu.to(v_layer.device, non_blocking=True)
        torch.npu.synchronize()


class NPUMHATokenToKOnlyPool(MHATokenToKOnlyPool):
    """NPU paged K-only cache used by MiniMax sparse index-only layers."""

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        self.use_fia = get_bool_env_var("ASCEND_USE_FIA", "False")
        super(MHATokenToKOnlyPool, self).__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            start_layer=start_layer,
            end_layer=end_layer,
        )
        self.model_layer_ids = tuple(
            range(self.start_layer, self.start_layer + self.layer_num)
        )
        self.head_num = head_num
        self.head_dim = head_dim

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.k_buffer = torch.zeros(
                (
                    self.layer_num,
                    self.size // self.page_size + 1,
                    self.page_size,
                    self.head_num,
                    self.head_dim,
                ),
                dtype=self.store_dtype,
                device=self.device,
            )
            # Keep a reference to the contiguous tensor for HiCache
            # D2H/H2D transfers (transfer_kv_dim_exchange expects a
            # tensor, not the per-layer list used in FIA mode below).
            self.k_buffer_tensor = self.k_buffer
            if self.use_fia:
                self.k_buffer = [
                    self.k_buffer[i].view(-1, 1, self.head_num, self.head_dim)
                    for i in range(self.layer_num)
                ]

        self._finalize_allocation_log(size)

    def _get_key_buffer(self, layer_id: int):
        k_buffer = self.k_buffer[layer_id - self.start_layer]
        if self.store_dtype != self.dtype:
            return k_buffer.view(self.dtype)
        return k_buffer

    def set_k_buffer(
        self,
        layer_id: int,
        loc_info,
        cache_k: torch.Tensor,
    ) -> None:
        loc, _, _ = unwrap_write_loc(loc_info)
        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)
        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)

        k_buffer_layer = self.k_buffer[layer_id - self.start_layer].view(
            -1, self.head_num, self.head_dim
        )
        loc = loc.to(device=cache_k.device, dtype=torch.int32).contiguous()
        torch_npu.npu_scatter_nd_update_(
            k_buffer_layer,
            loc.view(-1, 1),
            cache_k.contiguous().view(-1, self.head_num, self.head_dim),
        )

    def get_contiguous_buf_infos(self):
        data_ptrs = [
            self.get_key_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        data_lens = [
            self.get_key_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        if self.use_fia:
            item_lens = [
                self.get_key_buffer(i)[0].nbytes * self.page_size
                for i in range(self.start_layer, self.start_layer + self.layer_num)
            ]
        else:
            item_lens = [
                self.get_key_buffer(i)[0].nbytes
                for i in range(self.start_layer, self.start_layer + self.layer_num)
            ]
        return data_ptrs, data_lens, item_lens

    def get_kv_size_bytes(self):
        return get_tensor_size_bytes(self.k_buffer), 0


class NPUMiniMaxSparseKVPool(MiniMaxSparseKVPool):
    """MiniMax sparse wrapper backed by NPU paged MHA/index pools."""

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            main_pool_cls=NPUMHATokenToKVPool,
            index_kv_pool_cls=NPUMHATokenToKVPool,
            index_k_pool_cls=NPUMHATokenToKOnlyPool,
            **kwargs,
        )

    def get_index_k_state_buf_infos(self):
        pool = self.index_k_pool
        n = pool.layer_num
        data_ptrs = [pool.get_key_buffer(i).data_ptr() for i in range(n)]
        data_lens = [pool.get_key_buffer(i).nbytes for i in range(n)]
        if pool.use_fia:
            item_lens = [
                pool.get_key_buffer(i)[0].nbytes * pool.page_size for i in range(n)
            ]
        else:
            item_lens = [pool.get_key_buffer(i)[0].nbytes for i in range(n)]
        return data_ptrs, data_lens, item_lens


class NPUMLATokenToKVPool(MLATokenToKVPool):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        index_head_dim: Optional[int] = None,
        index_size: Optional[int] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        index_page_size: Optional[int] = None,
        indexer_layer_ids: Optional[Sequence[int]] = None,
        kv_cache_dim: Optional[int] = None,
        is_draft_worker: bool = False,
    ):
        # MLAPO historically owned NZ writes. Keep the allocation unchanged and
        # write into the NZ-addressed view below so ordinary MLA (including
        # Kimi-K3 MTP) can use FIA NZ without MLAPO.
        self.use_fia_nz = get_bool_env_var("SGLANG_USE_FIA_NZ")
        super(MLATokenToKVPool, self).__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            start_layer=start_layer,
            end_layer=end_layer,
        )

        self.model_layer_ids = tuple(
            range(self.start_layer, self.start_layer + self.layer_num)
        )
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.index_head_dim = index_head_dim
        self.enable_sparsity_driven_kv_offload = (
            envs.SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD.get()
        )
        if self.enable_sparsity_driven_kv_offload and self.index_head_dim is None:
            raise ValueError("Sparsity-driven KV offload requires an index KV cache.")
        if index_head_dim is None:
            self.indexer_layer_ids = ()
        elif indexer_layer_ids is None:
            self.indexer_layer_ids = tuple(
                range(self.start_layer, self.start_layer + self.layer_num)
            )
        else:
            self.indexer_layer_ids = tuple(indexer_layer_ids)
        self.num_indexer_layers = len(self.indexer_layer_ids)
        self.indexer_layer_id_to_slot = {
            layer_id: slot for slot, layer_id in enumerate(self.indexer_layer_ids)
        }
        assert len(self.indexer_layer_id_to_slot) == self.num_indexer_layers
        assert all(
            self.start_layer <= i < self.start_layer + self.layer_num
            for i in self.indexer_layer_ids
        )
        requested_kv_cache_dim = kv_cache_dim
        self.dsa_kv_cache_store_fp8 = (
            index_head_dim is not None
            and dtype == torch.float8_e4m3fn
            and requested_kv_cache_dim is not None
        )
        # F: the packed 8-bit SFA switch -- the same 656 B row as the packed-fp8
        # DSA mode above (latent | rope BF16 | per-128-tile FP32 scales), but
        # entered from the model's compute dtype (bf16, which stays self.dtype)
        # with the row stored in the 8-bit dtype. On A2/A3 the latent is INT8
        # (KvQuantSparseFlashAttention accepts INT8-only there). Decoupled from
        # --kv-cache-dtype and from the index-K store dtype below.
        self.sfa_kv_store_8bit = (
            index_head_dim is not None
            and not self.dsa_kv_cache_store_fp8
            and envs.SGLANG_NPU_SFA_KV_8BIT.get()
            and requested_kv_cache_dim is not None
        )
        self.sfa_packed_kv_storage = (
            self.dsa_kv_cache_store_fp8 or self.sfa_kv_store_8bit
        )
        if self.sfa_packed_kv_storage:
            assert index_head_dim == 128 and kv_lora_rank % 128 == 0
            assert requested_kv_cache_dim == (
                kv_lora_rank + kv_lora_rank // 128 * 4 + qk_rope_head_dim * 2
            )
            # One byte per element: the in-tree mode stores FP8 (there
            # store_dtype IS the compute dtype); the flag stores the latent
            # INT8 on A2/A3.
            self.store_dtype = torch.int8 if self.sfa_kv_store_8bit else dtype
        self.kv_cache_dim = (
            requested_kv_cache_dim if self.sfa_packed_kv_storage else kv_lora_rank
        )
        self.kr_cache_dim = (
            0
            if (self.dsa_kv_cache_store_fp8 or self.sfa_packed_kv_storage)
            else qk_rope_head_dim
        )
        if self.sfa_kv_store_8bit and get_parallel().attn_dcp_size <= 1:
            raise ValueError(
                "SGLANG_NPU_SFA_KV_8BIT stores the SFA latent KV as packed "
                "8-bit rows, whose serving reads are the DCP dequant-on-read "
                "gathers (extend prefix gather, decode top-k gather). With "
                "attn_dcp_size 1 there is no DCP gather and the paged read of "
                "the packed rows is not wired here (F2). Run with the flag "
                "under DCP, or without the flag."
            )
        # The index-K keeps its OWN store dtype, decoupled from the packed
        # latent row's store dtype. Under the 8-bit SFA flag the latent row is
        # int8 but the index-K stays the model's bf16: storing it in the row's
        # store dtype would reinterpret bf16 bits as int8 codes and break
        # npu_quant_lightning_indexer.
        if self.sfa_kv_store_8bit:
            self.index_k_store_dtype = self.dtype
            self.index_k_scale_dtype = None
        else:
            self.index_k_store_dtype = self.store_dtype
            self.index_k_scale_dtype = (
                torch.uint8 if self.dsa_kv_cache_store_fp8 else None
            )
        self.index_k_scale_buffer = None
        self.indexer_hadamard_128 = None
        self.index_page_size = page_size if index_page_size is None else index_page_size
        self.index_size = size if index_size is None else index_size
        parallel = get_parallel()
        self.dcp_size = parallel.attn_dcp_size
        self.dcp_rank = parallel.attn_dcp_rank
        global_page_padding = self.dcp_size if self.dcp_size > 1 else 1
        kv_page_padding = global_page_padding if is_draft_worker else 1
        index_page_padding = global_page_padding if index_head_dim is not None else 1
        if kv_page_padding < 1 or index_page_padding < 1:
            raise ValueError("NPU MLA page padding must be positive")
        self.kv_page_padding = kv_page_padding
        self.index_page_padding = index_page_padding
        self.is_draft_worker = is_draft_worker

        # The DCP extend write's owner filter, opened per forward by
        # plan_dcp_extend_write. None keeps the capturable row-0 path.
        self._dcp_extend_write_loc: Optional[torch.Tensor] = None
        self._dcp_extend_write_plan = None

        self.custom_mem_pool = None

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            # The padded slot 0 is used for writing dummy outputs from padded tokens.
            if self.enable_sparsity_driven_kv_offload:
                self.k_buffer = None
                self.v_buffer = None
            else:
                self.k_buffer = torch.zeros(
                    (
                        layer_num,
                        self.size // self.page_size + self.kv_page_padding,
                        self.page_size,
                        1,
                        self.kv_cache_dim,
                    ),
                    dtype=self.store_dtype,
                    device=self.device,
                )
                self.v_buffer = torch.zeros(
                    (
                        layer_num,
                        self.size // self.page_size + self.kv_page_padding,
                        self.page_size,
                        1,
                        self.kr_cache_dim,
                    ),
                    dtype=(
                        torch.bfloat16
                        if self.sfa_packed_kv_storage
                        else self.store_dtype
                    ),
                    device=self.device,
                )
            self.index_k_buffer = None
            if self.index_head_dim is not None:
                # index_size, not self.size: the indexer is replicated.
                self.index_k_buffer = torch.zeros(
                    (
                        self.num_indexer_layers,
                        self.index_size // self.index_page_size
                        + self.index_page_padding,
                        self.index_page_size,
                        1,
                        self.index_head_dim,
                    ),
                    dtype=self.index_k_store_dtype,
                    device=self.device,
                )
                if self.dsa_kv_cache_store_fp8 and self.num_indexer_layers > 0:
                    from sglang.srt.layers.attention.dsa.dsa_npu_indexer import (
                        create_npu_hadamard_128,
                    )

                    # MXFP8 index-k scale cache for quant_lightning_indexer (v2),
                    # quant_mode 3: one E8M0 byte per 32-element block, per-token
                    # tail (k_n, d/64, 2) == (1, 2, 2). Stored as uint8 so the
                    # scatter writer stays on a generic int dtype; consumers get
                    # the E8M0 view via get_index_k_scale_buffer().
                    self.index_k_scale_buffer = torch.zeros(
                        (*self.index_k_buffer.shape[:-1], self.index_head_dim // 64, 2),
                        dtype=torch.uint8,
                        device=self.device,
                    )
                    self.indexer_hadamard_128 = create_npu_hadamard_128(
                        self.index_head_dim, self.device
                    )

        if self.sfa_kv_store_8bit:
            # The startup pool table the flag's benefit is measured against:
            # rows, row bytes, and the capacity delta vs the bf16 row it
            # replaces (1152 B at kv_lora_rank 512 / rope 64). _finalize_
            # allocation_log below reports the pool's own bytes; this line
            # states the per-token contract.
            bf16_row_bytes = (kv_lora_rank + qk_rope_head_dim) * 2
            print_info_once(
                "NPU MLA pool: packed 8-bit SFA KV (SGLANG_NPU_SFA_KV_8BIT) "
                f"is active: row = latent int8 x {kv_lora_rank} + rope bf16 x "
                f"{qk_rope_head_dim} + per-128-tile fp32 scales x "
                f"{kv_lora_rank // 128} = {self.kv_cache_dim} B/token/layer "
                f"(vs {bf16_row_bytes} B bf16, x"
                f"{bf16_row_bytes / self.kv_cache_dim:.2f} capacity at fixed "
                f"GiB); {self.layer_num} KV layers, {size} rows, index-K "
                f"{self.index_k_store_dtype}"
            )
        self._finalize_allocation_log(size)

    def _mla_buffer_info(self):
        from sglang.srt.mem_cache.device_pool_info import MLABufferInfo

        return MLABufferInfo(
            buffers=self._mla_native_buffers(),
            native_buffers=self._mla_native_buffers,
            data_ptrs=None,
            layer_indices=tuple(range(self.layer_num)),
            device_layer_num=self.layer_num,
            page_size=self.page_size,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            kv_cache_dim=self.kv_cache_dim + self.kr_cache_dim,
            dtype=self.store_dtype,
            # self.device is the bare backend name; tensors carry the rank index.
            device=next(
                buffer.device
                for buffer in (self.k_buffer, self.index_k_buffer)
                if buffer is not None
            ),
            packed_fp8=self.dsa_kv_cache_store_fp8,
        )

    def _mla_native_buffers(self):
        from sglang.srt.mem_cache.device_pool_info import MLASplitBuffers

        return MLASplitBuffers(
            k=self.k_buffer,
            v=self.v_buffer,
            index_k=self.index_k_buffer,
            index_scales=self.index_k_scale_buffer,
            index_layer_indices=tuple(
                i - self.start_layer for i in self.indexer_layer_ids
            ),
            k_width=self.kv_cache_dim,
            v_width=self.kr_cache_dim,
            index_width=self.index_head_dim or 0,
        )

    def _copy_indices_for_buffer(self, indices, uses_global_slots):
        if uses_global_slots or self.dcp_size <= 1:
            return indices
        # Buffer ROWS, not pages, so this must be the pool's own interleave.
        local_indices = localize_dcp_indices(
            indices, self.dcp_size, self.dcp_rank, dcp_interleave_size()
        )
        return local_indices[local_indices >= 0]

    def get_kv_size_bytes(self):
        kv_size_bytes = 0

        if getattr(self, "k_buffer", None) is not None:
            for k_cache in self.k_buffer:
                kv_size_bytes += get_tensor_size_bytes(k_cache)
        if getattr(self, "v_buffer", None) is not None:
            for v_cache in self.v_buffer:
                kv_size_bytes += get_tensor_size_bytes(v_cache)
        if getattr(self, "index_k_buffer", None) is not None:
            for index_k_cache in self.index_k_buffer:
                kv_size_bytes += get_tensor_size_bytes(index_k_cache)
        if self.index_k_scale_buffer is not None:
            kv_size_bytes += get_tensor_size_bytes(self.index_k_scale_buffer)
        return kv_size_bytes

    def _raise_if_native_kv_cache_disabled(self):
        if (
            getattr(self, "k_buffer", None) is None
            or getattr(self, "v_buffer", None) is None
        ):
            raise RuntimeError(
                "Native NPU MLA device KV cache is disabled; "
                "k_buffer/v_buffer are not available. Use the sparse KV manager "
                "path, or re-enable native device KV cache for this code path."
            )

    def get_kv_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        self._raise_if_native_kv_cache_disabled()
        return (
            self.k_buffer[layer_id - self.start_layer],
            self.v_buffer[layer_id - self.start_layer],
        )

    def get_mla_kv_buffer(
        self,
        layer: "RadixAttention",
        loc: torch.Tensor,
        dst_dtype: Optional[torch.dtype] = None,
        layer_id: Optional[int] = None,
    ):
        """Gather ``(nope, rope)`` at physical rows ``loc``.

        Overridden because the inherited Triton read assumes one fused buffer
        and this pool keeps the halves apart. ``loc`` is already rank-local and
        physical (the read door's contract), so no owner filter here.
        ``layer_id`` reads another layer's KV, for the gather prefetch.
        """
        read_layer_id = layer.layer_id if layer_id is None else layer_id
        if self.sfa_kv_store_8bit:
            # F: the LSE sparse attention is bf16-only on A2/A3, so a read over
            # the int8 packed pool selects the whole 656 B row and decodes it
            # here (F1's temporary read; F2 fuses the dequant into the quant
            # operator for the paged read). The in-tree packed-fp8 mode above
            # is unchanged (the sparse operator reads it whole).
            return self._gather_packed_sfa_rows(read_layer_id, loc, dst_dtype)
        idx = loc.to(torch.int64)
        if self.dsa_kv_cache_store_fp8:
            # One packed record per row -- nope, rope and the dequant scales
            # together -- so there is no second half to return. The sparse
            # operator reads it whole at KV_D == kv_cache_dim, and splitting it
            # here would only force a re-pack before the gather.
            assert dst_dtype is None or dst_dtype == self.store_dtype, (
                "the packed FP8 KV record cannot be cast on read; asked for "
                f"{dst_dtype}"
            )
            # uint8, not the store dtype: aclnnIndexSelect has no FP8 in its
            # dtype list. The record is opaque bytes to every consumer except
            # the sparse operator, which the backend hands an FP8 view.
            packed = (
                self.k_buffer[read_layer_id - self.start_layer]
                .view(torch.uint8)
                .view(-1, self.kv_cache_dim)
            )
            return packed.index_select(0, idx).unsqueeze(1), None
        k = self.get_key_buffer(read_layer_id).view(-1, self.kv_lora_rank)
        v = self.get_value_buffer(read_layer_id).view(-1, self.qk_rope_head_dim)
        cache_k_nope = k.index_select(0, idx).unsqueeze(1)
        cache_k_rope = v.index_select(0, idx).unsqueeze(1)
        if dst_dtype is not None and dst_dtype != cache_k_nope.dtype:
            cache_k_nope = cache_k_nope.to(dst_dtype)
            cache_k_rope = cache_k_rope.to(dst_dtype)
        return cache_k_nope, cache_k_rope

    def _gather_packed_sfa_rows(
        self,
        read_layer_id: int,
        loc: torch.Tensor,
        dst_dtype: Optional[torch.dtype],
    ):
        """Dequantise selected packed 8-bit SFA rows into ``(nope, rope)``.

        The row layout is the pack's, from _pack_dsa_fp8_kv_cache: latent 8-bit
        (kv_lora_rank) | rope BF16 (qk_rope_head_dim) | per-128-tile FP32 scales
        (kv_lora_rank // 128), one byte per element in the store dtype. Dequant
        is the pack's inverse: the scale from npu_dynamic_quant is a per-tile
        multiplier, so ``latent.to(t) * scale.to(t)`` tile-wise, and the rope
        bytes are a straight bf16 reinterpretation. Reading k_buffer directly
        (not through get_key_buffer) because the storage view IS the packed row.
        """
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(
                read_layer_id - self.start_layer
            )
        storage = self.k_buffer[read_layer_id - self.start_layer].view(
            -1, self.kv_cache_dim
        )
        rows = storage.index_select(0, loc.to(torch.int64))
        out_dtype = dst_dtype if dst_dtype is not None else self.dtype
        rope_bytes = self.qk_rope_head_dim * 2
        tiles = self.kv_lora_rank // 128
        # Dequantise in FP32 (the scale is exact there; a bf16 scale would add
        # a second, unearned rounding) and round to the compute dtype once.
        scale = (
            rows[:, self.kv_lora_rank + rope_bytes :]
            .view(torch.uint8)
            .view(torch.float32)
            .reshape(-1, tiles, 1)
        )
        nope = (
            rows[:, : self.kv_lora_rank]
            .to(torch.float32)
            .reshape(-1, tiles, 128)
            * scale
        ).reshape(-1, self.kv_lora_rank)
        rope = (
            rows[:, self.kv_lora_rank : self.kv_lora_rank + rope_bytes]
            .view(torch.uint8)
            .view(torch.bfloat16)
        )
        # rope is a strided column slice of the packed row (dim-0 stride
        # kv_cache_dim), so it is NOT contiguous, and .to(out_dtype) is a no-op
        # copy once out_dtype is already bf16 -- npu_sparse_flash_attention_lse
        # rejects a non-contiguous key_rope. Materialise both halves to match the
        # non-packed path's contiguous index_select contract (nope is already a
        # fresh copy from the fp32 dequant; .contiguous() is a no-op there).
        return (
            nope.to(out_dtype).contiguous().unsqueeze(1),
            rope.to(out_dtype).contiguous().unsqueeze(1),
        )

    def gather_packed_sfa_rows_raw(
        self, read_layer_id: int, loc: torch.Tensor
    ) -> torch.Tensor:
        """The raw 656 B packed rows at physical rows ``loc``, one byte each.

        The DCP extend gather ships these as-is (F2): the all-gather runs over
        the opaque one-byte row, so this is a plain index_select into the
        storage view. Returned as uint8 so the caller can all-gather over a
        single, collective-safe element type and hand the quant operator back
        its storage dtype with one view.
        """
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(
                read_layer_id - self.start_layer
            )
        storage = self.k_buffer[read_layer_id - self.start_layer].view(
            -1, self.kv_cache_dim
        )
        return storage.index_select(0, loc.to(torch.int64)).view(torch.uint8)

    def view_packed_sfa_rows(self, rows: torch.Tensor):
        """The (latent, rope, scale) views out of raw 656 B packed rows.

        The inverse of _pack_dsa_fp8_kv_cache's record layout, as VIEWS (no
        copy): latent int8 (kv_lora_rank) | rope BF16 (qk_rope_head_dim) |
        per-128-tile FP32 scales (kv_lora_rank // 128). F2 hands these to the
        quant operator, which dequantises in-kernel; the views are strided
        (dim-0 stride kv_cache_dim), so the caller materialises them (the
        operator's TND contract).
        """
        rows = rows.reshape(-1, self.kv_cache_dim)
        rope_bytes = self.qk_rope_head_dim * 2
        latent = rows[:, : self.kv_lora_rank].view(torch.int8)
        rope = rows[
            :, self.kv_lora_rank : self.kv_lora_rank + rope_bytes
        ].view(torch.bfloat16)
        scale = (
            rows[:, self.kv_lora_rank + rope_bytes :]
            .view(torch.uint8)
            .view(torch.float32)
        )
        return latent, rope, scale

    def dequant_packed_sfa_rows(
        self, rows: torch.Tensor, dst_dtype: Optional[torch.dtype] = None
    ):
        """The F1 (nope, rope) halves of raw 656 B packed rows (F2 fallback).

        The same dequant as _gather_packed_sfa_rows (the per-128-tile scale
        applied in FP32, the rope a straight bf16 reinterpretation) but on rows
        the caller already holds, so the F2 read can fall back to the F1 bf16
        halves from the F2 buffer when the quant op declines on this box.
        """
        rows = rows.reshape(-1, self.kv_cache_dim)
        out_dtype = dst_dtype if dst_dtype is not None else self.dtype
        rope_bytes = self.qk_rope_head_dim * 2
        tiles = self.kv_lora_rank // 128
        scale = (
            rows[:, self.kv_lora_rank + rope_bytes :]
            .view(torch.uint8)
            .view(torch.float32)
            .reshape(-1, tiles, 1)
        )
        # The buffer is uint8 (the all-gather's own element type), so the latent
        # half must be REINTERPRETED as int8 before it is turned into a float --
        # .to(f32) on uint8 would give the byte values 0..255, not the signed
        # int8 the quant operator stored (npu_dynamic_quant, dst_type int8).
        nope = (
            rows[:, : self.kv_lora_rank]
            .view(torch.int8)
            .to(torch.float32)
            .reshape(-1, tiles, 128)
            * scale
        ).reshape(-1, self.kv_lora_rank)
        rope = (
            rows[:, self.kv_lora_rank : self.kv_lora_rank + rope_bytes]
            .view(torch.uint8)
            .view(torch.bfloat16)
        )
        return (
            nope.to(out_dtype).contiguous().unsqueeze(1),
            rope.to(out_dtype).contiguous().unsqueeze(1),
        )

    def get_state_buf_infos(self):
        if self.index_head_dim is None:
            return [], [], []
        # The buffer is compacted to indexer layers; range(layer_num) overruns.
        buffers = list(self.index_k_buffer)
        if self.index_k_scale_buffer is not None:
            buffers += list(self.index_k_scale_buffer)
        data_ptrs = [buf.data_ptr() for buf in buffers]
        data_lens = [buf.nbytes for buf in buffers]
        item_lens = [buf[0].nbytes for buf in buffers]
        return data_ptrs, data_lens, item_lens

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        self._raise_if_native_kv_cache_disabled()

        if self.sfa_packed_kv_storage:
            # F: the packed row is opaque storage, not a value of the compute
            # dtype: viewing 656 int8 elements as bf16 would halve the row and
            # corrupt the layout the quant operator and the DCP dequant address.
            # The readers are the two packed-aware paths (the quant sparse
            # attention operator, and get_mla_kv_buffer's dequant).
            return self.k_buffer[layer_id - self.start_layer]
        if self.store_dtype != self.dtype:
            return self.k_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.k_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        self._raise_if_native_kv_cache_disabled()

        if self.store_dtype != self.dtype:
            return self.v_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.v_buffer[layer_id - self.start_layer]

    def get_index_k_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        if getattr(self, "index_k_buffer", None) is None:
            raise RuntimeError("NPU MLA index KV cache is not allocated.")

        # Unlike the latent (get_key_buffer views through store_dtype), the
        # index-K is stored element-wise in its OWN dtype (index_k_store_dtype)
        # and its only NPU reader -- the LightningIndexer -- consumes it in
        # exactly that dtype, handing the scale to the quant op separately. It
        # is never packed/reinterpreted, so viewing it through the latent store
        # dtype is wrong: under the 8-bit SFA flag store_dtype is int8 and
        # viewing the int8 index-K as bf16 halves the row and mis-types the key
        # (npu_quant_lightning_indexer rejects key=DT_BFLOAT16).
        return self.index_k_buffer[self._get_indexer_slot(layer_id)]

    def _get_indexer_slot(self, layer_id: int) -> int:
        return self.indexer_layer_id_to_slot[layer_id]

    def get_index_k_scale_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        # E8M0 view of the raw byte storage: (block_num, block_size, k_n, d/64, 2),
        # the k_descale layout quant_lightning_indexer (v2) expects with
        # quant_mode 3 and layout_k PA_BBND.
        return self.index_k_scale_buffer[self._get_indexer_slot(layer_id)].view(
            torch.float8_e8m0fnu
        )

    def set_index_k_scale_buffer(self, layer_id: int, loc, scale):
        # Scatter whole per-token scale rows (k_n, d/64, 2) so the MX block
        # scales keep the layout quant_lightning_indexer (v2) reads back.
        assert scale.element_size() == 1, (
            f"index_k scale must be a 1-byte (E8M0) MX scale, got dtype {scale.dtype}"
        )
        buf = self.index_k_scale_buffer[self._get_indexer_slot(layer_id)]
        tail = buf.shape[2:]
        torch_npu.npu_scatter_nd_update_(
            buf.view(-1, *tail),
            loc.view(-1, 1),
            scale.reshape(-1, *tail).view(torch.uint8),
        )

    def _get_disagg_buffer_entries(self):
        """Return (buffer, uses_global_slots) entries in PD transfer order."""
        self._raise_if_native_kv_cache_disabled()
        global_kv = self.is_draft_worker
        entries = [(buffer, global_kv) for buffer in self.k_buffer]
        if not getattr(self, "sfa_packed_kv_storage", False):
            entries += [(buffer, global_kv) for buffer in self.v_buffer]
        if self.index_head_dim is not None:
            entries += [(buffer, True) for buffer in self.index_k_buffer]
            if self.index_k_scale_buffer is not None:
                entries += [(buffer, True) for buffer in self.index_k_scale_buffer]
        return entries

    # for disagg
    def get_contiguous_buf_infos(self):
        entries = self._get_disagg_buffer_entries()
        return (
            [buffer.data_ptr() for buffer, _ in entries],
            [buffer.nbytes for buffer, _ in entries],
            [
                buffer[0].nbytes * (self.dcp_size if uses_global_slots else 1)
                for buffer, uses_global_slots in entries
            ],
        )

    def get_dcp_remote_decode_layout(self) -> list[bool]:
        """Whether each PD entry uses allocator-global slots on decode."""
        return [
            uses_global_slots
            for _, uses_global_slots in self._get_disagg_buffer_entries()
        ]

    def get_kv_layer_ids(self):
        return (
            list(range(self.start_layer, self.start_layer + self.layer_num)) * 2
            + self.get_state_layer_ids()
        )

    def get_state_layer_ids(self):
        return list(self.indexer_layer_ids) * (
            2 if self.index_k_scale_buffer is not None else 1
        )

    def _pack_dsa_fp8_kv_cache(self, cache_k, cache_v):
        latent = cache_k.reshape(-1, self.kv_lora_rank)
        # dst_type is the STORAGE dtype, not the compute dtype: under the 8-bit
        # SFA flag the model computes in bf16 and the latent must still quantise
        # to the store dtype (int8 on A2/A3; in the in-tree mode the two
        # coincide).
        quantized, scale = torch_npu.npu_dynamic_quant(
            latent.reshape(-1, 128), dst_type=self.store_dtype
        )
        rows = latent.shape[0]
        # Opaque record: latent 8-bit | rope BF16 bytes | per-tile FP32 scales.
        packed = torch.cat(
            (
                quantized.reshape(rows, self.kv_lora_rank).view(torch.uint8),
                cache_v.to(torch.bfloat16)
                .reshape(rows, self.qk_rope_head_dim)
                .contiguous()
                .view(torch.uint8),
                scale.to(torch.float32)
                .reshape(rows, self.kv_lora_rank // 128)
                .contiguous()
                .view(torch.uint8),
            ),
            dim=-1,
        )
        return packed.view(self.store_dtype)

    def pack_sfa_kv_cache(self, cache_k, cache_v):
        """Public pack for the DCP extend gather's own-chunk tail (F2).

        The gather appends this forward's own KV to the packed buffer, holding
        it as bf16 (nope, rope); packing it here with the same quantise+record
        the pool uses in set_kv_buffer makes the buffer's tail the pool's own
        rows (npu_dynamic_quant is deterministic, so the two quantisations
        agree) instead of a bf16 row the operator's packed reader cannot address.
        """
        return self._pack_dsa_fp8_kv_cache(cache_k, cache_v)

    def _resolve_dcp_write(
        self,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        """Redirect a DCP-widened loc so this rank's tokens land on its rows.

        CUDA filters inside the Triton store; here it cannot, because
        ``npu_scatter_nd_update_`` is a fused vendor operator with no body.

        Decode rewrites the destination rather than dropping rows: a boolean
        index has a data-dependent shape, so torch_npu resolves it with
        ``aclnnNonzeroV2`` and syncs the stream, which capture refuses.
        Non-owned rows are aimed at physical row 0, which the allocator never
        hands out (``free_pages`` starts at 1) and nothing reads.

        Extend drops them instead -- never captured, and one write location is
        shared by every layer, so the filter costs one sync per forward.
        """
        dcp_size = get_parallel().attn_dcp_size
        # Against the widened space, as the CUDA pool does.
        maybe_detect_oob(
            loc,
            0,
            (self.size + self.page_size) * dcp_size,
            "set_kv_buffer (NPU MLA, widened loc)",
        )
        if dcp_size == 1:
            return loc, cache_k, cache_v

        if loc is self._dcp_extend_write_loc:
            owned_idx, dest = self._dcp_extend_write_index(loc, dcp_size)
            return (
                dest,
                cache_k.index_select(0, owned_idx),
                cache_v.index_select(0, owned_idx),
            )

        # Static shape, no NonZero, no stream sync: capturable.
        owned, dest = dcp_owner_and_row(
            loc, dcp_size, get_parallel().attn_dcp_rank, dcp_interleave_size()
        )
        return torch.where(owned, dest, loc.new_zeros(())), cache_k, cache_v

    def plan_dcp_extend_write(self, loc: torch.Tensor) -> None:
        """Let this extend forward's KV write drop the rows this rank does not own.

        ``_resolve_dcp_write`` recognises the forward by tensor identity: decode's
        loc is a different object, and the filter is discarded on every call.
        """
        self._dcp_extend_write_loc = loc
        self._dcp_extend_write_plan = None

    def _dcp_extend_write_index(self, loc: torch.Tensor, dcp_size: int):
        """This forward's owner filter, computed once and reused by every layer."""
        if self._dcp_extend_write_plan is None:
            self._dcp_extend_write_plan = plan_dcp_owner_write(
                loc,
                dcp_size,
                get_parallel().attn_dcp_rank,
                dcp_interleave_size(),
            )
        return self._dcp_extend_write_plan

    def set_kv_buffer(
        self,
        layer: "RadixAttention",
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        loc, _, _ = unwrap_write_loc(loc_info)
        self._raise_if_native_kv_cache_disabled()
        layer_id = layer.layer_id
        # BUG 6: the D1v3 S1 gather produces [num_tokens_pad]-wide KV
        # (tp * ceil(num_tokens/tp)), but out_cache_loc has exactly
        # num_tokens rows. Trim to the loc width before the scatter;
        # a no-op for divisible batches and for base DCP.
        n_loc = loc.numel()
        if cache_k.shape[0] > n_loc:
            cache_k = cache_k[:n_loc]
            if cache_v is not None:
                cache_v = cache_v[:n_loc]
        if self.sfa_packed_kv_storage:
            if cache_v is None:
                cache_k, cache_v = cache_k.split(
                    [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
                )
            # The packed FP8 record is latent KV, so it is sharded under DCP
            # exactly like the bf16 form: filter to this rank's rows and
            # collapse the widened loc BEFORE packing. Without this the scatter
            # writes at the virtual loc, which runs up to dcp_size times past
            # the end of k_buffer -- and nothing catches it, because this
            # branch returns before the bounds check in _resolve_dcp_write and
            # the sparse path's FP8-plus-DCP assert only guards the read.
            loc, cache_k, cache_v = self._resolve_dcp_write(loc, cache_k, cache_v)
            if loc.numel() == 0:
                return
            packed = self._pack_dsa_fp8_kv_cache(cache_k, cache_v)
            torch_npu.npu_scatter_nd_update_(
                self.k_buffer[layer_id - self.start_layer].view(
                    -1, 1, self.kv_cache_dim
                ),
                loc.view(-1, 1),
                packed.view(-1, 1, self.kv_cache_dim),
            )
            return

        if cache_v is None:
            cache_k, cache_v = cache_k.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )

        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

        if self.use_fia_nz:
            self._set_fia_nz_kv_buffer(layer_id, loc, cache_k, cache_v)
            return

        loc, cache_k, cache_v = self._resolve_dcp_write(loc, cache_k, cache_v)
        if loc.numel() == 0:
            # A rank can own none of a short chunk's rows.
            return

        torch_npu.npu_scatter_nd_update_(
            self.k_buffer[layer_id - self.start_layer].view(-1, 1, self.kv_lora_rank),
            loc.view(-1, 1),
            cache_k.view(-1, 1, self.kv_lora_rank),
        )
        torch_npu.npu_scatter_nd_update_(
            self.v_buffer[layer_id - self.start_layer].view(
                -1, 1, self.qk_rope_head_dim
            ),
            loc.view(-1, 1),
            cache_v.view(-1, 1, self.qk_rope_head_dim),
        )

    def _set_fia_nz_kv_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> None:
        """Store MLA latent and RoPE KV tensors in FIA's NZ tile order."""

        def scatter(cache: torch.Tensor, values: torch.Tensor, head_dim: int):
            num_tiles = head_dim // 16
            indices = _mla_fia_nz_scatter_indices(loc, head_dim, self.page_size)
            # Destination rows are ordered [page, tile, slot]. Source rows use
            # the matching [token, tile] order after this reshape.
            dst = cache.view(-1, 1, num_tiles, self.page_size, 16).view(-1, 16)
            src = values.contiguous().view(-1, num_tiles, 16).view(-1, 16)
            torch_npu.npu_scatter_nd_update_(dst, indices, src)

        offset = layer_id - self.start_layer
        scatter(self.k_buffer[offset], cache_k, self.kv_lora_rank)
        scatter(self.v_buffer[offset], cache_v, self.qk_rope_head_dim)

    def set_index_k_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
    ):
        # A write to an elided layer means the pool's mask disagrees with the
        # model's, so fail loudly. The indexer writes at the raw loc.
        maybe_detect_oob(
            loc,
            0,
            # Widened pages start at 1, so the top is one widened page past index_size.
            self.index_size + self.index_page_size * self.index_page_padding,
            "set_index_k_buffer (NPU MLA, raw virtual loc)",
        )
        assert layer_id in self.indexer_layer_id_to_slot, (
            f"layer {layer_id} owns no Indexer but wrote index-K; the pool's "
            "indexer_layer_ids disagrees with the model's indexer layout"
        )

        if index_k.dtype != self.dtype:
            index_k = index_k.to(self.dtype)

        # Bit-cast to the INDEX-K store dtype only where it differs. This used
        # to view onto self.store_dtype, which is the latent row's dtype: under
        # the 8-bit SFA flag that is int8 while the index-K is still the model's
        # bf16, and the view would reinterpret the bits rather than store them.
        if index_k.dtype != self.index_k_store_dtype:
            index_k = index_k.view(self.index_k_store_dtype)

        torch_npu.npu_scatter_nd_update_(
            self.index_k_buffer[self._get_indexer_slot(layer_id)].view(
                -1, 1, self.index_head_dim
            ),
            loc.view(-1, 1),
            index_k.view(-1, 1, self.index_head_dim),
        )

    def _chunk_copy_npu_to_cpu(
        self, buf_of_layers, indices, uses_global_slots_per_layer
    ):
        chunk_size = self.cpu_offloading_chunk_size
        out = []
        for tensors_per_layer, uses_global_slots in zip(
            buf_of_layers, uses_global_slots_per_layer, strict=True
        ):  # [k_buf, v_buf, ik_buf/None]
            layer_chunks = []
            for i in range(0, len(indices), chunk_size):
                ci = indices[i : i + chunk_size]
                layer_chunks.append(
                    [
                        t[self._copy_indices_for_buffer(ci, uses_global)].to(
                            "cpu", non_blocking=True
                        )
                        for t, uses_global in zip(
                            tensors_per_layer, uses_global_slots, strict=True
                        )
                        if t is not None
                    ]
                )
            out.append(layer_chunks)
        return out

    def _get_cpu_offload_layer_buffers(self, local_layer_id):
        # flatten(page, slot) also works for the zero-width packed V placeholder.
        buffers = [
            self.k_buffer[local_layer_id].flatten(0, 1),
            self.v_buffer[local_layer_id].flatten(0, 1),
        ]
        slot = self.indexer_layer_id_to_slot.get(local_layer_id + self.start_layer)
        if slot is not None:
            buffers.append(self.index_k_buffer[slot].flatten(0, 1))
            if self.index_k_scale_buffer is not None:
                buffers.append(self.index_k_scale_buffer[slot].flatten(0, 1))
        if self.sfa_packed_kv_storage:
            # Retraction copies opaque records; byte views also avoid 8-bit
            # (int8/fp8) advanced-indexing restrictions, without
            # decoding/requantizing.
            buffers = [
                buf.view(torch.uint8)
                if buf.dtype in (torch.float8_e4m3fn, torch.int8)
                else buf
                for buf in buffers
            ]
        return buffers

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        torch.npu.synchronize()
        buf_of_layers = [
            self._get_cpu_offload_layer_buffers(i) for i in range(self.layer_num)
        ]
        uses_global_slots_per_layer = []
        for buffers in buf_of_layers:
            # MLA K/V is rank-local under DCP. The replicated
            # indexer buffers retain allocator-global slot identities.
            uses_global_slots_per_layer.append(
                [self.is_draft_worker, self.is_draft_worker]
                + [True] * (len(buffers) - 2)
            )

        kv_cache_cpu = self._chunk_copy_npu_to_cpu(
            buf_of_layers, indices, uses_global_slots_per_layer
        )
        torch.npu.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(
        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
        torch.npu.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for local_layer_id in range(self.layer_num):
            buffers = self._get_cpu_offload_layer_buffers(local_layer_id)
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                chunk = kv_cache_cpu[local_layer_id][i // chunk_size]
                cpu_index = 0
                for buffer, uses_global_slots in zip(
                    buffers,
                    [self.is_draft_worker, self.is_draft_worker]
                    + [True] * (len(buffers) - 2),
                    strict=True,
                ):
                    if buffer is None:
                        continue
                    cpu = chunk[cpu_index]
                    cpu_index += 1
                    target_indices = self._copy_indices_for_buffer(
                        chunk_indices, uses_global_slots
                    )
                    assert cpu.shape[0] == len(target_indices)
                    buffer[target_indices] = cpu.to(buffer.device, non_blocking=True)
        torch.npu.synchronize()
