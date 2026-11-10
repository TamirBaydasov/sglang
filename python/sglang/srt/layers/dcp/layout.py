# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Pure index math for decode context parallel (DCP): per-rank lengths,
the owner-rule local-index filter, and the NPU extend-gather plan."""

import logging
from functools import lru_cache
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.unified_cache.component_type import BASE_COMPONENT_TYPE
from sglang.srt.runtime_context import get_parallel, get_schedule
from sglang.srt.utils import print_info_once

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _dcp_page_interleave_flag() -> bool:
    # Cached, not read at import: a module-level read happens before any test
    # can set the variable. Same reason as _dsa_token_shard_flag.
    return envs.SGLANG_NPU_DCP_PAGE_INTERLEAVE.get()


def dcp_interleave_size() -> int:
    """Positions a rank holds in a row before the next rank's.

    NPU only. CUDA's Triton store (``kernels/ops/kvcache/mla_buffer.py:42``)
    and the MHA read paths hardcode 1, so shared callers pass 1 explicitly.
    """
    return get_schedule().page_size if _dcp_page_interleave_flag() else 1


def dcp_owner_count(length: int, dcp_size: int, dcp_rank: int, interleave: int) -> int:
    """Scalar ``get_dcp_lens``: how many of ``[0, length)`` this rank owns."""
    cycle = dcp_size * interleave
    full, rem = divmod(length, cycle)
    return full * interleave + min(max(rem - dcp_rank * interleave, 0), interleave)


def dcp_local_row(loc: torch.Tensor, dcp_size: int, interleave: int) -> torch.Tensor:
    """The row ``loc`` occupies in its owner's pool."""
    if interleave == 1:
        return loc // dcp_size
    return (loc // (interleave * dcp_size)) * interleave + loc % interleave


def dcp_owner_and_row(
    loc: torch.Tensor, dcp_size: int, dcp_rank: int, interleave: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(owned mask, destination row)``, for write paths.

    Not ``localize_dcp_indices``: that returns a ``-1`` sentinel the write
    paths would only have to mask away again, and at ``interleave == 1`` its
    divide-by-1 and mod-1 still launch kernels on a per-layer path.
    """
    block = loc if interleave == 1 else loc // interleave
    return (block % dcp_size) == dcp_rank, dcp_local_row(loc, dcp_size, interleave)


def get_dcp_lens(
    lens: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    start: torch.Tensor | None = None,
    interleave_size: int = 1,
) -> torch.Tensor:
    """Return the KV length owned by one DCP rank.

    Consecutive runs of ``interleave_size`` positions rotate across ranks.  The
    default value preserves the token-interleaved layout used by the generic
    CUDA/ROCm DCP paths; NPU DSA passes its physical KV page size.
    """
    if dcp_size == 1:
        return lens
    if interleave_size < 1:
        raise ValueError(f"interleave_size must be positive, got {interleave_size}")

    cycle_size = dcp_size * interleave_size

    def _count_before(end: torch.Tensor) -> torch.Tensor:
        full_cycles = end // cycle_size
        remainder = end % cycle_size
        rank_remainder = torch.clamp(
            remainder - dcp_rank * interleave_size,
            min=0,
            max=interleave_size,
        )
        return full_cycles * interleave_size + rank_remainder

    if start is None:
        return _count_before(lens)
    return _count_before(start + lens) - _count_before(start)


def localize_dcp_indices(
    indices: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    interleave_size: int = 1,
) -> torch.Tensor:
    """Map global DCP indices to one rank, using ``-1`` for non-local rows."""
    if dcp_size == 1:
        return indices
    if interleave_size < 1:
        raise ValueError(f"interleave_size must be positive, got {interleave_size}")

    interleave_block = torch.div(indices, interleave_size, rounding_mode="floor")
    is_local = (indices >= 0) & (interleave_block % dcp_size == dcp_rank)
    local_block = torch.div(interleave_block, dcp_size, rounding_mode="floor")
    local_indices = local_block * interleave_size + indices % interleave_size
    return torch.where(is_local, local_indices, torch.full_like(indices, -1))


def maybe_dcp_kernel_indices(
    indices: torch.Tensor, dcp_size: int, dcp_rank: int
) -> torch.Tensor:
    """Widened logical slots -> this rank's physical rows.

    Owner rule: slot % dcp_size == dcp_rank, row = slot // dcp_size. The run
    starts page-aligned, so a strided view selects the owned slots without a mask.
    """
    if dcp_size == 1:
        return indices
    return indices[dcp_rank::dcp_size] // dcp_size


def filter_dcp_local_kv_indices(kv_indices: torch.Tensor):
    """Keep this rank's share of a read-index tensor, still WIDENED.

    Selection only; the caller collapses via translate_dcp_read_ids.
    """
    parallel = get_parallel()
    if parallel.dcp_enabled:
        kv_indices = kv_indices[kv_indices % parallel.dcp_size == parallel.dcp_rank]
    return kv_indices


def filter_dcp_local_chunk_kv_indices(
    kv_indices: torch.Tensor,
    chunk_starts_cpu: torch.Tensor,
    chunk_seq_lens_cpu: torch.Tensor,
) -> torch.Tensor:
    parallel = get_parallel()
    if not parallel.dcp_enabled:
        return kv_indices

    dcp_size = parallel.dcp_size
    parts = []
    offset = 0
    for start, length in zip(chunk_starts_cpu.tolist(), chunk_seq_lens_cpu.tolist()):
        first = (parallel.dcp_rank - start) % dcp_size
        parts.append(kv_indices[offset + first : offset + length : dcp_size])
        offset += length
    return torch.cat(parts)


def remap_dcp_sparse_indices(
    topk_indices: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    interleave_size: int = 1,
) -> torch.Tensor:
    """Map global sparse token indices to one rank's compact DCP KV layout.

    Keep indices owned by this rank, convert them to local KV positions, and
    stably move them before ``-1`` padding while preserving their score order.
    """
    if dcp_size == 1:
        return topk_indices
    if interleave_size < 1:
        raise ValueError(f"interleave_size must be positive, got {interleave_size}")

    # Float32 is faster than integer division/remainder on Ascend for this hot
    # path.  Keep the math equivalent to localize_dcp_indices above. Exact only
    # below 2**24 positions, 16.4x the 1M target, and it degrades to a wrong KV
    # row rather than failing -- test_the_float32_bound_on_the_position_itself.
    topk_indices_fp32 = topk_indices.to(torch.float32)
    interleave_blocks = torch.floor(topk_indices_fp32 / interleave_size)
    local_owner_mask = (topk_indices_fp32 >= 0) & (
        torch.remainder(interleave_blocks, dcp_size) == dcp_rank
    )
    local_offsets = torch.remainder(topk_indices_fp32, interleave_size)
    local_indices = (
        torch.floor(interleave_blocks / dcp_size) * interleave_size + local_offsets
    )
    remapped_indices = torch.where(
        local_owner_mask,
        local_indices,
        torch.full_like(topk_indices_fp32, -1.0),
    ).to(topk_indices.dtype)

    # Move valid entries before padding without changing their top-k order.
    topk_count = topk_indices.shape[-1]
    original_order = torch.arange(
        topk_count, dtype=torch.float32, device=topk_indices.device
    ).expand_as(topk_indices_fp32)
    pack_keys = original_order + (~local_owner_mask).to(torch.float32) * topk_count
    pack_order = torch.argsort(pack_keys, dim=-1).to(torch.int64)
    return torch.gather(remapped_indices, dim=-1, index=pack_order)


def get_dcp_chain_spec_lens(
    total_kv_lens: torch.Tensor,
    tokens_per_req: int,
    dcp_size: int,
    dcp_rank: int,
    interleave_size: int = 1,
) -> torch.Tensor:
    """Return request-major local KV frontiers for a speculative chain."""
    if tokens_per_req < 1:
        raise ValueError(f"tokens_per_req must be >= 1, got {tokens_per_req}")
    total_kv_lens = total_kv_lens.int()
    steps = torch.arange(
        1, tokens_per_req + 1, dtype=total_kv_lens.dtype, device=total_kv_lens.device
    )
    global_query_lens = total_kv_lens.unsqueeze(1) - tokens_per_req + steps
    global_query_lens = torch.where(
        total_kv_lens.unsqueeze(1) >= tokens_per_req,
        global_query_lens,
        torch.zeros_like(global_query_lens),
    )
    return get_dcp_lens(
        global_query_lens.reshape(-1),
        dcp_size,
        dcp_rank,
        interleave_size=interleave_size,
    ).int()


def update_local_kv_lens_for_dcp(kv_len_arr):
    """In-place per-rank KV length: the start=0 case of get_dcp_lens.

    floor((len - rank - 1) / N) + 1  ==  len // N + (rank < len % N)  for len >= 0
    (bit-identical; see test/registered/cp/test_dcp_layout_unit.py). Kept as an
    in-place mutation because callers (plan_dcp_decode_metadata, the FlashInfer-MLA
    cuda-graph replay path) rely on it.
    """
    parallel = get_parallel()
    if not parallel.dcp_enabled:
        return
    kv_len_arr.copy_(get_dcp_lens(kv_len_arr, parallel.dcp_size, parallel.dcp_rank))


def plan_dcp_owner_write(
    loc: torch.Tensor, dcp_size: int, dcp_rank: int, interleave_size: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rows of ``loc`` this rank owns, and the physical rows they land on.

    Across the group the ``owned_idx`` partition every row of ``loc`` exactly
    once, so a filtered write covers the same rows as an unfiltered one.

    ``torch.nonzero`` has a data-dependent output shape, so it syncs the stream
    and cannot run under capture -- extend only, where one write location is
    shared by every layer. Decode aims non-owned rows at a padding row instead.
    """
    owned, dest = dcp_owner_and_row(loc, dcp_size, dcp_rank, interleave_size)
    owned_idx = torch.nonzero(owned).squeeze(1)
    return owned_idx, dest[owned_idx]


# Reusable device buffers for the extend gather, keyed by purpose, dtype, device
# and row shape. See dcp_extend_gather_buffer.
_dcp_extend_gather_buffers: Dict[Tuple, torch.Tensor] = {}


def dcp_extend_gather_buffer(name: str, ref: torch.Tensor, rows: int) -> torch.Tensor:
    """A reusable ``rows``-row buffer shaped and typed like ``ref``. Grow-only.

    Allocated per layer, the peak moved with whatever the allocator held when a
    layer asked for another ~1.1 GiB. Two callers sharing ``name`` share the
    buffer and must not hold slices across each other's calls.
    """
    key = (name, ref.dtype, ref.device, tuple(ref.shape[1:]))
    buf = _dcp_extend_gather_buffers.get(key)
    if buf is None or buf.shape[0] < rows:
        # Drop every reference first, or the peak is briefly both buffers.
        _dcp_extend_gather_buffers.pop(key, None)
        buf = None
        buf = torch.empty((rows, *ref.shape[1:]), dtype=ref.dtype, device=ref.device)
        _dcp_extend_gather_buffers[key] = buf
        logger.info(
            "DCP extend gather buffer %r reserved: %d rows, %.3f GiB",
            name,
            rows,
            buf.numel() * buf.element_size() / (1 << 30),
        )
    return buf[:rows]


class DcpSharedPrefix(NamedTuple):
    """What a batch's prefixes share, by radix node.

    ``walked`` is each request's tree-owned prefix rebuilt from its match and
    ``protected`` is the scheduler's own count of it, so the two disagreeing
    means the walk is wrong. ``union_rows`` counts a shared node once: the rows
    a deduplicated gather would send for the tree-owned part.
    """

    walked: List[int]
    protected: List[int]
    union_rows: int


def _dcp_node_rows(node: object) -> Optional[int]:
    """Tokens a radix node holds on device, or None when it holds none.

    Two node shapes reach here. ``UnifiedTreeNode`` -- what the default
    ``UnifiedRadixCache`` builds, so what a served run uses -- keeps the Full KV
    under ``component_data``; ``RadixCache``'s ``TreeNode`` has a plain
    ``value``. Reading only the latter counts every node as empty.
    """
    data = getattr(node, "component_data", None)
    value = (
        data[BASE_COMPONENT_TYPE].value
        if data is not None
        else getattr(node, "value", None)
    )
    return None if value is None else len(value)


def dcp_shared_prefix(
    last_nodes: Sequence[object], protected_lens: Sequence[int]
) -> DcpSharedPrefix:
    """Measure prefix sharing across a batch from the radix match alone.

    A request's tree-owned prefix is its root-to-``last_node`` path, so sharing
    is the paths' common ancestors -- no device read and no index comparison.
    Duck-typed on ``.parent``/``.value`` to stay import-free and CPU-testable.

    Only the tree-owned part is shareable. Under ``page_size > 1`` a chunked
    request also carries a partial page that ``checkpoint`` keeps in
    ``prefix_indices`` but not in the tree; it belongs to that request alone, so
    it is outside both ``walked`` and ``union_rows`` and a dedup must still send
    it per request.
    """
    seen: Dict[int, int] = {}
    walked: List[int] = []
    for last_node in last_nodes:
        node, rows = last_node, 0
        while node is not None:
            n = _dcp_node_rows(node)
            if n:
                rows += n
                seen.setdefault(id(node), n)
            node = getattr(node, "parent", None)
        walked.append(rows)
    return DcpSharedPrefix(
        walked=walked, protected=list(protected_lens), union_rows=sum(seen.values())
    )


class DcpExtendGatherPiece(NamedTuple):
    """One collective of the extend gather. The all-gather lays the sends out
    rank-major, this chunk's own KV follows, and ``index`` maps output rows
    ``[out_start, out_end)`` onto that scratch."""

    send_start: int
    send_end: int
    extend_start: int
    extend_end: int
    out_start: int
    out_end: int
    index: torch.Tensor
    scratch_rows: int


class DcpExtendGatherPlan(NamedTuple):
    """How one rank gathers a batch's prefix KV at extend. Sends are padded
    to ``padded_lens`` so every rank's is ``send_rows`` long; the output is each
    request's prefix then its extend, in order, tiled by ``pieces``."""

    local_lens: List[int]
    padded_lens: List[int]
    send_rows: int
    scratch_rows: int
    pieces: List[DcpExtendGatherPiece]


def plan_dcp_extend_gather(
    prefix_lens: Sequence[int],
    extend_lens: Sequence[int],
    dcp_size: int,
    dcp_rank: int,
    max_piece_gather_rows: int,
    interleave_size: int = 1,
) -> DcpExtendGatherPlan:
    """Plan the extend-time gather of the prefix KV into position order.

    Position p lives on rank ``(p // B) % dcp_size`` at local row
    ``(p // (B * dcp_size)) * B + p % B``, ``B = interleave_size``. Local shards
    are concatenated per request, as ``dcp_local_prefix_kv_indices`` lists them.

    Pieces gather at most ``max_piece_gather_rows`` rows each, and cut on whole
    ``B``-blocks: that keeps local rows ``[row, row + take)`` covering global
    positions ``[row * dcp_size, (row + take) * dcp_size)`` on every rank. The
    pieces are identical on every rank; only ``local_lens`` is not.
    """
    prefix_lens = [int(p) for p in prefix_lens]
    extend_lens = [int(e) for e in extend_lens]
    b = interleave_size
    local_lens = [dcp_owner_count(p, dcp_size, dcp_rank, b) for p in prefix_lens]
    # Rank 0 holds the most; every rank pads to it.
    padded_lens = [dcp_owner_count(p, dcp_size, 0, b) for p in prefix_lens]
    # Rounded down to a whole block, so a cut never splits one rank's run.
    piece_rows = max(b, (max_piece_gather_rows // dcp_size) // b * b)

    pieces = []
    parts = []
    send_start = extend_start = out_start = 0
    send_end = extend_end = out_end = 0
    send_offset = 0
    for prefix_len, extend_len, padded_len in zip(
        prefix_lens, extend_lens, padded_lens
    ):
        row = 0
        while row < padded_len:
            if send_end - send_start == piece_rows:
                pieces.append(
                    _dcp_extend_gather_piece(
                        parts,
                        dcp_size,
                        send_start,
                        send_end,
                        extend_start,
                        extend_end,
                        out_start,
                        out_end,
                        b,
                    )
                )
                parts = []
                send_start, extend_start, out_start = send_end, extend_end, out_end
            take = min(padded_len - row, piece_rows - (send_end - send_start))
            positions = torch.arange(
                row * dcp_size,
                min((row + take) * dcp_size, prefix_len),
                dtype=torch.int64,
            )
            parts.append(("prefix", positions, send_offset))
            row += take
            send_end += take
            out_end += positions.numel()
        if extend_len:
            parts.append(("extend", extend_end, extend_len))
            extend_end += extend_len
            out_end += extend_len
        send_offset += padded_len
    if parts:
        pieces.append(
            _dcp_extend_gather_piece(
                parts,
                dcp_size,
                send_start,
                send_end,
                extend_start,
                extend_end,
                out_start,
                out_end,
                b,
            )
        )
    return DcpExtendGatherPlan(
        local_lens=local_lens,
        padded_lens=padded_lens,
        send_rows=sum(padded_lens),
        scratch_rows=max((p.scratch_rows for p in pieces), default=0),
        pieces=pieces,
    )


def _dcp_extend_gather_piece(
    parts,
    dcp_size: int,
    send_start: int,
    send_end: int,
    extend_start: int,
    extend_end: int,
    out_start: int,
    out_end: int,
    interleave_size: int = 1,
) -> DcpExtendGatherPiece:
    """Close one piece: index its output rows into its rank-major scratch."""
    send_len = send_end - send_start
    gathered_rows = send_len * dcp_size
    b = interleave_size
    index = []
    for kind, first, second in parts:
        if kind == "prefix":
            positions, request_send_offset = first, second
            owner = (positions // b) % dcp_size
            local_row = (positions // (b * dcp_size)) * b + positions % b
            index.append(
                owner * send_len + (request_send_offset - send_start) + local_row
            )
        else:
            start = gathered_rows + first - extend_start
            index.append(torch.arange(start, start + second, dtype=torch.int64))
    return DcpExtendGatherPiece(
        send_start=send_start,
        send_end=send_end,
        extend_start=extend_start,
        extend_end=extend_end,
        out_start=out_start,
        out_end=out_end,
        index=torch.cat(index),
        scratch_rows=gathered_rows + extend_end - extend_start,
    )


def dcp_crop_free_extend(forward_batch, index_topk: Optional[int]) -> bool:
    """May this extend forward drop the operator's causal crop (sparse_mode 0)?

    Only if every prefix reaches ``index_topk``: below it the top-k takes every
    key, so the crop is what keeps the result causal. Cached per forward, and
    the FIRST caller's ``index_topk`` wins -- the cache does not key on it.
    """
    cached = getattr(forward_batch, "npu_dcp_crop_free", None)
    if cached is not None:
        return cached
    prefix_lens = getattr(forward_batch, "extend_prefix_lens_cpu", None)
    ok = bool(
        index_topk is not None
        and prefix_lens
        and all(dcp_packed_causal_crop_is_dead(p, index_topk) for p in prefix_lens)
    )
    if not ok and index_topk is not None and prefix_lens:
        print_info_once(
            f"DCP extend keeps the operator's causal crop: a request here has a "
            f"prefix under index_topk={index_topk} (shortest {min(prefix_lens)})"
        )
    forward_batch.npu_dcp_crop_free = ok
    return ok


class DcpPackedReadPlan(NamedTuple):
    """Where each KV position of a ONE-REQUEST extend lives in the packed buffer.

    ``plan_dcp_extend_gather`` permutes the all-gather's rank-major output back
    into position order so the sparse operator can read it. That permutation is
    a context-sized ``index_select`` -- 784 launches x 0.275 ms = 216 ms a
    forward -- plus the position-ordered output buffer it writes into, ~1.04 GiB
    a rank at a 972k context. Neither is needed: the operator reads the KV
    through ``sparse_indices``, so it does not care what order the rows are in,
    only that the indices point at the right ones. Remapping a ``[tokens, k]``
    index tensor is arithmetic on a few million integers; permuting the KV is a
    gigabyte of traffic. Upstream reached the same conclusion for CUDA in
    ``#39215``'s ``create_packed_dsa_dcp_kv_indices``.

    The packed buffer is exactly what the collective produces, in one piece::

        [ rank 0's send rows | rank 1's | ... | rank C-1's ][ this chunk's KV ]
          <--- send_rows ---> each                            <-- extend_len -->

    Under the owner rule ``pos % C == rank`` a request's position ``p`` sits in
    rank ``p % C``'s send at local row ``p // C``, so::

        p <  prefix_len   ->  (p % C) * send_rows + p // C
        p >= prefix_len   ->  gathered_rows + (p - prefix_len)

    ``send_rows`` is the padded local length, so ranks below ``prefix_len % C``
    hold one row more than the rest and the spare rows are never addressed.

    **ONE REQUEST ONLY, and the reason is the same one DSA-CP has.** With two
    requests the all-gather still lays out rank-major over the whole send, so a
    request's rows land in C separate blocks and the operator's cumulative
    ``actual_seq_lengths_kv`` -- which needs each request contiguous -- cannot
    describe them. Multi-request extends keep the permuting path. That covers
    87% of the tokens in the AISBench run as measured on 2026-09-18.
    """

    prefix_len: int
    extend_len: int
    dcp_size: int
    send_rows: int
    gathered_rows: int
    rows: int
    # D2v2: the sub-all-gather count the buffer was (or will be) filled with.
    # 1 is the monolithic rank-major layout above; above 1 the prefix part is
    # chunk-major (chunk, then rank, then row) and the remap gains a chunk-major
    # branch. See dcp_gather_chunks_for.
    chunks: int = 1


def plan_dcp_packed_read(
    prefix_len: int, extend_len: int, dcp_size: int
) -> DcpPackedReadPlan:
    """Sizes of the packed rank-major buffer for a one-request extend."""
    send_rows = -(-int(prefix_len) // dcp_size)
    gathered_rows = send_rows * dcp_size
    return DcpPackedReadPlan(
        prefix_len=int(prefix_len),
        extend_len=int(extend_len),
        dcp_size=dcp_size,
        send_rows=send_rows,
        gathered_rows=gathered_rows,
        rows=gathered_rows + int(extend_len),
        chunks=dcp_gather_chunks_for(send_rows),
    )


def dcp_gather_chunks_for(send_rows: int) -> int:
    """The effective D2v2 sub-all-gather count for a send this long.

    The env value clamped to what the send can actually be cut into: a chunk of
    zero rows would be a collective with nothing in it, and one row per chunk
    (send shorter than the count) keeps the arithmetic honest rather than being
    a special case. 1 is "off": the monolithic collective and the rank-major
    layout, bit for bit.
    """
    chunks = _dcp_gather_chunks_env
    if chunks <= 1:
        return 1
    return min(chunks, max(1, int(send_rows)))


def _chunk_bounds(plan: DcpPackedReadPlan):
    """(q, rem, n): the D2v2 chunk sizes.

    The first ``n - 1`` chunks hold ``q = send_rows // n`` local rows each; the
    last holds ``q + send_rows % n``. ``n`` is 1 only for the monolithic plan,
    where this is never called.
    """
    n = plan.chunks
    q, rem = divmod(plan.send_rows, n)
    return q, rem, n


def packed_row_of(pos: int, plan: DcpPackedReadPlan) -> int:
    """Reference implementation of the remap, one position at a time.

    Exists to be the thing ``remap_topk_to_packed`` is checked against on CPU:
    the tensor version has to fuse the branch into ``where`` and do floor
    division on a signed tensor, and those are the two places this kind of
    arithmetic goes wrong.
    """
    if pos < 0:
        return pos
    if pos < plan.prefix_len:
        c = plan.dcp_size
        if plan.chunks == 1:
            return (pos % c) * plan.send_rows + pos // c
        # Chunk-major: local row i of rank r = chunk k, in-chunk row j, and
        # chunk k's block starts at c * (k * q) rows in (chunks before it each
        # held q local rows x c ranks).
        q, rem, n = _chunk_bounds(plan)
        r = pos % c
        i = pos // c
        if i < (n - 1) * q:
            k = i // q
            j = i - k * q
            sk = q
        else:
            k = n - 1
            j = i - (n - 1) * q
            sk = q + rem
        return c * (k * q) + r * sk + j
    return plan.gathered_rows + (pos - plan.prefix_len)


def remap_topk_to_packed(
    topk_indices: torch.Tensor, plan: DcpPackedReadPlan
) -> torch.Tensor:
    """Top-k positions within a request -> rows of the packed buffer.

    Shape and dtype are preserved, and so are the negatives: ``_pad_topk_indices``
    fills unused slots with -1 and the operator reads that tail as "no more
    entries", so mapping them into a real row would silently add keys. Every
    branch here is elementwise, which is what keeps this ~ a millisecond against
    the 216 ms it replaces.
    """
    c = plan.dcp_size
    # floor division, not trunc: torch's // on a signed tensor rounds toward
    # zero for negatives, and the -1 sentinels must come through untouched.
    local_row = torch.div(topk_indices, c, rounding_mode="floor")
    if plan.chunks == 1:
        prefix_row = (topk_indices % c) * plan.send_rows + local_row
    else:
        # D2v2 chunk-major: local row i of rank r sits in chunk k = i // q
        # (the last chunk absorbs the remainder) at in-chunk row j, and chunk
        # k's block starts c * (k * q) rows in. Same arithmetic as
        # packed_row_of, fused into where/clamp because this runs on a
        # [tokens, k] tensor 21 times a forward.
        q, rem, n = _chunk_bounds(plan)
        k = torch.div(local_row, q, rounding_mode="floor").clamp(max=n - 1)
        j = torch.where(
            k == n - 1, local_row - (n - 1) * q, local_row - k * q
        )
        sk = q + (k == n - 1).to(local_row.dtype) * rem
        prefix_row = c * (k * q) + (topk_indices % c) * sk + j
    extend_row = plan.gathered_rows + (topk_indices - plan.prefix_len)
    packed = torch.where(topk_indices < plan.prefix_len, prefix_row, extend_row)
    return torch.where(topk_indices < 0, topk_indices, packed)


_enable_dcp_packed_read = envs.SGLANG_NPU_ENABLE_DCP_PACKED_READ.get()
_dcp_gather_chunks_env = envs.SGLANG_NPU_DCP_GATHER_CHUNKS.get()


class _PackedMissing:
    """Distinguishes "not resolved yet" from "resolved, and it is None"."""


_PACKED_MISSING = _PackedMissing()


def dcp_packed_read_enabled() -> bool:
    """Whether the packed rank-major read is on for this process."""
    return _enable_dcp_packed_read


def dcp_packed_causal_crop_is_dead(prefix_len: int, index_topk: int) -> bool:
    """Whether the operator's causal crop can be dropped for this chunk.

    THE PACKED READ NEEDS ``sparse_mode = 0``: the crop aligns query i to key
    ``K - Q + i``, which means nothing once the keys are in rank-major order.
    The top-k is causal only where there is something to select FROM. At or
    below ``index_topk`` the top-k is not a selection, it is everything, and the
    crop is the only thing making the result causal. Above it, every query in
    the chunk has at least ``index_topk`` keys strictly before it, the scores of
    later keys are -inf, and a top-k of that can only name causal keys. The bound
    is on the FIRST query in the chunk, which is the one with the least context.
    """
    return prefix_len + 1 >= index_topk


def _dsa_kv_store_is_fp8() -> bool:
    """Whether the active DSA KV pool stores FP8 (``dsa_kv_cache_store_fp8`` /
    MXFP8). The packed single-record read does not support it: under the FP8
    store ``get_mla_kv_buffer`` returns the packed uint8 view with ``rope=None``,
    which the packed path's ``_pad_dcp_packed_send`` cannot consume. Defensive:
    the forward context is not always up (a plan probe can run before a forward),
    so a missing context means "not FP8" and the other refusals still apply.
    """
    try:
        from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

        return bool(
            getattr(get_token_to_kv_pool(), "dsa_kv_cache_store_fp8", False)
        )
    except Exception:
        return False


def dcp_packed_read_plan(
    forward_batch, index_topk: Optional[int]
) -> Optional[DcpPackedReadPlan]:
    """This forward's packed-buffer plan, or None to keep the permuting path.

    Resolved once per forward and cached on the batch. It lives here, beside the
    arithmetic, rather than in the NPU model module, because BOTH the model (to
    fill the buffer and remap the top-k) and the attention backend (to set the
    operator's KV length and drop its causal crop) have to agree on it, and a
    backend importing a model module inverts the layering -- the same reason
    ``dsa_cp.py`` sits where it does.

    Every refusal is logged once. A silent refusal here would look exactly like
    a feature that is on and doing nothing.
    """
    if not _enable_dcp_packed_read:
        return None
    cached = getattr(forward_batch, "npu_dcp_packed_plan", _PACKED_MISSING)
    if cached is not _PACKED_MISSING:
        return cached

    plan = None
    extend_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
    prefix_lens = getattr(forward_batch, "extend_prefix_lens_cpu", None)
    if _dsa_kv_store_is_fp8():
        # FP8 DSA KV store (dsa_kv_cache_store_fp8 / MXFP8): the packed
        # single-record read is not supported for it (get_mla_kv_buffer returns
        # rope=None). The permuting path handles FP8 (it packs uint8 and the
        # backend's quant branch dequants), so decline to that fallback -- the
        # v1 packed-read refusal, re-expressed as a plan admission.
        print_info_once(
            "DCP packed read is off: the DSA KV is stored in FP8 "
            "(dsa_kv_cache_store_fp8); the packed single-record read is not "
            "supported for it, so the permuting path is used"
        )
    elif index_topk is None:
        # No top-k to measure, so the causal bound below cannot be checked.
        # Refuse, and CACHE the refusal: a forward has to take one path the
        # whole way down. Resolving later would leave earlier layers' top-k
        # unremapped while later layers read a packed buffer.
        print_info_once(
            "DCP packed read is off: this forward reached the extend gather "
            "with no top-k, so the causal bound cannot be checked"
        )
    elif not extend_lens or prefix_lens is None:
        print_info_once("DCP packed read is off: no CPU length metadata on this extend")
    elif len(extend_lens) != 1:
        # ONE REQUEST, for the layout reason in DcpPackedReadPlan: the
        # all-gather is rank-major over the WHOLE send, so with two requests a
        # request's rows land in dcp_size separate blocks and the operator's
        # cumulative actual_seq_lengths_kv cannot describe them. The permuting
        # path handles those.
        print_info_once(
            f"DCP packed read is off for multi-request extends "
            f"({len(extend_lens)} requests here); the all-gather is rank-major "
            "over the whole send, so no request is contiguous in it"
        )
    elif not dcp_crop_free_extend(forward_batch, index_topk):
        # dcp_crop_free_extend has already logged which request fell short.
        print_info_once(
            "DCP packed read is off: this chunk needs the operator's causal "
            "crop, and the packed layout cannot use one"
        )
    else:
        plan = plan_dcp_packed_read(
            prefix_lens[0], extend_lens[0], get_parallel().dcp_size
        )
        print_info_once(
            "DCP packed read is ON: the sparse operator reads the all-gather's "
            "own rank-major output and the top-k is remapped instead"
        )
    forward_batch.npu_dcp_packed_plan = plan
    return plan


def dcp_packed_kv_lens(forward_batch, plan: DcpPackedReadPlan, device) -> torch.Tensor:
    """The operator's ``actual_seq_lengths_kv`` for a packed read, built once.

    One request, so one entry, and it is the whole buffer: with the causal crop
    off there is nothing to shorten, and the padding rows the ranks sent to keep
    their sends equal are never named by any remapped index.

    Cached on the batch rather than built where it is used, because that use is
    once per layer and ``torch.tensor(list, device=npu)`` is a blocking
    host-to-device copy -- 78 of them a forward, each draining the queue.
    """
    cached = getattr(forward_batch, "npu_dcp_packed_kv_lens", None)
    if cached is None:
        cached = torch.tensor([plan.rows], dtype=torch.int32, device=device)
        forward_batch.npu_dcp_packed_kv_lens = cached
    return cached
