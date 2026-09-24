# Copyright 2023-2026 SGLang Team
#
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
"""CPU unit test for the NPU DCP extend-gather PREFETCH.

``SGLANG_NPU_ENABLE_DCP_EXTEND_GATHER_PREFETCH`` splits the per-layer gather in
``_dcp_gather_extend_kv_npu`` in two: the all-gathers, which read only prefix
rows that earlier forwards wrote, and the rest, which needs this layer's own KV.
The first half is then issued for layer L+1 while layer L computes, on a side
stream, into one of two scratch buffers picked by layer parity.

The NPU path itself cannot run here, so this pins the two properties that do not
need a device, over the real plan from ``plan_dcp_extend_gather``:

1. **Splitting the loop changes nothing.** Gathering every piece first and only
   then appending the own rows and running the index_select must produce the
   same position-ordered output as doing both per piece, for one key (an FP8
   cache packs the record into one) and for two (bf16 keeps nope and rope
   apart). A prefetch that carried only the first key would leave the second
   holding whatever the previous layer wrote -- fluent, wrong, and silent.

2. **The handshake is what makes it safe.** Two events per direction: the side
   stream waits the main stream's release before refilling a slot, and the main
   stream waits the side stream's ready before reading one. The model here
   asserts both, and the negative-control tests remove each in turn to prove
   the assertions can fire. Note which one does what: with a single buffer and
   the release wait the result is still *correct*, just serialised again, so
   the release wait buys correctness and the second buffer buys speed.

Values are tagged with the layer that wrote them, so a stale read -- the failure
mode of a dropped key or a missing wait -- is a mismatch rather than a pass.
"""

import unittest

from sglang.srt.layers.dcp.layout import plan_dcp_extend_gather
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

DCP_SIZES = [1, 2, 4, 8, 16]
# The prefetch requires a single-piece plan, which is what
# SGLANG_NPU_DCP_EXTEND_GATHER_PIECE_ROWS <= 0 produces.
ONE_PIECE = 1 << 62
LAYERS = 8

CASES = [
    ([0], [1]),
    ([1], [4]),
    ([7], [17]),
    ([16], [1]),
    ([129], [4]),
    ([0, 0], [1, 1]),
    ([5, 9], [3, 5]),
    ([16384, 32768], [1, 1]),
    ([129, 7, 64], [9, 2, 6]),
    ([1, 0, 33], [1, 1, 1]),
]


def reference(prefix_lens, extend_lens, layer, key):
    """Position-ordered output: each request's prefix, then its extend rows."""
    out = []
    for r, (p, e) in enumerate(zip(prefix_lens, extend_lens)):
        out += [(layer, key, r, "p", i) for i in range(p)]
        out += [(layer, key, r, "e", i) for i in range(e)]
    return out


def local_shard(prefix_lens, dcp_size, rank, layer, key):
    """What ``get_mla_kv_buffer`` hands one rank: its prefix rows, per request."""
    rows = []
    for r, p in enumerate(prefix_lens):
        rows += [(layer, key, r, "p", i) for i in range(p) if i % dcp_size == rank]
    return rows


def padded_send(shard_rows, plan):
    send = [None] * plan.send_rows
    src = dst = 0
    for local_len, padded_len in zip(plan.local_lens, plan.padded_lens):
        send[dst : dst + local_len] = shard_rows[src : src + local_len]
        src += local_len
        dst += padded_len
    return send


class StaleRead(AssertionError):
    """A stream read or wrote a scratch slot the other was still using."""


def run_forward(
    prefix_lens,
    extend_lens,
    dcp_size,
    prefetch,
    n_keys=1,
    slots_n=2,
    release_wait=True,
    ready_wait=True,
    prefetch_keys=None,
):
    """Simulate one forward over ``LAYERS`` layers; return [[per key] per layer].

    ``prefetch_keys`` limits which keys the side stream gathers, so a prefetch
    that drops one can be reproduced deliberately.
    """
    plans = [
        plan_dcp_extend_gather(prefix_lens, extend_lens, dcp_size, rank, ONE_PIECE)
        for rank in range(dcp_size)
    ]
    plan = plans[0]
    assert len(plan.pieces) <= 1, "the prefetch only runs on a single-piece plan"
    total = plan.pieces[-1].out_end if plan.pieces else 0
    if prefetch_keys is None:
        prefetch_keys = range(n_keys)

    # [key][slot], mirroring _dcp_extend_gather_scratches: latent_scratch and
    # rope_scratch, each with a "_b" twin for the odd layers.
    slots = [
        [[None] * plan.scratch_rows for _ in range(slots_n)] for _ in range(n_keys)
    ]
    unreleased = [[False] * slots_n for _ in range(n_keys)]
    filled = [[False] * slots_n for _ in range(n_keys)]
    inflight = [[False] * slots_n for _ in range(n_keys)]

    def gather_phase(slot, layer, on_side, keys):
        for key in keys:
            if on_side and unreleased[key][slot] and not release_wait:
                raise StaleRead("side stream refilled a slot the main stream had read")
            unreleased[key][slot] = False
            sends = [
                padded_send(
                    local_shard(prefix_lens, dcp_size, rk, layer, key), plans[rk]
                )
                for rk in range(dcp_size)
            ]
            for piece in plan.pieces:
                send_len = piece.send_end - piece.send_start
                for rk in range(dcp_size):
                    for j in range(send_len):
                        slots[key][slot][rk * send_len + j] = sends[rk][
                            piece.send_start + j
                        ]
            if on_side:
                inflight[key][slot] = True
            else:
                filled[key][slot] = True

    def wait_ready(slot, keys):
        for key in keys:
            if inflight[key][slot]:
                inflight[key][slot] = False
                filled[key][slot] = True

    def consume_phase(slot, layer):
        outs = []
        for key in range(n_keys):
            if not filled[key][slot]:
                raise StaleRead(
                    "main stream read a slot the side stream had not filled"
                )
            extend_rows = [
                (layer, key, r, "e", i)
                for r, e in enumerate(extend_lens)
                for i in range(e)
            ]
            out = [None] * total
            for piece in plan.pieces:
                gathered = (piece.send_end - piece.send_start) * dcp_size
                scratch = slots[key][slot]
                for k, j in enumerate(range(piece.extend_start, piece.extend_end)):
                    scratch[gathered + k] = extend_rows[j]
                out[piece.out_start : piece.out_end] = [scratch[i] for i in piece.index]
            filled[key][slot] = False
            unreleased[key][slot] = True
            outs.append(out)
        return outs

    results, pending = [], set()
    for layer in range(LAYERS):
        slot = (layer % slots_n) if prefetch else 0
        if prefetch and layer in pending:
            if ready_wait:
                wait_ready(slot, range(n_keys))
        else:
            gather_phase(slot, layer, on_side=False, keys=range(n_keys))
        results.append(consume_phase(slot, layer))
        if prefetch and layer + 1 < LAYERS:
            gather_phase(
                (layer + 1) % slots_n, layer + 1, on_side=True, keys=prefetch_keys
            )
            pending.add(layer + 1)
    return results


class TestDcpExtendGatherPrefetch(CustomTestCase):
    def _check(self, prefetch, n_keys, **kw):
        for dcp_size in DCP_SIZES:
            for prefix_lens, extend_lens in CASES:
                got = run_forward(
                    prefix_lens, extend_lens, dcp_size, prefetch, n_keys=n_keys, **kw
                )
                for layer, per_key in enumerate(got):
                    for key, out in enumerate(per_key):
                        self.assertEqual(
                            out,
                            reference(prefix_lens, extend_lens, layer, key),
                            f"dcp={dcp_size} prefix={prefix_lens} keys={n_keys} "
                            f"prefetch={prefetch} layer={layer} key={key}",
                        )

    def test_inline_gathers_rebuild_every_request_in_position_order(self):
        self._check(prefetch=False, n_keys=1)
        self._check(prefetch=False, n_keys=2)

    def test_prefetching_a_layer_ahead_changes_nothing(self):
        self._check(prefetch=True, n_keys=1)

    def test_a_bf16_cache_carries_both_keys_through_the_prefetch(self):
        self._check(prefetch=True, n_keys=2)

    def test_dropping_a_key_from_the_prefetch_is_caught(self):
        # The FP8-only gate this replaced would have left the rope half holding
        # the previous layer's rows.
        with self.assertRaises(StaleRead):
            run_forward([2048, 1024], [3, 5], 8, True, n_keys=2, prefetch_keys=[0])

    def test_the_main_stream_must_wait_the_ready_event(self):
        with self.assertRaises(StaleRead):
            run_forward([2048, 1024], [3, 5], 8, True, ready_wait=False)

    def test_the_side_stream_must_wait_the_release_event(self):
        with self.assertRaises(StaleRead):
            run_forward([2048, 1024], [3, 5], 8, True, release_wait=False)

    def test_one_slot_is_correct_but_serialised(self):
        # The release wait is what makes the prefetch correct; the second
        # buffer is what lets it overlap. With one slot and the wait kept, the
        # side stream waits for the read it is about to clobber -- safe, and
        # pointless.
        got = run_forward([2048, 1024], [3, 5], 8, True, n_keys=2, slots_n=1)
        for layer, per_key in enumerate(got):
            for key, out in enumerate(per_key):
                self.assertEqual(out, reference([2048, 1024], [3, 5], layer, key))


if __name__ == "__main__":
    unittest.main()
