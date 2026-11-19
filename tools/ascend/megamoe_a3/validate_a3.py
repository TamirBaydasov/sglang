#!/usr/bin/env python3
"""Run the pinned DeepEP accuracy oracle with A3 boundary/routing cases."""

import argparse
from functools import wraps
import importlib
import json
from pathlib import Path
import sys


def worker(local_rank, local_ranks, options):
    import torch

    sys.path.insert(0, str(options.sgl_kernel_npu / "tests/python/deepep"))
    oracle = importlib.import_module("test_fused_deep_moe_a3_w4a8_acc")
    if options.routing == "skew":

        def skew_topk(num_tokens, num_experts, num_topk, device):
            # Send every rank's rows to the same experts to stress receive bounds.
            ids = torch.arange(num_topk, device=device, dtype=torch.int32)
            ids = ids.expand(num_tokens, -1).contiguous()
            weights = torch.full(
                (num_tokens, num_topk), 1 / num_topk, device=device, dtype=torch.float32
            )
            return ids, weights

        oracle.make_topk_inputs = skew_topk

    original = oracle.run_fused_reference

    @wraps(original)
    def measured(*args, **kwargs):
        torch.npu.synchronize()
        torch.npu.reset_peak_memory_stats()
        before = torch.npu.memory_allocated()
        output = original(*args, **kwargs)
        torch.npu.synchronize()
        print(
            json.dumps(
                {
                    "local_rank": local_rank,
                    "quant_mode": options.quant_mode,
                    "routing": options.routing,
                    "rows": options.num_tokens,
                    "allocated_before": before,
                    "peak_allocated": torch.npu.max_memory_allocated(),
                    "peak_reserved": torch.npu.max_memory_reserved(),
                }
            ),
            flush=True,
        )
        return output

    oracle.run_fused_reference = measured
    if options.unequal_rows and local_rank:
        options.num_tokens = 1
    oracle.test_loop(local_rank, local_ranks, options)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sgl-kernel-npu", type=Path, required=True)
    parser.add_argument("--quant-mode", choices=("w4a8", "w8a8"), required=True)
    parser.add_argument("--num-tokens", type=int, default=1024)
    parser.add_argument("--num-processes", type=int, default=16)
    parser.add_argument("--num-servers", type=int, default=1)
    parser.add_argument("--server-index", type=int, default=0)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--routing", choices=("random", "skew"), default="random")
    parser.add_argument("--unequal-rows", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.num_tokens <= 4096:
        parser.error("--num-tokens must be in [1, 4096]")
    args.sgl_kernel_npu = args.sgl_kernel_npu.resolve(strict=True)
    args.num_ranks_per_server = args.num_processes
    args.hidden = 6144
    args.moe_intermediate_size = 2048
    args.num_experts = 256
    args.num_topk = 8
    args.activation = "swiglu"
    args.beta = "0"
    args.linear_beta = "0"
    args.seed = 2026
    args.hccl_buffsize = 768
    args.enable_performance = False
    args.performance_only = False
    args.performance_iters = 1
    sys.path.insert(0, str(args.sgl_kernel_npu / "tests/python/deepep"))
    oracle = importlib.import_module("test_fused_deep_moe_a3_w4a8_acc")
    oracle.apply_runtime_env_from_args(args)
    import torch.multiprocessing

    torch.multiprocessing.spawn(
        worker, args=(args.num_processes, args), nprocs=args.num_processes
    )


if __name__ == "__main__":
    main()
