# A3 MegaMoE compatibility and upstream memory fix

This bundle targets Ascend A3 (`Ascend910_93`), ordinary SwiGLU, and ModelSlim
per-channel W4A8/W8A8 INT experts. It contains only the small DeepEP compatibility
patch, an apply/check/reverse script, and target validation commands. Token
sharding and experimental residual-path changes are excluded.

## Native memory fix is already upstream

The capacity-sized MegaMoE workspace issue was addressed by CANN
[merge request !11324, "chunk send and recv for megamoe"](https://gitcode.com/cann/ops-transformer/merge_requests/11324),
merged into `master` on September 7, 2026 as
[`c465a55ac0cd9680f8a96795a29e6e5781a90fd9`](https://gitcode.com/cann/ops-transformer/commit/c465a55ac0cd9680f8a96795a29e6e5781a90fd9).
The fix is also present in
[mirrored master at d00268a7](https://github.com/hicann/ops-transformer/blob/d00268a7d36d1b8bf1e344ad0e8a086edecfb2d7/mc2/mega_moe/op_host/op_tiling/arch22/mega_moe_tiling_arch22.cpp#L1593).
Use a compatible native build containing this commit; do not reapply the old
backport to a source tree that already includes it. No native patch is bundled.

The observed GLM W4A8 failure requested 18.02 GiB for dispatch capacity 4096,
EP16/top-8, hidden 6144, and intermediate size 2048. The old receive bound was
`4096 * 16 * 8 = 524288` rows. The upstream implementation uses 1024-token chunks
and a receive-round budget to reuse the large GEMM workspaces. Source formulas
predict roughly 4.5 GiB of dominant workspace for that geometry, but this is not
a measured result or total process memory. The installed native operator's
revision and actual memory use still need verification on A3.

Dispatch capacity 1024 is a smaller mitigation when every rank's actual input
rows fit. It limits supported batches and does not replace the native fix.
Do not reduce receive capacity based on an assumption of balanced routing.
A CANN version string alone does not prove that an installed binary contains
the fix. Keep native host libraries, device kernels, and the torch extension
from the same compatible package set.

## Apply the DeepEP compatibility patch

The patch is pinned to `sgl-kernel-npu`
`00db6cd6bd2280650281ea187292fe8913619abf`. It adapts legacy operators only for
ordinary SwiGLU with no activation parameters, preserving modern APIs and W4A8
compensation biases. Oversized inputs are rejected before symmetric-buffer
allocation. This Python patch does not change native workspace allocation.
The source license is reproduced in `LICENSE.sgl-kernel-npu`.

Run inside the A3 serving environment, using a separate clean source checkout:

```bash
export SGLANG_ROOT=/path/to/sglang
export DEPS_ROOT=/path/to/megamoe-deps
export BUNDLE="$SGLANG_ROOT/tools/ascend/megamoe_a3"
mkdir -p "$DEPS_ROOT"
git clone https://github.com/sgl-project/sgl-kernel-npu.git "$DEPS_ROOT/sgl-kernel-npu"
git -C "$DEPS_ROOT/sgl-kernel-npu" checkout --detach 00db6cd6bd2280650281ea187292fe8913619abf
python3 "$BUNDLE/apply_patches.py" --sgl-kernel-npu "$DEPS_ROOT/sgl-kernel-npu" --check
python3 "$BUNDLE/apply_patches.py" --sgl-kernel-npu "$DEPS_ROOT/sgl-kernel-npu"
```

The script checks the revision, patch SHA-256, exact before/after file hashes,
and applicability before writing. Repeating it is safe. Partial applications,
edited files, staged changes, and unrelated tracked modifications are rejected.
It does not fetch, build, install, commit, or change branches.

## Build and install DeepEP on A3

Activate the serving Python environment and source its CANN `set_env.sh`.
Use a matching native MegaMoE package set containing the upstream workspace fix.
The apply script does not install or replace CANN or `ops-transformer`.

```bash
(cd "$DEPS_ROOT/sgl-kernel-npu" && bash build.sh -a deepep Ascend910_9382)
export DEEPEP_WHEEL=/absolute/path/to/new/deep_ep.whl
python3 -m pip install --no-deps --force-reinstall "$DEEPEP_WHEEL"
```

The upstream build may obtain its build dependencies. Save the previously
installed DeepEP wheel before installing, and restart serving processes afterward.
Capture toolkit/operator versions and paths, the native source/build revision,
wheel hashes, and environment values before validating:

```bash
npu-smi info
git -C "$SGLANG_ROOT" rev-parse HEAD
git -C "$DEPS_ROOT/sgl-kernel-npu" rev-parse HEAD
sha256sum "$DEEPEP_WHEEL"
python3 -m pip show torch torch-npu deep-ep sgl-kernel-npu cann-ops-transformer
python3 - <<'PY'
import inspect
import deep_ep
import cann_ops_transformer
from cann_ops_transformer.ops import mega_moe
print("deep_ep:", deep_ep.__file__)
print("cann_ops_transformer:", cann_ops_transformer.__file__)
print("DeepEP:", inspect.signature(deep_ep.Buffer.fused_deep_moe))
print("operator:", inspect.signature(mega_moe))
PY
```

## Validate capacity, routing, numerical output, and memory

`validate_a3.py` uses the pinned dependency's existing MC2 small-op numerical
oracle and expert-token-count checks. It keeps the original W4A8/W8A8 tolerances,
forces ordinary SwiGLU, and logs process-local peak allocated/reserved NPU bytes
around each fused invocation. These peaks include live weights/reference buffers;
they are not an isolated native-workspace measurement. Capture allocator/operator
logs as well, and run the same cases with the previous package set in a separate
environment for a memory comparison.

One A3 node must expose 16 NPU devices. For multi-node execution, set
`--num-processes` to local devices, `--num-servers`, `--server-index`, and the same
`--master-addr`/`--master-port` on each host. Run cases sequentially:

```bash
set -euo pipefail
mkdir -p "$DEPS_ROOT/validation"
for quant in w4a8 w8a8; do
  for rows in 1 1023 1024 1025 2048 4096; do
    python3 "$BUNDLE/validate_a3.py" \
      --sgl-kernel-npu "$DEPS_ROOT/sgl-kernel-npu" \
      --quant-mode "$quant" --num-tokens "$rows" --num-processes 16 \
      2>&1 | tee "$DEPS_ROOT/validation/${quant}-${rows}-random.log"
  done
  python3 "$BUNDLE/validate_a3.py" \
    --sgl-kernel-npu "$DEPS_ROOT/sgl-kernel-npu" \
    --quant-mode "$quant" --num-tokens 4096 --num-processes 16 \
    --routing skew --unequal-rows \
    2>&1 | tee "$DEPS_ROOT/validation/${quant}-4096-skew.log"
done
```

The skew case sends every rank to experts 0..7. Unequal local rows exercise
padding and masking. Stop on failed numerical or token-count checks; inspect
the complete logs rather than treating a completed native call as acceptance.
These operator checks do not establish full-model quality or NPUGraph correctness.

For full-weight serving, use the GLM tutorial's eager TP16/EP16 recipe with
`SGLANG_NPU_ENABLE_DSA_TOKEN_SHARD=0`, shared experts separate, and no speculation.
Start with dispatch capacity/global chunk 512; increase only after observing
actual rank-local entry rows. The global scheduler chunk and per-rank dispatch
capacity are different limits. This port makes no token-sharding assumption to
map one to the other. Retain resolved server arguments, dtype/checkpoint descriptor,
TP/DP/EP, HCCL/network configuration, package identities, model output, and memory
logs. Compare against DeepEP with the same settings before enabling graph replay.
The earlier 24.59 s recipe depended on additional changes; it is not a promised
baseline for this isolated port.

## Roll back

Stop processes using DeepEP. Restore the saved DeepEP wheel and restart them.
Reversing source changes alone does not revert the installed package.

```bash
python3 "$BUNDLE/apply_patches.py" --sgl-kernel-npu "$DEPS_ROOT/sgl-kernel-npu" --reverse --check
python3 "$BUNDLE/apply_patches.py" --sgl-kernel-npu "$DEPS_ROOT/sgl-kernel-npu" --reverse
```

## Host verification

The port passed 12 CPU/mock checks of weight layouts, scale bits, compensation,
atomic rejection, idempotence, mode 1/2 behavior, CLI constraints, routed scaling,
and legacy/modern API compatibility. The dependency helper was also exercised
directly. The retained compatibility patch passed exact-file hash verification,
forward/reverse and repeat application, and rejection of modified sources and
corrupted patch data. Python syntax and focused Ruff checks passed. Native
compilation, numerical correctness, graph replay, memory use, and performance
remain pending A3 validation. No packages were installed while preparing this
bundle.
