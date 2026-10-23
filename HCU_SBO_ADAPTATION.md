# HCU FP8/W4A8 single-batch overlap adaptation

This commit contains the SGLang-side FP8 pair32 and W4A8 masked HIPC
SBO adaptation on top of 52e8a1b20 (the remote HY4 branch as of 2026-10-23).
It includes completion-signal capability gating, runner/legacy overlap-state
lookup, FP8 paired weight packing and GEMM selection, kernel-returned completion
metadata, the W4A8 Tensor-return compatibility fix, and HCU DeepEP combine API
compatibility. It also includes HY4 main/MTP auxiliary-stream creation for ROCm
multi-stream or SBO. Other uncommitted HY4 changes are not part of this commit.

Production helpers live in the existing MoE layer and FP8 quantization files;
this commit adds no production modules. It preserves the remote branch's PP
support, DeepEP context-pool fix, and normal-mode contiguous FP8 packing.

## Apply

Use the format-patch generated from this commit in a compatible checkout:

```bash
git am /path/to/0001-hcu-fp8-w4a8-sbo-consolidated.patch
```

For an exact baseline, check out this commit's parent first. For other revisions,
review the diff and run `git apply --check /path/to/0001-hcu-fp8-w4a8-sbo-consolidated.patch`
before applying. Do not stack this patch over the older incremental SBO patches
or a working tree that already contains the same changes.

## External dependencies

Install the provided DeepGEMM wheel in the environment used by the server:

```bash
python3 -m pip install --no-deps --force-reinstall   /path/to/deepgemm-2.1.0+das.dtk26041.34449c5-cp310-cp310-linux_x86_64.whl
```

The wheel is not embedded in this patch. This patch does not change DeepGEMM or
DeepEP native source. The HCU DeepEP low-latency combine must support the matching
completion-signal contract (comp_signal, block_m, threshold and related args).
The FP8 path requires gfx938 plus the pair32 packer/wrapper/native kernel. The
W4A8 path requires the masked HIPC API with enable_overlap and signal parameters.
A matching Python signature alone is not proof of native binary compatibility.

The simplified implementation targets this fixed dependency combination. It
does not probe old HIPC signatures, test native pair32 registration, or fall back
between old/new packer aliases. In low-latency mode on gfx938 it calls the pair32
packer directly; other architectures retain the current wheel's legacy `_inplace`
packer. Normal and auto modes retain the remote branch's
`marlin_fp8_contiguous_weight` packer and do not advertise pair32 SBO. Missing
required symbols fail normally. HCU combine always omits the `overlap` keyword;
its signal protocol uses `comp_signal`. Expert capability is exposed once on
DeepEPMoE rather than repeating backend checks in three model hooks.

## Enable

Start from the model's already working DeepEP low-latency launch configuration.
Enable `--enable-single-batch-overlap`. For the validated HY4 configuration, keep
shared experts separate using `--disable-shared-experts-fusion`.

For W4A8, use the slimquant_w4a8_marlin quantization path with:

```bash
export SGLANG_USE_LIGHTOP_W4A8_MARLIN_MOE=0
export SGLANG_USE_W4A8_MASKED_HIPC=1
export SGLANG_ROCM_USE_MULTI_STREAM=1
```

For the HCU channelwise FP8 path, use its FP8 model/quantization configuration and:

```bash
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_ROCM_USE_MULTI_STREAM=1
```

Set these before Python imports the server modules, then restart the server.
Check the actual imported DeepGEMM version and location: historical activation
scripts or PYTHONPATH overrides can hide an installed wheel.

The validated steady-state profiling configuration used
`HIP_GRAPH_ACCUMULATE_DISPATCH=0` while keeping GPU graphs enabled. This variable
is a runtime dispatch setting, not the SBO enable flag. A launch script that
exports 1 later will override an earlier environment export of 0; update the
actual launch configuration or make a reviewed run-specific copy. Machine-specific
launch scripts and old activation paths are deliberately not shipped here.

## Behavior and compatibility boundaries

The code is gated by hardware, backend and quantization, not by the HY4 model
name. Other models sharing these paths can also change behavior. FP8 pair32
packing/kernel selection in low-latency mode may happen even with SBO disabled. Ordinary W4A8 HIPC
calls retain the Tensor-return fix even with SBO disabled. The HCU DeepEP API
compatibility rule applies to all low-latency quantization paths.

These changes wire down-GEMM/combine overlap for supported masked low-latency
paths; they do not automatically enable overlap for prefill, every normal-mode
consumer, every model, or every topology. Kernel-returned block_m/threshold must
match the installed DeepEP signal protocol. Disabling SBO does not restore the
old FP8 weight layout; packing and GEMM selection must be rolled back together.

## Tests and evidence

Run the two targeted CPU test files in an installed SGLang development environment:

```bash
PYTHONPATH=python python3 -m pytest -q   test/registered/unit/layers/moe/test_hcu_fp8_sbo.py   test/registered/unit/layers/moe/test_hcu_w4a8_sbo.py
```

These mock-based tests execute production functions extracted from the existing
modules, without importing GPU-only dependencies. They cover packing/kernel
selection, completion metadata, normal Tensor returns, and HY4 auxiliary-stream
creation. They do not establish model accuracy or GPU overlap.

Historical FP8 probes covered BM64/BM128 and Graph signal replay. The 2026-10-22
S5 W4A8 trace (TP8/DP8/DCP1/EP8, 128K input, 4 running requests per DP, 5 steps)
showed down-GEMM/combine kernel overlap on some ranks; the S4 SBO-off trace showed
none. These are functional observations from different machines, not a
same-machine performance comparison. This consolidation has not been rerun as a
full-model GPU test.

Dispatch/shared-expert scheduling and MTP's final auxiliary-stream wait already
existed; this commit does not introduce them. DCP2/SBO support is not claimed.
