# Op support

Which MIL ops the ANE compiles and runs, by processor. Each op runs in a small fp16 program
built by `tools/probe_ops.jl`; the number after a processor is the maximum absolute error
relative to the largest reference value (fp16 rounding gives ~1e-3), and **≈** marks results
within 10% (a coarse hardware approximation). "compile fails (InvalidMILProgram)" means the
ANE compiler's MIL validation rejected the op; every argument encoding we tried (inline or
`const` literals, other axes, other modes) was rejected. Core ML may lower these ops to
others before they reach the ANE. The ANE is a private, undocumented target, so results can
change with macOS releases as well as with the processor.

**Contributing a processor:** run `PROBE_BY="Your Name" julia --project=. tools/probe_ops.jl`
on it (after `julia --project=. -e 'using Pkg; Pkg.instantiate()'` in a fresh clone). The probe
writes `docs/op_support/<processor>.tsv` and regenerates this page; send both in a pull
request. Processors not yet measured have no entry in the tables.

## Processors

| generation | processor | measured | macOS | date | cases that run | by |
|---|---|---|---|---|---|---|
| M1 (2020–22) | M1 | not yet |  |  |  |  |
|  | M1 Pro | yes | 27.0.1 | 2026-11-18 | 139 of 195 | Camillo Schenone |
|  | M1 Max | not yet |  |  |  |  |
|  | M1 Ultra | not yet |  |  |  |  |
| M2 (2022–23) | M2 | not yet |  |  |  |  |
|  | M2 Pro | yes | 14.8.9 | 2026-11-20 | 130 of 195 | jd-foster |
|  | M2 Max | yes | 27.2.1 | 2026-11-16 | 148 of 195 | Nikos Pitsianis |
|  | M2 Ultra | not yet |  |  |  |  |
| M3 (2023–25) | M3 | not yet |  |  |  |  |
|  | M3 Pro | not yet |  |  |  |  |
|  | M3 Max | not yet |  |  |  |  |
|  | M3 Ultra | not yet |  |  |  |  |
| M4 (2024) | M4 | yes | 27.0.1 | 2026-10-30 | 130 of 157 | Mathieu Bouchard (#4) |
|  | M4 Pro | not yet |  |  |  |  |
|  | M4 Max | not yet |  |  |  |  |
| M5 (2025–26) | M5 | not yet |  |  |  |  |
|  | M5 Pro | not yet |  |  |  |  |
|  | M5 Max | not yet |  |  |  |  |
|  | M5 Ultra | not yet |  |  |  |  |

- **M4:** measured with the 167-case probe of 2026-10-30; the 10 cases the package has fixed or renamed since (`floor_div`, both `pad` modes, `gather_along_axis`, `gather_nd`, and the old `reduce_argmax`, `reduce_argmin`, `topk`, `range_1d`, `gather` cases) are left out until the probe is re-run on an M4.

## Differences between processors

- `cos` works on M4; fails on M1 Pro, M2 Pro, M2 Max.
- `sin` works on M4; fails on M1 Pro, M2 Pro, M2 Max.
- `logical_and` works on M1 Pro, M2 Max, M4; fails on M2 Pro.
- `select` works on M1 Pro, M2 Max, M4; fails on M2 Pro.
- `reduce_argmax (uint16) axis=1` works on M1 Pro, M2 Max; fails on M2 Pro.
- `reduce_argmax (uint16) axis=2` works on M1 Pro, M2 Max; fails on M2 Pro.
- `reduce_argmin (uint16) axis=1` works on M1 Pro, M2 Max; fails on M2 Pro.
- `reduce_argmin (uint16) axis=2` works on M1 Pro, M2 Max; fails on M2 Pro.
- `reduce_argmax (uint16) axis=1, C=129` works on M1 Pro, M2 Max; fails on M2 Pro.
- `reverse` works on M1 Pro, M2 Max, M4; fails on M2 Pro.
- `gather (int16 indices)` works on M2 Max; fails on M1 Pro, M2 Pro.
- `gather (uint16 indices)` works on M2 Max; fails on M1 Pro, M2 Pro.
- `gather axis=0` works on M2 Max; fails on M1 Pro, M2 Pro.
- `gather axis=2` works on M2 Max; fails on M1 Pro, M2 Pro.
- `gather axis=3` works on M2 Max; fails on M1 Pro, M2 Pro.
- `gather, runtime indices (cast fp16→uint16)` works on M2 Max; fails on M1 Pro, M2 Pro.
- `gather, runtime table, V=2048 rows` works on M2 Max; fails on M1 Pro, M2 Pro.
- `gather_along_axis` works on M2 Max; fails on M1 Pro, M2 Pro.
- `gather_nd` works on M2 Max; fails on M1 Pro, M2 Pro.
- `scaled_dot_product_attention` works on M1 Pro, M2 Max, M4; fails on M2 Pro.

Not probed (they need control flow, dynamic shapes or state, which this package does not
support yet): `cond`, `while_loop`, `make_list` and the `list_*` ops, `non_zero`,
`non_maximum_suppression`, `classify`, `read_state`, `slice_update`, `affine`, `resample`,
`crop_resize`, `reshape_like`, and the `constexpr_*` compression ops other than
`constexpr_affine_dequantize`.

Observed constraints, independent of the op (measured on the M2 Max):
- Computation must be fp16: the same ops on fp32 tensors do not compile (cast inputs).
- No int32 tensor may appear anywhere in a program: even a `cast` to int32 fails MIL
  validation. int16, uint16 and int8 work. Ops whose results are int32 by default take a
  uint16 option: `output_dtype=UInt16` for `reduce_argmax`/`reduce_argmin`,
  `indices_dtype=UInt16` for `topk`.
- `reduce_argmax`/`reduce_argmin` with uint16 output return fp16 bit patterns labelled
  uint16: after `cast(i, Float16)`, decode on the host with
  `reinterpret(Float16, UInt16(y))`. The decoded indices are exact up to index 128 and
  rounded beyond it (consistent with 7 significant bits); more than 2048 elements along
  the axis does not compile.
- Gather ops (`gather`, `gather_along_axis`, `gather_nd`) work with int16 or uint16
  indices, constant or computed (a computed index must be uint16, e.g. a cast of an
  fp16 input), when `validate_indices` is written out (the wrappers do), and at most
  2048 elements along the gathered axis. The scatter ops and `one_hot` still fail.
  The 2048 limits here and for argmax match the largest range in which fp16 holds
  every integer, so the ANE presumably carries indices as fp16.
- `topk` with uint16 indices passes MIL validation but the compiler backend rejects it
  (CompilationFailure) for every shape, axis, `k` and opset tried, even with
  `return_indices=false`.
- A trailing `identity` op cannot be a program output.
- Tensor buffers need a 64-byte aligned row stride, which this package provides, so any
  innermost size works.

## Elementwise unary

| op | works on | fails on | note |
|---|---|---|---|
| `abs` | M1 Pro (0.0), M2 Pro (0.0), M2 Max (0.0), M4 (0.0) |  |  |
| `acos` |  | M1 Pro, M2 Pro, M2 Max, M4 | compile fails (InvalidMILProgram) |
| `asin` |  | M1 Pro, M2 Pro, M2 Max, M4 | compile fails (InvalidMILProgram) |
| `atan` | M1 Pro ≈ (0.025), M2 Pro ≈ (0.025), M2 Max ≈ (0.025), M4 ≈ (0.025) |  |  |
| `atanh` |  | M1 Pro, M2 Pro, M2 Max, M4 | compile fails (InvalidMILProgram) |
| `ceil` | M1 Pro (0.0), M2 Pro (0.0), M2 Max (0.0), M4 (0.0) |  |  |
| `cos` | M4 (0.00063) | M1 Pro, M2 Pro, M2 Max | M1 Pro: compile fails (CompilationFailure); M2 Pro: compile fails (CompilationFailure); M2 Max: compile fails (CompilationFailure) |
| `cosh` |  | M1 Pro, M2 Pro, M2 Max, M4 | compile fails (InvalidMILProgram) |
| `erf` | M1 Pro (0.00084), M2 Pro (0.00084), M2 Max (0.00084), M4 (0.00084) |  |  |
| `exp` | M1 Pro (0.00076), M2 Pro (0.00076), M2 Max (0.00076), M4 (0.00076) |  |  |
| `exp2` | M1 Pro (0.00036), M2 Pro (0.00036), M2 Max (0.00036), M4 (0.00036) |  |  |
| `floor` | M1 Pro (0.0), M2 Pro (0.0), M2 Max (0.0), M4 (0.0) |  |  |
| `round` | M1 Pro (0.0), M2 Pro (0.0), M2 Max (0.0), M4 (0.0) |  |  |
| `sign` | M1 Pro (0.0), M2 Pro (0.0), M2 Max (0.0), M4 (0.0) |  |  |
| `sin` | M4 (0.00062) | M1 Pro, M2 Pro, M2 Max | M1 Pro: compile fails (CompilationFailure); M2 Pro: compile fails (CompilationFailure); M2 Max: compile fails (CompilationFailure) |
| `sinh` |  | M1 Pro, M2 Pro, M2 Max, M4 | compile fails (InvalidMILProgram) |
| `sqrt` | M1 Pro (0.00077), M2 Pro (0.00077), M2 Max (0.00077), M4 (0.00077) |  |  |
| `square` | M1 Pro (0.00029), M2 Pro (0.00029), M2 Max (0.00029), M4 (0.00029) |  |  |
| `tan` |  | M1 Pro, M2 Pro, M2 Max, M4 | compile fails (InvalidMILProgram) |
| `tanh` | M1 Pro (0.0016), M2 Pro (0.0016), M2 Max (0.0016), M4 (0.0016) |  |  |
| `inverse` | M1 Pro (0.00058), M2 Pro (0.00058), M2 Max (0.00058), M4 (0.00058) |  |  |
| `log` | M1 Pro (0.001), M2 Pro (0.001), M2 Max (0.001), M4 (0.001) |  |  |
| `rsqrt` | M1 Pro (0.0014), M2 Pro (0.0014), M2 Max (0.0014), M4 (0.0014) |  |  |
| `rsqrt (no epsilon)` |  | M1 Pro, M2 Pro, M2 Max, M4 | compile fails (InvalidMILProgram) |
| `threshold` | M1 Pro (1.4e-5), M2 Pro (1.4e-5), M2 Max (1.4e-5), M4 (1.4e-5) |  |  |
| `clip` | M1 Pro (0.0), M2 Pro (0.0), M2 Max (0.0), M4 (0.0) |  |  |
| `cast (fp16→int32→fp16)` |  | M1 Pro, M2 Pro, M2 Max | compile fails (InvalidMILProgram) |
| `cast (fp16→int16→fp16)` | M1 Pro (0.0), M2 Pro (0.0), M2 Max (0.0) |  |  |
| `cast (fp16→uint16→fp16)` | M1 Pro (0.0), M2 Pro (0.0), M2 Max (0.0) |  |  |
| `cast (fp16→fp32→fp16)` |  | M1 Pro, M2 Pro, M2 Max, M4 | compile fails (CompilationFailure) |
| `logical_not` | M1 Pro (0.0), M2 Pro (0.0), M2 Max (0.0), M4 (0.0) |  |  |

## Activation

| op | works on | fails on | note |
|---|---|---|---|
| `relu` | M1 Pro (0.0), M2 Pro (0.0), M2 Max (0.0), M4 (0.0) |  |  |
| `relu6` | M1 Pro (0.0), M2 Pro (0.0), M2 Max (0.0), M4 (0.0) |  |  |
| `sigmoid` | M1 Pro (0.0034), M2 Pro (0.0034), M2 Max (0.0034), M4 (0.0034) |  |  |
| `silu` | M1 Pro (0.0043), M2 Pro (0.0043), M2 Max (0.0043), M4 (0.0043) |  |  |
| `softplus` | M1 Pro (0.00076), M2 Pro (0.00076), M2 Max (0.00076), M4 (0.00076) |  |  |
| `softsign` | M1 Pro (0.00092), M2 Pro (0.00092), M2 Max (0.00092), M4 (0.00092) |  |  |
| `gelu` | M1 Pro (0.0016), M2 Pro (0.0016), M2 Max (0.0016), M4 (0.0016) |  |  |
| `gelu (tanh approx.)` | M1 Pro (0.0018), M2 Pro (0.0018), M2 Max (0.0018), M4 (0.0018) |  |  |
| `elu` | M1 Pro (0.00095), M2 Pro (0.00095), M2 Max (0.00095), M4 (0.00094) |  |  |
| `leaky_relu` | M1 Pro (4.1e-5), M2 Pro (4.1e-5), M2 Max (4.1e-5), M4 (5.5e-5) |  |  |
| `thresholded_relu` | M1 Pro (0.0), M2 Pro (0.0), M2 Max (0.0), M4 (0.0) |  |  |
| `clamped_relu` | M1 Pro (0.00015), M2 Pro (0.00015), M2 Max (0.00015), M4 (0.00015) |  |  |
| `linear_activation` | M1 Pro (0.00048), M2 Pro (0.00048), M2 Max (0.00048), M4 (0.00025) |  |  |
| `scaled_tanh` | M1 Pro (0.0017), M2 Pro (0.0017), M2 Max (0.0017), M4 (0.0017) |  |  |
| `sigmoid_hard` | M1 Pro (0.00029), M2 Pro (0.00029), M2 Max (0.00029), M4 (0.00029) |  |  |
| `prelu` | M1 Pro (0.00012), M2 Pro (0.00012), M2 Max (0.00012), M4 (9.4e-5) |  |  |
| `softplus_parametric` | M1 Pro (0.001), M2 Pro (0.001), M2 Max (0.001), M4 (0.0012) |  |  |
| `softmax` | M1 Pro (0.00057), M2 Pro (0.00057), M2 Max (0.00057), M4 (0.00074) |  |  |

## Elementwise binary

| op | works on | fails on | note |
|---|---|---|---|
| `add` | M1 Pro (0.00031), M2 Pro (0.00031), M2 Max (0.00031), M4 (0.00036) |  |  |
| `sub` | M1 Pro (0.00033), M2 Pro (0.00033), M2 Max (0.00033), M4 (0.00033) |  |  |
| `mul` | M1 Pro (0.00026), M2 Pro (0.00026), M2 Max (0.00026), M4 (0.00028) |  |  |
| `real_div` | M1 Pro (0.00068), M2 Pro (0.00068), M2 Max (0.00068), M4 (0.00073) |  |  |
| `pow` | M1 Pro (0.00075), M2 Pro (0.00075), M2 Max (0.00075), M4 (0.00075) |  |  |
| `maximum` | M1 Pro (0.0), M2 Pro (0.0), M2 Max (0.0), M4 (0.0) |  |  |
| `minimum` | M1 Pro (0.0), M2 Pro (0.0), M2 Max (0.0), M4 (0.0) |  |  |
| `floor_div` | M1 Pro (0.0), M2 Pro (0.0), M2 Max (0.0) |  |  |
| `mod` |  | M1 Pro, M2 Pro, M2 Max, M4 | compile fails (InvalidMILProgram) |
| `add (broadcast [1,C,1,1])` | M1 Pro (0.00031), M2 Pro (0.00031), M2 Max (0.00031), M4 (0.00036) |  |  |
| `mul (scalar literal)` | M1 Pro (0.00039), M2 Pro (0.00039), M2 Max (0.00039), M4 (0.00037) |  |  |
| `equal` | M1 Pro (0.0), M2 Pro (0.0), M2 Max (0.0), M4 (0.0) |  |  |
| `not_equal` | M1 Pro (0.0), M2 Pro (0.0), M2 Max (0.0), M4 (0.0) |  |  |
| `greater` | M1 Pro (0.0), M2 Pro (0.0), M2 Max (0.0), M4 (0.0) |  |  |
| `greater_equal` | M1 Pro (0.0), M2 Pro (0.0), M2 Max (0.0), M4 (0.0) |  |  |
| `less` | M1 Pro (0.0), M2 Pro (0.0), M2 Max (0.0), M4 (0.0) |  |  |
| `less_equal` | M1 Pro (0.0), M2 Pro (0.0), M2 Max (0.0), M4 (0.0) |  |  |
| `logical_and` | M1 Pro (0.0), M2 Max (0.0), M4 (0.0) | M2 Pro | M2 Pro: compile fails (InvalidMILProgram) |
| `logical_or` |  | M1 Pro, M2 Pro, M2 Max, M4 | compile fails (InvalidMILProgram) |
| `logical_xor` |  | M1 Pro, M2 Pro, M2 Max, M4 | compile fails (InvalidMILProgram) |
| `select` | M1 Pro (0.0), M2 Max (0.0), M4 (0.0) | M2 Pro | M2 Pro: compile fails (InvalidMILProgram) |

## Reduction

| op | works on | fails on | note |
|---|---|---|---|
| `reduce_sum axes=[1]` | M1 Pro (0.00087), M2 Pro (0.00087), M2 Max (0.00087), M4 (0.001) |  |  |
| `reduce_sum axes=[3]` | M1 Pro (0.00067), M2 Pro (0.00067), M2 Max (0.00067), M4 (0.0011) |  |  |
| `reduce_sum axes=[2, 3]` | M1 Pro (0.00056), M2 Pro (0.00056), M2 Max (0.00056), M4 (0.0011) |  |  |
| `reduce_mean axes=[1]` | M1 Pro (0.00067), M2 Pro (0.00067), M2 Max (0.00067), M4 (0.00057) |  |  |
| `reduce_mean axes=[3]` | M1 Pro (0.00091), M2 Pro (0.00091), M2 Max (0.00091), M4 (0.0013) |  |  |
| `reduce_mean axes=[2, 3]` | M1 Pro (0.00076), M2 Pro (0.00076), M2 Max (0.00076), M4 (0.00059) |  |  |
| `reduce_max axes=[1]` | M1 Pro (0.0), M2 Pro (0.0), M2 Max (0.0), M4 (0.0) |  |  |
| `reduce_max axes=[3]` | M1 Pro (0.0), M2 Pro (0.0), M2 Max (0.0), M4 (0.0) |  |  |
| `reduce_max axes=[2, 3]` | M1 Pro (0.0), M2 Pro (0.0), M2 Max (0.0), M4 (0.0) |  |  |
| `reduce_min axes=[1]` | M1 Pro (0.0), M2 Pro (0.0), M2 Max (0.0), M4 (0.0) |  |  |
| `reduce_min axes=[3]` | M1 Pro (0.0), M2 Pro (0.0), M2 Max (0.0), M4 (0.0) |  |  |
| `reduce_min axes=[2, 3]` | M1 Pro (0.0), M2 Pro (0.0), M2 Max (0.0), M4 (0.0) |  |  |
| `reduce_prod axes=[1]` |  | M1 Pro, M2 Pro, M2 Max, M4 | compile fails (InvalidMILProgram) |
| `reduce_l1_norm axes=[1]` | M1 Pro (0.0012), M2 Pro (0.0012), M2 Max (0.0012), M4 (0.0012) |  |  |
| `reduce_l1_norm axes=[3]` | M1 Pro (0.00069), M2 Pro (0.00069), M2 Max (0.00069), M4 (0.0011) |  |  |
| `reduce_l1_norm axes=[2, 3]` | M1 Pro (0.0008), M2 Pro (0.0008), M2 Max (0.0008), M4 (0.0009) |  |  |
| `reduce_l2_norm axes=[1]` | M1 Pro (0.00075), M2 Pro (0.00067), M2 Max (0.00067), M4 (0.00077) |  |  |
| `reduce_l2_norm axes=[3]` | M1 Pro (0.0006), M2 Pro (0.00058), M2 Max (0.00058), M4 (0.0008) |  |  |
| `reduce_l2_norm axes=[2, 3]` | M1 Pro (0.0011), M2 Pro (0.00083), M2 Max (0.00083), M4 (0.00057) |  |  |
| `reduce_sum_square axes=[1]` | M1 Pro (0.00083), M2 Pro (0.00083), M2 Max (0.00083), M4 (0.00063) |  |  |
| `reduce_sum_square axes=[3]` | M1 Pro (0.00077), M2 Pro (0.00077), M2 Max (0.00077), M4 (0.00077) |  |  |
| `reduce_sum_square axes=[2, 3]` | M1 Pro (0.00071), M2 Pro (0.00071), M2 Max (0.00071), M4 (0.001) |  |  |
| `reduce_log_sum axes=[1]` | M1 Pro (0.0012), M2 Pro (0.0012), M2 Max (0.0012), M4 (0.0014) |  |  |
| `reduce_log_sum axes=[3]` | M1 Pro (0.001), M2 Pro (0.001), M2 Max (0.001), M4 (0.0011) |  |  |
| `reduce_log_sum axes=[2, 3]` | M1 Pro (0.00099), M2 Pro (0.00099), M2 Max (0.00099), M4 (0.00095) |  |  |
| `reduce_log_sum_exp axes=[1]` | M1 Pro (0.00084), M2 Pro (0.00084), M2 Max (0.00084), M4 (0.00098) |  |  |
| `reduce_log_sum_exp axes=[3]` | M1 Pro (0.00077), M2 Pro (0.00077), M2 Max (0.00077), M4 (0.0011) |  |  |
| `reduce_log_sum_exp axes=[2, 3]` | M1 Pro (0.00095), M2 Pro (0.00095), M2 Max (0.00095), M4 (0.001) |  |  |
| `reduce_argmax (int32)` |  | M1 Pro, M2 Pro, M2 Max | compile fails (InvalidMILProgram) |
| `reduce_argmax (uint16) axis=0` |  | M1 Pro, M2 Pro, M2 Max | compile fails (CompilationFailure) |
| `reduce_argmax (uint16) axis=1` | M1 Pro (0.0), M2 Max (0.0) | M2 Pro (wrong) | M2 Pro: 479 of 512 indices wrong |
| `reduce_argmax (uint16) axis=2` | M1 Pro (0.0), M2 Max (0.0) | M2 Pro (wrong) | M2 Pro: 902 of 1024 indices wrong |
| `reduce_argmax (uint16) axis=3` |  | M1 Pro, M2 Pro, M2 Max | M1 Pro: compile fails (CompilationFailure); M2 Pro: compile fails; M2 Max: compile fails (CompilationFailure) |
| `reduce_argmin (int32)` |  | M1 Pro, M2 Pro, M2 Max | compile fails (InvalidMILProgram) |
| `reduce_argmin (uint16) axis=0` |  | M1 Pro, M2 Pro, M2 Max | compile fails (CompilationFailure) |
| `reduce_argmin (uint16) axis=1` | M1 Pro (0.0), M2 Max (0.0) | M2 Pro (wrong) | M2 Pro: 486 of 512 indices wrong |
| `reduce_argmin (uint16) axis=2` | M1 Pro (0.0), M2 Max (0.0) | M2 Pro (wrong) | M2 Pro: 894 of 1024 indices wrong |
| `reduce_argmin (uint16) axis=3` |  | M1 Pro, M2 Pro, M2 Max | M1 Pro: compile fails (CompilationFailure); M2 Pro: compile fails; M2 Max: compile fails (CompilationFailure) |
| `reduce_argmax (uint16) axis=1, C=129` | M1 Pro (0.0), M2 Max (0.0) | M2 Pro (wrong) | M2 Pro: 64 of 64 indices wrong |
| `reduce_argmax (uint16) axis=1, C=256` |  | M1 Pro (wrong), M2 Pro (wrong), M2 Max (wrong) | M1 Pro: 14 of 64 indices wrong; M2 Pro: 63 of 64 indices wrong; M2 Max: 14 of 64 indices wrong |
| `reduce_argmax (uint16) axis=1, C=1024` |  | M1 Pro (wrong), M2 Pro (wrong), M2 Max (wrong) | M1 Pro: 46 of 64 indices wrong; M2 Pro: 64 of 64 indices wrong; M2 Max: 46 of 64 indices wrong |

## Convolution

| op | works on | fails on | note |
|---|---|---|---|
| `conv 1x1` | M1 Pro (0.0004), M2 Pro (0.00036), M2 Max (0.0004), M4 (0.00036) |  |  |
| `conv 3x3 same` | M1 Pro (0.0003), M2 Pro (0.00036), M2 Max (0.0003), M4 (0.0004) |  |  |
| `conv 3x3 stride 2` | M1 Pro (0.00035), M2 Pro (0.00038), M2 Max (0.00035), M4 (0.00034) |  |  |
| `conv + bias` | M1 Pro (0.00031), M2 Pro (0.00049), M2 Max (0.00031), M4 (0.00051) |  |  |
| `conv depthwise 3x3` | M1 Pro (0.00038), M2 Pro (0.00043), M2 Max (0.00038), M4 (0.00034) |  |  |
| `conv int8 weights (constexpr_affine_dequantize)` | M1 Pro (0.0052), M2 Pro (0.0045), M2 Max (0.0052), M4 (0.0044) |  |  |
| `conv_transpose 2x2 stride 2` | M1 Pro (runs; no reference), M2 Pro (runs; no reference), M2 Max (runs; no reference), M4 (runs; no reference) |  |  |
| `conv 1D` | M1 Pro (runs; no reference), M2 Pro (runs; no reference), M2 Max (runs; no reference), M4 (runs; no reference) |  |  |

## Pooling

| op | works on | fails on | note |
|---|---|---|---|
| `max_pool` | M1 Pro (0.0), M2 Pro (0.0), M2 Max (0.0), M4 (0.0) |  |  |
| `avg_pool` | M1 Pro (0.00039), M2 Pro (0.00054), M2 Max (0.00039), M4 (0.00068) |  |  |
| `l2_pool` | M1 Pro (0.00055), M2 Pro (0.00062), M2 Max (0.00055), M4 (0.00051) |  |  |

## Linear algebra

| op | works on | fails on | note |
|---|---|---|---|
| `matmul` | M1 Pro (0.00038), M2 Pro (0.00025), M2 Max (0.00038), M4 (0.00045) |  |  |
| `matmul transpose_y` | M1 Pro (0.00023), M2 Pro (0.00026), M2 Max (0.00023), M4 (0.00028) |  |  |
| `matmul (constant)` | M1 Pro (0.00047), M2 Pro (0.00036), M2 Max (0.00047), M4 (0.00046) |  |  |
| `linear` | M1 Pro (0.00041), M2 Pro (0.00033), M2 Max (0.00041), M4 (0.00045) |  |  |
| `einsum nchw,nwhu->nchu` | M1 Pro (0.00034), M2 Pro (0.00038), M2 Max (0.00034), M4 (0.00032) |  |  |

## Normalization

| op | works on | fails on | note |
|---|---|---|---|
| `layer_norm` | M1 Pro (0.00093), M2 Pro (0.0012), M2 Max (0.00093), M4 (0.00091) |  |  |
| `layer_norm gamma/beta (last axis)` | M1 Pro (0.0011), M2 Pro (0.00078), M2 Max (0.0011), M4 (0.001) |  |  |
| `batch_norm` | M1 Pro (0.00056), M2 Pro (0.00056), M2 Max (0.00056), M4 (0.00056) |  |  |
| `instance_norm` | M1 Pro (0.0012), M2 Pro (0.00087), M2 Max (0.0012), M4 (0.00091) |  |  |
| `l2_norm` | M1 Pro (runs; no reference), M2 Pro (runs; no reference), M2 Max (runs; no reference), M4 (runs; no reference) |  |  |
| `local_response_norm` | M1 Pro (runs; no reference), M2 Pro (runs; no reference), M2 Max (runs; no reference), M4 (runs; no reference) |  |  |

## Tensor transformation

| op | works on | fails on | note |
|---|---|---|---|
| `reshape` | M1 Pro (0.0), M2 Pro (0.0), M2 Max (0.0), M4 (0.0) |  |  |
| `transpose` | M1 Pro (0.0), M2 Pro (0.0), M2 Max (0.0), M4 (0.0) |  |  |
| `transpose (last two)` | M1 Pro (0.0), M2 Pro (0.0), M2 Max (0.0), M4 (0.0) |  |  |
| `expand_dims + squeeze` | M1 Pro (0.0), M2 Pro (0.0), M2 Max (0.0), M4 (0.0) |  |  |
| `reverse` | M1 Pro (2.2e-6), M2 Max (2.2e-6), M4 (2.2e-6) | M2 Pro | M2 Pro: compile fails (InvalidMILProgram) |
| `slice_by_index` | M1 Pro (0.0), M2 Pro (0.0), M2 Max (0.0), M4 (0.0) |  |  |
| `slice_by_index (stride 2)` | M1 Pro (0.0), M2 Pro (0.0), M2 Max (0.0), M4 (0.0) |  |  |
| `slice_by_size` | M1 Pro (0.0), M2 Pro (0.0), M2 Max (0.0), M4 (0.0) |  |  |
| `depth_to_space` | M1 Pro (runs; no reference), M2 Pro (runs; no reference), M2 Max (runs; no reference), M4 (runs; no reference) |  |  |
| `space_to_depth` | M1 Pro (runs; no reference), M2 Pro (runs; no reference), M2 Max (runs; no reference), M4 (runs; no reference) |  |  |
| `pixel_shuffle` | M1 Pro (runs; no reference), M2 Pro (runs; no reference), M2 Max (runs; no reference), M4 (runs; no reference) |  |  |
| `pixel_unshuffle` |  | M1 Pro, M2 Pro, M2 Max, M4 | compile fails (InvalidMILProgram) |
| `sliding_windows` |  | M1 Pro, M2 Pro, M2 Max, M4 | compile fails (InvalidMILProgram) |

## Tensor operation

| op | works on | fails on | note |
|---|---|---|---|
| `concat` | M1 Pro (0.0), M2 Pro (0.0), M2 Max (0.0), M4 (0.0) |  |  |
| `concat (last axis)` | M1 Pro (0.0), M2 Pro (0.0), M2 Max (0.0), M4 (0.0) |  |  |
| `split` | M1 Pro (0.0), M2 Pro (0.0), M2 Max (0.0), M4 (0.0) |  |  |
| `stack` | M1 Pro (0.0), M2 Pro (0.0), M2 Max (0.0), M4 (0.0) |  |  |
| `tile` | M1 Pro (0.0), M2 Pro (0.0), M2 Max (0.0), M4 (0.0) |  |  |
| `pad (constant)` | M1 Pro (1.9e-6), M2 Pro (1.9e-6), M2 Max (1.9e-6) |  |  |
| `pad (reflect)` | M1 Pro (runs; no reference), M2 Pro (runs; no reference), M2 Max (runs; no reference) |  |  |
| `flatten2d` | M1 Pro (0.0), M2 Pro (0.0), M2 Max (0.0), M4 (0.0) |  |  |
| `band_part` |  | M1 Pro, M2 Pro, M2 Max, M4 | compile fails (InvalidMILProgram) |
| `cumsum` |  | M1 Pro, M2 Pro, M2 Max, M4 | compile fails (InvalidMILProgram) |
| `topk (int32)` |  | M1 Pro, M2 Pro, M2 Max | compile fails (InvalidMILProgram) |
| `topk (uint16) axis=1` |  | M1 Pro, M2 Pro, M2 Max | compile fails (CompilationFailure) |
| `topk (uint16) axis=2` |  | M1 Pro, M2 Pro, M2 Max | compile fails (CompilationFailure) |
| `topk (uint16) axis=3` |  | M1 Pro, M2 Pro, M2 Max | compile fails (CompilationFailure) |
| `topk (values only)` |  | M1 Pro, M2 Pro, M2 Max | compile fails (CompilationFailure) |
| `argsort` |  | M1 Pro, M2 Pro, M2 Max, M4 | M1 Pro: compile fails (InvalidMILProgram); M2 Pro: compile fails (CompilationFailure); M2 Max: compile fails (InvalidMILProgram); M4: compile fails (InvalidMILProgram) |
| `one_hot` |  | M1 Pro, M2 Pro, M2 Max, M4 | compile fails (InvalidMILProgram) |
| `fill` | M1 Pro (0.00037), M2 Pro (0.00034), M2 Max (0.00037), M4 (0.00037) |  |  |
| `range_1d (int32)` |  | M1 Pro, M2 Pro, M2 Max | compile fails (InvalidMILProgram) |
| `range_1d (fp16)` | M1 Pro (0.00044), M2 Pro (0.00047), M2 Max (0.00044) |  |  |
| `shape` |  | M1 Pro, M2 Pro, M2 Max, M4 | compile fails (InvalidMILProgram) |
| `identity` | M1 Pro (0.0), M2 Pro (0.0), M2 Max (0.0), M4 (0.0) |  |  |

## Scatter/gather

| op | works on | fails on | note |
|---|---|---|---|
| `gather (int32 indices)` |  | M1 Pro, M2 Pro, M2 Max | compile fails (InvalidMILProgram) |
| `gather (int16 indices)` | M2 Max (0.0) | M1 Pro, M2 Pro | M1 Pro: compile fails (CompilationFailure); M2 Pro: compile fails (InvalidMILProgram) |
| `gather (uint16 indices)` | M2 Max (0.0) | M1 Pro, M2 Pro | M1 Pro: compile fails (CompilationFailure); M2 Pro: compile fails (InvalidMILProgram) |
| `gather axis=0` | M2 Max (0.0) | M1 Pro, M2 Pro | M1 Pro: compile fails (CompilationFailure); M2 Pro: compile fails (InvalidMILProgram) |
| `gather axis=2` | M2 Max (0.0) | M1 Pro, M2 Pro | M1 Pro: compile fails (CompilationFailure); M2 Pro: compile fails (InvalidMILProgram) |
| `gather axis=3` | M2 Max (0.0) | M1 Pro, M2 Pro | M1 Pro: compile fails (CompilationFailure); M2 Pro: compile fails (InvalidMILProgram) |
| `gather, runtime indices (cast fp16→int16)` |  | M1 Pro, M2 Pro (wrong), M2 Max | M1 Pro: compile fails (InvalidMILProgram); M2 Max: compile fails (InvalidMILProgram) |
| `gather, runtime indices (cast fp16→uint16)` | M2 Max (0.0) | M1 Pro, M2 Pro (wrong) | M1 Pro: compile fails (CompilationFailure) |
| `gather, runtime table, V=2048 rows` | M2 Max (0.0) | M1 Pro, M2 Pro (wrong) | M1 Pro: compile fails (CompilationFailure) |
| `gather, runtime table, V=2049 rows` |  | M1 Pro, M2 Pro (wrong), M2 Max | M1 Pro: compile fails (InvalidMILProgram); M2 Max: compile fails (InvalidMILProgram) |
| `gather_along_axis` | M2 Max (0.0) | M1 Pro, M2 Pro | M1 Pro: compile fails (CompilationFailure); M2 Pro: compile fails (InvalidMILProgram) |
| `gather_nd` | M2 Max (0.0) | M1 Pro, M2 Pro | M1 Pro: compile fails (CompilationFailure); M2 Pro: compile fails (InvalidMILProgram) |
| `scatter` |  | M1 Pro, M2 Pro, M2 Max, M4 | compile fails (InvalidMILProgram) |
| `scatter_along_axis` |  | M1 Pro, M2 Pro, M2 Max, M4 | compile fails (InvalidMILProgram) |
| `scatter_nd` |  | M1 Pro, M2 Pro, M2 Max, M4 | compile fails (InvalidMILProgram) |

## Image resizing

| op | works on | fails on | note |
|---|---|---|---|
| `upsample_nearest_neighbor` | M1 Pro (2.4e-6), M2 Pro (1.5e-6), M2 Max (2.4e-6), M4 (2.2e-6) |  |  |
| `upsample_bilinear` | M1 Pro (runs; no reference), M2 Pro (runs; no reference), M2 Max (runs; no reference), M4 (runs; no reference) |  |  |
| `resize_nearest_neighbor` | M1 Pro (2.0e-6), M2 Pro (2.3e-6), M2 Max (2.0e-6), M4 (1.9e-6) |  |  |
| `resize_bilinear` | M1 Pro (runs; no reference), M2 Pro (runs; no reference), M2 Max (runs; no reference), M4 (runs; no reference) |  |  |
| `crop` | M1 Pro (0.00013), M2 Pro (0.00014), M2 Max (0.00013), M4 (0.0) |  |  |

## Attention

| op | works on | fails on | note |
|---|---|---|---|
| `scaled_dot_product_attention` | M1 Pro (0.00095), M2 Max (0.00095), M4 (0.0014) | M2 Pro | M2 Pro: compile fails (InvalidMILProgram) |

## Recurrent

| op | works on | fails on | note |
|---|---|---|---|
| `lstm` |  | M1 Pro, M2 Pro, M2 Max, M4 | compile fails (InvalidMILProgram) |
| `gru` |  | M1 Pro, M2 Pro, M2 Max, M4 | compile fails (InvalidMILProgram) |
| `rnn` |  | M1 Pro, M2 Pro, M2 Max, M4 | compile fails (InvalidMILProgram) |

## Quantization

| op | works on | fails on | note |
|---|---|---|---|
| `quantize → dequantize` | M1 Pro (0.013), M2 Pro (0.013), M2 Max (0.013), M4 (0.014) |  |  |

## Random

| op | works on | fails on | note |
|---|---|---|---|
| `random_normal` |  | M1 Pro, M2 Pro, M2 Max, M4 | compile fails (InvalidMILProgram) |
| `random_uniform` |  | M1 Pro, M2 Pro, M2 Max, M4 | compile fails (InvalidMILProgram) |
