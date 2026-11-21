# Changelog

All notable changes to this package are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the package uses
[semantic versioning](https://semver.org/) as interpreted by Julia's package
manager: before 1.0, a minor bump (0.1 → 0.2) may break things, a patch bump
(0.1.0 → 0.1.1) may not.

## [Unreleased]

### Changed
- The op support page now covers several processors. `tools/probe_ops.jl` writes one
  result file per processor to `docs/op_support/`, and `tools/op_support_table.jl` builds
  the page from all of them: for each op, the processors it works and fails on, a roster
  of every M processor (measured or not), and a list of the ops whose outcome differs.
  First M4 results by Mathieu Bouchard (#4): `sin` and `cos` compile and run on the M4,
  unlike the M2 Max. First M1 Pro results: 139 of 195 cases run, and `gather`,
  `gather_along_axis` and `gather_nd` do not compile there although they do on the M2 Max.
  First M2 Pro results by jd-foster (#8), on macOS 14.8.9: 130 of 195 cases run; the gather
  family fails as on the M1 Pro, and so do `select`, `logical_and`, `reverse`, uint16
  argmax/argmin and `scaled_dot_product_attention`, which work on the other processors
  (all measured on macOS 27), so some of the differences may come from the OS.
- Requires ObjectiveC.jl 7.2 (was 6.0.1). Version 7 replaced `@autoreleasepool` with
  per-call pools and managed wrappers; the bindings now use managed wrappers for
  temporary objects (released eagerly, so the ANE gets a closed model's resources
  back at once), `Out{NSError}` for error out-parameters, and still keep each
  program's model and request as explicitly retained pointers, so that a program's
  finalizer unloads the model before anything releases it. `evaluate!` with
  preallocated outputs still allocates nothing.

### Fixed
- The `pad` test compared constant-mode padding exactly on random inputs and failed
  about one run in five: the ANE's constant pad is lossy for `|x| < 2^-7` and
  overflows for `|x| ≥ 2^15` (now documented in the `pad` docstring and the hardware
  notes). The test now uses values in [0.5, 4].
- Tracing is safe from several threads or tasks at once: the tracer state was one
  global, so concurrent `compile(f, x)` / `trace` calls could mix up their graphs
  ("Var … belongs to a different Graph"). It is now task-local.

### Changed
- The gather and scatter wrappers write `validate_indices` (default `false`) for
  opsets from iOS17 on; the ANE compiler rejects these ops without it. With int16 or
  uint16 indices, `gather`, `gather_along_axis` and `gather_nd` now compile and run,
  for constant indices and for uint16 indices computed in the program (e.g. cast from
  an fp16 input), with at most 2048 elements along the gathered axis. The scatter ops
  and `one_hot` still fail. `range_1d` works with fp16 operands.

### Added
- An experimental Reactant.jl front end, as a package extension: after `using Reactant`,
  `trace(f, inputs...; frontend=:reactant)` and `compile(f, inputs...; frontend=:reactant)`
  trace with Reactant and translate the optimized StableHLO module to MIL. It covers
  indexing, concatenation, padding, reductions, matrix products and arrays of up to
  3 dimensions, and loops are unrolled. Data movement (indexing, shifts, concatenation,
  padding) is exact. Its tests live in `test/reactant`, outside the main test target.
- Hardware notes on data movement: a `concat` of two slices of one tensor that start
  at the same index fails to compile or silently returns wrong values; slices along
  the innermost axis at some nonzero offsets, `reverse`, replicate padding,
  convolutions and matrix products are not exact copies, while `transpose`,
  `reshape`, other slices and reflect padding are. Workarounds are given.
- `output_dtype` keyword (`Int32` or `UInt16`) for `MIL.reduce_argmax` and
  `MIL.reduce_argmin`, and `indices_dtype`, `sort` and `return_indices` keywords for
  `MIL.topk`. The ANE rejects int32 tensors; with uint16 output, argmax and argmin
  run along channels and height (the result is fp16 bits labelled uint16, decoded with
  `reinterpret(Float16, UInt16(y))`, exact up to index 128). `topk` still does not compile.
- `evaluate!(outputs, program, inputs...)` writes into preallocated output arrays
  and allocates nothing per call. Without it, allocating large outputs made call
  times erratic (median about twice the minimum for a 2048×2048 product).
- `benchmark/run.jl` reuses the output array, reports minimum and median times, and
  can time with BenchmarkTools (`BENCH_METHOD=benchmarktools`) to measure sustained load.
- Probe cases for int32/int16/uint16 casts, uint16 argmax/argmin by axis and size, and
  uint16 `topk`; the op support page records the int32 rule and the decoding.

### Fixed
- `compile(f, x, y)` / `trace` with more than one Float32 input failed with
  "declare all inputs before adding ops".

## [0.1.0] - 2026-10-29

First release, not yet registered. Tested on an M2 Max with macOS 27.0.1:
the full test suite with Julia 1.13.3, and the tests that do not need a
Neural Engine (CI) with Julia 1.10.12 and 1.13.3.

### Added
- Direct access to the Apple Neural Engine through the private
  `AppleNeuralEngine.framework`, via ObjectiveC.jl (no native build step).
- `compile` / `evaluate!` for MIL program text, with `Float32` or `Float16`
  inputs and outputs and any tensor size (64-byte aligned row strides).
- `compile(f, inputs...)`: trace ordinary Julia functions on vectors and
  matrices (broadcasting, matrix products, reductions) into ANE programs.
- `AppleNeuralEngine.MIL`: a typed graph builder covering the coremltools MIL
  op set, with int8 weight compression (`constant_int8`).
- Weight files in Core ML's format: `WeightFile`, `pack_weights`,
  `build_weight_blob`.
- `functional()` and `device_info()`.
- `tools/probe_ops.jl`, which measures op support on the current machine.
- Examples: `traced_ffn.jl`, `transformer_ffn.jl`, `minimal.jl`.
- `benchmark/run.jl`: ANE vs Accelerate vs OpenBLAS, printed as a Markdown table.

[Unreleased]: https://github.com/pitsianis/AppleNeuralEngine.jl/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/pitsianis/AppleNeuralEngine.jl/releases/tag/v0.1.0
