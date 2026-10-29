# Changelog

All notable changes to this package are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the package uses
[semantic versioning](https://semver.org/) as interpreted by Julia's package
manager: before 1.0, a minor bump (0.1 → 0.2) may break things, a patch bump
(0.1.0 → 0.1.1) may not.

## [Unreleased]

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
