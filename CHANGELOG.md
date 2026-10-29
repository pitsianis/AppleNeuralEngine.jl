# Changelog

All notable changes to this package are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the package uses
[semantic versioning](https://semver.org/) as interpreted by Julia's package
manager: before 1.0, a minor bump (0.1 → 0.2) may break things, a patch bump
(0.1.0 → 0.1.1) may not.

## [Unreleased]

## [0.1.0] - unreleased

First release.

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
