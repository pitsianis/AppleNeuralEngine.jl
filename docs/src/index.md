# AppleNeuralEngine.jl

Run Julia code on the **Apple Neural Engine** (ANE), the fp16 neural-network
accelerator in every Apple Silicon Mac.

```julia
using AppleNeuralEngine

W = randn(Float32, 512, 2048) ./ 23
layer(x) = max.(x * W, 0)                 # ordinary Julia

x = randn(Float16, 256, 512)
program = compile(layer, x)               # traced, compiled for the ANE
y = program(x)                            # runs on the Neural Engine
```

The package drives Apple's private `AppleNeuralEngine.framework` directly from
Julia through [ObjectiveC.jl](https://github.com/JuliaInterop/ObjectiveC.jl). There
is no native code to build and no Core ML model to export. It offers three
levels of control:

| level | you write | see |
|---|---|---|
| tracing | plain Julia on matrices: broadcasting, `*`, `sum`, ... | [Tracing Julia functions](@ref) |
| graph builder | MIL operations with inferred types, e.g. `MIL.conv(x, W)` | [The MIL graph builder](@ref) |
| raw runtime | MIL program text and buffers | [Raw MIL and the runtime](@ref) |

## Installation

```julia
using Pkg
Pkg.add("AppleNeuralEngine")
```

Requirements: an Apple Silicon Mac and Julia 1.10 or later. Nothing else needs to
be installed.

!!! warning "Private API"
    The ANE has no public programming interface; Apple exposes it only through
    Core ML. This package uses undocumented framework classes that may change
    with any macOS release. `initialize!()` checks that the classes it needs
    exist. Do not rely on it for production systems.

## What the Neural Engine is good at

- **fp16 matrix-matrix work.** A transformer feed-forward block (512 → 1536 → 512,
  256 tokens) runs in about 0.25 ms on an M2 Max, roughly 5 TFLOP/s. That is about
  15× faster than Float32 BLAS on the CPU (`examples/transformer_ffn.jl`).
- **Fixed graphs with constant weights.** Weights are compiled into the program.
- **Activations, normalizations, reductions, convolutions, attention.** See
  [Op support](@ref).

It is less suited to small, latency-bound work. Every call costs about 0.1–0.2 ms
on top of the compute, and the hardware computes only in fp16 (see
[Hardware notes](@ref)).

## Acknowledgements

- [maderix/ANE](https://github.com/maderix/ANE) reverse-engineered the private
  `_ANEInMemoryModel` workflow that this package uses: compiling MIL in memory,
  the model directory layout, and IOSurface-backed requests.
- [ObjectiveC.jl](https://github.com/JuliaInterop/ObjectiveC.jl) makes the
  Objective-C calls possible, and [Metal.jl](https://github.com/JuliaGPU/Metal.jl)
  showed how to build an Apple accelerator package on it.
- Apple's [coremltools](https://github.com/apple/coremltools) documents the MIL
  op set that the graph builder follows.
- [Orion](https://arxiv.org/abs/2603.06728) catalogs ANE constraints; our
  [Hardware notes](@ref) compare against them.
- Much of the package was written together with
  [Claude](https://claude.com/claude-code), Anthropic's AI assistant.
