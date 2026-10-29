# AppleNeuralEngine.jl

Run Julia code on the **Apple Neural Engine** (ANE), the fp16 neural-network
accelerator in every Apple Silicon Mac.

```julia
using AppleNeuralEngine

W = randn(Float32, 512, 2048) ./ 23
layer(x) = max.(x * W, 0)          # ordinary Julia

x = randn(Float16, 256, 512)
program = compile(layer, x)        # traced and compiled for the Neural Engine
y = program(x)                     # runs on the ANE
```

The package drives Apple's private `AppleNeuralEngine.framework` directly from
Julia through [ObjectiveC.jl](https://github.com/JuliaInterop/ObjectiveC.jl), the
same way [Metal.jl](https://github.com/JuliaGPU/Metal.jl) drives the GPU. There is
no native code to build and no Core ML model to export.

## Three levels

- **Tracing:** `compile(f, inputs...)` records an ordinary Julia function on
  vectors and matrices (broadcasting with any scalar function, matrix products,
  `sum`/`maximum`/`minimum`) as an ANE program. Captured arrays become weights.
- **Graph builder:** `AppleNeuralEngine.MIL` builds programs op by op, following
  the coremltools MIL op set, with type and shape inference, int8 weights and
  automatic weight files.
- **Raw runtime:** `compile(mil_text; input_shapes, output_shapes, weight_blob)`
  and `evaluate!` for hand-written MIL.

## Examples

```sh
julia --project=. examples/traced_ffn.jl       # the same Julia function on CPU and ANE
julia --project=. examples/transformer_ffn.jl  # the same block with the MIL graph builder
julia --project=. examples/minimal.jl          # hand-written MIL, raw runtime
```

`transformer_ffn.jl` runs a pre-norm SwiGLU block (512 → 1536 → 512, 256 tokens)
and checks it against plain Julia. On an M2 Max:

```
relative error vs Float32 CPU: 0.00422 (fp16 compute)
ANE (incl. copies)     0.255 ms   4734.8 GFLOP/s
CPU Float32            3.832 ms    315.2 GFLOP/s
```

## Benchmark

```sh
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark benchmark/run.jl
```

It times the FFN block and a few matrix products on the ANE, with Accelerate and
with OpenBLAS, and prints a Markdown table to share. On an M2 Max:

| workload | ANE fp16 ms | GFLOP/s | Accelerate fp32 ms | GFLOP/s | OpenBLAS fp32 ms | GFLOP/s |
|---|--:|--:|--:|--:|--:|--:|
| FFN block D=512 H=1536 T=256 | 0.255 | 4730 | 4.519 | 267 | 5.718 | 211 |
| x*W 256×1024 · 1024×1024 | 0.143 | 3750 | 0.850 | 632 | 1.564 | 343 |
| x*W 1024×1024 · 1024×1024 | 0.378 | 5677 | 3.643 | 590 | 6.142 | 350 |
| x*W 2048×2048 · 2048×2048 | 1.648 | 10422 | 29.314 | 586 | 47.979 | 358 |

## What works

`tools/probe_ops.jl` compiles and checks every MIL op on the current machine; the
result is in [docs/src/op_support.md](docs/src/op_support.md). On an M2 Max,
129 of 167 probed cases work: elementwise arithmetic, activations, reductions,
convolutions (including depthwise, strided and int8-weight), pooling,
matmul/linear/einsum, normalizations, scaled dot-product attention and the usual
reshaping ops. `sin`/`cos`/`tan`, `pad`, gather/scatter, `cumsum`, `topk`,
recurrent layers and random ops are rejected by the ANE compiler.

The hardware computes in fp16 only. Each call costs about 0.1–0.2 ms on top of the
compute, so the ANE pays off for matrix-matrix work, not single small vectors. See
[docs/src/hardware.md](docs/src/hardware.md).

## Documentation

```sh
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl     # then open docs/build/index.html
```

## Tests

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
```

Tests that need a Neural Engine are skipped when `functional()` is false (other
hardware, virtual machines) or when `APPLENEURALENGINE_SKIP_ANE=1`.

## Caveats

This package uses undocumented Apple APIs. They can change with any macOS release
and are not suitable for production use. `initialize!()` checks that the private
classes it needs exist and raises an `ANEError` otherwise.

## Acknowledgements

- [maderix/ANE](https://github.com/maderix/ANE) reverse-engineered the private
  `_ANEInMemoryModel` / `_ANERequest` / `_ANEIOSurfaceObject` workflow this
  package uses: compiling MIL in memory, the model directory layout, and
  IOSurface-backed requests. AppleNeuralEngine.jl started as a Julia wrapper around
  that project's Objective-C bridge, and the runtime calls in `src/native.jl`
  follow its sequence.
- [ObjectiveC.jl](https://github.com/JuliaInterop/ObjectiveC.jl) makes the
  Objective-C calls from Julia possible, and
  [Metal.jl](https://github.com/JuliaGPU/Metal.jl) showed how to build an Apple
  accelerator package on it.
- Apple's [coremltools](https://github.com/apple/coremltools) documents the MIL op
  set and the weight-file format.
- [Orion](https://arxiv.org/abs/2603.06728) catalogs ANE constraints that our
  hardware notes compare against.
- Much of the package (the ObjectiveC.jl port of the runtime, the MIL graph
  builder, the tracer, the tests and the op probe) was written together with
  [Claude](https://claude.com/claude-code), Anthropic's AI assistant.
