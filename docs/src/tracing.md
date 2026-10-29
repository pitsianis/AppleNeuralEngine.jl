# Tracing Julia functions

`compile(f, inputs...)` runs `f` once on placeholder arrays, records every
operation as a MIL graph, and compiles the graph for the Neural Engine. The same
function keeps working on ordinary arrays, so you can check the ANE result
against the CPU directly.

```julia
using AppleNeuralEngine

const EPS = 1f-5
silu(z) = z / (1 + exp(-z))

# A LLaMA-style feed-forward block: RMSNorm, SwiGLU, residual.
function ffn(x)
    n = x ./ sqrt.(sum(abs2, x; dims=2) ./ size(x, 2) .+ EPS) .* g'
    return x .+ (silu.(n * W1') .* (n * W3')) * W2'
end

x = randn(Float16, 256, 512)
program = compile(ffn, x)        # inputs: example arrays (or sizes)
y = program(x)                   # on the ANE
y_cpu = ffn(Float32.(x))         # the same function on the CPU
```

## What is supported

Inputs are vectors or matrices of `Float16` or `Float32`. The output has the
same element type as the first input. Inside the trace everything is computed
in fp16.

- **Broadcasting** with any scalar function built from `+ - * / ^`, `exp`,
  `exp2`, `log`, `sqrt`, `inv`, `abs`, `abs2`, `tanh`, `max`, `min`, `floor`,
  `ceil`, `round`, `sign`, comparisons and `ifelse`. Whole expressions such as
  `silu.(x)` or `ifelse.(x .> 0, x, 0.1f0 .* x)` are traced through the scalar
  function.
- **Matrix products** `A * B` between traced arrays and constant matrices,
  `adjoint`/`transpose`, and matrix-vector products.
- **Reductions** `sum`, `maximum`, `minimum`, with or without `dims`, and
  `sum(f, x; dims)`.
- **Constants:** any Julia array captured by the function (weights, biases) is
  stored in the program's weight file. An array used several times is stored
  once.
- **Several outputs:** return a tuple.

`x * W` with a constant `W` becomes a 1×1 convolution, the Neural Engine's
fastest operation, so write layers in that orientation: rows are tokens or
samples, columns are features.

## Layout

A traced `(T, D)` matrix is held as the MIL tensor `[1, D, 1, T]`. That is the
same memory as the Julia matrix and the layout the ANE is fastest with, so
inputs and outputs need no transposition.

## Limitations

- Only vectors and matrices; no indexing, `getindex`, `reshape` or mutation.
- `sin`, `cos` and `tan` are rejected by the ANE compiler (see [Op support](@ref)).
- Control flow must not depend on array values. The trace records one path.
- For anything else, drop to [The MIL graph builder](@ref): [`trace`](@ref)
  returns the graph, which you can extend before compiling.
