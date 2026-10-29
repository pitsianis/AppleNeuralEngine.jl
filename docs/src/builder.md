# The MIL graph builder

MIL ("Model Intermediate Language") is the program format the ANE compiler
reads, the same one Core ML uses. `AppleNeuralEngine.MIL` builds MIL programs
from Julia. Every op is a function on [`MIL.Var`](@ref)s that infers its output
type and shape, so the generated program is always well-typed. Names, constants,
weight files and input/output ordering are handled for you.

```julia
using AppleNeuralEngine
const M = AppleNeuralEngine.MIL

g = M.Graph()
x = M.input(g, :x, Float16, (1, 64, 1, 32))        # MIL shape [N, C, H, W]
W = M.constant(g, randn(Float32, 128, 64, 1, 1))    # stored as fp16 weights
h = M.relu(M.conv(x, W))
M.output!(g, M.reduce_mean(h; axes=[1], keep_dims=true), :y)

program = compile(g)
y = program(randn(Float16, 32, 64))                 # Julia sizes: see below
print(M.source(g))                                  # the generated MIL text
```

## Conventions

- **Shapes and axes are MIL's:** row-major, outermost dimension first, 0-based
  axes, negative axes count from the end. Op and argument names follow the
  [coremltools MIL op reference](https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html).
  Several names shadow `Base` (`exp`, `reshape`, `split`, ...), so call them
  qualified: `M.exp(x)`.
- **Constants are indexed the MIL way:** `A[i, j, k]` is MIL element `[i, j, k]`.
  A conv weight `[Cout, Cin, kh, kw]` is the Julia array `W[cout, cin, kh, kw]`.
  `M.constant_int8` stores int8 weights with per-channel scales.
- **Program inputs and outputs** are Julia arrays with the *same memory* as the
  MIL tensor: the dimensions reversed and singleton dimensions dropped. A MIL
  `[1, C, 1, S]` tensor is a Julia `(S, C)` matrix. Pass `julia_shape=` to
  [`MIL.input`](@ref) / [`MIL.output!`](@ref) to choose another shape with the
  same number of elements.
- **Compute in fp16.** Inputs and outputs may be `Float32` or `Float16`, but ops
  on fp32 tensors do not compile: `M.cast` inputs to `Float16` first. `Float16`
  inputs and outputs also halve the data copied per call.

## Escape hatch

Ops without a wrapper can be emitted with [`MIL.op!`](@ref), giving the output
types yourself:

```julia
y = M.op!(g, "softsign", (:x => x,), ((Float16, size(x)),))
```

## Building on a trace

[`trace`](@ref) returns the `MIL.Graph` of a Julia function, so a traced model
can be inspected (`M.source`) or extended before [`compile`](@ref).
