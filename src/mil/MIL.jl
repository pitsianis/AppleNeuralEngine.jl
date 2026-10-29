"""
    MIL

A builder for MIL programs, the input language of the ANE compiler. Ops are
Julia functions on [`Var`](@ref)s that infer output types and shapes, so the
generated program is always well-typed:

```julia
using AppleNeuralEngine, AppleNeuralEngine.MIL
g = MIL.Graph()
x = MIL.input(g, :x, Float16, (1, 64, 1, 32))
W = MIL.constant(g, randn(Float32, 64, 64, 1, 1))
MIL.output!(g, MIL.relu(MIL.conv(x, W)))
program = compile(g)
```

Function names and arguments follow the MIL op set as documented by
coremltools; several shadow Base names (`exp`, `reshape`, ...), so call them
qualified (`MIL.exp`) rather than `using` the module.
"""
module MIL

import ..AppleNeuralEngine: WeightFile, bytes

include("graph.jl")
include("ops.jl")

end # module MIL
