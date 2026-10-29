# API reference

## Compiling and running

```@docs
compile
trace
ANEProgram
evaluate!
close!
isopen
initialize!
functional
device_info
ANEError
```

## Weights

```@docs
WeightFile
pack_weights
build_weight_blob
AppleNeuralEngine.bytes
```

## MIL graph builder

```@docs
AppleNeuralEngine.MIL
AppleNeuralEngine.MIL.Graph
AppleNeuralEngine.MIL.Var
AppleNeuralEngine.MIL.input
AppleNeuralEngine.MIL.output!
AppleNeuralEngine.MIL.constant
AppleNeuralEngine.MIL.constant_int8
AppleNeuralEngine.MIL.op!
AppleNeuralEngine.MIL.source
AppleNeuralEngine.MIL.weight_blob
```

### Ops

```@autodocs
Modules = [AppleNeuralEngine.MIL]
Pages = ["ops.jl"]
```

## Tracing

```@docs
AppleNeuralEngine.Traced
```
