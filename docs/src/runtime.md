# Raw MIL and the runtime

At the lowest level, [`compile`](@ref) takes MIL program text, the shapes of
its input and output buffers, and an optional weight file:

```julia
using AppleNeuralEngine

program = compile(mil_source;
    input_shapes=[(64, 256)], output_shapes=[(64, 256)],
    input_types=Float32, output_types=Float32,
    weight_blob=build_weight_blob(W))

output, = evaluate!(program, input)
close!(program)
```

`examples/minimal.jl` is a complete program in this style.

## Buffers

Each input and output is a dense Julia array whose memory matches the MIL tensor.
For the usual `[1, C, 1, S]` activation layout that is an `(S, C)` matrix. The
first Julia dimension is the innermost MIL dimension. `evaluate!` copies inputs
into IOSurfaces shared with the Neural Engine, runs the program, and copies the
outputs back.

## Weights

All constants live in one file, `@model_path/weights/weight.bin`, in Core ML's
weight-file format. A MIL constant refers to a tensor by its offset:

```text
tensor<fp16, [64, 64, 1, 1]> W = const()[name = string("W"),
    val = tensor<fp16, [64, 64, 1, 1]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(64)))];
```

- [`build_weight_blob`](@ref) makes a file holding one fp16 matrix (offset 64).
- [`pack_weights`](@ref) packs several matrices and returns their offsets.
- [`WeightFile`](@ref) appends arrays of any rank and element type one at a time.

The runtime can also take weights split across several files, but whether it
accepts them then depends on the order of an internal dictionary. This
package always uses one file.

## Input and output order

The runtime binds a program's inputs and outputs to buffers in **sorted name
order**, not in the order they are declared. For `func main(b, a) -> (q, p)` the
first buffer is `a` and the first output is `p`. The graph builder names
inputs and outputs so that both orders agree.

## Errors

Compilation and evaluation failures raise an [`ANEError`](@ref) carrying the
runtime's own message, for example `InvalidMILProgram` when the ANE compiler's
MIL validation rejects the program.
