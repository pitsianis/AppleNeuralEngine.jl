"""
    ANEProgram

A MIL program compiled and loaded onto the Neural Engine, with its input and
output buffers. Create one with [`compile`](@ref); run it with
[`evaluate!`](@ref) or by calling it, `program(inputs...)`. A program holds
Neural Engine resources until [`close!`](@ref) is called or it is garbage
collected.
"""
mutable struct ANEProgram
    model::id{_ANEInMemoryModel}
    request::id{_ANERequest}
    input_surfaces::Vector{Ptr{Cvoid}}
    output_surfaces::Vector{Ptr{Cvoid}}
    model_dir::String
    input_shapes::Vector{Vector{Int}}
    output_shapes::Vector{Vector{Int}}
    input_types::Vector{DataType}
    output_types::Vector{DataType}
    input_layouts::Vector{Tuple{Int,Int}}
    output_layouts::Vector{Tuple{Int,Int}}
end

"""
    isopen(program::ANEProgram) -> Bool

Whether `program` is still loaded (not yet [`close!`](@ref)d).
"""
isopen(program::ANEProgram) = program.model != nil

"""
    close!(program::ANEProgram)

Unload `program` from the Neural Engine and free its buffers. Safe to call more
than once; afterwards the program cannot be evaluated.
"""
function close!(program::ANEProgram)
    if isopen(program)
        _unload_native(program.model)
        foreach(_free_surface, program.input_surfaces)
        foreach(_free_surface, program.output_surfaces)
        empty!(program.input_surfaces)
        empty!(program.output_surfaces)
        _release(program.request)
        _release(program.model)
        program.request = nil
        program.model = nil
        rm(program.model_dir; recursive=true, force=true)
    end
    return nothing
end

function Base.show(io::IO, program::ANEProgram)
    fmt(shapes, types) = join(("$T(" * join(s, ", ") * ")" for (s, T) in zip(shapes, types)), ", ")
    print(io, "ANEProgram(inputs=[", fmt(program.input_shapes, program.input_types), "], outputs=[",
          fmt(program.output_shapes, program.output_types), "]", isopen(program) ? "" : ", closed", ")")
end

function _shapes(shapes, label)
    result = [Int.(collect(shape)) for shape in shapes]
    isempty(result) && throw(ArgumentError("$label must contain at least one tensor shape"))
    for shape in result
        isempty(shape) && throw(ArgumentError("tensor shapes cannot be empty"))
        all(>(0), shape) || throw(ArgumentError("tensor dimensions must be positive"))
        try
            foldl(Base.checked_mul, shape; init=sizeof(Float32))
        catch err
            err isa OverflowError || rethrow()
            throw(ArgumentError("tensor is too large"))
        end
    end
    return result
end

# (row bytes, rows) of a tensor buffer: rows run along the first (innermost)
# Julia dimension.
_layout(shape, T=Float32) = (shape[1] * sizeof(T), prod(shape[2:end]; init=1))

function _types(types, n, label)
    result = types isa DataType ? fill(types, n) : collect(DataType, types)
    length(result) == n || throw(ArgumentError("$label must have one type per tensor"))
    all(in((Float32, Float16)), result) || throw(ArgumentError("$label must be Float32 or Float16"))
    return result
end

"""
    compile(mil; input_shapes, output_shapes, weight_blob=nothing,
            input_types=Float32, output_types=Float32)

Compile one MIL program with the private ANE runtime and load it onto the
Neural Engine. Inputs and outputs are contiguous buffers of `Float32` or
`Float16` (one type for all, or one per tensor, matching the MIL program); each
`shape` gives the size of the Julia array that is copied in or out, and its
memory layout must match the tensor in the MIL program.

`weight_blob` is the program's weight file, `@model_path/weights/weight.bin`,
usually made with [`build_weight_blob`](@ref) or, for several tensors,
[`pack_weights`](@ref).

The runtime binds MIL inputs and outputs to buffer indices in sorted name
order, not declaration order.
"""
function compile(mil::AbstractString; input_shapes, output_shapes,
                 weight_blob::Union{Nothing,AbstractVector{UInt8}}=nothing,
                 input_types=Float32, output_types=Float32, _layouts=nothing)
    initialize!()
    isempty(mil) && throw(ArgumentError("MIL source cannot be empty"))
    inputs = _shapes(input_shapes, "input_shapes")
    outputs = _shapes(output_shapes, "output_shapes")
    intypes = _types(input_types, length(inputs), "input_types")
    outtypes = _types(output_types, length(outputs), "output_types")
    # The runtime also accepts several weight files, but whether it then
    # verifies the model depends on the order of its weights dictionary, so
    # multiple tensors go into one file instead (see `pack_weights`).
    weights = isnothing(weight_blob) || isempty(weight_blob) ? Pair{String,Vector{UInt8}}[] :
        ["@model_path/weights/weight.bin" => collect(UInt8, weight_blob)]

    inlayouts, outlayouts = something(_layouts, (map(_layout, inputs, intypes), map(_layout, outputs, outtypes)))
    model, request, insurf, outsurf, dir = _compile_native(
        Vector{UInt8}(codeunits(mil)), weights, inlayouts, outlayouts)
    program = ANEProgram(model, request, insurf, outsurf, dir, inputs, outputs, intypes, outtypes, inlayouts, outlayouts)
    finalizer(close!, program)
    return program
end

"""
    evaluate!(program, inputs...) -> Vector{Array}

Copy `inputs` into the program's input buffers, run it on the Neural Engine
and return freshly allocated output arrays.
"""
function evaluate!(program::ANEProgram, inputs...)
    isopen(program) || error("ANEProgram is closed")
    length(inputs) == length(program.input_shapes) ||
        throw(DimensionMismatch("expected $(length(program.input_shapes)) inputs, got $(length(inputs))"))

    for (i, (input, shape, T)) in enumerate(zip(inputs, program.input_shapes, program.input_types))
        input isa Array{T} || throw(ArgumentError("input $i must be a dense Array{$T}"))
        size(input) == Tuple(shape) || throw(DimensionMismatch("input $i has size $(size(input)); expected $(Tuple(shape))"))
    end
    for (input, surface, layout) in zip(inputs, program.input_surfaces, program.input_layouts)
        GC.@preserve input _copy_surface!(surface, pointer(input), layout..., true)
    end

    _evaluate_native(program.model, program.request)

    return map(program.output_shapes, program.output_types, program.output_surfaces, program.output_layouts) do shape, T, surface, layout
        output = Array{T}(undef, Tuple(shape))
        GC.@preserve output _copy_surface!(surface, pointer(output), layout..., false)
        output
    end
end

"""
    compile(g::MIL.Graph) -> ANEProgram

Compile a program built with the [`MIL`](@ref) graph builder. Its inputs and
outputs are Julia arrays with the MIL dimensions reversed and singleton
dimensions dropped: a MIL `[1, C, 1, S]` tensor is a Julia `(S, C)` matrix.
"""
compile(g::MIL.Graph) = compile(MIL.source(g);
    input_shapes=map(MIL.julia_shape, g.inputs), output_shapes=map(MIL.julia_shape, g.outputs),
    input_types=[v.eltype for v in g.inputs], output_types=[v.eltype for v in g.outputs],
    weight_blob=MIL.weight_blob(g),
    # Buffer rows follow the MIL innermost dimension, which the Julia shape may
    # have dropped if it is 1.
    _layouts=(map(MIL.buffer_layout, g.inputs), map(MIL.buffer_layout, g.outputs)))
