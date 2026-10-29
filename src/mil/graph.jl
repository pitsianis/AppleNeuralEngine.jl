# A MIL program under construction: typed values (`Var`) produced by ops,
# printed as the MIL text the ANE compiler reads.

const _MIL_TYPES = Dict{DataType,String}(
    Float16 => "fp16", Float32 => "fp32", Int32 => "int32", Int16 => "int16", Int8 => "int8",
    UInt8 => "uint8", UInt16 => "uint16", Bool => "bool", String => "string")

"""
    Graph(; opset="ios18")

A MIL program being built. Create inputs with [`input`](@ref), apply ops to the
resulting [`Var`](@ref)s, mark results with [`output!`](@ref), then pass the
graph to [`compile`](@ref AppleNeuralEngine.compile) or print it with [`source`](@ref).
"""
mutable struct Graph
    opset::String
    inputs::Vector{Any}
    outputs::Vector{Any}
    body::Vector{String}
    weights::WeightFile
    names::Set{String}
    renames::Dict{String,String}
    julia_shapes::Dict{String,Tuple}
end

Graph(; opset::AbstractString="ios18") =
    Graph(String(opset), [], [], String[], WeightFile(), Set{String}(), Dict{String,String}(), Dict{String,Tuple}())

"""
    Var

A value in a [`Graph`](@ref): a tensor with a MIL element type and a MIL shape
(row-major, outermost dimension first; `()` for a scalar). Vars are produced by
[`input`](@ref), [`constant`](@ref) and the op functions.
"""
struct Var
    graph::Graph
    name::String
    eltype::DataType
    shape::Vector{Int}
end

Base.eltype(v::Var) = v.eltype
Base.size(v::Var) = Tuple(v.shape)
Base.size(v::Var, d::Integer) = v.shape[d]
Base.ndims(v::Var) = length(v.shape)
Base.length(v::Var) = prod(v.shape; init=1)
Base.show(io::IO, v::Var) = print(io, "Var(", v.name, "::", _miltype(v.eltype, v.shape), ")")

_miltype(T, shape) = "tensor<$(_MIL_TYPES[T]), [$(join(shape, ", "))]>"
_miltype(v::Var) = _miltype(v.eltype, v.shape)

# Words of the MIL text format that cannot be variable names.
const _RESERVED = Set(["const", "func", "program", "tensor", "string", "bool", "list", "dict", "state",
                       "fp16", "fp32", "fp64", "bf16", "int4", "int8", "int16", "int32", "int64",
                       "uint1", "uint2", "uint4", "uint6", "uint8", "uint16", "uint32", "uint64"])

function _fresh!(g::Graph, hint)
    base = replace(string(hint), r"[^A-Za-z0-9_]" => "_")
    (isempty(base) || isdigit(base[1])) && (base = "v_" * base)
    base in _RESERVED && (base *= "_v")
    name, n = base, 0
    while name in g.names
        name = "$(base)_$(n += 1)"
    end
    push!(g.names, name)
    return name
end

function _same_graph(g::Graph, v::Var)
    v.graph === g || throw(ArgumentError("$v belongs to a different Graph"))
    return v
end

# --- literals ---------------------------------------------------------------

function _float(x)
    isfinite(x) || throw(ArgumentError("MIL literals must be finite, got $x"))
    return string(Float64(x))
end

_lit(v::Var) = v.name
_lit(x::Bool) = "bool($x)"
_lit(x::Integer) = "int32($x)"
_lit(x::Float16) = "fp16($(_float(x)))"
_lit(x::Float32) = "fp32($(_float(x)))"
_lit(x::Float64) = throw(ArgumentError("give MIL float literals as Float16 or Float32, got the Float64 $x"))
_lit(x::Union{AbstractString,Symbol}) = "string(\"$x\")"
_lit(x::DataType) = _lit(_MIL_TYPES[x])
_lit(x::Tuple) = _lit(collect(x))

function _lit(xs::AbstractVector)
    if !isempty(xs) && all(x -> x isa Var, xs)
        return "(" * join((x.name for x in xs), ", ") * ")"
    elseif all(x -> x isa Bool, xs)
        return "tensor<bool, [$(length(xs))]>([$(join(xs, ", "))])"
    elseif all(x -> x isa Integer, xs)
        return "tensor<int32, [$(length(xs))]>([$(join(xs, ", "))])"
    elseif all(x -> x isa Union{Float16,Float32}, xs) && allequal(typeof.(xs))
        return "tensor<$(_MIL_TYPES[eltype(xs)]), [$(length(xs))]>([$(join(map(_float, xs), ", "))])"
    end
    throw(ArgumentError("cannot write $(typeof(xs)) as a MIL literal"))
end

# --- graph construction -----------------------------------------------------

# MIL type of a literal argument, for hoisting it into a `const` op.
_littype(::Bool) = "bool"
_littype(::Integer) = "int32"
_littype(::Float16) = "fp16"
_littype(::Float32) = "fp32"
_littype(::Union{AbstractString,Symbol,DataType}) = "string"
function _littype(xs::Union{Tuple,AbstractVector})
    lit = _lit(xs)
    return startswith(lit, "tensor<") ? first(Base.split(lit, ">(")) * ">" : nothing
end

# Literal arguments become named `const` ops, as in Core ML's own programs: the
# ANE compiler's validation rejects some inline literals.
function _hoist!(g::Graph, opname, key, value)
    (value isa Var || (value isa AbstractVector && !isempty(value) && all(x -> x isa Var, value))) && return value
    T = _littype(value)
    isnothing(T) && return value
    name = _fresh!(g, "$(opname)_$(key)")
    push!(g.body, "$T $name = const()[name = string(\"$name\"), val = $(_lit(value))];")
    return Var(g, name, Nothing, Int[])
end

"""
    op!(g, op, args, outputs; name=op) -> Var or Tuple{Var...}

Append the MIL operation `op` to `g`. `args` are `name => value` pairs (values
are `Var`s or literals; `nothing` omits an optional argument) and `outputs` are
`(eltype, shape)` pairs, one per result. This is the escape hatch for ops that
have no wrapper; the wrappers in this module call it after inferring shapes.
"""
function op!(g::Graph, op::AbstractString, args, outputs; name=op)
    args = [k => v for (k, v) in args if !isnothing(v)]
    args = [k => _hoist!(g, name, k, v) for (k, v) in args]
    foreach(((_, v),) -> v isa Var && _same_graph(g, v), args)
    foreach(((_, v),) -> v isa AbstractVector && foreach(x -> x isa Var && _same_graph(g, x), v), args)
    results = [Var(g, _fresh!(g, length(outputs) == 1 ? name : "$(name)_$(i - 1)"), T, collect(Int, shape))
               for (i, (T, shape)) in enumerate(outputs)]
    lhs = join(("$(_miltype(v)) $(v.name)" for v in results), ", ")
    rhs = join(("$k = $(_lit(v))" for (k, v) in args), ", ")
    push!(g.body, "$lhs = $op($rhs)[name = string(\"$(results[1].name)\")];")
    return length(results) == 1 ? only(results) : Tuple(results)
end

"""
    input(g, name, T, shape; julia_shape=nothing) -> Var

Declare a program input of element type `T` (`Float32` or `Float16`) and MIL
shape `shape`. Inputs are passed to [`evaluate!`](@ref AppleNeuralEngine.evaluate!) in declaration order, as
Julia arrays of size `julia_shape`; by default that is `shape` reversed with
singleton dimensions dropped, the array with the same memory layout.
"""
function input(g::Graph, name, T::DataType, shape; julia_shape=nothing)
    T in (Float32, Float16) || throw(ArgumentError("ANE program inputs must be Float32 or Float16, got $T"))
    isempty(g.body) || throw(ArgumentError("declare all inputs before adding ops"))
    # The runtime binds inputs by sorted name, so the index prefix fixes the order.
    v = Var(g, _fresh!(g, "in$(lpad(length(g.inputs), 3, '0'))_$name"), T, collect(Int, shape))
    _set_julia_shape!(g, v, julia_shape)
    push!(g.inputs, v)
    return v
end

"""
    output!(g, v, name=v.name; julia_shape=nothing)

Mark `v` as a program output. Outputs are returned by [`evaluate!`](@ref AppleNeuralEngine.evaluate!) in the
order they are marked, as Julia arrays of size `julia_shape` (see [`input`](@ref)).
They must be `Float32` or `Float16`.
"""
function output!(g::Graph, v::Var, name=v.name; julia_shape=nothing)
    _same_graph(g, v)
    v.eltype in (Float32, Float16) || throw(ArgumentError("ANE program outputs must be Float32 or Float16; cast $v first"))
    any(i -> i.name == v.name, g.inputs) && throw(ArgumentError("an input cannot be an output; apply an op to it first"))
    haskey(g.renames, v.name) && throw(ArgumentError("$v is already an output"))
    # Outputs are bound by sorted name too, so they get index-prefixed names.
    # The producing variable is renamed when the program is printed: the ANE
    # compiler rejects a trailing `identity` op as an output.
    new = _fresh!(g, "out$(lpad(length(g.outputs), 3, '0'))_$name")
    g.renames[v.name] = new
    o = Var(g, new, v.eltype, v.shape)
    _set_julia_shape!(g, o, julia_shape)
    push!(g.outputs, o)
    return o
end

"""
    constant(g, value; eltype=Float16) -> Var

A constant tensor. Arrays are stored in the program's weight file, converted
to `eltype` and indexed the MIL way (`A[i, j]` is MIL element `[i, j]`); see
[`WeightFile`](@ref). Numbers become scalar constants.
"""
function constant(g::Graph, value::AbstractArray; eltype::DataType=Float16, name="const")
    data = convert(Array{eltype}, value)
    offset = push!(g.weights, data)
    shape = collect(size(value))
    blob = "BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64($offset))"
    v = Var(g, _fresh!(g, name), eltype, shape)
    push!(g.body, "$(_miltype(v)) $(v.name) = const()[name = string(\"$(v.name)\"), val = $(_miltype(v))($blob)];")
    return v
end

function constant(g::Graph, value::Number; eltype::DataType=Float16, name="const")
    v = Var(g, _fresh!(g, name), eltype, Int[])
    push!(g.body, "$(_miltype(v)) $(v.name) = const()[name = string(\"$(v.name)\"), val = $(_lit(convert(eltype, value)))];")
    return v
end

const _BUILD_INFO = """[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]"""

"""
    source(g) -> String

The MIL text of `g`, as passed to the ANE compiler.
"""
function source(g::Graph)
    isempty(g.inputs) && throw(ArgumentError("the program has no inputs"))
    isempty(g.outputs) && throw(ArgumentError("the program has no outputs; mark them with output!"))
    args = join(("$(_miltype(v)) $(v.name)" for v in g.inputs), ", ")
    io = IOBuffer()
    println(io, "program(1.3)")
    println(io, _BUILD_INFO)
    println(io, "{")
    println(io, "    func main<$(g.opset)>($args) {")
    rename = isempty(g.renames) ? nothing :
        # Variables are never followed by "(", op names always are.
        Regex("\\b(" * join(keys(g.renames), "|") * ")\\b(?!\\()") => (m -> g.renames[m])
    for line in g.body
        println(io, "        ", isnothing(rename) ? line : replace(line, rename))
    end
    println(io, "    } -> (", join((v.name for v in g.outputs), ", "), ");")
    println(io, "}")
    return String(take!(io))
end

"""
    weight_blob(g) -> Union{Vector{UInt8},Nothing}

The weight file holding `g`'s constants, or `nothing` if it has none.
"""
weight_blob(g::Graph) = g.weights.count == 0 ? nothing : bytes(g.weights)

# Julia arrays for program inputs/outputs: the same memory as the MIL tensor,
# i.e. its dimensions reversed, with singleton dimensions dropped (so a MIL
# [1, C, 1, S] activation is a Julia (S, C) matrix).
function _set_julia_shape!(g, v, dims)
    isnothing(dims) && return
    prod(dims; init=1) == length(v) || throw(DimensionMismatch("julia_shape $dims does not have the $(length(v)) elements of $v"))
    g.julia_shapes[v.name] = Tuple(dims)
end

function julia_shape(v::Var)
    haskey(v.graph.julia_shapes, v.name) && return v.graph.julia_shapes[v.name]
    dims = Tuple(d for d in Base.reverse(v.shape) if d != 1)
    return isempty(dims) ? (1,) : dims
end

# (row bytes, rows) of the buffer holding `v`: rows run along the innermost MIL dimension.
buffer_layout(v::Var) = (last(v.shape) * sizeof(v.eltype), prod(v.shape[1:end-1]; init=1))
