# Tracing: run an ordinary Julia function on placeholder arrays and record the
# operations as a MIL graph.
#
# A traced Julia array of size (d1,) or (d1, d2) is held as the MIL tensor
# [1, 1, 1, d1] or [1, d2, 1, d1]: the same memory, in the [1, C, 1, S] layout
# the ANE is fastest with. Julia's broadcasting (aligning leading dimensions)
# then matches MIL's (aligning trailing dimensions). Computation is in fp16.

"""
    Traced

A placeholder for a Julia vector or matrix while a function is traced by
[`compile(f, inputs...)`](@ref). Supports broadcasting with arbitrary scalar
functions built from arithmetic, `exp`, `log`, `sqrt`, `tanh`, `abs`, `max`, `min`,
comparisons and `ifelse`; matrix products and `adjoint`; and `sum`, `maximum`,
`minimum` with `dims`.
"""
struct Traced{N}
    var::MIL.Var
    size::NTuple{N,Int}
end

Base.size(x::Traced) = x.size
Base.size(x::Traced, d::Integer) = d <= ndims(x) ? x.size[d] : 1
Base.ndims(::Traced{N}) where {N} = N
Base.length(x::Traced) = prod(x.size)
Base.axes(x::Traced) = map(Base.OneTo, x.size)
Base.eltype(::Traced) = Float16
Base.show(io::IO, x::Traced) = print(io, "Traced", x.size)

_milshape(dims::Tuple{Int}) = [1, 1, 1, dims[1]]
_milshape(dims::Tuple{Int,Int}) = [1, dims[2], 1, dims[1]]
_milshape(dims::Tuple) = throw(ArgumentError("tracing supports vectors and matrices, got size $dims"))

# Julia array -> constant in the traced layout; one constant per array object.
function _constant(g::MIL.Graph, cache::IdDict, A::AbstractArray)
    get!(cache, A) do
        ndims(A) <= 2 || throw(ArgumentError("tracing supports vector and matrix constants, got size $(size(A))"))
        data = ndims(A) == 1 ? Base.reshape(A, 1, 1, 1, :) : Base.reshape(permutedims(A), 1, size(A, 2), 1, size(A, 1))
        MIL.constant(g, Float32.(data))
    end
end

# Tracing state for the function being traced.
mutable struct _Tracer
    graph::MIL.Graph
    constants::IdDict{Any,MIL.Var}
end

const _TRACER = Ref{Union{Nothing,_Tracer}}(nothing)

function _tracer()
    t = _TRACER[]
    isnothing(t) && error("traced arrays can only be used while tracing")
    return t
end

_lift(x::Traced) = x.var
_lift(x::AbstractArray) = _constant(_tracer().graph, _tracer().constants, x)

# --- broadcasting -------------------------------------------------------------

struct TracedStyle <: Broadcast.BroadcastStyle end
Base.BroadcastStyle(::Type{<:Traced}) = TracedStyle()
Base.BroadcastStyle(s::TracedStyle, ::Broadcast.BroadcastStyle) = s
Base.broadcastable(x::Traced) = x

# The value of one broadcast operand inside a traced scalar function.
struct Elem
    var::MIL.Var
end

Base.Broadcast.materialize(bc::Broadcast.Broadcasted{TracedStyle}) = copy(bc)

function Base.copy(bc::Broadcast.Broadcasted{TracedStyle})
    flat = Broadcast.flatten(bc)
    dims = Broadcast.broadcast_shape(map(_bsize, flat.args)...)
    out = flat.f(map(_elem, flat.args)...)
    out isa Elem || throw(ArgumentError("a traced broadcast must produce an array, got $(typeof(out))"))
    var = out.var
    if var.shape != _milshape(dims)   # e.g. a broadcast of constants only, or of a lower rank
        var = MIL.mul(var, MIL.constant(var.graph, ones(Float32, _milshape(dims)...)))
    end
    return Traced(var, dims)
end

_bsize(x::Union{Traced,AbstractArray}) = size(x)
_bsize(x) = ()
_elem(x::Union{Traced,AbstractArray}) = Elem(_lift(x))
_elem(x::Number) = x
_elem(x::Base.RefValue) = _elem(x[])

_num(x::Number) = Float16(x)
for (f, op) in ((:+, MIL.add), (:-, MIL.sub), (:*, MIL.mul), (:/, MIL.real_div), (:^, MIL.pow),
                (:max, MIL.maximum), (:min, MIL.minimum))
    @eval begin
        Base.$f(a::Elem, b::Elem) = Elem($op(a.var, b.var))
        Base.$f(a::Elem, b::Number) = Elem($op(a.var, _num(b)))
        Base.$f(a::Number, b::Elem) = Elem($op(_num(a), b.var))
    end
end
for (f, op) in ((:<, MIL.less), (:<=, MIL.less_equal), (:>, MIL.greater), (:>=, MIL.greater_equal),
                (:(==), MIL.equal), (:!=, MIL.not_equal))
    @eval begin
        Base.$f(a::Elem, b::Elem) = Elem($op(a.var, b.var))
        Base.$f(a::Elem, b::Number) = Elem($op(a.var, _num(b)))
        Base.$f(a::Number, b::Elem) = Elem($op(_num(a), b.var))
    end
end
for (f, op) in ((:exp, MIL.exp), (:exp2, MIL.exp2), (:sqrt, MIL.sqrt), (:tanh, MIL.tanh),
                (:abs, MIL.abs), (:sin, MIL.sin), (:cos, MIL.cos), (:floor, MIL.floor),
                (:ceil, MIL.ceil), (:round, MIL.round), (:sign, MIL.sign))
    @eval Base.$f(a::Elem) = Elem($op(a.var))
end
Base.log(a::Elem) = Elem(MIL.log(a.var; epsilon=floatmin(Float16)))
Base.inv(a::Elem) = Elem(MIL.inverse(a.var; epsilon=floatmin(Float16)))
Base.:-(a::Elem) = Elem(MIL.mul(a.var, Float16(-1)))
Base.:+(a::Elem) = a
Base.abs2(a::Elem) = Elem(MIL.square(a.var))
Base.literal_pow(::typeof(^), a::Elem, ::Val{2}) = Elem(MIL.square(a.var))
Base.ifelse(c::Elem, a::Union{Elem,Number}, b::Union{Elem,Number}) =
    Elem(MIL.select(c.var, a isa Elem ? a.var : _num(a), b isa Elem ? b.var : _num(b)))

# --- linear algebra -------------------------------------------------------------

_as2d(x) = ndims(x) == 1 ? (size(x, 1), 1) : size(x)

# Julia A * B is MIL(B) · MIL(A) on the 2-D views, since MIL(M) is Mᵀ.
function _matmul(a, b, dims)
    t = _tracer()
    # x * W with a constant matrix W: a 1x1 convolution over x's channels in the
    # [1, C, 1, S] layout, the ANE's fast path.
    if a isa Traced{2} && b isa AbstractMatrix
        W = get!(t.constants, (:conv, objectid(b))) do
            MIL.constant(t.graph, Float32.(Base.reshape(permutedims(b), size(b, 2), size(b, 1), 1, 1)))
        end
        return Traced(MIL.conv(a.var, W), dims)
    end
    va, vb = _lift(a), _lift(b)
    sa, sb = _as2d(a), _as2d(b)
    ma = MIL.reshape(va, [sa[2], sa[1]])
    mb = MIL.reshape(vb, [sb[2], sb[1]])
    prod2 = MIL.matmul(mb, ma)                       # [p, m]
    return Traced(MIL.reshape(prod2, _milshape(dims)), dims)
end

_mulsize(a, b) = ndims(b) == 1 ? (size(a, 1),) : (size(a, 1), size(b, 2))
function _checkmul(a, b)
    (ndims(a) == 1 ? 1 : size(a, 2)) == size(b, 1) && (ndims(a) == 2 || ndims(b) == 2) ||
        throw(DimensionMismatch("cannot multiply sizes $(size(a)) and $(size(b))"))
end
Base.:*(a::Traced, b::Traced) = (_checkmul(a, b); _matmul(a, b, _mulsize(a, b)))
Base.:*(a::Traced, b::AbstractMatrix) = (_checkmul(a, b); _matmul(a, b, _mulsize(a, b)))
Base.:*(a::AbstractMatrix, b::Traced) = (_checkmul(a, b); _matmul(a, b, _mulsize(a, b)))

function Base.adjoint(x::Traced{2})
    v = MIL.transpose(x.var, [0, 3, 2, 1])
    return Traced(v, Base.reverse(x.size))
end
Base.adjoint(x::Traced{1}) = Traced(MIL.reshape(x.var, [1, size(x, 1), 1, 1]), (1, size(x, 1)))
Base.transpose(x::Traced) = adjoint(x)

# --- reductions ---------------------------------------------------------------

# Julia dims -> MIL axes of the traced layout.
_milaxes(x::Traced, dims::Colon) = ndims(x) == 1 ? [3] : [1, 3]
_milaxes(x::Traced, dims) = [d == 1 ? 3 : 1 for d in (dims isa Integer ? (dims,) : dims)]
_reducedsize(x::Traced, dims::Colon) = ntuple(_ -> 1, ndims(x))
_reducedsize(x::Traced, dims) = ntuple(d -> d in dims ? 1 : size(x, d), ndims(x))

for (f, op) in ((:sum, MIL.reduce_sum), (:maximum, MIL.reduce_max), (:minimum, MIL.reduce_min))
    @eval begin
        Base.$f(x::Traced; dims=:) = Traced($op(x.var; axes=_milaxes(x, dims), keep_dims=true), _reducedsize(x, dims))
        Base.$f(g::Function, x::Traced; dims=:) = $f(g.(x); dims)
    end
end

# --- compile --------------------------------------------------------------------

"""
    trace(f, inputs...) -> MIL.Graph

Record `f` applied to placeholders for `inputs` (example arrays, or sizes
`(n,)` / `(m, n)`, which mean `Float16` inputs) as a MIL graph. `f` returns an
array or a tuple of arrays.
"""
function trace(f, inputs...)
    g = MIL.Graph()
    args = map(enumerate(inputs)) do (i, spec)
        T, dims = spec isa AbstractArray ? (eltype(spec), size(spec)) : (Float16, Tuple(spec))
        T in (Float16, Float32) || throw(ArgumentError("inputs must be Float16 or Float32 arrays, got $T"))
        v = MIL.input(g, "x$i", T, _milshape(dims); julia_shape=dims)
        Traced(T == Float16 ? v : MIL.cast(v, Float16), dims)
    end
    outtype = isempty(inputs) || !(first(inputs) isa AbstractArray) ? Float16 : eltype(first(inputs))
    previous = _TRACER[]
    _TRACER[] = _Tracer(g, IdDict{Any,MIL.Var}())
    try
        result = f(args...)
        for (i, r) in enumerate(result isa Tuple ? result : (result,))
            r isa Traced || throw(ArgumentError("the traced function must return traced arrays, got $(typeof(r))"))
            # A program output must be produced by an op.
            v = r.var in g.inputs ? MIL.mul(r.var, Float16(1)) : r.var
            MIL.output!(g, outtype == Float16 ? v : MIL.cast(v, outtype), "y$i"; julia_shape=r.size)
        end
    finally
        _TRACER[] = previous
    end
    return g
end

"""
    compile(f, inputs...) -> ANEProgram

Trace `f` (see [`trace`](@ref)) and compile it for the Neural Engine. The
program is callable: `program(x...)` returns the output (or a tuple of outputs).

```julia
W = randn(Float32, 512, 2048)
program = compile(x -> max.(x * W, 0), randn(Float16, 256, 512))
y = program(x)
```
"""
compile(f::Function, inputs...) = compile(trace(f, inputs...))

function (program::ANEProgram)(inputs...)
    outputs = evaluate!(program, inputs...)
    return length(outputs) == 1 ? only(outputs) : Tuple(outputs)
end
