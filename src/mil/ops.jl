# Wrappers for the MIL op set (coremltools `mil/ops/defs`, iOS15–iOS18). Each
# infers its output types and calls `op!`. Argument names follow MIL; shapes and
# axes are MIL's (row-major, 0-based axes, negative axes count from the end).

# --- helpers ----------------------------------------------------------------

_graph(x::Var) = x.graph
_graph(xs::Union{Tuple,AbstractVector}) = first(v for v in xs if v isa Var).graph

_axis(a::Integer, rank) = (-rank <= a < rank) ? Base.mod(a, rank) : throw(ArgumentError("axis $a is out of range for rank $rank"))
_axes(as, rank) = sort!([_axis(a, rank) for a in (as isa Integer ? (as,) : as)])

# Scalar arguments take the element type of the tensor they apply to.
_like(x::Var, a::Var) = a
_like(x::Var, a::Number) = x.eltype <: AbstractFloat ? convert(x.eltype, a) : a
_like(x::Var, a::AbstractVector{<:Number}) = [_like(x, ai) for ai in a]
_like(x::Var, ::Nothing) = nothing

function _broadcast(shapes...)
    rank = Base.maximum(length, shapes)
    out = ones(Int, rank)
    for s in shapes, (i, d) in enumerate(Base.reverse(s))
        j = rank - i + 1
        if out[j] == 1
            out[j] = d
        elseif d != 1 && d != out[j]
            throw(DimensionMismatch("shapes $(join(shapes, ", ")) do not broadcast"))
        end
    end
    return out
end

_shape(x::Var) = x.shape
_shape(::Number) = Int[]

# --- elementwise unary --------------------------------------------------------

for op in (:abs, :acos, :asin, :atan, :atanh, :ceil, :cos, :cosh, :erf, :exp, :exp2,
           :floor, :round, :sign, :sin, :sinh, :sqrt, :square, :tan, :tanh,
           :relu, :relu6, :sigmoid, :silu, :softplus, :softsign)
    @eval function $op(x::Var; name=$(string(op)))
        op!(x.graph, $(string(op)), (:x => x,), ((x.eltype, x.shape),); name)
    end
end

"""
    gelu(x; mode="EXACT")

`mode` is `"EXACT"`, `"TANH_APPROXIMATION"` or `"SIGMOID_APPROXIMATION"`.
"""
gelu(x::Var; mode="EXACT", name="gelu") =
    op!(x.graph, "gelu", (:x => x, :mode => mode), ((x.eltype, x.shape),); name)

for (op, params) in ((:inverse, (:epsilon,)), (:log, (:epsilon,)), (:rsqrt, (:epsilon,)),
                     (:threshold, (:alpha,)), (:clip, (:alpha, :beta)),
                     (:elu, (:alpha,)), (:leaky_relu, (:alpha,)), (:thresholded_relu, (:alpha,)),
                     (:clamped_relu, (:alpha, :beta)), (:linear_activation, (:alpha, :beta)),
                     (:scaled_tanh, (:alpha, :beta)), (:sigmoid_hard, (:alpha, :beta)),
                     (:prelu, (:alpha,)), (:softplus_parametric, (:alpha, :beta)))
    kws = [Expr(:kw, p, :nothing) for p in params]
    args = [:($(QuoteNode(p)) => _like(x, $p)) for p in params]
    @eval function $op(x::Var; $(kws...), name=$(string(op)))
        op!(x.graph, $(string(op)), (:x => x, $(args...)), ((x.eltype, x.shape),); name)
    end
end

logical_not(x::Var; name="logical_not") = op!(x.graph, "logical_not", (:x => x,), ((Bool, x.shape),); name)

"""
    cast(x, T)

Convert `x` to element type `T` (`Float16`, `Float32`, `Int32`, `Int16`, `UInt8`, `Bool`, ...).
"""
cast(x::Var, T::DataType; name="cast") = op!(x.graph, "cast", (:x => x, :dtype => T), ((T, x.shape),); name)

identity(x::Var; name="identity") = op!(x.graph, "identity", (:x => x,), ((x.eltype, x.shape),); name)

"""
    softmax(x; axis=-1)
"""
softmax(x::Var; axis=-1, name="softmax") =
    op!(x.graph, "softmax", (:x => x, :axis => axis), ((x.eltype, x.shape),); name)

# --- elementwise binary -----------------------------------------------------

for op in (:add, :sub, :mul, :real_div, :pow, :maximum, :minimum, :floor_div, :mod)
    @eval function $op(x::Union{Var,Number}, y::Union{Var,Number}; name=$(string(op)))
        ref = x isa Var ? x : y
        op!(ref.graph, $(string(op)), (:x => _like(ref, x), :y => _like(ref, y)),
            ((ref.eltype, _broadcast(_shape(x), _shape(y))),); name)
    end
end

for op in (:equal, :not_equal, :greater, :greater_equal, :less, :less_equal,
           :logical_and, :logical_or, :logical_xor)
    @eval function $op(x::Union{Var,Number}, y::Union{Var,Number}; name=$(string(op)))
        ref = x isa Var ? x : y
        op!(ref.graph, $(string(op)), (:x => _like(ref, x), :y => _like(ref, y)),
            ((Bool, _broadcast(_shape(x), _shape(y))),); name)
    end
end

"""
    select(cond, a, b)

Elementwise `cond ? a : b` with broadcasting.
"""
function select(cond::Var, a::Union{Var,Number}, b::Union{Var,Number}; name="select")
    ref = a isa Var ? a : b
    op!(cond.graph, "select", (:cond => cond, :a => _like(ref, a), :b => _like(ref, b)),
        ((ref.eltype, _broadcast(cond.shape, _shape(a), _shape(b))),); name)
end

# --- reductions -------------------------------------------------------------

function _reduced(shape, axes, keep_dims)
    axes = isnothing(axes) ? collect(0:length(shape)-1) : _axes(axes, length(shape))
    return keep_dims ? [i - 1 in axes ? 1 : d for (i, d) in enumerate(shape)] :
                       [d for (i, d) in enumerate(shape) if !(i - 1 in axes)]
end

for op in (:reduce_sum, :reduce_mean, :reduce_max, :reduce_min, :reduce_prod,
           :reduce_l1_norm, :reduce_l2_norm, :reduce_log_sum, :reduce_log_sum_exp, :reduce_sum_square)
    @eval function $op(x::Var; axes=nothing, keep_dims=false, name=$(string(op)))
        op!(x.graph, $(string(op)), (:x => x, :axes => axes, :keep_dims => keep_dims),
            ((x.eltype, _reduced(x.shape, axes, keep_dims)),); name)
    end
end

for op in (:reduce_argmax, :reduce_argmin)
    @eval function $op(x::Var; axis=-1, keep_dims=false, name=$(string(op)))
        op!(x.graph, $(string(op)), (:x => x, :axis => axis, :keep_dims => keep_dims),
            ((Int32, _reduced(x.shape, (axis,), keep_dims)),); name)
    end
end

# --- linear algebra -----------------------------------------------------------

"""
    matmul(x, y; transpose_x=false, transpose_y=false)

Batched matrix product over the last two dimensions, with broadcasting over the rest.
"""
function matmul(x::Var, y::Var; transpose_x=false, transpose_y=false, name="matmul")
    sx, sy = copy(x.shape), copy(y.shape)
    length(sx) == 1 && (sx = [1; sx])
    length(sy) == 1 && (sy = [sy; 1])
    transpose_x && (sx[end-1:end] = Base.reverse(sx[end-1:end]))
    transpose_y && (sy[end-1:end] = Base.reverse(sy[end-1:end]))
    sx[end] == sy[end-1] || throw(DimensionMismatch("matmul: inner dimensions $(sx[end]) and $(sy[end-1]) differ"))
    out = [_broadcast(sx[1:end-2], sy[1:end-2]); sx[end-1]; sy[end]]
    length(x.shape) == 1 && deleteat!(out, length(out) - 1)
    length(y.shape) == 1 && pop!(out)
    op!(x.graph, "matmul", (:x => x, :y => y, :transpose_x => transpose_x, :transpose_y => transpose_y),
        ((x.eltype, out),); name)
end

"""
    linear(x, weight; bias=nothing)

`x * weight' .+ bias` over the last dimension; `weight` is `[out, in]`.
"""
function linear(x::Var, weight::Var; bias=nothing, name="linear")
    x.shape[end] == weight.shape[2] || throw(DimensionMismatch("linear: x has $(x.shape[end]) features, weight expects $(weight.shape[2])"))
    op!(x.graph, "linear", (:x => x, :weight => weight, :bias => bias),
        ((x.eltype, [x.shape[1:end-1]; weight.shape[1]]),); name)
end

"""
    einsum(equation, values...)

MIL supports a fixed set of equations, e.g. `"nchw,nwhu->nchu"` (batched matmul over
the last two dimensions), for two rank-3 or rank-4 inputs.
"""
function einsum(equation::AbstractString, values::Var...; name="einsum")
    lhs, rhs = Base.split(replace(equation, " " => ""), "->")
    terms = Base.split(lhs, ",")
    length(terms) == length(values) || throw(ArgumentError("einsum: $(length(terms)) terms for $(length(values)) values"))
    dims = Dict{Char,Int}()
    for (t, v) in zip(terms, values)
        length(t) == ndims(v) || throw(DimensionMismatch("einsum: \"$t\" does not match rank $(ndims(v))"))
        for (c, d) in zip(t, v.shape)
            get!(dims, c, d) == d || dims[c] == 1 || d == 1 || throw(DimensionMismatch("einsum: index '$c' has sizes $(dims[c]) and $d"))
            dims[c] = max(dims[c], d)
        end
    end
    op!(first(values).graph, "einsum", (:values => collect(values), :equation => String(equation)),
        ((first(values).eltype, [dims[c] for c in rhs]),); name)
end

# --- convolution and pooling ------------------------------------------------

_tuple(x, n) = x isa Integer ? Base.fill(Int(x), n) : collect(Int, x)

function _spatial(in, k, s, d, pad_type, pad, ceil_mode=false)
    keff = (k - 1) * d + 1
    rnd = ceil_mode ? cld : fld
    pad_type == "same" && return cld(in, s)
    pad_type == "same_lower" && return cld(in, s)
    pb, pe = pad
    pad_type == "valid" && (pb = pe = 0)
    return rnd(in + pb + pe - keff, s) + 1
end

function _spatial_out(xs, ks, strides, dilations, pad_type, pad; ceil_mode=false)
    n = length(ks)
    strides = _tuple(something(strides, 1), n)
    dilations = _tuple(something(dilations, 1), n)
    pads = isnothing(pad) ? zeros(Int, 2n) : _tuple(pad, 2n)
    return [_spatial(xs[i], ks[i], strides[i], dilations[i], pad_type, pads[2i-1:2i], ceil_mode) for i in 1:n],
           strides, dilations, pads
end

"""
    conv(x, weight; bias=nothing, strides=1, pad_type="valid", pad=nothing, dilations=1, groups=1)

`x` is `[N, Cin, spatial...]`, `weight` is `[Cout, Cin ÷ groups, kernel...]`.
`pad_type` is `"valid"`, `"same"`, `"same_lower"` or `"custom"` (then `pad` gives
`[begin, end]` per spatial dimension).
"""
function conv(x::Var, weight::Var; bias=nothing, strides=1, pad_type="valid", pad=nothing,
              dilations=1, groups=1, name="conv")
    n = ndims(x) - 2
    x.shape[2] == weight.shape[2] * groups ||
        throw(DimensionMismatch("conv: x has $(x.shape[2]) channels, weight expects $(weight.shape[2] * groups)"))
    out, strides, dilations, pads = _spatial_out(x.shape[3:end], weight.shape[3:end], strides, dilations, pad_type, pad)
    op!(x.graph, "conv",
        (:x => x, :weight => weight, :bias => bias, :strides => strides, :pad_type => pad_type,
         :pad => pads, :dilations => dilations, :groups => Int32(groups)),
        ((x.eltype, [x.shape[1]; weight.shape[1]; out]),); name)
end

"""
    conv_transpose(x, weight; bias=nothing, strides=1, pad_type="valid", pad=nothing, dilations=1, groups=1)

`weight` is `[Cin, Cout ÷ groups, kernel...]`.
"""
function conv_transpose(x::Var, weight::Var; bias=nothing, strides=1, pad_type="valid", pad=nothing,
                        dilations=1, groups=1, name="conv_transpose")
    n = ndims(x) - 2
    s, d = _tuple(strides, n), _tuple(dilations, n)
    pads = isnothing(pad) ? zeros(Int, 2n) : _tuple(pad, 2n)
    ks = weight.shape[3:end]
    out = [pad_type == "same" ? x.shape[2+i] * s[i] :
           (x.shape[2+i] - 1) * s[i] + (ks[i] - 1) * d[i] + 1 - (pad_type == "custom" ? pads[2i-1] + pads[2i] : 0)
           for i in 1:n]
    op!(x.graph, "conv_transpose",
        (:x => x, :weight => weight, :bias => bias, :strides => s, :pad_type => pad_type,
         :pad => pads, :dilations => d, :groups => Int32(groups)),
        ((x.eltype, [x.shape[1]; weight.shape[2] * groups; out]),); name)
end

for op in (:avg_pool, :max_pool, :l2_pool)
    extra = op === :avg_pool ? (:(:exclude_padding_from_average => exclude_padding_from_average),) : ()
    @eval function $op(x::Var; kernel_sizes, strides=1, pad_type="valid", pad=nothing, ceil_mode=false,
                       exclude_padding_from_average=false, name=$(string(op)))
        ks = collect(Int, kernel_sizes)
        out, strides, _, pads = _spatial_out(x.shape[3:end], ks, strides, 1, pad_type, pad; ceil_mode)
        op!(x.graph, $(string(op)),
            (:x => x, :kernel_sizes => ks, :strides => strides, :pad_type => pad_type, :pad => pads,
             :ceil_mode => ceil_mode, $(extra...)),
            ((x.eltype, [x.shape[1:2]; out]),); name)
    end
end

# --- normalization ------------------------------------------------------------

"""
    layer_norm(x; axes=[-1], gamma=nothing, beta=nothing, epsilon=1e-5)
"""
layer_norm(x::Var; axes=[-1], gamma=nothing, beta=nothing, epsilon=1f-5, name="layer_norm") =
    op!(x.graph, "layer_norm", (:x => x, :axes => axes, :gamma => gamma, :beta => beta, :epsilon => _like(x, epsilon)),
        ((x.eltype, x.shape),); name)

batch_norm(x::Var, mean::Var, variance::Var; gamma=nothing, beta=nothing, epsilon=1f-5, name="batch_norm") =
    op!(x.graph, "batch_norm", (:x => x, :mean => mean, :variance => variance, :gamma => gamma,
                                :beta => beta, :epsilon => _like(x, epsilon)), ((x.eltype, x.shape),); name)

instance_norm(x::Var; gamma=nothing, beta=nothing, epsilon=1f-5, name="instance_norm") =
    op!(x.graph, "instance_norm", (:x => x, :gamma => gamma, :beta => beta, :epsilon => _like(x, epsilon)),
        ((x.eltype, x.shape),); name)

l2_norm(x::Var; epsilon=1f-6, name="l2_norm") =
    op!(x.graph, "l2_norm", (:x => x, :epsilon => _like(x, epsilon)), ((x.eltype, x.shape),); name)

local_response_norm(x::Var; size, alpha=1f-4, beta=0.75f0, k=1f0, name="local_response_norm") =
    op!(x.graph, "local_response_norm", (:x => x, :size => Int32(size), :alpha => _like(x, alpha),
                                         :beta => _like(x, beta), :k => _like(x, k)), ((x.eltype, x.shape),); name)

# --- shape manipulation -------------------------------------------------------

"""
    reshape(x, shape)

`shape` may contain one `-1` (inferred) and `0` (copy the input dimension).
"""
function reshape(x::Var, shape; name="reshape")
    s = [d == 0 ? x.shape[i] : d for (i, d) in enumerate(shape)]
    if (i = findfirst(==(-1), s)) !== nothing
        s[i] = length(x) ÷ prod(d for d in s if d != -1; init=1)
    end
    prod(s; init=1) == length(x) || throw(DimensionMismatch("cannot reshape $(x.shape) to $shape"))
    op!(x.graph, "reshape", (:x => x, :shape => collect(Int, shape)), ((x.eltype, s),); name)
end

transpose(x::Var, perm; name="transpose") =
    op!(x.graph, "transpose", (:x => x, :perm => collect(Int, perm)),
        ((x.eltype, x.shape[[_axis(p, ndims(x)) + 1 for p in perm]]),); name)

function expand_dims(x::Var, axes; name="expand_dims")
    rank = ndims(x) + length(axes)
    s = copy(x.shape)
    for a in sort(_axes(axes, rank))
        insert!(s, a + 1, 1)
    end
    op!(x.graph, "expand_dims", (:x => x, :axes => collect(Int, axes)), ((x.eltype, s),); name)
end

function squeeze(x::Var; axes=nothing, name="squeeze")
    drop = isnothing(axes) ? findall(==(1), x.shape) .- 1 : _axes(axes, ndims(x))
    s = [d for (i, d) in enumerate(x.shape) if !(i - 1 in drop)]
    op!(x.graph, "squeeze", (:x => x, :axes => axes), ((x.eltype, s),); name)
end

reverse(x::Var; axes=nothing, name="reverse") = op!(x.graph, "reverse", (:x => x, :axes => axes), ((x.eltype, x.shape),); name)

"""
    slice_by_index(x, begin_, end_; stride=nothing, begin_mask=nothing, end_mask=nothing, squeeze_mask=nothing)

Python-style half-open slices `begin_:stride:end_` per dimension (0-based).
"""
function slice_by_index(x::Var, begin_, end_; stride=nothing, begin_mask=nothing, end_mask=nothing,
                        squeeze_mask=nothing, name="slice_by_index")
    r = ndims(x)
    st = isnothing(stride) ? ones(Int, r) : collect(Int, stride)
    out = Int[]
    for i in 1:r
        n = x.shape[i]
        b = !isnothing(begin_mask) && begin_mask[i] ? (st[i] > 0 ? 0 : n - 1) : Base.mod(begin_[i], n + (begin_[i] == n))
        e = !isnothing(end_mask) && end_mask[i] ? (st[i] > 0 ? n : -1) : (end_[i] < 0 ? end_[i] + n : min(end_[i], n))
        len = st[i] > 0 ? max(0, cld(e - b, st[i])) : max(0, cld(b - e, -st[i]))
        (isnothing(squeeze_mask) || !squeeze_mask[i]) && push!(out, len)
    end
    op!(x.graph, "slice_by_index",
        (:x => x, :begin => collect(Int, begin_), :end => collect(Int, end_), :stride => stride,
         :begin_mask => begin_mask, :end_mask => end_mask, :squeeze_mask => squeeze_mask),
        ((x.eltype, out),); name)
end

function slice_by_size(x::Var, begin_, size; name="slice_by_size")
    out = [s == -1 ? x.shape[i] - begin_[i] : s for (i, s) in enumerate(size)]
    op!(x.graph, "slice_by_size", (:x => x, :begin => collect(Int, begin_), :size => collect(Int, size)),
        ((x.eltype, out),); name)
end

function concat(values::Union{Tuple,AbstractVector}; axis, interleave=false, name="concat")
    v1 = first(values)
    a = _axis(axis, ndims(v1))
    s = copy(v1.shape)
    s[a+1] = sum(v.shape[a+1] for v in values)
    op!(v1.graph, "concat", (:values => collect(values), :axis => axis, :interleave => interleave),
        ((v1.eltype, s),); name)
end

"""
    split(x; axis, num_splits=nothing, split_sizes=nothing) -> Tuple{Var...}
"""
function split(x::Var; axis, num_splits=nothing, split_sizes=nothing, name="split")
    a = _axis(axis, ndims(x))
    sizes = isnothing(split_sizes) ? Base.fill(x.shape[a+1] ÷ num_splits, num_splits) : collect(Int, split_sizes)
    outs = [(x.eltype, [i == a + 1 ? sz : d for (i, d) in enumerate(x.shape)]) for sz in sizes]
    op!(x.graph, "split", (:x => x, :axis => axis, :num_splits => isnothing(num_splits) ? nothing : Int32(num_splits),
                           :split_sizes => split_sizes), outs; name)
end

function stack(values::Union{Tuple,AbstractVector}; axis, name="stack")
    v1 = first(values)
    s = copy(v1.shape)
    insert!(s, _axis(axis, ndims(v1) + 1) + 1, length(values))
    op!(v1.graph, "stack", (:values => collect(values), :axis => axis), ((v1.eltype, s),); name)
end

tile(x::Var, reps; name="tile") = op!(x.graph, "tile", (:x => x, :reps => collect(Int, reps)), ((x.eltype, x.shape .* collect(reps)),); name)

"""
    pad(x, pad; mode="constant", constant_val=0)

`pad` lists `[begin, end]` amounts for the last `length(pad) ÷ 2` dimensions.
"""
function pad(x::Var, pad; mode="constant", constant_val=0, name="pad")
    # The ANE expects amounts for every dimension.
    p = [zeros(Int, 2 * ndims(x) - length(pad)); collect(Int, pad)]
    s = copy(x.shape)
    k = length(p) ÷ 2
    for i in 1:k
        s[end-k+i] += p[2i-1] + p[2i]
    end
    op!(x.graph, "pad", (:x => x, :pad => p, :mode => mode, :constant_val => _like(x, constant_val)),
        ((x.eltype, s),); name)
end

flatten2d(x::Var; axis=1, name="flatten2d") =
    op!(x.graph, "flatten2d", (:x => x, :axis => axis),
        ((x.eltype, [prod(x.shape[1:_axis(axis, ndims(x) + 1)]; init=1), prod(x.shape[_axis(axis, ndims(x) + 1)+1:end]; init=1)]),); name)

function depth_to_space(x::Var; block_size, name="depth_to_space")
    n, c, h, w = x.shape
    op!(x.graph, "depth_to_space", (:x => x, :block_size => Int32(block_size)),
        ((x.eltype, [n, c ÷ block_size^2, h * block_size, w * block_size]),); name)
end

function space_to_depth(x::Var; block_size, name="space_to_depth")
    n, c, h, w = x.shape
    op!(x.graph, "space_to_depth", (:x => x, :block_size => Int32(block_size)),
        ((x.eltype, [n, c * block_size^2, h ÷ block_size, w ÷ block_size]),); name)
end

function pixel_shuffle(x::Var; upscale_factor, name="pixel_shuffle")
    n, c, h, w = x.shape
    r = upscale_factor
    op!(x.graph, "pixel_shuffle", (:x => x, :upscale_factor => Int32(r)), ((x.eltype, [n, c ÷ r^2, h * r, w * r]),); name)
end

function pixel_unshuffle(x::Var; downscale_factor, name="pixel_unshuffle")
    n, c, h, w = x.shape
    r = downscale_factor
    op!(x.graph, "pixel_unshuffle", (:x => x, :downscale_factor => UInt32(r)), ((x.eltype, [n, c * r^2, h ÷ r, w ÷ r]),); name)
end

band_part(x::Var; lower=-1, upper=-1, name="band_part") =
    op!(x.graph, "band_part", (:x => x, :lower => Int32(lower), :upper => Int32(upper)), ((x.eltype, x.shape),); name)

cumsum(x::Var; axis=0, exclusive=false, reverse=false, name="cumsum") =
    op!(x.graph, "cumsum", (:x => x, :axis => axis, :exclusive => exclusive, :reverse => reverse), ((x.eltype, x.shape),); name)

"""
    topk(x; k=1, axis=-1, ascending=false) -> (values, indices)
"""
function topk(x::Var; k=1, axis=-1, ascending=false, name="topk")
    s = copy(x.shape)
    s[_axis(axis, ndims(x))+1] = k
    op!(x.graph, "topk", (:x => x, :k => Int32(k), :axis => axis, :ascending => ascending),
        ((x.eltype, s), (Int32, s)); name)
end

argsort(x::Var; axis=-1, ascending=false, name="argsort") =
    op!(x.graph, "argsort", (:x => x, :axis => axis, :ascending => ascending), ((Int32, x.shape),); name)

function gather(x::Var, indices::Var; axis=0, batch_dims=0, name="gather")
    a = _axis(axis, ndims(x))
    s = [x.shape[1:a]; indices.shape[batch_dims+1:end]; x.shape[a+2:end]]
    op!(x.graph, "gather", (:x => x, :indices => indices, :axis => axis, :batch_dims => Int32(batch_dims)),
        ((x.eltype, s),); name)
end

gather_along_axis(x::Var, indices::Var; axis=0, name="gather_along_axis") =
    op!(x.graph, "gather_along_axis", (:x => x, :indices => indices, :axis => axis), ((x.eltype, indices.shape),); name)

function gather_nd(x::Var, indices::Var; batch_dims=0, name="gather_nd")
    s = [indices.shape[1:end-1]; x.shape[batch_dims+indices.shape[end]+1:end]]
    op!(x.graph, "gather_nd", (:x => x, :indices => indices, :batch_dims => Int32(batch_dims)), ((x.eltype, s),); name)
end

for op in (:scatter, :scatter_along_axis)
    @eval $op(data::Var, indices::Var, updates::Var; axis=0, mode="update", name=$(string(op))) =
        op!(data.graph, $(string(op)), (:data => data, :indices => indices, :updates => updates, :axis => axis, :mode => mode),
            ((data.eltype, data.shape),); name)
end

scatter_nd(data::Var, indices::Var, updates::Var; mode="update", name="scatter_nd") =
    op!(data.graph, "scatter_nd", (:data => data, :indices => indices, :updates => updates, :mode => mode),
        ((data.eltype, data.shape),); name)

function one_hot(indices::Var, one_hot_vector_size::Integer; axis=-1, on_value=1f0, off_value=0f0, eltype=Float16, name="one_hot")
    s = copy(indices.shape)
    insert!(s, _axis(axis, ndims(indices) + 1) + 1, one_hot_vector_size)
    op!(indices.graph, "one_hot", (:indices => indices, :one_hot_vector_size => Int32(one_hot_vector_size), :axis => axis,
                                   :on_value => convert(eltype, on_value), :off_value => convert(eltype, off_value)),
        ((eltype, s),); name)
end

shape(x::Var; name="shape") = op!(x.graph, "shape", (:x => x,), ((Int32, [ndims(x)]),); name)

fill(g::Graph, shape, value::Number; eltype=Float16, name="fill") =
    op!(g, "fill", (:shape => collect(Int, shape), :value => convert(eltype, value)), ((eltype, collect(Int, shape)),); name)

function range_1d(g::Graph, start::T, stop::T, step::T; name="range_1d") where {T<:Union{Int32,Float16,Float32}}
    n = max(0, Base.ceil(Int, (stop - start) / step))
    op!(g, "range_1d", (:start => start, :end => stop, :step => step), ((T == Int32 ? Int32 : T, [n]),); name)
end

function sliding_windows(x::Var; axis, size, stride=1, name="sliding_windows")
    a = _axis(axis, ndims(x))
    s = [x.shape[1:a]; (x.shape[a+1] - size) ÷ stride + 1; size; x.shape[a+2:end]]
    op!(x.graph, "sliding_windows", (:x => x, :axis => axis, :size => Int32(size), :stride => Int32(stride)), ((x.eltype, s),); name)
end

# --- image resizing -----------------------------------------------------------

function upsample_nearest_neighbor(x::Var; scale_factor_height=1, scale_factor_width=1, name="upsample_nearest_neighbor")
    s = copy(x.shape)
    s[end-1] = Base.floor(Int, s[end-1] * scale_factor_height)
    s[end] = Base.floor(Int, s[end] * scale_factor_width)
    op!(x.graph, "upsample_nearest_neighbor",
        (:x => x, :scale_factor_height => Float32(scale_factor_height), :scale_factor_width => Float32(scale_factor_width)),
        ((x.eltype, s),); name)
end

function upsample_bilinear(x::Var; scale_factor_height=1, scale_factor_width=1, align_corners=false,
                           half_pixel_centers=!align_corners, name="upsample_bilinear")
    s = copy(x.shape)
    s[end-1] = Base.floor(Int, s[end-1] * scale_factor_height)
    s[end] = Base.floor(Int, s[end] * scale_factor_width)
    op!(x.graph, "upsample_bilinear",
        (:x => x, :scale_factor_height => Float32(scale_factor_height), :scale_factor_width => Float32(scale_factor_width),
         :align_corners => align_corners, :half_pixel_centers => half_pixel_centers),
        ((x.eltype, s),); name)
end

resize_nearest_neighbor(x::Var; target_size_height, target_size_width, name="resize_nearest_neighbor") =
    op!(x.graph, "resize_nearest_neighbor",
        (:x => x, :target_size_height => Int32(target_size_height), :target_size_width => Int32(target_size_width)),
        ((x.eltype, [x.shape[1:end-2]; target_size_height; target_size_width]),); name)

resize_bilinear(x::Var; target_size_height, target_size_width, sampling_mode="DEFAULT", name="resize_bilinear") =
    op!(x.graph, "resize_bilinear",
        (:x => x, :target_size_height => Int32(target_size_height), :target_size_width => Int32(target_size_width),
         :sampling_mode => sampling_mode),
        ((x.eltype, [x.shape[1:end-2]; target_size_height; target_size_width]),); name)

crop(x::Var; crop_height, crop_width, name="crop") =
    op!(x.graph, "crop", (:x => x, :crop_height => collect(Int, crop_height), :crop_width => collect(Int, crop_width)),
        ((x.eltype, [x.shape[1:end-2]; x.shape[end-1] - sum(crop_height); x.shape[end] - sum(crop_width)]),); name)

# --- attention and recurrent layers --------------------------------------------

"""
    scaled_dot_product_attention(query, key, value; attn_mask=nothing)

`softmax(query * key' / sqrt(E) + attn_mask) * value` over the last two dimensions.
"""
scaled_dot_product_attention(query::Var, key::Var, value::Var; attn_mask=nothing, name="sdpa") =
    op!(query.graph, "scaled_dot_product_attention", (:query => query, :key => key, :value => value, :attn_mask => attn_mask),
        ((query.eltype, [query.shape[1:end-1]; value.shape[end]]),); name)

"""
    lstm(x, initial_h, initial_c, weight_ih, weight_hh; bias=nothing, output_sequence=false, ...) -> (output, h, c)

`x` is `[T, B, I]`; the weights are `[4H, I]` and `[4H, H]` (gate order i, f, o, g).
"""
function lstm(x::Var, initial_h::Var, initial_c::Var, weight_ih::Var, weight_hh::Var; bias=nothing,
              direction="forward", output_sequence=false, recurrent_activation="sigmoid",
              cell_activation="tanh", activation="tanh", name="lstm")
    T, B, _ = x.shape
    H = weight_hh.shape[2]
    out = output_sequence ? [T, B, H] : [1, B, H]
    op!(x.graph, "lstm",
        (:x => x, :initial_h => initial_h, :initial_c => initial_c, :weight_ih => weight_ih, :weight_hh => weight_hh,
         :bias => bias, :direction => direction, :output_sequence => output_sequence,
         :recurrent_activation => recurrent_activation, :cell_activation => cell_activation, :activation => activation),
        ((x.eltype, out), (x.eltype, [B, H]), (x.eltype, [B, H])); name)
end

for (op, gates) in ((:gru, 3), (:rnn, 1))
    @eval function $op(x::Var, initial_h::Var, weight_ih::Var, weight_hh::Var; bias=nothing,
                       direction="forward", output_sequence=false, activation="tanh", name=$(string(op)))
        T, B, _ = x.shape
        H = weight_hh.shape[2]
        out = output_sequence ? [T, B, H] : [1, B, H]
        op!(x.graph, $(string(op)),
            (:x => x, :initial_h => initial_h, :weight_ih => weight_ih, :weight_hh => weight_hh, :bias => bias,
             :direction => direction, :output_sequence => output_sequence, :activation => activation),
            ((x.eltype, out), (x.eltype, [B, H])); name)
    end
end

# --- quantization -------------------------------------------------------------

quantize(input::Var, scale; zero_point=nothing, axis=nothing, output_dtype=Int8, name="quantize") =
    op!(input.graph, "quantize", (:input => input, :zero_point => zero_point, :scale => scale, :axis => axis,
                                  :output_dtype => output_dtype), ((output_dtype, input.shape),); name)

dequantize(input::Var, scale; zero_point=nothing, axis=nothing, eltype=Float16, name="dequantize") =
    op!(input.graph, "dequantize", (:input => input, :zero_point => zero_point, :scale => scale, :axis => axis),
        ((eltype, input.shape),); name)

"""
    constant_int8(g, W; axis=0) -> Var

A constant stored as int8 with one fp16 scale per slice along `axis` (symmetric
quantization), dequantized by the ANE when the program is loaded
(`constexpr_affine_dequantize`). Halves the weight memory of fp16 constants.
"""
function constant_int8(g::Graph, W::AbstractArray{<:Real}; axis::Integer=0, name="const_int8")
    dims = Tuple(d for d in 1:ndims(W) if d != axis + 1)
    scale = vec(Float16.(Base.maximum(Base.abs, W; dims, init=0f0) ./ 127))
    scale[scale .== 0] .= 1
    shp = ntuple(d -> d == axis + 1 ? length(scale) : 1, ndims(W))
    q = Base.round.(Int8, clamp.(W ./ Float32.(Base.reshape(scale, shp)), -127, 127))
    blob(off) = "BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64($off))"
    oq = push!(g.weights, q)
    os = push!(g.weights, scale)
    oz = push!(g.weights, zeros(Int8, length(scale)))
    v = Var(g, _fresh!(g, name), Float16, collect(size(W)))
    push!(g.body, "$(_miltype(v)) $(v.name) = constexpr_affine_dequantize()[axis = int32($axis), " *
                  "name = string(\"$(v.name)\"), quantized_data = $(_miltype(Int8, size(W)))($(blob(oq))), " *
                  "scale = $(_miltype(Float16, size(scale)))($(blob(os))), " *
                  "zero_point = $(_miltype(Int8, size(scale)))($(blob(oz)))];")
    return v
end
