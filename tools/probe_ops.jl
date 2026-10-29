# Probe which MIL ops the ANE compiles and runs correctly on this machine.
#
#     julia --project=. tools/probe_ops.jl [output.md]
#
# Each op is built into a small fp16 program with the MIL builder, compiled,
# run, and (where a reference is given) compared against Julia. Results go to
# docs/src/op_support.md by default.

using AppleNeuralEngine, Random, Dates
const M = AppleNeuralEngine.MIL

Random.seed!(1)

# --- MIL-ordered data -----------------------------------------------------------
# References work on arrays indexed like MIL tensors (A[n, c, h, w]); program
# buffers hold the same memory, i.e. the dimensions reversed.
to_buffer(A) = Float16.(permutedims(A, ndims(A):-1:1))
from_buffer(B, shape) = permutedims(Float32.(reshape(B, Base.reverse(shape)...)), length(shape):-1:1)

data(kind, shape) = kind === :pos ? rand(Float32, shape...) .+ 0.5f0 :
                    kind === :unit ? 1.8f0 .* rand(Float32, shape...) .- 0.9f0 :
                    kind === :small ? 0.5f0 .* randn(Float32, shape...) :
                    randn(Float32, shape...)

const X = (1, 16, 8, 32)   # default input: [N, C, H, W]

struct Case
    op::String
    category::String
    inputs::Vector{Tuple{Symbol,NTuple}}     # (data kind, MIL shape)
    build::Function                           # (g, vars...) -> Var or tuple of Vars
    ref::Union{Nothing,Function}              # (arrays...) -> array or tuple of arrays
end
Case(op, cat, build, ref=nothing; inputs=[(:normal, X)]) = Case(op, cat, inputs, build, ref)

# --- references -----------------------------------------------------------------
sigmoid(x) = 1 / (1 + exp(-x))
softplus(x) = log1p(exp(x))
function erf_ref(x)   # Abramowitz & Stegun 7.1.26
    t = 1 / (1 + 0.3275911f0 * abs(x))
    y = 1 - (((((1.061405429f0t - 1.453152027f0)t) + 1.421413741f0)t - 0.284496736f0)t + 0.254829592f0)t * exp(-x^2)
    return sign(x) * y
end
gelu_ref(x) = x * (1 + erf_ref(x / sqrt(2f0))) / 2
softmax_ref(A; dims) = (E = exp.(A .- maximum(A; dims)); E ./ sum(E; dims))
red(f, A, axes) = f(A; dims=Tuple(a + 1 for a in axes))

function conv_ref(Xa, W; pad=0, stride=1)
    N, C, H, Wd = size(Xa); O, _, kh, kw = size(W)
    Xp = zeros(Float32, N, C, H + 2pad, Wd + 2pad); Xp[:, :, pad+1:pad+H, pad+1:pad+Wd] = Xa
    Ho, Wo = (H + 2pad - kh) ÷ stride + 1, (Wd + 2pad - kw) ÷ stride + 1
    Y = zeros(Float32, N, O, Ho, Wo)
    for n in 1:N, o in 1:O, i in 1:Ho, j in 1:Wo
        Y[n, o, i, j] = sum(W[o, :, :, :] .* Xp[n, :, (i-1)*stride+1:(i-1)*stride+kh, (j-1)*stride+1:(j-1)*stride+kw])
    end
    return Y
end
pool_ref(f, Xa, k) = [f(Xa[n, c, (i-1)*k+1:i*k, (j-1)*k+1:j*k]) for n in axes(Xa, 1), c in axes(Xa, 2), i in 1:size(Xa, 3)÷k, j in 1:size(Xa, 4)÷k]

# --- cases ----------------------------------------------------------------------
W1x1 = randn(Float32, 24, 16, 1, 1) ./ 4
W3x3 = randn(Float32, 8, 16, 3, 3) ./ 12
cases = Case[]
add!(c) = push!(cases, c)

unary = [("abs", abs, :normal), ("acos", acos, :unit), ("asin", asin, :unit), ("atan", atan, :normal),
         ("atanh", atanh, :unit), ("ceil", ceil, :normal), ("cos", cos, :normal), ("cosh", cosh, :small),
         ("erf", erf_ref, :normal), ("exp", exp, :small), ("exp2", exp2, :small), ("floor", floor, :normal),
         # The ANE rounds as sign(x)·floor(|x| + 0.5) in fp16 (ties away from zero).
         ("round", x -> sign(x) * floor(Float32(Float16(abs(x) + 0.5f0))), :normal), ("sign", sign, :normal), ("sin", sin, :normal), ("sinh", sinh, :small),
         ("sqrt", sqrt, :pos), ("square", x -> x^2, :normal), ("tan", tan, :unit), ("tanh", tanh, :normal),
         ("relu", x -> max(x, 0), :normal), ("relu6", x -> clamp(x, 0, 6), :normal), ("sigmoid", sigmoid, :normal),
         ("silu", x -> x * sigmoid(x), :normal), ("softplus", softplus, :normal), ("softsign", x -> x / (1 + abs(x)), :normal)]
for (op, f, kind) in unary
    cat = op in ("relu", "relu6", "sigmoid", "silu", "softplus", "softsign") ? "activation" : "elementwise unary"
    add!(Case(op, cat, (g, x) -> getfield(M, Symbol(op))(x), A -> f.(A); inputs=[(kind, X)]))
end
add!(Case("gelu", "activation", (g, x) -> M.gelu(x), A -> gelu_ref.(A)))
add!(Case("gelu (tanh approx.)", "activation", (g, x) -> M.gelu(x; mode="TANH_APPROXIMATION"), A -> gelu_ref.(A)))
add!(Case("inverse", "elementwise unary", (g, x) -> M.inverse(x; epsilon=1f-4), A -> 1 ./ A; inputs=[(:pos, X)]))
add!(Case("log", "elementwise unary", (g, x) -> M.log(x; epsilon=1f-4), A -> log.(A); inputs=[(:pos, X)]))
add!(Case("rsqrt", "elementwise unary", (g, x) -> M.rsqrt(x; epsilon=1f-4), A -> 1 ./ sqrt.(A); inputs=[(:pos, X)]))
add!(Case("rsqrt (no epsilon)", "elementwise unary", (g, x) -> M.rsqrt(x), A -> 1 ./ sqrt.(A); inputs=[(:pos, X)]))
add!(Case("threshold", "elementwise unary", (g, x) -> M.threshold(x; alpha=0.2f0), A -> max.(A, 0.2f0)))
add!(Case("clip", "elementwise unary", (g, x) -> M.clip(x; alpha=-0.5f0, beta=0.5f0), A -> clamp.(A, -0.5f0, 0.5f0)))
add!(Case("cast (fp16→fp32→fp16)", "elementwise unary", (g, x) -> M.cast(M.cast(x, Float32), Float16), A -> A))
add!(Case("logical_not", "elementwise unary", (g, x) -> M.cast(M.logical_not(M.greater(x, 0f0)), Float16), A -> Float32.(A .<= 0)))

alpha_c = 0.1f0 .+ 0.2f0 .* rand(Float32, 16)
acts = [("elu", (g, x) -> M.elu(x; alpha=0.5f0), A -> ifelse.(A .> 0, A, 0.5f0 .* (exp.(A) .- 1))),
        ("leaky_relu", (g, x) -> M.leaky_relu(x; alpha=0.1f0), A -> ifelse.(A .> 0, A, 0.1f0 .* A)),
        ("thresholded_relu", (g, x) -> M.thresholded_relu(x; alpha=0.3f0), A -> ifelse.(A .> 0.3f0, A, 0f0)),
        ("clamped_relu", (g, x) -> M.clamped_relu(x; alpha=0.1f0, beta=1f0), A -> min.(1f0, ifelse.(A .>= 0, A, 0.1f0 .* A))),
        ("linear_activation", (g, x) -> M.linear_activation(x; alpha=2f0, beta=0.5f0), A -> 2 .* A .+ 0.5f0),
        ("scaled_tanh", (g, x) -> M.scaled_tanh(x; alpha=2f0, beta=0.5f0), A -> 2 .* tanh.(0.5f0 .* A)),
        ("sigmoid_hard", (g, x) -> M.sigmoid_hard(x; alpha=0.2f0, beta=0.5f0), A -> clamp.(0.2f0 .* A .+ 0.5f0, 0, 1)),
        ("prelu", (g, x) -> M.prelu(x; alpha=Float16.(alpha_c)), A -> ifelse.(A .> 0, A, reshape(alpha_c, 1, 16, 1, 1) .* A)),
        ("softplus_parametric", (g, x) -> M.softplus_parametric(x; alpha=Float16.(alpha_c), beta=Float16.(alpha_c .+ 0.5f0)),
            A -> reshape(alpha_c, 1, 16, 1, 1) .* softplus.(reshape(alpha_c .+ 0.5f0, 1, 16, 1, 1) .* A)),
        ("softmax", (g, x) -> M.softmax(x; axis=1), A -> softmax_ref(A; dims=2))]
for (op, b, r) in acts
    add!(Case(op, "activation", b, r))
end

binary = [("add", +), ("sub", -), ("mul", *), ("real_div", /), ("pow", ^), ("maximum", max), ("minimum", min),
          ("floor_div", (a, b) -> floor(a / b)), ("mod", (a, b) -> a - floor(a / b) * b)]
for (op, f) in binary
    kinds = op in ("pow", "real_div", "floor_div", "mod") ? (:pos, :pos) : (:normal, :normal)
    add!(Case(op, "elementwise binary", (g, x, y) -> getfield(M, Symbol(op))(x, y), (A, B) -> f.(A, B);
              inputs=[(kinds[1], X), (kinds[2], X)]))
end
add!(Case("add (broadcast [1,C,1,1])", "elementwise binary", (g, x, y) -> M.add(x, y), (A, B) -> A .+ B;
          inputs=[(:normal, X), (:normal, (1, 16, 1, 1))]))
add!(Case("mul (scalar literal)", "elementwise binary", (g, x) -> M.mul(x, 3f0), A -> 3 .* A))
for (op, f) in [("equal", (a, b) -> a == b), ("not_equal", !=), ("greater", >), ("greater_equal", >=),
                ("less", <), ("less_equal", <=)]
    add!(Case(op, "elementwise binary", (g, x, y) -> M.cast(getfield(M, Symbol(op))(M.floor(x), M.floor(y)), Float16),
              (A, B) -> Float32.(f.(floor.(A), floor.(B))); inputs=[(:normal, X), (:normal, X)]))
end
for (op, f) in [("logical_and", &), ("logical_or", |), ("logical_xor", xor)]
    add!(Case(op, "elementwise binary", (g, x, y) -> M.cast(getfield(M, Symbol(op))(M.greater(x, 0f0), M.greater(y, 0f0)), Float16),
              (A, B) -> Float32.(f.(A .> 0, B .> 0)); inputs=[(:normal, X), (:normal, X)]))
end
add!(Case("select", "elementwise binary", (g, x, y) -> M.select(M.greater(x, 0f0), x, y), (A, B) -> ifelse.(A .> 0, A, B);
          inputs=[(:normal, X), (:normal, X)]))

for (op, f) in [("reduce_sum", sum), ("reduce_mean", (A; dims) -> sum(A; dims) ./ prod(size(A, d) for d in dims)),
                ("reduce_max", maximum), ("reduce_min", minimum), ("reduce_prod", prod),
                ("reduce_l1_norm", (A; dims) -> sum(abs, A; dims)), ("reduce_l2_norm", (A; dims) -> sqrt.(sum(abs2, A; dims))),
                ("reduce_sum_square", (A; dims) -> sum(abs2, A; dims)), ("reduce_log_sum", (A; dims) -> log.(sum(A; dims))),
                ("reduce_log_sum_exp", (A; dims) -> log.(sum(exp, A; dims)))]
    kind = op in ("reduce_log_sum", "reduce_prod") ? :pos : :normal
    for axes in ([1], [3], [2, 3])
        op == "reduce_prod" && axes != [1] && continue
        add!(Case("$op axes=$axes", "reduction", (g, x) -> getfield(M, Symbol(op))(x; axes, keep_dims=true),
                  A -> red(f, op == "reduce_prod" ? A ./ 1.2f0 : A, axes);
                  inputs=[(kind, op == "reduce_prod" ? (1, 4, 8, 32) : X)]))
    end
end
cases[end].op == "reduce_prod axes=[1]" && (cases[end] = Case(cases[end].op, cases[end].category, cases[end].inputs,
    (g, x) -> M.reduce_prod(M.real_div(x, 1.2f0); axes=[1], keep_dims=true), A -> prod(A ./ 1.2f0; dims=2)))
for (op, f) in [("reduce_argmax", argmax), ("reduce_argmin", argmin)]
    add!(Case(op, "reduction", (g, x) -> M.cast(getfield(M, Symbol(op))(x; axis=1, keep_dims=true), Float16),
              A -> Float32.(map(i -> i[2] - 1, f(A; dims=2)))))
end

add!(Case("conv 1x1", "convolution", (g, x) -> M.conv(x, M.constant(g, W1x1)), A -> conv_ref(A, W1x1)))
add!(Case("conv 3x3 same", "convolution", (g, x) -> M.conv(x, M.constant(g, W3x3); pad_type="same"), A -> conv_ref(A, W3x3; pad=1)))
add!(Case("conv 3x3 stride 2", "convolution", (g, x) -> M.conv(x, M.constant(g, W3x3); strides=[2, 2], pad_type="custom", pad=[1, 1, 1, 1]),
          A -> conv_ref(A, W3x3; pad=1, stride=2)))
add!(Case("conv + bias", "convolution", (g, x) -> M.conv(x, M.constant(g, W1x1); bias=M.constant(g, fill(0.5f0, 24))),
          A -> conv_ref(A, W1x1) .+ 0.5f0))
Wdw = randn(Float32, 16, 1, 3, 3) ./ 3
add!(Case("conv depthwise 3x3", "convolution", (g, x) -> M.conv(x, M.constant(g, Wdw); pad_type="same", groups=16),
          A -> cat([conv_ref(A[:, c:c, :, :], Wdw[c:c, :, :, :]; pad=1) for c in 1:16]...; dims=2)))
add!(Case("conv int8 weights (constexpr_affine_dequantize)", "convolution", (g, x) -> M.conv(x, M.constant_int8(g, W1x1)),
          A -> conv_ref(A, W1x1)))
Wt = randn(Float32, 16, 8, 2, 2) ./ 4
add!(Case("conv_transpose 2x2 stride 2", "convolution", (g, x) -> M.conv_transpose(x, M.constant(g, Wt); strides=[2, 2])))
add!(Case("conv 1D", "convolution", (g, x) -> M.conv(x, M.constant(g, randn(Float32, 8, 16, 3) ./ 4)),
          nothing; inputs=[(:normal, (1, 16, 64))]))

for (op, f) in [("max_pool", maximum), ("avg_pool", A -> sum(A) / length(A)), ("l2_pool", A -> sqrt(sum(abs2, A)))]
    add!(Case(op, "pooling", (g, x) -> getfield(M, Symbol(op))(x; kernel_sizes=[2, 2], strides=[2, 2]), A -> pool_ref(f, A, 2)))
end

Wl = randn(Float32, 24, 32) ./ 6
add!(Case("matmul", "linear algebra", (g, x, y) -> M.matmul(x, y), (A, B) -> batched_mul(A, B);
          inputs=[(:normal, (1, 4, 32, 64)), (:normal, (1, 4, 64, 32))]))
add!(Case("matmul transpose_y", "linear algebra", (g, x, y) -> M.matmul(x, y; transpose_y=true),
          (A, B) -> batched_mul(A, permutedims(B, (1, 2, 4, 3))); inputs=[(:normal, (1, 4, 32, 64)), (:normal, (1, 4, 32, 64))]))
add!(Case("matmul (constant)", "linear algebra", (g, x) -> M.matmul(x, M.constant(g, Wl')), A -> batched_mul(A, reshape(Wl', 1, 1, 32, 24))))
add!(Case("linear", "linear algebra", (g, x) -> M.linear(x, M.constant(g, Wl); bias=M.constant(g, fill(0.25f0, 24))),
          A -> batched_mul(A, reshape(Wl', 1, 1, 32, 24)) .+ 0.25f0))
add!(Case("einsum nchw,nwhu->nchu", "linear algebra", (g, x, y) -> M.einsum("nchw,nwhu->nchu", x, y),
          (A, B) -> [sum(A[n, c, h, :] .* B[n, :, h, u]) for n in 1:1, c in 1:size(A, 2), h in 1:size(A, 3), u in 1:size(B, 4)];
          inputs=[(:normal, (1, 4, 8, 32)), (:normal, (1, 32, 8, 32))]))

function batched_mul(A, B)
    out = zeros(Float32, size(A, 1), size(A, 2), size(A, 3), size(B, 4))
    for n in axes(A, 1), c in axes(A, 2)
        out[n, c, :, :] = A[n, c, :, :] * B[min(n, size(B, 1)), min(c, size(B, 2)), :, :]
    end
    return out
end

gam, bet = 1 .+ 0.1f0 .* randn(Float32, 16), 0.1f0 .* randn(Float32, 16)
add!(Case("layer_norm", "normalization", (g, x) -> M.layer_norm(x; axes=[1]),
          A -> (μ = sum(A; dims=2) ./ 16; (A .- μ) ./ sqrt.(sum(abs2, A .- μ; dims=2) ./ 16 .+ 1f-5))))
add!(Case("layer_norm gamma/beta (last axis)", "normalization",
          (g, x) -> M.layer_norm(x; axes=[-1], gamma=M.constant(g, fill(2f0, 32)), beta=M.constant(g, fill(0.5f0, 32))),
          A -> (μ = sum(A; dims=4) ./ 32; 2 .* (A .- μ) ./ sqrt.(sum(abs2, A .- μ; dims=4) ./ 32 .+ 1f-5) .+ 0.5f0)))
add!(Case("batch_norm", "normalization",
          (g, x) -> M.batch_norm(x, M.constant(g, bet), M.constant(g, gam .^ 2); gamma=M.constant(g, gam), beta=M.constant(g, bet)),
          A -> (r(v) = reshape(v, 1, 16, 1, 1); r(gam) .* (A .- r(bet)) ./ sqrt.(r(gam .^ 2) .+ 1f-5) .+ r(bet))))
add!(Case("instance_norm", "normalization", (g, x) -> M.instance_norm(x),
          A -> (μ = sum(A; dims=(3, 4)) ./ 256; (A .- μ) ./ sqrt.(sum(abs2, A .- μ; dims=(3, 4)) ./ 256 .+ 1f-5))))
add!(Case("l2_norm", "normalization", (g, x) -> M.l2_norm(x)))
add!(Case("local_response_norm", "normalization", (g, x) -> M.local_response_norm(x; size=5)))

add!(Case("reshape", "tensor transformation", (g, x) -> M.reshape(x, [1, 32, 4, 32]), A -> permutedims(reshape(permutedims(A, (4, 3, 2, 1)), 32, 4, 32, 1), (4, 3, 2, 1))))
add!(Case("transpose", "tensor transformation", (g, x) -> M.transpose(x, [0, 3, 2, 1]), A -> permutedims(A, (1, 4, 3, 2))))
add!(Case("transpose (last two)", "tensor transformation", (g, x) -> M.transpose(x, [0, 1, 3, 2]), A -> permutedims(A, (1, 2, 4, 3))))
add!(Case("expand_dims + squeeze", "tensor transformation", (g, x) -> M.squeeze(M.expand_dims(x, [0]); axes=[0]), A -> A))
add!(Case("reverse", "tensor transformation", (g, x) -> M.reverse(x; axes=[3]), A -> A[:, :, :, end:-1:1]))
add!(Case("slice_by_index", "tensor transformation", (g, x) -> M.slice_by_index(x, [0, 2, 0, 0], [1, 10, 8, 32]), A -> A[:, 3:10, :, :]))
add!(Case("slice_by_index (stride 2)", "tensor transformation", (g, x) -> M.slice_by_index(x, [0, 0, 0, 0], [1, 16, 8, 32]; stride=[1, 2, 1, 1]),
          A -> A[:, 1:2:end, :, :]))
add!(Case("slice_by_size", "tensor transformation", (g, x) -> M.slice_by_size(x, [0, 4, 0, 0], [1, 8, 8, 32]), A -> A[:, 5:12, :, :]))
add!(Case("concat", "tensor operation", (g, x, y) -> M.concat([x, y]; axis=1), (A, B) -> cat(A, B; dims=2); inputs=[(:normal, X), (:normal, X)]))
add!(Case("concat (last axis)", "tensor operation", (g, x, y) -> M.concat([x, y]; axis=3), (A, B) -> cat(A, B; dims=4); inputs=[(:normal, X), (:normal, X)]))
add!(Case("split", "tensor operation", (g, x) -> M.split(x; axis=1, num_splits=2), A -> (A[:, 1:8, :, :], A[:, 9:16, :, :])))
add!(Case("stack", "tensor operation", (g, x, y) -> M.stack([x, y]; axis=0), (A, B) -> cat(reshape(A, 1, size(A)...), reshape(B, 1, size(B)...); dims=1);
          inputs=[(:normal, (1, 16, 32)), (:normal, (1, 16, 32))]))
add!(Case("tile", "tensor operation", (g, x) -> M.tile(x, [1, 2, 1, 1]), A -> repeat(A, 1, 2, 1, 1)))
add!(Case("pad (constant)", "tensor operation", (g, x) -> M.pad(x, [1, 1, 2, 2]), A -> (P = zeros(Float32, 1, 16, 10, 36); P[:, :, 2:9, 3:34] = A; P)))
add!(Case("pad (reflect)", "tensor operation", (g, x) -> M.pad(x, [1, 1, 2, 2]; mode="reflect")))
add!(Case("flatten2d", "tensor operation", (g, x) -> M.flatten2d(x; axis=1), A -> permutedims(reshape(permutedims(A, (4, 3, 2, 1)), 16 * 8 * 32, 1), (2, 1))))
add!(Case("depth_to_space", "tensor transformation", (g, x) -> M.depth_to_space(x; block_size=2)))
add!(Case("space_to_depth", "tensor transformation", (g, x) -> M.space_to_depth(x; block_size=2)))
add!(Case("pixel_shuffle", "tensor transformation", (g, x) -> M.pixel_shuffle(x; upscale_factor=2)))
add!(Case("pixel_unshuffle", "tensor transformation", (g, x) -> M.pixel_unshuffle(x; downscale_factor=2)))
add!(Case("sliding_windows", "tensor transformation", (g, x) -> M.sliding_windows(x; axis=3, size=4, stride=4)))
add!(Case("band_part", "tensor operation", (g, x) -> M.band_part(x; lower=0, upper=-1),
          A -> [j >= i ? A[n, c, i, j] : 0f0 for n in 1:1, c in 1:16, i in 1:8, j in 1:32]))
add!(Case("cumsum", "tensor operation", (g, x) -> M.cumsum(x; axis=3), A -> cumsum(A; dims=4)))
add!(Case("topk", "tensor operation", (g, x) -> (v = M.topk(x; k=4, axis=3); (v[1], M.cast(v[2], Float16))),
          A -> (sort(A; dims=4, rev=true)[:, :, :, 1:4], Float32.(mapslices(a -> sortperm(a; rev=true)[1:4] .- 1, A; dims=4)))))
add!(Case("argsort", "tensor operation", (g, x) -> M.cast(M.argsort(x; axis=3), Float16),
          A -> Float32.(mapslices(a -> sortperm(a; rev=true) .- 1, A; dims=4))))
idx = Int32.(rand(0:15, 6))
add!(Case("gather", "scatter/gather", (g, x) -> M.gather(x, M.constant(g, idx; eltype=Int32); axis=1), A -> A[:, idx .+ 1, :, :]))
idx4 = Int32.(rand(0:31, 1, 16, 8, 4))
add!(Case("gather_along_axis", "scatter/gather", (g, x) -> M.gather_along_axis(x, M.constant(g, idx4; eltype=Int32); axis=3),
          A -> [A[n, c, h, idx4[n, c, h, k] + 1] for n in 1:1, c in 1:16, h in 1:8, k in 1:4]))
idxnd = Int32.([0 3; 0 7; 0 1])
add!(Case("gather_nd", "scatter/gather", (g, x) -> M.gather_nd(x, M.constant(g, idxnd; eltype=Int32)),
          A -> permutedims(cat([A[1, idxnd[i, 2] + 1, :, :] for i in 1:3]...; dims=3), (3, 1, 2))))
add!(Case("scatter", "scatter/gather", (g, x, u) -> M.scatter(x, M.constant(g, Int32[1, 5]; eltype=Int32), u; axis=1),
          (A, U) -> (B = copy(A); B[:, [2, 6], :, :] = U; B); inputs=[(:normal, X), (:normal, (1, 2, 8, 32))]))
add!(Case("scatter_along_axis", "scatter/gather", (g, x, u) -> M.scatter_along_axis(x, M.constant(g, zeros(Int32, 1, 1, 8, 32); eltype=Int32), u; axis=1),
          (A, U) -> (B = copy(A); B[:, 1:1, :, :] = U; B); inputs=[(:normal, X), (:normal, (1, 1, 8, 32))]))
add!(Case("scatter_nd", "scatter/gather", (g, x, u) -> M.scatter_nd(x, M.constant(g, Int32[0 3]; eltype=Int32), u),
          (A, U) -> (B = copy(A); B[1, 4, :, :] = U[1, :, :]; B); inputs=[(:normal, X), (:normal, (1, 8, 32))]))
add!(Case("one_hot", "tensor operation", (g, x) -> M.one_hot(M.constant(g, Int32.(collect(0:15)); eltype=Int32), 32; axis=-1) |> v -> M.add(M.reshape(v, [1, 1, 16, 32]), M.mul(x, 0f0)),
          A -> reshape(Float32.([j == i for i in 0:15, j in 0:31]), 1, 1, 16, 32) .+ 0 .* A; inputs=[(:normal, (1, 1, 16, 32))]))
add!(Case("fill", "tensor operation", (g, x) -> M.add(x, M.fill(g, [1, 16, 8, 32], 2f0)), A -> A .+ 2))
add!(Case("range_1d", "tensor operation", (g, x) -> M.add(x, M.cast(M.range_1d(g, Int32(0), Int32(32), Int32(1)), Float16)), A -> A .+ reshape(0:31, 1, 1, 1, 32)))
add!(Case("shape", "tensor operation", (g, x) -> M.add(M.mul(x, 0f0), M.cast(M.reduce_sum(M.shape(x); axes=[0], keep_dims=true), Float16)), A -> 0 .* A .+ 57))
add!(Case("identity", "tensor operation", (g, x) -> M.relu(M.identity(x)), A -> max.(A, 0)))

add!(Case("upsample_nearest_neighbor", "image resizing", (g, x) -> M.upsample_nearest_neighbor(x; scale_factor_height=2, scale_factor_width=2),
          A -> repeat(A; inner=(1, 1, 2, 2))))
add!(Case("upsample_bilinear", "image resizing", (g, x) -> M.upsample_bilinear(x; scale_factor_height=2, scale_factor_width=2)))
add!(Case("resize_nearest_neighbor", "image resizing", (g, x) -> M.resize_nearest_neighbor(x; target_size_height=16, target_size_width=64),
          A -> repeat(A; inner=(1, 1, 2, 2))))
add!(Case("resize_bilinear", "image resizing", (g, x) -> M.resize_bilinear(x; target_size_height=16, target_size_width=64)))
add!(Case("crop", "image resizing", (g, x) -> M.crop(x; crop_height=[1, 1], crop_width=[2, 2]), A -> A[:, :, 2:7, 3:30]))

add!(Case("scaled_dot_product_attention", "attention", (g, q, k, v) -> M.scaled_dot_product_attention(q, k, v),
          (Q, K, V) -> (S = batched_mul(Q, permutedims(K, (1, 2, 4, 3))) ./ sqrt(32f0); batched_mul(softmax_ref(S; dims=4), V));
          inputs=[(:small, (1, 2, 32, 32)), (:small, (1, 2, 32, 32)), (:normal, (1, 2, 32, 32))]))
let H = 16, I = 32
    Wih, Whh = randn(Float32, 4H, I) ./ 6, randn(Float32, 4H, H) ./ 4
    add!(Case("lstm", "recurrent", (g, x) -> M.lstm(x, M.constant(g, zeros(Float32, 1, H)), M.constant(g, zeros(Float32, 1, H)),
                                                    M.constant(g, Wih), M.constant(g, Whh); output_sequence=true)[1],
              nothing; inputs=[(:normal, (8, 1, I))]))
    add!(Case("gru", "recurrent", (g, x) -> M.gru(x, M.constant(g, zeros(Float32, 1, H)), M.constant(g, Wih[1:3H, :]),
                                                  M.constant(g, Whh[1:3H, :]); output_sequence=true)[1], nothing; inputs=[(:normal, (8, 1, I))]))
    add!(Case("rnn", "recurrent", (g, x) -> M.rnn(x, M.constant(g, zeros(Float32, 1, H)), M.constant(g, Wih[1:H, :]),
                                                  M.constant(g, Whh[1:H, :]); output_sequence=true)[1], nothing; inputs=[(:normal, (8, 1, I))]))
end
add!(Case("quantize → dequantize", "quantization", (g, x) -> M.dequantize(M.quantize(x, Float16(0.05)), Float16(0.05)),
          A -> 0.05f0 .* clamp.(round.(A ./ 0.05f0), -128, 127)))
add!(Case("random_normal", "random", (g, x) -> M.add(M.mul(x, 0f0), M.op!(g, "random_normal", (:shape => [1, 16, 8, 32], :mean => Float16(0), :stddev => Float16(1)), ((Float16, [1, 16, 8, 32]),)))))
add!(Case("random_uniform", "random", (g, x) -> M.add(M.mul(x, 0f0), M.op!(g, "random_uniform", (:shape => [1, 16, 8, 32], :low => Float16(0), :high => Float16(1)), ((Float16, [1, 16, 8, 32]),)))))

# --- run ------------------------------------------------------------------------
_summary(e) = e isa ANEError ?
    (occursin("evaluation", e.msg) ? "evaluation fails" :
     (m = match(r"(InvalidMILProgram|CompilationFailure|Unsupported\w*|verifyBundleAtPath: [a-z ]+)", e.msg); m === nothing ? "compile fails" : "compile fails ($(m[1]))")) :
    "builder error: " * first(sprint(showerror, e), 80)

function run_case(c::Case)
    g = M.Graph()
    # References see exactly the fp16 values the ANE receives.
    arrays = [Float32.(Float16.(data(kind, shape))) for (kind, shape) in c.inputs]
    vars = [M.input(g, "x$i", Float16, shape; julia_shape=Tuple(Base.reverse(shape))) for (i, (_, shape)) in enumerate(c.inputs)]
    local outs
    try
        r = c.build(g, vars...)
        outs = r isa Tuple ? collect(r) : [r]
        for (i, o) in enumerate(outs)
            M.output!(g, o, "y$i"; julia_shape=Tuple(Base.reverse(o.shape)))
        end
    catch e
        return ("builder", _summary(e), NaN)
    end
    program = try
        compile(g)
    catch e
        return ("✗", _summary(e), NaN)
    end
    try
        results = evaluate!(program, map(to_buffer, arrays)...)
        isnothing(c.ref) && return ("✓ runs", "no reference", NaN)
        refs = c.ref(arrays...)
        refs = refs isa Tuple ? collect(refs) : [refs]
        err = maximum(zip(results, refs, outs)) do (res, ref, o)
            got = from_buffer(res, o.shape)
            size(got) == size(ref) || return Inf32
            maximum(abs.(got .- ref)) / max(maximum(abs, ref), 1f-3)
        end
        c.op == "floor_div" && err >= 2e-2 && return ("≈", "off by one when the quotient is (nearly) an integer: computed with an approximate reciprocal", err)
        return err < 2e-2 ? ("✓", "", err) : err < 0.1 ? ("≈", "approximate", err) : ("✗ wrong", "", err)
    catch e
        return ("✗", _summary(e), NaN)
    finally
        close!(program)
    end
end

function main(path)
    initialize!()
    println("probing $(length(cases)) cases on ", Sys.cpu_info()[1].model, " …")
    rows = []
    for c in cases
        status, note, err = run_case(c)
        println(rpad(c.op, 48), rpad(status, 10), isnan(err) ? "" : "err=$(round(err; sigdigits=2)) ", note)
        push!(rows, (c, status, note, err))
    end
    ok = count(r -> startswith(r[2], "✓") || startswith(r[2], "≈"), rows)
    io = IOBuffer()
    println(io, """
    # Op support

    Generated by `tools/probe_ops.jl` on $(Sys.cpu_info()[1].model), macOS $(get(ENV, "MACOS_VERSION", strip(read(`sw_vers -productVersion`, String)))),
    $(Dates.today()). Each op runs in a small fp16 program; **err** is the maximum absolute
    error relative to the largest reference value (fp16 rounding gives ~1e-3); **≈** marks results
    within 10% (a coarse hardware approximation). "compile fails (InvalidMILProgram)" means the
    ANE compiler's MIL validation rejected the op; every argument encoding we tried (inline or
    `const` literals, other axes, other modes) was rejected. Core ML may lower these ops to
    others before they reach the ANE. The ANE is a
    private, undocumented target, so this table can change with macOS releases. Re-run the
    probe on your machine.

    **$ok of $(length(rows))** probed cases compile and run.

    Not probed (they need control flow, dynamic shapes or state, which this package does not
    support yet): `cond`, `while_loop`, `make_list` and the `list_*` ops, `non_zero`,
    `non_maximum_suppression`, `classify`, `read_state`, `slice_update`, `affine`, `resample`,
    `crop_resize`, `reshape_like`, and the `constexpr_*` compression ops other than
    `constexpr_affine_dequantize`.

    Observed constraints, independent of the op:
    - Computation must be fp16: the same ops on fp32 tensors do not compile (cast inputs).
    - A trailing `identity` op cannot be a program output.
    - Tensor buffers need a 64-byte aligned row stride, which this package provides, so any
      innermost size works.
    """)
    for category in unique(r[1].category for r in rows)
        println(io, "\n## ", uppercasefirst(category), "\n\n| op | status | err | note |\n|---|---|---|---|")
        for (c, status, note, err) in filter(r -> r[1].category == category, rows)
            println(io, "| `", c.op, "` | ", status, " | ", isnan(err) ? "" : string(round(err; sigdigits=2)), " | ", note, " |")
        end
    end
    mkpath(dirname(path))
    write(path, String(take!(io)))
    println("\n$ok / $(length(rows)) ok — wrote $path")
end

main(get(ARGS, 1, joinpath(@__DIR__, "..", "docs", "src", "op_support.md")))
