const M = AppleNeuralEngine.MIL

@testset "MIL builder (no hardware)" begin
    g = M.Graph()
    x = M.input(g, :x, Float32, (1, 64, 1, 32))
    @test size(x) == (1, 64, 1, 32)
    @test M.julia_shape(x) == (32, 64)
    h = M.cast(x, Float16)
    @test eltype(h) == Float16

    # shape inference
    W = M.constant(g, randn(Float32, 128, 64, 1, 1))
    c = M.conv(h, W)
    @test size(c) == (1, 128, 1, 32)
    @test size(M.conv(h, M.constant(g, randn(Float32, 8, 64, 3, 3)); pad_type="same")) == (1, 8, 1, 32)
    @test size(M.reduce_mean(c; axes=[1], keep_dims=true)) == (1, 1, 1, 32)
    @test size(M.reduce_sum(c; axes=[1, 3])) == (1, 1)
    @test size(M.add(c, M.reduce_max(c; axes=[3], keep_dims=true))) == (1, 128, 1, 32)
    @test_throws DimensionMismatch M.add(c, h)
    @test size(M.matmul(M.reshape(c, [128, 32]), M.reshape(c, [128, 32]); transpose_x=true)) == (32, 32)
    @test size(M.transpose(c, [0, 3, 2, 1])) == (1, 32, 1, 128)
    @test size(M.reshape(c, [-1, 32])) == (128, 32)
    a, b = M.split(c; axis=1, num_splits=2)
    @test size(a) == size(b) == (1, 64, 1, 32)
    @test size(M.concat([a, b, a]; axis=1)) == (1, 192, 1, 32)
    @test size(M.slice_by_index(c, [0, 0, 0, 4], [1, 64, 1, 20])) == (1, 64, 1, 16)
    @test eltype(M.greater(c, 0f0)) == Bool
    v, i = M.topk(c; k=3, axis=1)
    @test size(v) == size(i) == (1, 3, 1, 32) && eltype(i) == Int32
    @test eltype(M.topk(c; k=3, axis=1, indices_dtype=UInt16)[2]) == UInt16
    @test M.topk(c; k=3, axis=1, return_indices=false) isa M.Var
    @test_throws ArgumentError M.topk(c; indices_dtype=Int16)
    am = M.reduce_argmax(c; axis=1, keep_dims=true, output_dtype=UInt16)
    @test size(am) == (1, 1, 1, 32) && eltype(am) == UInt16
    @test eltype(M.reduce_argmin(c; axis=1)) == Int32
    gidx = M.constant(g, [0, 2]; eltype=UInt16)
    @test size(M.gather(c, gidx; axis=1)) == (1, 2, 1, 32)
    @test occursin("validate_indices", last(g.body))
    g16 = M.Graph(; opset="ios16"); x16 = M.input(g16, :x, Float16, (4, 8))
    M.gather(x16, M.constant(g16, [1]; eltype=UInt16))
    @test !occursin("validate_indices", last(g16.body))      # iOS17+ argument
    @test size(M.einsum("nchw,nwhu->nchu", M.reshape(c, [1, 4, 32, 32]), M.reshape(c, [1, 32, 32, 4]))) == (1, 4, 32, 4)

    # printing
    y = M.output!(g, M.cast(c, Float32), :y)
    src = M.source(g)
    @test occursin("func main<ios18>(tensor<fp32, [1, 64, 1, 32]> in000_x)", src)
    @test occursin("} -> (out000_y);", src)
    @test occursin("out000_y = cast(", src)          # the producing variable is renamed
    @test !occursin(r"\bconst\b = ", src)            # no reserved words as names
    @test M.weight_blob(g) isa Vector{UInt8}
    # renaming an output must not touch an argument with the same name (`pad = …` of a `pad` op)
    gp = M.Graph(); xp = M.input(gp, :x, Float16, (1, 2, 4, 4))
    M.output!(gp, M.pad(xp, [1, 1, 1, 1]), :y)
    @test occursin(r"out000_y = pad\(x = in000_x, pad = pad_pad", M.source(gp))

    @test_throws ArgumentError M.output!(g, x)        # an input cannot be an output
    @test_throws ArgumentError M.output!(g, M.greater(c, 0f0))
    @test_throws ArgumentError M.input(g, :late, Float32, (1, 4))   # inputs come first
    @test_throws ArgumentError M.source(M.Graph())
    @test_throws ArgumentError M.add(c, M.cast(M.input(M.Graph(), :z, Float32, (1,)), Float16))
end

if ANE_AVAILABLE
@testset "MIL builder on the ANE" begin
    # fp16 I/O, odd innermost sizes (needs the padded row stride), several outputs
    for S in (1, 7, 32)
        g = M.Graph()
        x = M.input(g, :x, Float16, (1, 8, 1, S))
        y = M.input(g, :y, Float32, (1, 8, 1, S))
        s = M.add(x, M.cast(y, Float16))
        M.output!(g, M.relu(s), :r)
        M.output!(g, M.cast(M.mul(s, 2f0), Float32), :d)
        p = compile(g)
        a, b = randn(Float16, M.julia_shape(x)...), randn(Float32, M.julia_shape(y)...)
        r, d = p(a, b)
        @test r isa Array{Float16} && d isa Array{Float32} && size(r) == M.julia_shape(x)
        @test r ≈ max.(a .+ Float16.(b), 0) rtol = 1e-2 atol = 1e-2
        @test d ≈ 2 .* (Float32.(a) .+ b) rtol = 1e-2 atol = 1e-2
        close!(p)
    end

    # pad (constant and reflect) on the ANE
    for mode in ("constant", "reflect")
        g = M.Graph()
        x = M.input(g, :x, Float16, (1, 2, 4, 6))
        M.output!(g, M.pad(x, [1, 1, 2, 2]; mode), :y)
        p = compile(g)
        x0 = randn(Float16, 6, 4, 2)                        # Julia (W, H, C)
        y = p(x0)
        ref = similar(x0, 10, 6, 2)
        for c in 1:2, h in 1:6, w in 1:10
            hh, ww = h - 1, w - 2
            r(i, n) = i < 1 ? 2 - i : i > n ? 2n - i : i
            ref[w, h, c] = mode == "constant" ? (1 <= hh <= 4 && 1 <= ww <= 6 ? x0[ww, hh, c] : 0) : x0[r(ww, 6), r(hh, 4), c]
        end
        @test y == ref
        close!(p)
    end

    # int8 weights dequantized on load
    g = M.Graph()
    x = M.input(g, :x, Float16, (1, 64, 1, 32))
    W = randn(Float32, 32, 64, 1, 1)
    M.output!(g, M.conv(x, M.constant_int8(g, W)), :y)
    p = compile(g)
    x0 = randn(Float16, 32, 64)
    y = p(x0)
    @test relerr(Float32.(y), Float32.(x0) * dropdims(W; dims=(3, 4))') < 2e-2
    close!(p)

    # argmax with uint16 indices: the ANE returns fp16 bit patterns labelled uint16
    g = M.Graph()
    x = M.input(g, :x, Float16, (1, 16, 1, 32))
    M.output!(g, M.cast(M.reduce_argmax(x; axis=1, keep_dims=true, output_dtype=UInt16), Float16), :i)
    p = compile(g)
    x0 = randn(Float16, 32, 16)
    i = Int.(reinterpret.(Float16, UInt16.(p(x0))))
    @test vec(i) == [argmax(x0[w, :]) - 1 for w in 1:32]
    close!(p)

    # gather with indices computed on the ANE (an embedding lookup)
    g = M.Graph()
    ids = M.input(g, :ids, Float16, (16,))
    E = randn(Float32, 300, 64)
    M.output!(g, M.gather(M.constant(g, E), M.cast(ids, UInt16); axis=0), :e)
    p = compile(g)
    t = rand(0:299, 16)
    @test p(Float16.(t)) == permutedims(Float16.(E[t .+ 1, :]))   # MIL [16, 64] is a Julia (64, 16)
    close!(p)
end
end
