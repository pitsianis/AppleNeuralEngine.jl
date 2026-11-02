@testset "concurrent traces" begin
    # Each traced function yields mid-trace, so the two tasks interleave even on one thread.
    W1, W2 = randn(Float32, 32, 48), randn(Float32, 32, 16)
    f1(x) = (yield(); y = x * W1; yield(); max.(y, 0))
    f2(x) = (yield(); y = x * W2; yield(); y .+ 1)
    t1 = Threads.@spawn trace(f1, (8, 32))
    t2 = Threads.@spawn trace(f2, (8, 32))
    g1, g2 = fetch(t1), fetch(t2)
    @test only(g1.outputs).shape == [1, 48, 1, 8]
    @test only(g2.outputs).shape == [1, 16, 1, 8]
    s1, s2 = AppleNeuralEngine.MIL.source(g1), AppleNeuralEngine.MIL.source(g2)
    @test occursin("[48, 32, 1, 1]", s1) && !occursin("[16, 32, 1, 1]", s1)   # each graph holds only its own weight
    @test occursin("[16, 32, 1, 1]", s2) && !occursin("[48, 32, 1, 1]", s2)
end

if ANE_AVAILABLE
@testset "tracing" begin
    W = randn(Float32, 64, 96) ./ 8
    v = randn(Float32, 64)
    silu(z) = z / (1 + exp(-z))
    cases = [
        ("broadcast + constant matmul", x -> silu.(x * W) .+ 1, (randn(Float16, 32, 64),)),
        ("row softmax", x -> (e = exp.(x .- maximum(x; dims=2)); e ./ sum(e; dims=2)), (randn(Float16, 16, 64),)),
        ("reductions", x -> sum(abs2, x; dims=1) .+ minimum(x; dims=1), (randn(Float16, 16, 64),)),
        ("ifelse / comparisons", x -> ifelse.(x .> 0, x, 0.1f0 .* x), (randn(Float16, 16, 64),)),
        ("traced * traced", (a, b) -> a * b, (randn(Float16, 16, 32), randn(Float16, 32, 48))),
        ("matrix * vector", x -> W' * x, (randn(Float16, 64),)),
        ("adjoint, Gram", x -> x' * x, (randn(Float16, 32, 16),)),
        ("vector broadcast", x -> x .* v' .+ v', (randn(Float16, 8, 64),)),
        ("Float32 I/O", x -> tanh.(x), (randn(Float32, 16, 32),)),
        ("two Float32 inputs", (a, b) -> a .- b, (randn(Float32, 16, 32), randn(Float32, 16, 32))),
    ]
    for (label, f, args) in cases
        @testset "$label" begin
            p = compile(f, args...)
            out = p(args...)
            @test eltype(out) == eltype(first(args))
            @test size(out) == size(f(map(a -> Float32.(a), args)...))
            @test relerr(Float32.(out), f(map(a -> Float32.(a), args)...)) < 1e-2
            close!(p)
        end
    end
    p = compile(x -> (x .+ 1, x .* 2), randn(Float16, 8, 32))
    a, b = p(ones(Float16, 8, 32))
    @test all(==(2), a) && all(==(2), b)
    close!(p)

    @test_throws DimensionMismatch trace(x -> x * W, (16, 32))
    @test_throws ArgumentError trace(x -> 1, (16, 32))
    @test_throws ArgumentError trace(x -> x, (2, 2, 2))
end
end
