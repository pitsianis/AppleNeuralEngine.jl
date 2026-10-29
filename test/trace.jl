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
