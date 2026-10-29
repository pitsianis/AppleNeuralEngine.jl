using AppleNeuralEngine
using LinearAlgebra
using Test

include("mil.jl")

const ON_APPLE_SILICON = Sys.isapple() && Sys.ARCH === :aarch64

# The ANE tests need the private runtime; allow skipping them explicitly
# (e.g. on CI runners without an ANE) with APPLENEURALENGINE_SKIP_ANE=1.
const ANE_AVAILABLE = get(ENV, "APPLENEURALENGINE_SKIP_ANE", "0") != "1" && functional()
ANE_AVAILABLE || @warn "No usable Neural Engine (or APPLENEURALENGINE_SKIP_ANE=1); skipping hardware tests"

# fp16 compute: compare in a norm sense rather than elementwise.
relerr(a, b) = norm(a .- b) / max(norm(b), eps(Float32))

@testset "AppleNeuralEngine" begin

@testset "device" begin
    @test functional() isa Bool
    if ON_APPLE_SILICON
        info = device_info()
        @test info.has_ane isa Bool && info.num_cores isa Int
        @test functional() == (info.has_ane && !info.virtual_machine && info.num_cores > 0)
    end
end

@testset "shape validation" begin
    @test AppleNeuralEngine._shapes([(2, 3), [4]], "inputs") == [[2, 3], [4]]
    @test AppleNeuralEngine._shapes(((1, 2, 3),), "inputs") == [[1, 2, 3]]
    @test_throws ArgumentError AppleNeuralEngine._shapes([], "inputs")
    @test_throws ArgumentError AppleNeuralEngine._shapes([()], "inputs")
    @test_throws ArgumentError AppleNeuralEngine._shapes([(0, 3)], "inputs")
    @test_throws ArgumentError AppleNeuralEngine._shapes([(2, -1)], "inputs")
    @test_throws ArgumentError AppleNeuralEngine._shapes([(typemax(Int) ÷ 2,)], "inputs")
    @test_throws ArgumentError AppleNeuralEngine._shapes([(2^32, 2^32)], "inputs")  # prod wraps to 0
    @test AppleNeuralEngine._layout([64, 256]) == (64 * 4, 256)
    @test AppleNeuralEngine._layout([64, 256], Float16) == (64 * 2, 256)
    @test AppleNeuralEngine._layout([3]) == (12, 1)
end

@testset "weight blob" begin
    W = Float32[1 2 3; 4 5 6]
    blob = build_weight_blob(W)
    @test blob isa Vector{UInt8}
    @test length(blob) == 128 + 2 * length(W)

    # Header layout expected by the ANE BLOBFILE loader.
    @test blob[1] == 0x01 && blob[5] == 0x02
    @test blob[65:68] == [0xEF, 0xBE, 0xAD, 0xDE]
    @test blob[69] == 0x01
    @test reinterpret(UInt32, blob[73:76])[1] == 2 * length(W)
    @test reinterpret(UInt32, blob[81:84])[1] == 128

    # Payload is fp16 in row-major order (W[i, :] contiguous).
    @test reinterpret(Float16, blob[129:end]) == Float16[1, 2, 3, 4, 5, 6]

    # Transposed blob stores W' row-major, i.e. W column-major.
    tblob = build_weight_blob(W; transposed=true)
    @test length(tblob) == length(blob)
    @test tblob[1:128] == blob[1:128]
    @test reinterpret(Float16, tblob[129:end]) == Float16[1, 4, 2, 5, 3, 6]

    # Rounding to fp16 and non-square/large matrices.
    R = randn(Float32, 37, 129)
    rblob = build_weight_blob(R)
    @test reinterpret(Float16, rblob[129:end]) == vec(Float16.(permutedims(R)))
    @test reinterpret(Float16, build_weight_blob(R; transposed=true)[129:end]) == vec(Float16.(R))

    # Accepts any AbstractMatrix{Float32}, rejects other element types.
    @test build_weight_blob(view(R, 1:2, 1:3)) == build_weight_blob(R[1:2, 1:3])
    @test build_weight_blob(Float32[7;;]) |> length == 130
    @test_throws MethodError build_weight_blob(Float64.(W))
end

@testset "pack_weights" begin
    A = randn(Float32, 3, 5)      # 30 bytes of fp16: padded to 64
    B = randn(Float32, 4, 16)     # 128 bytes: already aligned
    g = randn(Float32, 1, 7)
    blob, offsets = pack_weights(A, B, g)
    @test offsets == [64, 64 + 64 + 64, 64 + (64 + 64) + (64 + 128)]
    @test reinterpret(UInt32, blob[1:8]) == [3, 2]            # count, version
    for (W, off) in zip((A, B, g), offsets)
        meta = blob[off+1:off+64]
        @test meta[1:4] == [0xEF, 0xBE, 0xAD, 0xDE]
        @test reinterpret(UInt32, meta[5:8])[1] == 1          # fp16
        nbytes, data = reinterpret(UInt64, meta[9:24])
        @test nbytes == 2 * length(W)
        @test data == off + 64
        @test reinterpret(Float16, blob[data+1:data+nbytes]) == vec(Float16.(permutedims(W)))
    end
    @test length(blob) == offsets[end] + 64 + 2 * length(g)   # no trailing padding

    tblob, toffsets = pack_weights(A, B; transposed=true)
    @test reinterpret(Float16, tblob[toffsets[1]+65:toffsets[1]+64+30]) == vec(Float16.(A))

    @test pack_weights(A)[1] == build_weight_blob(A)
    @test pack_weights(A)[2] == [64]
    @test_throws ArgumentError pack_weights()
end

if ANE_AVAILABLE
@testset "ANE execution" begin
    @testset "1x1 convolution matches matmul" begin
        C, S = 128, 32
        W = randn(Float32, C, C) ./ sqrt(Float32(C))
        program = compile(conv_mil(C, S);
            input_shapes=[(S, C)], output_shapes=[(S, C)],
            weight_blob=build_weight_blob(W))
        try
            @test isopen(program)
            @test program.input_shapes == [[S, C]]
            @test program.output_shapes == [[S, C]]

            x = randn(Float32, S, C)
            y, = evaluate!(program, x)
            @test y isa Matrix{Float32}
            @test size(y) == (S, C)
            # Julia (S, C) memory is the MIL [1, C, 1, S] layout, so y = x * W'.
            @test relerr(y, x * W') < 1e-2

            # Program is reusable and evaluations are independent.
            x2 = randn(Float32, S, C)
            y2, = evaluate!(program, x2)
            @test relerr(y2, x2 * W') < 1e-2
            @test evaluate!(program, x)[1] == y
            @test all(iszero, evaluate!(program, zeros(Float32, S, C))[1])
        finally
            close!(program)
        end
    end

    @testset "transposed weight blob" begin
        C, S = 64, 16
        W = randn(Float32, C, C) ./ sqrt(Float32(C))
        program = compile(conv_mil(C, S);
            input_shapes=[(S, C)], output_shapes=[(S, C)],
            weight_blob=build_weight_blob(W; transposed=true))
        try
            x = randn(Float32, S, C)
            @test relerr(evaluate!(program, x)[1], x * W) < 1e-2
        finally
            close!(program)
        end
    end

    @testset "several tensors in one weight file" begin
        C, S = 64, 16
        W1 = randn(Float32, C, C) ./ sqrt(Float32(C))
        W2 = randn(Float32, C, C) ./ sqrt(Float32(C))
        blob, (o1, o2) = pack_weights(W1, W2)
        program = compile(two_conv_mil(C, S, o1, o2);
            input_shapes=[(S, C)], output_shapes=[(S, C)], weight_blob=blob)
        try
            x = randn(Float32, S, C)
            @test relerr(evaluate!(program, x)[1], max.(x * W1', 0) * W2') < 1e-2
        finally
            close!(program)
        end
    end

    @testset "the example from the README" begin
        C, S = 256, 64
        program = compile(conv_mil(C, S);
            input_shapes=[(S, C)], output_shapes=[(S, C)],
            weight_blob=build_weight_blob(ones(Float32, C, C)))
        try
            @test all(==(256f0), evaluate!(program, ones(Float32, S, C))[1])
        finally
            close!(program)
        end
    end

    @testset "multiple inputs, no weights" begin
        C, S = 64, 32
        for blob in (nothing, UInt8[])   # an empty blob means "no weights"
            program = compile(sub_mil(C, S);
                input_shapes=[(S, C), (S, C)], output_shapes=[(S, C)],
                weight_blob=blob)
            try
                x, y = randn(Float32, S, C), randn(Float32, S, C)
                z, = evaluate!(program, x, y)
                @test maximum(abs.(z .- (x .- y))) < 1e-2
            finally
                close!(program)
            end
        end
    end

    @testset "I/O tensors are bound in name order" begin
        # Observed runtime behaviour: inputs and outputs map to buffer indices
        # by sorted tensor name, not by declaration order. MIL authors must
        # declare/order them accordingly. This test documents that contract.
        C, S = 64, 16
        program = compile(unsorted_io_mil(C, S);
            input_shapes=[(S, C), (S, C)], output_shapes=[(S, C), (S, C)])
        try
            a, b = randn(Float32, S, C), randn(Float32, S, C)
            p, q = evaluate!(program, a, b)
            @test maximum(abs.(p .- max.(b, 0))) < 1e-2   # p = relu(b)
            @test maximum(abs.(q .- (b .- a))) < 1e-2     # q = b - a
        finally
            close!(program)
        end
    end

    @testset "evaluate! argument checks" begin
        C, S = 64, 16
        program = compile(sub_mil(C, S);
            input_shapes=[(S, C), (S, C)], output_shapes=[(S, C)])
        x = randn(Float32, S, C)
        @test_throws DimensionMismatch evaluate!(program, x)
        @test_throws DimensionMismatch evaluate!(program, x, x, x)
        @test_throws DimensionMismatch evaluate!(program, x, randn(Float32, C, S))
        @test_throws DimensionMismatch evaluate!(program, x, vec(x))
        @test_throws ArgumentError evaluate!(program, x, Float64.(x))
        @test_throws ArgumentError evaluate!(program, x, view(randn(Float32, S + 1, C), 1:S, :))

        dir = program.model_dir
        @test isdir(dir)
        @test !occursin("closed", sprint(show, program))
        close!(program)
        @test !isopen(program)
        @test isempty(program.input_surfaces)
        @test !isdir(dir)
        @test occursin("closed", sprint(show, program))
        @test_throws ErrorException evaluate!(program, x, x)
        close!(program)   # idempotent
        @test !isopen(program)
    end

    @testset "compile argument checks" begin
        @test_throws ArgumentError compile(""; input_shapes=[(4,)], output_shapes=[(4,)])
        @test_throws ArgumentError compile(sub_mil(4, 4); input_shapes=[], output_shapes=[(4, 4)])
        @test_throws ArgumentError compile(sub_mil(4, 4); input_shapes=[(4, 4)], output_shapes=[])
        @test_throws ArgumentError compile(sub_mil(4, 4); input_shapes=[(0, 4)], output_shapes=[(4, 4)])
        @test_throws UndefKeywordError compile(sub_mil(4, 4); output_shapes=[(4, 4)])
        # Invalid MIL is reported by the ANE compiler, with its diagnostic.
        err = try
            compile("this is not MIL"; input_shapes=[(4,)], output_shapes=[(4,)])
        catch e
            e
        end
        @test err isa ANEError
        @test occursin("InvalidMILProgram", sprint(showerror, err))
    end

    @testset "repeated compile / close" begin
        C, S = 64, 16
        mil = sub_mil(C, S)
        x, y = randn(Float32, S, C), randn(Float32, S, C)
        for _ in 1:10
            program = compile(mil; input_shapes=[(S, C), (S, C)], output_shapes=[(S, C)])
            @test maximum(abs.(evaluate!(program, x, y)[1] .- (x .- y))) < 1e-2
            close!(program)
        end
        # Programs left to the finalizer must not crash.
        for _ in 1:5
            compile(mil; input_shapes=[(S, C), (S, C)], output_shapes=[(S, C)])
        end
        GC.gc(true)
        @test true
    end

    @testset "initialize! is idempotent" begin
        @test initialize!() === nothing
        @test initialize!() === nothing
    end
end
else
    @info "Skipping ANE execution tests (no ANE runtime available)"
end

include("builder.jl")
include("trace.jl")

end
