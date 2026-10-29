# A transformer feed-forward block, as used in LLaMA-style models, on the
# Apple Neural Engine:
#
#     n = RMSNorm(x) .* g
#     y = x + W2 * (silu(W1 * n) .* (W3 * n))
#
# The block is built with the MIL graph builder, compiled once, checked against
# the same computation in plain Julia, and timed.
#
#     julia --project=. examples/transformer_ffn.jl

using AppleNeuralEngine
using Random

const D = 512    # model dimension
const H = 1536   # hidden dimension (3D)
const T = 256    # tokens processed per call
const EPS = 1f-5

# --- the block on the ANE -----------------------------------------------------
# Activations are MIL [1, channels, 1, tokens] tensors, the layout the ANE is
# fastest with. The program's Julia input is then a (T, D) matrix: row t is token t.
# Inputs and outputs are Float16 here, which halves the data copied per call.

function ffn_graph(g_gain, W1, W2, W3)
    g = MIL.Graph()
    x = MIL.input(g, :x, Float16, (1, D, 1, T))
    # RMSNorm over channels, then the learned per-channel gain.
    ms = MIL.reduce_mean(MIL.mul(x, x); axes=[1], keep_dims=true)
    n = MIL.mul(MIL.mul(x, MIL.rsqrt(ms; epsilon=EPS)), MIL.constant(g, reshape(g_gain, 1, D, 1, 1)))
    # Linear layers are 1x1 convolutions with [out, in, 1, 1] weights.
    weight(W) = MIL.constant(g, reshape(W, size(W)..., 1, 1))
    h = MIL.mul(MIL.silu(MIL.conv(n, weight(W1))), MIL.conv(n, weight(W3)))
    MIL.output!(g, MIL.add(x, MIL.conv(h, weight(W2))), :y)
    return g
end

# --- the same block in plain Julia -------------------------------------------
silu(z) = z / (1 + exp(-z))

function ffn_cpu(x, g, W1, W2, W3)
    n = x ./ sqrt.(sum(abs2, x; dims=2) ./ size(x, 2) .+ EPS) .* g'
    return x .+ (silu.(n * W1') .* (n * W3')) * W2'
end

# --- weights ------------------------------------------------------------------
Random.seed!(1)
W1 = randn(Float32, H, D) ./ sqrt(Float32(D))
W3 = randn(Float32, H, D) ./ sqrt(Float32(D))
W2 = randn(Float32, D, H) ./ sqrt(Float32(H))
g = 1 .+ 0.1f0 .* randn(Float32, D)

# --- compile, check, time -----------------------------------------------------
graph = ffn_graph(g, W1, W2, W3)
t_compile = @elapsed program = compile(graph)
println(program, "\ncompiled in ", round(t_compile; digits=2), " s")

relerr(a, b) = sqrt(sum(abs2, a .- b) / sum(abs2, b))

try
    x = randn(Float32, T, D)
    y_ane, = evaluate!(program, Float16.(x))
    y_cpu = ffn_cpu(x, g, W1, W2, W3)
    err = relerr(Float32.(y_ane), y_cpu)
    println("relative error vs Float32 CPU: ", round(err; sigdigits=3), " (fp16 compute)")
    err < 1e-2 || error("ANE result does not match the CPU reference")

    x16 = Float16.(x)
    flops = 2 * T * D * H * 3
    time_ane = minimum(@elapsed(evaluate!(program, x16)) for _ in 1:50)
    time_cpu = minimum(@elapsed(ffn_cpu(x, g, W1, W2, W3)) for _ in 1:50)
    for (name, t) in (("ANE (incl. copies)", time_ane), ("CPU Float32", time_cpu))
        println(rpad(name, 20), lpad(round(1e3t; digits=3), 8), " ms  ",
                lpad(round(flops / t / 1e9; digits=1), 7), " GFLOP/s")
    end
finally
    close!(program)
end
