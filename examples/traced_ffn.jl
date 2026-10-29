# The same Julia function on the CPU and, traced, on the Apple Neural Engine:
# a LLaMA-style feed-forward block (RMSNorm, SwiGLU, residual).
#
#     julia --project=. examples/traced_ffn.jl

using AppleNeuralEngine

D, H, T = 512, 1536, 256          # model dim, hidden dim, tokens
W1, W3 = randn(Float32, H, D) ./ sqrt(Float32(D)), randn(Float32, H, D) ./ sqrt(Float32(D))
W2 = randn(Float32, D, H) ./ sqrt(Float32(H))
g = 1 .+ 0.1f0 .* randn(Float32, D)

silu(z) = z / (1 + exp(-z))
function ffn(x)
    n = x ./ sqrt.(sum(abs2, x; dims=2) ./ size(x, 2) .+ 1f-5) .* g'
    return x .+ (silu.(n * W1') .* (n * W3')) * W2'
end

x = randn(Float16, T, D)
ane = compile(ffn, x)             # trace + compile for the Neural Engine
y = ane(x)                        # run on the ANE

y_cpu = ffn(Float32.(x))
println("relative error: ", sqrt(sum(abs2, Float32.(y) .- y_cpu) / sum(abs2, y_cpu)))

x32 = Float32.(x)
t_ane = minimum(@elapsed(ane(x)) for _ in 1:100)
t_cpu = minimum(@elapsed(ffn(x32)) for _ in 1:100)
gflop = 2 * T * D * H * 3 / 1e9
println("ANE (fp16):  ", round(1e3t_ane; digits=3), " ms  ", round(Int, gflop / t_ane), " GFLOP/s")
println("CPU (fp32):  ", round(1e3t_cpu; digits=3), " ms  ", round(Int, gflop / t_cpu), " GFLOP/s")
println(device_info())
