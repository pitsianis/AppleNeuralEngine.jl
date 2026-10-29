# Benchmark the Neural Engine against the CPU (OpenBLAS and Apple Accelerate)
# and print a Markdown table to paste into the Discourse thread or an issue.
#
#     julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
#     julia --project=benchmark benchmark/run.jl

using AppleNeuralEngine, LinearAlgebra, Printf

# Minimum over repetitions after a warm-up call: the steady-state cost per call.
besttime(f, x; n=100) = (f(x); minimum(@elapsed(f(x)) for _ in 1:n))

# --- workloads: each is the same Julia function on the CPU and the ANE ----------
silu(z) = z / (1 + exp(-z))

function ffn_workload(D=512, H=1536, T=256)
    W1, W3 = randn(Float32, H, D) ./ sqrt(D), randn(Float32, H, D) ./ sqrt(D)
    W2 = randn(Float32, D, H) ./ sqrt(H)
    g = 1 .+ 0.1f0 .* randn(Float32, D)
    f(x) = (n = x ./ sqrt.(sum(abs2, x; dims=2) ./ size(x, 2) .+ 1f-5) .* g';
            x .+ (silu.(n * W1') .* (n * W3')) * W2')
    return "FFN block D=$D H=$H T=$T", f, (T, D), 2 * T * D * H * 3
end

function gemm_workload(T, D, H)
    W = randn(Float32, D, H) ./ sqrt(D)
    return "x*W $(T)×$(D) · $(D)×$(H)", x -> x * W, (T, D), 2 * T * D * H
end

workloads = [ffn_workload(), gemm_workload(256, 1024, 1024), gemm_workload(1024, 1024, 1024),
             gemm_workload(2048, 2048, 2048)]

# --- run ------------------------------------------------------------------------
functional() || error("No usable Neural Engine on this machine (device_info: $(device_info()))")

results = map(workloads) do (name, f, dims, flops)
    x16 = randn(Float16, dims...)
    x32 = Float32.(x16)
    program = compile(f, x16)
    y, ref = Float32.(program(x16)), f(x32)
    err = sqrt(sum(abs2, y .- ref) / sum(abs2, ref))
    t_ane = besttime(program, x16)
    close!(program)
    (; name, flops, err, t_ane, t_cpu=besttime(f, x32))
end

openblas = [r.t_cpu for r in results]
using AppleAccelerate           # switches BLAS to Accelerate from here on
accelerate = [besttime(w[2], randn(Float32, w[3]...)) for w in workloads]

gflops(flops, t) = @sprintf("%.0f", flops / t / 1e9)
ms(t) = @sprintf("%.3f", 1e3t)
info = device_info()
macos = strip(read(`sw_vers -productVersion`, String))

println("\n**", Sys.cpu_info()[1].model, "**, ANE: ", info.num_cores, " cores (", info.architecture,
        "), macOS ", macos, ", Julia ", VERSION, ", ", BLAS.get_num_threads(), " BLAS threads\n")
println("| workload | ANE fp16 ms | GFLOP/s | Accelerate fp32 ms | GFLOP/s | OpenBLAS fp32 ms | GFLOP/s | rel. error |")
println("|---|--:|--:|--:|--:|--:|--:|--:|")
for (r, ta, to) in zip(results, accelerate, openblas)
    println("| ", r.name, " | ", ms(r.t_ane), " | ", gflops(r.flops, r.t_ane), " | ", ms(ta), " | ", gflops(r.flops, ta),
            " | ", ms(to), " | ", gflops(r.flops, to), " | ", @sprintf("%.1e", r.err), " |")
end
println("\nANE times include copying inputs and outputs. CPU times are Float32.")
