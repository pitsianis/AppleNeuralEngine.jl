# Benchmark the Neural Engine against the CPU (OpenBLAS and Apple Accelerate)
# and print a Markdown table to paste into the Discourse thread or an issue.
#
#     julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
#     julia --project=benchmark benchmark/run.jl
#
# BENCH_METHOD selects the timing method:
#   loop (default)    100 back-to-back calls after a warm-up; about 12 s in all.
#                     Measures short bursts.
#   benchmarktools    BenchmarkTools, BENCH_SECONDS seconds (default 2) per
#                     measurement; about a minute in all. Mostly measures sustained
#                     load, under which the ANE runs about 50% slower.

using AppleNeuralEngine, BenchmarkTools, LinearAlgebra, Printf, Statistics

const METHOD = get(ENV, "BENCH_METHOD", "loop")
METHOD in ("benchmarktools", "loop") || error("BENCH_METHOD must be benchmarktools or loop, got $METHOD")
const SECONDS = parse(Float64, get(ENV, "BENCH_SECONDS", "2"))

# Minimum and median time per call in seconds. The minimum is the steady-state cost;
# the median shows how much it varies.
function timing(f, x)
    f(x)                                    # compile and warm up
    if METHOD == "loop"
        ts = [@elapsed(f(x)) for _ in 1:100]
        return (; min=minimum(ts), med=median(ts))
    end
    trial = @benchmark $f($x) seconds = SECONDS samples = 100_000
    return (; min=minimum(trial).time / 1e9, med=median(trial).time / 1e9)
end

# --- workloads: each is the same Julia function on the CPU and the ANE ----------
silu(z) = z / (1 + exp(-z))

function ffn_workload(D=512, H=1536, T=256)
    W1, W3 = randn(Float32, H, D) ./ sqrt(Float32(D)), randn(Float32, H, D) ./ sqrt(Float32(D))
    W2 = randn(Float32, D, H) ./ sqrt(Float32(H))
    g = 1 .+ 0.1f0 .* randn(Float32, D)
    f(x) = (n = x ./ sqrt.(sum(abs2, x; dims=2) ./ size(x, 2) .+ 1f-5) .* g';
            x .+ (silu.(n * W1') .* (n * W3')) * W2')
    return "FFN block D=$D H=$H T=$T", f, (T, D), 2 * T * D * H * 3
end

function gemm_workload(T, D, H)
    W = randn(Float32, D, H) ./ sqrt(Float32(D))
    return "x*W $(T)×$(D) · $(D)×$(H)", x -> x * W, (T, D), 2 * T * D * H
end

workloads = [ffn_workload(), gemm_workload(256, 1024, 1024), gemm_workload(1024, 1024, 1024),
             gemm_workload(2048, 2048, 2048)]

# --- run ------------------------------------------------------------------------
functional() || error("No usable Neural Engine on this machine (device_info: $(device_info()))")

results = map(workloads) do (name, f, dims, flops)
    x16 = randn(Float16, dims...)
    x32 = Float32.(x16)
    # The CPU baseline must really be Float32: a Float64 weight would promote it.
    eltype(f(x32)) == Float32 || error("$name does not compute in Float32 on the CPU")
    program = compile(f, x16)
    y, ref = Float32.(program(x16)), f(x32)
    err = sqrt(sum(abs2, y .- ref) / sum(abs2, ref))
    out = program(x16)                      # preallocated output, reused by every call
    t_ane = timing(x -> evaluate!(out, program, x), x16)
    close!(program)
    (; name, flops, err, t_ane, t_cpu=timing(f, x32))
end

openblas = [r.t_cpu for r in results]
using AppleAccelerate           # switches BLAS to Accelerate from here on
accelerate = [timing(w[2], randn(Float32, w[3]...)) for w in workloads]

gflops(flops, t) = @sprintf("%.0f", flops / t / 1e9)
ms(t) = @sprintf("%.3f / %.3f", 1e3t.min, 1e3t.med)
info = device_info()
macos = strip(read(`sw_vers -productVersion`, String))

println("\n**", Sys.cpu_info()[1].model, "**, ANE: ", info.num_cores, " cores (", info.architecture,
        "), macOS ", macos, ", Julia ", VERSION, ", ", BLAS.get_num_threads(), " BLAS threads\n")
println("| workload | ANE fp16 ms | GFLOP/s | Accelerate fp32 ms | GFLOP/s | OpenBLAS fp32 ms | GFLOP/s | rel. error |")
println("|---|--:|--:|--:|--:|--:|--:|--:|")
for (r, ta, to) in zip(results, accelerate, openblas)
    println("| ", r.name, " | ", ms(r.t_ane), " | ", gflops(r.flops, r.t_ane.min), " | ", ms(ta), " | ", gflops(r.flops, ta.min),
            " | ", ms(to), " | ", gflops(r.flops, to.min), " | ", @sprintf("%.1e", r.err), " |")
end
println("\nTimes are minimum / median ", METHOD == "loop" ? "of 100 calls" : "over $(SECONDS) s of calls (BenchmarkTools)",
        "; GFLOP/s use the minimum.")
println("ANE times include copying inputs and outputs (into a preallocated output). CPU times are Float32.")
