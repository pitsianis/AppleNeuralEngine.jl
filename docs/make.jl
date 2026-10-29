using Documenter
using AppleNeuralEngine

makedocs(;
    sitename="AppleNeuralEngine.jl",
    # Source links need a git remote; until the repository exists, build without them.
    remotes=isdir(joinpath(@__DIR__, "..", ".git")) ? Documenter.Remotes.GitHub("pitsianis", "AppleNeuralEngine.jl") => nothing : nothing,
    modules=[AppleNeuralEngine, AppleNeuralEngine.MIL],
    format=Documenter.HTML(; prettyurls=get(ENV, "CI", "false") == "true"),
    pages=[
        "Home" => "index.md",
        "Guide" => [
            "Tracing Julia functions" => "tracing.md",
            "The MIL graph builder" => "builder.md",
            "Raw MIL and the runtime" => "runtime.md",
        ],
        "Op support" => "op_support.md",
        "Hardware notes" => "hardware.md",
        "API reference" => "api.md",
    ],
    # The examples need a Neural Engine, so they are not run when building the docs.
    doctest=false,
    warnonly=[:missing_docs],
)

deploydocs(; repo="github.com/pitsianis/AppleNeuralEngine.jl", devbranch="main")
