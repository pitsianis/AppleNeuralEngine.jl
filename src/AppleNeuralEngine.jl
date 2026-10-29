module AppleNeuralEngine

using Libdl
using ObjectiveC
using ObjectiveC.Foundation
import Base: isopen

export ANEError, ANEProgram, build_weight_blob, close!, compile, device_info, evaluate!, functional, initialize!, isopen, pack_weights, trace, WeightFile, MIL

include("native.jl")
include("weights.jl")
include("mil/MIL.jl")
include("program.jl")
include("trace.jl")

end # module AppleNeuralEngine
