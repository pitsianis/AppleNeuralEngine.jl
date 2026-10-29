using AppleNeuralEngine

const CHANNELS = 256
const SPATIAL = 64

const MIL = """
program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
{
    func main<ios18>(tensor<fp32, [1, $(CHANNELS), 1, $(SPATIAL)]> x) {
        string c_pad_type_0 = const()[name = string("c_pad_type_0"), val = string("valid")];
        tensor<int32, [2]> c_strides_0 = const()[name = string("c_strides_0"), val = tensor<int32, [2]>([1, 1])];
        tensor<int32, [4]> c_pad_0 = const()[name = string("c_pad_0"), val = tensor<int32, [4]>([0, 0, 0, 0])];
        tensor<int32, [2]> c_dilations_0 = const()[name = string("c_dilations_0"), val = tensor<int32, [2]>([1, 1])];
        int32 c_groups_0 = const()[name = string("c_groups_0"), val = int32(1)];
        string x_to_fp16_dtype_0 = const()[name = string("x_to_fp16_dtype_0"), val = string("fp16")];
        tensor<fp16, [1, $(CHANNELS), 1, $(SPATIAL)]> x_to_fp16 = cast(dtype = x_to_fp16_dtype_0, x = x)[name = string("cast_in")];
        tensor<fp16, [$(CHANNELS), $(CHANNELS), 1, 1]> W0 = const()[name = string("W0"), val = tensor<fp16, [$(CHANNELS), $(CHANNELS), 1, 1]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(64)))];
        tensor<fp16, [1, $(CHANNELS), 1, $(SPATIAL)]> c0 = conv(dilations = c_dilations_0, groups = c_groups_0, pad = c_pad_0, pad_type = c_pad_type_0, strides = c_strides_0, weight = W0, x = x_to_fp16)[name = string("c0")];
        string to_fp32 = const()[name = string("to_fp32"), val = string("fp32")];
        tensor<fp32, [1, $(CHANNELS), 1, $(SPATIAL)]> c = cast(dtype = to_fp32, x = c0)[name = string("cast_out")];
    } -> (c);
}
"""

# Julia's first dimension is contiguous, so (SPATIAL, CHANNELS) gives the
# channel-first [1, C, 1, S] byte order expected by the MIL program.
weights = ones(Float32, CHANNELS, CHANNELS)
weight_blob = build_weight_blob(weights)
program = compile(MIL;
    input_shapes=[(SPATIAL, CHANNELS)],
    output_shapes=[(SPATIAL, CHANNELS)],
    weight_blob)

try
    zero_output, = evaluate!(program, zeros(Float32, SPATIAL, CHANNELS))
    one_output, = evaluate!(program, ones(Float32, SPATIAL, CHANNELS))
    @assert all(iszero, zero_output)
    @assert all(==(256f0), one_output)
    println("zero input: all outputs are 0")
    println("one input: all outputs are 256")
finally
    close!(program)
end
