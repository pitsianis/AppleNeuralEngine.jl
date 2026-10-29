# Small MIL program generators used by the test suite. All programs take and
# return fp32 tensors of shape [1, C, 1, S] and compute in fp16 on the ANE, so a
# Julia array of size (S, C) has the matching memory layout.

const MIL_HEADER = """
program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
"""

_t(dtype, C, S) = "tensor<$dtype, [1, $C, 1, $S]>"

const _CASTS = """
        string to16 = const()[name = string("to16"), val = string("fp16")];
        string to32 = const()[name = string("to32"), val = string("fp32")];
"""

# y = W * x over channels (1x1 convolution), W read from the single weight blob.
function conv_mil(C, S)
    return MIL_HEADER * """
{
    func main<ios18>($(_t("fp32", C, S)) x) {
$_CASTS
        string pt = const()[name = string("pt"), val = string("valid")];
        tensor<int32, [2]> st = const()[name = string("st"), val = tensor<int32, [2]>([1, 1])];
        tensor<int32, [4]> pd = const()[name = string("pd"), val = tensor<int32, [4]>([0, 0, 0, 0])];
        tensor<int32, [2]> dl = const()[name = string("dl"), val = tensor<int32, [2]>([1, 1])];
        int32 gr = const()[name = string("gr"), val = int32(1)];
        $(_t("fp16", C, S)) x16 = cast(dtype = to16, x = x)[name = string("cx")];
        tensor<fp16, [$C, $C, 1, 1]> W = const()[name = string("W"), val = tensor<fp16, [$C, $C, 1, 1]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(64)))];
        $(_t("fp16", C, S)) c16 = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = W, x = x16)[name = string("c")];
        $(_t("fp32", C, S)) y = cast(dtype = to32, x = c16)[name = string("cy")];
    } -> (y);
}
"""
end

# z = x - y; two inputs, no weights.
function sub_mil(C, S)
    return MIL_HEADER * """
{
    func main<ios18>($(_t("fp32", C, S)) x, $(_t("fp32", C, S)) y) {
$_CASTS
        $(_t("fp16", C, S)) x16 = cast(dtype = to16, x = x)[name = string("cx")];
        $(_t("fp16", C, S)) y16 = cast(dtype = to16, x = y)[name = string("cy")];
        $(_t("fp16", C, S)) z16 = sub(x = x16, y = y16)[name = string("s")];
        $(_t("fp32", C, S)) z = cast(dtype = to32, x = z16)[name = string("cz")];
    } -> (z);
}
"""
end

# Inputs declared as (b, a) and outputs as (q, p), with q = b - a, p = relu(b).
# Used to pin down how the runtime binds tensors to buffer indices.
function unsorted_io_mil(C, S)
    return MIL_HEADER * """
{
    func main<ios18>($(_t("fp32", C, S)) b, $(_t("fp32", C, S)) a) {
$_CASTS
        $(_t("fp16", C, S)) b16 = cast(dtype = to16, x = b)[name = string("cb")];
        $(_t("fp16", C, S)) a16 = cast(dtype = to16, x = a)[name = string("ca")];
        $(_t("fp16", C, S)) s16 = sub(x = b16, y = a16)[name = string("s")];
        $(_t("fp16", C, S)) r16 = relu(x = b16)[name = string("r")];
        $(_t("fp32", C, S)) q = cast(dtype = to32, x = s16)[name = string("cq")];
        $(_t("fp32", C, S)) p = cast(dtype = to32, x = r16)[name = string("cp")];
    } -> (q, p);
}
"""
end

# z = W2 * relu(W1 * x), both weights packed in one file at the given offsets.
function two_conv_mil(C, S, off1, off2)
    W(name, off) = """tensor<fp16, [$C, $C, 1, 1]> $name = const()[name = string("$name"), val = tensor<fp16, [$C, $C, 1, 1]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64($off)))];"""
    return MIL_HEADER * """
{
    func main<ios18>($(_t("fp32", C, S)) x) {
$_CASTS
        string pt = const()[name = string("pt"), val = string("valid")];
        tensor<int32, [2]> st = const()[name = string("st"), val = tensor<int32, [2]>([1, 1])];
        tensor<int32, [4]> pd = const()[name = string("pd"), val = tensor<int32, [4]>([0, 0, 0, 0])];
        tensor<int32, [2]> dl = const()[name = string("dl"), val = tensor<int32, [2]>([1, 1])];
        int32 gr = const()[name = string("gr"), val = int32(1)];
        $(_t("fp16", C, S)) x16 = cast(dtype = to16, x = x)[name = string("cx")];
        $(W("W1", off1))
        $(W("W2", off2))
        $(_t("fp16", C, S)) h = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = W1, x = x16)[name = string("h")];
        $(_t("fp16", C, S)) r = relu(x = h)[name = string("r")];
        $(_t("fp16", C, S)) z16 = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = W2, x = r)[name = string("z")];
        $(_t("fp32", C, S)) z = cast(dtype = to32, x = z16)[name = string("cz")];
    } -> (z);
}
"""
end
