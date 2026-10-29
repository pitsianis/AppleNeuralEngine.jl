# Core ML weight files, as read by MIL `BLOBFILE(path = ..., offset = ...)`
# (format: coremltools mlmodel/src/MILBlob/Blob/StorageFormat.hpp):
#
#   file header (64 bytes):  UInt32 tensor count, UInt32 format version (2)
#   per tensor, 64-aligned:  metadata (64 bytes): UInt32 0xDEADBEEF, UInt32 dtype,
#                            UInt64 data size, UInt64 data offset;
#                            followed by the data
#
# A MIL constant refers to a tensor by the file offset of its metadata.

const _BLOB_DTYPES = Dict{DataType,UInt32}(
    Float16 => 1, Float32 => 2, UInt8 => 3, Int8 => 4, Int16 => 6, UInt16 => 7, Int32 => 14)

_le(x) = reinterpret(UInt8, [htol(x)])

"""
    WeightFile()

An ANE weight file that tensors are appended to one at a time with
`push!(file, array)`, which returns the offset a MIL `BLOBFILE` uses to refer
to the tensor. `bytes(file)` returns the file contents for [`compile`](@ref).

Arrays are stored in MIL (row-major) order: element `A[i, j, k]` is MIL element
`[i, j, k]`, so a conv weight `[Cout, Cin, kh, kw]` is the Julia array
`W[cout, cin, kh, kw]`. The element type is kept; supported types are
$(join(sort(string.(keys(_BLOB_DTYPES))), ", ")).
"""
mutable struct WeightFile
    data::Vector{UInt8}
    count::Int
    WeightFile() = new(zeros(UInt8, 64), 0)
end

function Base.push!(file::WeightFile, array::AbstractArray{T}) where {T}
    haskey(_BLOB_DTYPES, T) || throw(ArgumentError("weights of type $T are not supported; use one of $(join(keys(_BLOB_DTYPES), ", "))"))
    # Row-major order of `array`; Julia is column-major.
    payload = ndims(array) <= 1 ? collect(T, vec(array)) : vec(permutedims(array, ndims(array):-1:1))
    data = file.data
    append!(data, zeros(UInt8, cld(length(data), 64) * 64 - length(data)))  # 64-byte alignment
    meta = length(data)
    nbytes = sizeof(payload)
    header = zeros(UInt8, 64)
    header[1:4] .= (0xEF, 0xBE, 0xAD, 0xDE)
    header[5:8] .= _le(_BLOB_DTYPES[T])
    header[9:16] .= _le(UInt64(nbytes))
    header[17:24] .= _le(UInt64(meta + 64))
    append!(data, header)
    append!(data, reinterpret(UInt8, payload))
    file.count += 1
    return meta
end

"""
    bytes(file::WeightFile) -> Vector{UInt8}

The contents of the weight file, to pass to [`compile`](@ref) as `weight_blob`.
"""
function bytes(file::WeightFile)
    data = copy(file.data)
    data[1:4] .= _le(UInt32(file.count))
    data[5:8] .= _le(UInt32(2))
    return data
end

"""
    pack_weights(tensors...; transposed=false) -> (blob, offsets)

Encode several `Float32` matrices as fp16 tensors in one ANE weight file. Pass
`blob` to [`compile`](@ref) as `weight_blob` and refer to tensor `i` in the MIL
program as

    BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(offsets[i]))

Each tensor is stored in row-major order, so a conv weight `[Cout, Cin, 1, 1]`
is the matrix `W[cout, cin]`; with `transposed=true` every tensor is stored
transposed. A vector (e.g. a per-channel scale) can be passed as a `1×n` matrix.
See [`WeightFile`](@ref) for tensors of other ranks and element types.
"""
function pack_weights(tensors::AbstractMatrix{Float32}...; transposed::Bool=false)
    isempty(tensors) && throw(ArgumentError("pack_weights needs at least one tensor"))
    file = WeightFile()
    offsets = [push!(file, Float16.(transposed ? permutedims(t) : t)) for t in tensors]
    return bytes(file), offsets
end

"""
    build_weight_blob(weights::AbstractMatrix{Float32}; transposed=false)

Encode one `Float32` matrix as an fp16 ANE weight file, referred to in MIL as
`BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(64))`.
See [`pack_weights`](@ref) for several tensors and for the storage order.
"""
build_weight_blob(weights::AbstractMatrix{Float32}; transposed::Bool=false) =
    first(pack_weights(weights; transposed))
