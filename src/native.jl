# Direct bindings to the private AppleNeuralEngine framework, written with
# ObjectiveC.jl. The call sequence (in-memory MIL model -> compile -> load ->
# IOSurface-backed request -> evaluate) follows the reverse-engineering work in
# maderix/ANE (https://github.com/maderix/ANE).

const _ANE_FRAMEWORK = "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine"
const _IOSURFACE = "/System/Library/Frameworks/IOSurface.framework/IOSurface"
const _ANE_CLASSES = ("_ANEInMemoryModelDescriptor", "_ANEInMemoryModel", "_ANERequest", "_ANEIOSurfaceObject")
# QOS_CLASS_DEFAULT; the value used by maderix/ANE for every runtime call.
const _QOS = Cuint(0x15)
const _INITIALIZED = Ref(false)

# The private objects are owned explicitly by `ANEProgram` (see `_retain`), so
# their wrappers are unmanaged.
@objcwrapper managed = false _ANEInMemoryModelDescriptor <: NSObject
@objcwrapper managed = false _ANEInMemoryModel <: NSObject
@objcwrapper managed = false _ANERequest <: NSObject
@objcwrapper managed = false _ANEIOSurfaceObject <: NSObject

"""
    ANEError(msg)

Raised when the ANE runtime rejects a model or fails to run it. `msg` includes
the runtime's own `NSError` description when one is available.
"""
struct ANEError <: Exception
    msg::String
end

Base.showerror(io::IO, err::ANEError) = print(io, "ANEError: ", err.msg)

function _throw_ane(what, err::Ref{id{NSError}})
    detail = err[] == nil ? "unknown error" :
        String(NSString(@objc [err[]::id{NSError} description]::id{NSString}))
    throw(ANEError("$what: $detail"))
end

"""
    initialize!()

Load the private AppleNeuralEngine framework and check that the classes this
package relies on exist. Called automatically by [`compile`](@ref).
"""
function initialize!()
    _INITIALIZED[] && return nothing
    Sys.isapple() || error("AppleNeuralEngine requires macOS")
    Sys.ARCH === :aarch64 || error("AppleNeuralEngine requires Apple Silicon (arm64)")
    isnothing(Libdl.dlopen(_ANE_FRAMEWORK; throw_error=false)) &&
        throw(ANEError("could not load AppleNeuralEngine.framework"))
    for name in _ANE_CLASSES
        ccall(:objc_getClass, Ptr{Cvoid}, (Cstring,), name) == C_NULL &&
            throw(ANEError("private class $name is missing; this macOS release may have changed the ANE runtime"))
    end
    _INITIALIZED[] = true
    return nothing
end

_retain(obj::id{T}) where {T} = ccall(:objc_retain, id{T}, (id{T},), obj)
_release(obj::id) = ccall(:objc_release, Cvoid, (id{Object},), obj)

_nsdata(bytes::Vector{UInt8}) =
    @objc [NSData dataWithBytes:bytes::Ptr{UInt8} length:length(bytes)::NSUInteger]::id{NSData}

# Foundation collections from heterogeneous objects (ObjectiveC.jl's
# constructors need homogeneous, unmanaged element types).
function _nsdict(pairs)
    dict = @objc [NSMutableDictionary dictionary]::id{NSDictionary}
    for (key, value) in pairs
        @objc [dict::id{NSDictionary} setObject:value::id{Object} forKey:key::id{Object}]::Nothing
    end
    return dict
end

function _nsarray(items)
    array = @objc [NSMutableArray array]::id{NSArray}
    for item in items
        @objc [array::id{NSArray} addObject:item::id{Object}]::Nothing
    end
    return array
end

function _surface_key(name::Symbol)
    lib = Libdl.dlopen(_IOSURFACE)
    return unsafe_load(Ptr{id{NSString}}(Libdl.dlsym(lib, name)))
end

# Tensor buffers are 2-D IOSurfaces: one row per run of the innermost
# dimension, with the row stride padded to 64 bytes. The ANE rejects buffers
# whose rows are not 64-byte aligned; with the padded stride any innermost size
# works, down to a single element.
const _ROW_ALIGN = 64

function _create_surface((rowbytes, rows))
    stride = cld(rowbytes, _ROW_ALIGN) * _ROW_ALIGN
    props = _nsdict((
        _surface_key(:kIOSurfaceWidth) => NSNumber(Int(rowbytes)),
        _surface_key(:kIOSurfaceHeight) => NSNumber(Int(rows)),
        _surface_key(:kIOSurfaceBytesPerElement) => NSNumber(1),
        _surface_key(:kIOSurfaceBytesPerRow) => NSNumber(Int(stride)),
        _surface_key(:kIOSurfaceAllocSize) => NSNumber(Int(stride * rows)),
        _surface_key(:kIOSurfacePixelFormat) => NSNumber(0),
    ))
    surface = ccall((:IOSurfaceCreate, _IOSURFACE), Ptr{Cvoid}, (id{NSDictionary},), props)
    surface == C_NULL && throw(ANEError("could not allocate a $rows × $rowbytes-byte IOSurface"))
    return surface
end

_free_surface(surface::Ptr{Cvoid}) = ccall(:CFRelease, Cvoid, (Ptr{Cvoid},), surface)

const _LOCK_READONLY = UInt32(1)

# Copy between a dense Julia buffer (rows of `rowbytes`) and a surface, honoring
# the surface's row stride.
function _copy_surface!(surface::Ptr{Cvoid}, buffer::Ptr, rowbytes::Integer, rows::Integer, to_surface::Bool)
    lock = to_surface ? UInt32(0) : _LOCK_READONLY
    ccall((:IOSurfaceLock, _IOSURFACE), Cint, (Ptr{Cvoid}, UInt32, Ptr{UInt32}), surface, lock, C_NULL)
    base = ccall((:IOSurfaceGetBaseAddress, _IOSURFACE), Ptr{UInt8}, (Ptr{Cvoid},), surface)
    stride = Int(ccall((:IOSurfaceGetBytesPerRow, _IOSURFACE), Csize_t, (Ptr{Cvoid},), surface))
    buffer = Ptr{UInt8}(buffer)
    if stride == rowbytes
        to_surface ? unsafe_copyto!(base, buffer, rowbytes * rows) : unsafe_copyto!(buffer, base, rowbytes * rows)
    else
        for r in 0:rows-1
            to_surface ? unsafe_copyto!(base + r * stride, buffer + r * rowbytes, rowbytes) :
                         unsafe_copyto!(buffer + r * rowbytes, base + r * stride, rowbytes)
        end
    end
    ccall((:IOSurfaceUnlock, _IOSURFACE), Cint, (Ptr{Cvoid}, UInt32, Ptr{UInt32}), surface, lock, C_NULL)
    return nothing
end

# The runtime compiles from a directory named after the model's identifier in
# NSTemporaryDirectory(); populate it with the MIL text and weight files.
function _populate_model_dir(model::id{_ANEInMemoryModel}, mil, weights)
    hex = @objc [model::id{_ANEInMemoryModel} hexStringIdentifier]::id{NSString}
    tmp = ccall(:NSTemporaryDirectory, id{NSString}, ())
    dir = joinpath(String(NSString(tmp)), String(NSString(hex)))
    mkpath(joinpath(dir, "weights"))
    write(joinpath(dir, "model.mil"), mil)
    for (path, blob) in weights
        file = joinpath(dir, chopprefix(path, "@model_path/"))
        mkpath(dirname(file))
        write(file, blob)
    end
    return dir
end

function _load!(model::id{_ANEInMemoryModel}, err)
    @objc [model::id{_ANEInMemoryModel} loadWithQoS:_QOS::Cuint
                                        options:NSDictionary()::id{NSDictionary}
                                        error:err::Ptr{id{NSError}}]::Bool
end

# Compile and load `mil` with the given weight files, allocate one IOSurface per
# tensor and build the request binding them. Returns everything `ANEProgram`
# owns: (model, request, input surfaces, output surfaces, model directory).
# `inlayouts`/`outlayouts` are (row bytes, rows) per tensor.
@autoreleasepool function _compile_native(mil::Vector{UInt8}, weights, inlayouts, outlayouts)
    wdict = _nsdict(NSString(path) => _nsdict((NSString("offset") => NSNumber(0),
                                               NSString("data") => _nsdata(blob)))
                    for (path, blob) in weights)
    desc = @objc [_ANEInMemoryModelDescriptor modelWithMILText:_nsdata(mil)::id{NSData}
                                              weights:wdict::id{NSDictionary}
                                              optionsPlist:nil::id{NSDictionary}]::id{_ANEInMemoryModelDescriptor}
    desc == nil && throw(ANEError("the runtime rejected the MIL text or weights"))
    model = @objc [_ANEInMemoryModel inMemoryModelWithDescriptor:desc::id{_ANEInMemoryModelDescriptor}]::id{_ANEInMemoryModel}
    model == nil && throw(ANEError("could not create an in-memory model"))

    dir = _populate_model_dir(model, mil, weights)
    err = Ref{id{NSError}}(nil)
    try
        @objc([model::id{_ANEInMemoryModel} compileWithQoS:_QOS::Cuint
                                            options:NSDictionary()::id{NSDictionary}
                                            error:err::Ptr{id{NSError}}]::Bool) ||
            _throw_ane("ANE compilation failed", err)
        # Retry once: the ANE may need a moment to reclaim a slot from a
        # recently unloaded model.
        if !_load!(model, err)
            sleep(0.1)
            err[] = nil
            _load!(model, err) || _throw_ane("ANE load failed", err)
        end
    catch
        rm(dir; recursive=true, force=true)
        rethrow()
    end

    inputs = map(_create_surface, inlayouts)
    outputs = map(_create_surface, outlayouts)
    wrap(surface) = @objc [_ANEIOSurfaceObject objectWithIOSurface:surface::Ptr{Cvoid}]::id{_ANEIOSurfaceObject}
    request = @objc [_ANERequest requestWithInputs:_nsarray(map(wrap, inputs))::id{NSArray}
                                 inputIndices:_nsarray(NSNumber.(0:length(inputs)-1))::id{NSArray}
                                 outputs:_nsarray(map(wrap, outputs))::id{NSArray}
                                 outputIndices:_nsarray(NSNumber.(0:length(outputs)-1))::id{NSArray}
                                 weightsBuffer:nil::id{Object}
                                 perfStats:nil::id{Object}
                                 procedureIndex:NSNumber(0)::id{NSNumber}]::id{_ANERequest}
    return _retain(model), _retain(request), inputs, outputs, dir
end

@autoreleasepool function _evaluate_native(model::id{_ANEInMemoryModel}, request::id{_ANERequest})
    err = Ref{id{NSError}}(nil)
    @objc([model::id{_ANEInMemoryModel} evaluateWithQoS:_QOS::Cuint
                                        options:NSDictionary()::id{NSDictionary}
                                        request:request::id{_ANERequest}
                                        error:err::Ptr{id{NSError}}]::Bool) ||
        _throw_ane("ANE evaluation failed", err)
    return nothing
end

@autoreleasepool function _unload_native(model::id{_ANEInMemoryModel})
    err = Ref{id{NSError}}(nil)
    @objc [model::id{_ANEInMemoryModel} unloadWithQoS:_QOS::Cuint error:err::Ptr{id{NSError}}]::Bool
    return nothing
end

_device_string(sel) = (p = sel(); p == nil ? nothing : String(NSString(reinterpret(id{NSString}, p))))

"""
    device_info() -> NamedTuple

Properties of this machine's Neural Engine, as reported by the ANE framework:
`has_ane`, `virtual_machine`, `num_ane`, `num_cores`, `architecture` (e.g.
`"h14g"`) and `subtype`.
"""
@autoreleasepool function device_info()
    initialize!()
    return (has_ane=@objc([_ANEDeviceInfo hasANE]::Bool),
            virtual_machine=@objc([_ANEDeviceInfo isVirtualMachine]::Bool),
            num_ane=Int(@objc [_ANEDeviceInfo numANEs]::Clong),
            num_cores=Int(@objc [_ANEDeviceInfo numANECores]::Clong),
            architecture=_device_string(() -> @objc [_ANEDeviceInfo aneArchitectureType]::id{Object}),
            subtype=_device_string(() -> @objc [_ANEDeviceInfo aneSubType]::id{Object}))
end

"""
    functional() -> Bool

Whether programs can run on a Neural Engine here: Apple Silicon, the private
framework present, and a physical (not virtualized) ANE.
"""
function functional()
    Sys.isapple() && Sys.ARCH === :aarch64 || return false
    try
        info = device_info()
        return info.has_ane && !info.virtual_machine && info.num_cores > 0
    catch
        return false
    end
end
