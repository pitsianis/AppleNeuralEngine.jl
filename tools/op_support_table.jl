# Build docs/src/op_support.md from the per-processor probe results in docs/op_support/*.tsv.
#
#     julia tools/op_support_table.jl
#
# `tools/probe_ops.jl` writes one TSV per processor and then runs this script. Each TSV starts
# with `# key: value` lines (chip, macos, date, by, cases, note) and has the columns
# category, op, status, err, note. Ops come in the order of the most complete file.

const DATA = joinpath(@__DIR__, "..", "docs", "op_support")
const OUT = joinpath(@__DIR__, "..", "docs", "src", "op_support.md")

# Every Apple M processor released so far (generation, processors, year). Add new ones here.
const KNOWN = [("M1", ["M1", "M1 Pro", "M1 Max", "M1 Ultra"], "2020–22"),
               ("M2", ["M2", "M2 Pro", "M2 Max", "M2 Ultra"], "2022–23"),
               ("M3", ["M3", "M3 Pro", "M3 Max", "M3 Ultra"], "2023–25"),
               ("M4", ["M4", "M4 Pro", "M4 Max"], "2024"),
               ("M5", ["M5", "M5 Pro", "M5 Max", "M5 Ultra"], "2025–26")]
const ORDER = [p for (_, ps, _) in KNOWN for p in ps]

struct Probe
    chip::String                       # e.g. "M2 Max"
    meta::Dict{String,String}
    rows::Vector{NTuple{5,String}}     # category, op, status, err, note
end

function read_probe(path)
    meta = Dict{String,String}(); rows = NTuple{5,String}[]
    for l in eachline(path)
        isempty(strip(l)) && continue
        if startswith(l, "#")
            m = match(r"^#\s*(\w+):\s*(.*)$", l)
            m === nothing || (meta[m[1]] = strip(m[2]))
            continue
        end
        f = split(l, '\t'; keepempty=true)
        f[1] == "category" && continue
        length(f) == 5 || error("$path: expected 5 tab-separated fields, got $(length(f)): $l")
        push!(rows, Tuple(String.(f)))
    end
    chip = replace(get(meta, "chip", splitext(basename(path))[1]), r"^Apple\s+" => "")
    Probe(chip, meta, rows)
end

works(status) = startswith(status, "✓") || startswith(status, "≈")
rank(chip) = (i = findfirst(==(chip), ORDER); i === nothing ? length(ORDER) + 1 : i)

function entry(chip, status, err)
    status == "≈" && return "$chip ≈ ($err)"
    status == "✓ runs" && return "$chip (runs; no reference)"
    isempty(err) ? chip : "$chip ($err)"
end

function build(probes)
    sort!(probes; by=p -> rank(p.chip))
    # op order and categories: most complete file first, then ops only the others have
    main = argmax(p -> length(p.rows), probes)
    ops = [(r[1], r[2]) for r in main.rows]
    for p in probes, r in p.rows
        (r[1], r[2]) in ops || push!(ops, (r[1], r[2]))
    end
    result = Dict((p.chip, r[2]) => r for p in probes for r in p.rows)

    io = IOBuffer()
    println(io, """
    # Op support

    Which MIL ops the ANE compiles and runs, by processor. Each op runs in a small fp16 program
    built by `tools/probe_ops.jl`; the number after a processor is the maximum absolute error
    relative to the largest reference value (fp16 rounding gives ~1e-3), and **≈** marks results
    within 10% (a coarse hardware approximation). "compile fails (InvalidMILProgram)" means the
    ANE compiler's MIL validation rejected the op; every argument encoding we tried (inline or
    `const` literals, other axes, other modes) was rejected. Core ML may lower these ops to
    others before they reach the ANE. The ANE is a private, undocumented target, so results can
    change with macOS releases as well as with the processor.

    **Contributing a processor:** run `PROBE_BY="Your Name" julia --project=. tools/probe_ops.jl`
    on it (after `julia --project=. -e 'using Pkg; Pkg.instantiate()'` in a fresh clone). The probe
    writes `docs/op_support/<processor>.tsv` and regenerates this page; send both in a pull
    request. Processors not yet measured have no entry in the tables.

    ## Processors
    """)
    println(io, "| generation | processor | measured | macOS | date | cases that run | by |\n|---|---|---|---|---|---|---|")
    byname = Dict(p.chip => p for p in probes)
    for (gen, ps, years) in KNOWN, (i, chip) in enumerate(ps)
        g = i == 1 ? "$gen ($years)" : ""
        if haskey(byname, chip)
            p = byname[chip]; m = p.meta
            n = count(r -> works(r[3]), p.rows)
            println(io, "| $g | $chip | yes | $(get(m, "macos", "")) | $(get(m, "date", "")) | $n of $(length(p.rows)) | $(get(m, "by", "")) |")
        else
            println(io, "| $g | $chip | not yet |  |  |  |  |")
        end
    end
    for p in probes
        rank(p.chip) > length(ORDER) &&
            println(io, "| other | $(p.chip) | yes | $(get(p.meta, "macos", "")) | $(get(p.meta, "date", "")) | $(count(r -> works(r[3]), p.rows)) of $(length(p.rows)) | $(get(p.meta, "by", "")) |")
    end
    notes = [(p.chip, p.meta["note"]) for p in probes if haskey(p.meta, "note")]
    isempty(notes) || println(io, "\n", join(("- **$c:** $n" for (c, n) in notes), "\n"))

    # ops whose outcome differs between processors
    diffs = [(c, op) for (c, op) in ops
             if length(unique(works(result[(p.chip, op)][3]) for p in probes if haskey(result, (p.chip, op)))) > 1]
    println(io, "\n## Differences between processors\n")
    if isempty(diffs)
        println(io, "None among the ops measured on more than one processor.")
    else
        for (c, op) in diffs
            ok = [p.chip for p in probes if haskey(result, (p.chip, op)) && works(result[(p.chip, op)][3])]
            no = [p.chip for p in probes if haskey(result, (p.chip, op)) && !works(result[(p.chip, op)][3])]
            println(io, "- `$op` works on ", join(ok, ", "), "; fails on ", join(no, ", "), ".")
        end
    end

    println(io, """

    Not probed (they need control flow, dynamic shapes or state, which this package does not
    support yet): `cond`, `while_loop`, `make_list` and the `list_*` ops, `non_zero`,
    `non_maximum_suppression`, `classify`, `read_state`, `slice_update`, `affine`, `resample`,
    `crop_resize`, `reshape_like`, and the `constexpr_*` compression ops other than
    `constexpr_affine_dequantize`.

    Observed constraints, independent of the op (measured on the M2 Max):
    - Computation must be fp16: the same ops on fp32 tensors do not compile (cast inputs).
    - No int32 tensor may appear anywhere in a program: even a `cast` to int32 fails MIL
      validation. int16, uint16 and int8 work. Ops whose results are int32 by default take a
      uint16 option: `output_dtype=UInt16` for `reduce_argmax`/`reduce_argmin`,
      `indices_dtype=UInt16` for `topk`.
    - `reduce_argmax`/`reduce_argmin` with uint16 output return fp16 bit patterns labelled
      uint16: after `cast(i, Float16)`, decode on the host with
      `reinterpret(Float16, UInt16(y))`. The decoded indices are exact up to index 128 and
      rounded beyond it (consistent with 7 significant bits); more than 2048 elements along
      the axis does not compile.
    - Gather ops (`gather`, `gather_along_axis`, `gather_nd`) work with int16 or uint16
      indices, constant or computed (a computed index must be uint16, e.g. a cast of an
      fp16 input), when `validate_indices` is written out (the wrappers do), and at most
      2048 elements along the gathered axis. The scatter ops and `one_hot` still fail.
      The 2048 limits here and for argmax match the largest range in which fp16 holds
      every integer, so the ANE presumably carries indices as fp16.
    - `topk` with uint16 indices passes MIL validation but the compiler backend rejects it
      (CompilationFailure) for every shape, axis, `k` and opset tried, even with
      `return_indices=false`.
    - A trailing `identity` op cannot be a program output.
    - Tensor buffers need a 64-byte aligned row stride, which this package provides, so any
      innermost size works.""")

    for category in unique(first.(ops))
        println(io, "\n## ", uppercasefirst(category), "\n\n| op | works on | fails on | note |\n|---|---|---|---|")
        for (c, op) in ops
            c == category || continue
            here = [(p.chip, result[(p.chip, op)]) for p in probes if haskey(result, (p.chip, op))]
            ok = [entry(chip, r[3], r[4]) for (chip, r) in here if works(r[3])]
            no = [chip * (r[3] == "✗ wrong" ? " (wrong)" : "") for (chip, r) in here if !works(r[3])]
            ns = [(chip, r[5]) for (chip, r) in here if !isempty(r[5]) && r[5] != "approximate" && r[5] != "no reference"]
            note = length(unique(last.(ns))) == 1 && length(ns) == length(here) ? ns[1][2] :
                   join(("$chip: $n" for (chip, n) in ns), "; ")
            println(io, "| `", op, "` | ", join(ok, ", "), " | ", join(no, ", "), " | ", note, " |")
        end
    end
    String(take!(io))
end

function main()
    files = sort(filter(f -> endswith(f, ".tsv"), readdir(DATA; join=true)))
    isempty(files) && error("no probe results in $DATA")
    write(OUT, build(read_probe.(files)))
    println("wrote $OUT from ", join(basename.(files), ", "))
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()
