# Hardware notes

Observations made while building and testing this package on an M2 Max
(16-core ANE, macOS 27). They complement the restrictions catalogued by
[Orion (arXiv 2603.06728)](https://arxiv.org/abs/2603.06728).

## Precision

- The ANE computes in **fp16**. Ops on fp32 tensors do not compile, and neither
  does a cast to fp32 in the middle of a graph. Only program inputs and outputs
  may be fp32.
- `round` rounds ties away from zero, computed as `sign(x)·floor(|x| + 0.5)` in
  fp16. For `|x|` just below a half-integer the addition can round up.
- `floor_div` uses an approximate reciprocal and can be off by one when the
  quotient is (nearly) an integer. `atan` is accurate to about 2.5%.

## Buffers

- **Row alignment.** Tensor buffers are 2-D IOSurfaces with one row per run of
  the innermost MIL dimension. The ANE rejects rows that are not 64-byte aligned
  (16 fp32 or 32 fp16 values). The runtime reports this as an evaluation failure.
  With the row *stride* padded to 64 bytes, any innermost size works, down to
  1. This package does that, so the "pad the sequence dimension to ≥ 16" rule
  reported elsewhere does not apply.
- **Buffer sizes** must match what the compiled model expects. The runtime reports
  "IOSurface smaller than the model expects" otherwise.

## Program structure

- Inputs and outputs are bound in sorted-name order (see
  [Raw MIL and the runtime](@ref)).
- A program output cannot be produced by a trailing `identity` op.
- Variable names must not be MIL keywords (`const`, `tensor`, type names).

## Cost

- Compiling and loading a program takes about 20 ms for a tiny graph and about
  70–80 ms for a transformer block. We ran 300 compile/run/close cycles in one
  process without hitting a compile limit, as long as programs are closed.
- Each evaluation costs about 0.1–0.2 ms on top of the compute, for the call and
  the buffer copies. Small programs are therefore latency-bound. Batch work into
  larger matrix-matrix products.
- Calling the runtime through ObjectiveC.jl instead of a compiled Objective-C
  bridge adds about 0.1 µs per call, which is not measurable against the ANE
  itself.
