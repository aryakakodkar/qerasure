# qerasure

`qerasure` adds an erasure-aware circuit model and fast classical sampling pipeline on top of Stim-compatible circuits.

## Quickstart

1. Build an `ErasureCircuit` (directly or via `qerasure::gen::SurfaceCodeRotated`).
2. Compile it with an `ErasureModel` into `CompiledErasureProgram`.
3. Sample:
   - `ErasureSampler` for retained shot traces (`SampledBatch`)
   - `StreamSampler` for high-throughput callback-based processing
4. Convert sampled events back into a Stim circuit with `Injector` (for retained batches), or decode directly in stream callbacks.

## Documentation

- [Docs index](docs/index.md)
- [Circuit model quickstart](docs/circuit_model_quickstart.md)
- [Gate reference (Stim-style)](docs/gate_reference.md)
- [Sampling modes](docs/sampling_modes.md)
- [Checks and resets semantics](docs/checks_resets_semantics.md)
- [Examples](docs/examples.md)
- [FAQ](docs/faq.md)
