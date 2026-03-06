# qerasure

`qerasure` adds an erasure-aware circuit model and classical sampling pipeline on top of Stim-compatible circuits.

## Quickstart

1. Build an `ErasureCircuit`
2. Compile it with an `ErasureModel`, which describes how erasure onsets, spreads, checks, and resets are to be translated to Pauli errors
3. Sample:
   - `ErasureSampler` for retained shot traces (`SampledBatch`), for debugging and testing purposes
   - `StreamSampler` for high-throughput sampling
4. Decode sampled events using dynamically re-weighted matching graph (Note: currently only implemented for codes with checks which are immediately followed by resets)
