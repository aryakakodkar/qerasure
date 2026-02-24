# C++ Lowering Reference

## Overview
`qerasure::Lowerer` converts erasure simulation events into Pauli error events using a Stim-like instruction model.

## Instruction Model

Lowering is defined by a `LoweringParams` object. This contains:
1. A `SpreadProgram` 
2. A reset error

The reset error defines the type of Pauli error on the qubit upon reset.

A `SpreadProgram` is an ordered list of instructions defined for each data qubit. A `SpreadProgram` can be provided with an instruction in the following form:

```
SpreadProgram program;
program.append("ERROR_TYPE(p) TARGET")
```

where `ERROR_TYPE` must be one of the following error types:
- `X_ERROR`
- `Z_ERROR`
- `Y_ERROR`

and target must be in the form: `(STABILIZER_TYPE)_(ANCILLA_INDEX)`. For example, `X_2` is the ancilla which measures the second (in time) X-stabilizer for the data qubit with which this `SpreadProgram` is associated. Note that currently any instruction only accepts one target.

Conditional errors can be introduced using `COND` and `ELSE` modifiers as described below.

All probabilities are sampled with splitmix64-based threshold sampling.

### Semantics

- `COND_ERROR_TYPE`
  - e.g. `COND_X_ERROR(p)`
  - Samples error with probability `p`
  - If sampled, applies all listed target errors and sets internal `conditional_flag=1`.
  - If not sampled, sets `conditional_flag=0`.

- `ELSE_ERROR_TYPE`
  - e.g. `ELSE_X_ERROR(p)`
  - Executes only when `conditional_flag==0`.
  - If executed, samples with probability `p` and updates flag to `conditional_flag=1`
  - If `correlated_flag==1`, instruction is skipped.

## Program Assignment

`LoweringParams` supports:
- preferred construction with one default program for all data qubits
  - `LoweringParams(default_program)`
  - `LoweringParams(default_program, reset_params)`
- optional mutation with `set_default_data_program(...)`
- per-data-qubit overrides (`set_data_qubit_program(data_idx, program)`)

## Example

```cpp
#include "qerasure/core/lowering/lowering.h"

using namespace qerasure;

SpreadProgram prog;
prog.append("Z_ERROR(0.5) X_1; Z_ERROR(0.5) X_2") // applies Z-errors on X_1 and X_2 independently with prob. 0.5
prog.append("COND_X_ERROR(0.5) Z_1; ELSE_X_ERROR(1) Z_2") // either applies an X-errir to Z_1 or Z_2 symmetrically

LoweredErrorParams reset = LoweredErrorParams(PauliError.DEPOLARIZE, 0.5)
LoweringParams params(prog, reset);
```

## Runtime Notes

- Lowering spread programs apply to **erased data qubits**.
- Reset lowering is still controlled by `reset_params_`. Will update soon.
- Ancilla partner slots are precomputed in `RotatedSurfaceCode`, so lowering avoids repeated partner discovery at runtime.
