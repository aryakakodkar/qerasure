# C++ Lowering Reference

## Overview
`qerasure::Lowerer` converts erasure simulation events into Pauli error events using a Stim-like instruction model.

Core files:
- `include/qerasure/core/lowering/lowering.h`
- `src/core/lowering/lowering.cpp`

Partner-slot precomputation is provided by:
- `include/qerasure/core/code/rotated_surface_code.h`
- `src/core/code/rotated_surface_code.cpp`

## Instruction Model

Lowering is defined by a `SpreadProgram`, which is an ordered list of instructions:

- `ERROR_CHANNEL(p) targets...`
- `CORRELATED_ERROR(p) targets...`
- `ELSE_CORRELATED_ERROR(p) targets...`

Where a target is `(PauliError, PartnerSlot)`:
- `PartnerSlot::X_1`, `PartnerSlot::X_2`, `PartnerSlot::Z_1`, `PartnerSlot::Z_2`

All probabilities are sampled with splitmix64-based threshold sampling.

### Semantics

- `ERROR_CHANNEL`
  - Samples independently with probability `p`.
  - If sampled, applies all listed target errors.

- `CORRELATED_ERROR`
  - Samples with probability `p`.
  - If sampled, applies all listed target errors and sets internal `correlated_flag=1`.
  - If not sampled, sets `correlated_flag=0`.

- `ELSE_CORRELATED_ERROR`
  - Executes only when `correlated_flag==0`.
  - If executed, samples with probability `p` and updates flag exactly like `CORRELATED_ERROR`.
  - If `correlated_flag==1`, instruction is skipped.

## Program Assignment

`LoweringParams` supports:
- preferred construction with one default program for all data qubits
  - `LoweringParams(default_program)`
  - `LoweringParams(default_program, reset_params)`
- optional mutation with `set_default_data_program(...)`
- per-data-qubit overrides (`set_data_qubit_program(data_idx, program)`)

Legacy constructors are retained and internally mapped into default `ERROR_CHANNEL` instructions.

## Minimal Example

```cpp
#include "qerasure/core/lowering/lowering.h"

using namespace qerasure;

SpreadProgram prog;
prog.add_correlated_error(0.5, {{PauliError::X_ERROR, PartnerSlot::X_1}});
prog.add_else_correlated_error(1.0, {{PauliError::X_ERROR, PartnerSlot::X_2}});

LoweredErrorParams reset = {PauliError::Z_ERROR, 1.0};
LoweringParams params(prog, reset);
// or params.set_data_qubit_program(data_idx, prog);
```

## Python Equivalent

```python
prog = SpreadProgram()
prog.add_error_channel(0.5, [SpreadTargetOp(PauliError.X_ERROR, PartnerSlot.X_1)])
reset = LoweredErrorParams(PauliError.Z_ERROR, 1.0)
params = LoweringParams(default_program=prog, reset_params=reset)
# Also supported: params = LoweringParams(prog, reset)
```

## Runtime Notes

- Lowering spread programs apply to **erased data qubits**.
- Reset lowering is still controlled by `reset_params_`.
- Ancilla partner slots are precomputed in `RotatedSurfaceCode`, so lowering avoids repeated partner discovery at runtime.

## Testing

Stim-like chain behavior is validated by:
- `tests/lowering_standard_test.cpp`
