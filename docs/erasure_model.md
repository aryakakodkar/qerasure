# qerasure Erasure Model Reference

This page documents the Python-facing erasure-model API in `qerasure` for the circuit model.

## Contents
- [Overview](#overview)
- [Channel Slots](#channel-slots)
- [Supported Channel Specs](#supported-channel-specs)
- [Check-Error Parameters](#check-error-parameters)
- [Precedence Rules](#precedence-rules)
- [Examples](#examples)
- [Debugging and Transparency](#debugging-and-transparency)

## Overview
The user-friendly path is:
- `qe.make_erasure_model(...)`
- or `qe.ErasureModel.from_specs(...)`

These build a model from named channel slots and optional check-error parameters, then pass the compiled result to C++ unchanged.

## Channel Slots
The model uses four named Pauli-channel slots:

- `onset`: Pauli error channel applied on erasure onset.
- `reset`: Pauli error channel applied on reset failure.
- `spread_control`: spread channel used when erased qubit is control in `CX`.
- `spread_target`: spread channel used when erased qubit is target in `CX`.

You can pass these via:
- `channels={...}` map, or
- direct keyword arguments (`onset=...`, `reset=...`, etc.).

## Supported Channel Specs
Each channel slot accepts:
- string spec
- `qe.PauliChannel`
- tuple/list `(p_x, p_y, p_z)`

Supported string forms:
- `PAULI_CHANNEL(px,py,pz)`
- `X_ERROR(p)`
- `Y_ERROR(p)`
- `Z_ERROR(p)`
- `DEPOLARIZE1(p)` (mapped to `PAULI_CHANNEL(p/3,p/3,p/3)`)

Validation:
- all probabilities must be in `[0,1]`
- for `PAULI_CHANNEL`, `p_x + p_y + p_z <= 1`

## Check-Error Parameters
Three check-related parameters are available:

- `check_error_prob`: convenience value that sets both FN and FP.
- `check_false_negative_prob`: explicit FN override.
- `check_false_positive_prob`: explicit FP override.

If only one explicit side is provided and `check_error_prob` is omitted, the other side stays at default `0.0`.

## Precedence Rules
When constructing the model:

1. Channel slot source:
   - explicit slot kwargs (`onset=...`) override `channels={...}` values.
2. Check probabilities:
   - apply `check_error_prob` first to both FN and FP,
   - then apply explicit `check_false_negative_prob` / `check_false_positive_prob` overrides.

## Examples

### Minimal named-slot model
```python
import qerasure as qe

model = qe.make_erasure_model(
    max_persistence=2,
    channels={
        "onset": "PAULI_CHANNEL(0.25,0.25,0.25)",
        "reset": "PAULI_CHANNEL(0.25,0.25,0.25)",
        "spread_control": "X_ERROR(0.5)",
        "spread_target": "Z_ERROR(0.5)",
    },
    check_error_prob=0.02,
)
```

### Explicit FN/FP override
```python
model = qe.make_erasure_model(
    max_persistence=2,
    channels={
        "onset": "DEPOLARIZE1(0.75)",
        "reset": "DEPOLARIZE1(0.75)",
        "spread_control": "X_ERROR(0.5)",
        "spread_target": "Z_ERROR(0.5)",
    },
    check_error_prob=0.02,           # sets FN=FP=0.02
    check_false_positive_prob=0.0,   # override FP only
)
```

### Legacy low-level constructor (still supported)
```python
model = qe.ErasureModel(
    2,
    qe.PauliChannel(0.25, 0.25, 0.25),
    qe.PauliChannel(0.25, 0.25, 0.25),
    qe.PauliChannel(0.5, 0.0, 0.0),
    qe.PauliChannel(0.0, 0.0, 0.5),
)
model.check_false_negative_prob = 0.01
model.check_false_positive_prob = 0.0
```

## Debugging and Transparency
Use:
- `print(model.explain())`

This prints:
- `max_persistence`
- parsed channel values for `onset`, `reset`, `spread_control`, `spread_target`
- `check_false_negative_prob`, `check_false_positive_prob`

This is the recommended way to verify model semantics in notebooks and scripts.
