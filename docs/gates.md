# qerasure Gate Reference

This page documents qerasure-specific operations.  
Stim-native gates (`H`, `CX`, `M`, `R`, `MR`, `X_ERROR`, `Z_ERROR`, `DEPOLARIZE1`, `DETECTOR`, `OBSERVABLE_INCLUDE`) follow Stim semantics.

## Contents
- [Common Rules](#common-rules)
- [ERASE](#erase)
- [ERASE2](#erase2)
- [ERASE2_ANY](#erase2_any)
- [EC](#ec)
- [ECR](#ecr)
- [COND_ER](#cond_er)

## Common Rules
- Every instruction must have at least one target.
- Two-qubit instructions require an even number of targets (`ERASE2`, `ERASE2_ANY`).
- Probability arguments must be in [0, 1].
- Erasure behavior depends on the `ErasureModel` passed into `CompiledErasureProgram`:
  - `onset` channel
  - `spread.control_spread` and `spread.target_spread`
  - `reset` channel
  - `check_false_negative_prob`, `check_false_positive_prob`
  - `max_persistence`

## ERASE
Single-qubit erasure onset.

Argument:
- `p`: erasure probability.

Targets:
- One or more qubits.

Semantics:
- For each target qubit, with probability `p`, the qubit becomes erased.
- Erased qubits can subsequently generate persistent spread errors on future entangling gates, according to the erasure model.

Example:
```stim
ERASE(0.001) 5
ERASE(0.002) 12 14
```

## ERASE2
Directional two-qubit erasure onset.

Argument:
- `p`: erasure probability.

Targets:
- Qubit pairs `(q0 q1)`.

Semantics (per pair):
- With probability `p`, `q0` is erased.
- If `q0` onset fires, `q1` receives an onset spread sampled from the model onset channel.
- Behavior is directional: the first target is the erasable qubit for that pair.

Example:
```stim
# Erase qubit 4; apply onset spread onto qubit 6.
ERASE2(0.001) 4 6

# Pairs are processed independently.
ERASE2(0.001) 4 5 8 9
```

## ERASE2_ANY
Symmetric two-qubit erasure onset.

Argument:
- `p`: pair-erasure trigger probability.

Targets:
- Qubit pairs `(q0 q1)`.

Semantics (per pair):
- With probability `p`, exactly one of the two qubits is erased, chosen uniformly.
- The non-erased partner receives onset spread sampled from the model onset channel.
- If either qubit is already erased at runtime, the pair event is skipped.

Example:
```stim
ERASE2_ANY(0.01) 9 17
ERASE2_ANY(0.01) 9 17 10 18
```

## EC
Erasure check (no reset).

Argument:
- Optional numeric argument accepted by parser; check behavior is controlled by the erasure model false-positive/false-negative parameters.

Targets:
- Qubits to check.

Semantics:
- Produces a binary erasure-flag outcome per checked qubit.
- If qubit is erased: true positive unless false negative occurs.
- If qubit is not erased: false positive may occur.
- If erased qubit has survived `max_persistence - 1` missed checks (or reaches its final check in the circuit), check is forced to report positive.

Example:
```stim
EC 11 12 13 14
```

## ECR
Erasure check followed by conditional reset.

Argument:
- `p_reset_fail`: reset failure probability.

Targets:
- Qubits to check and conditionally reset.

Semantics:
- Runs the same check logic as `EC`.
- If the check flags positive, reset is attempted:
  - with probability `p_reset_fail`, a Pauli reset-failure error is sampled from the model reset channel,
  - otherwise reset succeeds with no reset-channel Pauli.
- Reset clears erasure state for that qubit.
- If max-persistence is exceeded, reset is forced (effectively no reset-failure sampling for that forced case).

Example:
```stim
ECR(0.0) 11 12 13 14
ECR(0.02) 11 12 13 14
```

## COND_ER
Conditional reset only (no check).

Argument:
- `p_reset_fail`: reset failure probability.

Targets:
- Qubits to conditionally reset.

Semantics:
- Uses the most recent check outcome on each target qubit.
- Reset executes only if the most recent check was positive.
- If reset executes, failure is sampled exactly as in `ECR` using the model reset channel.

Example:
```stim
EC 11 12 13 14
COND_ER(0.01) 11 12 13 14
```
