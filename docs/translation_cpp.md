# C++ Translation Reference

## Overview
`qerasure` translation utilities generate Stim-compatible circuit text from code geometry/schedule metadata.

Current implementation:
- `include/qerasure/core/translation/stim_translation.h`
- `src/core/translation/stim_translation.cpp`

Backend:
- Built with Stim's C++ `stim::Circuit` API (`safe_append_u` / `safe_append_ua`), not manual text concatenation.
- `CMakeLists.txt` fetches Stim automatically via `FetchContent` and links `libstim`.

## Surface-Code Stim Generator

API:

```cpp
std::string build_surface_code_stim_circuit(const RotatedSurfaceCode& code, std::size_t qec_rounds);
```

### Round semantics

- Requires `qec_rounds >= 2`.
- Runs `qec_rounds - 1` syndrome-extraction cycles.
- Each cycle emits:
  - `H` on all X ancillas
  - 4 `CX` layers from the rotated-surface schedule
  - `H` on all X ancillas
  - `MR` on all ancillas
  - `DETECTOR` parity checks:
    - first extraction round: Z-ancilla boundary detectors
    - later rounds: ancilla parity against previous-round ancilla measurement

After extraction cycles:
- `M` on all data qubits
- final Z-plaquette detectors using:
  - latest Z-ancilla measurement
  - final data measurements in that plaquette support
- `OBSERVABLE_INCLUDE(0)` for logical Z (data qubits on minimum x-coordinate column).

## Python API

```python
from qerasure import RotatedSurfaceCode, build_surface_code_stim_circuit

code = RotatedSurfaceCode(3)
stim_text = build_surface_code_stim_circuit(code, qec_rounds=3)
print(stim_text)
```

Note: this generator intentionally does not emit `QUBIT_COORDS` or `TICK` lines yet.
