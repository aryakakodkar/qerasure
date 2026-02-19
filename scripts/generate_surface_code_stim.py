#!/usr/bin/env python3
"""Generate and save a rotated-surface-code Stim circuit."""

from __future__ import annotations

import argparse
from pathlib import Path

from qerasure import RotatedSurfaceCode, build_surface_code_stim_circuit


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a rotated-surface-code Stim circuit and save it to disk."
    )
    parser.add_argument("--distance", type=int, required=True, help="Odd code distance (>=3).")
    parser.add_argument(
        "--qec-rounds",
        type=int,
        required=True,
        help="Requested number of syndrome-extraction rounds in output.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .stim path (default: tests/surface_code_d<d>_r<rounds>.stim).",
    )
    args = parser.parse_args()

    if args.qec_rounds < 1:
        raise ValueError("--qec-rounds must be >= 1")

    code = RotatedSurfaceCode(args.distance)

    # Translation API currently uses extraction_rounds = qec_rounds - 1 semantics.
    # Add one so CLI argument directly maps to emitted extraction rounds.
    stim_text = build_surface_code_stim_circuit(code, args.qec_rounds + 1)

    output_path = args.output
    if output_path is None:
        output_path = Path(f"tests/surface_code_d{args.distance}_r{args.qec_rounds}.stim")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(stim_text)

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
