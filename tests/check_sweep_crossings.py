#!/usr/bin/env python3
"""Inspect pairwise distance-curve crossings in a sweep JSON."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def _close(a: float, b: float, tol: float = 1e-15) -> bool:
    return abs(a - b) <= tol


def _crossings_linear(
    curve_small: list[tuple[float, float]],
    curve_large: list[tuple[float, float]],
    include_boundary_roots: bool,
) -> list[float]:
    if len(curve_small) != len(curve_large):
        raise ValueError("Distance curves have different point counts.")

    roots: list[float] = []
    for i in range(len(curve_small) - 1):
        p0, y0s = curve_small[i]
        p1, y1s = curve_small[i + 1]
        q0, y0l = curve_large[i]
        q1, y1l = curve_large[i + 1]
        if not (_close(p0, q0) and _close(p1, q1)):
            raise ValueError("Distance curves are on different e-grids.")

        d0 = y0s - y0l
        d1 = y1s - y1l

        if d0 * d1 < 0.0:
            frac = -d0 / (d1 - d0)
            roots.append(p0 + frac * (p1 - p0))
            continue
        if include_boundary_roots and d0 == 0.0:
            roots.append(p0)
            continue
        if include_boundary_roots and d1 == 0.0:
            roots.append(p1)

    unique: list[float] = []
    for x in sorted(roots):
        if not unique or abs(x - unique[-1]) > 1e-12:
            unique.append(x)
    return unique


def _crossings_logx(
    curve_small: list[tuple[float, float]],
    curve_large: list[tuple[float, float]],
    include_boundary_roots: bool,
) -> list[float]:
    if len(curve_small) != len(curve_large):
        raise ValueError("Distance curves have different point counts.")

    roots: list[float] = []
    for i in range(len(curve_small) - 1):
        p0, y0s = curve_small[i]
        p1, y1s = curve_small[i + 1]
        q0, y0l = curve_large[i]
        q1, y1l = curve_large[i + 1]
        if not (_close(p0, q0) and _close(p1, q1)):
            raise ValueError("Distance curves are on different e-grids.")
        if p0 <= 0.0 or p1 <= 0.0:
            raise ValueError("logx interpolation requires positive e values.")

        d0 = y0s - y0l
        d1 = y1s - y1l
        if d0 * d1 < 0.0:
            frac = -d0 / (d1 - d0)
            x0 = math.log(p0)
            x1 = math.log(p1)
            roots.append(math.exp(x0 + frac * (x1 - x0)))
            continue
        if include_boundary_roots and d0 == 0.0:
            roots.append(p0)
            continue
        if include_boundary_roots and d1 == 0.0:
            roots.append(p1)

    unique: list[float] = []
    for x in sorted(roots):
        if not unique or abs(x - unique[-1]) > 1e-12:
            unique.append(x)
    return unique


def _filter_rows(
    rows: list[dict],
    q_value: float | None,
    p_value: float | None,
    scheme: str | None,
) -> list[dict]:
    out = rows
    if q_value is not None:
        out = [r for r in out if abs(float(r["check_prob"]) - q_value) < 1e-15]
    if p_value is not None:
        out = [r for r in out if abs(float(r["pauli_prob"]) - p_value) < 1e-15]
    if scheme is not None:
        out = [r for r in out if str(r["scheme"]) == scheme]
    return out


def _build_curves(rows: list[dict], metric: str) -> dict[int, list[tuple[float, float]]]:
    curves: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        d = int(row["distance"])
        e = float(row["erasure_prob"])
        y = float(row[metric])
        curves[d].append((e, y))
    for d in curves:
        curves[d].sort(key=lambda t: t[0])
    return dict(curves)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Report pairwise crossings for sweep curves and compare boundary-aware "
            "vs interior-only threshold estimates."
        )
    )
    parser.add_argument("json_path", type=Path)
    parser.add_argument(
        "--metric",
        type=str,
        default="logical_error_rate_per_round",
        choices=["logical_error_rate_per_round", "logical_error_rate"],
    )
    parser.add_argument(
        "--interp",
        type=str,
        default="linear",
        choices=["linear", "logx"],
        help="Interpolation of crossing location within each e-segment.",
    )
    parser.add_argument("--scheme", type=str, default=None)
    parser.add_argument("--q", type=float, default=None)
    parser.add_argument("--p", type=float, default=None)
    args = parser.parse_args()

    payload = json.loads(args.json_path.read_text())
    rows = list(payload.get("rows", []))
    if not rows:
        raise ValueError("No rows found in input JSON.")
    rows = _filter_rows(rows, args.q, args.p, args.scheme)
    if not rows:
        raise ValueError("No rows left after filtering.")

    schemes = sorted(set(str(r["scheme"]) for r in rows))
    cross_fn = _crossings_linear if args.interp == "linear" else _crossings_logx

    print(f"input={args.json_path}")
    print(f"metric={args.metric}")
    print(f"interp={args.interp}")
    if args.q is not None or args.p is not None:
        print(f"filters: q={args.q} p={args.p}")
    if args.scheme is not None:
        print(f"filter scheme={args.scheme}")

    for scheme in schemes:
        srows = [r for r in rows if str(r["scheme"]) == scheme]
        curves = _build_curves(srows, args.metric)
        distances = sorted(curves.keys())
        if len(distances) < 2:
            print(f"\n[{scheme}] skipped: need >=2 distances, got {distances}")
            continue

        boundary_selected: list[float] = []
        interior_selected: list[float] = []

        print(f"\n[{scheme}]")
        print(f"distances={distances}")
        for i in range(len(distances) - 1):
            d_small = distances[i]
            d_large = distances[i + 1]
            c_with = cross_fn(
                curves[d_small], curves[d_large], include_boundary_roots=True
            )
            c_without = cross_fn(
                curves[d_small], curves[d_large], include_boundary_roots=False
            )

            selected_with = c_with[0] if c_with else None
            selected_without = c_without[0] if c_without else None

            if selected_with is not None:
                boundary_selected.append(float(selected_with))
            if selected_without is not None:
                interior_selected.append(float(selected_without))

            print(
                f"pair d={d_small}-{d_large} "
                f"crossings_with_boundary={c_with} selected_with={selected_with} "
                f"crossings_interior={c_without} selected_interior={selected_without}"
            )

        th_with = statistics.mean(boundary_selected) if boundary_selected else None
        th_without = statistics.mean(interior_selected) if interior_selected else None
        print(
            "threshold_with_boundary="
            f"{th_with} (pairs={len(boundary_selected)})"
        )
        print(
            "threshold_interior_only="
            f"{th_without} (pairs={len(interior_selected)})"
        )


if __name__ == "__main__":
    main()
