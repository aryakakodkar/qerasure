#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    if "rows" not in payload:
        raise ValueError(f"{path} does not contain a 'rows' field.")
    return list(payload["rows"])


def _build_curves(rows: list[dict], metric: str) -> dict[int, list[tuple[float, float]]]:
    curves: dict[int, list[tuple[float, float]]] = {}
    for row in rows:
        d = int(row["distance"])
        p = float(row["p_two_qubit_erasure"])
        y = float(row[metric])
        curves.setdefault(d, []).append((p, y))

    for d in curves:
        curves[d].sort(key=lambda t: t[0])
    return curves


def _crossings_linear(
    curve_small: list[tuple[float, float]],
    curve_large: list[tuple[float, float]],
) -> list[float]:
    # Assumes matched p-grid (true for current sweep format).
    if len(curve_small) != len(curve_large):
        raise ValueError("Distance curves have different point counts; interpolation not implemented.")

    crossings: list[float] = []
    for i in range(len(curve_small) - 1):
        p0, y0s = curve_small[i]
        p1, y1s = curve_small[i + 1]
        q0, y0l = curve_large[i]
        q1, y1l = curve_large[i + 1]
        if abs(p0 - q0) > 1e-15 or abs(p1 - q1) > 1e-15:
            raise ValueError("Distance curves use different p grids; interpolation not implemented.")

        d0 = y0s - y0l
        d1 = y1s - y1l

        if d0 == 0.0:
            crossings.append(p0)
            continue
        if d0 * d1 < 0.0:
            # Linear interpolation of the sign change of Δ(p)=L_small(p)-L_large(p).
            frac = -d0 / (d1 - d0)
            crossings.append(p0 + frac * (p1 - p0))
        elif d1 == 0.0:
            crossings.append(p1)

    # Deduplicate near-identical roots from endpoint handling.
    unique: list[float] = []
    for x in crossings:
        if not unique or abs(x - unique[-1]) > 1e-12:
            unique.append(x)
    return unique


def estimate_threshold(rows: list[dict], metric: str) -> dict:
    curves = _build_curves(rows, metric)
    distances = sorted(curves.keys())
    if len(distances) < 2:
        raise ValueError("Need at least two code distances to estimate a threshold.")

    pair_results: list[dict] = []
    pair_points: list[float] = []
    for i in range(len(distances) - 1):
        d_small = distances[i]
        d_large = distances[i + 1]
        crossings = _crossings_linear(curves[d_small], curves[d_large])
        pair_results.append(
            {
                "pair": [d_small, d_large],
                "crossings": crossings,
                "selected": crossings[0] if crossings else None,
            }
        )
        if crossings:
            pair_points.append(crossings[0])

    # Literature-standard finite-size crossing estimate:
    # use crossings of logical-error curves for increasing distance.
    #
    # References:
    # - E. Dennis, A. Kitaev, A. Landahl, and J. Preskill,
    #   "Topological quantum memory", J. Math. Phys. 43, 4452 (2002).
    # - C. Wang, J. Harrington, and J. Preskill,
    #   "Confinement-Higgs transition in a disordered gauge theory and
    #   the accuracy threshold for quantum memory", Ann. Phys. 303, 31-58 (2003).
    threshold = statistics.mean(pair_points) if pair_points else None
    spread = statistics.pstdev(pair_points) if len(pair_points) > 1 else 0.0

    return {
        "metric": metric,
        "distances": distances,
        "pairwise_crossings": pair_results,
        "threshold_estimate": threshold,
        "threshold_spread": spread,
        "num_pairs_with_crossing": len(pair_points),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate threshold from sweep JSON via finite-size curve crossings."
    )
    parser.add_argument("json_path", type=Path, help="Path to sweep JSON (with rows).")
    parser.add_argument(
        "--metric",
        type=str,
        default="logical_error_rate_per_round",
        choices=["logical_error_rate_per_round", "logical_error_rate"],
        help="Y-axis metric used for crossing.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional output JSON path for threshold summary.",
    )
    args = parser.parse_args()

    rows = _load_rows(args.json_path)
    result = estimate_threshold(rows, args.metric)

    print(f"input: {args.json_path}")
    print(f"metric: {result['metric']}")
    print(f"distances: {result['distances']}")
    for pair in result["pairwise_crossings"]:
        print(
            f"pair d={pair['pair'][0]}-{pair['pair'][1]} "
            f"crossings={pair['crossings']} selected={pair['selected']}"
        )
    print(f"threshold_estimate: {result['threshold_estimate']}")
    print(f"threshold_spread: {result['threshold_spread']}")
    print(f"num_pairs_with_crossing: {result['num_pairs_with_crossing']}")

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(result, indent=2))
        print(f"saved: {args.save}")


if __name__ == "__main__":
    main()
