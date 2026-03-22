#!/usr/bin/env python3
"""Find pairwise curve crossings from sweep JSON artifacts."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def _close(a: float, b: float, tol: float = 1e-15) -> bool:
	return abs(a - b) <= tol


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


def _interp_linear(curve: list[tuple[float, float]], x: float) -> float:
	if x < curve[0][0] or x > curve[-1][0]:
		raise ValueError("Interpolation point is outside the curve domain.")
	for i in range(len(curve) - 1):
		x0, y0 = curve[i]
		x1, y1 = curve[i + 1]
		if _close(x, x0):
			return y0
		if _close(x, x1):
			return y1
		if x0 <= x <= x1:
			t = (x - x0) / (x1 - x0)
			return y0 + t * (y1 - y0)
	raise ValueError("Interpolation segment was not found.")


def _interp_logx(curve: list[tuple[float, float]], x: float) -> float:
	if x <= 0.0:
		raise ValueError("logx interpolation requires positive x.")
	if x < curve[0][0] or x > curve[-1][0]:
		raise ValueError("Interpolation point is outside the curve domain.")
	for i in range(len(curve) - 1):
		x0, y0 = curve[i]
		x1, y1 = curve[i + 1]
		if _close(x, x0):
			return y0
		if _close(x, x1):
			return y1
		if x0 <= x <= x1:
			if x0 <= 0.0 or x1 <= 0.0:
				raise ValueError("logx interpolation requires positive grid points.")
			t = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
			return y0 + t * (y1 - y0)
	raise ValueError("Interpolation segment was not found.")


def _crossings(
	curve_small: list[tuple[float, float]],
	curve_large: list[tuple[float, float]],
	interp: str,
	include_endpoints: bool,
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
			t = -d0 / (d1 - d0)
			if interp == "linear":
				roots.append(p0 + t * (p1 - p0))
			else:
				if p0 <= 0.0 or p1 <= 0.0:
					raise ValueError("logx interpolation requires positive e values.")
				roots.append(math.exp(math.log(p0) + t * (math.log(p1) - math.log(p0))))
			continue
		if include_endpoints and d0 == 0.0:
			roots.append(p0)
		if include_endpoints and d1 == 0.0:
			roots.append(p1)

	unique: list[float] = []
	for x in sorted(roots):
		if not unique or abs(x - unique[-1]) > 1e-12:
			unique.append(x)
	return unique


def _nearest_three_way_meet(
	curves: dict[int, list[tuple[float, float]]],
	distances: list[int],
	interp: str,
	include_endpoints: bool,
) -> tuple[float, float, float] | None:
	if len(distances) != 3:
		return None
	d1, d2, d3 = distances
	r12 = _crossings(curves[d1], curves[d2], interp, include_endpoints)
	r23 = _crossings(curves[d2], curves[d3], interp, include_endpoints)
	if not r12 or not r23:
		return None

	best: tuple[float, float, float] | None = None
	for x12 in r12:
		for x23 in r23:
			gap = abs(x12 - x23)
			xm = 0.5 * (x12 + x23)
			if best is None or gap < best[0]:
				best = (gap, x12, x23)
	assert best is not None
	_, x12, x23 = best
	xm = 0.5 * (x12 + x23)

	interp_fn = _interp_linear if interp == "linear" else _interp_logx
	y1 = interp_fn(curves[d1], xm)
	y2 = interp_fn(curves[d2], xm)
	y3 = interp_fn(curves[d3], xm)
	spread = max(y1, y2, y3) - min(y1, y2, y3)
	return xm, abs(x12 - x23), spread


def main() -> None:
	parser = argparse.ArgumentParser(description="Find pairwise crossings in sweep data.")
	parser.add_argument("json_path", type=Path)
	parser.add_argument(
		"--metric",
		type=str,
		default="logical_error_rate",
		choices=["logical_error_rate", "logical_error_rate_per_round"],
	)
	parser.add_argument("--q", type=float, default=None)
	parser.add_argument("--p", type=float, default=None)
	parser.add_argument("--scheme", type=str, default=None)
	parser.add_argument(
		"--interp", type=str, default="linear", choices=["linear", "logx"]
	)
	parser.add_argument(
		"--include-endpoints",
		action="store_true",
		help="Include exact equality at sampled endpoints as crossings.",
	)
	parser.add_argument(
		"--out-json",
		type=Path,
		default=None,
		help="Optional output path for machine-readable results.",
	)
	args = parser.parse_args()

	payload = json.loads(args.json_path.read_text())
	rows = list(payload.get("rows", []))
	if not rows:
		raise ValueError("No rows found in input JSON.")

	rows = _filter_rows(rows, args.q, args.p, args.scheme)
	if not rows:
		raise ValueError("No rows left after filtering.")

	schemes = sorted(set(str(r["scheme"]) for r in rows))
	result: dict[str, dict] = {}

	print(f"input={args.json_path}")
	print(f"metric={args.metric}")
	print(f"interp={args.interp}")
	print(f"include_endpoints={args.include_endpoints}")
	if args.q is not None or args.p is not None:
		print(f"filters: q={args.q} p={args.p}")
	if args.scheme is not None:
		print(f"filter scheme={args.scheme}")

	for scheme in schemes:
		srows = [r for r in rows if str(r["scheme"]) == scheme]
		curves = _build_curves(srows, args.metric)
		distances = sorted(curves.keys())
		if len(distances) < 2:
			continue

		scheme_out: dict[str, object] = {"distances": distances, "pairs": []}
		print(f"\n[{scheme}]")
		print(f"distances={distances}")

		for i in range(len(distances) - 1):
			d_small = distances[i]
			d_large = distances[i + 1]
			roots = _crossings(
				curves[d_small],
				curves[d_large],
				args.interp,
				args.include_endpoints,
			)
			print(f"pair d={d_small}-{d_large} crossings={roots}")
			(scheme_out["pairs"]).append(
				{"d_small": d_small, "d_large": d_large, "crossings": roots}
			)

		meet = _nearest_three_way_meet(
			curves, distances, args.interp, args.include_endpoints
		)
		if meet is None:
			print("three_way_meet: none (missing pairwise crossings)")
			scheme_out["three_way_meet"] = None
		else:
			e_meet, pair_gap, y_spread = meet
			print(
				f"three_way_meet_estimate e={e_meet} "
				f"(pair_gap={pair_gap}, y_spread={y_spread})"
			)
			scheme_out["three_way_meet"] = {
				"e": e_meet,
				"pair_gap": pair_gap,
				"y_spread": y_spread,
			}

		result[scheme] = scheme_out

	if args.out_json is not None:
		args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True))
		print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
	main()
