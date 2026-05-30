#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cs240_ordering import (
    adaptive_subsequences,
    build_distance_matrix,
    evaluate_ordering,
    generate_synthetic_views,
    greedy_order,
    held_karp_path,
    load_blender_views,
    path_cost,
    two_opt_order,
    windowed_dp_order,
)
from cs240_ordering.metrics import timed_call
from cs240_ordering.plotting import save_adjacency_plot, save_sequence_strip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CS240 multi-view sequence-ordering experiments.")
    parser.add_argument(
        "--data",
        choices=["synthetic", "blender"],
        default="synthetic",
        help="dataset source: generated synthetic views or NeRF Synthetic/Blender scene",
    )
    parser.add_argument("--source", type=Path, default=None, help="Blender scene directory for --data blender")
    parser.add_argument("--split", default="train", help="Blender split name, e.g. train, val, or test")
    parser.add_argument("--n", type=int, default=12, help="number of synthetic views")
    parser.add_argument("--seed", type=int, default=240, help="random seed for synthetic data")
    parser.add_argument("--limit", type=int, default=None, help="limit number of loaded Blender views")
    parser.add_argument("--white-background", action="store_true", help="composite RGBA Blender images on white")
    parser.add_argument(
        "--distance",
        choices=["pose", "feature", "hybrid", "all"],
        default="all",
        help="distance matrix type to evaluate",
    )
    parser.add_argument("--out", type=Path, default=Path("output/cs240_ordering"), help="output directory")
    parser.add_argument("--dp-limit", type=int, default=16, help="maximum N for exact Held-Karp DP")
    parser.add_argument("--window-size", type=int, default=8, help="window size for windowed DP refinement")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.20, 0.35, 1.01],
        help="normalized thresholds for adaptive subsequence construction",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    samples = _load_samples(args)
    distance_kinds = ["pose", "feature", "hybrid"] if args.distance == "all" else [args.distance]

    rows: list[dict[str, Any]] = []
    orders: dict[str, Any] = {
        "config": {
            "data": args.data,
            "source": str(args.source) if args.source is not None else None,
            "split": args.split,
            "n": args.n,
            "num_samples": len(samples),
            "seed": args.seed,
            "distance": args.distance,
            "dp_limit": args.dp_limit,
            "window_size": args.window_size,
            "thresholds": args.thresholds,
        },
        "results": {},
    }

    for kind in distance_kinds:
        cost = build_distance_matrix(samples, kind=kind)
        np.save(args.out / f"cost_{kind}.npy", cost)
        orders["results"][kind] = {}

        greedy_result, greedy_seconds, greedy_peak = timed_call(greedy_order, cost, 0)
        algorithms: list[tuple[str, Any, float, int]] = [
            ("naive", list(range(args.n)), 0.0, 0),
            ("greedy", greedy_result, greedy_seconds, greedy_peak),
        ]

        adaptive_result, adaptive_seconds, adaptive_peak = timed_call(adaptive_subsequences, cost, args.thresholds)
        algorithms.append(("adaptive", adaptive_result, adaptive_seconds, adaptive_peak))

        if args.n <= args.dp_limit:
            dp_result, dp_seconds, dp_peak = timed_call(held_karp_path, cost, args.dp_limit)
            dp_order, _dp_cost = dp_result
            algorithms.append(("held_karp", dp_order, dp_seconds, dp_peak))

        two_opt_result, two_opt_seconds, two_opt_peak = timed_call(two_opt_order, cost, greedy_result)
        algorithms.append(("greedy_2opt", two_opt_result, two_opt_seconds, two_opt_peak))

        if args.window_size <= args.dp_limit:
            window_result, window_seconds, window_peak = timed_call(
                windowed_dp_order,
                cost,
                greedy_result,
                args.window_size,
            )
            algorithms.append(("greedy_windowed_dp", window_result, window_seconds, window_peak))

        for algorithm_name, result, seconds, peak_bytes in algorithms:
            metrics = evaluate_ordering(cost, result)
            row = {
                "distance": kind,
                "algorithm": algorithm_name,
                "runtime_seconds": f"{seconds:.8f}",
                "peak_memory_bytes": peak_bytes,
                **_format_metrics(metrics),
            }
            rows.append(row)
            orders["results"][kind][algorithm_name] = {
                "order_or_subsequences": result,
                "metrics": metrics,
                "runtime_seconds": seconds,
                "peak_memory_bytes": peak_bytes,
            }

            if _is_single_order(result):
                order = list(map(int, result))
                safe_name = f"{kind}_{algorithm_name}"
                save_adjacency_plot(
                    cost,
                    order,
                    title=f"{kind} / {algorithm_name} adjacency costs",
                    path=args.out / "plots" / f"{safe_name}_adjacency.png",
                )
                save_sequence_strip(samples, order, args.out / "strips" / f"{safe_name}_sequence.png")

    _write_csv(args.out / "metrics.csv", rows)
    with (args.out / "orders.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe(orders), f, indent=2)

    print(f"Wrote CS240 ordering experiment outputs to {args.out}")
    print(_summary_table(rows))


def _load_samples(args: argparse.Namespace):
    if args.data == "synthetic":
        return generate_synthetic_views(n=args.n, seed=args.seed)

    if args.source is None:
        raise ValueError("--source is required when --data blender")
    return load_blender_views(
        args.source,
        split=args.split,
        limit=args.limit,
        white_background=args.white_background,
    )


def _format_metrics(metrics: dict[str, float | int]) -> dict[str, str | int]:
    formatted: dict[str, str | int] = {}
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted[key] = f"{value:.8f}"
        else:
            formatted[key] = value
    return formatted


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _is_single_order(result: Any) -> bool:
    return bool(result) and not isinstance(result[0], (list, tuple))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _summary_table(rows: list[dict[str, Any]]) -> str:
    headers = ["distance", "algorithm", "total_path_cost", "max_adjacent_jump", "runtime_seconds"]
    lines = [" | ".join(headers), " | ".join(["-" * len(header) for header in headers])]
    for row in rows:
        lines.append(" | ".join(str(row[header]) for header in headers))
    return "\n".join(lines)


if __name__ == "__main__":
    main()
