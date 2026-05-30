from __future__ import annotations

import time
import tracemalloc
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .algorithms import path_cost


def evaluate_ordering(cost: np.ndarray, order_or_subsequences: Sequence[Any]) -> dict[str, float | int]:
    """Compute algorithm-level smoothness and coverage metrics."""

    sequences = _as_sequences(order_or_subsequences)
    n = cost.shape[0]
    covered = [vertex for sequence in sequences for vertex in sequence]
    unique_covered = set(covered)
    adjacent_costs = [
        float(cost[sequence[i], sequence[i + 1]])
        for sequence in sequences
        for i in range(len(sequence) - 1)
    ]

    lengths = [len(sequence) for sequence in sequences]
    total = float(sum(path_cost(cost, sequence) for sequence in sequences))
    return {
        "total_path_cost": total,
        "max_adjacent_jump": float(max(adjacent_costs)) if adjacent_costs else 0.0,
        "mean_adjacent_cost": float(np.mean(adjacent_costs)) if adjacent_costs else 0.0,
        "std_adjacent_cost": float(np.std(adjacent_costs)) if adjacent_costs else 0.0,
        "coverage": float(len(unique_covered) / n) if n else 0.0,
        "num_subsequences": len(sequences),
        "min_subsequence_length": int(min(lengths)) if lengths else 0,
        "max_subsequence_length": int(max(lengths)) if lengths else 0,
        "duplicate_visits": int(len(covered) - len(unique_covered)),
    }


def timed_call(func: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, float, int]:
    """Run an algorithm while recording wall time and peak Python allocation."""

    tracemalloc.start()
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak


def _as_sequences(order_or_subsequences: Sequence[Any]) -> list[list[int]]:
    if not order_or_subsequences:
        return []

    first = order_or_subsequences[0]
    if isinstance(first, (list, tuple, np.ndarray)):
        return [list(map(int, sequence)) for sequence in order_or_subsequences]
    return [list(map(int, order_or_subsequences))]
