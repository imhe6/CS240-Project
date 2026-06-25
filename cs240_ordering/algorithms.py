from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def path_cost(cost: np.ndarray, order: Sequence[int]) -> float:
    if len(order) < 2:
        return 0.0
    return float(sum(cost[order[i], order[i + 1]] for i in range(len(order) - 1)))


def greedy_order(cost: np.ndarray, start: int = 0) -> list[int]:
    cost = _validate_cost(cost)
    n = cost.shape[0]
    if not 0 <= start < n:
        raise ValueError("start must be a valid vertex index")

    unvisited = set(range(n))
    order = [start]
    unvisited.remove(start)
    while unvisited:
        current = order[-1]
        next_vertex = min(unvisited, key=lambda idx: (cost[current, idx], idx))
        order.append(next_vertex)
        unvisited.remove(next_vertex)
    return order


def adaptive_subsequences(cost: np.ndarray, thresholds: Iterable[float]) -> list[list[int]]:
    cost = _validate_cost(cost)
    thresholds = list(thresholds)
    if not thresholds:
        raise ValueError("thresholds must not be empty")

    uncovered = set(range(cost.shape[0]))
    sequences: list[list[int]] = []
    for threshold in thresholds:
        if not uncovered:
            break

        starts = sorted(uncovered)
        for start in starts:
            if start not in uncovered:
                continue
            sequence = [start]
            uncovered.remove(start)
            current = start

            while uncovered:
                candidate = min(uncovered, key=lambda idx: (cost[current, idx], idx))
                if cost[current, candidate] > threshold:
                    break
                sequence.append(candidate)
                uncovered.remove(candidate)
                current = candidate

            sequences.append(sequence)

    for vertex in sorted(uncovered):
        sequences.append([vertex])
    return sequences


def held_karp_path(cost: np.ndarray, max_n: int = 16) -> tuple[list[int], float]:
    cost = _validate_cost(cost)
    n = cost.shape[0]
    if n > max_n:
        print(f"Held-Karp DP is limited to N <= {max_n}, got {n}, skipping")
        return [0], float("inf")
    if n == 1:
        return [0], 0.0

    dp: dict[tuple[int, int], float] = {}
    parent: dict[tuple[int, int], int] = {}
    for j in range(n):
        dp[(1 << j, j)] = 0.0

    for mask in range(1, 1 << n):
        if mask & (mask - 1) == 0:
            continue
        for j in range(n):
            if not mask & (1 << j):
                continue
            prev_mask = mask ^ (1 << j)
            best_cost = np.inf
            best_prev = -1
            for i in range(n):
                if not prev_mask & (1 << i):
                    continue
                candidate = dp[(prev_mask, i)] + cost[i, j]
                if candidate < best_cost:
                    best_cost = candidate
                    best_prev = i
            dp[(mask, j)] = float(best_cost)
            parent[(mask, j)] = best_prev

    full_mask = (1 << n) - 1
    end = min(range(n), key=lambda j: (dp[(full_mask, j)], j))
    best = dp[(full_mask, end)]

    order = [end]
    mask = full_mask
    current = end
    while mask & (mask - 1):
        previous = parent[(mask, current)]
        order.append(previous)
        mask ^= 1 << current
        current = previous

    order.reverse()
    return order, float(best)


def windowed_dp_order(cost: np.ndarray, initial_order: Sequence[int], window_size: int = 8) -> list[int]:
    cost = _validate_cost(cost)
    order = list(initial_order)
    _validate_order(cost, order)
    if window_size < 2:
        raise ValueError("window_size must be at least 2")

    previous_total = path_cost(cost, order)
    for start in range(0, len(order), max(1, window_size // 2)):
        end = min(len(order), start + window_size)
        if end - start < 3:
            continue
        window = order[start:end]
        sub_cost = cost[np.ix_(window, window)]
        local_order, _ = held_karp_path(sub_cost, max_n=window_size)
        candidate_window = [window[i] for i in local_order]
        candidate = order[:start] + candidate_window + order[end:]
        candidate_total = path_cost(cost, candidate)
        if candidate_total <= previous_total:
            order = candidate
            previous_total = candidate_total
    return order


def two_opt_order(cost: np.ndarray, initial_order: Sequence[int]) -> list[int]:
    cost = _validate_cost(cost)
    order = list(initial_order)
    _validate_order(cost, order)
    n = len(order)
    if n < 4:
        return order

    improved = True
    while improved:
        improved = False
        for i in range(n - 2):
            for k in range(i + 2, n):
                before = cost[order[i], order[i + 1]]
                after = 0.0 if k == n - 1 else cost[order[k], order[k + 1]]
                old = before + after

                new_first = cost[order[i], order[k]]
                new_second = 0.0 if k == n - 1 else cost[order[i + 1], order[k + 1]]
                new = new_first + new_second
                if new + 1e-12 < old:
                    order[i + 1 : k + 1] = reversed(order[i + 1 : k + 1])
                    improved = True
                    break
            if improved:
                break
    return order


def _validate_cost(cost: np.ndarray) -> np.ndarray:
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1]:
        raise ValueError("cost must be a square matrix")
    if not np.all(np.isfinite(cost)):
        raise ValueError("cost must contain only finite values")
    return cost


def _validate_order(cost: np.ndarray, order: Sequence[int]) -> None:
    n = cost.shape[0]
    if sorted(order) != list(range(n)):
        raise ValueError("order must visit every vertex exactly once")
