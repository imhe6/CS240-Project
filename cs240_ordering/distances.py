from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from .data import ViewSample


def build_distance_matrix(
    samples: Sequence[ViewSample],
    kind: str = "hybrid",
    pose_weight: float = 0.7,
    feature_weight: float = 0.3,
) -> np.ndarray:
    """Build a symmetric normalized dissimilarity matrix for posed views."""

    if not samples:
        raise ValueError("samples must not be empty")

    kind = kind.lower()
    if kind == "pose":
        return _normalize_off_diagonal(_pose_distance_matrix(samples))
    if kind == "feature":
        return _normalize_off_diagonal(_feature_distance_matrix(samples))
    if kind == "hybrid":
        if pose_weight < 0 or feature_weight < 0 or pose_weight + feature_weight <= 0:
            raise ValueError("pose_weight and feature_weight must be non-negative with positive sum")
        pose = _normalize_off_diagonal(_pose_distance_matrix(samples))
        feature = _normalize_off_diagonal(_feature_distance_matrix(samples))
        total = pose_weight + feature_weight
        hybrid = (pose_weight / total) * pose + (feature_weight / total) * feature
        return _normalize_off_diagonal(hybrid)

    raise ValueError("kind must be one of: pose, feature, hybrid")


def _pose_distance_matrix(samples: Sequence[ViewSample]) -> np.ndarray:
    n = len(samples)
    centers = np.stack([sample.camera_center for sample in samples])
    dirs = np.stack([sample.viewing_direction for sample in samples])
    cost = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            center_dist = np.linalg.norm(centers[i] - centers[j])
            cosine = np.clip(float(np.dot(dirs[i], dirs[j])), -1.0, 1.0)
            angle = np.arccos(cosine) / np.pi
            cost[i, j] = cost[j, i] = center_dist + angle
    return cost


def _feature_distance_matrix(samples: Sequence[ViewSample]) -> np.ndarray:
    features = [_extract_orb_features(sample.image) for sample in samples]
    n = len(samples)
    cost = np.zeros((n, n), dtype=np.float64)
    finite_values = []

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    for i in range(n):
        for j in range(i + 1, n):
            des_i = features[i]
            des_j = features[j]
            if des_i is None or des_j is None:
                dist = np.inf
            else:
                matches = matcher.match(des_i, des_j)
                dist = float(np.mean([match.distance for match in matches])) if matches else np.inf
            cost[i, j] = cost[j, i] = dist
            if np.isfinite(dist):
                finite_values.append(dist)

    if not finite_values:
        return _fallback_color_distance_matrix(samples)

    replacement = max(finite_values) * 1.25 + 1.0
    cost[~np.isfinite(cost)] = replacement
    np.fill_diagonal(cost, 0.0)
    return cost


def _extract_orb_features(image: np.ndarray) -> np.ndarray | None:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(nfeatures=500)
    _kp, descriptors = orb.detectAndCompute(gray, None)
    return descriptors


def _fallback_color_distance_matrix(samples: Sequence[ViewSample]) -> np.ndarray:
    histograms = []
    for sample in samples:
        image = sample.image[:, :, :3].astype(np.float32) / 255.0
        hist = []
        for channel in range(3):
            counts, _bins = np.histogram(image[:, :, channel], bins=16, range=(0.0, 1.0), density=True)
            hist.append(counts)
        histograms.append(np.concatenate(hist))

    n = len(samples)
    cost = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.linalg.norm(histograms[i] - histograms[j]))
            cost[i, j] = cost[j, i] = dist
    return cost


def _normalize_off_diagonal(cost: np.ndarray) -> np.ndarray:
    cost = np.asarray(cost, dtype=np.float64).copy()
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1]:
        raise ValueError("cost must be a square matrix")
    cost = (cost + cost.T) / 2.0
    np.fill_diagonal(cost, 0.0)

    mask = ~np.eye(cost.shape[0], dtype=bool)
    off_diag = cost[mask]
    finite = off_diag[np.isfinite(off_diag)]
    if finite.size == 0:
        cost[mask] = 1.0
        return cost

    replacement = finite.max() * 1.25 + 1.0
    cost[~np.isfinite(cost)] = replacement

    off_diag = cost[mask]
    min_value = off_diag.min()
    max_value = off_diag.max()
    if np.isclose(max_value, min_value):
        cost[mask] = 1.0 if max_value > 0 else 0.0
    else:
        cost[mask] = (off_diag - min_value) / (max_value - min_value)
    np.fill_diagonal(cost, 0.0)
    return cost
