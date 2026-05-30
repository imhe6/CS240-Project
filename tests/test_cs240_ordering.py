import unittest
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from cs240_ordering import (
    adaptive_subsequences,
    build_distance_matrix,
    evaluate_ordering,
    generate_synthetic_views,
    greedy_counterexample_matrix,
    greedy_order,
    held_karp_path,
    load_blender_views,
    path_cost,
    two_opt_order,
)
from cs240_ordering.synthetic import expected_counterexample_optimum


class CS240OrderingTests(unittest.TestCase):
    def test_held_karp_returns_known_counterexample_optimum(self):
        cost = greedy_counterexample_matrix()
        expected_order, expected_cost = expected_counterexample_optimum()

        order, total = held_karp_path(cost)

        self.assertEqual(order, expected_order)
        self.assertAlmostEqual(total, expected_cost)
        self.assertLess(total, path_cost(cost, greedy_order(cost, start=0)))

    def test_greedy_visits_every_node_once(self):
        cost = build_distance_matrix(generate_synthetic_views(n=10, seed=7), kind="pose")

        order = greedy_order(cost, start=0)

        self.assertEqual(len(order), 10)
        self.assertEqual(sorted(order), list(range(10)))

    def test_two_opt_never_increases_cost(self):
        cost = build_distance_matrix(generate_synthetic_views(n=12, seed=123), kind="hybrid")
        initial = greedy_order(cost, start=0)

        refined = two_opt_order(cost, initial)

        self.assertEqual(sorted(refined), list(range(12)))
        self.assertLessEqual(path_cost(cost, refined), path_cost(cost, initial) + 1e-12)

    def test_adaptive_subsequences_respect_thresholds_and_cover(self):
        cost = build_distance_matrix(generate_synthetic_views(n=9, seed=99), kind="pose")
        thresholds = [0.10, 0.25, 1.01]

        sequences = adaptive_subsequences(cost, thresholds)
        metrics = evaluate_ordering(cost, sequences)

        self.assertAlmostEqual(metrics["coverage"], 1.0)
        flattened = [idx for sequence in sequences for idx in sequence]
        self.assertEqual(sorted(flattened), list(range(9)))
        for sequence in sequences:
            for i in range(len(sequence) - 1):
                self.assertLessEqual(cost[sequence[i], sequence[i + 1]], max(thresholds) + 1e-12)

    def test_hybrid_distance_is_symmetric_finite_zero_diagonal_normalized(self):
        samples = generate_synthetic_views(n=8, seed=240)

        cost = build_distance_matrix(samples, kind="hybrid")

        self.assertEqual(cost.shape, (8, 8))
        self.assertTrue(np.all(np.isfinite(cost)))
        self.assertTrue(np.allclose(cost, cost.T))
        self.assertTrue(np.allclose(np.diag(cost), 0.0))
        self.assertGreaterEqual(cost.min(), 0.0)
        self.assertLessEqual(cost.max(), 1.0)

    def test_load_blender_views_reads_nerf_synthetic_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train").mkdir()
            Image.new("RGBA", (16, 16), (255, 0, 0, 255)).save(root / "train" / "r_0.png")
            Image.new("RGBA", (16, 16), (0, 255, 0, 255)).save(root / "train" / "r_1.png")
            transforms = {
                "camera_angle_x": 0.7,
                "frames": [
                    {"file_path": "train/r_0", "transform_matrix": np.eye(4).tolist()},
                    {
                        "file_path": "train/r_1",
                        "transform_matrix": (
                            np.array(
                                [
                                    [1.0, 0.0, 0.0, 1.0],
                                    [0.0, 1.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0, 0.0],
                                    [0.0, 0.0, 0.0, 1.0],
                                ]
                            ).tolist()
                        ),
                    },
                ],
            }
            (root / "transforms_train.json").write_text(__import__("json").dumps(transforms), encoding="utf-8")

            samples = load_blender_views(root, split="train")

            self.assertEqual([sample.name for sample in samples], ["r_0.png", "r_1.png"])
            self.assertEqual(samples[0].image.shape, (16, 16, 3))
            self.assertTrue(np.allclose(samples[1].camera_center, [1.0, 0.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
