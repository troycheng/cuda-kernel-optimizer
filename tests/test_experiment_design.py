from __future__ import annotations

import importlib.util
import random
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "skills" / "cuda-kernel-optimizer" / "scripts" / "experiment_design.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("experiment_design", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BalancedPairOrdersTests(unittest.TestCase):
    def test_schedule_is_balanced_and_reproducible(self) -> None:
        module = _load_module()
        for blocks in (5, 12):
            first = module.balanced_pair_orders(blocks, seed=17)
            self.assertEqual(first, module.balanced_pair_orders(blocks, seed=17))
            self.assertEqual(len(first), blocks)
            self.assertLessEqual(abs(first.count("AB") - first.count("BA")), 1)

    def test_validation_does_not_mutate_global_rng(self) -> None:
        module = _load_module()
        random.seed(99)
        before = random.getstate()
        module.balanced_pair_orders(4, seed=3)
        self.assertEqual(random.getstate(), before)
        for blocks in (0, -1, True, 1.5):
            with self.subTest(blocks=blocks), self.assertRaises(ValueError):
                module.balanced_pair_orders(blocks)
        for seed in (True, 1.5):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                module.balanced_pair_orders(2, seed=seed)


if __name__ == "__main__":
    unittest.main()
