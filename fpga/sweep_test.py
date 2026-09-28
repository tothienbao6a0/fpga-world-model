import unittest

from fpga.sweep import run_sweep


class SweepTest(unittest.TestCase):
    def test_both_banks_and_environments_use_same_decision_count(self):
        rows = run_sweep(2, 3, 7, 2)
        self.assertEqual(len(rows), 4)
        self.assertEqual({row["decisions"] for row in rows}, {12})
        self.assertEqual({row["candidate_count"] for row in rows}, {9, 81})
        self.assertEqual({row["environment"] for row in rows}, {"linear", "nonlinear"})

    def test_empty_seed_set_rejected(self):
        with self.assertRaises(ValueError):
            run_sweep(2, 3, 7, 0)


if __name__ == "__main__":
    unittest.main()
