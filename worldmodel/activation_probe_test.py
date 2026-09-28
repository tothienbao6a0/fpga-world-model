import unittest
from importlib.util import find_spec

from worldmodel.activation_probe import dequantized_outputs, error_stats, full_qkv_error_stats


class ActivationProbeTest(unittest.TestCase):
    def test_dequantization_uses_one_activation_scale_and_per_row_weight_scales(self):
        self.assertEqual(
            dequantized_outputs(((10, -10), (20, -20)), 0.5, (0.1, 0.2)),
            ((0.5, -1.0), (1.0, -2.0)),
        )

    def test_error_stats_compare_each_candidate_and_lane(self):
        result = error_stats(((1.0, 2.0), (3.0, 4.0)), ((1.0, 1.0), (3.0, 3.0)))
        self.assertEqual(result["values_compared"], 4)
        self.assertEqual(result["max_abs_error"], 1.0)
        self.assertAlmostEqual(result["rmse"], 2**0.5 / 2)

    def test_error_stats_reject_mismatched_candidate_count(self):
        with self.assertRaisesRegex(ValueError, "matching"):
            error_stats(((1.0,),), ((1.0,), (2.0,)))

    @unittest.skipUnless(find_spec("torch"), "model environment is not installed")
    def test_full_qkv_error_covers_all_candidates_and_tokens(self):
        import torch

        inputs = torch.tensor([[[1.0, 2.0], [2.0, 1.0]], [[2.0, 2.0], [1.0, 1.0]]])
        weights = torch.tensor([[1.0, 1.0]])
        biases = torch.tensor([0.0])
        reference = inputs.sum(dim=2, keepdim=True)
        for scope in ("token", "layer"):
            result = full_qkv_error_stats(inputs, reference, weights, biases, scope)
            self.assertEqual(result["values_compared"], 4)
            self.assertLess(result["relative_l2_error"], 0.01)
        with self.assertRaisesRegex(ValueError, "shapes must align"):
            full_qkv_error_stats(inputs, reference, weights[:, :1], biases, "layer")


if __name__ == "__main__":
    unittest.main()
