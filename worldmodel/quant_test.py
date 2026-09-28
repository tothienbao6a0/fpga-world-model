import unittest

from worldmodel.quant import quantize_candidate_tile, quantized_tile, symmetric_int8
from worldmodel.verify_rtl import max_abs_dequantized_error


class QuantTest(unittest.TestCase):
    def test_symmetric_quantization_preserves_zero_and_extrema(self):
        values, scale = symmetric_int8((-2.0, 0.0, 1.0, 2.0))
        self.assertEqual(values, (-127, 0, 64, 127))
        self.assertAlmostEqual(scale, 2 / 127)

    def test_integer_tile_includes_bias_and_negative_products(self):
        self.assertEqual(quantized_tile((2, -3), ((4, 5), (-2, 1)), (7, -1)), (0, -8))

    def test_error_uses_original_activations(self):
        self.assertEqual(
            max_abs_dequantized_error((0.5,), [[2.0]], [0.0], (0,), 1.0, (1.0,)),
            1.0,
        )

    def test_candidates_share_one_activation_scale_and_keep_bias(self):
        activations, weights, biases, scale, weight_scales = quantize_candidate_tile(
            ((1.0, -2.0), (2.0, -4.0)), ((0.5, -1.0),), (1.0,),
        )
        self.assertEqual(activations, ((32, -64), (64, -127)))
        self.assertEqual(weights, ((64, -127),))
        self.assertEqual(biases, (round(1.0 / (scale * weight_scales[0])),))

    def test_candidate_quantization_rejects_misaligned_rows(self):
        with self.assertRaisesRegex(ValueError, "same width"):
            quantize_candidate_tile(((1.0, 2.0), (3.0,)), ((1.0, 2.0),), (0.0,))

    def test_explicit_layer_scale_is_shared_and_clamps_outliers(self):
        activations, _, _, scale, _ = quantize_candidate_tile(
            ((0.5, 2.0), (4.0, -4.0)), ((1.0, 1.0),), (0.0,), activation_scale=0.01,
        )
        self.assertEqual(scale, 0.01)
        self.assertEqual(activations, ((50, 127), (127, -127)))


if __name__ == "__main__":
    unittest.main()
