import unittest

from worldmodel.quant import quantized_tile, symmetric_int8
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


if __name__ == "__main__":
    unittest.main()
