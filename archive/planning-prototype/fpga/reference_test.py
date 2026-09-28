import unittest

from fpga.reference import dot4, evaluate, pack_weights, to_q


class ReferenceTest(unittest.TestCase):
    def test_q88_conversion_and_saturation(self):
        self.assertEqual(to_q(1.25), 320)
        self.assertEqual(dot4((256, 0, 0, 0), 40000, 0, 0), 32767)
        with self.assertRaises(ValueError):
            to_q(128)

    def test_packed_weights_keep_signed_fields(self):
        packed = pack_weights((256, -128, 0, 1))
        self.assertEqual((packed >> 16) & 0xFFFF, 0xFF80)

    def test_oracle_selects_lower_cost_first_action(self):
        weights_p = (256, 0, 256, 0)
        weights_v = (0, 0, 0, 0)
        result = evaluate(0, 0, 256, weights_p, weights_v, ((-1,), (0,), (1,)))
        self.assertEqual(result.action, 1)
        self.assertEqual(result.cost, 2621)


if __name__ == "__main__":
    unittest.main()
