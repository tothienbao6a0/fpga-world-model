import unittest

from worldmodel.verify_candidate_rtl import candidate_vectors, testbench, traffic_bytes


class CandidateTileTest(unittest.TestCase):
    def test_vectors_are_distinct_and_clamped_to_int8(self):
        vectors = candidate_vectors((127, -127, 5, 0, -4), 4)
        self.assertEqual(len(vectors), 4)
        self.assertEqual(vectors[0], (127, -127, 5, 0, -4))
        self.assertEqual(len(set(vectors)), 4)
        self.assertTrue(all(-127 <= value <= 127 for row in vectors for value in row))

    def test_broadcast_reduces_weight_bytes_but_not_arithmetic(self):
        traffic = traffic_bytes(16, 400, 4)
        self.assertEqual(traffic["serial_weight_bytes"], 25600)
        self.assertEqual(traffic["broadcast_weight_bytes"], 6400)
        self.assertEqual(traffic["macs_both"], 25600)

    def test_testbench_rejects_missing_weights(self):
        with self.assertRaisesRegex(ValueError, "nonempty"):
            testbench(((1, 2),), (), ())


if __name__ == "__main__":
    unittest.main()
