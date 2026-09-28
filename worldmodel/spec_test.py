import unittest

from worldmodel.spec import workload_macs, validate_weight_shapes


class SpecTest(unittest.TestCase):
    def test_attention_grows_faster_than_matrix_work_with_frames(self):
        short = workload_macs(2)
        full = workload_macs(4)
        self.assertEqual(short["tokens_per_sample"], 512)
        self.assertEqual(full["tokens_per_sample"], 1024)
        self.assertEqual(full["matrix_macs"], 2 * short["matrix_macs"])
        self.assertEqual(full["attention_macs"], 4 * short["attention_macs"])

    def test_checkpoint_shape_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "predictor_embed.weight"):
            validate_weight_shapes({"predictor_embed.weight": (1, 1)})


if __name__ == "__main__":
    unittest.main()
