import unittest

from worldbench.dynamics import fit_model, quantize, training_samples


class DynamicsTest(unittest.TestCase):
    def test_learned_model_predicts_held_out_transitions(self):
        model = fit_model(training_samples(256, 7))
        for state, action, expected in training_samples(30, 99):
            actual = model.predict(state, action)
            for got, want in zip(actual, expected):
                self.assertAlmostEqual(got, want, places=7)

    def test_quantization_changes_step_and_rejects_negative_bits(self):
        self.assertEqual(quantize(0.31, 2), 0.25)
        with self.assertRaises(ValueError):
            quantize(1, -1)

    def test_empty_training_data_rejected(self):
        with self.assertRaises(ValueError):
            fit_model([])


if __name__ == "__main__":
    unittest.main()
