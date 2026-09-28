import unittest

from worldbench.dynamics import fit_model, training_samples
from worldbench.experiment import initial_conditions, run_episode, summarize


class ExperimentTest(unittest.TestCase):
    def test_paired_episode_has_decision_traces(self):
        model = fit_model(training_samples(256, 7))
        state, target = initial_conditions(1, 8)[0]
        traces, final_cost = run_episode(model, state, target, 0, 4, 3, 3)
        self.assertEqual(len(traces), 4)
        self.assertTrue(all(trace.bits == 3 and trace.transitions == 81 for trace in traces))
        self.assertGreaterEqual(final_cost, 0)
        self.assertEqual(summarize(traces, [final_cost])["decisions"], 4)

    def test_invalid_episode_count(self):
        with self.assertRaises(ValueError):
            initial_conditions(0, 1)

    def test_episode_uses_selected_environment_dynamics(self):
        model = fit_model(training_samples(256, 7))
        frozen = lambda state, action: state
        _, cost = run_episode(model, (0.25, 0.0), 1.0, 0, 3, 2, None, dynamics=frozen)
        self.assertAlmostEqual(cost, 0.75 ** 2)


if __name__ == "__main__":
    unittest.main()
