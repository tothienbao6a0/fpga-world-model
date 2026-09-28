import unittest

from worldbench.dynamics import fit_model, training_samples
from worldbench.planner import candidate_sequences, choose_action, choose_action_refined


class PlannerTest(unittest.TestCase):
    def setUp(self):
        self.model = fit_model(training_samples(256, 7))

    def test_candidate_count_and_deterministic_choice(self):
        candidates = candidate_sequences(3)
        self.assertEqual(len(candidates), 27)
        first = choose_action(self.model, (0.0, 0.0), 1.0, candidates)
        second = choose_action(self.model, (0.0, 0.0), 1.0, candidates)
        self.assertEqual(first, second)
        self.assertEqual(first.transitions, 81)

    def test_margin_compares_distinct_first_actions(self):
        candidates = candidate_sequences(3)
        decision = choose_action(self.model, (0.0, 0.0), 1.0, candidates)
        self.assertGreaterEqual(decision.margin, 0)
        self.assertEqual(decision.score_for_action(decision.action), decision.score)
        self.assertEqual(len(decision.first_action_scores), 3)

    def test_low_precision_can_change_the_selected_action(self):
        candidates = candidate_sequences(3)
        state = (-0.9206500566419157, -0.45622154261778847)
        full = choose_action(self.model, state, -1.25, candidates)
        low = choose_action(self.model, state, -1.25, candidates, 3)
        self.assertEqual(full.action, 0)
        self.assertEqual(low.action, 1)

    def test_refining_all_candidates_recovers_full_precision(self):
        candidates = candidate_sequences(3)
        state = (-0.9206500566419157, -0.45622154261778847)
        full = choose_action(self.model, state, -1.25, candidates)
        refined = choose_action_refined(self.model, state, -1.25, candidates, 3, 9)
        self.assertEqual(refined.action, full.action)
        self.assertEqual(refined.sequence, full.sequence)
        self.assertEqual(refined.transitions, 162)

    def test_refinement_rejects_empty_shortlist(self):
        with self.assertRaises(ValueError):
            choose_action_refined(self.model, (0, 0), 1, candidate_sequences(2), 3, 0)

    def test_invalid_horizon_and_empty_candidates(self):
        with self.assertRaises(ValueError):
            candidate_sequences(0)
        with self.assertRaises(ValueError):
            choose_action(self.model, (0, 0), 1, ())

    def test_margin_is_unknown_without_an_alternative_action(self):
        decision = choose_action(self.model, (0, 0), 1, ((1, 1), (1, 0)))
        self.assertIsNone(decision.margin)


if __name__ == "__main__":
    unittest.main()
