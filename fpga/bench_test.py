import unittest

from fpga.bench import candidate_bank, run_benchmark


class BenchTest(unittest.TestCase):
    def test_bank_covers_all_first_actions(self):
        bank = candidate_bank()
        self.assertEqual(len(bank), 9)
        self.assertEqual({sequence[0] for sequence in bank}, {-1, 0, 1})

    def test_closed_loop_contract_runs(self):
        result = run_benchmark(2, 3, 7, False)
        self.assertEqual(result["decisions"], 6)
        self.assertEqual(result["ideal_stream_cycles_per_decision"], 37)
        self.assertGreaterEqual(result["hardware_contract_mean_final_cost"], 0)


if __name__ == "__main__":
    unittest.main()
