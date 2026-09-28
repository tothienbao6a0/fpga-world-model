import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from worldbench.cli import main


class CliTest(unittest.TestCase):
    def test_emits_summary_and_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.jsonl"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = main(["--episodes", "2", "--steps", "2", "--horizon", "2", "--bits", "3", "--trace", str(trace)])
            summary = json.loads(output.getvalue())
            lines = [json.loads(line) for line in trace.read_text().splitlines()]
            self.assertEqual(result, 0)
            self.assertEqual(summary["results"]["3"]["decisions"], 4)
            self.assertEqual(len(lines), 12)
            self.assertIn("3+refine2", summary["results"])
            self.assertTrue(all(line["score_regret"] >= 0 for line in lines))

    def test_invalid_horizon_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                main(["--horizon", "0"])
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
