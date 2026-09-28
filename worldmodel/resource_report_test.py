import unittest

from worldmodel.resource_report import parse_resources


class ResourceReportTest(unittest.TestCase):
    def test_parser_reads_final_local_counts(self):
        log = """=== qkv_tile ===
  1 cells
    1 DSP48E2
=== qkv_tile ===
  4 cells
    2 DSP48E2
    3 LUT6
    1 FDRE
"""
        self.assertEqual(parse_resources(log, "qkv_tile")["DSP48E2"], 2)
        self.assertEqual(parse_resources(log, "qkv_tile")["LUT_total"], 3)

    def test_parser_rejects_missing_target(self):
        with self.assertRaisesRegex(ValueError, "no final"):
            parse_resources("", "qkv_tile")


if __name__ == "__main__":
    unittest.main()
