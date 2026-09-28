import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from worldmodel.bench import verify_source
from worldmodel.spec import MODEL


class BenchSourceTest(unittest.TestCase):
    def test_dirty_upstream_is_rejected_even_at_pinned_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "app/plan_common/models/AdaLN_vit.py"
            source.parent.mkdir(parents=True)
            source.write_text("", encoding="utf-8")
            with patch("worldmodel.bench.subprocess.run", side_effect=[
                SimpleNamespace(stdout=MODEL.upstream_commit + "\n"),
                SimpleNamespace(stdout=" M app/plan_common/models/AdaLN_vit.py\n"),
            ]):
                with self.assertRaisesRegex(ValueError, "local modifications"):
                    verify_source(Path(directory))


if __name__ == "__main__":
    unittest.main()
