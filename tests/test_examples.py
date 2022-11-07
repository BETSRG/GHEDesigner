import subprocess
import sys
import unittest
from pathlib import Path


class TestExamples(unittest.TestCase):

    def test_examples(self):
        example_dir_path = Path(__file__).parent.parent / "examples"
        example_files = ["find_bi_polygon.py"]

        python_exe = Path(sys.executable)

        for f in example_files:
            f_path = example_dir_path / f
            try:
                subprocess.check_call([f"{python_exe}", f_path])
            except:
                self.assertFalse(True)
