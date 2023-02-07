import os
import unittest
import tempfile

from pathlib import Path

from ghedesigner.manager import run_manager_from_cli_worker

demos_path = Path(__file__).parent.parent.parent / "demos"


class TestDemoFiles(unittest.TestCase):

    def test_demo_files(self):
        for _, _, files in os.walk(demos_path):
            for demo_file in files:
                try:
                    demo_file_path = demos_path / demo_file
                    tmp_dir = Path(tempfile.mkdtemp())
                    out_file = tmp_dir / "out.json"
                    run_manager_from_cli_worker(demo_file_path, out_file)
                except:
                    print(f"Demo file failed: {demo_file}")
                    exit(1)
