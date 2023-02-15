import os
import unittest
from datetime import datetime
from pathlib import Path

from ghedesigner.manager import run_manager_from_cli_worker

demos_path = Path(__file__).parent.parent.parent / "demos"
demo_output_parent_dir = Path(__file__).parent.parent.parent / "demo_outputs"


class TestDemoFiles(unittest.TestCase):

    def test_demo_files(self):

        time_str = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")

        for _, _, files in os.walk(demos_path):
            for f in files:
                try:
                    demo_file_path = demos_path / f
                    out_dir = demo_output_parent_dir / time_str / f.replace('.json', '')
                    os.makedirs(out_dir)
                    run_manager_from_cli_worker(input_file_path=demo_file_path, output_directory=out_dir)
                except:
                    print(f"Demo file failed: {f}")
                    exit(1)
