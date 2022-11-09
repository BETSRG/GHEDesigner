from pathlib import Path
from unittest import TestCase


class GHEBaseTest(TestCase):
    def setUp(self) -> None:
        cur_file = Path(__file__).resolve()
        self.tests_directory = cur_file.parent
        self.test_data_directory = self.tests_directory / 'test_data'
        self.project_root_directory = self.tests_directory.parent
        self.test_outputs_directory = self.tests_directory / 'test_outputs'
        self.test_outputs_directory.mkdir(exist_ok=True)
