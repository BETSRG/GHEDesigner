from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import ClassVar
from unittest import TestCase

LOG_FILE: Path | None = None

time_str = datetime.now().strftime("%Y%m%d_%H%M%S")


class GHEBaseTest(TestCase):
    # Type hints for mypy since these are dynamically created attributes
    test_data_directory: ClassVar[Path]
    tests_directory: ClassVar[Path]
    project_root_directory: ClassVar[Path]
    test_outputs_directory: ClassVar[Path]
    demos_path: ClassVar[Path]
    demo_output_parent_dir: ClassVar[Path]

    class LogMessageTypes(Enum):
        Debug = "DEBUG"
        Info = "INFO"
        Warning = "WARNING"
        Error = "ERROR"

        @staticmethod
        def get_string(message_type_enum) -> str:
            strs = {
                GHEBaseTest.LogMessageTypes.Debug: "DEBUG",
                GHEBaseTest.LogMessageTypes.Info: "INFO",
                GHEBaseTest.LogMessageTypes.Warning: "WARNING",
                GHEBaseTest.LogMessageTypes.Error: "ERROR",
            }
            return strs[message_type_enum]

    def setup_log_file(self) -> None:
        global LOG_FILE  # noqa: PLW0603
        cur_file = Path(__file__).resolve()
        tests_directory = cur_file.parent
        log_directory = tests_directory / "test_logs"
        log_directory.mkdir(exist_ok=True)
        date_time_string = datetime.now().strftime("%Y%m%d_%H%M%S")
        LOG_FILE = log_directory / f"{date_time_string}.log"
        self.log("Tests Initialized")

    @classmethod
    def setUpClass(cls: type[GHEBaseTest]) -> None:
        cur_file = Path(__file__).resolve()
        cls.tests_directory = cur_file.parent
        cls.test_data_directory = cls.tests_directory / "test_data"
        cls.project_root_directory = cls.tests_directory.parent
        cls.test_outputs_directory = cls.tests_directory / "test_outputs" / time_str
        cls.test_outputs_directory.mkdir(exist_ok=True, parents=True)
        cls.demos_path = Path(__file__).parent.parent.parent / "demos"
        cls.demo_output_parent_dir = Path(__file__).parent.parent.parent / "demo_outputs"

    def log(self, message, message_type: LogMessageTypes = LogMessageTypes.Info):
        if LOG_FILE is None:
            self.setup_log_file()
        date_time_string = datetime.now().strftime("%Y%m%d_%H%M%S")
        message_type_string = GHEBaseTest.LogMessageTypes.get_string(message_type)
        message_string = str(message).strip()
        if LOG_FILE:
            with LOG_FILE.open("a") as fp:
                fp.write(f"{date_time_string},{message_type_string},{message_string}\n")

    def get_atlanta_loads(self) -> list[float]:
        # read in the csv file and convert the loads to a list of length 8760
        glhe_json_data = self.test_data_directory / "Atlanta_Office_Building_Loads.csv"
        raw_lines = glhe_json_data.read_text().split("\n")
        return [float(x) for x in raw_lines[1:] if x.strip() != ""]

    def get_multiyear_loads(self) -> list[float]:
        # read in the csv file and convert the loads to a list of length 8760
        glhe_json_data = self.test_data_directory / "Multiyear_Loading_Example.csv"
        raw_lines = glhe_json_data.read_text().split("\n")
        return [float(x) for x in raw_lines[1:] if x.strip() != ""]

    @staticmethod
    def rel_error_within_tol(test: float, base: float, tol: float) -> bool:
        return abs((test - base) / base) <= tol

    def assert_value_matches_any(
        self,
        actual_value: float,
        expected_values: list[float],
        delta: float = 0.1,
        label: str = "value",
    ) -> None:
        for expected_value in expected_values:
            if abs(actual_value - expected_value) <= delta:
                return

        expected_summary = ", ".join(f"{expected_value:.6f}±{delta:.2f}" for expected_value in expected_values)
        self.fail(f"Unexpected {label}: {actual_value:.6f}; expected one of {expected_summary}")

    def assert_design_matches_any(
        self,
        search,
        expected_outcomes: list[tuple[float, int]],
        delta: float = 0.25,
    ) -> None:
        actual_height = search.ghe.bhe.borehole.H
        actual_boreholes = len(search.ghe.gFunction.bore_locations)

        for expected_height, expected_boreholes in expected_outcomes:
            if actual_boreholes == expected_boreholes and abs(actual_height - expected_height) <= delta:
                return

        expected_summary = ", ".join(
            f"(height={expected_height:.2f}±{delta:.2f}, boreholes={expected_boreholes})"
            for expected_height, expected_boreholes in expected_outcomes
        )
        self.fail(
            "Unexpected design result: "
            f"height={actual_height:.6f}, boreholes={actual_boreholes}; "
            f"expected one of {expected_summary}"
        )
