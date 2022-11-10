from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional
from unittest import TestCase

LOG_FILE: Optional[Path] = None


class GHEBaseTest(TestCase):
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
                GHEBaseTest.LogMessageTypes.Error: "ERROR"
            }
            return strs[message_type_enum]

    def setup_log_file(self) -> None:
        global LOG_FILE
        cur_file = Path(__file__).resolve()
        tests_directory = cur_file.parent
        log_directory = tests_directory / 'test_logs'
        log_directory.mkdir(exist_ok=True)
        date_time_string = datetime.now().strftime("%d%m%Y_%H%M%S")
        LOG_FILE = log_directory / f"{date_time_string}.log"
        self.log("Tests Initialized")

    def setUp(self) -> None:
        cur_file = Path(__file__).resolve()
        self.tests_directory = cur_file.parent
        self.test_data_directory = self.tests_directory / 'test_data'
        self.project_root_directory = self.tests_directory.parent
        self.test_outputs_directory = self.tests_directory / 'test_outputs'
        self.test_outputs_directory.mkdir(exist_ok=True)

    # noinspection PyMethodMayBeStatic
    def log(self, message, message_type: LogMessageTypes = LogMessageTypes.Info):
        if LOG_FILE is None:
            self.setup_log_file()
        date_time_string = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
        message_type_string = GHEBaseTest.LogMessageTypes.get_string(message_type)
        message_string = str(message).strip()
        with LOG_FILE.open('a') as fp:
            fp.write(f"{date_time_string},{message_type_string},{message_string}\n")

    def get_atlanta_loads(self):
        # read in the csv file and convert the loads to a list of length 8760
        glhe_json_data = self.test_data_directory / 'Atlanta_Office_Building_Loads.csv'
        raw_lines = glhe_json_data.read_text().split('\n')
        return [float(x) for x in raw_lines[1:] if x.strip() != '']

    # def get_polygon_building_csv_list(self):
    #     building_file = self.test_data_directory / 'polygon_building.csv'
    #     build_polygon_df: pd.DataFrame = pd.read_csv(str(building_file))
    #     self.building_polygon_ar: list = build_polygon_df.values.tolist()
