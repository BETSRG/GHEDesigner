import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from jsonschema.exceptions import ValidationError

from ghedesigner.district_system import GHEHPSystem
from ghedesigner.enums import SimCompType
from ghedesigner.tests.test_base_case import GHEBaseTest
from ghedesigner.validate import validate_input_file


class TestDistrictSys(GHEBaseTest):
    def test_simulate_1_pipe_3_ghe_6_bldg_district(self):
        f_path_json = self.demos_path / "simulate_1_pipe_3_ghe_6_bldg_district_HOURLY.json"
        system = GHEHPSystem(f_path_json)
        system.size_and_simulate()

        # don't put in the timestamped directory for now
        # system.create_output(self.test_outputs_directory / "test_district_sys" / "output_simulate_3_bldg_3_ghe.csv")
        system.create_output(
            self.tests_directory / self.test_data_directory / "simulate_1_pipe_3_ghe_6_bldg_district.csv"
        )

    def test_simulate_1_pipe_1_ghe_1_bldg_district(self):
        f_path_json = self.demos_path / "simulate_1_pipe_1_ghe_1_bldg_district.json"
        system = GHEHPSystem(f_path_json)
        system.size_and_simulate()

        # don't put in the timestamped directory for now
        # system.create_output(self.test_outputs_directory / "test_simple_district" / "output_simple_district.csv")
        system.create_output(
            self.tests_directory / self.test_data_directory / "simulate_1_pipe_1_ghe_1_bldg_district.csv"
        )

    def test_simulate_1_pipe_1_ghe_1_hx_1_bldg_district(self):
        f_path_json = self.demos_path / "simulate_1_pipe_1_ghe_1_hx_1_bldg_district.json"
        system = GHEHPSystem(f_path_json)
        system.size_and_simulate()

        # don't put in the timestamped directory for now
        # system.create_output(self.test_outputs_directory / "test_simple_district" / "output_simple_district.csv")
        system.create_output(
            self.tests_directory / self.test_data_directory / "simulate_1_pipe_1_ghe_1_hx_1_bldg_district.csv"
        )

    def test_simulate_1_pipe_1_ghe_1_hx_1_bldg_w_loads_district(self):
        f_path_json = self.demos_path / "simulate_1_pipe_1_ghe_1_hx_1_bldg_w_loads_district.json"
        system = GHEHPSystem(f_path_json)
        system.size_and_simulate()

        # don't put in the timestamped directory for now
        # system.create_output(self.test_outputs_directory / "test_simple_district" / "output_simple_district.csv")
        system.create_output(
            self.tests_directory / self.test_data_directory / "simulate_1_pipe_1_ghe_1_hx_1_bldg_w_loads_district.csv"
        )

    def test_two_pipe_inlet_indices_are_assigned(self):
        f_path_json = self.demos_path / "simulate_2_pipe_3_ghe_6_bldg_district_HOURLY.json"
        system = GHEHPSystem(f_path_json)

        buildings = [comp for comp in system.components if comp.comp_type == SimCompType.BUILDING]
        ghes = [comp for comp in system.components if comp.comp_type == SimCompType.GROUND_HEAT_EXCHANGER]

        assert buildings
        assert ghes
        assert {comp.inlet_index for comp in buildings} == {buildings[0].row_index}
        assert {comp.inlet_index for comp in ghes} == {ghes[0].row_index}
        assert all(isinstance(comp.inlet_index, int) for comp in buildings + ghes)

    def test_simulation_only_initializes_output_bookkeeping(self):
        f_path_json = self.demos_path / "simulate_1_pipe_1_ghe_1_bldg_district.json"
        data = json.loads(f_path_json.read_text())
        data["simulation_control"]["search_method"] = "SIMULATION_ONLY"
        for building_data in data["building"].values():
            for load_data in building_data.values():
                if isinstance(load_data, dict) and "file_path" in load_data:
                    load_data["file_path"] = str((f_path_json.parent / load_data["file_path"]).resolve())

        with TemporaryDirectory() as tmp_dir:
            sim_only_path = Path(tmp_dir) / "simulation_only.json"
            sim_only_path.write_text(json.dumps(data))

            system = GHEHPSystem(sim_only_path)

        assert system.nbh_selections == []
        assert system.coordinate_locations == {}
        assert system.total_loads.shape == (8760,)

    def test_horizontal_piping_schema_rejects_missing_pipe(self):
        f_path_json = self.demos_path / "simulate_1_pipe_3_ghe_6_bldg_district_HOURLY_horizontal.json"
        data = json.loads(f_path_json.read_text())
        first_pipe = next(iter(data["horizontal_piping"].values()))
        del first_pipe["pipe"]

        with TemporaryDirectory() as tmp_dir:
            invalid_path = Path(tmp_dir) / "missing_horizontal_pipe.json"
            invalid_path.write_text(json.dumps(data))

            with pytest.raises(ValidationError):
                validate_input_file(invalid_path)
