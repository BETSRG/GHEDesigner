import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest
from jsonschema.exceptions import ValidationError
from pandas.testing import assert_frame_equal

from ghedesigner.district_system import GHEHPSystem
from ghedesigner.enums import SimCompType
from ghedesigner.tests.test_base_case import GHEBaseTest
from ghedesigner.validate import validate_input_file


class TestDistrictSys(GHEBaseTest):
    def assert_simulation_output_matches_baseline(self, system: GHEHPSystem, baseline_name: str):
        baseline_path = self.test_data_directory / baseline_name
        with TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / baseline_name
            system.create_output(output_path)
            actual = pd.read_csv(output_path)

        expected = pd.read_csv(baseline_path)
        assert_frame_equal(actual, expected, check_dtype=False, check_exact=False, rtol=0.0, atol=1e-2)

    def test_simulate_1_pipe_3_ghe_6_bldg_district(self):
        f_path_json = self.demos_path / "simulate_1_pipe_3_ghe_6_bldg_district_HOURLY.json"
        system = GHEHPSystem(f_path_json)
        system.size_and_simulate()
        self.assert_simulation_output_matches_baseline(system, "simulate_1_pipe_3_ghe_6_bldg_district_HOURLY.csv")

    def test_simulate_1_pipe_1_ghe_1_bldg_district(self):
        f_path_json = self.demos_path / "simulate_1_pipe_1_ghe_1_bldg_district.json"
        system = GHEHPSystem(f_path_json)
        system.size_and_simulate()
        self.assert_simulation_output_matches_baseline(system, "simulate_1_pipe_1_ghe_1_bldg_district.csv")

    def test_one_pipe_single_loop_startup_does_not_oscillate(self):
        f_path_json = self.demos_path / "simulate_1_pipe_1_ghe_1_bldg_district.json"
        system = GHEHPSystem(f_path_json)
        system.size_and_simulate()

        building = next(comp for comp in system.components if comp.comp_type == SimCompType.BUILDING)

        assert np.all(np.diff(building.t_in[:20]) < 0.0)

    def test_one_pipe_eft_state_flows_downstream_for_identical_series_components(self):
        f_path_json = self.demos_path / "simulate_1_pipe_1_ghe_1_bldg_district.json"
        data = json.loads(f_path_json.read_text())
        data["topology"] = [
            {"type": "building", "name": "building1"},
            {"type": "ground_heat_exchanger", "name": "ghe1"},
            {"type": "building", "name": "building2"},
            {"type": "ground_heat_exchanger", "name": "ghe2"},
        ]
        data["building"]["building2"] = copy.deepcopy(data["building"]["building1"])
        data["ground_heat_exchanger"]["ghe2"] = copy.deepcopy(data["ground_heat_exchanger"]["ghe1"])
        data["simulation_control"]["search_method"] = "SIMULATION_ONLY"
        data["simulation_control"]["constant_cop"] = True
        for building_data in data["building"].values():
            for load_data in building_data.values():
                if isinstance(load_data, dict) and "file_path" in load_data:
                    load_data["file_path"] = str((f_path_json.parent / load_data["file_path"]).resolve())
                    load_data.pop("heat_pump_name", None)
                    load_data["heat_pump_cop"] = 4.0

        with TemporaryDirectory() as tmp_dir:
            one_pipe_path = Path(tmp_dir) / "one_pipe_identical_series.json"
            one_pipe_path.write_text(json.dumps(data))
            system = GHEHPSystem(one_pipe_path)
            system.size_and_simulate()

        building1, building2 = [comp for comp in system.components if comp.comp_type == SimCompType.BUILDING]
        ghe1, ghe2 = [comp for comp in system.components if comp.comp_type == SimCompType.GROUND_HEAT_EXCHANGER]

        return_fraction = system.loop_return_current_fraction
        expected_first_inlet = return_fraction * ghe2.t_out[1:] + (1.0 - return_fraction) * ghe2.t_out[:-1]

        assert np.allclose(building1.t_out, ghe1.t_in)
        assert np.allclose(ghe1.t_out, building2.t_in)
        assert np.allclose(building2.t_out, ghe2.t_in)
        assert np.allclose(expected_first_inlet, building1.t_in[1:])
        assert not np.allclose(building1.t_in, building2.t_in)
        assert not np.allclose(ghe1.t_in, ghe2.t_in)

    def test_simulate_1_pipe_1_ghe_1_hx_1_bldg_district(self):
        f_path_json = self.demos_path / "simulate_1_pipe_1_ghe_1_hx_1_bldg_district.json"
        system = GHEHPSystem(f_path_json)
        system.size_and_simulate()
        self.assert_simulation_output_matches_baseline(system, "simulate_1_pipe_1_ghe_1_hx_1_bldg_district.csv")

    def test_simulate_1_pipe_1_ghe_1_hx_1_bldg_w_loads_district(self):
        f_path_json = self.demos_path / "simulate_1_pipe_1_ghe_1_hx_1_bldg_w_loads_district.json"
        system = GHEHPSystem(f_path_json)
        system.size_and_simulate()
        self.assert_simulation_output_matches_baseline(system, "simulate_1_pipe_1_ghe_1_hx_1_bldg_w_loads_district.csv")

    def test_two_pipe_inlet_indices_follow_topology_order(self):
        f_path_json = self.demos_path / "simulate_2_pipe_3_ghe_6_bldg_district_HOURLY.json"
        system = GHEHPSystem(f_path_json)

        buildings = [comp for comp in system.components if comp.comp_type == SimCompType.BUILDING]
        ghes = [comp for comp in system.components if comp.comp_type == SimCompType.GROUND_HEAT_EXCHANGER]

        assert buildings
        assert ghes
        assert [comp.inlet_index for comp in buildings] == [comp.row_index for comp in buildings]
        assert [comp.inlet_index for comp in ghes] == [comp.row_index for comp in ghes]
        assert len({comp.inlet_index for comp in buildings}) == len(buildings)
        assert len({comp.inlet_index for comp in ghes}) == len(ghes)
        assert all(isinstance(comp.inlet_index, int) for comp in buildings + ghes)

    def test_two_pipe_eft_state_flows_downstream(self):
        f_path_json = self.demos_path / "simulate_2_pipe_3_ghe_6_bldg_district_HOURLY.json"
        system = GHEHPSystem(f_path_json)
        system.size_and_simulate()

        buildings = [comp for comp in system.components if comp.comp_type == SimCompType.BUILDING]
        ghes = [comp for comp in system.components if comp.comp_type == SimCompType.GROUND_HEAT_EXCHANGER]

        assert not np.allclose(buildings[0].t_in, buildings[2].t_in)
        assert not np.allclose(ghes[0].t_in, ghes[1].t_in)

    def test_two_pipe_non_constant_cop_building_matrix_is_generated(self):
        f_path_json = self.demos_path / "simulate_2_pipe_3_ghe_6_bldg_district_HOURLY.json"
        data = json.loads(f_path_json.read_text())
        data["simulation_control"]["constant_cop"] = False
        heat_pump_name = next(iter(data["heat_pump"]))
        for building_data in data["building"].values():
            for load_data in building_data.values():
                if isinstance(load_data, dict) and "file_path" in load_data:
                    load_data["file_path"] = str((f_path_json.parent / load_data["file_path"]).resolve())
                    load_data["heat_pump_name"] = heat_pump_name

        with TemporaryDirectory() as tmp_dir:
            two_pipe_path = Path(tmp_dir) / "two_pipe_non_constant_cop.json"
            two_pipe_path.write_text(json.dumps(data))
            system = GHEHPSystem(two_pipe_path)

        building = next(comp for comp in system.components if comp.comp_type == SimCompType.BUILDING)
        building.matrix_size = system.matrix_size
        building.cp = system.cp
        mass_bldg = building.calc_mass_flow_rate(building.t_in[0], 0)

        rows, rhs = building.generate_matrix(
            mass_bldg,
            mass_bldg * system.loop_flow_factor,
            mass_bldg,
            0.0,
            0.0,
            1,
            system.loop_config,
            system.load_method,
        )

        assert len(rows) == 2
        assert len(rhs) == 2

        captured = {}

        def capture_r1_r2(t_in, idx_timestep):
            captured["t_in"] = t_in
            captured["idx_timestep"] = idx_timestep
            return 0.0, 0.0

        building.t_in[0] = 12.5
        building.t_in[1] = 99.5
        building.calc_r1_r2 = capture_r1_r2

        rows, rhs = building.generate_matrix(
            mass_bldg,
            mass_bldg * system.loop_flow_factor,
            mass_bldg,
            0.0,
            0.0,
            2,
            system.loop_config,
            system.load_method,
        )

        assert len(rows) == 2
        assert len(rhs) == 2
        assert captured == {"t_in": 12.5, "idx_timestep": 1}

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

    def test_fixed_cop_building_pump_power_uses_building_pump_parameters(self):
        f_path_json = self.demos_path / "simulate_1_pipe_1_ghe_1_bldg_district.json"
        data = json.loads(f_path_json.read_text())
        data["simulation_control"]["constant_cop"] = True
        for building_data in data["building"].values():
            building_data["design_pressure_loss"] = 25000.0
            building_data["pump_efficiency"] = 0.5
            for load_data in building_data.values():
                if isinstance(load_data, dict) and "file_path" in load_data:
                    load_data["file_path"] = str((f_path_json.parent / load_data["file_path"]).resolve())
                    load_data.pop("heat_pump_name", None)
                    load_data["heat_pump_cop"] = 4.0

        with TemporaryDirectory() as tmp_dir:
            fixed_cop_path = Path(tmp_dir) / "fixed_cop_pump_power.json"
            fixed_cop_path.write_text(json.dumps(data))
            system = GHEHPSystem(fixed_cop_path)

        building = next(comp for comp in system.components if comp.comp_type == SimCompType.BUILDING)
        building.cp = system.cp
        building.calc_mass_flow_rate(building.t_in[0], 0)
        building.calc_energy()

        expected = building.m_flow[0] / (building.fluid.rho * 0.5) * 25000.0
        assert building.power_circ_pump[0] == pytest.approx(expected)
        assert building.power_circ_pump[0] > 0.0

    def test_fixed_loads_simulation_uses_valid_building_flow_indices(self):
        f_path_json = self.demos_path / "simulate_1_pipe_1_ghe_1_bldg_district.json"
        data = json.loads(f_path_json.read_text())
        data["simulation_control"]["fixed_loads"] = True
        data["simulation_control"]["constant_cop"] = True
        for building_data in data["building"].values():
            for load_data in building_data.values():
                if isinstance(load_data, dict) and "file_path" in load_data:
                    load_data["file_path"] = str((f_path_json.parent / load_data["file_path"]).resolve())

        with TemporaryDirectory() as tmp_dir:
            fixed_loads_path = Path(tmp_dir) / "fixed_loads.json"
            fixed_loads_path.write_text(json.dumps(data))
            system = GHEHPSystem(fixed_loads_path)

        system.size_and_simulate()

        assert system.number_of_simulations == 1
        assert all(building.m_flow[-1] >= 0.0 for building in system.buildings)

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
