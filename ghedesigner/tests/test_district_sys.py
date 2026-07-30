import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from jsonschema.exceptions import ValidationError
from pandas.testing import assert_frame_equal

from ghedesigner.district_system import CoupledHorizontalPipe, GHEHPSystem, IsolatedHorizontalPipe
from ghedesigner.enums import DesignGeomType, SimCompType
from ghedesigner.ghe.hp_hybrid_loads_processor import ProcessLoads
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.media import Fluid, Soil
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

    def test_simulate_1_pipe_3_ghe_6_bldg_district_horizontal(self):
        f_path_json = self.demos_path / "simulate_1_pipe_3_ghe_6_bldg_district_HOURLY_horizontal.json"
        system = GHEHPSystem(f_path_json)
        system.size_and_simulate()
        self.assert_simulation_output_matches_baseline(
            system, "simulate_1_pipe_3_ghe_6_bldg_district_HOURLY_horizontal.csv"
        )

    def test_simulate_1_pipe_3_ghe_6_bldg_district_horizontal_loadagg(self):
        f_path_json = self.demos_path / "simulate_1_pipe_3_ghe_6_bldg_district_LOADAGGHOURLY_horizontal.json"
        system = GHEHPSystem(f_path_json)
        system.size_and_simulate()
        self.assert_simulation_output_matches_baseline(
            system, "simulate_1_pipe_3_ghe_6_bldg_district_LOADAGGHOURLY_horizontal.csv"
        )

    def test_horizontal_loadagg_uses_subhourly_timestep(self):
        time_array = np.linspace(0.0, 1.0, 20, endpoint=False)
        pipe = Pipe.init_single_u_tube(
            inner_diameter=0.1524,
            outer_diameter=0.1624,
            shank_spacing=0.0,
            roughness=1e-6,
            conductivity=0.4,
            rho_cp=1542000,
        )
        soil = Soil(k=2.0, rho_cp=2343520, ugt=15.0)
        fluid = Fluid(fluid_name="WATER", percent=0, temperature=70)

        def q_prime_interp(tau):
            return np.zeros_like(np.asarray(tau, dtype=float))

        horiz_pipe = IsolatedHorizontalPipe(
            name="subhourly_pipe",
            length=10.0,
            num_segments=1,
            pipe=pipe,
            soil=soil,
            fluid=fluid,
            num_timesteps=time_array.size,
            time_array=time_array,
            q_prime_interp=q_prime_interp,
            beta=0.344,
            ugt_avg=15.0,
            ugt_amp1=0.0,
            ugt_phase1=0.0,
            ugt_amp2=0.0,
            ugt_phase2=0.0,
            depth=1.0,
            load_method="hourlyloadagg",
        )

        aggregator = horiz_pipe.aggregators[0]
        assert aggregator.base_dt_sec == pytest.approx(180.0)
        aggregator.shift_and_add(2.0, 180.0, 1)
        assert aggregator.energy_bins[0] == pytest.approx(360.0)

    def test_coupled_loadagg_uses_neighbor_ground_temperature(self):
        time_array = np.array([0.0, 1.0, 2.0])
        pipe = Pipe.init_single_u_tube(
            inner_diameter=0.1524,
            outer_diameter=0.1624,
            shank_spacing=0.0,
            roughness=1e-6,
            conductivity=0.4,
            rho_cp=1542000,
        )
        soil = Soil(k=2.0, rho_cp=2343520, ugt=15.0)
        fluid = Fluid(fluid_name="WATER", percent=0, temperature=70)

        def q_prime_interp(tau):
            return np.zeros_like(np.asarray(tau, dtype=float))

        def make_pipe(name, ugt_avg):
            return CoupledHorizontalPipe(
                name=name,
                length=10.0,
                num_segments=1,
                pipe=pipe,
                soil=soil,
                fluid=fluid,
                num_timesteps=time_array.size,
                time_array=time_array,
                q_prime_even_interp=q_prime_interp,
                q_prime_odd_interp=q_prime_interp,
                beta=0.344,
                ugt_avg=ugt_avg,
                ugt_amp1=0.0,
                ugt_phase1=0.0,
                ugt_amp2=0.0,
                ugt_phase2=0.0,
                depth=1.0,
                load_method="hourlyloadagg",
            )

        pipe_a = make_pipe("pipe_a", 10.0)
        pipe_b = make_pipe("pipe_b", 20.0)
        pipe_a.coupled_pipe = pipe_b
        pipe_b.coupled_pipe = pipe_a
        pipe_a.t_mean_seg[0, 1] = 15.0
        pipe_b.t_mean_seg[0, 1] = 30.0

        pipe_a.compute_history_terms(2)

        assert pipe_a.aggregators[0].energy_bins[0] == pytest.approx(5.0 * 3600.0)
        assert pipe_b.aggregators[0].energy_bins[0] == pytest.approx(10.0 * 3600.0)

    def test_hybrid_fixed_cop_loads_do_not_require_heat_pump_names(self):
        processor = ProcessLoads()
        processor.read_hp_load_from_json(
            {
                "building": {
                    "building": {
                        "heating_load": {"load_values": [1.0, 2.0], "heat_pump_cop": 3.5},
                        "cooling_load": {"load_values": [3.0, 4.0], "heat_pump_cop": 4.5},
                    }
                }
            }
        )

        zone = processor.zones[0]
        np.testing.assert_array_equal(zone.q_htg_1yr, [1.0, 2.0])
        np.testing.assert_array_equal(zone.q_clg_1yr, [3.0, 4.0])
        assert zone.COP_htg == pytest.approx(3.5)
        assert zone.COP_clg == pytest.approx(4.5)

    def test_rowwise_spacing_bounds_use_constraint_intersection(self):
        def make_ghe(min_spacing, max_spacing):
            constraint = SimpleNamespace(min_spacing=min_spacing, max_spacing=max_spacing)
            manager = SimpleNamespace(geom_type=DesignGeomType.ROWWISE, geometric_constraint=constraint)
            return SimpleNamespace(ghe_manager=manager)

        bounds = GHEHPSystem._get_rowwise_spacing_bounds([make_ghe(4.5, 10.0), make_ghe(6.0, 8.0)])

        assert bounds == (6.0, 8.0)

    def test_rowwise_spacing_bounds_reject_disjoint_constraints(self):
        def make_ghe(min_spacing, max_spacing):
            constraint = SimpleNamespace(min_spacing=min_spacing, max_spacing=max_spacing)
            manager = SimpleNamespace(geom_type=DesignGeomType.ROWWISE, geometric_constraint=constraint)
            return SimpleNamespace(ghe_manager=manager)

        with pytest.raises(ValueError, match="do not have a common range"):
            GHEHPSystem._get_rowwise_spacing_bounds([make_ghe(4.5, 5.0), make_ghe(6.0, 8.0)])

    def test_simulate_1_pipe_1_ghe_1_bldg_district(self):
        f_path_json = self.demos_path / "simulate_1_pipe_1_ghe_1_bldg_district.json"
        system = GHEHPSystem(f_path_json)
        system.size_and_simulate()
        self.assert_simulation_output_matches_baseline(system, "simulate_1_pipe_1_ghe_1_bldg_district.csv")

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
