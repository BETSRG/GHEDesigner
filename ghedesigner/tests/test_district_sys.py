import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from jsonschema.exceptions import ValidationError
from pandas.testing import assert_frame_equal

from ghedesigner.constants import TWO_PI
from ghedesigner.district_system import (
    GHX,
    CoupledHorizontalPipe,
    DynamicAggregator,
    GHEHPSystem,
    IsolatedHorizontalPipe,
    timestep_params_generator,
)
from ghedesigner.enums import BHType, DesignGeomType, SimCompType
from ghedesigner.ghe.horizontal_pipe_heat_exchange import calc_pipe_wall_resistance
from ghedesigner.ghe.hp_hybrid_loads_processor import ProcessLoads, Zone, enforce_minimum_timestep
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.media import Fluid, Soil
from ghedesigner.tests.test_base_case import GHEBaseTest
from ghedesigner.utilities import load_input_file
from ghedesigner.validate import validate_input_file


class TestDistrictSys(GHEBaseTest):
    @staticmethod
    def make_history_test_ghx(*, constant_time_step: bool) -> GHX:
        ghx = object.__new__(GHX)
        fixture = cast(Any, ghx)
        fixture.load_method = "hourly"
        fixture.dq = np.zeros(3, dtype=float)
        fixture.q_ghe = np.array([1.0, 0.0, 0.0])
        fixture.two_pi_k_recip = 1.0
        fixture.constant_time_step = constant_time_step
        fixture.total_values_ghe = np.zeros(4, dtype=float)
        fixture.history_terms = np.zeros(5, dtype=float)
        fixture.ghe_manager = SimpleNamespace(soil=SimpleNamespace(ugt=0.0))
        fixture.ts = 3600.0
        fixture.g = np.exp
        return ghx

    def test_constant_timestep_ghe_history_uses_full_elapsed_time(self):
        ghx = self.make_history_test_ghx(constant_time_step=True)
        ghx.gfunction_evals = np.array([3.0, 2.0, 1.0])
        ghx.step_gfunction_evals = np.ones(3, dtype=float)

        history_term = ghx.calc_history_term(2)

        assert history_term == pytest.approx(-1.0)

    def test_variable_timestep_ghe_history_uses_current_interval(self):
        ghx = self.make_history_test_ghx(constant_time_step=False)
        ghx.time_array = np.array([0.0, 1.0, 3.0, 6.0])
        ghx.gfunction_evals = np.array([6.0, 5.0, 3.0])
        ghx.step_gfunction_evals = np.array([1.0, 2.0, 3.0])

        history_term = ghx.calc_history_term(2)

        assert history_term == pytest.approx(-1.0)

    def test_loadagg_ghe_history_uses_elapsed_response_age(self):
        ghx = self.make_history_test_ghx(constant_time_step=True)
        ghx.load_method = "hourlyloadagg"
        ghx.time_array = np.array([0.0, 1.0, 2.0, 3.0])
        ghx.aggregator = DynamicAggregator(3.0 * 3600.0)
        ghx.g_agg = ghx.g(np.log(ghx.aggregator.response_ages / ghx.ts))
        ghx.step_gfunction_evals = np.ones(3, dtype=float)

        history_term = ghx.calc_history_term(2)

        assert ghx.aggregator.response_ages[0] == pytest.approx(2.0 * 3600.0)
        assert history_term == pytest.approx(-1.0)

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
            return np.asarray(tau, dtype=float)

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
            time_step_params=timestep_params_generator(time_array),
            load_method="hourlyloadagg",
        )

        aggregator = horiz_pipe.aggregators[0]
        assert aggregator.base_dt_sec == pytest.approx(180.0)
        assert horiz_pipe.y_agg_evals[0] == pytest.approx(
            horiz_pipe.two_pi_k * 2.0 * aggregator.base_dt_sec / horiz_pipe.t_p
        )
        aggregator.shift_and_add(2.0, 180.0, 1)
        assert aggregator.energy_bins[0] == pytest.approx(360.0)

    def test_horizontal_pipe_response_uses_configured_conductivity(self):
        source_path = self.demos_path / "simulate_1_pipe_3_ghe_6_bldg_district_HOURLY_horizontal.json"
        standard_system = GHEHPSystem(source_path)
        modified_data = load_input_file(source_path)
        pipe_name = "building1_building2_line"
        modified_data["horizontal_piping"][pipe_name]["pipe"]["conductivity"] = 0.01

        with TemporaryDirectory() as tmp_dir:
            modified_path = Path(tmp_dir) / "low_conductivity_horizontal.json"
            modified_path.write_text(json.dumps(modified_data))
            low_conductivity_system = GHEHPSystem(modified_path)

        standard_pipe = next(comp for comp in standard_system.components if comp.name == pipe_name)
        low_conductivity_pipe = next(comp for comp in low_conductivity_system.components if comp.name == pipe_name)

        standard_wall_resistance = calc_pipe_wall_resistance(
            Pipe.init_single_u_tube(
                inner_diameter=0.1016,
                outer_diameter=0.1116,
                shank_spacing=0.0,
                roughness=1e-6,
                conductivity=0.4,
                rho_cp=1542000,
            )
        )
        assert standard_pipe.beta == pytest.approx(TWO_PI * 2.0 * standard_wall_resistance)
        assert low_conductivity_pipe.beta == pytest.approx(40.0 * standard_pipe.beta)
        assert standard_pipe.q_prime_interp(0.0) == pytest.approx(1.0 / 0.344)
        assert low_conductivity_pipe.q_prime_interp(0.0) == pytest.approx(1.0 / 12.0)

    def test_horizontal_ground_temperature_model_is_nested_under_soil(self):
        source_path = self.demos_path / "simulate_1_pipe_3_ghe_6_bldg_district_HOURLY_horizontal.json"
        data = load_input_file(source_path)
        system = GHEHPSystem(source_path)
        horizontal_pipes = [
            component
            for component in system.components
            if component.comp_type in (SimCompType.ISOLATED_HORIZONTAL_PIPE, SimCompType.COUPLED_HORIZONTAL_PIPE)
        ]

        assert "ground_temperature_model" not in data
        assert "annual_average" not in data["soil"]["ground_temperature_model"]
        assert horizontal_pipes
        assert all(pipe.ugt_avg == pytest.approx(data["soil"]["undisturbed_temp"]) for pipe in horizontal_pipes)
        assert all(pipe.soil.k == pytest.approx(data["soil"]["conductivity"]) for pipe in horizontal_pipes)
        assert all(pipe.soil.rho_cp == pytest.approx(data["soil"]["rho_cp"]) for pipe in horizontal_pipes)

    def test_horizontal_component_soil_is_rejected(self):
        source_path = self.demos_path / "simulate_1_pipe_3_ghe_6_bldg_district_HOURLY_horizontal.json"
        data = load_input_file(source_path)
        first_pipe = next(iter(data["horizontal_piping"].values()))
        first_pipe["soil"] = {
            "conductivity": data["soil"]["conductivity"],
            "rho_cp": data["soil"]["rho_cp"],
        }

        with TemporaryDirectory() as tmp_dir:
            legacy_path = Path(tmp_dir) / "component_soil.json"
            legacy_path.write_text(json.dumps(data))

            with pytest.raises(ValidationError):
                validate_input_file(legacy_path)

    def test_root_ground_temperature_model_is_rejected(self):
        source_path = self.demos_path / "simulate_1_pipe_3_ghe_6_bldg_district_HOURLY_horizontal.json"
        data = load_input_file(source_path)
        ground_temperature_model = data["soil"].pop("ground_temperature_model")
        ground_temperature_model["annual_average"] = data["soil"]["undisturbed_temp"]
        data["ground_temperature_model"] = ground_temperature_model

        with TemporaryDirectory() as tmp_dir:
            legacy_path = Path(tmp_dir) / "root_ground_temperature_model.json"
            legacy_path.write_text(json.dumps(data))

            with pytest.raises(ValidationError):
                validate_input_file(legacy_path)

    def test_disabled_horizontal_simulation_does_not_require_ground_temperature_model(self):
        source_path = self.demos_path / "simulate_1_pipe_3_ghe_6_bldg_district_HOURLY_horizontal.json"
        data = load_input_file(source_path)
        data["simulation_control"]["horizontal_simulation_considered"] = False
        del data["soil"]["ground_temperature_model"]

        with TemporaryDirectory() as tmp_dir:
            disabled_path = Path(tmp_dir) / "horizontal_disabled.json"
            disabled_path.write_text(json.dumps(data))
            validate_input_file(disabled_path)
            system = GHEHPSystem(disabled_path)

        horizontal_types = (SimCompType.ISOLATED_HORIZONTAL_PIPE, SimCompType.COUPLED_HORIZONTAL_PIPE)
        assert all(component.comp_type not in horizontal_types for component in system.components)
        expected_component_names = [
            component["name"]
            for component in data["topology"]
            if SimCompType[component["type"].upper()] not in horizontal_types
        ]
        assert [component.name for component in system.components] == expected_component_names

        system.num_timesteps = 2
        system.solve_system_standard()
        assert all(np.all(np.isfinite(component.t_in[:2])) for component in system.components)

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
            return np.asarray(tau, dtype=float)

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
                time_step_params=timestep_params_generator(time_array),
                load_method="hourlyloadagg",
            )

        pipe_a = make_pipe("pipe_a", 10.0)
        pipe_b = make_pipe("pipe_b", 20.0)
        assert pipe_a.y_self_agg_evals[0] == pytest.approx(
            pipe_a.two_pi_k * 2.0 * pipe_a.aggregators[0].base_dt_sec / pipe_a.t_p
        )
        assert pipe_a.y_cross_agg_evals[0] == pytest.approx(0.0)
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

    def test_hybrid_grid_has_one_hour_minimum_timestep(self):
        grid = enforce_minimum_timestep([0.2, 0.8, 1.2, 1.9, 2.4, 3.0])

        assert grid[0] == 0.0
        assert grid[-1] == 3.0
        assert np.all(np.diff(grid) >= 1.0)

    def test_hybrid_grid_resampling_conserves_energy(self):
        source_time = np.array([0.5, 1.5, 3.0])
        source_load = np.array([2.0, 4.0, 6.0])
        target_time = np.array([0.0, 1.0, 2.0, 3.0])

        mapped_load = Zone.average_loads_on_time_grid(source_time, source_load, target_time)

        np.testing.assert_allclose(mapped_load, [0.0, 3.0, 5.0, 6.0])
        source_energy = np.dot(source_load, np.diff(np.insert(source_time, 0, 0.0)))
        mapped_energy = np.dot(mapped_load[1:], np.diff(target_time))
        assert mapped_energy == pytest.approx(source_energy)

    def test_vertical_soil_is_required_at_top_level(self):
        source_path = self.demos_path / "simulate_1_pipe_1_ghe_1_bldg_district.json"
        data = load_input_file(source_path)
        ghe_data = next(iter(data["ground_heat_exchanger"].values()))
        ghe_data["soil"] = data.pop("soil")

        with TemporaryDirectory() as tmp_dir:
            legacy_path = Path(tmp_dir) / "nested_soil.json"
            legacy_path.write_text(json.dumps(data))

            with pytest.raises(ValidationError):
                validate_input_file(legacy_path)

    def test_hybrid_reference_properties_follow_topology_not_ghe_key_order(self):
        source_path = self.demos_path / "Network_Sizing_3GHE_6HP_BUPCRS.json"
        original_data = load_input_file(source_path)
        original_data["ground_heat_exchanger"]["ghe_1"]["flow_rate"] = 0.4
        original_data["ground_heat_exchanger"]["ghe_2"]["flow_rate"] = 0.8
        reordered_data = json.loads(json.dumps(original_data))
        reordered_data["ground_heat_exchanger"] = dict(reversed(list(reordered_data["ground_heat_exchanger"].items())))

        original_processor = ProcessLoads()
        reordered_processor = ProcessLoads()
        original_processor.read_data_from_json_file(original_data)
        reordered_processor.read_data_from_json_file(reordered_data)

        assert original_processor.mass_flow_rate == pytest.approx(0.4)
        assert reordered_processor.mass_flow_rate == pytest.approx(0.4)
        assert original_processor.soil.ugt == pytest.approx(original_data["soil"]["undisturbed_temp"])
        assert reordered_processor.soil.ugt == pytest.approx(original_data["soil"]["undisturbed_temp"])

    def test_hybrid_reference_preserves_pipe_arrangement_and_converts_borehole_flow(self):
        cases = [
            ("Network_Sizing_3GHE_6HP_BUPCRS.json", BHType.SINGLEUTUBE),
            ("find_design_bi_rectangle_double_u_tube_series.json", BHType.DOUBLEUTUBESERIES),
            ("find_design_rectangle_double_u_tube.json", BHType.DOUBLEUTUBEPARALLEL),
            ("find_design_rectangle_coaxial.json", BHType.COAXIAL),
        ]

        class FakeBhe:
            def to_single(self):
                return self

            def calc_sts_g_functions(self):
                pass

        for pipe_demo, expected_type in cases:
            with self.subTest(pipe_demo=pipe_demo):
                data = load_input_file(self.demos_path / "Network_Sizing_3GHE_6HP_BUPCRS.json")
                pipe_source = load_input_file(self.demos_path / pipe_demo)
                pipe_data = next(iter(pipe_source["ground_heat_exchanger"].values()))["pipe"]
                reference_ghe = data["ground_heat_exchanger"]["ghe_1"]
                reference_ghe["pipe"] = pipe_data
                reference_ghe["flow_rate"] = 0.8
                captured = {}

                def fake_get_bhe_object(bhe_type, mass_flow_borehole, *_args):
                    captured["bhe_type"] = bhe_type
                    captured["mass_flow_borehole"] = mass_flow_borehole
                    return FakeBhe()

                processor = ProcessLoads()
                with patch(
                    "ghedesigner.ghe.hp_hybrid_loads_processor.get_bhe_object",
                    side_effect=fake_get_bhe_object,
                ):
                    processor.read_data_from_json_file(data)
                    processor.prepare_bhe_for_hybrid()

                assert processor.pipe.type == expected_type
                assert captured["bhe_type"] == expected_type
                assert captured["mass_flow_borehole"] == pytest.approx(0.8 / 1000.0 * processor.fluid.rho)

    def test_hybrid_system_flow_uses_pre_designed_borehole_count(self):
        data = load_input_file(self.demos_path / "Network_Sizing_3GHE_6HP_BUPCRS.json")
        reference_ghe = data["ground_heat_exchanger"]["ghe_1"]
        reference_ghe.pop("geometric_constraints")
        reference_ghe.pop("design")
        reference_ghe["pre_designed"] = {
            "arrangement": "RECTANGLE",
            "H": 100.0,
            "boreholes_in_x_dimension": 2,
            "boreholes_in_y_dimension": 3,
            "spacing_in_x_dimension": 5.0,
            "spacing_in_y_dimension": 5.0,
        }
        reference_ghe["flow_rate"] = 6.0
        reference_ghe["flow_type"] = "SYSTEM"
        captured = {}

        class FakeBhe:
            def to_single(self):
                return self

            def calc_sts_g_functions(self):
                pass

        def fake_get_bhe_object(_bhe_type, mass_flow_borehole, *_args):
            captured["mass_flow_borehole"] = mass_flow_borehole
            return FakeBhe()

        processor = ProcessLoads()
        with patch(
            "ghedesigner.ghe.hp_hybrid_loads_processor.get_bhe_object",
            side_effect=fake_get_bhe_object,
        ):
            processor.read_data_from_json_file(data)
            processor.prepare_bhe_for_hybrid()

        assert processor.num_boreholes == 6
        expected_mass_flow = 6.0 / 6 / 1000.0 * processor.fluid.rho
        assert captured["mass_flow_borehole"] == pytest.approx(expected_mass_flow)

    def test_hybrid_system_flow_rejects_sizable_field_without_borehole_count(self):
        data = load_input_file(self.demos_path / "Network_Sizing_3GHE_6HP_BUPCRS.json")
        reference_ghe = data["ground_heat_exchanger"]["ghe_1"]
        reference_ghe["flow_type"] = "SYSTEM"
        processor = ProcessLoads()
        processor.read_data_from_json_file(data)

        with pytest.raises(ValueError, match="borehole count is not known until after sizing"):
            processor.prepare_bhe_for_hybrid()

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

    def test_two_pipe_horizontal_pipes_connect_outlet_to_downstream(self):
        source_path = self.demos_path / "simulate_1_pipe_3_ghe_6_bldg_district_HOURLY_horizontal.json"
        data = load_input_file(source_path)
        data["central_loop"]["pipe_configuration"] = "TWOPIPE"

        with TemporaryDirectory() as tmp_dir:
            two_pipe_path = Path(tmp_dir) / "two_pipe_horizontal.json"
            two_pipe_path.write_text(json.dumps(data))
            system = GHEHPSystem(two_pipe_path)

        horizontal_pipes = [
            comp
            for comp in system.components
            if comp.comp_type in (SimCompType.ISOLATED_HORIZONTAL_PIPE, SimCompType.COUPLED_HORIZONTAL_PIPE)
        ]
        assert horizontal_pipes

        for pipe in horizontal_pipes:
            pipe.matrix_size = system.matrix_size
            rows, _ = pipe.generate_matrix(
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
                1,
                system.loop_config,
                system.load_method,
            )
            topology_row = rows[0]
            outlet_index = pipe.row_index + 3 * pipe.num_segments

            assert pipe.inlet_index == pipe.row_index
            assert topology_row[outlet_index] == 1.0
            assert topology_row[pipe.downstream_index] == -1.0
            assert np.count_nonzero(topology_row) == 2

        system.num_timesteps = 2
        system.solve_system_standard()
        assert all(np.all(pipe.t_in[1:3] > 0.0) and np.all(pipe.t_out[1:3] > 0.0) for pipe in horizontal_pipes)

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

    def test_heat_pump_curve_limits_apply_to_matrix_load_estimate_and_energy(self):
        system = GHEHPSystem(self.demos_path / "simulate_1_pipe_1_ghe_1_bldg_district.json")
        building = system.buildings[0]

        building.htg_vals[0] = 1000.0
        building.clg_vals[0] = 0.0
        assert building.calc_r1_r2(0.0, 0) == pytest.approx(building.calc_r1_r2(9.0, 0))
        assert building.calc_r1_r2(36.0, 0) == pytest.approx(building.calc_r1_r2(50.0, 0))

        building.htg_vals[0] = 0.0
        building.clg_vals[0] = 1000.0
        assert building.calc_r1_r2(0.0, 0) == pytest.approx(building.calc_r1_r2(9.0, 0))
        assert building.calc_r1_r2(36.0, 0) == pytest.approx(building.calc_r1_r2(50.0, 0))

        building.min_eft = 0.0
        building.max_eft = 50.0
        building.generate_constant_cop_loads(ugt=20.0, beta=0.0)
        expected_load = (
            building.hp_htg.heating_ratio(0.0) * building.htg_vals
            - building.hp_clg.cooling_ratio(50.0) * building.clg_vals
        )
        np.testing.assert_allclose(building.loads, expected_load)

        building.t_in[:3] = [0.0, 20.0, 50.0]
        building.htg_vals[:3] = 1000.0
        building.clg_vals[:3] = 1000.0
        building.calc_energy()
        expected_heating_power = 1000.0 * (1.0 - building.hp_htg.heating_ratio(building.t_in[:3]))
        expected_cooling_power = np.abs(1000.0 * (building.hp_clg.cooling_ratio(building.t_in[:3]) - 1.0))
        np.testing.assert_allclose(building.power_hp_htg[:3], expected_heating_power)
        np.testing.assert_allclose(building.power_hp_clg[:3], expected_cooling_power)

    def test_nonconstant_cop_matrix_terms_scale_with_cooling_load(self):
        system = GHEHPSystem(self.demos_path / "simulate_1_pipe_1_ghe_1_bldg_district.json")
        building = system.buildings[0]
        building.htg_vals[0] = 0.0

        building.clg_vals[0] = 1000.0
        matrix_terms_1000_w = building.calc_r1_r2(25.0, 0)
        building.clg_vals[0] = 2000.0
        matrix_terms_2000_w = building.calc_r1_r2(25.0, 0)

        assert matrix_terms_2000_w == pytest.approx(tuple(2.0 * value for value in matrix_terms_1000_w))

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

    def test_horizontal_segments_must_be_a_positive_integer(self):
        f_path_json = self.demos_path / "simulate_1_pipe_3_ghe_6_bldg_district_HOURLY_horizontal.json"
        for horizontal_segments in (0, -1, 1.5, True):
            with self.subTest(horizontal_segments=horizontal_segments):
                data = json.loads(f_path_json.read_text())
                data["simulation_control"]["horizontal_segments"] = horizontal_segments

                with TemporaryDirectory() as tmp_dir:
                    invalid_path = Path(tmp_dir) / "invalid_horizontal_segments.json"
                    invalid_path.write_text(json.dumps(data))

                    with pytest.raises(ValidationError):
                        validate_input_file(invalid_path)
                    with pytest.raises(ValueError, match="horizontal_segments must be a positive integer"):
                        GHEHPSystem(invalid_path)

    def test_coupled_horizontal_pipe_requires_complete_coupling_fields(self):
        f_path_json = self.demos_path / "simulate_1_pipe_3_ghe_6_bldg_district_HOURLY_horizontal.json"
        for missing_field in ("coupled_to", "spacing"):
            with self.subTest(missing_field=missing_field):
                data = load_input_file(f_path_json)
                del data["horizontal_piping"]["horiz_supply_line"][missing_field]

                with TemporaryDirectory() as tmp_dir:
                    invalid_path = Path(tmp_dir) / "incomplete_coupled_pipe.json"
                    invalid_path.write_text(json.dumps(data))

                    with pytest.raises(ValidationError):
                        validate_input_file(invalid_path)
                    with pytest.raises(ValueError, match=rf"missing required field.*{missing_field}"):
                        GHEHPSystem(invalid_path)

    def test_coupled_horizontal_pipes_must_reference_each_other(self):
        f_path_json = self.demos_path / "simulate_1_pipe_3_ghe_6_bldg_district_HOURLY_horizontal.json"
        data = load_input_file(f_path_json)
        data["horizontal_piping"]["horiz_return_line"]["coupled_to"] = "horiz_return_line"

        with TemporaryDirectory() as tmp_dir:
            invalid_path = Path(tmp_dir) / "nonreciprocal_coupled_pipe.json"
            invalid_path.write_text(json.dumps(data))

            with pytest.raises(ValueError, match="must reference each other"):
                GHEHPSystem(invalid_path)

    def test_coupled_horizontal_pipes_must_have_compatible_properties(self):
        f_path_json = self.demos_path / "simulate_1_pipe_3_ghe_6_bldg_district_HOURLY_horizontal.json"
        data = load_input_file(f_path_json)
        data["horizontal_piping"]["horiz_return_line"]["spacing"] = 0.5

        with TemporaryDirectory() as tmp_dir:
            invalid_path = Path(tmp_dir) / "incompatible_coupled_pipe.json"
            invalid_path.write_text(json.dumps(data))

            with pytest.raises(ValueError, match="matching properties: spacing"):
                GHEHPSystem(invalid_path)

    def test_coupled_horizontal_partner_lookup_is_case_insensitive(self):
        f_path_json = self.demos_path / "simulate_1_pipe_3_ghe_6_bldg_district_HOURLY_horizontal.json"
        data = load_input_file(f_path_json)
        data["horizontal_piping"]["horiz_supply_line"]["coupled_to"] = "HORIZ_RETURN_LINE"
        data["horizontal_piping"]["horiz_return_line"]["coupled_to"] = "HORIZ_SUPPLY_LINE"

        with TemporaryDirectory() as tmp_dir:
            valid_path = Path(tmp_dir) / "case_insensitive_coupled_pipe.json"
            valid_path.write_text(json.dumps(data))
            system = GHEHPSystem(valid_path)

        supply_pipe = next(comp for comp in system.components if comp.name == "horiz_supply_line")
        return_pipe = next(comp for comp in system.components if comp.name == "horiz_return_line")
        assert supply_pipe.coupled_pipe is return_pipe
        assert return_pipe.coupled_pipe is supply_pipe
