import json

import numpy as np
import pytest

from ghedesigner.tests.test_base_case import GHEBaseTest
from ghedesigner.utilities import HPmodel, read_csv_column


class TestUtilities(GHEBaseTest):
    def test_heat_pump_quadratics_hold_boundary_values_outside_curve_range(self):
        demo = json.loads((self.demos_path / "simulate_1_pipe_1_ghe_1_bldg_district.json").read_text())
        heat_pump = HPmodel("hp1", demo["heat_pump"]["hp1"], ugt=20.0)
        temperatures = np.array([0.0, 10.0, 20.0, 35.0, 50.0])

        heating_ratios = heat_pump.heating_ratio(temperatures)
        cooling_ratios = heat_pump.cooling_ratio(temperatures)
        heating_capacities = heat_pump.heating_capacity(temperatures)
        cooling_capacities = heat_pump.cooling_capacity(temperatures)

        assert heating_ratios[0] == pytest.approx(heating_ratios[1])
        assert heating_ratios[-1] == pytest.approx(heating_ratios[-2])
        assert cooling_ratios[0] == pytest.approx(cooling_ratios[1])
        assert cooling_ratios[-1] == pytest.approx(cooling_ratios[-2])
        assert heating_capacities[0] == pytest.approx(heating_capacities[1])
        assert heating_capacities[-1] == pytest.approx(heating_capacities[-2])
        assert cooling_capacities[0] == pytest.approx(cooling_capacities[1])
        assert cooling_capacities[-1] == pytest.approx(cooling_capacities[-2])

        assert heat_pump.heating_ratio_slope(0.0) == 0.0
        assert heat_pump.heating_ratio_slope(50.0) == 0.0
        assert heat_pump.cooling_ratio_slope(0.0) == 0.0
        assert heat_pump.cooling_ratio_slope(50.0) == 0.0
        assert heat_pump.heating_ratio_slope(20.0) != 0.0
        assert heat_pump.cooling_ratio_slope(20.0) != 0.0

    def test_heat_pump_rejects_reversed_curve_temperature_limits(self):
        demo = json.loads((self.demos_path / "simulate_1_pipe_1_ghe_1_bldg_district.json").read_text())
        heat_pump_data = demo["heat_pump"]["hp1"]
        heat_pump_data["cooling_performance"]["minimum_curve_temperature"] = 40.0
        heat_pump_data["cooling_performance"]["maximum_curve_temperature"] = 30.0

        with pytest.raises(ValueError, match="Cooling minimum curve temperature"):
            HPmodel("hp1", heat_pump_data, ugt=20.0)

    def test_read_csv_data(self):
        # test typical csv file
        f_path_loads = self.test_data_directory / "Atlanta_Office_Building_Loads.csv"

        # find by column name
        data = read_csv_column(f_path_loads, column="Hourly heat extraction (W)")
        self.assertEqual(len(data), 8760)
        self.assertEqual(data[0], 0)
        self.assertAlmostEqual(data[5000], -192013, delta=1)
        self.assertEqual(data[-1], 0)

        # find by column index
        data = read_csv_column(f_path_loads, column=0)
        self.assertEqual(len(data), 8760)
        self.assertEqual(data[0], 0)
        self.assertAlmostEqual(data[5000], -192013, delta=1)
        self.assertEqual(data[-1], 0)

        # test irregular csv file
        f_path_data = self.test_data_directory / "rowwise_reference_values.csv"

        # find by column name
        data = read_csv_column(f_path_data, column="test_normal_spacing_target_spacings")
        self.assertEqual(data[0], 10)
        self.assertEqual(data[-1], 20)

        # find by column index
        data = read_csv_column(f_path_data, column=2)
        self.assertEqual(data[0], 10)
        self.assertEqual(data[-1], 20)

        # try passing incorrect column type
        with pytest.raises(TypeError):
            read_csv_column(f_path_data, column=1.5)

        # try non-numeric
        data = read_csv_column(
            f_path_data, column="test_shape_methods_point_intersections", try_convert_to_numeric=False
        )
        self.assertEqual(data[0], "TRUE")
        self.assertEqual(data[-1], "")
