from pathlib import Path
from unittest import TestCase

from ghedesigner.heat_pump import HeatPump


class TestHeatPump(TestCase):
    def test_fixed_cop_heating(self):
        heat_pump = HeatPump("load 1")
        heat_pump.set_loads_from_lambda(
            lambda _: 1000  # 1 kW, positive indicating heating the building, taking energy from loop
        )
        heat_pump.set_fixed_cop(10)  # COP of 10, meaning loop load should be 900
        loop_flow = 9 / 41  # made up to get the m_dot*cp to come out as 900, thus Q/m_dot_cp is 1, so DT = 1
        inlet_temp = 10
        outlet_temp = heat_pump.calculate(1, inlet_temp, loop_flow)
        self.assertAlmostEqual(outlet_temp, inlet_temp + 1)

    def test_fixed_cop_cooling(self):
        heat_pump = HeatPump("load 1")
        heat_pump.set_loads_from_lambda(
            lambda _: -1000  # 1 kW, positive indicating heating the building, taking energy from loop
        )
        heat_pump.set_fixed_cop(10)  # COP of 10, meaning loop load should be 1100
        loop_flow = 11 / 41  # made up to get the m_dot*cp to come out as 1100, thus Q/m_dot_cp is 1, so DT = 1
        inlet_temp = 10
        outlet_temp = heat_pump.calculate(1, inlet_temp, loop_flow)
        self.assertAlmostEqual(outlet_temp, inlet_temp - 1)

    def test_load_lookup(self):
        heat_pump = HeatPump("load 1")
        building_loads_path = Path(__file__).parent / "test_data" / "test_bldg_loads.csv"
        heat_pump.set_loads_from_file(building_loads_path)
        heat_pump.set_fixed_cop(3)
        loop_flow = 1.05
        inlet_temp = 10
        known_zero_load_time = 1  # first hour of the ground_loads has zero load
        outlet_temp = heat_pump.calculate(known_zero_load_time, inlet_temp, loop_flow)
        self.assertAlmostEqual(outlet_temp, inlet_temp)
        known_non_zero_load_time = 79
        outlet_temp = heat_pump.calculate(known_non_zero_load_time, inlet_temp, loop_flow)
        assert outlet_temp != inlet_temp
