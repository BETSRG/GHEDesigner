from ghedesigner.tests.test_base_case import GHEBaseTest
from ghedesigner.ghe.hybrid_loads_stub_out import HybridLoadsCalc


class TestHybridUpdate(GHEBaseTest):
    def test_step_1(self):

        cop = 3

        building_loads = [100, 200, 300, -100, -200, -300]
        expected_ghe_loads = [x * (1 - 1 / cop) if x >=
                              0 else x * (1 + 1/cop) for x in building_loads]

        actual_ghe_loads = HybridLoadsCalc(building_loads).step_1_bldg_to_ground_load()
        for idx, actual in enumerate(actual_ghe_loads):
            expected = expected_ghe_loads[idx]
            self.assertAlmostEqual(actual, expected)

    def test_step_2(self):
        pass
