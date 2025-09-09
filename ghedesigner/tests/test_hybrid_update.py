from ghedesigner.tests.test_base_case import GHEBaseTest
from ghedesigner.ghe.hybrid_loads_stub_out import HybridLoadsCalc


class TestHybridUpdate(GHEBaseTest):
    def get_default_hybrid_object(self):
        building_loads = [100, 200, 300, -100, -200, -300]
        return HybridLoadsCalc(building_loads)

    def test_step_1(self):

        cop = 3

        building_loads = [100, 200, 300, -100, -200, -300]
        expected_ghe_loads = [x * (1 - 1 / cop) if x >=
                              0 else x * (1 + 1/cop) for x in building_loads]

        actual_ghe_loads = HybridLoadsCalc(building_loads, cop_h=3, cop_c=3).step_1_bldg_to_ground_load()
        for idx, actual in enumerate(actual_ghe_loads):
            expected = expected_ghe_loads[idx]
            self.assertAlmostEqual(actual, expected)

    def test_step_2(self):
        
        # check 1: zero ground load should result in zero normalized load
        zero_loads = [0.0] * 8760
        normalized = HybridLoadsCalc.step_2_normalize_loads(zero_loads)
        self.assertTrue(all([x == 0.0 for x in normalized]))

        # check 2: all positive ground loads
        positive_ground_loads = [1.0 * x for x in range(8760)]
        normalized = HybridLoadsCalc.step_2_normalize_loads(positive_ground_loads)
        # new_min = min(normalized)
        new_max = max(normalized)
        # self.assertAlmostEqual(0.0, new_min)
        self.assertAlmostEqual(4000.0, new_max)

        # check 3: all negative ground loads
        negative_ground_loads = [-1.0 * x for x in range(8760)]
        normalized = HybridLoadsCalc.step_2_normalize_loads(negative_ground_loads)
        new_min = min(normalized)
        # new_max = max(normalized)
        self.assertAlmostEqual(-4000.0, new_min)
        # self.assertAlmostEqual(0.0, new_max)

        # check 4: typical ground loads
        # ground_loads = h.step_1_bldg_to_ground_load()
        # HybridLoadsCalc.step_2_normalize_loads(ground_loads)

