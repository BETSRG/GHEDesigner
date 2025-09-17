from ghedesigner.tests.test_base_case import GHEBaseTest
from ghedesigner.ghe.hybrid_loads_stub_out import HybridLoadsCalc


class TestHybridUpdate(GHEBaseTest):
    def get_default_hybrid_object(self):
        return HybridLoadsCalc()

    def test_step_1(self):

        cop = 3

        building_loads = [100, 200, 300, -100, -200, -300]
        expected_ghe_loads = [x * (1 - 1 / cop) if x >=
                              0 else x * (1 + 1/cop) for x in building_loads]

        actual_ghe_loads = HybridLoadsCalc(cop_h=3, cop_c=3).step_1_bldg_to_ground_load(building_loads)
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

        # check 4: same peak up and down
        ground_loads = [-1.0, 1.0]
        normalized = HybridLoadsCalc.step_2_normalize_loads(ground_loads)
        new_min = min(normalized)
        new_max = max(normalized)
        self.assertAlmostEqual(-4000.0, new_min)
        self.assertAlmostEqual(4000.0, new_max)

        # check 5: unbalanced peaks, higher positive
        ground_loads = [-1.0, 2.0]
        normalized = HybridLoadsCalc.step_2_normalize_loads(ground_loads)
        new_min = min(normalized)
        new_max = max(normalized)
        self.assertAlmostEqual(-2000.0, new_min)
        self.assertAlmostEqual(4000.0, new_max)

        # check 6: unbalanced peaks, higher negative
        ground_loads = [-2.0, 1.0]
        normalized = HybridLoadsCalc.step_2_normalize_loads(ground_loads)
        new_min = min(normalized)
        new_max = max(normalized)
        self.assertAlmostEqual(-4000.0, new_min)
        self.assertAlmostEqual(2000.0, new_max)

    def test_step_3(self):
        h = self.get_default_hybrid_object()
        building_loads = self.get_atlanta_loads()
        ground_loads = HybridLoadsCalc().step_1_bldg_to_ground_load(building_loads)
        normalized_loads = HybridLoadsCalc.step_2_normalize_loads(ground_loads)
        with open('/tmp/normalized.csv', 'w') as f:
            f.writelines([str(n) + '\n' for n in normalized_loads])
        h.step_3_calc_monthly_load_metrics(normalized_loads)
        self.assertEqual(13, len(h.monthly_total_ground_load))
        self.assertEqual(13, len(h.monthly_peak_extraction))
        self.assertEqual(13, len(h.monthly_peak_rejection))
        self.assertEqual(13, len(h.monthly_ave_ground_load))

        #check values
        #check absolute value of peak is 4000
        self.assertAlmostEqual(4000, max(h.monthly_peak_extraction)|max(abs(h.monthly_peak_rejection)))

        # check absolute highest monthly average is still below 4000 w
        self.assertLessEqual (4000, max(abs(h.monthly_ave_ground_load)))

        #check the higheset total monthly load is still <= the maximum value of 4000 w per hour for a 31 day month
        self.assertLessEqual (4000 * 31 * 24, max(abs(h.monthly_total_ground_load)))

