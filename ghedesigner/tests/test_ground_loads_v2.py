"""Tests for HybridLoadV2 (Spitler 2024 algorithm).

Includes:
- Unit tests for static/pure methods
- Unit tests for pipeline steps (peak identification, selection)
- Integration test with real BHE and Atlanta loads
- Edge case tests (expected to fail until special-case handling is added)
"""

import unittest
from calendar import monthrange
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from scipy.interpolate import interp1d

from ghedesigner.constants import HOURS_IN_YEAR, HRS_IN_DAY, MONTHS_IN_YEAR
from ghedesigner.ghe.ground_loads import first_month_hour, last_month_hour
from ghedesigner.ghe.ground_loads_v2 import HybridLoadV2


# ---------------------------------------------------------------------------
# Helper: build a minimal mock BHE so we can construct HybridLoadV2
# without computing real g-functions (for unit tests only).
# ---------------------------------------------------------------------------
def _make_mock_bhe():
    """Create a mock SingleUTube with minimal attributes for HybridLoadV2."""
    bhe = MagicMock()
    bhe.soil.k = 2.0
    bhe.calc_effective_borehole_resistance.return_value = 0.15
    bhe.t_s = 1.0e10  # large ts so log(t/ts) stays in range

    # STS g-function: simple linear g = 0.5 * ln(t/ts) clamped to [0, 10]
    def g_sts_func(lntts):
        vals = np.asarray(lntts, dtype=float)
        result = np.clip(0.5 * vals + 5.0, 0.0, 10.0)
        return result

    bhe.g_sts = g_sts_func
    return bhe


def _make_full_v2(loads, start_month=1, end_month=12):
    """Build a complete HybridLoadV2 with mock BHE."""
    bhe = _make_mock_bhe()
    return HybridLoadV2(loads, bhe, bhe, start_month, end_month)


def _make_v2_with_loads(loads):
    """Build HybridLoadV2 with mocked BHE so we can inspect monthly splits."""
    bhe = _make_mock_bhe()

    # Patch _run_hourly_simulation to avoid the full sim
    with (
        patch.object(
            HybridLoadV2,
            "_run_hourly_simulation",
            return_value=[0.0] * (len(loads) + 1),
        ),
        patch.object(HybridLoadV2, "_find_peak_durations"),
        patch.object(HybridLoadV2, "_process_month_loads"),
    ):
        obj = HybridLoadV2(loads, bhe, bhe, 1, 12)

    return obj


def _make_v2_with_synthetic_dt(hourly_delta_t, loads=None):
    """Build HybridLoadV2 with a prescribed hourly_delta_t for testing steps 4-5."""
    if loads is None:
        loads = [100.0] * HOURS_IN_YEAR  # dummy non-zero loads

    bhe = _make_mock_bhe()

    with (
        patch.object(
            HybridLoadV2,
            "_run_hourly_simulation",
            return_value=hourly_delta_t,
        ),
        patch.object(HybridLoadV2, "_find_peak_durations"),
        patch.object(HybridLoadV2, "_process_month_loads"),
    ):
        obj = HybridLoadV2(loads, bhe, bhe, 1, 12)

    return obj


def _analyze_and_export(
    loads: list,
    output_dir: Path,
    file_prefix: str,
    undisturbed_ground_t: float = 20.0,
    m_dot: float = 0.1,
    cp: float = 4186.0,
) -> None:
    """Build HybridLoadV2 from any hourly load profile and export analysis CSVs.

    Exports two files to output_dir:
      - {file_prefix}_hourly_results.csv  : per-hour load, delta_t, tf_ave, eft, exft
      - {file_prefix}_hybrid_results.csv  : hybrid time steps in step-function format
                                            with predicted delta_T

    :param loads: 8760 hourly loads in Watts
    :param output_dir: directory to write CSV files
    :param file_prefix: prefix for output file names
    :param undisturbed_ground_t: undisturbed ground temperature in °C
    :param m_dot: fluid mass flow rate in kg/s
    :param cp: fluid specific heat in J/(kg·K)
    """
    bhe = _make_mock_bhe()
    obj = HybridLoadV2(loads, bhe, bhe, 1, 12)

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    print(f"\n===== {file_prefix}: Peak Temperature & Load Summary =====")
    print(f"{'Month':<6} {'Max dT':>8} {'Hour':>6} {'Peak CL (kW)':>13} "
          f"{'Min dT':>8} {'Hour':>6} {'Peak HL (kW)':>13}")
    print("-" * 68)
    for m in range(MONTHS_IN_YEAR):
        print(f"{month_names[m]:<6} {obj.monthly_max_dt[m]:>8.3f} {obj.monthly_max_dt_hour[m]:>6d} "
              f"{obj.monthly_peak_cl[m]:>13.3f} {obj.monthly_min_dt[m]:>8.3f} "
              f"{obj.monthly_min_dt_hour[m]:>6d} {obj.monthly_peak_hl[m]:>13.3f}")
    print(f"\nPeak cooling months: {[month_names[m] for m in obj.peak_cooling_months]}")
    print(f"Peak heating months: {[month_names[m] for m in obj.peak_heating_months]}")
    print("\nPeak durations (hours):")
    for m in obj.peak_cooling_months:
        print(f"  {month_names[m]} cooling: {obj.monthly_peak_cl_duration[m]:.2f}")
    for m in obj.peak_heating_months:
        print(f"  {month_names[m]} heating: {obj.monthly_peak_hl_duration[m]:.2f}")
    print("=" * 68)

    print(f"\n===== {file_prefix}: Hybrid Load Profile (All Steps) =====")
    print(f"{'Step':>4}  {'Start Hr':>10}  {'End Hr':>10}  {'Duration':>10}  {'Load (kW)':>10}")
    print("-" * 52)
    for i in range(1, len(obj.hour)):
        h_start = obj.hour[i - 1]
        h_end = obj.hour[i]
        print(f"{i:>4}  {h_start:>10.1f}  {h_end:>10.1f}  {h_end - h_start:>10.1f}  {obj.load[i]:>10.3f}")
    print("=" * 52)

    # Hourly results: delta_t, tf_ave, eft, exft
    # q is aligned to hourly_delta_t indexing (8761 entries, index 0 = 0)
    q = np.hstack((0.0, obj.normalized_loads))
    tf_ave = np.array(obj.hourly_delta_t) + undisturbed_ground_t
    eft = tf_ave + q / (m_dot * cp)
    exft = tf_ave - q / (m_dot * cp)

    hourly_path = output_dir / f"{file_prefix}_hrly_results.csv"
    with open(hourly_path, "w") as f:
        f.write("hour,load_w,delta_t,tf_ave,eft,exft\n")
        for hour_idx, (load_val, dt, tf_ave_val, eft_val, exft_val) in enumerate(
            zip(q, obj.hourly_delta_t, tf_ave, eft, exft)
        ):
            f.write(f"{hour_idx},{load_val:.6f},{dt:.6f},{tf_ave_val:.6f},{eft_val:.6f},{exft_val:.6f}\n")
    print(f"\nExported hourly results to {hourly_path}")
    print(f"  entries={len(obj.hourly_delta_t)}, "
          f"delta_T=[{min(obj.hourly_delta_t):.4f}, {max(obj.hourly_delta_t):.4f}], "
          f"EFT=[{min(eft):.4f}, {max(eft):.4f}] °C")

    # Hybrid results: run simulate_hourly on hybrid sequence, export step-function format
    hybrid_delta_t = obj.hybrid_dt

    hybrid_path = output_dir / f"{file_prefix}_hybrid_results.csv"
    with open(hybrid_path, "w") as f:
        f.write("hour,load_kw,predicted_delta_t\n")
        for i in range(len(obj.hour)):
            f.write(f"{obj.hour[i]:.4f},{obj.load[i]:.6f},{hybrid_delta_t[i]:.6f}\n")
            # Duplicate timestep at load transitions to create vertical step edges for plotting
            if i < len(obj.hour) - 1 and obj.load[i] != obj.load[i + 1]:
                f.write(f"{obj.hour[i]:.4f},{obj.load[i + 1]:.6f},{hybrid_delta_t[i]:.6f}\n")
    print(f"Exported hybrid results to {hybrid_path}")


class TestNormalizeLoads(unittest.TestCase):
    """Step 1: normalize_loads static method."""

    def test_basic_normalization(self):
        """Peak absolute value should become 4000 W."""
        loads = [1000, -2000, 500, -1500]
        result = HybridLoadV2.normalize_loads(loads)
        # max(|loads|) = 2000, scale = 4000/2000 = 2
        np.testing.assert_allclose(result, [2000, -4000, 1000, -3000])

    def test_peak_equals_norm_target(self):
        """After normalization, abs peak should be NORM_LOAD_W."""
        loads = [300, -700, 200, 100, -400]
        result = HybridLoadV2.normalize_loads(loads)
        self.assertAlmostEqual(np.max(np.abs(result)), HybridLoadV2.NORM_LOAD_W)

    def test_all_zeros(self):
        """All-zero loads should remain all zeros (no divide-by-zero)."""
        loads = [0.0] * 100
        result = HybridLoadV2.normalize_loads(loads)
        np.testing.assert_array_equal(result, np.zeros(100))

    def test_uniform_positive(self):
        """Uniform positive loads should all become NORM_LOAD_W."""
        loads = [500.0] * HOURS_IN_YEAR
        result = HybridLoadV2.normalize_loads(loads)
        np.testing.assert_allclose(result, [HybridLoadV2.NORM_LOAD_W] * HOURS_IN_YEAR)

    def test_single_value(self):
        """Single non-zero value should become +/- NORM_LOAD_W."""
        result = HybridLoadV2.normalize_loads([-123.0])
        self.assertAlmostEqual(result[0], -HybridLoadV2.NORM_LOAD_W)


class TestSimulateHourly(unittest.TestCase):
    """Hourly g-function superposition simulation (static method)."""

    def test_zero_load_gives_zero_dt(self):
        """Zero load at all time steps should produce zero delta_T."""
        hour_time = np.arange(25)  # 0..24
        q = np.zeros(25)
        # Simple g-function: g(x) = 1.0 for all x
        g_sts = interp1d([-50, 50], [1.0, 1.0], fill_value=1.0, bounds_error=False)
        dt = HybridLoadV2.simulate_hourly(hour_time, q, g_sts, resist_bh=0.1, two_pi_k=12.0, ts=1e10, h=100.0)
        self.assertEqual(len(dt), 25)
        for val in dt:
            self.assertAlmostEqual(val, 0.0)

    def test_constant_load_monotonic(self):
        """Constant positive rejection should give monotonically increasing delta_T."""
        hour_time = np.arange(51)  # 0..50
        q = np.hstack((0.0, np.ones(50) * 1000.0))
        g_sts = interp1d([-100, 100], [0.0, 10.0], fill_value=(0.0, 10.0), bounds_error=False)
        dt = HybridLoadV2.simulate_hourly(hour_time, q, g_sts, resist_bh=0.1, two_pi_k=12.0, ts=1e10, h=100.0)
        # delta_T should be non-negative and generally increasing
        for i in range(2, len(dt)):
            self.assertGreaterEqual(dt[i], dt[i - 1] - 1e-10)

    def test_output_length(self):
        """Output length should match hour_time length."""
        n = 100
        hour_time = np.arange(n + 1)
        q = np.hstack((0.0, np.ones(n) * 500.0))
        g_sts = interp1d([-50, 50], [1.0, 1.0], fill_value=1.0, bounds_error=False)
        dt = HybridLoadV2.simulate_hourly(hour_time, q, g_sts, resist_bh=0.1, two_pi_k=12.0, ts=1e10, h=100.0)
        self.assertEqual(len(dt), n + 1)

    def test_first_element_is_zero(self):
        """First element of delta_T should always be 0."""
        hour_time = np.arange(11)
        q = np.hstack((0.0, np.ones(10) * 2000.0))
        g_sts = interp1d([-50, 50], [1.0, 1.0], fill_value=1.0, bounds_error=False)
        dt = HybridLoadV2.simulate_hourly(hour_time, q, g_sts, resist_bh=0.1, two_pi_k=12.0, ts=1e10, h=100.0)
        self.assertAlmostEqual(dt[0], 0.0)


class TestSplitLoadsByMonth(unittest.TestCase):
    """Step 2: split_loads_by_month."""

    def test_monthly_totals_all_rejection(self):
        """All-rejection loads: monthly_cl should have totals, monthly_hl should be zero."""
        # 8760 hours, constant -1000 W (rejection)
        loads = [-1000.0] * HOURS_IN_YEAR
        obj = _make_v2_with_loads(loads)
        # Every month should have non-zero cooling total
        for m in range(MONTHS_IN_YEAR):
            self.assertGreater(obj.monthly_cl[m], 0.0)
            self.assertAlmostEqual(obj.monthly_hl[m], 0.0)

    def test_monthly_totals_all_extraction(self):
        """All-extraction loads: monthly_hl should have totals, monthly_cl should be zero."""
        loads = [1000.0] * HOURS_IN_YEAR
        obj = _make_v2_with_loads(loads)
        for m in range(MONTHS_IN_YEAR):
            self.assertAlmostEqual(obj.monthly_cl[m], 0.0)
            self.assertGreater(obj.monthly_hl[m], 0.0)

    def test_peak_values_correct(self):
        """Verify peak values match the expected max hourly load in each month."""
        loads = [0.0] * HOURS_IN_YEAR
        # January: hours 0-743 (744 hours in Jan)
        for h in range(744):
            loads[h] = 500.0  # extraction
        loads[100] = 2000.0  # peak extraction in January
        # February: hours 744-1415 (672 hours in Feb for non-leap year)
        for h in range(744, 744 + 672):
            loads[h] = -800.0  # rejection
        loads[800] = -3000.0  # peak rejection in February

        obj = _make_v2_with_loads(loads)
        # January (index 0) peak extraction = 2000 W = 2.0 kW
        self.assertAlmostEqual(obj.monthly_peak_hl[0], 2.0)
        # February (index 1) peak rejection = 3000 W = 3.0 kW
        self.assertAlmostEqual(obj.monthly_peak_cl[1], 3.0)

    def test_total_hours_match_8760(self):
        """Sum of hours across all months should be 8760 for non-leap year."""
        loads = [100.0] * HOURS_IN_YEAR
        obj = _make_v2_with_loads(loads)
        total_hours = sum(HRS_IN_DAY * obj.days_in_month[m] for m in range(MONTHS_IN_YEAR))
        self.assertEqual(total_hours, HOURS_IN_YEAR)


class TestPeakTemperatureIdentification(unittest.TestCase):
    """Steps 4-5: _find_monthly_peak_temperatures and _select_peak_months."""

    def test_peak_months_correct_count(self):
        """Should select exactly NUM_PEAK_MONTHS for cooling and heating."""
        # Create synthetic delta_T: ramp up Jan-Jun, ramp down Jul-Dec
        dt = [0.0]  # index 0 placeholder
        for h in range(HOURS_IN_YEAR):
            # Simple pattern: higher in summer, lower in winter
            month_frac = h / HOURS_IN_YEAR
            dt.append(10.0 * np.sin(2 * np.pi * month_frac))
        obj = _make_v2_with_synthetic_dt(dt)
        self.assertEqual(len(obj.peak_cooling_months), HybridLoadV2.NUM_PEAK_MONTHS)
        self.assertEqual(len(obj.peak_heating_months), HybridLoadV2.NUM_PEAK_MONTHS)

    def test_peak_cooling_months_have_highest_dt(self):
        """Peak cooling months should be those with highest max delta_T."""
        dt = [0.0]
        for h in range(HOURS_IN_YEAR):
            month_frac = h / HOURS_IN_YEAR
            dt.append(10.0 * np.sin(2 * np.pi * month_frac))
        obj = _make_v2_with_synthetic_dt(dt)

        # Peak cooling months should have max_dt >= all non-peak months
        peak_max_dts = [obj.monthly_max_dt[m] for m in obj.peak_cooling_months]
        non_peak_months = [m for m in range(MONTHS_IN_YEAR) if m not in obj.peak_cooling_months]
        non_peak_max_dts = [obj.monthly_max_dt[m] for m in non_peak_months]
        self.assertGreaterEqual(min(peak_max_dts), max(non_peak_max_dts))

    def test_peak_heating_months_have_lowest_dt(self):
        """Peak heating months should be those with lowest min delta_T."""
        dt = [0.0]
        for h in range(HOURS_IN_YEAR):
            month_frac = h / HOURS_IN_YEAR
            dt.append(10.0 * np.sin(2 * np.pi * month_frac))
        obj = _make_v2_with_synthetic_dt(dt)

        peak_min_dts = [obj.monthly_min_dt[m] for m in obj.peak_heating_months]
        non_peak_months = [m for m in range(MONTHS_IN_YEAR) if m not in obj.peak_heating_months]
        non_peak_min_dts = [obj.monthly_min_dt[m] for m in non_peak_months]
        self.assertLessEqual(max(peak_min_dts), min(non_peak_min_dts))

    def test_peak_hour_within_month_bounds(self):
        """Peak hour-of-year should fall within the correct month's hour range."""
        rng = np.random.default_rng(42)

        h = np.arange(HOURS_IN_YEAR)
        noise = rng.normal(0.0, 0.1, size=HOURS_IN_YEAR)

        dt = np.concatenate(
            (
                [0.0],
                5.0 * np.sin(2 * np.pi * h / HOURS_IN_YEAR) + noise,
            )
        )

        obj = _make_v2_with_synthetic_dt(dt.tolist())

        hours_in_previous = 0
        for m in range(MONTHS_IN_YEAR):
            hours_in_month = HRS_IN_DAY * obj.days_in_month[m]
            start_hr = hours_in_previous + 1
            end_hr = hours_in_previous + hours_in_month
            if obj.monthly_max_dt_hour[m] > 0:
                self.assertGreaterEqual(obj.monthly_max_dt_hour[m], start_hr)
                self.assertLessEqual(obj.monthly_max_dt_hour[m], end_hr)
            if obj.monthly_min_dt_hour[m] > 0:
                self.assertGreaterEqual(obj.monthly_min_dt_hour[m], start_hr)
                self.assertLessEqual(obj.monthly_min_dt_hour[m], end_hr)
            hours_in_previous += hours_in_month


class TestProcessMonthLoads(unittest.TestCase):
    """Step 7: _process_month_loads output structure."""

    def test_hour_array_monotonically_increasing(self):
        """Hour array should be monotonically increasing after the initial zeros.

        The array format has two leading zeros [0, 0, ...] matching the
        original HybridLoad convention. After that, hours should strictly increase.
        """
        loads = [1000.0 * np.sin(2 * np.pi * h / HOURS_IN_YEAR) for h in range(HOURS_IN_YEAR)]
        obj = _make_full_v2(loads)
        # Skip the initial [0, 0] pair (indices 0 and 1)
        for i in range(2, len(obj.hour)):
            self.assertGreater(
                obj.hour[i],
                obj.hour[i - 1],
                f"Hour array not monotonic at index {i}: {obj.hour[i]} <= {obj.hour[i - 1]}",
            )

    def test_load_and_hour_same_length(self):
        """load and hour arrays should be the same length."""
        loads = [500.0] * HOURS_IN_YEAR
        obj = _make_full_v2(loads)
        self.assertEqual(len(obj.load), len(obj.hour))

    def test_step_func_load_length(self):
        """step_func_load should have len(hour) elements.

        The array format includes a leading zero, then n-1 step values,
        matching the original HybridLoad convention.
        """
        loads = [500.0] * HOURS_IN_YEAR
        obj = _make_full_v2(loads)
        self.assertEqual(len(obj.step_func_load), len(obj.hour))

    def test_step_func_load_values(self):
        """step_func_load[i] (for i>=1) should equal load[i] - load[i-1].

        Index 0 is a leading zero from initialization.
        """
        loads = [1000.0 * np.sin(2 * np.pi * h / HOURS_IN_YEAR) for h in range(HOURS_IN_YEAR)]
        obj = _make_full_v2(loads)
        # Index 0 is the initial zero
        self.assertAlmostEqual(obj.step_func_load[0], 0.0)
        # Indices 1.n-1 hold the step changes
        for i in range(1, len(obj.step_func_load)):
            expected = obj.load[i] - obj.load[i - 1]
            self.assertAlmostEqual(
                obj.step_func_load[i], expected, places=10, msg=f"step_func_load mismatch at index {i}"
            )

    def test_final_hour_covers_full_year(self):
        """Last hour should be at or near 8760 for a 12-month simulation."""
        loads = [500.0] * HOURS_IN_YEAR
        obj = _make_full_v2(loads)
        self.assertAlmostEqual(obj.hour[-1], HOURS_IN_YEAR, delta=1.0)

    def test_dual_peak_month_constructs_successfully(self):
        """A month with both heating and cooling peaks should produce valid output.

        Creates loads where July has strong rejection AND strong extraction,
        well-separated in time (so no overlap concern). Verifies the month
        appears in both peak lists, and the output arrays contain both a
        positive (cooling) and negative (heating) peak load step for July.
        """
        loads = [0.0] * HOURS_IN_YEAR

        # July: hour 4344..5087 (744 hours)
        july_start = 744 + 672 + 744 + 720 + 744 + 720

        # Strong rejection (cooling) in early July
        for h in range(july_start + 50, july_start + 60):
            loads[h] = -6000.0

        # Strong extraction (heating) in late July — well separated
        for h in range(july_start + 600, july_start + 610):
            loads[h] = 6000.0

        # Mild loads in other months, so July dominates both peaks
        for h in range(july_start):
            loads[h] = 300.0  # mild extraction (winter)
        for h in range(july_start + 744, HOURS_IN_YEAR):
            loads[h] = -300.0  # mild rejection (fall)
        # Fill non-peak July hours with small values
        for h in range(july_start, july_start + 744):
            if loads[h] == 0.0:
                loads[h] = 50.0

        obj = _make_full_v2(loads)

        # July (index 6) should be in both peak lists
        self.assertIn(6, obj.peak_cooling_months, f"July not in cooling peaks: {obj.peak_cooling_months}")
        self.assertIn(6, obj.peak_heating_months, f"July not in heating peaks: {obj.peak_heating_months}")

        # Both durations should be positive
        self.assertGreater(obj.monthly_peak_cl_duration[6], 0.0)
        self.assertGreater(obj.monthly_peak_hl_duration[6], 0.0)

        # The output arrays should contain both a positive and negative
        # peak load within July's hour range (first_month_hour uses 1-indexed months)
        fmh = first_month_hour(7, obj.years)
        lmh = last_month_hour(7, obj.years)

        found_cooling_peak = False
        found_heating_peak = False
        for i in range(1, len(obj.hour)):
            if fmh <= obj.hour[i] <= lmh:
                if obj.load[i] > 0.5:
                    found_cooling_peak = True
                elif obj.load[i] < -0.5:
                    found_heating_peak = True

        self.assertTrue(found_cooling_peak, "No positive (cooling) peak load found in July's hybrid steps")
        self.assertTrue(found_heating_peak, "No negative (heating) peak load found in July's hybrid steps")

        # Arrays should still be well-formed
        self.assertEqual(len(obj.load), len(obj.hour))
        for i in range(2, len(obj.hour)):
            self.assertGreater(obj.hour[i], obj.hour[i - 1], f"Hour not monotonic at {i}")

        # Energy conservation: hybrid energy for July should match original
        hybrid_energy = 0.0
        for i in range(1, len(obj.hour)):
            h_start = obj.hour[i - 1]
            h_end = obj.hour[i]
            if h_end <= fmh or h_start >= lmh:
                continue
            step_start = max(h_start, fmh)
            step_end = min(h_end, lmh)
            duration = step_end - step_start
            hybrid_energy += obj.load[i] * duration

        original_energy = obj.monthly_cl[6] - obj.monthly_hl[6]
        if abs(original_energy) > 1.0:
            rel_error = abs(hybrid_energy - original_energy) / abs(original_energy)
            self.assertLess(
                rel_error,
                0.05,
                f"July energy not conserved: hybrid={hybrid_energy:.1f} kWh vs "
                f"original={original_energy:.1f} kWh (rel error {rel_error:.3f})",
            )


class TestIntegrationWithAtlantaLoads(unittest.TestCase):
    """Integration test: construct HybridLoadV2 with real-ish BHE and Atlanta loads."""

    @classmethod
    def setUpClass(cls):
        """Load the Atlanta building loads CSV."""

        csv_path = Path(__file__).parent / "test_data" / "Atlanta_Office_Building_Loads.csv"
        if csv_path.exists():
            raw_lines = csv_path.read_text().split("\n")
            cls.atlanta_loads = [float(x) for x in raw_lines[1:] if x.strip() != ""]
        else:
            cls.atlanta_loads = None

    def test_construction_completes(self):
        """HybridLoadV2 should construct without errors using Atlanta loads."""
        if self.atlanta_loads is None:
            self.skipTest("Atlanta loads CSV not found")
        bhe = _make_mock_bhe()
        obj = HybridLoadV2(self.atlanta_loads, bhe, bhe, 1, 12)
        self.assertGreater(len(obj.load), 0)
        self.assertGreater(len(obj.hour), 0)
        self.assertEqual(len(obj.peak_cooling_months), HybridLoadV2.NUM_PEAK_MONTHS)
        self.assertEqual(len(obj.peak_heating_months), HybridLoadV2.NUM_PEAK_MONTHS)

    def test_export_atlanta_analysis(self):
        """Export hourly and hybrid results CSVs for the Atlanta load profile."""
        if self.atlanta_loads is None:
            self.skipTest("Atlanta loads CSV not found")
        output_dir = Path(__file__).parent / "output"
        _analyze_and_export(self.atlanta_loads, output_dir, "ATL")

    def test_peak_durations_positive(self):
        """All peak durations should be positive for real building loads."""
        if self.atlanta_loads is None:
            self.skipTest("Atlanta loads CSV not found")
        bhe = _make_mock_bhe()
        obj = HybridLoadV2(self.atlanta_loads, bhe, bhe, 1, 12)
        for m in obj.peak_cooling_months:
            self.assertGreater(
                obj.monthly_peak_cl_duration[m], 0.0, f"Cooling peak duration for month {m} should be > 0"
            )
        for m in obj.peak_heating_months:
            self.assertGreater(
                obj.monthly_peak_hl_duration[m], 0.0, f"Heating peak duration for month {m} should be > 0"
            )

    def test_normalized_loads_peak_is_4000(self):
        """Normalized loads should have absolute peak of 4000 W."""
        if self.atlanta_loads is None:
            self.skipTest("Atlanta loads CSV not found")
        bhe = _make_mock_bhe()
        obj = HybridLoadV2(self.atlanta_loads, bhe, bhe, 1, 12)
        self.assertAlmostEqual(np.max(np.abs(obj.normalized_loads)), HybridLoadV2.NORM_LOAD_W, places=1)

    def test_monthly_energy_conservation(self):
        """For peak months, the hybrid load steps should conserve monthly energy."""
        if self.atlanta_loads is None:
            self.skipTest("Atlanta loads CSV not found")
        bhe = _make_mock_bhe()
        obj = HybridLoadV2(self.atlanta_loads, bhe, bhe, 1, 12)

        # Verify that for each month in the hybrid representation,
        # the total energy (load * duration) roughly equals the original monthly net
        for m in range(MONTHS_IN_YEAR):
            # first_month_hour/last_month_hour use 1-indexed months
            fmh = first_month_hour(m + 1, obj.years)
            lmh = last_month_hour(m + 1, obj.years)

            # Find all hybrid steps within this month
            hybrid_energy = 0.0
            for i in range(1, len(obj.hour)):
                h_start = obj.hour[i - 1]
                h_end = obj.hour[i]
                # Check if this step overlaps with the month
                if h_end <= fmh or h_start >= lmh:
                    continue
                # Clamp to month boundaries
                step_start = max(h_start, fmh)
                step_end = min(h_end, lmh)
                duration = step_end - step_start
                hybrid_energy += obj.load[i] * duration

            # Original monthly net energy (kWh): rejection - extraction
            original_energy = obj.monthly_cl[m] - obj.monthly_hl[m]

            # Allow some tolerance since the hybrid scheme approximates
            if abs(original_energy) > 1.0:  # skip months with negligible energy
                rel_error = abs(hybrid_energy - original_energy) / abs(original_energy)
                self.assertLess(
                    rel_error,
                    0.05,
                    f"Month {m}: hybrid energy {hybrid_energy:.1f} vs original {original_energy:.1f} "
                    f"(rel error {rel_error:.3f})",
                )


# ============================================================================
# Edge case tests - these exercise known limitations.
# Expected to FAIL until special-case handling is implemented.
# ============================================================================


class TestEdgeCase1PeakAtMonthStart(unittest.TestCase):
    """Edge case 1: Peak temperature occurs at the very first hour of a month.

    When the peak temperature is at hour 1 of a month, centering the peak
    duration around it would require extending into the previous month.
    Instead, the peak should be shifted forward within the month so
    that it starts at the month boundary and the full duration is preserved.
    """

    def test_peak_at_january_hour_1(self):
        """Peak at hour 1 of January: peak should shift forward, keeping full duration.

        Creates loads where the highest extraction occurs at hour 0 (first
        hour of January). Centering would push the start before the month,
        so the peak is shifted forward to start at fmh with full duration.
        """
        # Build loads: strong extraction spike at hour 0, mild rest of year
        loads = [100.0] * HOURS_IN_YEAR
        loads[0] = 5000.0  # huge extraction at hour 0 (first hour of Jan)
        # Add some mild rejection in summer so we have cooling peaks too
        for h in range(4000, 5000):
            loads[h] = -2000.0

        bhe = _make_mock_bhe()
        obj = HybridLoadV2(loads, bhe, bhe, 1, 12)

        # The heating peak for January should be month 0
        self.assertIn(0, obj.peak_heating_months)

        # The peak duration should be >= 1 hour
        d = obj.monthly_peak_hl_duration[0]
        self.assertGreaterEqual(d, 1.0, "Peak duration should be >= 1 hour")

        # first_month_hour uses 1-indexed months
        fmh = first_month_hour(1, obj.years)

        # Find the heating peak step in the load array
        peak_found = False
        for i in range(1, len(obj.hour)):
            if obj.load[i] < -0.5:  # negative = extraction in output convention
                peak_start = obj.hour[i - 1]
                peak_end = obj.hour[i]
                actual_duration = peak_end - peak_start

                # Peak should start at or after the month boundary (shifted forward)
                self.assertGreaterEqual(peak_start, fmh - 1, "Peak should not extend before the month start")

                # The full duration should be preserved (not truncated)
                self.assertAlmostEqual(
                    actual_duration,
                    d,
                    delta=1.0,
                    msg=f"Peak duration should be preserved: expected ~{d:.1f}h, got {actual_duration:.1f}h",
                )
                peak_found = True
                break

        self.assertTrue(peak_found, "Could not find a heating peak event in the load array")

        # Hour array must remain monotonic
        for i in range(2, len(obj.hour)):
            self.assertGreater(obj.hour[i], obj.hour[i - 1], f"Hour not monotonic at index {i}")


class TestEdgeCase2PeakSpanningMonthBoundary(unittest.TestCase):
    """Edge case 2: Peak event duration so long it spans a month boundary.

    When a peak event has a long duration and occurs near the start or end
    of a month, the event can't fit entirely within the month. The current
    code clamps to month boundaries, which truncates the peak and violates
    energy conservation.

    Per the PDF (page 6, Case 2): Adjacent peak events from neighboring
    months that collide should be merged or handled specially.
    """

    @unittest.expectedFailure
    def test_adjacent_month_peaks_collision(self):
        """Two adjacent months with peaks near their shared boundary.

        Creates loads where January has a strong peak near the end of the
        month and February has a strong peak near the start. With long
        durations, the two peak events would overlap at the month boundary.
        """
        loads = [100.0] * HOURS_IN_YEAR  # mild baseline extraction

        # January: strong extraction spike near end of month (hour ~740)
        jan_hours = 744
        for h in range(jan_hours - 10, jan_hours):
            loads[h] = 4000.0  # big extraction near end of Jan

        # February: strong extraction spike near start of month (hour ~745)
        for h in range(jan_hours, jan_hours + 10):
            loads[h] = 4000.0  # big extraction near start of Feb

        # Summer rejection for cooling peaks
        for h in range(4000, 5000):
            loads[h] = -3000.0

        bhe = _make_mock_bhe()
        obj = HybridLoadV2(loads, bhe, bhe, 1, 12)

        # Both months should be in peak heating months (0-indexed: Jan=0, Feb=1)
        jan_is_peak = 0 in obj.peak_heating_months
        feb_is_peak = 1 in obj.peak_heating_months

        if jan_is_peak and feb_is_peak:
            # Check that the hour array remains strictly monotonic
            for i in range(1, len(obj.hour)):
                self.assertGreater(
                    obj.hour[i],
                    obj.hour[i - 1],
                    f"Hour array not monotonic at index {i}: "
                    f"hour[{i}]={obj.hour[i]}, hour[{i - 1}]={obj.hour[i - 1]}. "
                    f"Adjacent month peaks may have collided at the boundary.",
                )

            # Additionally, no load step should have negative duration
            for i in range(1, len(obj.hour)):
                duration = obj.hour[i] - obj.hour[i - 1]
                self.assertGreater(duration, 0, f"Negative or zero duration at step {i}: {duration}")
        else:
            self.fail(f"Expected Jan and Feb to be heating peak months. Got heating peaks: {obj.peak_heating_months}")


class TestEdgeCase3TwoPeaksCollideInMonth(unittest.TestCase):
    """Edge case 3: Two peak events (cooling and heating) collide within one month.

    When a month has both a cooling peak and a heating peak, and the two
    temperature peaks occur close together in time, their durations can
    overlap. The overlap-prevention logic should push the second peak
    forward so the hour array stays strictly monotonic and both peaks
    are still represented.
    """

    def test_close_peaks_same_month(self):
        """Cooling and heating peaks close together in July should not overlap.

        Places a strong rejection spike and a strong extraction spike only
        ~20 hours apart in July. With long peak durations the events would
        overlap if not handled, but the overlap-prevention logic should
        shift the second event forward.
        """
        loads = [0.0] * HOURS_IN_YEAR
        july_start = 744 + 672 + 744 + 720 + 744 + 720  # hour 4344

        # Strong rejection (cooling) peak at hour ~4394 (50 hours into July)
        for h in range(july_start + 45, july_start + 55):
            loads[h] = -6000.0

        # Strong extraction (heating) peak at hour ~4414 (70 hours into July)
        # Only 20 hours after the cooling peak — durations will likely overlap
        for h in range(july_start + 65, july_start + 75):
            loads[h] = 6000.0

        # Mild loads in other months, so July dominates both peak lists
        for h in range(july_start):
            loads[h] = 300.0
        for h in range(july_start + 744, HOURS_IN_YEAR):
            loads[h] = -300.0
        for h in range(july_start, july_start + 744):
            if loads[h] == 0.0:
                loads[h] = 50.0

        obj = _make_full_v2(loads)

        # July (index 6) should appear in both peak lists
        self.assertIn(6, obj.peak_cooling_months, f"July not in cooling peaks: {obj.peak_cooling_months}")
        self.assertIn(6, obj.peak_heating_months, f"July not in heating peaks: {obj.peak_heating_months}")

        # Both durations should be positive
        self.assertGreater(obj.monthly_peak_cl_duration[6], 0.0)
        self.assertGreater(obj.monthly_peak_hl_duration[6], 0.0)

        # Hour array must be strictly monotonic (the main thing overlap breaks)
        for i in range(2, len(obj.hour)):
            self.assertGreater(
                obj.hour[i],
                obj.hour[i - 1],
                f"Hour not monotonic at index {i}: hour[{i}]={obj.hour[i]}, hour[{i - 1}]={obj.hour[i - 1]}",
            )

        # Both a cooling and heating peak load should appear in July
        # (first_month_hour/last_month_hour use 1-indexed months)
        fmh = first_month_hour(7, obj.years)
        lmh = last_month_hour(7, obj.years)

        found_cooling = False
        found_heating = False
        for i in range(1, len(obj.hour)):
            if fmh <= obj.hour[i] <= lmh:
                if obj.load[i] > 0.5:
                    found_cooling = True
                elif obj.load[i] < -0.5:
                    found_heating = True

        self.assertTrue(found_cooling, "No cooling peak load found in July's hybrid steps")
        self.assertTrue(found_heating, "No heating peak load found in July's hybrid steps")

        # Load/hour arrays should have consistent lengths
        self.assertEqual(len(obj.load), len(obj.hour))
        self.assertEqual(len(obj.step_func_load), len(obj.hour))


class TestMultiYearSimulation(unittest.TestCase):
    """Tests for multi-year simulation and leap year handling."""

    def test_multi_year_construction(self):
        """HybridLoadV2 should construct successfully with end_month > 12."""
        loads = [500.0] * HOURS_IN_YEAR
        obj = _make_full_v2(loads, start_month=1, end_month=240)
        self.assertGreater(len(obj.load), 0)
        self.assertGreater(len(obj.hour), 0)
        self.assertEqual(len(obj.load), len(obj.hour))

    def test_multi_year_years_list(self):
        """years list should have one entry per simulated year."""
        loads = [500.0] * HOURS_IN_YEAR
        obj = _make_full_v2(loads, start_month=1, end_month=240)
        self.assertEqual(len(obj.years), 20)

    def test_single_year_years_list(self):
        """Single-year simulation should have exactly one year."""
        loads = [500.0] * HOURS_IN_YEAR
        obj = _make_full_v2(loads, start_month=1, end_month=12)
        self.assertEqual(len(obj.years), 1)

    def test_multi_year_hour_monotonic(self):
        """Hour array should be strictly monotonic for multi-year simulation."""
        loads = [1000.0 * np.sin(2 * np.pi * h / HOURS_IN_YEAR) for h in range(HOURS_IN_YEAR)]
        obj = _make_full_v2(loads, start_month=1, end_month=240)
        for i in range(2, len(obj.hour)):
            self.assertGreater(
                obj.hour[i],
                obj.hour[i - 1],
                f"Hour array not monotonic at index {i}: {obj.hour[i]} <= {obj.hour[i - 1]}",
            )

    def test_multi_year_final_hour(self):
        """Last hour should cover 20 years of hours."""
        loads = [500.0] * HOURS_IN_YEAR
        obj = _make_full_v2(loads, start_month=1, end_month=240)
        # 20 years ~= 20 * 8760 = 175200 hours (varies with leap years)
        self.assertGreater(obj.hour[-1], 175000)
        self.assertLess(obj.hour[-1], 176000)

    def test_multi_year_load_hour_same_length(self):
        """load and hour arrays must be the same length for multi-year."""
        loads = [1000.0 * np.sin(2 * np.pi * h / HOURS_IN_YEAR) for h in range(HOURS_IN_YEAR)]
        obj = _make_full_v2(loads, start_month=1, end_month=240)
        self.assertEqual(len(obj.load), len(obj.hour))
        self.assertEqual(len(obj.step_func_load), len(obj.hour))

    def test_leap_year_february_hours(self):
        """Leap year February should have 696 hours (29 days), not 672."""

        loads = [500.0] * HOURS_IN_YEAR
        obj = _make_full_v2(loads, start_month=1, end_month=48)  # 4 years

        # Base year is 2026. Year 3 (index 2) is 2028, which is a leap year.
        # Month 26 = February of year 3 (2028).
        self.assertEqual(obj.years[2], 2028)
        self.assertEqual(monthrange(2028, 2)[1], 29)

        # Verify total simulation hours for 4 years includes one leap year (2028).
        expected_hours = 0
        for y in range(4):
            year = 2026 + y
            for m in range(1, 13):
                expected_hours += monthrange(year, m)[1] * HRS_IN_DAY
        self.assertAlmostEqual(obj.hour[-1], expected_hours, delta=1.0)

    def test_leap_year_total_hours_differ_from_non_leap(self):
        """A simulation spanning a leap year should have 24 more hours than
        the same span without a leap year."""
        loads = [500.0] * HOURS_IN_YEAR

        # 2 years starting 2026: 2026 (non-leap) + 2027 (non-leap)
        obj_no_leap = _make_full_v2(loads, start_month=1, end_month=24)

        # 3 years: 2026 + 2027 + 2028 (leap)
        obj_with_leap = _make_full_v2(loads, start_month=1, end_month=36)

        # Years 1-2 (2026-2027): no leap years, ~17520 hours
        hours_first_two = obj_no_leap.hour[-1]
        # Year 3 (2028): leap year, should add 8784 hours
        hours_three = obj_with_leap.hour[-1]
        year_3_hours = hours_three - hours_first_two
        self.assertEqual(year_3_hours, 366 * HRS_IN_DAY)

    def test_step_func_load_values_multi_year(self):
        """step_func_load[i] should equal load[i] - load[i-1] for multi-year."""
        loads = [1000.0 * np.sin(2 * np.pi * h / HOURS_IN_YEAR) for h in range(HOURS_IN_YEAR)]
        obj = _make_full_v2(loads, start_month=1, end_month=60)
        self.assertAlmostEqual(obj.step_func_load[0], 0.0)
        for i in range(1, len(obj.step_func_load)):
            expected = obj.load[i] - obj.load[i - 1]
            self.assertAlmostEqual(
                obj.step_func_load[i], expected, places=10, msg=f"step_func_load mismatch at index {i}"
            )

    def test_multiyear_loading_example_csv(self):
        """Multi-year simulation using first year of Multiyear_Loading_Example.csv.

        The CSV contains 35064 hourly values spanning 4 years (2026-2029),
        including leap year 2028. This test uses the first year (8760 hours)
        as the repeating load profile and runs a 4-year (48-month) simulation.
        """

        csv_path = Path(__file__).parent / "test_data" / "Multiyear_Loading_Example.csv"
        if not csv_path.exists():
            self.skipTest("Multiyear_Loading_Example.csv not found")

        raw_lines = csv_path.read_text().split("\n")
        all_loads = [float(x) for x in raw_lines[1:] if x.strip() != ""]
        self.assertEqual(len(all_loads), 35064, "Expected 35064 hourly values (4 years)")

        # Use first year as the repeating load profile
        first_year_loads = all_loads[:HOURS_IN_YEAR]

        obj = _make_full_v2(first_year_loads, start_month=1, end_month=48)

        # Years list should cover 4 years including leap year 2028
        self.assertEqual(len(obj.years), 4)
        self.assertIn(2028, obj.years)

        # Arrays should be well-formed
        self.assertEqual(len(obj.load), len(obj.hour))
        self.assertEqual(len(obj.step_func_load), len(obj.hour))

        # Hour array should be strictly monotonic
        for i in range(2, len(obj.hour)):
            self.assertGreater(obj.hour[i], obj.hour[i - 1], f"Hour array not monotonic at index {i}")

        # Final hour should account for leap year 2028
        # 2026: 8760, 2027: 8760, 2028: 8784, 2029: 8760 = 35064 total
        expected_hours = 0
        for year in obj.years:
            for m in range(1, 13):
                expected_hours += monthrange(year, m)[1] * HRS_IN_DAY
        self.assertAlmostEqual(obj.hour[-1], expected_hours, delta=1.0)

        # Peak months should be identified
        self.assertEqual(len(obj.peak_cooling_months), HybridLoadV2.NUM_PEAK_MONTHS)
        self.assertEqual(len(obj.peak_heating_months), HybridLoadV2.NUM_PEAK_MONTHS)

        # All peak durations should be positive
        for m in obj.peak_cooling_months:
            self.assertGreater(obj.monthly_peak_cl_duration[m], 0.0)
        for m in obj.peak_heating_months:
            self.assertGreater(obj.monthly_peak_hl_duration[m], 0.0)


if __name__ == "__main__":
    unittest.main()
