"""
Focused test for perform_hrly_ExFT_simulation method
This test specifically targets the ExFT calculation to identify why values are too high
"""

from unittest.mock import Mock

import numpy as np

# Add the path to your hybrid_loads module
# sys.path.append(str(Path(__file__).parent))


def create_test_mock_objects():
    """Create properly configured mock objects for testing"""

    # Mock soil with realistic properties
    mock_soil = Mock()
    mock_soil.k = 2.0  # W/m-K - typical soil thermal conductivity

    # Mock borehole heat exchanger
    mock_bhe = Mock()
    mock_bhe.soil = mock_soil
    mock_bhe.calc_effective_borehole_resistance.return_value = 0.13  # K/(W/m)

    # Mock radial numerical with g-function
    mock_radial = Mock()
    mock_radial.t_s = 3600.0  # Reference time scale in seconds

    def realistic_g_function(log_time_values):
        """
        Realistic g-function that mimics actual ground heat exchanger behavior
        Typical g-function values range from about 2 to 15
        """
        log_times = np.atleast_1d(np.array(log_time_values))

        # More realistic g-function: slower growth, reasonable bounds
        # This approximates actual infinite line source solutions
        result = 2.5 + 1.2 * np.log(np.maximum(np.abs(log_times), -10.0))
        result = np.clip(result, 1.0, 12.0)  # Reasonable g-function bounds

        if np.isscalar(log_time_values):
            return float(result[0])
        return result

    mock_radial.g_sts = realistic_g_function

    return mock_bhe, mock_radial


def test_perform_hrly_exft_simulation():
    """Main test function for the ExFT simulation method"""

    print("=== Testing perform_hrly_ExFT_simulation Method ===")

    # Import the HybridLoads2 class
    try:
        from ghedesigner.ghe.hybrid_loads import HybridLoads2

        print("✓ Successfully imported HybridLoads2")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return

    # Create mock objects
    mock_bhe, mock_radial = create_test_mock_objects()

    # Test with simple building loads
    # Start with a very simple case: constant load
    simple_loads = [1000.0] * 24  # 1 kW for 24 hours

    print("\nTest 1: Simple constant load")
    print(f"Building loads: {len(simple_loads)} hours at {simple_loads[0]} W")

    try:
        # Create HybridLoads2 instance
        hybrid = HybridLoads2(
            building_loads=simple_loads, bhe=mock_bhe, radial_numerical=mock_radial, years=[2025], cop_h=4.0, cop_c=5.0
        )
        print("✓ HybridLoads2 instance created successfully")

        # Test the ground load conversion first
        ground_loads = hybrid.bldg_to_ground_load(simple_loads)
        print(f"Ground loads sample: {ground_loads[:5]}")
        print(f"Ground loads range: {min(ground_loads):.1f} to {max(ground_loads):.1f} W")

        # Test normalization
        normalized_loads = hybrid.normalize_loads(ground_loads)
        print(f"Normalized loads sample: {normalized_loads[:5]}")
        print(f"Normalized loads range: {min(normalized_loads):.1f} to {max(normalized_loads):.1f} W")

        # Now test the ExFT simulation
        print("\nTesting ExFT simulation...")
        exft_results = hybrid.perform_hrly_ExFT_simulation(normalized_loads)

        print(f"ExFT results length: {len(exft_results)}")
        print(f"ExFT results sample (first 10): {exft_results[:10]}")
        print(f"ExFT results range: {min(exft_results):.2f} to {max(exft_results):.2f} °C")

        # Check if values are reasonable
        max_exft_change = max(abs(x) for x in exft_results)
        print(f"Maximum ExFT change from ground temperature: {max_exft_change:.2f} °C")

        if max_exft_change > 20.0:
            print("⚠ WARNING: ExFT changes seem unusually high!")
            debug_calculation_steps(hybrid, normalized_loads[:5])
        else:
            print("✓ ExFT values appear reasonable")

    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback

        traceback.print_exc()


def debug_calculation_steps(hybrid, load_profile):
    """Debug the calculation steps to identify where high values come from"""

    print("\n=== DEBUGGING CALCULATION STEPS ===")

    # Extract the key parameters used in the calculation
    ts = hybrid.borehole.t_s
    two_pi_k = 12.566 * hybrid.bhe.soil.k  # TWO_PI * k
    resist_bh_effective = 0.13  # Fixed value from your code
    g = hybrid.borehole.g_sts

    print(f"Ground temperature scale (ts): {ts}")
    print(f"Soil thermal conductivity: {hybrid.bhe.soil.k} W/m-K")
    print(f"Two_pi_k: {two_pi_k:.3f}")
    print(f"Borehole resistance: {resist_bh_effective} K/(W/m)")
    print(f"Load profile (first 5): {load_profile}")

    # Test the g-function with typical values
    test_times = np.array([1.0, 2.0, 4.0, 8.0, 24.0])  # Hours
    log_times = np.log(test_times * 3600 / ts)  # Convert to log(t/ts)
    g_values = g(log_times)
    print("G-function test:")
    print(f"  Times (hours): {test_times}")
    print(f"  Log(t/ts): {log_times}")
    print(f"  G-values: {g_values}")

    # Manual calculation for first few time steps
    print("\nManual calculation verification:")

    q_arr = np.array(load_profile)
    print(f"Load array: {q_arr}")

    if len(q_arr) > 1:
        q_dt = q_arr[1:] - q_arr[:-1]
        print(f"Load differences (q_dt): {q_dt}")

        # Check calculation for hour 1
        hour = 1
        _time = np.array([hour])  # Time difference from start
        log_time_val = np.log(_time * 3600 / ts)
        g_val = g(log_time_val)

        print(f"\nHour {hour} calculation:")
        print(f"  Time difference: {_time[0]} hours")
        print(f"  Log time value: {log_time_val[0]:.3f}")
        print(f"  G-function value: {g_val[0]:.3f}")
        print(f"  Load difference: {q_dt[0]:.1f} W")
        print(f"  Delta Tb component: {(q_dt[0] / two_pi_k) * g_val[0]:.3f} °C")
        print(f"  Resistance component: {q_arr[hour] * resist_bh_effective:.3f} °C")
        print(f"  Total ExFT: {(q_dt[0] / two_pi_k) * g_val[0] + q_arr[hour] * resist_bh_effective:.3f} °C")

    # Identify potential issues
    print("\n=== POTENTIAL ISSUES ANALYSIS ===")

    # Check if the time scale is appropriate
    if ts > 10000:
        print(f"⚠ Time scale (ts={ts}) seems very high. Typical values are 3600-7200.")

    # Check if soil conductivity is reasonable
    if hybrid.bhe.soil.k < 0.5 or hybrid.bhe.soil.k > 5.0:
        print(f"⚠ Soil conductivity ({hybrid.bhe.soil.k}) is outside typical range (0.5-5.0 W/m-K)")

    # Check g-function values
    typical_g_val = g(np.log(24 * 3600 / ts))  # 24-hour g-value
    if isinstance(typical_g_val, np.ndarray):
        typical_g_val = typical_g_val[0]

    if typical_g_val > 20:
        print(f"⚠ G-function value ({typical_g_val:.2f}) seems very high for 24 hours")
    elif typical_g_val < 1:
        print(f"⚠ G-function value ({typical_g_val:.2f}) seems very low for 24 hours")

    # Check load magnitudes after conversion
    max_load = max(abs(x) for x in load_profile)
    if max_load > 10000:
        print(f"⚠ Normalized loads ({max_load:.1f} W) might be too high")


def test_with_different_load_patterns():
    """Test with different load patterns to isolate the issue"""

    print("\n=== TESTING DIFFERENT LOAD PATTERNS ===")

    mock_bhe, mock_radial = create_test_mock_objects()

    test_cases = [
        ("Zero loads", [0.0] * 10),
        ("Small constant load", [100.0] * 10),
        ("Medium constant load", [1000.0] * 10),
        ("Large constant load", [5000.0] * 10),
        ("Step change", [0.0] * 5 + [1000.0] * 5),
        ("Ramp", list(range(0, 1000, 100))),
    ]

    try:
        from ghedesigner.ghe.hybrid_loads import HybridLoads2

        for test_name, building_loads in test_cases:
            print(f"\n--- {test_name} ---")

            # Extend to minimum required length if needed
            if len(building_loads) < 24:
                building_loads = building_loads * (24 // len(building_loads) + 1)
                building_loads = building_loads[:24]

            try:
                hybrid = HybridLoads2(
                    building_loads=building_loads,
                    bhe=mock_bhe,
                    radial_numerical=mock_radial,
                    years=[2025],
                    cop_h=4.0,
                    cop_c=5.0,
                )

                # Convert to ground loads and normalize
                ground_loads = hybrid.bldg_to_ground_load(building_loads)
                normalized_loads = hybrid.normalize_loads(ground_loads)

                # Run ExFT simulation
                exft_results = hybrid.perform_hrly_ExFT_simulation(normalized_loads)

                max_exft = max(abs(x) for x in exft_results)
                print(f"  Building loads range: {min(building_loads):.1f} to {max(building_loads):.1f} W")
                print(f"  Ground loads range: {min(ground_loads):.1f} to {max(ground_loads):.1f} W")
                print(f"  Normalized range: {min(normalized_loads):.1f} to {max(normalized_loads):.1f} W")
                print(f"  Max ExFT change: {max_exft:.2f} °C")

                if max_exft > 15.0:
                    print("  ⚠ HIGH ExFT detected! Investigating...")
                    # Print more details for problematic cases
                    print(f"    First 5 ExFT values: {exft_results[:5]}")
                    print(f"    Last 5 ExFT values: {exft_results[-5:]}")

            except Exception as e:
                print(f"  ✗ Error with {test_name}: {e}")

    except ImportError as e:
        print(f"✗ Could not import HybridLoads2: {e}")


def test_unit_conversion_check():
    """Check if there might be unit conversion issues"""

    print("\n=== UNIT CONVERSION CHECK ===")

    mock_bhe, mock_radial = create_test_mock_objects()

    # Test with known values and check units throughout the calculation
    building_load_watts = 2000.0  # 2 kW heating load

    print(f"Original building load: {building_load_watts} W")

    try:
        from ghedesigner.ghe.hybrid_loads import HybridLoads2

        # Create minimal instance just for unit testing
        building_loads = [building_load_watts] * 48  # 48 hours

        hybrid = HybridLoads2(
            building_loads=building_loads,
            bhe=mock_bhe,
            radial_numerical=mock_radial,
            years=[2025],
            cop_h=4.0,
            cop_c=5.0,
        )

        # Step 1: Building to ground load conversion
        ground_load = hybrid.bldg_to_ground_load([building_load_watts])[0]
        print(f"Ground load after COP conversion: {ground_load:.1f} W")

        # For heating: ground_load = (cop_h - 1) / cop_h * building_load
        expected_ground = (4.0 - 1) / 4.0 * building_load_watts
        print(f"Expected ground load: {expected_ground:.1f} W")

        # Step 2: Normalization
        ground_loads_list = [ground_load] * 48
        max_ground = max(ground_loads_list)
        normalized_single = 40 * 100 / max_ground * ground_load  # From normalize_loads method
        print(f"Normalized load: {normalized_single:.1f} W")
        print(f"Normalization factor: {40 * 100 / max_ground:.6f}")

        # This normalization seems to normalize to 4000W peak, which might be the issue!
        print("⚠ POTENTIAL ISSUE: Normalization scales to 4000W peak (40 * 100)")

        # Step 3: Check ExFT calculation parameters
        ts = mock_radial.t_s
        two_pi_k = 2 * np.pi * mock_bhe.soil.k
        resist_bh = 0.13

        print("\nCalculation parameters:")
        print(f"  ts (time scale): {ts} seconds")
        print(f"  two_pi_k: {two_pi_k:.3f} W/m-K")
        print(f"  Borehole resistance: {resist_bh} K/(W/m)")

        # The resistance component alone for normalized load:
        resistance_component = normalized_single * resist_bh
        print(
            f"  Resistance component: {normalized_single:.1f} W × {resist_bh} K/(W/m) = {resistance_component:.2f} °C"
        )

        if resistance_component > 10.0:
            print(f"⚠ WARNING: Resistance component alone gives {resistance_component:.2f}°C!")
            print("           This suggests the normalization is creating unreasonably high loads")

        # Test full ExFT simulation
        normalized_loads = [normalized_single] * 48
        exft_result = hybrid.perform_hrly_ExFT_simulation(normalized_loads)

        print("\nExFT simulation results:")
        print(f"  Result length: {len(exft_result)}")
        print(f"  First 5 values: {exft_result[:5]}")
        print(f"  Max ExFT: {max(exft_result):.2f} °C")
        print(f"  Min ExFT: {min(exft_result):.2f} °C")

        # Diagnosis
        max_exft = max(abs(x) for x in exft_result)
        if max_exft > 20.0:
            print("\n🔍 DIAGNOSIS:")
            print(f"   ExFT values of {max_exft:.2f}°C are too high!")
            print("   Likely causes:")
            print("   1. Normalization to 4000W peak is too aggressive")
            print("   2. Units might be inconsistent (W vs kW)")
            print("   3. G-function or time scaling issues")
            print("   4. Borehole resistance value too high")

    except Exception as e:
        print(f"✗ Error in unit conversion test: {e}")
        import traceback

        traceback.print_exc()


def test_normalization_issue():
    """Specifically test if the normalization is causing the problem"""

    print("\n=== TESTING NORMALIZATION ISSUE ===")

    mock_bhe, mock_radial = create_test_mock_objects()

    try:
        from ghedesigner.ghe.hybrid_loads import HybridLoads2

        # Test with different peak loads to see normalization effect
        test_cases = [
            ("Small peak", [100.0, 50.0, 200.0, 75.0] * 12),  # 200W peak
            ("Medium peak", [1000.0, 500.0, 2000.0, 750.0] * 12),  # 2kW peak
            ("Large peak", [5000.0, 2500.0, 10000.0, 3750.0] * 12),  # 10kW peak
        ]

        for test_name, building_loads in test_cases:
            print(f"\n--- {test_name} ---")

            hybrid = HybridLoads2(
                building_loads=building_loads,
                bhe=mock_bhe,
                radial_numerical=mock_radial,
                years=[2025],
                cop_h=4.0,
                cop_c=5.0,
            )

            # Check ground loads
            ground_loads = hybrid.bldg_to_ground_load(building_loads)
            max_ground = max(ground_loads)
            print(f"  Max ground load: {max_ground:.1f} W")

            # Check normalization
            normalized = hybrid.normalize_loads(ground_loads)
            max_normalized = max(normalized)
            normalization_factor = max_normalized / max_ground if max_ground > 0 else 0

            print(f"  Max normalized load: {max_normalized:.1f} W")
            print(f"  Normalization factor: {normalization_factor:.6f}")
            print("  Expected normalized peak: 4000 W")

            # The issue might be here - check if normalization is correct
            expected_norm_factor = 4000.0 / max_ground if max_ground > 0 else 0
            print(f"  Expected norm factor: {expected_norm_factor:.6f}")

            # Run ExFT simulation with smaller normalized loads to test
            test_normalized = [x * 0.1 for x in normalized]  # Scale down by 10x
            exft_small = hybrid.perform_hrly_ExFT_simulation(test_normalized)

            exft_full = hybrid.perform_hrly_ExFT_simulation(normalized)

            print(f"  ExFT with full normalized loads: {max(abs(x) for x in exft_full):.2f} °C")
            print(f"  ExFT with 10% normalized loads: {max(abs(x) for x in exft_small):.2f} °C")

            # The scaling should be roughly linear
            scaling_ratio = (
                max(abs(x) for x in exft_full) / max(abs(x) for x in exft_small)
                if max(abs(x) for x in exft_small) > 0
                else 0
            )
            print(f"  Scaling ratio: {scaling_ratio:.1f} (should be ~10)")

    except Exception as e:
        print(f"✗ Error in normalization test: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Run all tests"""
    print("Starting ExFT Simulation Debug Tests")
    print("=" * 60)

    test_perform_hrly_exft_simulation()
    test_unit_conversion_check()
    test_normalization_issue()

    print("\n" + "=" * 60)
    print("SUMMARY OF POTENTIAL ISSUES:")
    print("1. Check if normalization to 4000W is appropriate for your system size")
    print("2. Verify that the g-function values are realistic for your borehole geometry")
    print("3. Confirm that the borehole resistance value (0.13 K/(W/m)) is correct")
    print("4. Check that time scaling (ts) is appropriate")
    print("5. Verify unit consistency throughout the calculation chain")


if __name__ == "__main__":
    main()
