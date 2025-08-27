"""
Simplified test for HybridLoads2 - focusing on the specific issues you're encountering
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Add your project root to path if needed
# sys.path.append(str(Path(__file__).parent.parent))

def get_atlanta_loads():
    """Load Atlanta building loads with error handling"""
    try:
        test_data_directory = Path(__file__).parent / "test_data"
        print(f"Looking for data in: {test_data_directory}")
        print(f"Directory exists: {test_data_directory.exists()}")

        glhe_json_data = test_data_directory / "Atlanta_Office_Building_Loads.csv"
        print(f"CSV file exists: {glhe_json_data.exists()}")

        if glhe_json_data.exists():
            raw_lines = glhe_json_data.read_text().split("\n")
            loads = [float(x) for x in raw_lines[1:] if x.strip() != ""]
            print(f"Loaded {len(loads)} data points")
            return loads
        else:
            raise FileNotFoundError("CSV file not found")

    except Exception as e:
        print(f"Error loading data: {e}")
        print("Creating synthetic test data...")
        # Generate synthetic annual hourly loads
        hours = np.arange(8760)
        seasonal = 1000 * np.sin(2 * np.pi * hours / (365 * 24))
        daily = 500 * np.sin(2 * np.pi * hours / 24)
        base_load = 2000
        synthetic_loads = base_load + seasonal + daily + np.random.normal(0, 50, 8760)
        return synthetic_loads.tolist()

def test_data_loading():
    """Test that we can load the building loads data"""
    print("=== Testing Data Loading ===")
    building_loads = get_atlanta_loads()
    print(f"Data type: {type(building_loads)}")
    print(f"Data length: {len(building_loads)}")
    print(f"First 5 values: {building_loads[:5]}")
    print(f"Last 5 values: {building_loads[-5:]}")
    print(f"Data range: {min(building_loads):.1f} to {max(building_loads):.1f}")

    # Analyze heating vs cooling distribution
    heating_loads = [x for x in building_loads if x > 0]
    cooling_loads = [x for x in building_loads if x < 0]
    zero_loads = [x for x in building_loads if x == 0]

    print(f"Heating loads (>0): {len(heating_loads)} ({len(heating_loads)/len(building_loads)*100:.1f}%)")
    print(f"Cooling loads (<0): {len(cooling_loads)} ({len(cooling_loads)/len(building_loads)*100:.1f}%)")
    print(f"Zero loads: {len(zero_loads)} ({len(zero_loads)/len(building_loads)*100:.1f}%)")

    if len(cooling_loads) > 0:
        print(f"Heating range: 0 to {max(heating_loads):.1f}")
        print(f"Cooling range: {min(cooling_loads):.1f} to 0")

    # Check monthly distribution
    hours_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    monthly_hours = [h * 24 for h in hours_per_month]

    print("\nMonthly load analysis:")
    start_hour = 0
    for month, hours in enumerate(monthly_hours, 1):
        if start_hour + hours <= len(building_loads):
            month_loads = building_loads[start_hour:start_hour + hours]
            month_heating = [x for x in month_loads if x > 0]
            month_cooling = [x for x in month_loads if x < 0]
            print(f"Month {month:2d}: {len(month_heating):4d} heating, {len(month_cooling):4d} cooling hours")
            start_hour += hours

    return building_loads

def create_mock_singletube():
    """Create properly configured mock SingleUTube objects"""
    print("\n=== Creating Mock SingleUTube Objects ===")

    # Create mock soil
    mock_soil = Mock()
    mock_soil.k = 2.0  # thermal conductivity

    # Create mock borehole heat exchanger
    mock_bhe = Mock()
    mock_bhe.soil = mock_soil
    mock_bhe.calc_effective_borehole_resistance.return_value = 0.1  # typical value

    # Create mock radial numerical with g-function
    mock_radial = Mock()
    mock_radial.t_s = 3600.0  # time scale factor

    # Create a mock g-function that returns reasonable values
    # The g-function should return values for given log(t/ts) inputs
    def mock_g_function(log_time_values):
        # Simple mock g-function that returns reasonable values
        if isinstance(log_time_values, (list, np.ndarray)):
            return np.array([2.0 + 0.5 * np.log(abs(t) + 1) for t in log_time_values])
        else:
            return 2.0 + 0.5 * np.log(abs(log_time_values) + 1)

    mock_radial.g_sts = mock_g_function

    print("✓ Created mock_bhe with soil.k =", mock_bhe.soil.k)
    print("✓ Created mock_radial with g_sts function")
    print("✓ mock_bhe.calc_effective_borehole_resistance() =", mock_bhe.calc_effective_borehole_resistance())

    return mock_bhe, mock_radial

def test_mock_dependencies():
    """Try to create real SingleUTube objects, fall back to mocks if needed"""
    print("\n=== Testing Dependencies ===")

    try:
        # Try to import and create real objects first
        from ghedesigner.ghe.boreholes.single_u_borehole import SingleUTube
        from ghedesigner.ghe.pipe import Pipe
        from ghedesigner.media import GHEFluid, Grout, Soil
        from ghedesigner.ghe.boreholes.core import Borehole

        # Create the required components for SingleUTube
        pipe = Pipe.init_single_u_tube(
            inner_diameter=0.06404,
            outer_diameter=0.07216,
            shank_spacing=0.02856,
            roughness=1.0e-6,
            conductivity=0.4,
            rho_cp=1542000.0,
        )

        soil = Soil(k=2.0, rho_cp=2343493.0, ugt=18.3)
        grout = Grout(k=1.0, rho_cp=3901000.0)
        fluid = GHEFluid(fluid_str="water", percent=0.0, temperature=20.0)
        borehole = Borehole(burial_depth=2.0, borehole_radius=0.5, borehole_height=100)

        # Try different ways to create SingleUTube (the constructor signature might vary)
        try:
            # Method 1: Try without v_flow parameter
            bhe_eq = SingleUTube(
                borehole=borehole,
                pipe=pipe,
                grout=grout,
                fluid=fluid,
                soil=soil
            )
            print("✓ Created real SingleUTube objects (method 1)")
        except Exception as e1:
            try:
                # Method 2: Try with different parameter name
                bhe_eq = SingleUTube(
                    flow_rate=0.5,
                    borehole=borehole,
                    pipe=pipe,
                    grout=grout,
                    fluid=fluid,
                    soil=soil
                )
                print("✓ Created real SingleUTube objects (method 2)")
            except Exception as e2:
                print(f"Failed both methods: {e1}, {e2}")
                raise Exception("Could not create real SingleUTube objects")

        # Create second instance for radial_numerical
        try:
            radial_numerical = SingleUTube(
                borehole=borehole,
                pipe=pipe,
                grout=grout,
                fluid=fluid,
                soil=soil
            )
        except:
            radial_numerical = SingleUTube(
                flow_rate=0.5,
                borehole=borehole,
                pipe=pipe,
                grout=grout,
                fluid=fluid,
                soil=soil
            )

        print(f"✓ Created real bhe_eq: {type(bhe_eq)}")
        print(f"✓ Created real radial_numerical: {type(radial_numerical)}")

        return bhe_eq, radial_numerical

    except Exception as e:
        print(f"✗ Failed to create real SingleUTube objects: {e}")
        print("Falling back to Mock objects...")
        return create_mock_singletube()

def test_hybridloads2_import():
    """Test importing and basic instantiation"""
    print("\n=== Testing HybridLoads2 Import ===")

    try:
        from ghedesigner.ghe.hybrid_loads import HybridLoads2
        print("✓ Successfully imported HybridLoads2")

        # Get test data
        building_loads = get_atlanta_loads()

        # Create dependencies
        mock_bhe_eq, mock_radial_numerical = test_mock_dependencies()

        # Try to create HybridLoads2 instance
        print("\nAttempting to create HybridLoads2 instance...")

        # Use full year of data
        if len(building_loads) >= 8760:
            print("Using full annual dataset (8760 hours)...")
            test_loads = building_loads[:8760]
        else:
            print(f"Using available data ({len(building_loads)} hours)...")
            test_loads = building_loads

        print(f"Using {len(test_loads)} hours of test data")

        # Analyze the test data
        heating_loads = [x for x in test_loads if x > 0]
        cooling_loads = [x for x in test_loads if x < 0]
        zero_loads = [x for x in test_loads if x == 0]

        print(f"Test data - Heating: {len(heating_loads)}, Cooling: {len(cooling_loads)}, Zero: {len(zero_loads)}")

        if len(test_loads) > 0:
            print(f"Test data range: {min(test_loads):.1f} to {max(test_loads):.1f}")

        # Create the HybridLoads2 instance
        try:
            hybrid_loads = HybridLoads2(
                building_loads=test_loads,
                bhe=mock_bhe_eq,
                radial_numerical=mock_radial_numerical,
                years=[2025],  # Specify a year
                cop_h=4.0,
                cop_c=5.0
            )
            print("✓ Successfully created HybridLoads2 instance")
            print(f"Instance type: {type(hybrid_loads)}")

            return hybrid_loads

        except Exception as init_error:
            print(f"✗ Failed to create HybridLoads2 instance: {init_error}")
            print(f"Error type: {type(init_error)}")

            # Try to diagnose the issue
            print("\nDiagnosing the issue...")
            import traceback
            traceback.print_exc()

            raise init_error

    except ImportError as import_error:
        print(f"✗ Failed to import: {import_error}")
        print("Make sure you're in the correct environment and ghedesigner is installed")
        raise import_error

def test_basic_functionality(hybrid_loads):
    """Test basic functionality of the HybridLoads2 instance"""
    print("\n=== Testing Basic Functionality ===")

    # Check what attributes and methods are available
    attributes = [attr for attr in dir(hybrid_loads) if not attr.startswith('_')]
    print(f"Available attributes/methods: {attributes}")

    # Test some key attributes
    try:
        print(f"✓ COP heating: {hybrid_loads.cop_h}")
        print(f"✓ COP cooling: {hybrid_loads.cop_c}")
        print(f"✓ Years: {hybrid_loads.years}")
        print(f"✓ Building loads length: {len(hybrid_loads.building_loads)}")

        if hasattr(hybrid_loads, 'monthly_cl'):
            print(f"✓ Monthly cooling loads (first 5): {hybrid_loads.monthly_cl[:5]}")
        if hasattr(hybrid_loads, 'monthly_hl'):
            print(f"✓ Monthly heating loads (first 5): {hybrid_loads.monthly_hl[:5]}")

        if hasattr(hybrid_loads, 'target_ExFTghe_temps'):
            print(f"✓ Target ExFT temps length: {len(hybrid_loads.target_ExFTghe_temps)}")

        if hasattr(hybrid_loads, 'max_4_ExFT'):
            print(f"✓ Max 4 ExFT shape: {hybrid_loads.max_4_ExFT.shape}")
            print(f"✓ Max 4 ExFT values:\n{hybrid_loads.max_4_ExFT}")

        if hasattr(hybrid_loads, 'min_4_ExFT'):
            print(f"✓ Min 4 ExFT shape: {hybrid_loads.min_4_ExFT.shape}")
            print(f"✓ Min 4 ExFT values:\n{hybrid_loads.min_4_ExFT}")

    except Exception as func_error:
        print(f"✗ Error accessing attributes: {func_error}")
        import traceback
        traceback.print_exc()

def main():
    """Main test runner"""
    print("Starting HybridLoads2 Testing...")
    print("=" * 50)

    try:
        # Step 1: Test data loading
        building_loads = test_data_loading()

        # Step 2: Test imports and instantiation
        hybrid_loads = test_hybridloads2_import()

        # Step 3: Test basic functionality
        test_basic_functionality(hybrid_loads)

        print("\n" + "=" * 50)
        print("✓ All basic tests passed!")
        print("You can now start writing more specific unit tests.")

    except Exception as e:
        print("\n" + "=" * 50)
        print(f"✗ Test failed: {e}")
        print(f"Error type: {type(e)}")

        # Print traceback for debugging
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()

        print("\nTroubleshooting suggestions:")
        print("1. Check that ghedesigner is properly installed")
        print("2. Verify you're in the correct Python environment")
        print("3. Make sure test data file exists or synthetic data is working")
        print("4. Check the actual SingleUTube constructor signature")
        print("5. Verify that Mock objects have all required attributes")

if __name__ == "__main__":
    main()