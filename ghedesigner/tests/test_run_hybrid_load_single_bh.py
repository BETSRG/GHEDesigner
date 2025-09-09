from pathlib import Path

from ghedesigner.enums import FlowConfigType, TimestepType
from ghedesigner.ghe.boreholes.core import Borehole
from ghedesigner.ghe.design.near_square import DesignNearSquare, GeometricConstraintsNearSquare
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.media import Fluid, Grout, Soil
from ghedesigner.utilities import length_of_side


def get_test_loads():
    """Load test building loads with error handling"""

    test_data_directory = Path(__file__).parent / "test_data"
    print(f"Looking for data in: {test_data_directory}")
    print(f"Directory exists: {test_data_directory.exists()}")

    bldg_load_data = test_data_directory / "bldg_loads_1b_amadfard.csv"
    print(f"CSV file exists: {bldg_load_data.exists()}")

    if bldg_load_data.exists():
        raw_lines = bldg_load_data.read_text().split("\n")
        loads = [float(x) for x in raw_lines if x.strip() != ""]
        print(f"Loaded {len(loads)} data points from CSV")
        return loads
    else:
        raise FileNotFoundError("CSV file not found")


def test_main():
    pipe = Pipe.init_single_u_tube(
        inner_diameter=0.0274,
        outer_diameter=0.0334,
        shank_spacing=0.075,
        roughness=1.0e-6,
        conductivity=0.43,
        rho_cp=1540000.0,
    )

    soil = Soil(k=1.8, rho_cp=2073600, ugt=17.5)
    grout = Grout(k=1.4, rho_cp=3900000.0)
    fluid = Fluid(fluid_name="water", percent=0.0, temperature=20.0)
    borehole = Borehole(burial_depth=4.0, borehole_radius=0.075, borehole_height=100)
    building_loads = get_test_loads()
    b = 5.0  # distance in between bore holes
    number_of_boreholes = 1
    length = length_of_side(number_of_boreholes, b)

    geometry = GeometricConstraintsNearSquare(b=b, length=length)
    design = DesignNearSquare(
        v_flow=0.4427,  # L/s nominal flow rate
        borehole=borehole,
        fluid=fluid,
        pipe=pipe,
        grout=grout,
        soil=soil,
        start_month=1,
        end_month=12,
        max_eft=35,
        min_eft=5,
        max_height=135,
        min_height=50,
        continue_if_design_unmet=True,
        max_boreholes=None,
        geometric_constraints=geometry,
        hourly_extraction_ground_loads=building_loads,
        method=TimestepType.HYBRID,
        flow_type=FlowConfigType.BOREHOLE,
    )
    search = design.find_design()
    search.ghe.compute_g_functions(60, 135)
    search.ghe.size(method=TimestepType.HYBRID, min_height=60, max_height=135, design_min_eft=5, design_max_eft=35)


if __name__ == "__main__":
    test_main()
