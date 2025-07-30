from pathlib import Path

from ghedesigner.enums import FlowConfigType, TimestepType
from ghedesigner.ghe.boreholes.core import Borehole
from ghedesigner.ghe.design.near_square import DesignNearSquare, GeometricConstraintsNearSquare
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.media import GHEFluid, Grout, Soil
from ghedesigner.utilities import length_of_side


def get_atlanta_loads():
    test_data_directory = Path(__file__).parent / "test_data"
    print(test_data_directory.exists())

    glhe_json_data = test_data_directory / "Atlanta_Office_Building_Loads.csv"
    raw_lines = glhe_json_data.read_text().split("\n")
    return [float(x) for x in raw_lines[1:] if x.strip() != ""]


def main():
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
    ground_loads = get_atlanta_loads()
    b = 5.0
    number_of_boreholes = 1
    length = length_of_side(number_of_boreholes, b)

    geometry = GeometricConstraintsNearSquare(b=b, length=length)
    design = DesignNearSquare(
        v_flow=0.5,
        borehole=borehole,
        fluid=fluid,
        pipe=pipe,
        grout=grout,
        soil=soil,
        start_month=1,
        end_month=240,
        max_eft=35,
        min_eft=5,
        max_height=135,
        min_height=60,
        continue_if_design_unmet=True,
        max_boreholes=None,
        geometric_constraints=geometry,
        hourly_extraction_ground_loads=ground_loads,
        method=TimestepType.HYBRID,
        flow_type=FlowConfigType.BOREHOLE,
    )
    search = design.find_design()
    search.ghe.compute_g_functions(60, 135)
    search.ghe.size(method=TimestepType.HYBRID, min_height=60, max_height=135, design_min_eft=5, design_max_eft=35)


if __name__ == "__main__":
    main()
