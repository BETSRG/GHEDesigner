from ghedt import utilities, geometry, design, search_routines
from ghedt.peak_load_analysis_tool import media, borehole_heat_exchangers
from pathlib import Path
import pygfunction as gt


def main():
    # Provide path to ORNL parameters file
    file_path = "ornl_params.json"
    ornl_param_file_data = utilities.js_load(file_path)

    # Borehole dimensions
    # -------------------
    H = ornl_param_file_data["max_height"]  # Borehole length (m)
    D = 2.0  # Borehole buried depth (m)
    r_b = ornl_param_file_data["bore_hole_radius"]  # Borehole radius (m)
    B = ornl_param_file_data["B_spacing"]  # Borehole spacing (m)

    # Pipe dimensions
    # ---------------
    # Single and Multiple U-tubes
    # Pipe outer radius (m)
    r_out = ornl_param_file_data["pipe_out_diameter"] / 2.0
    t = ornl_param_file_data["pipe_thickness"]  # Pipe thickness (m)
    r_in = r_out - t  # Pipe inner radius (m)
    # Inner-tube to inner-tube Shank spacing (m)
    s = ornl_param_file_data["u-tube_distance"]
    epsilon = 1.0e-6  # Pipe roughness (m)

    # Pipe positions
    # --------------
    # Single U-tube [(x_in, y_in), (x_out, y_out)]
    pos_single = media.Pipe.place_pipes(s, r_out, 1)
    # Single U-tube BHE object
    single_u_tube = borehole_heat_exchangers.SingleUTube

    # Thermal conductivities
    # ----------------------
    # Pipe thermal conductivity (W/m.K)
    k_p = ornl_param_file_data["pipe_thermal_conductivity"]
    # Ground thermal conductivity (W/m.K)
    k_s = ornl_param_file_data["ground_thermal_conductivity"]
    # Grout thermal conductivity (W/m.K)
    k_g = ornl_param_file_data["grout_thermal_conductivity"]

    # Volumetric heat capacities
    # --------------------------
    rhoCp_p = 1542.0 * 1000.0  # Pipe volumetric heat capacity (J/K.m3)
    # Soil volumetric heat capacity (J/K.m3)
    rhoCp_s = ornl_param_file_data["ground_thermal_heat_capacity"]
    rhoCp_g = 3901.0 * 1000.0  # Grout volumetric heat capacity (J/K.m3)

    # Thermal properties
    # ------------------
    # Pipe
    pipe_single = media.Pipe(pos_single, r_in, r_out, s, epsilon, k_p, rhoCp_p)
    # Soil
    # Undisturbed ground temperature (degrees Celsius)
    ugt = ornl_param_file_data["ground_temperature"]
    soil = media.Soil(k_s, rhoCp_s, ugt)
    # Grout
    grout = media.Grout(k_g, rhoCp_g)

    # Inputs related to fluid
    # -----------------------
    # Fluid properties
    fluid = gt.media.Fluid(fluid_str="Water", percent=0.0)

    # Fluid properties
    # Volumetric flow rate (L/s)
    V_flow = ornl_param_file_data["design_flow_rate"] * 1000.0
    # Note: The flow parameter can be borehole or system.
    flow = "system"

    # Define a borehole
    borehole = gt.boreholes.Borehole(H, D, r_b, x=0.0, y=0.0)

    # Simulation parameters
    # ---------------------
    # Simulation start month and end month
    start_month = 1
    n_years = 20
    end_month = n_years * 12
    # Maximum and minimum allowable fluid temperatures
    max_EFT_allowable = 35  # degrees Celsius
    min_EFT_allowable = 5  # degrees Celsius
    # Maximum and minimum allowable heights
    max_Height = 135.0  # in meters
    min_Height = 60  # in meters
    sim_params = media.SimulationParameters(
        start_month,
        end_month,
        max_EFT_allowable,
        min_EFT_allowable,
        max_Height,
        min_Height,
    )

    # Process loads from file
    # -----------------------
    # read in the csv file and convert the loads to a list of length 8760
    project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
    csv_file = project_root / 'examples' / 'data' / 'Atlanta_Office_Building_Loads.csv'
    raw_lines = csv_file.read_text().split('\n')
    hourly_extraction_ground_loads = [float(x) for x in raw_lines[1:] if x.strip() != '']

    # Geometric constraints for the `near-square` routine
    geometric_constraints = geometry.GeometricConstraints(b=B)

    # Single U-tube
    # -------------
    design_single_u_tube = design.DesignNearSquare(
        V_flow,
        borehole,
        single_u_tube,
        fluid,
        pipe_single,
        grout,
        soil,
        sim_params,
        geometric_constraints,
        hourly_extraction_ground_loads,
        flow=flow,
    )

    # Find the near-square design for a single U-tube and size it.
    bisection_search = design_single_u_tube.find_design()
    bisection_search.ghe.compute_g_functions()
    bisection_search.ghe.size(method="hybrid")

    # Export the g-function to a file named `ghedt_output`. A json file will be
    # created.
    search_routines.oak_ridge_export(bisection_search, file_name="ghedt_output")


if __name__ == "__main__":
    main()
