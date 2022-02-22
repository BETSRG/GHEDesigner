# Jack C. Cook
# December, Wednesday 29, 2021

# Purpose: Show how to select and size a ground heat exchanging utilizing the
# current ORNL parameters json file that OpenStudio creates

import ghedt as dt
import ghedt.peak_load_analysis_tool as plat
import pygfunction as gt
import pandas as pd
from time import time as clock


def main():
    # Provide path to ORNL parameters file
    file_path = 'ornl_params.json'
    ornl_param_file_data = dt.utilities.js_load(file_path)

    # Borehole dimensions
    # -------------------
    H = ornl_param_file_data['max_height']  # Borehole length (m)
    D = 2.  # Borehole buried depth (m)
    r_b = ornl_param_file_data['bore_hole_radius']  # Borehole radius (m)
    B = ornl_param_file_data['B_spacing']  # Borehole spacing (m)

    # Pipe dimensions
    # ---------------
    # Single and Multiple U-tubes
    # Pipe outer radius (m)
    r_out = ornl_param_file_data['pipe_out_diameter'] / 2.
    t = ornl_param_file_data['pipe_thickness']  # Pipe thickness (m)
    r_in = r_out - t  # Pipe inner radius (m)
    # Inner-tube to inner-tube Shank spacing (m)
    s = ornl_param_file_data['u-tube_distance']
    epsilon = 1.0e-6  # Pipe roughness (m)

    # Pipe positions
    # --------------
    # Single U-tube [(x_in, y_in), (x_out, y_out)]
    pos_single = plat.media.Pipe.place_pipes(s, r_out, 1)
    # Single U-tube BHE object
    single_u_tube = plat.borehole_heat_exchangers.SingleUTube

    # Thermal conductivities
    # ----------------------
    # Pipe thermal conductivity (W/m.K)
    k_p = ornl_param_file_data['pipe_thermal_conductivity']
    # Ground thermal conductivity (W/m.K)
    k_s = ornl_param_file_data['ground_thermal_conductivity']
    # Grout thermal conductivity (W/m.K)
    k_g = ornl_param_file_data['grout_thermal_conductivity']

    # Volumetric heat capacities
    # --------------------------
    rhoCp_p = 1542. * 1000.  # Pipe volumetric heat capacity (J/K.m3)
    # Soil volumetric heat capacity (J/K.m3)
    rhoCp_s = ornl_param_file_data['ground_thermal_heat_capacity']
    rhoCp_g = 3901. * 1000.  # Grout volumetric heat capacity (J/K.m3)

    # Thermal properties
    # ------------------
    # Pipe
    pipe_single = \
        plat.media.Pipe(pos_single, r_in, r_out, s, epsilon, k_p, rhoCp_p)
    # Soil
    # Undisturbed ground temperature (degrees Celsius)
    ugt = ornl_param_file_data['ground_temperature']
    soil = plat.media.Soil(k_s, rhoCp_s, ugt)
    # Grout
    grout = plat.media.Grout(k_g, rhoCp_g)

    # Inputs related to fluid
    # -----------------------
    # Fluid properties
    mixer = 'MEG'  # Ethylene glycol mixed with water
    percent = 0.  # Percentage of ethylene glycol added in
    fluid = gt.media.Fluid(mixer=mixer, percent=percent)

    # Fluid properties
    # Volumetric flow rate (L/s)
    V_flow = ornl_param_file_data['design_flow_rate'] * 1000.
    # Note: The flow parameter can be borehole or system.
    flow = 'system'

    # Define a borehole
    borehole = gt.boreholes.Borehole(H, D, r_b, x=0., y=0.)

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
    max_Height = 135.  # in meters
    min_Height = 60  # in meters
    sim_params = plat.media.SimulationParameters(
        start_month, end_month, max_EFT_allowable, min_EFT_allowable,
        max_Height, min_Height)

    # Process loads from file
    # -----------------------
    # read in the csv file and convert the loads to a list of length 8760
    hourly_extraction: dict = \
        pd.read_csv('../../Atlanta_Office_Building_Loads.csv').to_dict('list')
    # Take only the first column in the dictionary
    hourly_extraction_ground_loads: list = \
        hourly_extraction[list(hourly_extraction.keys())[0]]

    # Geometric constraints for the `near-square` routine
    geometric_constraints = dt.media.GeometricConstraints(B=B)

    # Single U-tube
    # -------------
    design_single_u_tube = dt.design.Design(
        V_flow, borehole, single_u_tube, fluid, pipe_single, grout,
        soil, sim_params, geometric_constraints, hourly_extraction_ground_loads,
        routine='near-square', flow=flow)

    # Find the near-square design for a single U-tube and size it.
    bisection_search = design_single_u_tube.find_design()
    bisection_search.ghe.compute_g_functions()
    bisection_search.ghe.size(method='hybrid')

    # Export the g-function to a file named `ghedt_output`. A json file will be
    # created.
    dt.search_routines.oak_ridge_export(
        bisection_search, file_name='ghedt_output')


if __name__ == '__main__':
    main()
