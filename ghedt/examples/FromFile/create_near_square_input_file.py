# Jack C. Cook
# Friday, December 10, 2021

import ghedt as dt
import ghedt.peak_load_analysis_tool as plat
import ghedt.pygfunction as gt
import pandas as pd


def main():
    # Borehole dimensions
    # -------------------
    H = 96.  # Borehole length (m)
    D = 2.  # Borehole buried depth (m)
    r_b = 0.075  # Borehole radius (m)
    B = 5.  # Borehole spacing (m)

    # Pipe dimensions
    # ---------------
    # Single and Multiple U-tubes
    r_out = 26.67 / 1000. / 2.  # Pipe outer radius (m)
    r_in = 21.6 / 1000. / 2.  # Pipe inner radius (m)
    s = 32.3 / 1000.  # Inner-tube to inner-tube Shank spacing (m)
    epsilon = 1.0e-6  # Pipe roughness (m)
    # Coaxial tube
    r_in_in = 44.2/1000./2.
    r_in_out = 50./1000./2.
    # Outer pipe radii
    r_out_in = 97.4 / 1000. / 2.
    r_out_out = 110./1000./2.
    # Pipe radii
    # Note: This convention is different from pygfunction
    r_inner = [r_in_in, r_in_out]   # The radii of the inner pipe from in to out
    r_outer = [r_out_in, r_out_out] # The radii of the outer pipe from in to out

    # Pipe positions
    # --------------
    # Single U-tube [(x_in, y_in), (x_out, y_out)]
    pos_single = plat.media.Pipe.place_pipes(s, r_out, 1)
    # Single U-tube BHE object
    single_u_tube = plat.borehole_heat_exchangers.SingleUTube
    # Double U-tube
    pos_double = plat.media.Pipe.place_pipes(s, r_out, 2)
    double_u_tube = plat.borehole_heat_exchangers.MultipleUTube
    # Coaxial tube
    pos_coaxial = (0, 0)
    coaxial_tube = plat.borehole_heat_exchangers.CoaxialPipe

    # Thermal conductivities
    # ----------------------
    k_p = 0.4  # Pipe thermal conductivity (W/m.K)
    k_s = 2.0  # Ground thermal conductivity (W/m.K)
    k_g = 1.0  # Grout thermal conductivity (W/m.K)

    # Volumetric heat capacities
    # --------------------------
    rhoCp_p = 1542. * 1000.  # Pipe volumetric heat capacity (J/K.m3)
    rhoCp_s = 2343.493 * 1000.  # Soil volumetric heat capacity (J/K.m3)
    rhoCp_g = 3901. * 1000.  # Grout volumetric heat capacity (J/K.m3)

    # Thermal properties
    # ------------------
    # Pipe
    pipe_single = \
        plat.media.Pipe(pos_single, r_in, r_out, s, epsilon, k_p, rhoCp_p)
    pipe_double = \
        plat.media.Pipe(pos_double, r_in, r_out, s, epsilon, k_p, rhoCp_p)
    pipe_coaxial = \
        plat.media.Pipe(pos_coaxial, r_inner, r_outer, 0, epsilon, k_p, rhoCp_p)
    # Soil
    ugt = 18.3  # Undisturbed ground temperature (degrees Celsius)
    soil = plat.media.Soil(k_s, rhoCp_s, ugt)
    # Grout
    grout = plat.media.ThermalProperty(k_g, rhoCp_g)

    # Inputs related to fluid
    # -----------------------
    # Fluid properties
    mixer = 'MEG'  # Ethylene glycol mixed with water
    percent = 0.  # Percentage of ethylene glycol added in
    fluid = gt.media.Fluid(mixer=mixer, percent=percent)

    # Fluid properties
    V_flow_borehole = 0.2  # Borehole volumetric flow rate (L/s)

    # Define a borehole
    borehole = gt.boreholes.Borehole(H, D, r_b, x=0., y=0.)

    # Simulation start month and end month
    # --------------------------------
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
        pd.read_csv('../Atlanta_Office_Building_Loads.csv').to_dict('list')
    # Take only the first column in the dictionary
    hourly_extraction_ground_loads: list = \
        hourly_extraction[list(hourly_extraction.keys())[0]]

    # Perform field selection using bisection search between a 1x1 and 32x32
    coordinates_domain = dt.domains.square_and_near_square(1, 32, B)

    # Geometric constraints for the `near-square` routine
    geometric_constraints = dt.media.GeometricConstraints(
        B_max_x=B, unconstrained=True)

    # Note: Flow functionality is currently only on a borehole basis. Future
    # development will include the ability to change the flow rate to be on a
    # system flow rate basis.
    design_single_u_tube = dt.design.Design(
        V_flow_borehole, borehole, single_u_tube, fluid, pipe_single, grout,
        soil, sim_params, geometric_constraints, coordinates_domain,
        hourly_extraction_ground_loads, routine='near-square')

    # Output the design interface object to a json file so it can be reused
    dt.utilities.create_input_file(
        design_single_u_tube, file_name='ghedt_input')

    design_double_u_tube = dt.design.Design(
        V_flow_borehole, borehole, double_u_tube, fluid, pipe_double, grout,
        soil, sim_params, geometric_constraints, coordinates_domain,
        hourly_extraction_ground_loads, routine='near-square')

    dt.utilities.create_input_file(
        design_double_u_tube, file_name='double_u_tube')

    design_coaxial_u_tube = dt.design.Design(
        V_flow_borehole, borehole, coaxial_tube, fluid, pipe_coaxial, grout,
        soil, sim_params, geometric_constraints, coordinates_domain,
        hourly_extraction_ground_loads, routine='near-square')

    dt.utilities.create_input_file(
        design_coaxial_u_tube, file_name='coaxial_tube')


if __name__ == '__main__':
    main()
