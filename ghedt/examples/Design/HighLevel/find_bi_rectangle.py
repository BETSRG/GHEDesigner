# Jack C. Cook
# Monday, December 27, 2021

# Purpose: Design a constrained bi-rectangular field using the common design
# interface with a single U-tube, multiple U-tube and coaxial tube borehole
# heat exchanger.


import ghedt as dt
import ghedt.peak_load_analysis_tool as plat
import ghedt.pygfunction as gt
import pandas as pd
from time import time as clock


def main():
    # Borehole dimensions
    # -------------------
    H = 96.  # Borehole length (m)
    D = 2.  # Borehole buried depth (m)
    r_b = 0.075  # Borehole radius (m)

    # Pipe dimensions
    # ---------------
    # Single and Multiple U-tubes
    r_out = 26.67 / 1000. / 2.  # Pipe outer radius (m)
    r_in = 21.6 / 1000. / 2.  # Pipe inner radius (m)
    s = 32.3 / 1000.  # Inner-tube to inner-tube Shank spacing (m)
    epsilon = 1.0e-6  # Pipe roughness (m)
    # Coaxial tube
    r_in_in = 44.2 / 1000. / 2.
    r_in_out = 50. / 1000. / 2.
    # Outer pipe radii
    r_out_in = 97.4 / 1000. / 2.
    r_out_out = 110. / 1000. / 2.
    # Pipe radii
    # Note: This convention is different from pygfunction
    r_inner = [r_in_in, r_in_out]  # The radii of the inner pipe from in to out
    r_outer = [r_out_in,
               r_out_out]  # The radii of the outer pipe from in to out

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
    k_p_coax = [0.4, 0.4]  # Pipes thermal conductivity (W/m.K)
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
        plat.media.Pipe(pos_coaxial, r_inner, r_outer, 0, epsilon, k_p_coax,
                        rhoCp_p)
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
        pd.read_csv('../../Atlanta_Office_Building_Loads.csv').to_dict('list')
    # Take only the first column in the dictionary
    hourly_extraction_ground_loads: list = \
        hourly_extraction[list(hourly_extraction.keys())[0]]

    # Rectangular design constraints are the land and range of B-spacing
    length = 85.  # m
    width = 36.5  # m
    B_min = 4.45  # m
    B_max_x = 10.  # m
    B_max_y = 12.  # m

    # Geometric constraints for the `find_rectangle` routine
    # Required geometric constraints for the uniform rectangle design: length,
    # width, B_min, B_max
    geometric_constraints = dt.media.GeometricConstraints(
        length=length, width=width, B_min=B_min, B_max_x=B_max_x,
        B_max_y=B_max_y)

    # Note: Flow functionality is currently only on a borehole basis. Future
    # development will include the ability to change the flow rate to be on a
    # system flow rate basis.

    # Single U-tube
    # -------------
    design_single_u_tube = dt.design.Design(
        V_flow_borehole, borehole, single_u_tube, fluid, pipe_single, grout,
        soil, sim_params, geometric_constraints, hourly_extraction_ground_loads,
        routine='bi-rectangle')

    # Find a constrained rectangular design for a single U-tube and size it.
    tic = clock()
    bisection_search = design_single_u_tube.find_design()
    bisection_search.ghe.compute_g_functions()
    bisection_search.ghe.size(method='hybrid')
    toc = clock()
    title = 'HighLevel/find_bi_rectangle.py results'
    print(title + '\n' + len(title) * '=')
    subtitle = '* Single U-tube'
    print(subtitle + '\n' + len(subtitle) * '-')
    print('Calculation time: {0:.2f} seconds'.format(toc - tic))
    print('Height: {0:.4f} meters'.format(bisection_search.ghe.bhe.b.H))
    nbh = len(bisection_search.ghe.GFunction.bore_locations)
    print('Number of boreholes: {}'.format(nbh))
    print('Total Drilling: {0:.1f} meters\n'.
          format(bisection_search.ghe.bhe.b.H * nbh))

    # Double U-tube
    # -------------
    design_double_u_tube = dt.design.Design(
        V_flow_borehole, borehole, double_u_tube, fluid, pipe_double, grout,
        soil, sim_params, geometric_constraints, hourly_extraction_ground_loads,
        routine='bi-rectangle')

    # Find a constrained rectangular design for a double U-tube and size it.
    tic = clock()
    bisection_search = design_double_u_tube.find_design()
    bisection_search.ghe.compute_g_functions()
    bisection_search.ghe.size(method='hybrid')
    toc = clock()
    subtitle = '* Double U-tube'
    print(subtitle + '\n' + len(subtitle) * '-')
    print('Calculation time: {0:.2f} seconds'.format(toc - tic))
    print('Height: {0:.4f} meters'.format(bisection_search.ghe.bhe.b.H))
    nbh = len(bisection_search.ghe.GFunction.bore_locations)
    print('Number of boreholes: {}'.format(nbh))
    print('Total Drilling: {0:.1f} meters\n'.
          format(bisection_search.ghe.bhe.b.H * nbh))

    # Coaxial tube
    # -------------
    design_coaxial_u_tube = dt.design.Design(
        V_flow_borehole, borehole, coaxial_tube, fluid, pipe_coaxial, grout,
        soil, sim_params, geometric_constraints, hourly_extraction_ground_loads,
        routine='bi-rectangle')

    # Find a constrained rectangular design for a coaxial tube and size it.
    tic = clock()
    bisection_search = design_coaxial_u_tube.find_design()
    bisection_search.ghe.compute_g_functions()
    bisection_search.ghe.size(method='hybrid')
    toc = clock()
    subtitle = '* Coaxial tube'
    print(subtitle + '\n' + len(subtitle) * '-')
    print('Calculation time: {0:.2f} seconds'.format(toc - tic))
    print('Height: {0:.4f} meters'.format(bisection_search.ghe.bhe.b.H))
    nbh = len(bisection_search.ghe.GFunction.bore_locations)
    print('Number of boreholes: {}'.format(nbh))
    print('Total Drilling: {0:.1f} meters\n'.
          format(bisection_search.ghe.bhe.b.H * nbh))


if __name__ == '__main__':
    main()
