# Jack C. Cook
# Saturday, October 30, 2021

import ghedt
import ghedt.PLAT as PLAT
import ghedt.PLAT.pygfunction as gt
import pandas as pd
from time import time as clock


def main():
    # Borehole dimensions
    # -------------------
    H = 96.  # Borehole length (m)
    D = 2.  # Borehole buried depth (m)
    r_b = 0.075  # Borehole radius]
    B = 5.  # Borehole spacing (m)

    # Pipe dimensions
    # ---------------
    r_out = 26.67 / 1000. / 2.  # Pipe outer radius (m)
    r_in = 21.6 / 1000. / 2.  # Pipe inner radius (m)
    s = 32.3 / 1000.  # Inner-tube to inner-tube Shank spacing (m)
    epsilon = 1.0e-6  # Pipe roughness (m)

    # Pipe positions
    # --------------
    # Single U-tube [(x_in, y_in), (x_out, y_out)]
    pos = PLAT.media.Pipe.place_pipes(s, r_out, 1)
    # Single U-tube BHE object
    bhe_object = PLAT.borehole_heat_exchangers.SingleUTube

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
    pipe = PLAT.media.Pipe(pos, r_in, r_out, s, epsilon, k_p, rhoCp_p)
    # Soil
    ugt = 18.3  # Undisturbed ground temperature (degrees Celsius)
    soil = PLAT.media.Soil(k_s, rhoCp_s, ugt)
    # Grout
    grout = PLAT.media.ThermalProperty(k_g, rhoCp_g)

    # Inputs related to fluid
    # -----------------------
    # Fluid properties
    mixer = 'MEG'  # Ethylene glycol mixed with water
    percent = 0.  # Percentage of ethylene glycol added in
    fluid = gt.media.Fluid(mixer=mixer, percent=percent)

    # Fluid properties
    V_flow_borehole = 0.2  # System volumetric flow rate (L/s)

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
    sim_params = PLAT.media.SimulationParameters(
        start_month, end_month, max_EFT_allowable, min_EFT_allowable,
        max_Height, min_Height)

    # Process loads from file
    # -----------------------
    # read in the csv file and convert the loads to a list of length 8760
    hourly_extraction: dict = \
        pd.read_csv('../../GHE/Atlanta_Office_Building_Loads.csv').to_dict('list')
    # Take only the first column in the dictionary
    hourly_extraction_ground_loads: list = \
        hourly_extraction[list(hourly_extraction.keys())[0]]

    # --------------------------------------------------------------------------

    # Rectangular design constraints are the land and range of B-spacing
    length = 85.  # m
    width = 36.5  # m
    B_min = 4.45  # m
    B_max_x = 10.  # m
    B_max_y = 12.

    coordinates_domain_nested = \
        ghedt.domains.bi_rectangle_zoned_nested(
            length, width, B_min, B_max_x, B_max_y)

    T_excess_values_outer = []
    total_drilling_depth_sized = []

    for i in range(11, len(coordinates_domain_nested)):
        try:
            bisection_search = ghedt.search_routines.Bisection1D(
                coordinates_domain_nested[i], V_flow_borehole, borehole,
                bhe_object, fluid, pipe, grout, soil, sim_params,
                hourly_extraction_ground_loads, disp=False)

            calculated_excess_values = \
                list(bisection_search.calculated_temperatures.values())
            calculated_excess_indices = \
                list(bisection_search.calculated_temperatures.keys())
            idx = max(calculated_excess_indices)
            T_excess_outer = bisection_search.calculated_temperatures[idx]
            T_excess_values_outer.append(T_excess_outer)

            nbh = len(bisection_search.selected_coordinates)
            print('Number of boreholes: {}'.format(nbh))
            print('Borehole spacing: {}'.format(bisection_search.ghe.GFunction.B))

            # Perform sizing in between the min and max bounds
            tic = clock()
            ghe = bisection_search.ghe
            ghe.compute_g_functions()

            ghe.size(method='hybrid')
            toc = clock()
            print('Time to compute g-functions and size: {} seconds'.format(toc - tic))

            print('Sized height of boreholes: {0:.2f} m'.format(ghe.bhe.b.H))

            print('Total drilling depth: {0:.1f} m'.format(ghe.bhe.b.H * nbh))
            total_drilling_depth = ghe.bhe.b.H * nbh

            total_drilling_depth_sized.append(total_drilling_depth)

        except ValueError as msg:
            print(i)
            print(msg)

    fig = gt.utilities._initialize_figure()
    ax = fig.add_subplot(111)

    ax.scatter(T_excess_values_outer, total_drilling_depth_sized)

    ax.set_xlabel('Outer domain excess temperature ($\degree$C)')
    ax.set_ylabel('Sizing result of search on inner domain (m)')

    ax.grid()
    ax.set_axisbelow(True)

    fig.tight_layout()

    fig.savefig('outer_vs_inner_sizing.png')


if __name__ == '__main__':
    main()
