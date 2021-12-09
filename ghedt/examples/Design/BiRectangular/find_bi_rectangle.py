# Jack C. Cook
# Thursday, October 28, 2021

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

    # Perform field selection using bisection search between a 1x1 and 32x32
    coordinates_domain_nested = \
        ghedt.domains.bi_rectangle_nested(length, width, B_min, B_max_x,
                                          B_max_y, disp=False)

    coordinates_domain = coordinates_domain_nested[0]

    output_folder = 'Bi-Rectangle_Domain'
    ghedt.domains.visualize_domain(coordinates_domain, output_folder)

    tic = clock()
    bisection_search = ghedt.search_routines.Bisection2D(
        coordinates_domain_nested, V_flow_borehole, borehole, bhe_object,
        fluid, pipe, grout, soil, sim_params, hourly_extraction_ground_loads,
        disp=True)
    toc = clock()
    print('Time to perform bisection search: {} seconds'.format(toc - tic))

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

    # Plot go and no-go zone with corrected borefield
    # -----------------------------------------------
    coordinates = bisection_search.selected_coordinates

    perimeter = [[0., 0.], [85., 0.], [85., 80.], [0., 80.]]
    l_x_building = 50
    l_y_building = 33.3
    origin_x, origin_y = (15, 36.5)
    no_go = [[origin_x, origin_y], [origin_x+l_x_building, origin_y],
             [origin_x+l_x_building, origin_y+l_y_building],
             [origin_x, origin_y+l_y_building]]

    fig, ax = ghedt.gfunction.GFunction.visualize_area_and_constraints(
        perimeter, coordinates, no_go=no_go)

    fig.savefig('bi-rectangle_case.png', bbox_inches='tight', pad_inches=0.1)

    # Export plots for animation
    folder = 'Calculated_Temperature_Fields/'
    ghedt.utilities.create_if_not(folder)

    outer_domain = [coordinates_domain_nested[0][0]]
    for i in range(len(coordinates_domain_nested)):
        outer_domain.append(coordinates_domain_nested[i][-1])

    calculated_temperatures_nested = \
        bisection_search.calculated_temperatures_nested[0]

    count = 0
    for key in calculated_temperatures_nested:
        _coordinates = outer_domain[key]
        fig, ax = ghedt.gfunction.GFunction.visualize_area_and_constraints(
            perimeter, _coordinates, no_go=no_go)
        name = str(count).zfill(2)
        fig.savefig(folder + name + '.png', bbox_inches='tight', pad_inches=0.1)
        count += 1

    calculated_temperatures = bisection_search.calculated_temperatures
    inner_domain = bisection_search.coordinates_domain
    for key in calculated_temperatures:
        _coordinates = inner_domain[key]
        fig, ax = ghedt.gfunction.GFunction.visualize_area_and_constraints(
            perimeter, _coordinates, no_go=no_go)
        name = str(count).zfill(2)
        fig.savefig(folder + name + '.png', bbox_inches='tight', pad_inches=0.1)
        count += 1


if __name__ == '__main__':
    main()
