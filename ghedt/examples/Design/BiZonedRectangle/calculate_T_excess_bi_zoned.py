# Jack C. Cook
# Thursday, October 28, 2021

import ghedt
import ghedt.PLAT as PLAT
import ghedt.PLAT.pygfunction as gt
import pandas as pd
from time import time as clock


def compute_domain(coordinates_domain, V_flow_borehole, borehole, bhe_object,
                   log_time, fluid, pipe, grout, soil, sim_params,
                   hourly_extraction_ground_loads):
    # Need to compute the whole domain for the plot
    nbh_values = []
    T_excess_values = []
    for i in range(len(coordinates_domain)):
        coordinates = coordinates_domain[i]

        V_flow_system = V_flow_borehole * float(
            len(coordinates))  # System volumetric flow rate (L/s)
        m_flow_borehole = V_flow_borehole / 1000. * fluid.rho

        B = ghedt.utilities.borehole_spacing(borehole, coordinates)

        borehole.H = sim_params.max_Height

        g_function = ghedt.gfunction.compute_live_g_function(
            B, [borehole.H], [borehole.r_b], [borehole.D], m_flow_borehole,
            bhe_object, log_time, coordinates, fluid, pipe, grout,
            soil)

        # Initialize the GHE object
        ghe = ghedt.ground_heat_exchangers.GHE(
            V_flow_system, B, bhe_object, fluid, borehole, pipe, grout,
            soil, g_function, sim_params,
            hourly_extraction_ground_loads)

        T_excess = ghe.cost(*ghe.simulate(method='hybrid'))

        nbh_values.append(len(coordinates))
        T_excess_values.append(T_excess)

    return nbh_values, T_excess_values


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
    # Total fluid mass flow rate per borehole (kg/s)
    m_flow_borehole = V_flow_borehole / 1000. * fluid.rho

    # Define a borehole
    borehole = gt.boreholes.Borehole(H, D, r_b, x=0., y=0.)

    log_time = ghedt.utilities.Eskilson_log_times()

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
        ghedt.domains.bi_rectangle_zoned_nested(
            length, width, B_min, B_max_x, B_max_y)

    outer_domain = [coordinates_domain_nested[0][0]]
    for i in range(len(coordinates_domain_nested)):
        outer_domain.append(coordinates_domain_nested[i][-1])

    nbh_values, T_excess_values = compute_domain(
        outer_domain, V_flow_borehole, borehole, bhe_object,
        log_time, fluid, pipe, grout, soil, sim_params,
        hourly_extraction_ground_loads)

    bisection_search = ghedt.search_routines.Bisection1D(
        outer_domain, V_flow_borehole, borehole, bhe_object,
        fluid, pipe, grout, soil, sim_params, hourly_extraction_ground_loads,
        disp=False)

    d = {'Outer_Domain': {
        'Domain': {'nbh': nbh_values, 'T_excess': T_excess_values}}}

    nbh_values = []
    T_excess_values = []
    for i in bisection_search.calculated_temperatures:
        coordinates = bisection_search.coordinates_domain[i]
        nbh_values.append(len(coordinates))
        T_excess_values.append(bisection_search.calculated_temperatures[i])

    d['Outer_Domain']['Searched'] = \
        {'nbh': nbh_values, 'T_excess': T_excess_values}

    selected_domain = \
        coordinates_domain_nested[bisection_search.selection_key-1]

    nbh_values, T_excess_values = compute_domain(
        selected_domain, V_flow_borehole, borehole, bhe_object,
        log_time, fluid, pipe, grout, soil, sim_params,
        hourly_extraction_ground_loads)

    bisection_search = ghedt.search_routines.Bisection1D(
        selected_domain, V_flow_borehole, borehole, bhe_object,
        fluid, pipe, grout, soil, sim_params, hourly_extraction_ground_loads,
        disp=False)

    d['Selected_Domain'] = \
        {'Domain': {'nbh': nbh_values, 'T_excess': T_excess_values}}

    nbh_values = []
    T_excess_values = []
    for i in bisection_search.calculated_temperatures:
        coordinates = bisection_search.coordinates_domain[i]
        nbh_values.append(len(coordinates))
        T_excess_values.append(bisection_search.calculated_temperatures[i])

    d['Selected_Domain']['Searched'] = \
        {'nbh': nbh_values, 'T_excess': T_excess_values}

    d['Entire_Domain'] = {}

    # Now compute the whole domain for all domains
    for i in range(len(coordinates_domain_nested)):
        coordinates_domain = coordinates_domain_nested[i]

        nbh_values, T_excess_values = compute_domain(
            coordinates_domain, V_flow_borehole, borehole, bhe_object,
            log_time, fluid, pipe, grout, soil, sim_params,
            hourly_extraction_ground_loads)

        d['Entire_Domain'][i] = \
            {'nbh': nbh_values, 'T_excess': T_excess_values}

    file_name = 'bi_zoned_search'

    ghedt.utilities.js_dump(file_name, d)


if __name__ == '__main__':
    main()
