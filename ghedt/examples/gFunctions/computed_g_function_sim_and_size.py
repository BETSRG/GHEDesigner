# Jack C. Cook
# Monday, December 20, 2021

# Purpose: Show how to simulate and size a g-function that has been previously
# computed and stored in a `cpgfunction-output` style json file.

import ghedt.peak_load_analysis_tool as plat
import pandas as pd
import pygfunction as gt
import ghedt as dt


def main():
    # Borehole dimensions
    # -------------------
    H = 100.0  # Borehole length (m)
    D = 2.0  # Borehole buried depth (m)
    r_b = 150.0 / 1000.0 / 2.0  # Borehole radius]
    B = 5.0  # Borehole spacing (m)

    # Pipe dimensions
    # ---------------
    r_out = 26.67 / 1000.0 / 2.0  # Pipe outer radius (m)
    r_in = 21.6 / 1000.0 / 2.0  # Pipe inner radius (m)
    s = 32.3 / 1000.0  # Inner-tube to inner-tube Shank spacing (m)
    epsilon = 1.0e-6  # Pipe roughness (m)

    # Pipe positions
    # --------------
    # Single U-tube [(x_in, y_in), (x_out, y_out)]
    pos = plat.media.Pipe.place_pipes(s, r_out, 1)
    # Single U-tube BHE object
    bhe_object = plat.borehole_heat_exchangers.SingleUTube

    # Thermal conductivities
    # ----------------------
    k_p = 0.4  # Pipe thermal conductivity (W/m.K)
    k_s = 2.0  # Ground thermal conductivity (W/m.K)
    k_g = 1.0  # Grout thermal conductivity (W/m.K)

    # Volumetric heat capacities
    # --------------------------
    rhoCp_p = 1542.0 * 1000.0  # Pipe volumetric heat capacity (J/K.m3)
    rhoCp_s = 2343.493 * 1000.0  # Soil volumetric heat capacity (J/K.m3)
    rhoCp_g = 3901.0 * 1000.0  # Grout volumetric heat capacity (J/K.m3)

    # Thermal properties
    # ------------------
    # Pipe
    pipe = plat.media.Pipe(pos, r_in, r_out, s, epsilon, k_p, rhoCp_p)
    # Soil
    ugt = 18.3  # Undisturbed ground temperature (degrees Celsius)
    soil = plat.media.Soil(k_s, rhoCp_s, ugt)
    # Grout
    grout = plat.media.Grout(k_g, rhoCp_g)

    # Read in g-functions from GLHEPro
    file = "GLHEPRO_gFunctions_12x13.json"
    data = dt.utilities.js_load(file)

    # Configure the database data for input to the goethermal GFunction object
    geothermal_g_input = dt.gfunction.GFunction.configure_database_file_for_usage(data)

    # Initialize the GFunction object
    g_function = dt.gfunction.GFunction(**geothermal_g_input)

    # Inputs related to fluid
    # -----------------------
    V_flow_system = 31.2  # System volumetric flow rate (L/s)
    mixer = "MEG"  # Ethylene glycol mixed with water
    percent = 0.0  # Percentage of ethylene glycol added in
    # Fluid properties
    fluid = gt.media.Fluid(mixer=mixer, percent=percent)

    # Define a borehole
    borehole = gt.boreholes.Borehole(H, D, r_b, x=0.0, y=0.0)

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
    max_Height = 200  # in meters
    min_Height = 60  # in meters
    sim_params = plat.media.SimulationParameters(
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
    hourly_extraction: dict = pd.read_csv(
        "../Atlanta_Office_Building_Loads.csv"
    ).to_dict("list")
    # Take only the first column in the dictionary
    hourly_extraction_ground_loads: list = hourly_extraction[
        list(hourly_extraction.keys())[0]
    ]

    # --------------------------------------------------------------------------

    # Initialize GHE object
    ghe = dt.ground_heat_exchangers.GHE(
        V_flow_system,
        B,
        bhe_object,
        fluid,
        borehole,
        pipe,
        grout,
        soil,
        g_function,
        sim_params,
        hourly_extraction_ground_loads,
    )

    ghe.size()

    calculation_details = "GLHEPRO_gFunctions_12x13.json".split(".")[0]
    print(calculation_details)
    print("Height of boreholes: {0:.3f}".format(ghe.bhe.b.H))


if __name__ == "__main__":
    main()
