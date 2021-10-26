# Jack C. Cook
# Wednesday, October 13, 2021


import ghedt.PLAT as PLAT
import matplotlib.pyplot as plt
import pandas as pd
import ghedt.PLAT.pygfunction as gt
import gFunctionDatabase as gfdb
import ghedt
from time import time as clock


def main():
    # --------------------------------------------------------------------------

    g_function_files = ['12_Equal_Segments_Similarities_UIFT.json',
                        '12_Equal_Segments_Equivalent_UIFT.json',
                        '8_Unequal_Segments_Equivalent_UIFT.json',
                        '12_Equal_Segments_UBWT.json',
                        'GLHEPRO_gFunctions_12x13.json',
                        '96_Equal_Segments_Similarities_UIFT.json']

    d_out = {}

    years = [10, 20, 30]

    for j in range(len(years)):
        print('YEARS = {}'.format(years[j]))

        d_out[years[j]] = {}

        for i in range(len(g_function_files)):

            # Borehole dimensions
            # -------------------
            H = 100.  # Borehole length (m)
            D = 2.  # Borehole buried depth (m)
            r_b = 150. / 1000. / 2.  # Borehole radius]
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

            # Read in g-functions from GLHEPro
            file = '12x13_Calculated_g_Functions/' + g_function_files[i]
            r_data, _ = gfdb.fileio.read_file(file)

            # Configure the database data for input to the goethermal GFunction object
            geothermal_g_input = gfdb.Management. \
                application.GFunction.configure_database_file_for_usage(r_data)

            # Initialize the GFunction object
            GFunction = gfdb.Management.application.GFunction(**geothermal_g_input)

            # Inputs related to fluid
            # -----------------------
            V_flow_system = 31.2  # System volumetric flow rate (L/s)
            mixer = 'MEG'  # Ethylene glycol mixed with water
            percent = 0.  # Percentage of ethylene glycol added in
            # Fluid properties
            fluid = gt.media.Fluid(mixer=mixer, percent=percent)

            # Define a borehole
            borehole = gt.boreholes.Borehole(H, D, r_b, x=0., y=0.)

            # Simulation start month and end month
            # --------------------------------
            # Simulation start month and end month
            start_month = 1
            n_years = years[j]
            end_month = n_years * 12
            # Maximum and minimum allowable fluid temperatures
            max_EFT_allowable = 35  # degrees Celsius
            min_EFT_allowable = 5  # degrees Celsius
            # Maximum and minimum allowable heights
            max_Height = 200  # in meters
            min_Height = 60  # in meters
            sim_params = PLAT.media.SimulationParameters(
                start_month, end_month, max_EFT_allowable, min_EFT_allowable,
                max_Height, min_Height)

            # Process loads from file
            # -----------------------
            # read in the csv file and convert the loads to a list of length 8760
            hourly_extraction: dict = \
                pd.read_csv('../GHE/Atlanta_Office_Building_Loads.csv').to_dict('list')
            # Take only the first column in the dictionary
            hourly_extraction_ground_loads: list = \
                hourly_extraction[list(hourly_extraction.keys())[0]]

            # --------------------------------------------------------------------------

            # Initialize Hybrid GLHE object
            GHE = ghedt.ground_heat_exchangers.GHE(
                V_flow_system, B, bhe_object, fluid, borehole, pipe, grout, soil,
                GFunction, sim_params, hourly_extraction_ground_loads)

            GHE.size()

            calculation_details = g_function_files[i].split('.')[0]
            print(calculation_details)
            print('Height of boreholes: {}'.format(GHE.bhe.b.H))

            d_out[n_years][calculation_details] = GHE.bhe.b.H

        output_location = 'Sized_Computed_g_Functions.xlsx'
        pd.DataFrame(d_out).to_excel(output_location)


if __name__ == '__main__':
    main()
