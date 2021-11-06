# Jack C. Cook
# Monday, October 25, 2021

import ghedt
import ghedt.PLAT as PLAT
import numpy as np
import gFunctionDatabase as gfdb
import pygfunction as gt
import os
import json


def js_r(filename: str):
    with open(filename) as f_in:
        return json.load(f_in)


def js_out(name, d):
    with open(name + '.json', 'w+') as fp:
        json.dump(d, fp, indent=4)


def create_if_not(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return


def main():

    calculations = ['12_Equal_Segments_Similarities_UBWT',
                    '12_Equal_Segments_Similarities_MIFT',
                    '12_Equal_Segments_Equivalent_MIFT',
                    '8_Unequal_Segments_Equivalent_MIFT']

    H_values = [48., 96., 192., 384.]

    for k in range(len(H_values)):

        output_folder = \
            'Calculated_g_Functions/RowWise/' + str(int(H_values[k])) \
            + 'm_Depth/'

        submission_folder = 'Submission_Files/' + str(int(H_values[k])) \
                            + 'm_Depth/'
        submission_files = os.listdir(submission_folder)

        for j in range(len(calculations)):

            calculation = calculations[j]

            calculation_folder = output_folder + calculation + '/'
            create_if_not(calculation_folder)

            # ------------------------------------------------------------------
            # Simulation parameters
            # ------------------------------------------------------------------
            for i in range(len(submission_files)):

                submission_file = submission_folder + submission_files[i]
                submission_data = js_r(submission_file)

                # Borehole dimensions
                # -------------------
                name = submission_data['name']
                H = submission_data['H']  # Borehole length (m)
                D = submission_data['D']  # Borehole buried depth (m)
                r_b = submission_data['r_b']  # Borehole radius]
                B = submission_data['B']  # Borehole spacing (m)

                # Pipe dimensions
                # ---------------
                r_out = submission_data['r_out']  # Pipe outer radius (m)
                r_in = submission_data['r_in']  # Pipe inner radius (m)
                s = 32.3 / 1000.  # Inner-tube to inner-tube Shank spacing (m)
                epsilon = submission_data['epsilon']  # Pipe roughness (m)

                # Pipe positions
                # --------------
                # Single U-tube [(x_in, y_in), (x_out, y_out)]
                pos = PLAT.media.Pipe.place_pipes(s, r_out, 1)
                # Single U-tube BHE object
                bhe_object = PLAT.borehole_heat_exchangers.SingleUTube

                # Thermal conductivities
                # ----------------------
                k_p = submission_data['k_p']  # Pipe thermal conductivity (W/m.K)
                k_s = submission_data['k_s']  # Ground thermal conductivity (W/m.K)
                k_g = submission_data['k_g']  # Grout thermal conductivity (W/m.K)

                # Volumetric heat capacities
                # --------------------------
                rhoCp_p = 1542. * 1000.  # Pipe volumetric heat capacity (J/K.m3)
                rhoCp_s = submission_data['rhoCp_s']  # Soil volumetric heat capacity (J/K.m3)
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

                alpha = soil.k / soil.rhoCp

                # Eskilson's original ln(t/ts) values
                ts = H ** 2 / (9. * alpha)  # Bore field characteristic time
                log_time = np.array(gfdb.utilities.Eskilson_log_times())
                time_values = np.exp(log_time) * ts

                # Inputs related to fluid
                # -----------------------
                # Fluid properties
                mixer = submission_data['mixer']  # Ethylene glycol mixed with water
                percent = submission_data['percent']  # Percentage of ethylene glycol added in
                fluid = gt.media.Fluid(mixer=mixer, percent=percent)

                # Total fluid mass flow rate per borehole (kg/s)
                m_flow_borehole = submission_data['m_flow_borehole']

                # Define a borehole
                borehole = gt.boreholes.Borehole(H, D, r_b, x=0., y=0.)

                # Coordinates
                coordinates = submission_data['bore_locations']

                # Calculate g-functions
                # g-Function calculation options
                disp = True

                # Calculate a uniform inlet fluid temperature g-function with 12 equal
                # segments using the similarities solver

                calc_details = calculation.split('_')
                nSegments = int(calc_details[0])
                segments = calc_details[1]
                solver = calc_details[3].lower()
                boundary = calc_details[4]

                gfunc = ghedt.gfunction.calculate_g_function(
                    m_flow_borehole, bhe_object, time_values, coordinates,
                    borehole, fluid, pipe, grout, soil, nSegments=nSegments,
                    segments=segments, solver=solver, boundary=boundary,
                    disp=disp)

                key = '{}_{}_{}_{}'.format(B, H, r_b, D)

                d_out = {'g': {}, 'bore_locations': coordinates,
                         'logtime': log_time.tolist()}
                d_out['g'][key] = gfunc.gFunc.tolist()

                js_out(calculation_folder + name, d_out)


if __name__ == '__main__':
    main()
