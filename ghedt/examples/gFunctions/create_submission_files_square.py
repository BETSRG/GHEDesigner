# Jack C. Cook
# Monday, October 25, 2021

import os
import gFunctionDatabase as gfdb
import json
import pygfunction as gt


def create_if_not(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return


def js_o(filename, d):
    with open(filename + '.json', 'w+') as fp:
        json.dump(d, fp, indent=4)


def main():
    output_folder = 'Submission_Files/'
    create_if_not(output_folder)

    # Borehole dimensions
    D = 2.          # Borehole buried depth (m)
    H = 96.        # Borehole length (m)
    r_b = 0.075    # Borehole radius (m)
    B = 5.          # Uniform borehole spacing (m)

    # Pipe dimensions (all configurations)
    epsilon = 1.0e-6    # Pipe roughness (m)

    # Pipe dimensions (single U-tube and double U-tube)
    r_out = 26.67 / 1000. / 2.      # Pipe outer radius (m)
    r_in = 21.6 / 1000. / 2.       # Pipe inner radius (m)

    # Ground properties
    k_p = 0.4  # Pipe thermal conductivity (W/m.K)
    k_s = 2.0  # Ground thermal conductivity (W/m.K)
    k_g = 1.0  # Grout thermal conductivity (W/m.K)
    rhoCp_s = 2343.493 * 1000.  # J/kg.m3
    alpha = k_s / rhoCp_s

    # Fluid properties
    mixer = 'MEG'  # Ethylene glycol mixed with water
    percent = 0.  # Percentage of ethylene glycol added in
    fluid = gt.media.Fluid(mixer=mixer, percent=percent)

    # Fluid properties
    V_flow_borehole = 0.2  # System volumetric flow rate (L/s)
    # Total fluid mass flow rate per borehole (kg/s)
    m_flow_borehole = V_flow_borehole / 1000. * fluid.rho

    s = 32.3 / 1000.  # Inner-tube to inner-tube Shank spacing (m)

    pos = [(-0.029484999999999997, 3.610871087285971e-18),
           (0.029484999999999997, -7.221742174571942e-18)]

    log_time = [-8.5, -7.8, -7.2, -6.5, -5.9, -5.2, -4.5,
                -3.963, -3.27, -2.864, -2.577, -2.171, -1.884,
                -1.191, -0.497, -0.274, -0.051, 0.196, 0.419,
                0.642, 0.873, 1.112, 1.335, 1.679, 2.028,
                2.275, 3.003]

    for i in range(1, 21):
        for j in range(1):
            print('{}x{}'.format(i, i+j))
            coordinates = gfdb.coordinates.rectangle(i, i+j, B, B)
            name = '{}x{}rectangle'.format(i, j+i)
            d_out = {
                'B': B,
                'D': D,
                'H': H,
                'r_b': r_b,
                'alpha': alpha,
                'r_out': r_out,
                'r_in': r_in,
                'epsilon': epsilon,
                'pos': pos,
                'm_flow_borehole': m_flow_borehole,
                'k_p': k_p,
                'k_s': k_s,
                'k_g': k_g,
                'rhoCp_s': rhoCp_s,
                'mixer': mixer,
                'percent': percent,
                'bore_locations': coordinates,
                'logtime': log_time,
                'name': name
            }

            js_o(output_folder + 'sub-' + str(i-1), d_out)

    return


if __name__ == '__main__':
    main()
