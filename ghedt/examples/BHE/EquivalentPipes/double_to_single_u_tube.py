# Jack C. Cook
# Saturday, August 21, 2021

import ghedt.peak_load_analysis_tool as plat
import pygfunction as gt


def main():
    # Borehole dimensions
    H = 100.  # Borehole length (m)
    D = 2.  # Borehole buried depth (m)
    r_b = 150. / 1000. / 2.  # Borehole radius

    # Pipe dimensions
    r_out = 26.67 / 1000. / 2.  # Pipe outer radius (m)
    r_in = 21.6 / 1000. / 2.  # Pipe inner radius (m)
    s = 32.3 / 1000.  # Inner-tube to inner-tube Shank spacing (m)
    epsilon = 1.0e-6  # Pipe roughness (m)

    # Thermal conductivities
    k_p = 0.4  # Pipe thermal conductivity (W/m.K)
    k_s = 2.0  # Ground thermal conductivity (W/m.K)
    k_g = 1.0  # Grout thermal conductivity (W/m.K)

    # Volumetric heat capacities
    rhoCp_p = 1542. * 1000.  # Pipe volumetric heat capacity (J/K.m3)
    rhoCp_s = 2343.493 * 1000.  # Soil volumetric heat capacity (J/K.m3)
    rhoCp_g = 3901. * 1000.  # Grout volumetric heat capacity (J/K.m3)

    # Pipe positions
    # Double U-tube [(x_in, y_in), (x_out, y_out), (x_in, y_in), (x_out, y_out)]
    pos = plat.media.Pipe.place_pipes(s, r_out, 2)

    # Thermal properties
    # Pipe
    pipe = plat.media.Pipe(pos, r_in, r_out, s, epsilon, k_p, rhoCp_p)
    # Soil
    ugt = 18.3  # Undisturbed ground temperature (degrees Celsius)
    soil = plat.media.Soil(k_s, rhoCp_s, ugt)
    # Grout
    grout = plat.media.Grout(k_g, rhoCp_g)

    # Fluid properties
    fluid = gt.media.Fluid(mixer='MEG', percent=0.)
    V_flow_borehole = 0.2  # Volumetric flow rate per borehole (L/s)
    # Total fluid mass flow rate per borehole (kg/s)
    m_flow_borehole = V_flow_borehole / 1000. * fluid.rho

    # Define a borehole
    borehole = gt.boreholes.Borehole(H, D, r_b, x=0., y=0.)

    # Double U-tube defaults to parallel
    double_u_tube = plat.borehole_heat_exchangers.MultipleUTube(
        m_flow_borehole, fluid, borehole, pipe, grout, soil)

    val = 'Intermediate variables'
    print(val + '\n' + len(val) * '-')
    # Intermediate variables
    V_fluid, V_pipe, R_conv, R_pipe = \
        plat.equivalance.u_tube_volumes(double_u_tube)
    print('Fluid volume per meter (m^2): {0:.8f}'.format(V_fluid))
    print('Pipe volume per meter (m^2): {0:.8f}'.format(V_pipe))
    print('Total Convective Resistance (K/(W/m)): {0:.8f}'.format(R_conv))
    print('Total Pipe Resistance (K/(W/m)): {0:.8f}'.format(R_pipe))

    single_u_tube = plat.equivalance.compute_equivalent(double_u_tube)
    val = 'Single U-tube equivalent parameters'
    print('\n' + val + '\n' + len(val) * '-')
    print('Fluid volumetric flow rate (L/s): {0:.8f}'.
          format(single_u_tube.m_flow_pipe * 1000. / single_u_tube.fluid.rho))
    print('Radius of inner pipe (m): {0:.8f}'.format(single_u_tube.r_in))
    print('Radius of outer pipe (m): {0:.8f}'.format(single_u_tube.r_out))
    print('Shank spacing (m): {0:.8f}'.format(single_u_tube.pipe.s))
    print('Convection coefficient (): {0:.8f}'.format(single_u_tube.h_f))
    print('Pipe thermal conductivity (W/m.K): {0:.8f}'.format(single_u_tube.pipe.k))
    print('Grout thermal conductivity (W/m.K): {0:.8f}'.
          format(single_u_tube.grout.k))

    Rb = single_u_tube.compute_effective_borehole_resistance()
    print('Effective borehole resistance (m.K/W): {0:.8f}'.format(Rb))

    print(single_u_tube.__repr__())

    # Plot equivalent single U-tube
    fig = single_u_tube.visualize_pipes()

    # save equivalent
    fig.savefig('double_to_single_equivalent.png')


# Main function
if __name__ == '__main__':
    main()
