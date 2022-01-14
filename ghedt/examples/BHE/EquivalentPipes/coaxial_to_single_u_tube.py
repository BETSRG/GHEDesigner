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
    # Inner pipe radii
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
    epsilon = 1.0e-6  # Pipe roughness (m)

    # Pipe positioning
    pos = (0, 0)
    s = 0

    # Thermal properties
    k_p = [0.4, 0.4]  # Inner and outer pipe thermal conductivity (W/m.K)
    k_s = 2.0  # Ground thermal conductivity (W/m.K)
    k_g = 1.0  # Grout thermal conductivity (W/m.K)

    # Volumetric heat capacities
    rhoCp_p = 1542. * 1000.  # Pipe volumetric heat capacity (J/K.m3)
    rhoCp_s = 2343.493 * 1000.  # Soil volumetric heat capacity (J/K.m3)
    rhoCp_g = 3901. * 1000.  # Grout volumetric heat capacity (J/K.m3)

    # Thermal properties
    # Pipe
    pipe = plat.media.Pipe(pos, r_inner, r_outer, s, epsilon, k_p, rhoCp_p)
    # Soil
    ugt = 18.3  # Undisturbed ground temperature (degrees Celsius)
    soil = plat.media.Soil(k_s, rhoCp_s, ugt)
    # Grout
    grout = plat.media.ThermalProperty(k_g, rhoCp_g)

    # Fluid properties
    fluid = gt.media.Fluid(mixer='MEG', percent=0.)
    V_flow_borehole = 0.2  # Volumetric flow rate per borehole (L/s)
    # Total fluid mass flow rate per borehole (kg/s)
    m_flow_borehole = V_flow_borehole / 1000. * fluid.rho

    # Define a borehole
    borehole = gt.boreholes.Borehole(H, D, r_b, x=0., y=0.)

    Coaxial = plat.borehole_heat_exchangers.CoaxialPipe(
        m_flow_borehole, fluid, borehole, pipe, grout, soil)

    var = 'Intermediate variables'
    print(var)
    print(len(var) * '-')
    V_fluid, V_pipe, R_conv, R_pipe = \
        plat.equivalance.concentric_tube_volumes(Coaxial)
    print('Fluid volume per meter (m^2): {0:.8f}'.format(V_fluid))
    print('Pipe volume per meter (m^2): {0:.8f}'.format(V_pipe))
    print('Total Convective Resistance (K/(W/m)): {0:.8f}'.format(R_conv))
    print('Total Pipe Resistance (K/(W/m)): {0:.8f}'.format(R_pipe))
    print('\n')

    single_u_tube = plat.equivalance.compute_equivalent(Coaxial)

    val = 'Single U-tube equivalent parameters'
    print('\n' + val + '\n' + len(val) * '-')
    print('Fluid volumetric flow rate (L/s): {0:.8f}'.
          format(single_u_tube.m_flow_pipe * 1000. / single_u_tube.fluid.rho))
    print('Radius of inner pipe (m): {0:.8f}'.format(single_u_tube.r_in))
    print('Radius of outer pipe (m): {0:.8f}'.format(single_u_tube.r_out))
    print('Shank spacing (m): {0:.8f}'.format(single_u_tube.pipe.s))
    print('Convection coefficient (): {0:.8f}'.format(single_u_tube.h_f))
    print('Pipe thermal conductivity (W/m.K): {0:.8f}'.format(
        single_u_tube.pipe.k))
    print('Grout thermal conductivity (W/m.K): {0:.8f}'.
          format(single_u_tube.grout.k))

    Rb = single_u_tube.compute_effective_borehole_resistance()
    print('Effective borehole resistance (m.K/W): {0:.8f}'.format(Rb))

    print(single_u_tube.__repr__())

    # Plot equivalent single U-tube
    fig = single_u_tube.visualize_pipes()

    # save equivalent
    fig.savefig('coaxial_to_single_equivalent.png')


# Main function
if __name__ == '__main__':
    main()
