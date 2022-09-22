# Jack C. Cook
# Monday, August 16, 2021

import pygfunction as gt
import ghedt.peak_load_analysis_tool as plat


def main():
    # Borehole dimensions
    H = 100.0  # Borehole length (m)
    D = 2.0  # Borehole buried depth (m)
    r_b = 150.0 / 1000.0 / 2.0  # Borehole radius

    # Pipe dimensions
    r_out = 26.67 / 1000.0 / 2.0  # Pipe outer radius (m)
    r_in = 21.6 / 1000.0 / 2.0  # Pipe inner radius (m)
    s = 32.3 / 1000.0  # Inner-tube to inner-tube Shank spacing (m)
    epsilon = 1.0e-6  # Pipe roughness (m)

    # Pipe positions
    # Single U-tube [(x_in, y_in), (x_out, y_out)]
    pos = plat.media.Pipe.place_pipes(s, r_out, 1)

    # Thermal conductivities
    k_p = 0.4  # Pipe thermal conductivity (W/m.K)
    k_s = 2.0  # Ground thermal conductivity (W/m.K)
    k_g = 1.0  # Grout thermal conductivity (W/m.K)

    # Volumetric heat capacities
    rhoCp_p = 1542.0 * 1000.0  # Pipe volumetric heat capacity (J/K.m3)
    rhoCp_s = 2343.493 * 1000.0  # Soil volumetric heat capacity (J/K.m3)
    rhoCp_g = 3901.0 * 1000.0  # Grout volumetric heat capacity (J/K.m3)

    # Thermal properties
    # Pipe
    pipe = plat.media.Pipe(pos, r_in, r_out, s, epsilon, k_p, rhoCp_p)
    # Soil
    ugt = 18.3  # Undisturbed ground temperature (degrees Celsius)
    soil = plat.media.Soil(k_s, rhoCp_s, ugt)
    # Grout
    grout = plat.media.Grout(k_g, rhoCp_g)

    # Fluid properties
    fluid = gt.media.Fluid(mixer="MEG", percent=0.0)
    V_flow_borehole = 0.2  # Volumetric flow rate per borehole (L/s)
    # Total fluid mass flow rate per borehole (kg/s)
    m_flow_borehole = V_flow_borehole / 1000.0 * fluid.rho

    # Define a borehole
    borehole = gt.boreholes.Borehole(H, D, r_b, x=0.0, y=0.0)

    single_u_tube = plat.borehole_heat_exchangers.SingleUTube(
        m_flow_borehole, fluid, borehole, pipe, grout, soil
    )

    print(single_u_tube)

    # Intermediate variables
    Re = plat.borehole_heat_exchangers.compute_Reynolds(
        single_u_tube.m_flow_pipe, r_in, epsilon, fluid
    )
    print("Reynolds number: {}".format(Re))
    R_p = single_u_tube.R_p
    print("Pipe resistance (K/(W/m)) : {}".format(R_p))
    h_f = single_u_tube.h_f
    print("Convection coefficient (W/m2.K): {}".format(h_f))
    R_fp = single_u_tube.R_fp
    print("Convective resistance (K/(W/m)): {}".format(R_fp))

    R_b = single_u_tube.compute_effective_borehole_resistance()

    print("Borehole thermal resistance: {0:.4f} m.K/W".format(R_b))

    # Create a borehole top view
    fig = single_u_tube.visualize_pipes()

    # Save the figure as a png
    fig.savefig("single_u_tube.png")


# Main function
if __name__ == "__main__":
    main()
