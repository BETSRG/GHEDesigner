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

    # Thermal conductivities
    k_p = 0.4  # Pipe thermal conductivity (W/m.K)
    k_s = 2.0  # Ground thermal conductivity (W/m.K)
    k_g = 1.0  # Grout thermal conductivity (W/m.K)

    # Volumetric heat capacities
    rhoCp_p = 1542.0 * 1000.0  # Pipe volumetric heat capacity (J/K.m3)
    rhoCp_s = 2343.493 * 1000.0  # Soil volumetric heat capacity (J/K.m3)
    rhoCp_g = 3901.0 * 1000.0  # Grout volumetric heat capacity (J/K.m3)

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
    fluid = gt.media.Fluid(fluid_str="Water", percent=0.0)
    V_flow_borehole = 0.2  # Volumetric flow rate per borehole (L/s)
    # Total fluid mass flow rate per borehole (kg/s)
    m_flow_borehole = V_flow_borehole / 1000.0 * fluid.rho

    # Define a borehole
    borehole = gt.boreholes.Borehole(H, D, r_b, x=0.0, y=0.0)

    double_u_tube_series = plat.borehole_heat_exchangers.MultipleUTube(
        m_flow_borehole, fluid, borehole, pipe, grout, soil, config="series"
    )

    double_u_tube_parallel = plat.borehole_heat_exchangers.MultipleUTube(
        m_flow_borehole, fluid, borehole, pipe, grout, soil, config="parallel"
    )

    print(double_u_tube_parallel)

    R_b_series = double_u_tube_series.compute_effective_borehole_resistance()
    R_B_parallel = double_u_tube_parallel.compute_effective_borehole_resistance()

    # Intermediate variables
    Re = plat.borehole_heat_exchangers.compute_Reynolds(
        double_u_tube_parallel.m_flow_pipe, r_in, epsilon, fluid
    )

    print("Reynolds number: {}".format(Re))
    R_p = double_u_tube_parallel.R_p
    print("Pipe resistance (K/(W/m)) : {}".format(R_p))
    h_f = double_u_tube_parallel.h_f
    print("Convection coefficient (W/m2.K): {}".format(h_f))
    R_fp = double_u_tube_parallel.R_fp
    print("Convective resistance (K/(W/m)): {}".format(R_fp))

    print("Borehole thermal resistance (series): {0:.4f} m.K/W".format(R_b_series))
    print("Borehole thermal resistance (parallel): {0:.4f} m.K/W".format(R_B_parallel))

    # Create a borehole top view
    fig = double_u_tube_series.visualize_pipes()

    # Save the figure as a png
    fig.savefig("double_u_tube_series.png")

    # Create a borehole top view
    fig = double_u_tube_parallel.visualize_pipes()

    # Save the figure as a png
    fig.savefig("double_u_tube_parallel.png")


# Main function
if __name__ == "__main__":
    main()
