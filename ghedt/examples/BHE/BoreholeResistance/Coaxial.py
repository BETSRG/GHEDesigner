import pygfunction as gt
import ghedt.peak_load_analysis_tool as plat
import matplotlib.pyplot as plt


def main():
    # Borehole dimensions
    H = 100.0  # Borehole length (m)
    D = 2.0  # Borehole buried depth (m)
    r_b = 150.0 / 1000.0 / 2.0  # Borehole radius

    # Pipe dimensions
    # Inner pipe radii
    r_in_in = 44.2 / 1000.0 / 2.0
    r_in_out = 50.0 / 1000.0 / 2.0
    # Outer pipe radii
    r_out_in = 97.4 / 1000.0 / 2.0
    r_out_out = 110.0 / 1000.0 / 2.0
    # Pipe radii
    # Note: This convention is different from pygfunction
    r_inner = [r_in_in, r_in_out]  # The radii of the inner pipe from in to out
    r_outer = [r_out_in, r_out_out]  # The radii of the outer pipe from in to out
    epsilon = 1.0e-6  # Pipe roughness (m)

    # Pipe positioning
    pos = (0, 0)
    s = 0

    # Thermal properties
    k_p = [0.4, 0.4]  # Inner and outer pipe thermal conductivity (W/m.K)
    k_s = 2.0  # Ground thermal conductivity (W/m.K)
    k_g = 1.0  # Grout thermal conductivity (W/m.K)

    # Volumetric heat capacities
    rhoCp_p = 1542.0 * 1000.0  # Pipe volumetric heat capacity (J/K.m3)
    rhoCp_s = 2343.493 * 1000.0  # Soil volumetric heat capacity (J/K.m3)
    rhoCp_g = 3901.0 * 1000.0  # Grout volumetric heat capacity (J/K.m3)

    # Thermal properties
    # Pipe
    pipe = plat.media.Pipe(pos, r_inner, r_outer, s, epsilon, k_p, rhoCp_p)
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

    Coaxial = plat.borehole_heat_exchangers.CoaxialPipe(
        m_flow_borehole, fluid, borehole, pipe, grout, soil
    )

    print(Coaxial)

    R_b = Coaxial.compute_effective_borehole_resistance()

    val = "Intermediate variables"
    print(val + "\n" + len(val) * "-")
    Re = plat.borehole_heat_exchangers.compute_reynolds(
        Coaxial.m_flow_pipe, Coaxial.pipe.r_out[1], epsilon, fluid
    )
    print("Reynolds number: {}".format(Re))
    # R_p = Coaxial.R_p
    # print('Pipe resistance (K/(W/m)) : {}'.format(R_p))
    h_f = Coaxial.h_fluid_a_out
    print("Convection coefficient (W/m2.K): {}".format(h_f))
    R_fp = Coaxial.R_fp
    print("Convective resistance (K/(W/m)): {}".format(R_fp))
    print("Borehole thermal resistance: {0:4f} m.K/W".format(R_b))

    fig = Coaxial.visualize_pipes()

    fig.savefig("Coaxial.png")

    plt.close(fig)


# Main function
if __name__ == "__main__":
    main()
