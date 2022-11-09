from ghedt.peak_load_analysis_tool import media, equivalance, borehole_heat_exchangers
import pygfunction as gt
from .ghe_base_case import GHEBaseTest


class TestEquivalentPipes(GHEBaseTest):
    def test_equiv_pipes_coaxial_to_single_u_tube(self):
        # Borehole dimensions
        h = 100.0  # Borehole length (m)
        d = 2.0  # Borehole buried depth (m)
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
        rho_cp_p = 1542.0 * 1000.0  # Pipe volumetric heat capacity (J/K.m3)
        rho_cp_s = 2343.493 * 1000.0  # Soil volumetric heat capacity (J/K.m3)
        rho_cp_g = 3901.0 * 1000.0  # Grout volumetric heat capacity (J/K.m3)

        # Thermal properties
        # Pipe
        pipe = media.Pipe(pos, r_inner, r_outer, s, epsilon, k_p, rho_cp_p)
        # Soil
        ugt = 18.3  # Undisturbed ground temperature (degrees Celsius)
        soil = media.Soil(k_s, rho_cp_s, ugt)
        # Grout
        grout = media.Grout(k_g, rho_cp_g)

        # Fluid properties
        fluid = gt.media.Fluid(fluid_str="Water", percent=0.0)
        v_flow_borehole = 0.2  # Volumetric flow rate per borehole (L/s)
        # Total fluid mass flow rate per borehole (kg/s)
        m_flow_borehole = v_flow_borehole / 1000.0 * fluid.rho

        # Define a borehole
        borehole = gt.boreholes.Borehole(h, d, r_b, x=0.0, y=0.0)

        coaxial = borehole_heat_exchangers.CoaxialPipe(
            m_flow_borehole, fluid, borehole, pipe, grout, soil
        )

        var = "Intermediate variables"
        print(var)
        print(len(var) * "-")
        v_fluid, v_pipe, r_conv, r_pipe = equivalance.concentric_tube_volumes(coaxial)
        print("Fluid volume per meter (m^2): {0:.8f}".format(v_fluid))
        print("Pipe volume per meter (m^2): {0:.8f}".format(v_pipe))
        print("Total Convective Resistance (K/(W/m)): {0:.8f}".format(r_conv))
        print("Total Pipe Resistance (K/(W/m)): {0:.8f}".format(r_pipe))
        print("\n")

        single_u_tube = equivalance.compute_equivalent(coaxial)

        val = "Single U-tube equivalent parameters"
        print("\n" + val + "\n" + len(val) * "-")
        print(
            "Fluid volumetric flow rate (L/s): {0:.8f}".format(
                single_u_tube.m_flow_pipe * 1000.0 / single_u_tube.fluid.rho
            )
        )
        print("Radius of inner pipe (m): {0:.8f}".format(single_u_tube.r_in))
        print("Radius of outer pipe (m): {0:.8f}".format(single_u_tube.r_out))
        print("Shank spacing (m): {0:.8f}".format(single_u_tube.pipe.s))
        print("Convection coefficient (): {0:.8f}".format(single_u_tube.h_f))
        print("Pipe thermal conductivity (W/m.K): {0:.8f}".format(single_u_tube.pipe.k))
        print("Grout thermal conductivity (W/m.K): {0:.8f}".format(single_u_tube.grout.k))

        rb = single_u_tube.compute_effective_borehole_resistance()
        print("Effective borehole resistance (m.K/W): {0:.8f}".format(rb))

        print(single_u_tube.__repr__())

        # Plot equivalent single U-tube
        fig = single_u_tube.visualize_pipes()

        # save equivalent
        output_plot = self.test_outputs_directory / 'coaxial_to_single_equivalent.png'
        fig.savefig(str(output_plot))

    def test_equiv_pipes_double_to_single_u_tube(self):
        # Borehole dimensions
        h = 100.0  # Borehole length (m)
        d = 2.0  # Borehole buried depth (m)
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
        rho_cp_p = 1542.0 * 1000.0  # Pipe volumetric heat capacity (J/K.m3)
        rho_cp_s = 2343.493 * 1000.0  # Soil volumetric heat capacity (J/K.m3)
        rho_cp_g = 3901.0 * 1000.0  # Grout volumetric heat capacity (J/K.m3)

        # Pipe positions
        # Double U-tube [(x_in, y_in), (x_out, y_out), (x_in, y_in), (x_out, y_out)]
        pos = media.Pipe.place_pipes(s, r_out, 2)

        # Thermal properties
        # Pipe
        pipe = media.Pipe(pos, r_in, r_out, s, epsilon, k_p, rho_cp_p)
        # Soil
        ugt = 18.3  # Undisturbed ground temperature (degrees Celsius)
        soil = media.Soil(k_s, rho_cp_s, ugt)
        # Grout
        grout = media.Grout(k_g, rho_cp_g)

        # Fluid properties
        fluid = gt.media.Fluid(fluid_str="Water", percent=0.0)
        v_flow_borehole = 0.2  # Volumetric flow rate per borehole (L/s)
        # Total fluid mass flow rate per borehole (kg/s)
        m_flow_borehole = v_flow_borehole / 1000.0 * fluid.rho

        # Define a borehole
        borehole = gt.boreholes.Borehole(h, d, r_b, x=0.0, y=0.0)

        # Double U-tube defaults to parallel
        double_u_tube = borehole_heat_exchangers.MultipleUTube(
            m_flow_borehole, fluid, borehole, pipe, grout, soil
        )

        val = "Intermediate variables"
        print(val + "\n" + len(val) * "-")
        # Intermediate variables
        v_fluid, v_pipe, r_conv, r_pipe = equivalance.u_tube_volumes(double_u_tube)
        print("Fluid volume per meter (m^2): {0:.8f}".format(v_fluid))
        print("Pipe volume per meter (m^2): {0:.8f}".format(v_pipe))
        print("Total Convective Resistance (K/(W/m)): {0:.8f}".format(r_conv))
        print("Total Pipe Resistance (K/(W/m)): {0:.8f}".format(r_pipe))

        single_u_tube = equivalance.compute_equivalent(double_u_tube)
        val = "Single U-tube equivalent parameters"
        print("\n" + val + "\n" + len(val) * "-")
        print(
            "Fluid volumetric flow rate (L/s): {0:.8f}".format(
                single_u_tube.m_flow_pipe * 1000.0 / single_u_tube.fluid.rho
            )
        )
        print("Radius of inner pipe (m): {0:.8f}".format(single_u_tube.r_in))
        print("Radius of outer pipe (m): {0:.8f}".format(single_u_tube.r_out))
        print("Shank spacing (m): {0:.8f}".format(single_u_tube.pipe.s))
        print("Convection coefficient (): {0:.8f}".format(single_u_tube.h_f))
        print("Pipe thermal conductivity (W/m.K): {0:.8f}".format(single_u_tube.pipe.k))
        print("Grout thermal conductivity (W/m.K): {0:.8f}".format(single_u_tube.grout.k))

        rb = single_u_tube.compute_effective_borehole_resistance()
        print("Effective borehole resistance (m.K/W): {0:.8f}".format(rb))

        print(single_u_tube.__repr__())

        # Plot equivalent single U-tube
        fig = single_u_tube.visualize_pipes()

        # save equivalent
        output_plot = self.test_outputs_directory / 'double_to_single_equivalent.png'
        fig.savefig(str(output_plot))
