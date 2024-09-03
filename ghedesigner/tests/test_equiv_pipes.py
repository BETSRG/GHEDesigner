from ghedesigner.borehole import GHEBorehole
from ghedesigner.borehole_heat_exchangers import CoaxialPipe, MultipleUTube
from ghedesigner.media import GHEFluid, Grout, Pipe, Soil
from ghedesigner.tests.test_base_case import GHEBaseTest


class TestEquivalentPipes(GHEBaseTest):
    def test_equiv_pipes_coaxial_to_single_u_tube(self):
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

        # Pipe
        pos = (0, 0)
        s = 0
        k_pipe_inner = 0.4  # Inner pipe thermal conductivity (W/m.K)
        k_pipe_outer = 0.4  # Outer pipe thermal conductivity (W/m.K)
        rho_cp_p = 1542000.0  # Pipe volumetric heat capacity (J/K.m3)
        pipe = Pipe(pos, r_inner, r_outer, s, epsilon, (k_pipe_inner, k_pipe_outer), rho_cp_p)

        # Soil
        k_s = 2.0  # Ground thermal conductivity (W/m.K)
        rho_cp_s = 2343493.0  # Soil volumetric heat capacity (J/K.m3)
        ugt = 18.3  # Undisturbed ground temperature (degrees Celsius)
        soil = Soil(k_s, rho_cp_s, ugt)

        # Grout
        k_g = 1.0  # Grout thermal conductivity (W/m.K)
        rho_cp_g = 3901000.0  # Grout volumetric heat capacity (J/K.m3)
        grout = Grout(k_g, rho_cp_g)

        # Fluid properties
        fluid = GHEFluid(fluid_str="Water", percent=0.0)
        v_flow_borehole = 0.8  # Volumetric flow rate per borehole (L/s)
        # Total fluid mass flow rate per borehole (kg/s)
        m_flow_borehole = v_flow_borehole / 1000.0 * fluid.rho

        # Define a borehole
        h = 100.0  # Borehole length (m)
        d = 2.0  # Borehole buried depth (m)
        dia = 140.0 / 1000.0  # Borehole diameter
        borehole = GHEBorehole(h, d, dia / 2.0, x=0.0, y=0.0)

        # borehole heat exchanger
        coaxial = CoaxialPipe(m_flow_borehole, fluid, borehole, pipe, grout, soil)

        var = "Intermediate variables"
        self.log(var)
        self.log(len(var) * "-")
        v_fluid, v_pipe, r_conv, r_pipe = coaxial.concentric_tube_volumes()
        self.log(f"Fluid volume per meter (m^2): {v_fluid:0.8f}")
        self.log(f"Pipe volume per meter (m^2): {v_pipe:0.8f}")
        self.log(f"Total Convective Resistance (K/(W/m)): {r_conv:0.8f}")
        self.log(f"Total Pipe Resistance (K/(W/m)): {r_pipe:0.8f}")
        self.log("\n")

        single_u_tube = coaxial.to_single()

        val = "Single U-tube equivalent parameters"
        self.log("\n" + val + "\n" + len(val) * "-")
        self.log(
            f"Fluid volumetric flow rate (L/s): "
            f"{single_u_tube.m_flow_borehole * 1000.0 / single_u_tube.fluid.rho:0.8f}"
        )
        self.log(f"Diameter of inner pipe (m): {single_u_tube.r_in * 2.0:0.8f}")
        self.log(f"Diameter of outer pipe (m): {single_u_tube.r_out * 2.0:0.8f}")
        self.log(f"Shank spacing (m): {single_u_tube.pipe.s:0.8f}")
        self.log(f"Convection coefficient (): {single_u_tube.h_f:08f}")
        self.log(f"Pipe thermal conductivity (W/m.K): {single_u_tube.pipe.k:0.8f}")
        self.log(f"Grout thermal conductivity (W/m.K): {single_u_tube.grout.k:0.8f}")

        rb = single_u_tube.calc_effective_borehole_resistance()
        self.log(f"Effective borehole resistance (m.K/W): {rb:0.8f}")

        self.log(single_u_tube.as_dict())

    def test_equiv_pipes_double_to_single_u_tube(self):
        # Borehole dimensions
        h = 100.0  # Borehole length (m)
        d = 2.0  # Borehole buried depth (m)
        dia = 140.0 / 1000.0  # Borehole diameter

        # Pipe dimensions
        d_out = 0.04216  # Pipe outer diameter (m)
        d_in = 0.03404  # Pipe inner diameter (m)
        r_out = d_out / 2.0
        r_in = d_in / 2.0
        s = 0.01856  # Inner-tube to inner-tube Shank spacing (m)
        epsilon = 1.0e-6  # Pipe roughness (m)

        # Thermal conductivities
        k_p = 0.4  # Pipe thermal conductivity (W/m.K)
        k_s = 2.0  # Ground thermal conductivity (W/m.K)
        k_g = 1.0  # Grout thermal conductivity (W/m.K)

        # Volumetric heat capacities
        rho_cp_p = 1542000.0  # Pipe volumetric heat capacity (J/K.m3)
        rho_cp_s = 2343493.0  # Soil volumetric heat capacity (J/K.m3)
        rho_cp_g = 3901000.0  # Grout volumetric heat capacity (J/K.m3)

        # Pipe positions
        # Double U-tube [(x_in, y_in), (x_out, y_out), (x_in, y_in), (x_out, y_out)]
        pos = Pipe.place_pipes(s, r_out, 2)

        # Thermal properties
        # Pipe
        pipe = Pipe(pos, r_in, r_out, s, epsilon, k_p, rho_cp_p)
        # Soil
        ugt = 18.3  # Undisturbed ground temperature (degrees Celsius)
        soil = Soil(k_s, rho_cp_s, ugt)
        # Grout
        grout = Grout(k_g, rho_cp_g)

        # Fluid properties
        fluid = GHEFluid(fluid_str="Water", percent=0.0)
        v_flow_borehole = 0.5  # Volumetric flow rate per borehole (L/s)
        # Total fluid mass flow rate per borehole (kg/s)
        m_flow_borehole = v_flow_borehole / 1000.0 * fluid.rho

        # Define a borehole
        borehole = GHEBorehole(h, d, dia / 2.0, x=0.0, y=0.0)

        # Double U-tube defaults to parallel
        double_u_tube = MultipleUTube(m_flow_borehole, fluid, borehole, pipe, grout, soil)

        val = "Intermediate variables"
        self.log(val + "\n" + len(val) * "-")
        # Intermediate variables
        v_fluid, v_pipe, r_conv, r_pipe = double_u_tube.u_tube_volumes()
        self.log(f"Fluid volume per meter (m^2): {v_fluid:0.8f}")
        self.log(f"Pipe volume per meter (m^2): {v_pipe:0.8f}")
        self.log(f"Total Convective Resistance (K/(W/m)): {r_conv:0.8f}")
        self.log(f"Total Pipe Resistance (K/(W/m)): {r_pipe:0.8f}")

        single_u_tube = double_u_tube.to_single()
        val = "Single U-tube equivalent parameters"
        self.log("\n" + val + "\n" + len(val) * "-")
        self.log(
            f"Fluid volumetric flow rate (L/s): "
            f"{single_u_tube.m_flow_borehole * 1000.0 / single_u_tube.fluid.rho:0.8f}"
        )
        self.log(f"Diameter of inner pipe (m): {single_u_tube.r_in * 2.0:0.8f}")
        self.log(f"Diameter of outer pipe (m): {single_u_tube.r_out * 2.0:0.8f}")
        self.log(f"Shank spacing (m): {single_u_tube.pipe.s:0.8f}")
        self.log(f"Convection coefficient (-): {single_u_tube.h_f:0.8f}")
        self.log(f"Pipe thermal conductivity (W/m.K): {single_u_tube.pipe.k:0.8f}")
        self.log(f"Grout thermal conductivity (W/m.K): {single_u_tube.grout.k:0.8f}")

        rb = single_u_tube.calc_effective_borehole_resistance()
        self.log(f"Effective borehole resistance (m.K/W): {rb:0.8f}")

        self.log(single_u_tube.as_dict())
