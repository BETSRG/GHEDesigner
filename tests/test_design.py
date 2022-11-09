import copy

import pygfunction as gt

from ghedt import design, utilities, geometry
from ghedt.peak_load_analysis_tool import borehole_heat_exchangers, media
from .ghe_base_case import GHEBaseTest


class TestNearSquare(GHEBaseTest):

    def test_design_selection(self):
        # Single U-tube
        # -------------
        # Design a single U-tube with a system volumetric flow rate

        # Borehole dimensions
        # -------------------
        h = 96.0  # Borehole length (m)
        d = 2.0  # Borehole buried depth (m)
        r_b = 0.075  # Borehole radius (m)

        # Pipe dimensions
        # ---------------
        # Single and Multiple U-tubes
        r_out = 26.67 / 1000.0 / 2.0  # Pipe outer radius (m)
        r_in = 21.6 / 1000.0 / 2.0  # Pipe inner radius (m)
        s = 32.3 / 1000.0  # Inner-tube to inner-tube Shank spacing (m)
        epsilon = 1.0e-6  # Pipe roughness (m)
        # Coaxial tube
        # r_in_in = 44.2 / 1000.0 / 2.0
        # r_in_out = 50.0 / 1000.0 / 2.0
        # # Outer pipe radii
        # r_out_in = 97.4 / 1000.0 / 2.0
        # r_out_out = 110.0 / 1000.0 / 2.0
        # Pipe radii
        # Note: This convention is different from pygfunction
        # r_inner = [r_in_in, r_in_out]  # The radii of the inner pipe from in to out
        # r_outer = [r_out_in, r_out_out]  # The radii of the outer pipe from in to out

        # Pipe positions
        # --------------
        # Single U-tube [(x_in, y_in), (x_out, y_out)]
        pos_single = media.Pipe.place_pipes(s, r_out, 1)
        # Single U-tube BHE object
        self.single_u_tube = borehole_heat_exchangers.SingleUTube
        # Double U-tube
        # pos_double = plat.media.Pipe.place_pipes(s, r_out, 2)
        # double_u_tube = plat.borehole_heat_exchangers.MultipleUTube
        # Coaxial tube
        # pos_coaxial = (0, 0)
        # coaxial_tube = plat.borehole_heat_exchangers.CoaxialPipe

        # Thermal conductivities
        # ----------------------
        k_p = 0.4  # Pipe thermal conductivity (W/m.K)
        # k_p_coax = [0.4, 0.4]  # Pipes thermal conductivity (W/m.K)
        k_s = 2.0  # Ground thermal conductivity (W/m.K)
        k_g = 1.0  # Grout thermal conductivity (W/m.K)

        # Volumetric heat capacities
        # --------------------------
        rho_cp_p = 1542.0 * 1000.0  # Pipe volumetric heat capacity (J/K.m3)
        rho_cp_s = 2343.493 * 1000.0  # Soil volumetric heat capacity (J/K.m3)
        rho_cp_g = 3901.0 * 1000.0  # Grout volumetric heat capacity (J/K.m3)

        # Thermal properties
        # ------------------
        # Pipe
        self.pipe_single = media.Pipe(
            pos_single, r_in, r_out, s, epsilon, k_p, rho_cp_p
        )
        # pipe_double = plat.media.Pipe(pos_double, r_in, r_out, s, epsilon, k_p, rhoCp_p)
        # pipe_coaxial = plat.media.Pipe(pos_coaxial, r_inner, r_outer, 0, epsilon, k_p_coax, rhoCp_p)
        # Soil
        ugt = 18.3  # Undisturbed ground temperature (degrees Celsius)
        self.soil = media.Soil(k_s, rho_cp_s, ugt)
        # Grout
        self.grout = media.Grout(k_g, rho_cp_g)

        # Inputs related to fluid
        # -----------------------
        # Fluid properties
        self.fluid = gt.media.Fluid(fluid_str="Water", percent=0.0)

        # Fluid flow rate
        v_flow = 0.2  # Borehole volumetric flow rate (L/s)
        self.V_flow_borehole = copy.deepcopy(v_flow)

        # Define a borehole
        self.borehole = gt.boreholes.Borehole(h, d, r_b, x=0.0, y=0.0)

        # Simulation parameters
        # ---------------------
        # Simulation start month and end month
        start_month = 1
        n_years = 20
        end_month = n_years * 12
        # Maximum and minimum allowable fluid temperatures
        max_eft_allowable = 35  # degrees Celsius
        min_eft_allowable = 5  # degrees Celsius
        # Maximum and minimum allowable heights
        max_height = 135.0  # in meters
        min_height = 60  # in meters
        self.sim_params = media.SimulationParameters(
            start_month,
            end_month,
            max_eft_allowable,
            min_eft_allowable,
            max_height,
            min_height,
        )

        # Note: Based on these inputs, the resulting near-square test will
        # determine a system with 156 boreholes.
        self.V_flow_system = self.V_flow_borehole * 156.0

        # Process loads from file
        self.hourly_extraction_ground_loads = self.get_atlanta_loads()

        # Geometric constraints for the `near-square` routine
        # Required geometric constraints for the uniform rectangle design: B
        b = 5.0  # Borehole spacing (m)
        number_of_boreholes = 32
        length = utilities.length_of_side(number_of_boreholes, b)
        self.geometric_constraints = geometry.GeometricConstraints(b=b, length=length)

        design_single_u_tube_a = design.DesignNearSquare(
            self.V_flow_system,
            self.borehole,
            self.single_u_tube,
            self.fluid,
            self.pipe_single,
            self.grout,
            self.soil,
            self.sim_params,
            self.geometric_constraints,
            self.hourly_extraction_ground_loads,
            flow="system",
        )
        # Find the near-square design for a single U-tube and size it.
        bisection_search = design_single_u_tube_a.find_design()
        bisection_search.ghe.compute_g_functions()
        bisection_search.ghe.size(method="hybrid")
        h_single_u_tube_a = bisection_search.ghe.bhe.b.H

        design_single_u_tube_b = design.DesignNearSquare(
            self.V_flow_borehole,
            self.borehole,
            self.single_u_tube,
            self.fluid,
            self.pipe_single,
            self.grout,
            self.soil,
            self.sim_params,
            self.geometric_constraints,
            self.hourly_extraction_ground_loads,
            flow="borehole",
        )
        # Find the near-square design for a single U-tube and size it.
        bisection_search = design_single_u_tube_b.find_design()
        bisection_search.ghe.compute_g_functions()
        bisection_search.ghe.size(method="hybrid")
        h_single_u_tube_b = bisection_search.ghe.bhe.b.H

        # Verify that the `flow` toggle is properly working
        self.assertAlmostEqual(h_single_u_tube_a, h_single_u_tube_b, places=8)
        # Verify that the proper height as been found
        # Note: This reference was calculated on macOS. It seems that on Linux
        # the values are not equal starting around the 9th decimal place.
        h_reference = 130.27
        self.assertAlmostEqual(h_reference, h_single_u_tube_a, delta=0.01)
