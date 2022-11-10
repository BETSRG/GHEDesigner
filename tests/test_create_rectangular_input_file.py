from ghedesigner import design, geometry, utilities, borehole_heat_exchangers, media
from ghedesigner.borehole import GHEBorehole
from ghedesigner.fluid import GHEFluid
from .ghe_base_case import GHEBaseTest


class TestCreateRectangularInputFile(GHEBaseTest):
    def test_create_rectangular_input_file(self):
        # Borehole dimensions
        # -------------------
        h = 96.0  # Borehole length (m)
        d = 2.0  # Borehole buried depth (m)
        r_b = 0.075  # Borehole radius (m)
        # B = 5.0  # Borehole spacing (m)

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
        # Outer pipe radii
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
        single_u_tube = borehole_heat_exchangers.SingleUTube
        # Double U-tube
        # pos_double = plat.media.Pipe.place_pipes(s, r_out, 2)
        # double_u_tube = plat.borehole_heat_exchangers.MultipleUTube
        # Coaxial tube
        # pos_coaxial = (0, 0)
        # coaxial_tube = plat.borehole_heat_exchangers.CoaxialPipe

        # Thermal conductivities
        # ----------------------
        k_p = 0.4  # Pipe thermal conductivity (W/m.K)
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
        pipe_single = media.Pipe(pos_single, r_in, r_out, s, epsilon, k_p, rho_cp_p)
        # pipe_double = plat.media.Pipe(pos_double, r_in, r_out, s, epsilon, k_p, rhoCp_p)
        # pipe_coaxial = plat.media.Pipe(
        #     pos_coaxial, r_inner, r_outer, 0, epsilon, k_p, rhoCp_p
        # )
        # Soil
        ugt = 18.3  # Undisturbed ground temperature (degrees Celsius)
        soil = media.Soil(k_s, rho_cp_s, ugt)
        # Grout
        grout = media.Grout(k_g, rho_cp_g)

        # Inputs related to fluid
        # -----------------------
        # Fluid properties
        fluid = GHEFluid(fluid_str="Water", percent=0.0)

        # Fluid properties
        v_flow_borehole = 0.2  # Borehole volumetric flow rate (L/s)

        # Define a borehole
        borehole = GHEBorehole(h, d, r_b, x=0.0, y=0.0)

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
        sim_params = media.SimulationParameters(
            start_month,
            end_month,
            max_eft_allowable,
            min_eft_allowable,
            max_height,
            min_height,
        )

        hourly_extraction_ground_loads = self.get_atlanta_loads()

        # Rectangular design constraints are the land and range of B-spacing
        length = 85.0  # m
        width = 36.5  # m
        b_min = 3.0  # m
        b_max = 10.0  # m

        # Geometric constraints for the `near-square` routine
        geometric_constraints = geometry.GeometricConstraints(
            length=length, width=width, b_min=b_min, b_max_x=b_max  # , unconstrained=False
        )

        # Note: Flow functionality is currently only on a borehole basis. Future
        # development will include the ability to change the flow rate to be on a
        # system flow rate basis.
        design_single_u_tube = design.DesignRectangle(
            v_flow_borehole,
            borehole,
            single_u_tube,
            fluid,
            pipe_single,
            grout,
            soil,
            sim_params,
            geometric_constraints,
            hourly_extraction_ground_loads,
            utilities.DesignMethod.Hybrid,
        )

        # Output the design interface object to a json file, so it can be reused
        input_file_path = self.test_data_directory / 'ghedt_input_rectangular.obj'
        utilities.create_input_file(design_single_u_tube, input_file_path)
