from ghedesigner.borehole import GHEBorehole
from ghedesigner.borehole_heat_exchangers import SingleUTube, MultipleUTube, CoaxialPipe
from ghedesigner.design import DesignNearSquare
from ghedesigner.geometry import GeometricConstraintsNearSquare
from ghedesigner.media import Pipe, Soil, Grout, GHEFluid
from ghedesigner.simulation import SimulationParameters
from ghedesigner.tests.ghe_base_case import GHEBaseTest
from ghedesigner.utilities import DesignMethodTimeStep, create_input_file, read_input_file


class TestCreateNearSquareInputFile(GHEBaseTest):
    def test_create_near_square_input_file(self):
        # Borehole dimensions
        # -------------------
        h = 96.0  # Borehole length (m)
        d = 2.0  # Borehole buried depth (m)
        r_b = 0.075  # Borehole radius (m)
        b = 5.0  # Borehole spacing (m)

        # Pipe dimensions
        # ---------------
        # Single and Multiple U-tubes
        r_out = 26.67 / 1000.0 / 2.0  # Pipe outer radius (m)
        r_in = 21.6 / 1000.0 / 2.0  # Pipe inner radius (m)
        s = 32.3 / 1000.0  # Inner-tube to inner-tube Shank spacing (m)
        epsilon = 1.0e-6  # Pipe roughness (m)
        # Coaxial tube
        r_in_in = 44.2 / 1000.0 / 2.0
        r_in_out = 50.0 / 1000.0 / 2.0
        # Outer pipe radii
        r_out_in = 97.4 / 1000.0 / 2.0
        r_out_out = 110.0 / 1000.0 / 2.0
        # Pipe radii
        # Note: This convention is different from pygfunction
        r_inner = [r_in_in, r_in_out]  # The radii of the inner pipe from in to out
        r_outer = [r_out_in, r_out_out]  # The radii of the outer pipe from in to out

        # Pipe positions
        # --------------
        # Single U-tube [(x_in, y_in), (x_out, y_out)]
        pos_single = Pipe.place_pipes(s, r_out, 1)
        # Single U-tube BHE object
        single_u_tube = SingleUTube
        # Double U-tube
        pos_double = Pipe.place_pipes(s, r_out, 2)
        double_u_tube = MultipleUTube
        # Coaxial tube
        pos_coaxial = (0, 0)
        coaxial_tube = CoaxialPipe

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
        pipe_single = Pipe(pos_single, r_in, r_out, s, epsilon, k_p, rho_cp_p)
        pipe_double = Pipe(pos_double, r_in, r_out, s, epsilon, k_p, rho_cp_p)
        pipe_coaxial = Pipe(
            pos_coaxial, r_inner, r_outer, 0, epsilon, k_p, rho_cp_p
        )
        # Soil
        ugt = 18.3  # Undisturbed ground temperature (degrees Celsius)
        soil = Soil(k_s, rho_cp_s, ugt)
        # Grout
        grout = Grout(k_g, rho_cp_g)

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
        sim_params = SimulationParameters(
            start_month,
            end_month,
            max_eft_allowable,
            min_eft_allowable,
            max_height,
            min_height,
        )

        hourly_extraction_ground_loads = self.get_atlanta_loads()

        # Geometric constraints for the `near-square` routine
        geometric_constraints = GeometricConstraintsNearSquare(b, 300)  # , unconstrained=True)
        # TODO: b and length were not specified above, so I made up 5 and 300

        # Note: Flow functionality is currently only on a borehole basis. Future
        # development will include the ability to change the flow rate to be on a
        # system flow rate basis.

        # Single U-tube
        # -------------
        design_single_u_tube = DesignNearSquare(
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
            DesignMethodTimeStep.Hybrid,
        )

        # Output the design interface object to a json file, so it can be reused
        input_file_path = self.test_outputs_directory / 'ghedt_input_near_square_single_u_tube.obj'
        create_input_file(design_single_u_tube, input_file_path)

        # Double U-tube
        # -------------
        design_double_u_tube = DesignNearSquare(
            v_flow_borehole,
            borehole,
            double_u_tube,
            fluid,
            pipe_double,
            grout,
            soil,
            sim_params,
            geometric_constraints,
            hourly_extraction_ground_loads,
            DesignMethodTimeStep.Hybrid,
        )

        input_file_path = self.test_outputs_directory / 'ghedt_input_near_square_double_u_tube.obj'
        create_input_file(design_double_u_tube, input_file_path)

        # Coaxial tube
        # ------------
        design_coaxial_u_tube = DesignNearSquare(
            v_flow_borehole,
            borehole,
            coaxial_tube,
            fluid,
            pipe_coaxial,
            grout,
            soil,
            sim_params,
            geometric_constraints,
            hourly_extraction_ground_loads,
            DesignMethodTimeStep.Hybrid,
        )

        input_file_path = self.test_outputs_directory / 'ghedt_input_near_square_coaxial_tube.obj'
        create_input_file(design_coaxial_u_tube, input_file_path)

        # Test reading the input file back in now
        #
        # Enter the path to the input file named `ghedt_input.obj` which was created above
        path_to_file = self.test_outputs_directory / 'ghedt_input.obj'
        # Initialize a Design object that is based on the content of the
        # path_to_file variable.
        _design = read_input_file(path_to_file)
        # Find the design based on the inputs.
        bisection_search = _design.find_design()
        # Perform sizing in between the min and max bounds.
        ghe = bisection_search.ghe
        ghe.compute_g_functions()
        ghe.size(method=DesignMethodTimeStep.Hybrid)
        # Export the g-function to a json file
        output_file = self.test_outputs_directory / "ghedt_output_from_input.json"
        bisection_search.oak_ridge_export(output_file)
