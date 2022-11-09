from json import loads

import pygfunction as gt

from ghedt import geometry, design, search_routines, utilities
from ghedt.peak_load_analysis_tool import media, borehole_heat_exchangers
from .ghe_base_case import GHEBaseTest


class TestDesignFromORNLJsonInput(GHEBaseTest):
    def test_design_from_ornl_json_input(self):
        # Provide path to ORNL parameters file
        file_path = self.test_data_directory / 'ornl_params.json'
        ornl_param_file_data = loads(file_path.read_text())

        # Borehole dimensions
        # -------------------
        h = ornl_param_file_data["max_height"]  # Borehole length (m)
        d = 2.0  # Borehole buried depth (m)
        r_b = ornl_param_file_data["bore_hole_radius"]  # Borehole radius (m)
        b = ornl_param_file_data["B_spacing"]  # Borehole spacing (m)

        # Pipe dimensions
        # ---------------
        # Single and Multiple U-tubes
        # Pipe outer radius (m)
        r_out = ornl_param_file_data["pipe_out_diameter"] / 2.0
        t = ornl_param_file_data["pipe_thickness"]  # Pipe thickness (m)
        r_in = r_out - t  # Pipe inner radius (m)
        # Inner-tube to inner-tube Shank spacing (m)
        s = ornl_param_file_data["u-tube_distance"]
        epsilon = 1.0e-6  # Pipe roughness (m)

        # Pipe positions
        # --------------
        # Single U-tube [(x_in, y_in), (x_out, y_out)]
        pos_single = media.Pipe.place_pipes(s, r_out, 1)
        # Single U-tube BHE object
        single_u_tube = borehole_heat_exchangers.SingleUTube

        # Thermal conductivities
        # ----------------------
        # Pipe thermal conductivity (W/m.K)
        k_p = ornl_param_file_data["pipe_thermal_conductivity"]
        # Ground thermal conductivity (W/m.K)
        k_s = ornl_param_file_data["ground_thermal_conductivity"]
        # Grout thermal conductivity (W/m.K)
        k_g = ornl_param_file_data["grout_thermal_conductivity"]

        # Volumetric heat capacities
        # --------------------------
        rho_cp_p = 1542.0 * 1000.0  # Pipe volumetric heat capacity (J/K.m3)
        # Soil volumetric heat capacity (J/K.m3)
        rho_cp_s = ornl_param_file_data["ground_thermal_heat_capacity"]
        rho_cp_g = 3901.0 * 1000.0  # Grout volumetric heat capacity (J/K.m3)

        # Thermal properties
        # ------------------
        # Pipe
        pipe_single = media.Pipe(pos_single, r_in, r_out, s, epsilon, k_p, rho_cp_p)
        # Soil
        # Undisturbed ground temperature (degrees Celsius)
        ugt = ornl_param_file_data["ground_temperature"]
        soil = media.Soil(k_s, rho_cp_s, ugt)
        # Grout
        grout = media.Grout(k_g, rho_cp_g)

        # Inputs related to fluid
        # -----------------------
        # Fluid properties
        fluid = gt.media.Fluid(fluid_str="Water", percent=0.0)

        # Fluid properties
        # Volumetric flow rate (L/s)
        v_flow = ornl_param_file_data["design_flow_rate"] * 1000.0
        # Note: The flow parameter can be borehole or system.
        flow = "system"

        # Define a borehole
        borehole = gt.boreholes.Borehole(h, d, r_b, x=0.0, y=0.0)

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

        # Process loads from file
        hourly_extraction_ground_loads = self.get_atlanta_loads()

        # Geometric constraints for the `near-square` routine
        geometric_constraints = geometry.GeometricConstraints(b=b, length=300)
        # TODO: length wasn't specified in the line above, but it is needed for near-square design, so I made up 300

        # Single U-tube
        # -------------
        design_single_u_tube = design.DesignNearSquare(
            v_flow,
            borehole,
            single_u_tube,
            fluid,
            pipe_single,
            grout,
            soil,
            sim_params,
            geometric_constraints,
            hourly_extraction_ground_loads,
            flow=flow,
            method=utilities.DesignMethod.Hybrid,
        )

        # Find the near-square design for a single U-tube and size it.
        bisection_search = design_single_u_tube.find_design()
        bisection_search.ghe.compute_g_functions()
        bisection_search.ghe.size(method=utilities.DesignMethod.Hybrid)

        # Export the g-function to a json file
        output_file_path = self.test_outputs_directory / 'ghedt_output_design_from_ornl_json_input.json'
        search_routines.oak_ridge_export(bisection_search, output_file_path)
