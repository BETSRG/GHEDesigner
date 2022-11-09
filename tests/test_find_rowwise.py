# Purpose: Design a constrained RowWise field using the common
# design interface with a single U-tube borehole heat exchanger.

# This search is described in <placeholder>.

import csv
from math import pi
from time import time as clock

import pygfunction as gt

from ghedt import geometry, design, utilities
from ghedt.output import output_design_details
from ghedt.peak_load_analysis_tool import media, borehole_heat_exchangers
from ghedt.rowwise.rowwise_generation import gen_shape
from .ghe_base_case import GHEBaseTest


class TestFindRowWise(GHEBaseTest):
    def test_find_row_wise(self):

        # This file contains two examples utilizing the RowWise design algorithm for a single U tube design.
        # The 1st example doesn't treat perimeter boreholes different, and the second one maintains a perimeter target
        # spacing to interior target-spacing ratio of .8.
        # The results from these examples are exported to the "DesignExampleOutput" folder.

        # W/O Separate Perimeter Spacing Example

        # Output File Configuration
        project_name = "Atlanta Office Building: Design Example"
        note = "RowWise Usage Example w/o Perimeter Spacing: Single U Tube"
        author = "John Doe"
        iteration_name = "Example 5"
        output_file_directory = self.test_outputs_directory / "DesignExampleOutput"

        # Borehole dimensions
        h = 96.0  # Borehole length (m)
        d = 2.0  # Borehole buried depth (m)
        r_b = 0.075  # Borehole radius (m)

        # Single and Multiple U-tube Pipe Dimensions
        r_out = 26.67 / 1000.0 / 2.0  # Pipe outer radius (m)
        r_in = 21.6 / 1000.0 / 2.0  # Pipe inner radius (m)
        s = 32.3 / 1000.0  # Inner-tube to inner-tube Shank spacing (m)
        epsilon = 1.0e-6  # Pipe roughness (m)

        # Single U Tube Pipe Positions
        pos_single = media.Pipe.place_pipes(s, r_out, 1)
        single_u_tube = borehole_heat_exchangers.SingleUTube

        # Thermal conductivities
        k_p = 0.4  # Pipe thermal conductivity (W/m.K)
        k_s = 2.0  # Ground thermal conductivity (W/m.K)
        k_g = 1.0  # Grout thermal conductivity (W/m.K)

        # Volumetric heat capacities
        rho_cp_p = 1542.0 * 1000.0  # Pipe volumetric heat capacity (J/K.m3)
        rho_cp_s = 2343.493 * 1000.0  # Soil volumetric heat capacity (J/K.m3)
        rho_cp_g = 3901.0 * 1000.0  # Grout volumetric heat capacity (J/K.m3)

        # Instantiating Pipe
        pipe_single = media.Pipe(pos_single, r_in, r_out, s, epsilon, k_p, rho_cp_p)

        # Instantiating Soil Properties
        ugt = 18.3  # Undisturbed ground temperature (degrees Celsius)
        soil = media.Soil(k_s, rho_cp_s, ugt)

        # Instantiating Grout Properties
        grout = media.Grout(k_g, rho_cp_g)

        # Fluid properties
        fluid = gt.media.Fluid(fluid_str="Water", percent=0.0)

        # Fluid Flow Properties
        v_flow = 0.2  # Volumetric flow rate (L/s)
        # Note: The flow parameter can be borehole or system.
        flow = "borehole"

        # Instantiate a Borehole
        borehole = gt.boreholes.Borehole(h, d, r_b, x=0.0, y=0.0)

        # Simulation parameters
        start_month = 1
        n_years = 20
        end_month = n_years * 12
        max_eft_allowable = 35  # degrees Celsius (HP EFT)
        min_eft_allowable = 5  # degrees Celsius (HP EFT)
        max_height = 135.0  # in meters  # TODO: This is causing a warning, need to be higher?
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

        # RowWise Design Constraints

        p_spacing = 0.8  # Dimensionless
        spacing_start = 10.0  # in meters
        spacing_stop = 20.0  # in meters
        spacing_step = 0.1  # in meters
        rotate_step = 0.5  # in degrees
        rotate_start = -90.0 * (pi / 180.0)  # in radians
        rotate_stop = 0 * (pi / 180.0)  # in radians

        # Building Description
        property_boundary_file = self.test_data_directory / "polygon_property_boundary.csv"
        no_go_zone_file = self.test_data_directory / "polygon_no_go_zone1.csv"

        prop_a = []  # in meters
        ng_a = []  # in meters

        with open(property_boundary_file, "r", newline="") as pF:
            c_r = csv.reader(pF)
            for line in c_r:
                l_list = []
                for row in line:
                    l_list.append(float(row))
                prop_a.append(l_list)

        # for file in os.listdir(no_go_zone_file):
        with open(no_go_zone_file, "r", newline="") as ngF:
            c_r = csv.reader(ngF)
            ng_a.append([])
            for line in c_r:
                l_list = []
                for row in line:
                    l_list.append(float(row))
                ng_a[-1].append(l_list)

        build_vert, no_go_vert = gen_shape(prop_a, ng_zones=ng_a)

        """ Geometric constraints for the `row-wise` routine:
          - list of vertices for the no-go zones (no_go_vert)
          - perimeter target-spacing to interior target-spacing ratio
          - the lower bound target-spacing (spacing_start)
          - the upper bound target-spacing (spacing_stop)
          - the range around the selected target-spacing over-which to to do an exhaustive search
          - the lower bound rotation (rotateStart)
          - the upper bound rotation (rotateStop)
          - list of vertices for the property boundary (buildVert)
        """
        geometric_constraints = geometry.GeometricConstraints(
            ng_zones=no_go_vert,
            p_spacing=p_spacing,
            spacing_start=spacing_start,
            spacing_stop=spacing_stop,
            spacing_step=spacing_step,
            rotate_start=rotate_start,
            rotate_stop=rotate_stop,
            rotate_step=rotate_step,
            prop_bound=build_vert,
        )

        # Single U-tube
        # -------------
        design_single_u_tube = design.DesignRowWise(
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
            method=utilities.DesignMethod.Hybrid,
            flow=flow,
        )

        # Find the near-square design for a single U-tube and size it.
        tic = clock()  # Clock Start Time
        bisection_search = design_single_u_tube.find_design(
            disp=True, use_perimeter=False
        )  # Finding GHE Design
        bisection_search.ghe.compute_g_functions()  # Calculating G-functions for Chosen Design
        bisection_search.ghe.size(
            method=utilities.DesignMethod.Hybrid
        )  # Calculating the Final Height for the Chosen Design
        toc = clock()  # Clock Stop Time

        # Print Summary of Findings
        subtitle = "* Single U-tube"  # Subtitle for the printed summary
        self.log(subtitle + "\n" + len(subtitle) * "-")
        self.log("Calculation time: {0:.2f} seconds".format(toc - tic))
        self.log("Height: {0:.4f} meters".format(bisection_search.ghe.bhe.b.H))
        nbh = len(bisection_search.ghe.GFunction.bore_locations)
        self.log("Number of boreholes: {}".format(nbh))
        self.log("Total Drilling: {0:.1f} meters\n".format(bisection_search.ghe.bhe.b.H * nbh))

        # Generating Output File
        output_design_details(
            bisection_search,
            toc - tic,
            project_name,
            note,
            author,
            iteration_name,
            output_directory=output_file_directory,
            summary_file="SummaryOfResults_SU_WOP.txt",
            csv_f_1="TimeDependentValues_SU_WOP.csv",
            csv_f_2="BorefieldData_SU_WOP.csv",
            csv_f_3="Loadings_SU_WOP.csv",
            csv_f_4="GFunction_SU_WOP.csv",
            load_method=utilities.DesignMethod.Hybrid,
        )

        # *************************************************************************************************************
        # Perimeter Spacing Example

        note = "RowWise Usage Example w/o Perimeter Spacing: Single U Tube"

        # Single U-tube
        # -------------
        design_single_u_tube = design.DesignRowWise(
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
            method=utilities.DesignMethod.Hybrid,
            flow=flow,
        )

        # Find the near-square design for a single U-tube and size it.
        tic = clock()  # Clock Start Time
        bisection_search = design_single_u_tube.find_design(
            disp=True, use_perimeter=True
        )  # Finding GHE Design
        bisection_search.ghe.compute_g_functions()  # Calculating G-functions for Chosen Design
        bisection_search.ghe.size(
            method=utilities.DesignMethod.Hybrid
        )  # Calculating the Final Height for the Chosen Design
        toc = clock()  # Clock Stop Time

        # Print Summary of Findings
        subtitle = "* Single U-tube"  # Subtitle for the printed summary
        self.log(subtitle + "\n" + len(subtitle) * "-")
        self.log("Calculation time: {0:.2f} seconds".format(toc - tic))
        self.log("Height: {0:.4f} meters".format(bisection_search.ghe.bhe.b.H))
        nbh = len(bisection_search.ghe.GFunction.bore_locations)
        self.log("Number of boreholes: {}".format(nbh))
        self.log("Total Drilling: {0:.1f} meters\n".format(bisection_search.ghe.bhe.b.H * nbh))

        # Generating Output File
        output_design_details(
            bisection_search,
            toc - tic,
            project_name,
            note,
            author,
            iteration_name,
            output_directory=output_file_directory,
            summary_file="SummaryOfResults_SU_WP.txt",
            csv_f_1="TimeDependentValues_SU_WP.csv",
            csv_f_2="BorefieldData_SU_WP.csv",
            csv_f_3="Loadings_SU_WP.csv",
            csv_f_4="GFunction_SU_WP.csv",
            load_method=utilities.DesignMethod.Hybrid,
        )
