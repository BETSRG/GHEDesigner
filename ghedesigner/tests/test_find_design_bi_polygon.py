# Purpose: Design a bi-uniform constrained polygonal field using the common
# design interface with a single U-tube, multiple U-tube and coaxial tube
# borehole heat exchanger.

# This search is described in section 4.4.5 from pages 146-148 in Cook (2021).

import csv
from time import time as clock

from ghedesigner.borehole import GHEBorehole
from ghedesigner.borehole_heat_exchangers import SingleUTube, MultipleUTube, CoaxialPipe
from ghedesigner.design import DesignBiRectangleConstrained
from ghedesigner.geometry import GeometricConstraintsBiRectangleConstrained
from ghedesigner.media import Pipe, Soil, Grout, GHEFluid
from ghedesigner.simulation import SimulationParameters
from ghedesigner.output import OutputManager
from ghedesigner.tests.ghe_base_case import GHEBaseTest
from ghedesigner.utilities import DesignMethodTimeStep


class TestFindBiPolygonDesign(GHEBaseTest):

    def test_find_bi_polygon_design(self):

        # This file contains 3 examples utilizing the bi-uniform polygonal design algorithm for a single U, double U,
        # and coaxial tube  The results from these examples are exported to the "DesignExampleOutput" folder.

        # Single U-tube Example

        # Output File Configuration
        project_name = "Atlanta Office Building: Design Example"
        note = "Bi-Uniform Polygon Usage Example: Single U Tube"
        author = "Jane Doe"
        iteration_name = "Example 6"
        output_file_directory = self.test_outputs_directory / "TestFindBiPolygonDesign"

        # Borehole dimensions
        h = 96.0  # Borehole length (m)
        d = 2.0  # Borehole buried depth (m)
        r_b = 0.075  # Borehole radius (m)
        # B = 5.0  # Borehole spacing (m)

        # Single and Multiple U-tube Pipe Dimensions
        r_out = 0.013335  # Pipe outer radius (m)
        r_in = 0.0108  # Pipe inner radius (m)
        s = 0.0323  # Inner-tube to inner-tube Shank spacing (m)
        epsilon = 1.0e-6  # Pipe roughness (m)

        # Single U Tube Pipe Positions
        pos_single = Pipe.place_pipes(s, r_out, 1)
        single_u_tube = SingleUTube

        # Thermal conductivities
        k_p = 0.4  # Pipe thermal conductivity (W/m.K)
        k_s = 2.0  # Ground thermal conductivity (W/m.K)
        k_g = 1.0  # Grout thermal conductivity (W/m.K)

        # Volumetric heat capacities
        rho_cp_p = 1542000.0  # Pipe volumetric heat capacity (J/K.m3)
        rho_cp_s = 2343493.0  # Soil volumetric heat capacity (J/K.m3)
        rho_cp_g = 3901000.0  # Grout volumetric heat capacity (J/K.m3)

        # Instantiating Pipe
        pipe_single = Pipe(pos_single, r_in, r_out, s, epsilon, k_p, rho_cp_p)

        # Instantiating Soil Properties
        ugt = 18.3  # Undisturbed ground temperature (degrees Celsius)
        soil = Soil(k_s, rho_cp_s, ugt)

        # Instantiating Grout Properties
        grout = Grout(k_g, rho_cp_g)

        # Fluid properties
        fluid = GHEFluid(fluid_str="Water", percent=0.0)

        # Fluid Flow Properties
        v_flow = 0.2  # Volumetric flow rate (L/s)
        # Note: The flow parameter can be borehole or system.
        flow = "borehole"

        # Instantiate a Borehole
        borehole = GHEBorehole(h, d, r_b, x=0.0, y=0.0)

        # Simulation parameters
        start_month = 1
        n_years = 20
        end_month = n_years * 12
        max_eft_allowable = 35  # degrees Celsius (HP_EFT)
        min_eft_allowable = 5  # degrees Celsius (HP_EFT)
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

        # Process loads from file
        hourly_extraction_ground_loads = self.get_atlanta_loads()

        # Polygonal design constraints are the land and range of B-spacing
        b_min = 5  # in m
        b_max_x = 25  # in m
        b_max_y = b_max_x  # in m

        # Building Description
        property_boundary_file = self.test_data_directory / "polygon_property_boundary.csv"
        no_go_zone_file = self.test_data_directory / "polygon_no_go_zone1.csv"

        prop_a = []  # in meters
        ng_a = []  # in meters

        with open(property_boundary_file, "r", newline="") as pF:
            c_r = csv.reader(pF)
            for line in c_r:
                l_prop_a = []
                for row in line:
                    l_prop_a.append(float(row))
                prop_a.append(l_prop_a)

        with open(no_go_zone_file, "r", newline="") as ngF:
            c_r = csv.reader(ngF)
            ng_a.append([])
            for line in c_r:
                l_prop_a = []
                for row in line:
                    l_prop_a.append(float(row))
                ng_a[-1].append(l_prop_a)

        """ Geometric constraints for the `bi-rectangle_constrained` routine:
          - B_min
          - B_max_x
          - B_max_y
        """
        geometric_constraints = GeometricConstraintsBiRectangleConstrained(b_min, b_max_y, b_max_x)

        # Single U-tube
        # -------------
        design_single_u_tube = DesignBiRectangleConstrained(
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
            method=DesignMethodTimeStep.Hybrid,
            flow=flow,
            property_boundary=prop_a,
            building_descriptions=ng_a,
        )

        # Find the near-square design for a single U-tube and size it.
        tic = clock()  # Clock Start Time
        bisection_search = design_single_u_tube.find_design(disp=True)  # Finding GHE Design
        bisection_search.ghe.compute_g_functions()  # Calculating g-functions for Chosen Design
        bisection_search.ghe.size(
            method=DesignMethodTimeStep.Hybrid)  # Calculating the Final Height for the Chosen Design
        toc = clock()  # Clock Stop Time

        # Print Summary of Findings
        subtitle = "* Single U-tube"  # Subtitle for the printed summary
        self.log(subtitle + "\n" + len(subtitle) * "-")
        self.log(f"Calculation time: {toc - tic:0.2f} seconds")
        self.log(f"Height: {bisection_search.ghe.bhe.b.H:0.4f} meters")
        nbh = len(bisection_search.ghe.gFunction.bore_locations)
        self.log(f"Number of boreholes: {nbh}")
        self.log(f"Total Drilling: {bisection_search.ghe.bhe.b.H * nbh:0.1f} meters\n")

        # Generating Output File
        o = OutputManager(
            bisection_search,
            toc - tic,
            project_name,
            note,
            author,
            iteration_name,
            load_method=DesignMethodTimeStep.Hybrid,
        )  # this will just go through GHEManager methods eventually
        o.write_all_output_files(
            output_directory=output_file_directory,
            file_suffix="_SU",
        )

        # *************************************************************************************************************
        # Double U-tube Example

        note = "Bi-Uniform Polygon Usage Example: Double U Tube"

        # Double U-tube
        pos_double = Pipe.place_pipes(s, r_out, 2)
        double_u_tube = MultipleUTube
        pipe_double = Pipe(pos_double, r_in, r_out, s, epsilon, k_p, rho_cp_p)

        # Double U-tube
        # -------------
        design_double_u_tube = DesignBiRectangleConstrained(
            v_flow,
            borehole,
            double_u_tube,
            fluid,
            pipe_double,
            grout,
            soil,
            sim_params,
            geometric_constraints,
            hourly_extraction_ground_loads,
            method=DesignMethodTimeStep.Hybrid,
            flow=flow,
            property_boundary=prop_a,
            building_descriptions=ng_a,
        )

        # Find the near-square design for a single U-tube and size it.
        tic = clock()  # Clock Start Time
        bisection_search = design_double_u_tube.find_design(disp=True)  # Finding GHE Design
        bisection_search.ghe.compute_g_functions()  # Calculating G-functions for Chosen Design
        bisection_search.ghe.size(
            method=DesignMethodTimeStep.Hybrid)  # Calculating the Final Height for the Chosen Design
        toc = clock()  # Clock Stop Time

        # Print Summary of Findings
        subtitle = "* Double U-tube"  # Subtitle for the printed summary
        self.log(subtitle + "\n" + len(subtitle) * "-")
        self.log(f"Calculation time: {toc - tic:0.2f} seconds")
        self.log(f"Height: {bisection_search.ghe.bhe.b.H:0.4f} meters")
        nbh = len(bisection_search.ghe.gFunction.bore_locations)
        self.log(f"Number of boreholes: {nbh}")
        self.log(f"Total Drilling: {bisection_search.ghe.bhe.b.H * nbh:0.1f} meters\n")

        # Generating Output File
        o = OutputManager(
            bisection_search,
            toc - tic,
            project_name,
            note,
            author,
            iteration_name,
            load_method=DesignMethodTimeStep.Hybrid,
        )  # this will just go through GHEManager methods eventually
        o.write_all_output_files(
            output_directory=output_file_directory,
            file_suffix="_DU",
        )

        # *************************************************************************************************************
        # Coaxial Tube Example

        note = "Bi-Uniform Polygon Usage Example: Coaxial Tube"

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

        k_p_coax = [0.4, 0.4]  # Pipes thermal conductivity (W/m.K)

        # Coaxial tube
        pos_coaxial = (0, 0)
        coaxial_tube = CoaxialPipe
        pipe_coaxial = Pipe(pos_coaxial, r_inner, r_outer, 0, epsilon, k_p_coax, rho_cp_p)

        # Coaxial Tube
        # -------------
        design_coax_tube = DesignBiRectangleConstrained(
            v_flow,
            borehole,
            coaxial_tube,
            fluid,
            pipe_coaxial,
            grout,
            soil,
            sim_params,
            geometric_constraints,
            hourly_extraction_ground_loads,
            method=DesignMethodTimeStep.Hybrid,
            flow=flow,
            property_boundary=prop_a,
            building_descriptions=ng_a,
        )

        # Find the near-square design for a single U-tube and size it.
        tic = clock()  # Clock Start Time
        bisection_search = design_coax_tube.find_design(disp=True)  # Finding GHE Design
        bisection_search.ghe.compute_g_functions()  # Calculating G-functions for Chosen Design
        bisection_search.ghe.size(
            method=DesignMethodTimeStep.Hybrid)  # Calculating the Final Height for the Chosen Design
        toc = clock()  # Clock Stop Time

        # Print Summary of Findings
        subtitle = "* Coaxial Tube"  # Subtitle for the printed summary
        self.log(subtitle + "\n" + len(subtitle) * "-")
        self.log(f"Calculation time: {toc - tic:0.2f} seconds")
        self.log(f"Height: {bisection_search.ghe.bhe.b.H:0.4f} meters")
        nbh = len(bisection_search.ghe.gFunction.bore_locations)
        self.log(f"Number of boreholes: {nbh}")
        self.log(f"Total Drilling: {bisection_search.ghe.bhe.b.H * nbh:0.1f} meters\n")

        # Generating Output File
        o = OutputManager(
            bisection_search,
            toc - tic,
            project_name,
            note,
            author,
            iteration_name,
            load_method=DesignMethodTimeStep.Hybrid,
        )  # this will just go through GHEManager methods eventually
        o.write_all_output_files(
            output_directory=output_file_directory,
            file_suffix="_C",
        )