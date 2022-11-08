# Purpose: Design a constrained RowWise field using the common
# design interface with a single U-tube borehole heat exchanger.

# This search is described in <placeholder>.

from ghedt import geometry, design
from ghedt.peak_load_analysis_tool import media, borehole_heat_exchangers
import pygfunction as gt
from pathlib import Path
from time import time as clock
from ghedt.output import output_design_details
from math import pi
from ghedt.RowWise.rowwise_generation import gen_shape
from unittest import TestCase
import csv


class TestFindRowWise(TestCase):
    def test_find_row_wise(self):

        # This file contains two examples utilizing the RowWise design algorithm for a single U tube design.
        # The 1st example doesn't treat perimeter boreholes different, and the second one maintains a perimeter target-
        # spacing to interior target-spacing ratio of .8.
        # The results from these examples are exported to the "DesignExampleOutput" folder.

        # W/O Separate Perimeter Spacing Example

        # Output File Configuration
        projectName = "Atlanta Office Building: Design Example"
        note = "RowWise Usage Example w/o Perimeter Spacing: Single U Tube"
        author = "John Doe"
        IterationName = "Example 5"
        outputFileDirectory = "DesignExampleOutput"

        # Borehole dimensions
        H = 96.0  # Borehole length (m)
        D = 2.0  # Borehole buried depth (m)
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
        rhoCp_p = 1542.0 * 1000.0  # Pipe volumetric heat capacity (J/K.m3)
        rhoCp_s = 2343.493 * 1000.0  # Soil volumetric heat capacity (J/K.m3)
        rhoCp_g = 3901.0 * 1000.0  # Grout volumetric heat capacity (J/K.m3)

        # Instantiating Pipe
        pipe_single = media.Pipe(pos_single, r_in, r_out, s, epsilon, k_p, rhoCp_p)

        # Instantiating Soil Properties
        ugt = 18.3  # Undisturbed ground temperature (degrees Celsius)
        soil = media.Soil(k_s, rhoCp_s, ugt)

        # Instantiating Grout Properties
        grout = media.Grout(k_g, rhoCp_g)

        # Fluid properties
        fluid = gt.media.Fluid(fluid_str="Water", percent=0.0)

        # Fluid Flow Properties
        V_flow = 0.2  # Volumetric flow rate (L/s)
        # Note: The flow parameter can be borehole or system.
        flow = "borehole"

        # Instantiate a Borehole
        borehole = gt.boreholes.Borehole(H, D, r_b, x=0.0, y=0.0)

        # Simulation parameters
        start_month = 1
        n_years = 20
        end_month = n_years * 12
        max_EFT_allowable = 35  # degrees Celsius (HPEFT)
        min_EFT_allowable = 5  # degrees Celsius (HPEFT)
        max_Height = 135.0  # in meters
        min_Height = 60  # in meters
        sim_params = media.SimulationParameters(
            start_month,
            end_month,
            max_EFT_allowable,
            min_EFT_allowable,
            max_Height,
            min_Height,
        )

        # Process loads from file
        # read in the csv file and convert the loads to a list of length 8760
        test_data_dir = Path(__file__).resolve().parent / 'test_data'
        csv_file = test_data_dir / 'Atlanta_Office_Building_Loads.csv'
        raw_lines = csv_file.read_text().split('\n')
        hourly_extraction_ground_loads = [float(x) for x in raw_lines[1:] if x.strip() != '']

        # RowWise Design Constraints

        pSpacs = 0.8  # Unitless
        spacStart = 10.0  # in meters
        spacStop = 20.0  # in meters
        spacStep = 0.1  # in meters
        rotateStep = 0.5  # in degrees
        rotateStart = -90.0 * (pi / 180.0)  # in radians
        rotateStop = 0 * (pi / 180.0)  # in radians

        # Building Description
        property_boundary_file = test_data_dir / 'property_polygons' / "property_boundary.csv"
        no_go_zone_file = test_data_dir / 'property_polygons' / "NogoZone1.csv"

        propA = []  # in meters
        ngA = []  # in meters

        with open(property_boundary_file, "r", newline="") as pF:
            cR = csv.reader(pF)
            for line in cR:
                L = []
                for row in line:
                    L.append(float(row))
                propA.append(L)

        # for file in os.listdir(no_go_zone_file):
        with open(no_go_zone_file, "r", newline="") as ngF:
            cR = csv.reader(ngF)
            ngA.append([])
            for line in cR:
                L = []
                for row in line:
                    L.append(float(row))
                ngA[-1].append(L)

        buildVert, nogoVert = gen_shape(propA, ng_zones=ngA)

        """ Geometric constraints for the `row-wise` routine:
          - list of vertices for the nogo zones (nogoVert)
          - perimeter target-spacing to interior target-spacing ratio
          - the lower bound target-spacing (spacStart)
          - the upper bound target-spacing (spacStop)
          - the range around the selected target-spacing over-which to to do an exhaustive search
          - the lower bound rotation (rotateStart)
          - the upper bound rotation (rotateStop)
          - list of vertices for the property boundary (buildVert)
        """
        geometric_constraints = geometry.GeometricConstraints(
            ng_zones=nogoVert,
            p_spacing=pSpacs,
            spacing_start=spacStart,
            spacing_stop=spacStop,
            spacing_step=spacStep,
            rotate_start=rotateStart,
            rotate_stop=rotateStop,
            rotate_step=rotateStep,
            prop_bound=buildVert,
        )

        # Single U-tube
        # -------------
        design_single_u_tube = design.DesignRowWise(
            V_flow,
            borehole,
            single_u_tube,
            fluid,
            pipe_single,
            grout,
            soil,
            sim_params,
            geometric_constraints,
            hourly_extraction_ground_loads,
            method="hybrid",
            flow=flow,
        )

        # Find the near-square design for a single U-tube and size it.
        tic = clock()  # Clock Start Time
        bisection_search = design_single_u_tube.find_design(
            disp=True, use_perimeter=False
        )  # Finding GHE Design
        bisection_search.ghe.compute_g_functions()  # Calculating Gfunctions for Chosen Design
        bisection_search.ghe.size(
            method="hybrid"
        )  # Calculating the Final Height for the Chosen Design
        toc = clock()  # Clock Stop Time

        # Print Summary of Findings
        subtitle = "* Single U-tube"  # Subtitle for the printed summary
        print(subtitle + "\n" + len(subtitle) * "-")
        print("Calculation time: {0:.2f} seconds".format(toc - tic))
        print("Height: {0:.4f} meters".format(bisection_search.ghe.bhe.b.H))
        nbh = len(bisection_search.ghe.GFunction.bore_locations)
        print("Number of boreholes: {}".format(nbh))
        print("Total Drilling: {0:.1f} meters\n".format(bisection_search.ghe.bhe.b.H * nbh))

        # Generating Ouptut File
        output_design_details(
            bisection_search,
            toc - tic,
            projectName,
            note,
            author,
            IterationName,
            output_directory=outputFileDirectory,
            summary_file="SummaryOfResults_SU_WOP.txt",
            csv_f_1="TimeDependentValues_SU_WOP.csv",
            csv_f_2="BorefieldData_SU_WOP.csv",
            csv_f_3="Loadings_SU_WOP.csv",
            csv_f_4="GFunction_SU_WOP.csv",
        )

        # *************************************************************************************************************
        # Perimeter Spacing Example

        note = "RowWise Usage Example w/o Perimeter Spacing: Single U Tube"

        # Single U-tube
        # -------------
        design_single_u_tube = design.DesignRowWise(
            V_flow,
            borehole,
            single_u_tube,
            fluid,
            pipe_single,
            grout,
            soil,
            sim_params,
            geometric_constraints,
            hourly_extraction_ground_loads,
            method="hybrid",
            flow=flow,
        )

        # Find the near-square design for a single U-tube and size it.
        tic = clock()  # Clock Start Time
        bisection_search = design_single_u_tube.find_design(
            disp=True, use_perimeter=True
        )  # Finding GHE Design
        bisection_search.ghe.compute_g_functions()  # Calculating Gfunctions for Chosen Design
        bisection_search.ghe.size(
            method="hybrid"
        )  # Calculating the Final Height for the Chosen Design
        toc = clock()  # Clock Stop Time

        # Print Summary of Findings
        subtitle = "* Single U-tube"  # Subtitle for the printed summary
        print(subtitle + "\n" + len(subtitle) * "-")
        print("Calculation time: {0:.2f} seconds".format(toc - tic))
        print("Height: {0:.4f} meters".format(bisection_search.ghe.bhe.b.H))
        nbh = len(bisection_search.ghe.GFunction.bore_locations)
        print("Number of boreholes: {}".format(nbh))
        print("Total Drilling: {0:.1f} meters\n".format(bisection_search.ghe.bhe.b.H * nbh))

        # Generating Ouptut File
        output_design_details(
            bisection_search,
            toc - tic,
            projectName,
            note,
            author,
            IterationName,
            output_directory=outputFileDirectory,
            summary_file="SummaryOfResults_SU_WP.txt",
            csv_f_1="TimeDependentValues_SU_WP.csv",
            csv_f_2="BorefieldData_SU_WP.csv",
            csv_f_3="Loadings_SU_WP.csv",
            csv_f_4="GFunction_SU_WP.csv",
        )
