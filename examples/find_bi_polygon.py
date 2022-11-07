# Purpose: Design a bi-uniform constrained polygonal field using the common
# design interface with a single U-tube, multiple U-tube and coaxial tube
# borehole heat exchanger.

# This search is described in section 4.4.5 from pages 146-148 in Cook (2021).

import csv
import tempfile
from pathlib import Path
from sys import path
from time import time as clock

import pandas as pd
import pygfunction as gt

# use the ghedt import as a flag for determining whether we need to add to path
try:
    import ghedt  # noqa: F401
except ImportError:
    # we are probably in VSCode or some other development setup
    # just add the root of the repo to path just like it will be in deployment
    root_dir = Path(__file__).parent.parent.resolve()
    path.insert(0, str(root_dir))

from ghedt import design, geometry
from ghedt.output import output_design_details
from ghedt.peak_load_analysis_tool import media, borehole_heat_exchangers


def main():

    # This file contains three examples utilizing the bi-uniform polygonal design algorithm for a single U, double U,
    # and coaxial tube design. The results from these examples are exported to the "DesignExampleOutput" folder.

    # Single U-tube Example

    # Output File Configuration
    project_name = "Atlanta Office Building: Design Example"
    note = "Bi-Uniform Polygon Usage Example: Single U Tube"
    author = "Jane Doe"
    iteration_name = "Example 6"
    output_file_directory = tempfile.mkdtemp()

    # Borehole dimensions
    h = 96.0  # Borehole length (m)
    d = 2.0  # Borehole buried depth (m)
    r_b = 0.075  # Borehole radius (m)
    # B = 5.0  # Borehole spacing (m)

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
    max_eft_allowable = 35  # degrees Celsius (HP_EFT)
    min_eft_allowable = 5  # degrees Celsius (HP_EFT)
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
    # read in the csv file and convert the loads to a list of length 8760
    hourly_extraction: dict = pd.read_csv(
        Path(__file__).parent / "data" / "Atlanta_Office_Building_Loads.csv"
    ).to_dict("list")
    # Take only the first column in the dictionary
    hourly_extraction_ground_loads: list = hourly_extraction[
        list(hourly_extraction.keys())[0]
    ]

    # Polygonal design constraints are the land and range of B-spacing
    b_min = 5  # in m
    b_max_x = 25  # in m
    b_max_y = b_max_x  # in m

    # Building Description
    property_boundary_file = Path(__file__).parent / "data" / "PropBound.csv"
    no_go_zone_file = Path(__file__).parent / "data" / "NogoZone1.csv"

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
    geometric_constraints = geometry.GeometricConstraints(
        b_min=b_min, b_max_y=b_max_y, b_max_x=b_max_x
    )

    # Single U-tube
    # -------------
    design_single_u_tube = design.Design(
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
        method="hybrid",
        flow=flow,
        routine="bi-rectangle_constrained",
        property_boundary=prop_a,
        building_descriptions=ng_a,
    )

    # Find the near-square design for a single U-tube and size it.
    tic = clock()  # Clock Start Time
    bisection_search = design_single_u_tube.find_design(disp=True)  # Finding GHE Design
    bisection_search.ghe.compute_g_functions()  # Calculating g-functions for Chosen Design
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

    # Generating Output File
    output_design_details(
        bisection_search,
        toc - tic,
        project_name,
        note,
        author,
        iteration_name,
        output_directory=output_file_directory,
        summary_file="SummaryOfResults_SU.txt",
        csv_f_1="TimeDependentValues_SU.csv",
        csv_f_2="BorefieldData_SU.csv",
        csv_f_3="Loadings_SU.csv",
        csv_f_4="GFunction_SU.csv",
    )

    # *************************************************************************************************************
    # Double U-tube Example

    note = "Bi-Uniform Polygon Usage Example: Double U Tube"

    # Double U-tube
    pos_double = media.Pipe.place_pipes(s, r_out, 2)
    double_u_tube = borehole_heat_exchangers.MultipleUTube
    pipe_double = media.Pipe(pos_double, r_in, r_out, s, epsilon, k_p, rho_cp_p)

    # Double U-tube
    # -------------
    design_double_u_tube = design.Design(
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
        method="hybrid",
        flow=flow,
        routine="bi-rectangle_constrained",
        property_boundary=prop_a,
        building_descriptions=ng_a,
    )

    # Find the near-square design for a single U-tube and size it.
    tic = clock()  # Clock Start Time
    bisection_search = design_double_u_tube.find_design(disp=True)  # Finding GHE Design
    bisection_search.ghe.compute_g_functions()  # Calculating G-functions for Chosen Design
    bisection_search.ghe.size(
        method="hybrid"
    )  # Calculating the Final Height for the Chosen Design
    toc = clock()  # Clock Stop Time

    # Print Summary of Findings
    subtitle = "* Double U-tube"  # Subtitle for the printed summary
    print(subtitle + "\n" + len(subtitle) * "-")
    print("Calculation time: {0:.2f} seconds".format(toc - tic))
    print("Height: {0:.4f} meters".format(bisection_search.ghe.bhe.b.H))
    nbh = len(bisection_search.ghe.GFunction.bore_locations)
    print("Number of boreholes: {}".format(nbh))
    print("Total Drilling: {0:.1f} meters\n".format(bisection_search.ghe.bhe.b.H * nbh))

    # Generating Output File
    output_design_details(
        bisection_search,
        toc - tic,
        project_name,
        note,
        author,
        iteration_name,
        output_directory=output_file_directory,
        summary_file="SummaryOfResults_DU.txt",
        csv_f_1="TimeDependentValues_DU.csv",
        csv_f_2="BorefieldData_DU.csv",
        csv_f_3="Loadings_DU.csv",
        csv_f_4="GFunction_DU.csv",
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
    coaxial_tube = borehole_heat_exchangers.CoaxialPipe
    pipe_coaxial = media.Pipe(
        pos_coaxial, r_inner, r_outer, 0, epsilon, k_p_coax, rho_cp_p
    )

    # Coaxial Tube
    # -------------
    design_coax_tube = design.Design(
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
        method="hybrid",
        flow=flow,
        routine="bi-rectangle_constrained",
        property_boundary=prop_a,
        building_descriptions=ng_a,
    )

    # Find the near-square design for a single U-tube and size it.
    tic = clock()  # Clock Start Time
    bisection_search = design_coax_tube.find_design(disp=True)  # Finding GHE Design
    bisection_search.ghe.compute_g_functions()  # Calculating G-functions for Chosen Design
    bisection_search.ghe.size(
        method="hybrid"
    )  # Calculating the Final Height for the Chosen Design
    toc = clock()  # Clock Stop Time

    # Print Summary of Findings
    subtitle = "* Coaxial Tube"  # Subtitle for the printed summary
    print(subtitle + "\n" + len(subtitle) * "-")
    print("Calculation time: {0:.2f} seconds".format(toc - tic))
    print("Height: {0:.4f} meters".format(bisection_search.ghe.bhe.b.H))
    nbh = len(bisection_search.ghe.GFunction.bore_locations)
    print("Number of boreholes: {}".format(nbh))
    print("Total Drilling: {0:.1f} meters\n".format(bisection_search.ghe.bhe.b.H * nbh))

    # Generating Output File
    output_design_details(
        bisection_search,
        toc - tic,
        project_name,
        note,
        author,
        iteration_name,
        output_directory=output_file_directory,
        summary_file="SummaryOfResults_C.txt",
        csv_f_1="TimeDependentValues_C.csv",
        csv_f_2="BorefieldData_C.csv",
        csv_f_3="Loadings_C.csv",
        csv_f_4="GFunction_C.csv",
    )


if __name__ == "__main__":
    main()