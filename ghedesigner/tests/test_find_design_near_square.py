# Purpose: Design a square or near-square field using the common design
# interface with a single U-tube, multiple U-tube and coaxial tube.

# This search is described in section 4.3.2 of Cook (2021) from pages 123-129.

from time import time as clock

from ghedesigner.borehole import GHEBorehole
from ghedesigner.borehole_heat_exchangers import MultipleUTube, CoaxialPipe
from ghedesigner.design import DesignNearSquare
from ghedesigner.geometry import GeometricConstraints
from ghedesigner.manager import GHEManager
from ghedesigner.media import Pipe, Soil, Grout, GHEFluid, SimulationParameters
from ghedesigner.output import write_output_files
from ghedesigner.tests.ghe_base_case import GHEBaseTest
from ghedesigner.utilities import DesignMethod, length_of_side


class TestFindNearSquareDesign(GHEBaseTest):

    def test_find_single_u_tube_design(self):
        manager = GHEManager()
        manager.set_pipe(
            inner_radius=(21.6 / 1000.0 / 2.0), outer_radius=(26.67 / 1000.0 / 2.0),
            shank_spacing=(32.3 / 1000.0), roughness=1.0e-6, conductivity=0.4, rho_cp=(1542.0 * 1000.0)
        )
        manager.set_soil(conductivity=2.0, rho_cp=(2343.493 * 1000.0), undisturbed_temp=18.3)
        manager.set_grout(conductivity=1.0, rho_cp=(3901.0 * 1000.0))
        manager.set_fluid()
        manager.set_borehole(length=96.0, buried_depth=2.0, radius=0.075)
        manager.set_simulation_parameters(num_months=240, max_eft=35, min_eft=5, max_height=135, min_height=60)
        manager.set_ground_loads_from_hourly_list(self.get_atlanta_loads())
        manager.set_geometry_constraints(b=5.0, length=155)  # borehole spacing and field side length
        # perform a design search assuming "system" flow?
        manager.set_design(flow_rate=6.4, flow_type="system")
        tic = clock()  # Clock Start Time
        manager.find_design()
        toc = clock()  # Clock Stop Time

        self.assertAlmostEqual(manager.u_tube_height, 124.92, delta=1e-2)

        # Print Summary of Findings
        subtitle = "* Single U-tube"  # Subtitle for the printed summary
        self.log(subtitle + "\n" + len(subtitle) * "-")
        self.log(f"Calculation time: {toc - tic:0.2f} seconds")
        self.log(f"Height: {manager.u_tube_height:0.4f} meters")

        # Output File Configuration
        project_name = "Atlanta Office Building: Design Example"
        note = "Square-Near-Square Usage Example: Single U Tube"
        author = "John Doe"
        iteration_name = "Example 1"
        output_file_directory = self.test_outputs_directory / "DesignExampleOutput"

        # Generating Output File
        write_output_files(
            manager._search,  # TODO: Make so we don't have to access a protected method for this
            toc - tic,
            project_name,
            note,
            author,
            iteration_name,
            output_directory=output_file_directory,
            file_suffix="_SU",
            load_method=DesignMethod.Hybrid,
        )

    def test_find_double_u_tube_design(self):
        # *************************************************************************************************************
        # Double U-tube Example

        project_name = "Atlanta Office Building: Design Example"
        author = "John Doe"
        iteration_name = "Example 1"
        output_file_directory = self.test_outputs_directory / "DesignExampleOutput"

        # Borehole dimensions
        h = 96.0  # Borehole length (m)
        d = 2.0  # Borehole buried depth (m)
        r_b = 0.075  # Borehole radius (m)
        b = 5.0  # Borehole spacing (m)

        # Single and Multiple U-tube Pipe Dimensions
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
        max_eft_allowable = 35  # degrees Celsius (HP EFT)
        min_eft_allowable = 5  # degrees Celsius (HP EFT)
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

        number_of_boreholes = 32
        length = length_of_side(number_of_boreholes, b)
        geometric_constraints = GeometricConstraints(b=b, length=length)
        hourly_extraction_ground_loads = self.get_atlanta_loads()

        note = "Square-Near-Square Usage Example: Double U Tube"

        # Double U-tube
        pos_double = Pipe.place_pipes(s, r_out, 2)
        double_u_tube = MultipleUTube
        pipe_double = Pipe(pos_double, r_in, r_out, s, epsilon, k_p, rho_cp_p)

        # Double U-tube
        # -------------
        design_double_u_tube = DesignNearSquare(
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
            method=DesignMethod.Hybrid,
            flow=flow,
        )

        # Find the near-square design for a single U-tube and size it.
        tic = clock()  # Clock Start Time
        bisection_search = design_double_u_tube.find_design(disp=True)  # Finding GHE Design
        bisection_search.ghe.compute_g_functions()  # Calculating G-functions for Chosen Design
        bisection_search.ghe.size(
            method=DesignMethod.Hybrid
        )  # Calculating the Final Height for the Chosen Design
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
        write_output_files(
            bisection_search,
            toc - tic,
            project_name,
            note,
            author,
            iteration_name,
            output_directory=output_file_directory,
            file_suffix="_DU",
            load_method=DesignMethod.Hybrid,
        )

    def test_find_coaxial_pipe_design(self):
        # *************************************************************************************************************
        # Coaxial Tube Example

        project_name = "Atlanta Office Building: Design Example"
        author = "John Doe"
        iteration_name = "Example 1"
        output_file_directory = self.test_outputs_directory / "DesignExampleOutput"

        # Borehole dimensions
        h = 96.0  # Borehole length (m)
        d = 2.0  # Borehole buried depth (m)
        r_b = 0.075  # Borehole radius (m)
        b = 5.0  # Borehole spacing (m)

        # Single and Multiple U-tube Pipe Dimensions
        epsilon = 1.0e-6  # Pipe roughness (m)

        # Thermal conductivities
        k_s = 2.0  # Ground thermal conductivity (W/m.K)
        k_g = 1.0  # Grout thermal conductivity (W/m.K)

        # Volumetric heat capacities
        rho_cp_p = 1542.0 * 1000.0  # Pipe volumetric heat capacity (J/K.m3)
        rho_cp_s = 2343.493 * 1000.0  # Soil volumetric heat capacity (J/K.m3)
        rho_cp_g = 3901.0 * 1000.0  # Grout volumetric heat capacity (J/K.m3)

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
        max_eft_allowable = 35  # degrees Celsius (HP EFT)
        min_eft_allowable = 5  # degrees Celsius (HP EFT)
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

        number_of_boreholes = 32
        length = length_of_side(number_of_boreholes, b)
        geometric_constraints = GeometricConstraints(b=b, length=length)
        hourly_extraction_ground_loads = self.get_atlanta_loads()

        note = "Square-Near-Square Usage Example: Coaxial Tube"

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
        pipe_coaxial = Pipe(
            pos_coaxial, r_inner, r_outer, 0, epsilon, k_p_coax, rho_cp_p
        )

        # Coaxial Tube
        # -------------
        design_coax_tube = DesignNearSquare(
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
            method=DesignMethod.Hybrid,
            flow=flow,
        )

        # Find the near-square design for a single U-tube and size it.
        tic = clock()  # Clock Start Time
        bisection_search = design_coax_tube.find_design(disp=True)  # Finding GHE Design
        bisection_search.ghe.compute_g_functions()  # Calculating G-functions for Chosen Design
        bisection_search.ghe.size(
            method=DesignMethod.Hybrid
        )  # Calculating the Final Height for the Chosen Design
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
        write_output_files(
            bisection_search,
            toc - tic,
            project_name,
            note,
            author,
            iteration_name,
            output_directory=output_file_directory,
            file_suffix="_C",
            load_method=DesignMethod.Hybrid,
        )
