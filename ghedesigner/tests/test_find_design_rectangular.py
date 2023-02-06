from time import time as clock

from ghedesigner.borehole import GHEBorehole
from ghedesigner.borehole_heat_exchangers import SingleUTube, MultipleUTube, CoaxialPipe
from ghedesigner.design import DesignRectangle
from ghedesigner.geometry import GeometricConstraints
from ghedesigner.media import Pipe, Soil, Grout, GHEFluid, SimulationParameters
from ghedesigner.tests.ghe_base_case import GHEBaseTest
from ghedesigner.utilities import DesignMethod


class TestFindRectangleDesign(GHEBaseTest):
    def test_single_u_tube(self):
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

        # Pipe positions
        # --------------
        # Single U-tube [(x_in, y_in), (x_out, y_out)]
        pos_single = Pipe.place_pipes(s, r_out, 1)
        # Single U-tube BHE object
        single_u_tube = SingleUTube

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
        v_flow = 0.2  # Volumetric flow rate (L/s)
        # Note: The flow parameter can be borehole or system.
        flow = "borehole"

        # Define a borehole
        borehole = GHEBorehole(h, d, r_b, x=0.0, y=0.0)

        # Simulation start month and end month
        # --------------------------------
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

        # Process loads from file
        hourly_extraction_ground_loads = self.get_atlanta_loads()

        # Rectangular design constraints are the land and range of B-spacing
        length = 85.0  # m
        width = 36.5  # m
        b_min = 3.0  # m
        b_max = 10.0  # m

        """ Geometric constraints for the `find_rectangle` routine.
        Required geometric constraints for the uniform rectangle design:
          - length
          - width
          - B_min
          - B_max
        """
        geometric_constraints = GeometricConstraints(
            length=length, width=width, b_min=b_min, b_max_x=b_max
        )

        title = "Find rectangle..."
        self.log(title + "\n" + len(title) * "=")

        # Single U-tube
        # -------------
        design_single_u_tube = DesignRectangle(
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
            method=DesignMethod.Hybrid,
        )

        # Find a constrained rectangular design for a single U-tube and size it.
        tic = clock()
        bisection_search = design_single_u_tube.find_design()
        bisection_search.ghe.compute_g_functions()
        bisection_search.ghe.size(method=DesignMethod.Hybrid)
        toc = clock()
        subtitle = "* Single U-tube"
        self.log(subtitle + "\n" + len(subtitle) * "-")
        self.log(f"Calculation time: {toc - tic:0.2f} seconds")
        self.log(f"Height: {bisection_search.ghe.bhe.b.H:0.4f} meters")
        nbh = len(bisection_search.ghe.gFunction.bore_locations)
        self.log(f"Number of boreholes: {nbh}")
        self.log(f"Total Drilling: {bisection_search.ghe.bhe.b.H * nbh:0.1f} meters\n")

    def test_double_u_tube(self):
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

        # Pipe positions
        # --------------
        # Double U-tube
        pos_double = Pipe.place_pipes(s, r_out, 2)
        double_u_tube = MultipleUTube

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
        pipe_double = Pipe(pos_double, r_in, r_out, s, epsilon, k_p, rho_cp_p)

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
        v_flow = 0.2  # Volumetric flow rate (L/s)
        # Note: The flow parameter can be borehole or system.
        flow = "borehole"

        # Define a borehole
        borehole = GHEBorehole(h, d, r_b, x=0.0, y=0.0)

        # Simulation start month and end month
        # --------------------------------
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

        # Process loads from file
        hourly_extraction_ground_loads = self.get_atlanta_loads()

        # Rectangular design constraints are the land and range of B-spacing
        length = 85.0  # m
        width = 36.5  # m
        b_min = 3.0  # m
        b_max = 10.0  # m

        """ Geometric constraints for the `find_rectangle` routine.
        Required geometric constraints for the uniform rectangle design:
          - length
          - width
          - B_min
          - B_max
        """
        geometric_constraints = GeometricConstraints(
            length=length, width=width, b_min=b_min, b_max_x=b_max
        )

        title = "Find rectangle..."
        self.log(title + "\n" + len(title) * "=")

        # Double U-tube
        # -------------
        design_double_u_tube = DesignRectangle(
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
            flow=flow,
            method=DesignMethod.Hybrid,
        )

        # Find a constrained rectangular design for a double U-tube and size it.
        tic = clock()
        bisection_search = design_double_u_tube.find_design()
        bisection_search.ghe.compute_g_functions()
        bisection_search.ghe.size(method=DesignMethod.Hybrid)
        toc = clock()
        subtitle = "* Double U-tube"
        self.log(subtitle + "\n" + len(subtitle) * "-")
        self.log(f"Calculation time: {toc - tic:0.2f} seconds")
        self.log(f"Height: {bisection_search.ghe.bhe.b.H:0.4f} meters")
        nbh = len(bisection_search.ghe.gFunction.bore_locations)
        self.log(f"Number of boreholes: {nbh}")
        self.log(f"Total Drilling: {bisection_search.ghe.bhe.b.H * nbh:0.1f} meters\n")

    def test_coaxial_pipe(self):
        # Borehole dimensions
        # -------------------
        h = 96.0  # Borehole length (m)
        d = 2.0  # Borehole buried depth (m)
        r_b = 0.075  # Borehole radius (m)

        # Pipe dimensions
        # ---------------
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
        # Coaxial tube
        pos_coaxial = (0, 0)
        coaxial_tube = CoaxialPipe

        # Thermal conductivities
        # ----------------------
        k_p_coax = [0.4, 0.4]  # Pipes thermal conductivity (W/m.K)
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
        pipe_coaxial = Pipe(
            pos_coaxial, r_inner, r_outer, 0, epsilon, k_p_coax, rho_cp_p
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
        v_flow = 0.2  # Volumetric flow rate (L/s)
        # Note: The flow parameter can be borehole or system.
        flow = "borehole"

        # Define a borehole
        borehole = GHEBorehole(h, d, r_b, x=0.0, y=0.0)

        # Simulation start month and end month
        # --------------------------------
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

        # Process loads from file
        hourly_extraction_ground_loads = self.get_atlanta_loads()

        # Rectangular design constraints are the land and range of B-spacing
        length = 85.0  # m
        width = 36.5  # m
        b_min = 3.0  # m
        b_max = 10.0  # m

        """ Geometric constraints for the `find_rectangle` routine.
        Required geometric constraints for the uniform rectangle design:
          - length
          - width
          - B_min
          - B_max
        """
        geometric_constraints = GeometricConstraints(
            length=length, width=width, b_min=b_min, b_max_x=b_max
        )

        title = "Find rectangle..."
        self.log(title + "\n" + len(title) * "=")

        # Coaxial tube
        # -------------
        design_coaxial_u_tube = DesignRectangle(
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
            flow=flow,
            method=DesignMethod.Hybrid,
        )

        # Find a constrained rectangular design for a coaxial tube and size it.
        tic = clock()
        bisection_search = design_coaxial_u_tube.find_design()
        bisection_search.ghe.compute_g_functions()
        bisection_search.ghe.size(method=DesignMethod.Hybrid)
        toc = clock()
        subtitle = "* Coaxial tube"
        self.log(subtitle + "\n" + len(subtitle) * "-")
        self.log(f"Calculation time: {toc - tic:0.2f} seconds")
        self.log(f"Height: {bisection_search.ghe.bhe.b.H:0.4f} meters")
        nbh = len(bisection_search.ghe.gFunction.bore_locations)
        self.log(f"Number of boreholes: {nbh}")
        self.log(f"Total Drilling: {bisection_search.ghe.bhe.b.H * nbh:0.1f} meters\n")