from time import time as clock

from ghedesigner.borehole import GHEBorehole
from ghedesigner.borehole_heat_exchangers import CoaxialPipe
from ghedesigner.design import DesignRectangle
from ghedesigner.geometry import GeometricConstraints
from ghedesigner.manager import GHEManager, DesignMethodGeometry
from ghedesigner.media import Pipe, Soil, Grout, GHEFluid, SimulationParameters
from ghedesigner.tests.ghe_base_case import GHEBaseTest
from ghedesigner.utilities import DesignMethodTimeStep


class TestFindRectangleDesign(GHEBaseTest):
    def test_single_u_tube(self):

        manager = GHEManager()
        manager.set_single_u_tube_pipe(
            inner_radius=(21.6 / 1000.0 / 2.0), outer_radius=(26.67 / 1000.0 / 2.0), shank_spacing=(32.3 / 1000.0),
            roughness=1.0e-6, conductivity=0.4, rho_cp=(1542.0 * 1000.0)
        )
        manager.set_soil(conductivity=2.0, rho_cp=(2343.493 * 1000.0), undisturbed_temp=18.3)
        manager.set_grout(conductivity=1.0, rho_cp=(3901.0 * 1000.0))
        manager.set_fluid()
        manager.set_borehole(length=96.0, buried_depth=2.0, radius=0.075)
        manager.set_simulation_parameters(num_months=240, max_eft=35, min_eft=5, max_height=135, min_height=60)
        manager.set_ground_loads_from_hourly_list(self.get_atlanta_loads())
        manager.set_geometry_constraints_rectangular(length=85.0, width=36.5, b_min=3.0, b_max=10.0)
        manager.set_design(flow_rate=0.2, flow_type="borehole", design_method_geo=DesignMethodGeometry.Rectangular)
        manager.find_design()

    def test_double_u_tube(self):

        manager = GHEManager()
        manager.set_double_u_tube_pipe(
            inner_radius=(21.6 / 1000.0 / 2.0), outer_radius=(26.67 / 1000.0 / 2.0), shank_spacing=(32.3 / 1000.0),
            roughness=1.0e-6, conductivity=0.4, rho_cp=(1542.0 * 1000.0)
        )
        manager.set_soil(conductivity=2.0, rho_cp=(2343.493 * 1000.0), undisturbed_temp=18.3)
        manager.set_grout(conductivity=1.0, rho_cp=(3901.0 * 1000.0))
        manager.set_fluid()
        manager.set_borehole(length=96.0, buried_depth=2.0, radius=0.075)
        manager.set_simulation_parameters(num_months=240, max_eft=35, min_eft=5, max_height=135, min_height=60)
        manager.set_ground_loads_from_hourly_list(self.get_atlanta_loads())
        manager.set_geometry_constraints_rectangular(length=85.0, width=36.5, b_min=3.0, b_max=10.0)
        manager.set_design(flow_rate=0.2, flow_type="borehole", design_method_geo=DesignMethodGeometry.Rectangular)
        manager.find_design()

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
            method=DesignMethodTimeStep.Hybrid,
        )

        # Find a constrained rectangular design for a coaxial tube and size it.
        tic = clock()
        bisection_search = design_coaxial_u_tube.find_design()
        bisection_search.ghe.compute_g_functions()
        bisection_search.ghe.size(method=DesignMethodTimeStep.Hybrid)
        toc = clock()
        subtitle = "* Coaxial tube"
        self.log(subtitle + "\n" + len(subtitle) * "-")
        self.log(f"Calculation time: {toc - tic:0.2f} seconds")
        self.log(f"Height: {bisection_search.ghe.bhe.b.H:0.4f} meters")
        nbh = len(bisection_search.ghe.gFunction.bore_locations)
        self.log(f"Number of boreholes: {nbh}")
        self.log(f"Total Drilling: {bisection_search.ghe.bhe.b.H * nbh:0.1f} meters\n")
