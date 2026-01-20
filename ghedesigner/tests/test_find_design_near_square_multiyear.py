# Purpose: Design a square or near-square field using the common design
# interface with a single U-tube, multiple U-tube and coaxial tube.

# This search is described in section 4.3.2 of Cook (2021) from pages 123-129.

from ghedesigner.enums import TimestepType
from ghedesigner.ghe.boreholes.core import Borehole
from ghedesigner.ghe.design.near_square import DesignNearSquare, GeometricConstraintsNearSquare
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.media import Fluid, Grout, Soil
from ghedesigner.tests.test_base_case import GHEBaseTest


class TestFindNearSquareMultiyearDesign(GHEBaseTest):
    # This file contains three examples utilizing the square-near-square design algorithm
    # (utilizing a multi-year loading) for a single U, double U, and coaxial tube  The
    # results from these examples are exported to the "DesignExampleOutput" folder.
    def get_design(self, pipe: Pipe, flow_rate: float):
        soil = Soil(k=2.0, rho_cp=2343493.0, ugt=18.3)
        grout = Grout(k=1.0, rho_cp=3901000.0)
        fluid = Fluid(fluid_name="water", percent=0.0, temperature=20.0)
        borehole = Borehole(burial_depth=2.0, borehole_radius=0.07)
        ground_loads = self.get_multiyear_loads()
        b = 5.0
        length = 155.0
        geometry = GeometricConstraintsNearSquare(b=b, length=length)
        design = DesignNearSquare(
            v_flow=flow_rate,
            borehole=borehole,
            fluid=fluid,
            pipe=pipe,
            grout=grout,
            soil=soil,
            start_month=1,
            end_month=48,
            max_eft=35,
            min_eft=5,
            max_height=135,
            min_height=60,
            continue_if_design_unmet=True,
            max_boreholes=None,
            geometric_constraints=geometry,
            hourly_extraction_ground_loads=ground_loads,
            method=TimestepType.HYBRID,
        )
        search = design.find_design()
        search.ghe.compute_g_functions(60, 135)
        search.ghe.size(method=TimestepType.HYBRID, min_height=60, max_height=135, design_min_eft=5, design_max_eft=35)
        return search

    def test_multiyear_loading_single_u_tube(self):
        pipe = Pipe.init_single_u_tube(
            inner_diameter=0.03404,
            outer_diameter=0.04216,
            shank_spacing=0.01856,
            roughness=1.0e-6,
            conductivity=0.4,
            rho_cp=1542000.0,
        )
        search = self.get_design(pipe, 0.5)
        u_tube_height = search.ghe.bhe.borehole.H
        self.assertAlmostEqual(132.7, u_tube_height, delta=0.1)
        borehole_location_data_rows = search.ghe.gFunction.bore_locations
        self.assertEqual(16, len(borehole_location_data_rows))

    def test_multiyear_loading_double_u_tube(self):
        pipe = Pipe.init_double_u_tube_parallel(
            inner_diameter=0.03404,
            outer_diameter=0.04216,
            shank_spacing=0.01856,
            roughness=1.0e-6,
            conductivity=0.4,
            rho_cp=1542000.0,
        )
        search = self.get_design(pipe, 0.5)
        u_tube_height = search.ghe.bhe.borehole.H
        self.assertAlmostEqual(127.0, u_tube_height, delta=0.1)
        borehole_location_data_rows = search.ghe.gFunction.bore_locations
        self.assertEqual(12, len(borehole_location_data_rows))

    def test_multiyear_loading_coaxial(self):
        pipe = Pipe.init_coaxial(
            inner_pipe_d_in=0.0442,
            inner_pipe_d_out=0.050,
            outer_pipe_d_in=0.0974,
            outer_pipe_d_out=0.11,
            roughness=1.0e-6,
            conductivity=(0.4, 0.4),
            rho_cp=1542000.0,
        )
        search = self.get_design(pipe, 0.8)
        u_tube_height = search.ghe.bhe.borehole.H
        self.assertAlmostEqual(112.03, u_tube_height, delta=0.1)
        borehole_location_data_rows = search.ghe.gFunction.bore_locations
        self.assertEqual(12, len(borehole_location_data_rows))
