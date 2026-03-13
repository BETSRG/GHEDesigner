# Purpose: Design a constrained bi-rectangular field using the common design
# interface with a single U-tube, multiple U-tube and coaxial tube borehole
# heat exchanger.

# This search is described in section 4.4.2 of Cook (2021) from pages 134-138.

from ghedesigner.enums import TimestepType
from ghedesigner.ghe.boreholes.core import Borehole
from ghedesigner.ghe.design.titled_line import DesignTiltedLine, GeometricConstraintsTiltedLine
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.media import Fluid, Grout, Soil
from ghedesigner.tests.test_base_case import GHEBaseTest


class TestFindTiltedLineDesign(GHEBaseTest):
    def get_design(self, pipe: Pipe, flow_rate: float):
        soil = Soil(k=2.0, rho_cp=2343493.0, ugt=18.3)
        fluid = Fluid("water", 0.0, 20.0)
        grout = Grout(1.0, 3901000.0)
        ground_loads = self.get_tilted_loads()
        borehole = Borehole(burial_depth=2.0, borehole_radius=0.07)
        geometry = GeometricConstraintsTiltedLine(b=5, length=500, tilt=0.261799387799)
        design = DesignTiltedLine(
            v_flow=flow_rate,
            _borehole=borehole,
            fluid=fluid,
            pipe=pipe,
            grout=grout,
            soil=soil,
            start_month=1,
            end_month=240,
            max_eft=35,
            min_eft=5,
            max_height=135,
            min_height=121.5,
            continue_if_design_unmet=True,
            max_boreholes=None,
            geometric_constraints=geometry,
            hourly_extraction_ground_loads=ground_loads,
            method=TimestepType.HYBRID,
        )
        search = design.find_design()
        print("GHE B Spacing", search.ghe.b_spacing)
        search.ghe.compute_and_merge_g_functions([121.5])
        search.ghe.size(
            method=TimestepType.HYBRID, min_height=121.5, max_height=135, design_min_eft=5, design_max_eft=35
        )
        return search

    def test_single_u_tube(self):
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
        self.assertAlmostEqual(133.29, u_tube_height, delta=0.1)
        borehole_location_data_rows = search.ghe.gFunction.bore_locations
        self.assertEqual(70, len(borehole_location_data_rows))

    def test_double_u_tube(self):
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
        self.assertAlmostEqual(134.29, u_tube_height, delta=0.1)
        borehole_location_data_rows = search.ghe.gFunction.bore_locations
        self.assertEqual(60, len(borehole_location_data_rows))

    def test_coaxial(self):
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
        self.assertAlmostEqual(134.14, u_tube_height, delta=0.1)
        borehole_location_data_rows = search.ghe.gFunction.bore_locations
        self.assertEqual(55, len(borehole_location_data_rows))
