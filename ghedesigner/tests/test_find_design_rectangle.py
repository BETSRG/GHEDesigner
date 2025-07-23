from ghedesigner.enums import TimestepType
from ghedesigner.ghe.boreholes.core import Borehole
from ghedesigner.ghe.design.rectangle import DesignRectangle, GeometricConstraintsRectangle
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.media import GHEFluid, Grout, Soil
from ghedesigner.tests.test_base_case import GHEBaseTest


class TestFindRectangleDesign(GHEBaseTest):
    def get_design(self, pipe: Pipe, flow_rate: float):
        fluid = GHEFluid("water", 0.0, 20.0)
        grout = Grout(1.0, 3901000.0)
        soil = Soil(2.0, 2343493.0, 18.3)
        ground_loads = self.get_atlanta_loads()
        borehole = Borehole(burial_depth=2.0, borehole_radius=0.07)
        num_months = 240
        geometry = GeometricConstraintsRectangle(width=36.5, length=85.0, b_min=3.0, b_max=10)
        min_height = 60
        max_height = 135
        min_eft = 5
        max_eft = 35
        design = DesignRectangle(
            v_flow=flow_rate,
            borehole=borehole,
            fluid=fluid,
            pipe=pipe,
            grout=grout,
            soil=soil,
            start_month=1,
            end_month=num_months,
            max_eft=max_eft,
            min_eft=min_eft,
            max_height=max_height,
            min_height=min_height,
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
        self.assertAlmostEqual(120.9, u_tube_height, delta=0.1)
        borehole_location_data_rows = search.ghe.gFunction.bore_locations
        self.assertEqual(180, len(borehole_location_data_rows))

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
        self.assertAlmostEqual(125.4, u_tube_height, delta=0.1)
        borehole_location_data_rows = search.ghe.gFunction.bore_locations
        self.assertEqual(144, len(borehole_location_data_rows))

    def test_coaxial_pipe(self):
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
        self.assertAlmostEqual(119.4, u_tube_height, delta=0.1)
        borehole_location_data_rows = search.ghe.gFunction.bore_locations
        self.assertEqual(144, len(borehole_location_data_rows))
