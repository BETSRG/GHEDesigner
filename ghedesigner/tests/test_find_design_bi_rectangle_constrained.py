# Purpose: Design a bi-uniform constrained polygonal field using the common
# design interface with a single U-tube, multiple U-tube and coaxial tube
# borehole heat exchanger.

# This search is described in section 4.4.5 from pages 146-148 in Cook (2021).

from ghedesigner.enums import TimestepType
from ghedesigner.ghe.boreholes.core import Borehole
from ghedesigner.ghe.design.birectangle_constrained import (
    DesignBiRectangleConstrained,
    GeometricConstraintsBiRectangleConstrained,
)
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.media import GHEFluid, Grout, Soil
from ghedesigner.tests.test_base_case import GHEBaseTest

prop_boundary = [
    [19.46202532, 108.8860759],
    [19.67827004, 94.46835443],
    [24.65189873, 75.3164557],
    [37.19409283, 56.59493671],
    [51.68248945, 45.83544304],
    [84.33544304, 38.94936709],
    [112.0147679, 38.94936709],
    [131.0443038, 35.50632911],
    [147.2626582, 28.83544304],
    [160.8860759, 18.07594937],
    [171.6983122, 18.29113924],
    [167.157173, 72.94936709],
    [169.1033755, 80.48101266],
    [177.3206751, 99.63291139],
    [182.2943038, 115.7721519],
    [182.2943038, 121.3670886],
    [155.0474684, 118.5696203],
    [53.19620253, 112.3291139],
]

no_go_zones = [
    [
        [74.38818565, 80.69620253],
        [73.0907173, 53.36708861],
        [93.85021097, 52.50632911],
        [120.0158228, 53.15189873],
        [121.5295359, 62.18987342],
        [128.8818565, 63.26582278],
        [128.8818565, 78.5443038],
        [129.0981013, 80.91139241],
        [108.5548523, 81.34177215],
        [104.0137131, 110],
        [95.58016878, 110],
        [95.7964135, 81.7721519],
    ]
]

prop_boundaries_multiple_bf_outlines = [
    [[0.0, 0.0], [45.0, 0.0], [90.0, 45.0], [45.0, 90.0], [0.0, 45.0]],
    [[0.0, 90.0], [80.0, 90.0], [90.0, 120.0], [10.0, 115.0]],
    [[120.0, 120.0], [165.0, 180.0], [165.0, 237.0], [140.0, 250.0], [135.0, 151.0], [120.0, 135.0]],
]

no_go_zones_multiple_bf_outlines = [
    [[60.0, 0.0], [80.0, 0.0], [80.0, 200.0], [60.0, 200.0]],
    [[60.0, 100.0], [80.0, 50.0], [140.0, 140.0], [130.0, 175.0]],
    [[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0]],
]


class TestFindBiRectangleConstrainedDesign(GHEBaseTest):
    def get_design(
        self,
        pipe: Pipe,
        flow_rate: float,
        borehole_radius: float,
        _prop_boundary: list[list[float]] | list[list[list[float]]],
        _no_go_boundaries: list[list[list[float]]],
    ):
        soil = Soil(k=2.0, rho_cp=2343493.0, ugt=18.3)
        fluid = GHEFluid("water", 0.0, 20.0)
        grout = Grout(1.0, 3901000.0)
        ground_loads = self.get_atlanta_loads()
        borehole = Borehole(burial_depth=2.0, borehole_radius=borehole_radius)
        geometry = GeometricConstraintsBiRectangleConstrained(
            b_min=5.0,
            b_max_x=25.0,
            b_max_y=25.0,
            property_boundary=_prop_boundary,
            no_go_boundaries=_no_go_boundaries,
        )
        design = DesignBiRectangleConstrained(
            v_flow=flow_rate,
            borehole=borehole,
            fluid=fluid,
            pipe=pipe,
            grout=grout,
            soil=soil,
            start_month=1,
            end_month=240,
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

    def test_single_u_tube(self):
        pipe = Pipe.init_single_u_tube(
            inner_diameter=0.03404,
            outer_diameter=0.04216,
            shank_spacing=0.01856,
            roughness=1.0e-6,
            conductivity=0.4,
            rho_cp=1542000.0,
        )
        search = self.get_design(pipe, 0.5, 0.07, prop_boundary, no_go_zones)
        u_tube_height = search.ghe.bhe.borehole.H
        self.assertAlmostEqual(133.5, u_tube_height, delta=0.1)
        borehole_location_data_rows = search.ghe.gFunction.bore_locations
        self.assertEqual(74, len(borehole_location_data_rows))

    def test_single_u_tube_multiple_bf_outlines(self):
        pipe = Pipe.init_single_u_tube(
            inner_diameter=0.03404,
            outer_diameter=0.04216,
            shank_spacing=0.01856,
            roughness=1.0e-6,
            conductivity=0.4,
            rho_cp=1542000.0,
        )
        search = self.get_design(
            pipe,
            0.2,
            0.075,
            prop_boundaries_multiple_bf_outlines,
            no_go_zones_multiple_bf_outlines,
        )
        u_tube_height = search.ghe.bhe.borehole.H
        self.assertAlmostEqual(133.7, u_tube_height, delta=0.1)
        borehole_location_data_rows = search.ghe.gFunction.bore_locations
        self.assertEqual(67, len(borehole_location_data_rows))

    def test_double_u_tube(self):
        pipe = Pipe.init_double_u_tube_parallel(
            inner_diameter=0.03404,
            outer_diameter=0.04216,
            shank_spacing=0.01856,
            roughness=1.0e-6,
            conductivity=0.4,
            rho_cp=1542000.0,
        )
        search = self.get_design(pipe, 0.5, 0.07, prop_boundary, no_go_zones)
        u_tube_height = search.ghe.bhe.borehole.H
        self.assertAlmostEqual(133.1, u_tube_height, delta=0.1)
        borehole_location_data_rows = search.ghe.gFunction.bore_locations
        self.assertEqual(63, len(borehole_location_data_rows))

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
        search = self.get_design(pipe, 0.8, 0.07, prop_boundary, no_go_zones)
        u_tube_height = search.ghe.bhe.borehole.H
        self.assertAlmostEqual(132.4, u_tube_height, delta=0.1)
        borehole_location_data_rows = search.ghe.gFunction.bore_locations
        self.assertEqual(57, len(borehole_location_data_rows))
