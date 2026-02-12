from typing import cast

from ghedesigner.constants import DEG_TO_RAD
from ghedesigner.enums import TimestepType
from ghedesigner.ghe.boreholes.core import Borehole
from ghedesigner.ghe.design.rowwise import DesignRowWise, GeometricConstraintsRowWise
from ghedesigner.ghe.ground_heat_exchangers import GHE
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.media import Fluid, Grout, Soil
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


# This file contains two examples utilizing the RowWise design algorithm for a single U tube
# The 1st example doesn't treat perimeter boreholes different, and the second one maintains a perimeter target
# spacing to interior target-spacing ratio of .8.


class TestFindRowWiseDesign(GHEBaseTest):
    # Purpose: Design a constrained RowWise field using the common
    # design interface with a single U-tube borehole heat exchanger.
    def get_design(self, pipe: Pipe, flow_rate: float, spacing_ratio: float | None = None):
        soil = Soil(k=2.0, rho_cp=2343493.0, ugt=18.3)
        fluid = Fluid("water")
        grout = Grout(1.0, 3901000.0)
        ground_loads = self.get_atlanta_loads()
        borehole = Borehole(burial_depth=2.0, borehole_radius=0.07)
        geometry = GeometricConstraintsRowWise(
            perimeter_spacing_ratio=spacing_ratio,
            min_spacing=10.0,
            max_spacing=20.0,
            spacing_step=0.1,
            min_rotation=-90.0 * DEG_TO_RAD,
            max_rotation=0.0 * DEG_TO_RAD,
            rotate_step=0.5,
            property_boundary=prop_boundary,
            no_go_boundaries=no_go_zones,
        )
        design = DesignRowWise(
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
            max_height=200,
            min_height=60,
            continue_if_design_unmet=True,
            max_boreholes=None,
            geometric_constraints=geometry,
            hourly_extraction_ground_loads=ground_loads,
            method=TimestepType.HYBRID,
        )
        search = design.find_design()
        search.ghe = cast(GHE, search.ghe)  # Cast the type to GHE
        search.ghe.compute_g_functions(60, 200)
        search.ghe.size(method=TimestepType.HYBRID, min_height=60, max_height=200, design_min_eft=5, design_max_eft=35)
        return search

    def test_find_row_wise_design_wo_perimeter(self):
        pipe = Pipe.init_single_u_tube(
            inner_diameter=0.03404,
            outer_diameter=0.04216,
            shank_spacing=0.01856,
            roughness=1.0e-6,
            conductivity=0.4,
            rho_cp=1542000.0,
        )
        search = self.get_design(pipe, 0.5, None)
        u_tube_height = search.ghe.bhe.borehole.H
        self.assertAlmostEqual(197.4, u_tube_height, delta=0.1)
        borehole_location_data_rows = search.ghe.gFunction.bore_locations
        self.assertEqual(40, len(borehole_location_data_rows))

    def test_find_row_wise_design_with_perimeter(self):
        pipe = Pipe.init_single_u_tube(
            inner_diameter=0.03404,
            outer_diameter=0.04216,
            shank_spacing=0.01856,
            roughness=1.0e-6,
            conductivity=0.4,
            rho_cp=1542000.0,
        )
        search = self.get_design(pipe, 0.5, 0.8)
        u_tube_height = search.ghe.bhe.borehole.H
        self.assertAlmostEqual(199.4, u_tube_height, delta=0.1)
        borehole_location_data_rows = search.ghe.gFunction.bore_locations
        self.assertEqual(40, len(borehole_location_data_rows))
