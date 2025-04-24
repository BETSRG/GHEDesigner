from ghedesigner.enums import BHPipeType, TimestepType
from ghedesigner.ghe.boreholes.core import Borehole
from ghedesigner.ghe.design.rectangle import DesignRectangle, GeometricConstraintsRectangle
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.ghe.search.bisection_1d import Bisection1D
from ghedesigner.media import GHEFluid, Grout, Soil
from ghedesigner.tests.test_base_case import GHEBaseTest


class TestLoads(GHEBaseTest):
    def setUp(self):
        self.load = 20000
        self.num_hr_in_month = 730

    @staticmethod
    def get_designed_ghe(loads: list[float]) -> Bisection1D:
        pipe = Pipe.init_single_u_tube(
            conductivity=0.4,
            rho_cp=1542000.0,
            inner_diameter=0.03404,
            outer_diameter=0.04216,
            shank_spacing=0.01856,
            roughness=1.0e-6,
            num_pipes=1,
        )
        fluid = GHEFluid("water", 0.0, 20.0)
        grout = Grout(1.0, 3901000.0)
        soil = Soil(2.0, 2343493.0, 18.3)
        borehole = Borehole(burial_depth=2.0, borehole_radius=0.07)
        geometry = GeometricConstraintsRectangle(width=36.5, length=85.0, b_min=3.0, b_max_x=10.0)
        design = DesignRectangle(
            v_flow=0.5,
            _borehole=borehole,
            bhe_type=BHPipeType.SINGLEUTUBE,
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
            max_boreholes=100,
            geometric_constraints=geometry,
            hourly_extraction_ground_loads=loads,
            method=TimestepType.HYBRID,  # TODO: Is this the problem?
        )
        search = design.find_design()
        search.ghe.compute_g_functions(60, 135)
        search.ghe.size(method=TimestepType.HYBRID, min_height=60, max_height=135, design_min_eft=5, design_max_eft=35)
        return search

    def test_balanced_loads(self):
        loads = [-self.load if 3 <= i <= 8 else self.load for i in range(12) for _ in range(self.num_hr_in_month)]
        search = self.get_designed_ghe(loads)
        u_tube_height = search.ghe.bhe.b.H
        self.assertAlmostEqual(118.53, u_tube_height, delta=0.1)
        borehole_location_data_rows = search.ghe.gFunction.bore_locations
        self.assertEqual(6, len(borehole_location_data_rows))

    def test_imbalanced_heating_loads(self):
        loads = [self.load for _ in range(12) for _ in range(self.num_hr_in_month)]
        search = self.get_designed_ghe(loads)
        u_tube_height = search.ghe.bhe.b.H
        self.assertAlmostEqual(130.40, u_tube_height, delta=0.1)
        borehole_location_data_rows = search.ghe.gFunction.bore_locations
        self.assertEqual(10, len(borehole_location_data_rows))

    def test_imbalanced_cooling_loads(self):
        loads = [-self.load for _ in range(12) for _ in range(self.num_hr_in_month)]
        search = self.get_designed_ghe(loads)
        u_tube_height = search.ghe.bhe.b.H
        self.assertAlmostEqual(126.99, u_tube_height, delta=0.1)
        borehole_location_data_rows = search.ghe.gFunction.bore_locations
        self.assertEqual(8, len(borehole_location_data_rows))
