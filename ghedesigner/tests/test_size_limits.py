from ghedesigner.enums import BHPipeType, TimestepType
from ghedesigner.ghe.boreholes.core import Borehole
from ghedesigner.ghe.design.near_square import DesignNearSquare, GeometricConstraintsNearSquare
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.media import GHEFluid, Grout, Soil
from ghedesigner.tests.test_base_case import GHEBaseTest


class TestFindNearSquareDesign(GHEBaseTest):
    @staticmethod
    def get_pipe() -> Pipe:
        # 1-1/4" in DR-11 HDPE
        return Pipe.init_single_u_tube(
            conductivity=0.4,
            rho_cp=1542000.0,
            inner_diameter=0.03404,
            outer_diameter=0.04216,
            shank_spacing=0.01856,
            roughness=1.0e-6,
            num_pipes=1,
        )

    def test_small_loads(self):
        pipe = self.get_pipe()
        fluid = GHEFluid("water", 0.0, 20.0)
        grout = Grout(1.0, 3901000.0)
        soil = Soil(3.493, 2.5797e06, 10.0)
        ground_loads = [1.0e2] * 8760
        borehole = Borehole(burial_depth=2.0, borehole_radius=0.0751)
        geometry = GeometricConstraintsNearSquare(b=6.096, length=20)
        design = DesignNearSquare(
            v_flow=1.0,
            _borehole=borehole,
            bhe_type=BHPipeType.SINGLEUTUBE,
            fluid=fluid,
            pipe=pipe,
            grout=grout,
            soil=soil,
            start_month=1,
            end_month=12,
            max_eft=35,
            min_eft=5,
            max_height=135,
            min_height=60,
            continue_if_design_unmet=True,
            max_boreholes=100,
            geometric_constraints=geometry,
            hourly_extraction_ground_loads=ground_loads,
            method=TimestepType.HYBRID,
        )
        search = design.find_design()
        u_tube_height = search.ghe.bhe.b.H
        self.assertAlmostEqual(60, u_tube_height, delta=0.1)
        borehole_location_data_rows = search.ghe.gFunction.bore_locations
        self.assertEqual(1, len(borehole_location_data_rows))

    def test_big_loads(self):
        pipe = self.get_pipe()
        fluid = GHEFluid("water", 0.0, 20.0)
        grout = Grout(1.0, 3901000.0)
        soil = Soil(3.493, 2.5797e06, 10.0)
        ground_loads = [1.0e6] * 8760
        borehole = Borehole(burial_depth=2.0, borehole_radius=0.0751)
        geometry = GeometricConstraintsNearSquare(b=6.096, length=20)
        design = DesignNearSquare(
            v_flow=1.0,
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
            max_height=213,
            min_height=60,
            continue_if_design_unmet=True,
            max_boreholes=100,
            geometric_constraints=geometry,
            hourly_extraction_ground_loads=ground_loads,
            method=TimestepType.HYBRID,
        )
        search = design.find_design()
        u_tube_height = search.ghe.bhe.b.H
        self.assertAlmostEqual(213, u_tube_height, delta=0.1)
        borehole_location_data_rows = search.ghe.gFunction.bore_locations
        self.assertEqual(20, len(borehole_location_data_rows))

    def test_big_loads_with_max_boreholes(self):
        pipe = self.get_pipe()
        fluid = GHEFluid("water", 0.0, 20.0)
        grout = Grout(1.0, 3901000.0)
        soil = Soil(3.493, 2.5797e06, 10.0)
        ground_loads = [1.0e6] * 8760
        borehole = Borehole(burial_depth=2.0, borehole_radius=0.0751)
        geometry = GeometricConstraintsNearSquare(b=6.096, length=100)
        design = DesignNearSquare(
            v_flow=1.0,
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
            max_height=213,
            min_height=60,
            continue_if_design_unmet=True,
            max_boreholes=100,
            geometric_constraints=geometry,
            hourly_extraction_ground_loads=ground_loads,
            method=TimestepType.HYBRID,
        )
        search = design.find_design()
        u_tube_height = search.ghe.bhe.b.H
        self.assertAlmostEqual(213, u_tube_height, delta=0.1)
        borehole_location_data_rows = search.ghe.gFunction.bore_locations
        self.assertEqual(90, len(borehole_location_data_rows))
