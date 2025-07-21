from pathlib import Path

from ghedesigner.building import Building
from ghedesigner.enums import TimestepType
from ghedesigner.ghe.boreholes.core import Borehole
from ghedesigner.ghe.design.rectangle import DesignRectangle, GeometricConstraintsRectangle
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.heat_pump import HeatPump
from ghedesigner.media import GHEFluid, Grout, Soil
from ghedesigner.system import System
from ghedesigner.tests.test_base_case import GHEBaseTest


class TestNewWorkflows(GHEBaseTest):
    def test_new_workflow(self):
        system = System()
        system.set_simulation_parameters(num_months=240)

        # read building loads
        heat_pump = HeatPump("load 1")
        building_loads_path = Path(__file__).parent / "test_data" / "test_bldg_loads.csv"
        heat_pump.set_loads_from_file(building_loads_path)
        heat_pump.set_fixed_cop(3)
        heat_pump.do_sizing()

        # size building heat pump
        building = Building("building")
        building.add_heat_pump(heat_pump)

        # size ghe
        pipe = Pipe.init_single_u_tube(
            inner_diameter=0.03404,
            outer_diameter=0.04216,
            shank_spacing=0.01856,
            roughness=1.0e-6,
            conductivity=0.4,
            rho_cp=1542000.0,
        )
        soil = Soil(k=2.0, rho_cp=2343493.0, ugt=18.3)
        fluid = GHEFluid("water", 0.0, 20.0)
        grout = Grout(1.0, 3901000.0)
        ground_loads = self.get_atlanta_loads()
        borehole = Borehole(burial_depth=2.0, borehole_radius=0.07)
        geometry = GeometricConstraintsRectangle(width=36.5, length=85.0, b_min=3.0, b_max=10)
        min_height = 60
        max_height = 135
        min_eft = 5
        max_eft = 35
        design = DesignRectangle(
            v_flow=0.5,
            borehole=borehole,
            fluid=fluid,
            pipe=pipe,
            grout=grout,
            soil=soil,
            start_month=1,
            end_month=240,
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
        search.ghe.compute_g_functions(min_height, max_height)
        search.ghe.size(
            method=TimestepType.HYBRID,
            min_height=min_height,
            max_height=max_height,
            design_min_eft=5,
            design_max_eft=35,
        )
        # simulate hourly
        system.set_building(building)
        system.set_ghe(search.ghe)
        system.simulate()
