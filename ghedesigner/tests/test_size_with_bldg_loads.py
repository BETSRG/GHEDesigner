from ghedesigner.enums import TimestepType
from ghedesigner.ghe.boreholes.core import Borehole
from ghedesigner.ghe.design.rectangle import DesignRectangle, GeometricConstraintsRectangle
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.heat_pump_fixed_cop import HeatPumpFixedCOP
from ghedesigner.media import Fluid, Grout, Soil
from ghedesigner.tests.test_base_case import GHEBaseTest


class TestSizeWithBldgLoads(GHEBaseTest):
    def test_size_with_bldg_loads(self):
        hp_data = {
            "total_load": {
                "column_number": 0,
                "file_path": self.test_data_directory / "test_bldg_loads.csv",
                "heat_pump_cop": 3,
            }
        }

        heat_pump = HeatPumpFixedCOP("hp 1", hp_data)
        ground_loads = heat_pump.get_ground_loads()

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
        fluid = Fluid("water")
        grout = Grout(1.0, 3901000.0)
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
