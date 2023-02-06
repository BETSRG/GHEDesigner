from ghedesigner.manager import GHEManager, DesignMethodGeometry
from ghedesigner.tests.ghe_base_case import GHEBaseTest


class TestFindRectangleDesign(GHEBaseTest):
    def test_single_u_tube(self):
        manager = GHEManager()
        manager.set_single_u_tube_pipe(inner_radius=(21.6 / 1000.0 / 2.0), outer_radius=(26.67 / 1000.0 / 2.0),
                                       shank_spacing=(32.3 / 1000.0),
                                       roughness=1.0e-6, conductivity=0.4, rho_cp=(1542.0 * 1000.0))

        manager.set_soil(conductivity=2.0, rho_cp=(2343.493 * 1000.0), undisturbed_temp=18.3)
        manager.set_grout(conductivity=1.0, rho_cp=(3901.0 * 1000.0))
        manager.set_fluid()
        manager.set_borehole(length=96.0, buried_depth=2.0, radius=0.075)
        manager.set_simulation_parameters(num_months=240, max_eft=35, min_eft=5, max_height=135, min_height=60)
        manager.set_ground_loads_from_hourly_list(self.get_atlanta_loads())
        manager.set_geometry_constraints_rectangular(length=85.0, width=36.5, b_min=3.0, b_max=10.0)
        manager.set_design(flow_rate=0.2, flow_type="borehole", design_method_geo=DesignMethodGeometry.Rectangular)
        manager.find_design()
        self.assertAlmostEqual(123.26, manager.u_tube_height, delta=0.01)
        self.assertEqual(180, len(manager._search.selected_coordinates))

    def test_double_u_tube(self):
        manager = GHEManager()
        manager.set_double_u_tube_pipe(inner_radius=(21.6 / 1000.0 / 2.0), outer_radius=(26.67 / 1000.0 / 2.0),
                                       shank_spacing=(32.3 / 1000.0),
                                       roughness=1.0e-6, conductivity=0.4, rho_cp=(1542.0 * 1000.0))

        manager.set_soil(conductivity=2.0, rho_cp=(2343.493 * 1000.0), undisturbed_temp=18.3)
        manager.set_grout(conductivity=1.0, rho_cp=(3901.0 * 1000.0))
        manager.set_fluid()
        manager.set_borehole(length=96.0, buried_depth=2.0, radius=0.075)
        manager.set_simulation_parameters(num_months=240, max_eft=35, min_eft=5, max_height=135, min_height=60)
        manager.set_ground_loads_from_hourly_list(self.get_atlanta_loads())
        manager.set_geometry_constraints_rectangular(length=85.0, width=36.5, b_min=3.0, b_max=10.0)
        manager.set_design(flow_rate=0.2, flow_type="borehole", design_method_geo=DesignMethodGeometry.Rectangular)
        manager.find_design()
        self.assertAlmostEqual(127.66, manager.u_tube_height, delta=0.01)
        self.assertEqual(144, len(manager._search.selected_coordinates))

    def test_coaxial_pipe(self):
        manager = GHEManager()
        manager.set_coaxial_pipe(inner_pipe_r_in=(44.2 / 1000.0 / 2.0), inner_pipe_r_out=(50.0 / 1000.0 / 2.0),
                                 outer_pipe_r_in=(97.4 / 1000.0 / 2.0), outer_pipe_r_out=(110.0 / 1000.0 / 2.0),
                                 roughness=1.0e-6, conductivity_inner=0.4, conductivity_outer=0.4,
                                 rho_cp=(1542.0 * 1000.0))

        manager.set_soil(conductivity=2.0, rho_cp=(2343.493 * 1000.0), undisturbed_temp=18.3)
        manager.set_grout(conductivity=1.0, rho_cp=(3901.0 * 1000.0))
        manager.set_fluid()
        manager.set_borehole(length=96.0, buried_depth=2.0, radius=0.075)
        manager.set_simulation_parameters(num_months=240, max_eft=35, min_eft=5, max_height=135, min_height=60)
        manager.set_ground_loads_from_hourly_list(self.get_atlanta_loads())
        manager.set_geometry_constraints_rectangular(length=85.0, width=36.5, b_min=3.0, b_max=10.0)
        manager.set_design(flow_rate=0.2, flow_type="borehole", design_method_geo=DesignMethodGeometry.Rectangular)
        manager.find_design()
        self.assertAlmostEqual(132.58, manager.u_tube_height, delta=0.01)
        self.assertEqual(144, len(manager._search.selected_coordinates))
