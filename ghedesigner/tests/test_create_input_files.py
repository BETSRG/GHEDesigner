import os
from json import loads, dumps

from recursive_diff import recursive_eq

from ghedesigner.manager import GHEManager
from ghedesigner.tests.ghe_base_case import GHEBaseTest


class TestCreateInputFiles(GHEBaseTest):

    def test_num_files_match_num_tests(self):
        # update this to match the number of tests in this file
        # there should be a test corresponding to each demo file
        num_tests = 10

        for _, _, files in os.walk(self.demos_path):
            self.assertTrue(len(files) == num_tests)

    def test_create_input_file_rowwise_single_u_tube(self):
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
            [53.19620253, 112.3291139]
        ]

        no_go_zones = [[
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
            [95.7964135, 81.7721519]
        ]]

        ghe = GHEManager()
        ghe.set_single_u_tube_pipe(
            inner_radius=0.0108, outer_radius=0.0133, shank_spacing=0.0323,
            roughness=1.0e-6, conductivity=0.4, rho_cp=1542000.0)
        ghe.set_soil(conductivity=2.0, rho_cp=2343493.0, undisturbed_temp=18.3)
        ghe.set_grout(conductivity=1.0, rho_cp=3901000.0)
        ghe.set_fluid()
        ghe.set_borehole(height=96.0, buried_depth=2.0, radius=0.075)
        ghe.set_simulation_parameters(num_months=240, max_eft=35, min_eft=5, max_height=135, min_height=60)
        ghe.set_ground_loads_from_hourly_list(self.get_atlanta_loads())
        ghe.set_geometry_constraints_rowwise(perimeter_spacing_ratio=0.8,
                                             spacing_start=10.0, spacing_stop=20.0, spacing_step=0.1,
                                             rotate_start=-90.0, rotate_stop=0.0, rotate_step=0.5,
                                             property_boundary=prop_boundary, no_go_boundaries=no_go_zones)
        ghe.set_design(flow_rate=0.2, flow_type="borehole", design_method_geo=ghe.DesignGeomType.RowWise)
        out_path = self.test_outputs_directory / "test_create_input_file_rowwise_single_u_tube.json"
        ghe.write_input_file(out_path)

        # get demo file
        demo_file = self.demos_path / 'find_design_rowwise_single_u_tube.json'
        d_demo = loads(demo_file.read_text())

        # uncomment to write formatted demo file in the test_output directory
        # with open(self.test_outputs_directory / 'find_design_bi_zoned_rectangle_single_u_tube.json', 'w') as f:
        #     f.write(dumps(d_demo, sort_keys=True, indent=2, separators=(',', ': ')))

        # get new file
        d_new = loads(out_path.read_text())

        # compare files
        recursive_eq(d_demo, d_new)

    def test_create_input_file_bi_rectangle_constrained_single_u_tube(self):
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
            [53.19620253, 112.3291139]
        ]

        no_go_zones = [[
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
            [95.7964135, 81.7721519]
        ]]

        ghe = GHEManager()
        ghe.set_single_u_tube_pipe(
            inner_radius=0.0108, outer_radius=0.0133, shank_spacing=0.0323,
            roughness=1.0e-6, conductivity=0.4, rho_cp=1542000.0)
        ghe.set_soil(conductivity=2.0, rho_cp=2343493.0, undisturbed_temp=18.3)
        ghe.set_grout(conductivity=1.0, rho_cp=3901000.0)
        ghe.set_fluid()
        ghe.set_borehole(height=96.0, buried_depth=2.0, radius=0.075)
        ghe.set_simulation_parameters(num_months=240, max_eft=35, min_eft=5, max_height=135, min_height=60)
        ghe.set_ground_loads_from_hourly_list(self.get_atlanta_loads())
        ghe.set_geometry_constraints_bi_rectangle_constrained(b_min=5.0, b_max_x=25.0, b_max_y=25.0,
                                                              property_boundary=prop_boundary,
                                                              no_go_boundaries=no_go_zones)
        ghe.set_design(flow_rate=0.2, flow_type="borehole", design_method_geo=ghe.DesignGeomType.BiRectangleConstrained)
        out_path = self.test_outputs_directory / "test_create_input_file_bi_rectangle_constrained_single_u_tube.json"
        ghe.write_input_file(out_path)

        # get demo file
        demo_file = self.demos_path / 'find_design_bi_rectangle_constrained_single_u_tube.json'
        d_demo = loads(demo_file.read_text())

        # uncomment to write formatted demo file in the test_output directory
        # with open(self.test_outputs_directory / 'find_design_bi_zoned_rectangle_single_u_tube.json', 'w') as f:
        #     f.write(dumps(d_demo, sort_keys=True, indent=2, separators=(',', ': ')))

        # get new file
        d_new = loads(out_path.read_text())

        # compare files
        recursive_eq(d_demo, d_new)

    def test_create_input_file_bi_zoned_rectangle_single_u_tube(self):
        ghe = GHEManager()
        ghe.set_single_u_tube_pipe(
            inner_radius=0.0108, outer_radius=0.0133, shank_spacing=0.0323,
            roughness=1.0e-6, conductivity=0.4, rho_cp=1542000.0)
        ghe.set_soil(conductivity=2.0, rho_cp=2343493.0, undisturbed_temp=18.3)
        ghe.set_grout(conductivity=1.0, rho_cp=3901000.0)
        ghe.set_fluid()
        ghe.set_borehole(height=96.0, buried_depth=2.0, radius=0.075)
        ghe.set_simulation_parameters(num_months=240, max_eft=35, min_eft=5, max_height=135, min_height=60)
        ghe.set_ground_loads_from_hourly_list(self.get_atlanta_loads())
        ghe.set_geometry_constraints_bi_zoned_rectangle(length=100, width=100.0, b_min=2.0, b_max_x=10.0, b_max_y=12.0)
        ghe.set_design(flow_rate=0.2, flow_type="borehole", design_method_geo=ghe.DesignGeomType.BiZonedRectangle)
        out_path = self.test_outputs_directory / "test_create_input_file_bi_zoned_rectangle_single_u_tube.json"
        ghe.write_input_file(out_path)

        # get demo file
        demo_file = self.demos_path / 'find_design_bi_zoned_rectangle_single_u_tube.json'
        d_demo = loads(demo_file.read_text())

        # uncomment to write formatted demo file in the test_output directory
        # with open(self.test_outputs_directory / 'find_design_bi_zoned_rectangle_single_u_tube.json', 'w') as f:
        #     f.write(dumps(d_demo, sort_keys=True, indent=2, separators=(',', ': ')))

        # get new file
        d_new = loads(out_path.read_text())

        # compare files
        recursive_eq(d_demo, d_new)

    def test_create_input_file_bi_rectangle_single_u_tube(self):
        ghe = GHEManager()
        ghe.set_single_u_tube_pipe(
            inner_radius=0.0108, outer_radius=0.0133, shank_spacing=0.0323,
            roughness=1.0e-6, conductivity=0.4, rho_cp=1542000.0)
        ghe.set_soil(conductivity=2.0, rho_cp=2343493.0, undisturbed_temp=18.3)
        ghe.set_grout(conductivity=1.0, rho_cp=3901000.0)
        ghe.set_fluid()
        ghe.set_borehole(height=96.0, buried_depth=2.0, radius=0.075)
        ghe.set_simulation_parameters(num_months=240, max_eft=35, min_eft=5, max_height=135, min_height=60)
        ghe.set_ground_loads_from_hourly_list(self.get_atlanta_loads())
        ghe.set_geometry_constraints_bi_rectangle(length=100, width=100.0, b_min=2.0, b_max_x=10.0, b_max_y=12.0)
        ghe.set_design(flow_rate=0.2, flow_type="borehole", design_method_geo=ghe.DesignGeomType.BiRectangle)
        out_path = self.test_outputs_directory / "test_create_input_file_bi_rectangle_single_u_tube.json"
        ghe.write_input_file(out_path)

        # get demo file
        demo_file = self.demos_path / 'find_design_bi_rectangle_single_u_tube.json'
        d_demo = loads(demo_file.read_text())

        # get new file
        d_new = loads(out_path.read_text())

        # compare files
        recursive_eq(d_demo, d_new)

    def test_create_input_file_near_square_coaxial(self):
        ghe = GHEManager()
        ghe.set_coaxial_pipe(
            inner_pipe_r_in=0.0221, inner_pipe_r_out=0.025, outer_pipe_r_in=0.0487, outer_pipe_r_out=0.055,
            roughness=1.0e-6, conductivity_inner=0.4, conductivity_outer=0.4, rho_cp=1542000.0)
        ghe.set_soil(conductivity=2.0, rho_cp=2343493.0, undisturbed_temp=18.3)
        ghe.set_grout(conductivity=1.0, rho_cp=3901000.0)
        ghe.set_fluid()
        ghe.set_borehole(height=96.0, buried_depth=2.0, radius=0.075)
        ghe.set_simulation_parameters(num_months=240, max_eft=35, min_eft=5, max_height=135, min_height=60)
        ghe.set_ground_loads_from_hourly_list(self.get_atlanta_loads())
        ghe.set_geometry_constraints_near_square(b=5.0, length=100.0)
        ghe.set_design(flow_rate=0.2, flow_type="borehole", design_method_geo=ghe.DesignGeomType.NearSquare)
        out_path = self.test_outputs_directory / "test_create_input_file_near_square_coaxial.json"
        ghe.write_input_file(out_path)

        # get demo file
        demo_file = self.demos_path / 'find_design_near_square_coaxial.json'
        d_demo = loads(demo_file.read_text())

        # get new file
        d_new = loads(out_path.read_text())

        # compare files
        recursive_eq(d_demo, d_new)

    def test_create_input_file_near_square_double_u_tube(self):
        ghe = GHEManager()
        ghe.set_double_u_tube_pipe(inner_radius=0.0108, outer_radius=0.0133, shank_spacing=0.0323,
                                   roughness=1.0e-6, conductivity=0.4, rho_cp=1542000.0)
        ghe.set_soil(conductivity=2.0, rho_cp=2343493.0, undisturbed_temp=18.3)
        ghe.set_grout(conductivity=1.0, rho_cp=3901000.0)
        ghe.set_fluid()
        ghe.set_borehole(height=96.0, buried_depth=2.0, radius=0.075)
        ghe.set_simulation_parameters(num_months=240, max_eft=35, min_eft=5, max_height=135, min_height=60)
        ghe.set_ground_loads_from_hourly_list(self.get_atlanta_loads())
        ghe.set_geometry_constraints_near_square(b=5.0, length=100.0)
        ghe.set_design(flow_rate=0.2, flow_type="borehole", design_method_geo=ghe.DesignGeomType.NearSquare)
        out_path = self.test_outputs_directory / "test_create_input_file_near_square_double_u_tube.json"
        ghe.write_input_file(out_path)

        # get demo file
        demo_file = self.demos_path / 'find_design_near_square_double_u_tube.json'
        d_demo = loads(demo_file.read_text())

        # get new file
        d_new = loads(out_path.read_text())

        # compare files
        recursive_eq(d_demo, d_new)

    def test_create_input_file_near_square_single_u_tube(self):
        ghe = GHEManager()
        ghe.set_single_u_tube_pipe(
            inner_radius=0.0108, outer_radius=0.0133,
            shank_spacing=0.0323, roughness=1.0e-6, conductivity=0.4, rho_cp=1542000.0
        )
        ghe.set_soil(conductivity=2.0, rho_cp=2343493.0, undisturbed_temp=18.3)
        ghe.set_grout(conductivity=1.0, rho_cp=3901000.0)
        ghe.set_fluid()
        ghe.set_borehole(height=96.0, buried_depth=2.0, radius=0.075)
        ghe.set_simulation_parameters(num_months=240, max_eft=35, min_eft=5, max_height=135, min_height=60)
        ghe.set_ground_loads_from_hourly_list(self.get_atlanta_loads())
        ghe.set_geometry_constraints_near_square(b=5, length=100)
        ghe.set_design(flow_rate=0.2, flow_type="borehole", design_method_geo=ghe.DesignGeomType.NearSquare)
        out_path = self.test_outputs_directory / "test_create_input_file_near_square_single_u_tube.json"
        ghe.write_input_file(out_path)

        # get demo file
        demo_file = self.demos_path / 'find_design_near_square_single_u_tube.json'
        d_demo = loads(demo_file.read_text())

        # get new file
        d_new = loads(out_path.read_text())

        # compare files
        recursive_eq(d_demo, d_new)

    def test_create_input_file_rectangle_coaxial(self):
        ghe = GHEManager()
        ghe.set_coaxial_pipe(
            inner_pipe_r_in=0.0221, inner_pipe_r_out=0.025, outer_pipe_r_in=0.0487, outer_pipe_r_out=0.055,
            roughness=1.0e-6, conductivity_inner=0.4, conductivity_outer=0.4, rho_cp=1542000.0)
        ghe.set_soil(conductivity=2.0, rho_cp=2343493.0, undisturbed_temp=18.3)
        ghe.set_grout(conductivity=1.0, rho_cp=3901000.0)
        ghe.set_fluid()
        ghe.set_borehole(height=96.0, buried_depth=2.0, radius=0.075)
        ghe.set_simulation_parameters(num_months=240, max_eft=35, min_eft=5, max_height=100, min_height=60)
        ghe.set_ground_loads_from_hourly_list(self.get_atlanta_loads())
        ghe.set_geometry_constraints_rectangle(length=100.0, width=100., b_min=3.0, b_max=10.0)
        ghe.set_design(flow_rate=0.2, flow_type="borehole", design_method_geo=ghe.DesignGeomType.Rectangle)
        out_path = self.test_outputs_directory / "test_create_input_file_rectangle_coaxial.json"
        ghe.write_input_file(out_path)

        # get demo file
        demo_file = self.demos_path / 'find_design_rectangle_coaxial.json'
        d_demo = loads(demo_file.read_text())

        # get new file
        d_new = loads(out_path.read_text())

        # compare files
        recursive_eq(d_demo, d_new)

    def test_create_input_file_rectangle_double_u_tube(self):
        ghe = GHEManager()
        ghe.set_double_u_tube_pipe(
            inner_radius=0.0108, outer_radius=0.0133, shank_spacing=0.0323,
            roughness=1.0e-6, conductivity=0.4, rho_cp=1542000.0)
        ghe.set_soil(conductivity=2.0, rho_cp=2343493.0, undisturbed_temp=18.3)
        ghe.set_grout(conductivity=1.0, rho_cp=3901000.0)
        ghe.set_fluid()
        ghe.set_borehole(height=96.0, buried_depth=2.0, radius=0.075)
        ghe.set_simulation_parameters(num_months=240, max_eft=35, min_eft=5, max_height=135, min_height=60)
        ghe.set_ground_loads_from_hourly_list(self.get_atlanta_loads())
        ghe.set_geometry_constraints_rectangle(length=100.0, width=100.0, b_min=3.0, b_max=10.0)
        ghe.set_design(flow_rate=0.2, flow_type="borehole", design_method_geo=ghe.DesignGeomType.Rectangle)
        out_path = self.test_outputs_directory / "test_create_input_file_rectangle_double_u_tube.json"
        ghe.write_input_file(out_path)

        # get demo file
        demo_file = self.demos_path / 'find_design_rectangle_double_u_tube.json'
        d_demo = loads(demo_file.read_text())

        # get new file
        d_new = loads(out_path.read_text())

        # compare files
        recursive_eq(d_demo, d_new)

    def test_create_input_file_rectangle_single_u_tube(self):
        ghe = GHEManager()
        ghe.set_single_u_tube_pipe(
            inner_radius=0.0108, outer_radius=0.0133, shank_spacing=0.0323,
            roughness=1.0e-6, conductivity=0.4, rho_cp=1542000.0)
        ghe.set_soil(conductivity=2.0, rho_cp=2343493.0, undisturbed_temp=18.3)
        ghe.set_grout(conductivity=1.0, rho_cp=3901000.0)
        ghe.set_fluid()
        ghe.set_borehole(height=96.0, buried_depth=2.0, radius=0.075)
        ghe.set_simulation_parameters(num_months=240, max_eft=35, min_eft=5, max_height=135, min_height=60)
        ghe.set_ground_loads_from_hourly_list(self.get_atlanta_loads())
        ghe.set_geometry_constraints_rectangle(length=100.0, width=100.0, b_min=3.0, b_max=10.0)
        ghe.set_design(flow_rate=0.2, flow_type="borehole", design_method_geo=ghe.DesignGeomType.Rectangle)
        out_path = self.test_outputs_directory / "test_create_input_file_rectangle_single_u_tube.json"
        ghe.write_input_file(out_path)

        # get demo file
        demo_file = self.demos_path / 'find_design_rectangle_single_u_tube.json'
        d_demo = loads(demo_file.read_text())

        with open(self.test_outputs_directory / 'find_design_rectangle_single_u_tube.json', 'w') as f:
            f.write(dumps(d_demo, sort_keys=True, indent=2, separators=(',', ': ')))

        # get new file
        d_new = loads(out_path.read_text())

        # compare files
        recursive_eq(d_demo, d_new)
