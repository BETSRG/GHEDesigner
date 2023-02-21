from json import loads, dumps
from recursive_diff import recursive_eq

from ghedesigner.manager import GHEManager
from ghedesigner.tests.ghe_base_case import GHEBaseTest


class TestCreateInputFiles(GHEBaseTest):

    # def test_num_files_match_num_tests(self):

    #     # update this to match the number of tests in this file
    #     # there should be a corresponding test to each demo file
    #     num_tests = 7

    #     for _, _, files in os.walk(self.demos_path):
    #         print(files)
    #         self.assertTrue(len(files) == num_tests)

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

        # TODO: need to understand why this is here. this was just pulled from the original example file
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

        # TODO: need to understand why this is here. this was just pulled from the original example file
        ghe.set_geometry_constraints_near_square(b=5.0, length=100.0)
        ghe.set_design(flow_rate=0.2, flow_type="borehole", design_method_geo=ghe.DesignGeomType.NearSquare)
        out_path = self.test_outputs_directory / "test_create_input_file_near_square_double_u_tube.json"
        ghe.write_input_file(out_path)

        # get demo file
        demo_file = self.demos_path / 'find_design_near_square_double_u_tube.json'
        d_demo = loads(demo_file.read_text())

        with open(self.test_outputs_directory / 'find_design_near_square_double_u_tube.json', 'w') as f:
            f.write(dumps(d_demo, sort_keys=True, indent=2, separators=(',', ': ')))

        # get new file
        d_new = loads(out_path.read_text())

        # compare files
        recursive_eq(d_demo, d_new)
