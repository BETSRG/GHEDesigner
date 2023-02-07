from ghedesigner.manager import GHEManager
from ghedesigner.tests.ghe_base_case import GHEBaseTest


# Pipe dimensions
# ---------------
# Single and Multiple U-tubes
# Coaxial tube
# r_in_in = 44.2 / 1000.0 / 2.0
# r_in_out = 50.0 / 1000.0 / 2.0
# # Outer pipe radii
# r_out_in = 97.4 / 1000.0 / 2.0
# r_out_out = 110.0 / 1000.0 / 2.0
# Pipe radii
# Note: This convention is different from pygfunction
# r_inner = [r_in_in, r_in_out]  # The radii of the inner pipe from in to out
# r_outer = [r_out_in, r_out_out]  # The radii of the outer pipe from in to out

# Pipe positions
# --------------
# Single U-tube BHE object
# Double U-tube
# pos_double = plat.Pipe.place_pipes(s, r_out, 2)
# double_u_tube = MultipleUTube
# Coaxial tube
# pos_coaxial = (0, 0)
# coaxial_tube = CoaxialPipe

class TestNearSquare(GHEBaseTest):

    def test_design_selection_system(self):
        ghe = GHEManager()
        ghe.set_single_u_tube_pipe(
            inner_radius=0.0108, outer_radius=0.013335, shank_spacing=0.0323,
            roughness=1.0e-6, conductivity=0.4, rho_cp=1542000.0
        )
        ghe.set_soil(conductivity=2.0, rho_cp=2343493.0, undisturbed_temp=18.3)
        ghe.set_grout(conductivity=1.0, rho_cp=3901000.0)
        ghe.set_fluid()
        ghe.set_borehole(length=96.0, buried_depth=2.0, radius=0.075)
        ghe.set_simulation_parameters(num_months=240, max_eft=35, min_eft=5, max_height=135, min_height=60)
        ghe.set_ground_loads_from_hourly_list(self.get_atlanta_loads())
        ghe.set_geometry_constraints_near_square(b=5.0, length=155)  # borehole spacing and field side length
        # perform a design search assuming "system" flow?
        ghe.set_design(flow_rate=31.2, flow_type="system", design_method_geo=ghe.DesignGeomType.NearSquare)
        ghe.find_design()
        output_file_directory = self.test_outputs_directory / "TestDesignSelectionSystem"
        outputs = ghe.collect_outputs("Project Name", "Notes", "Author", "Iteration Name", output_file_directory)
        u_tube_height = outputs['ghe_system']['active_borehole_length']
        self.assertAlmostEqual(130.27, u_tube_height, delta=0.01)

    def test_design_selection_borehole(self):
        ghe = GHEManager()
        ghe.set_single_u_tube_pipe(
            inner_radius=0.0108, outer_radius=0.013335, shank_spacing=0.0323,
            roughness=1.0e-6, conductivity=0.4, rho_cp=1542000.0
        )
        ghe.set_soil(conductivity=2.0, rho_cp=2343493.0, undisturbed_temp=18.3)
        ghe.set_grout(conductivity=1.0, rho_cp=3901000.0)
        ghe.set_fluid()
        ghe.set_borehole(length=96.0, buried_depth=2.0, radius=0.075)
        ghe.set_simulation_parameters(num_months=240, max_eft=35, min_eft=5, max_height=135, min_height=60)
        ghe.set_ground_loads_from_hourly_list(self.get_atlanta_loads())
        ghe.set_geometry_constraints_near_square(b=5.0, length=155)  # borehole spacing and field side length
        # perform a design search assuming "borehole" flow?
        ghe.set_design(flow_rate=0.2, flow_type="borehole", design_method_geo=ghe.DesignGeomType.NearSquare)
        ghe.find_design()
        output_file_directory = self.test_outputs_directory / "TestDesignSelectionBorehole"
        outputs = ghe.collect_outputs("Project Name", "Notes", "Author", "Iteration Name", output_file_directory)
        u_tube_height = outputs['ghe_system']['active_borehole_length']
        self.assertAlmostEqual(130.27, u_tube_height, delta=0.01)
