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

    def test_design_selection(self):
        manager = GHEManager()
        manager.set_pipe(
            inner_radius=(21.6 / 1000.0 / 2.0), outer_radius=(26.67 / 1000.0 / 2.0), shank_spacing=(32.3 / 1000.0),
            roughness=1.0e-6, conductivity=0.4, rho_cp=(1542.0 * 1000.0)
        )
        manager.set_soil(
            conductivity=2.0, rho_cp=(2343.493 * 1000.0), undisturbed_temp=18.3
        )
        manager.set_grout(conductivity=1.0, rho_cp=(3901.0 * 1000.0))
        manager.set_fluid()  # defaults to water
        manager.set_borehole(length=96.0, buried_depth=2.0, radius=0.075)
        manager.set_simulation_parameters(
            num_months=240, max_eft=35, min_eft=5, max_height=135, min_height=60
        )
        with open('/tmp/blah.csv', 'w') as f:
            f.write(',\n'.join([str(x) for x in self.get_atlanta_loads()]))
        manager.set_ground_loads_from_hourly_list(self.get_atlanta_loads())
        manager.set_geometry_constraints(b=5.0, length=155)  # borehole spacing and field side length
        # perform a design search assuming "system" flow?
        manager.set_design(flow_rate=31.2, flow_type="system")
        manager.find_design()
        h_single_u_tube_a = manager.u_tube_height
        # perform a design search assuming "borehole" flow?
        manager.set_design(flow_rate=0.2, flow_type="borehole")
        manager.find_design()
        h_single_u_tube_b = manager.u_tube_height

        # Verify that the `flow` toggle is properly working
        self.assertAlmostEqual(h_single_u_tube_a, h_single_u_tube_b, places=8)
        # Verify that the proper height as been found
        # Note: This reference was calculated on macOS. It seems that on Linux
        # the values are not equal starting around the 9th decimal place.
        h_reference = 130.27
        self.assertAlmostEqual(h_reference, h_single_u_tube_a, delta=0.01)
