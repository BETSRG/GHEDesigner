from ghedesigner.enums import BHType, TimestepType
from ghedesigner.ghe.boreholes.core import Borehole
from ghedesigner.ghe.coordinates import rectangle
from ghedesigner.ghe.gfunction import calc_g_func_for_multiple_lengths
from ghedesigner.ghe.ground_heat_exchangers import GHE
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.media import Fluid, Grout, Soil
from ghedesigner.tests.test_base_case import GHEBaseTest
from ghedesigner.utilities import eskilson_log_times


class TestLiveGFunctionSimAndSize(GHEBaseTest):
    def test_live_g_function_sim_and_size(self):
        # Borehole dimensions
        # -------------------
        h = 96.0  # Borehole length (m)
        d = 2.0  # Borehole buried depth (m)
        dia = 0.140  # Borehole diameter
        b = 5.0  # Borehole spacing (m)

        # Pipe dimensions
        # ---------------
        d_out = 0.04216  # Pipe outer diameter (m)
        d_in = 0.03404  # Pipe inner diameter (m)
        s = 0.01856  # Inner-tube to inner-tube Shank spacing (m)
        epsilon = 1.0e-6  # Pipe roughness (m)

        # Pipe positions
        # --------------
        # Single U-tube [(x_in, y_in), (x_out, y_out)]

        # Thermal conductivities
        # ----------------------
        k_p = 0.4  # Pipe thermal conductivity (W/m.K)
        k_s = 2.0  # Ground thermal conductivity (W/m.K)
        k_g = 1.0  # Grout thermal conductivity (W/m.K)

        # Volumetric heat capacities
        # --------------------------
        rho_cp_p = 1542000.0  # Pipe volumetric heat capacity (J/K.m3)
        rho_cp_s = 2343493.0  # Soil volumetric heat capacity (J/K.m3)
        rho_cp_g = 3901000.0  # Grout volumetric heat capacity (J/K.m3)

        # Thermal properties
        # ------------------
        # Pipe
        pipe = Pipe.init_single_u_tube(k_p, rho_cp_p, d_in, d_out, s, epsilon, 1)
        # Soil
        ugt = 18.3  # Undisturbed ground temperature (degrees Celsius)
        soil = Soil(k_s, rho_cp_s, ugt)
        # Grout
        grout = Grout(k_g, rho_cp_g)

        # Eskilson's original ln(t/ts) values
        log_time = eskilson_log_times()

        # Inputs related to fluid
        # -----------------------
        # Fluid properties
        fluid = Fluid(fluid_name="Water", percent=0.0)

        # Coordinates
        nx = 12
        ny = 13
        coordinates = rectangle(nx, ny, b, b)

        # Fluid properties
        v_flow_borehole = 0.5
        # System volumetric flow rate (L/s)
        v_flow_system = v_flow_borehole * float(nx * ny)
        # Total fluid mass flow rate per borehole (kg/s)
        m_flow_borehole = v_flow_borehole / 1000.0 * fluid.rho

        # Define a borehole
        borehole = Borehole(borehole_height=h, burial_depth=d, borehole_radius=dia / 2.0)

        # Simulation start month and end month
        # --------------------------------
        # Simulation start month and end month
        # start_month = 1
        n_years = 20
        num_months = n_years * 12

        # Process loads from file
        hourly_extraction_ground_loads = self.get_atlanta_loads()

        # Calculate a g-function for uniform inlet fluid temperature with
        g_function = calc_g_func_for_multiple_lengths(
            b,
            [h],
            dia / 2.0,
            d,
            m_flow_borehole,
            BHType.SINGLEUTUBE,
            log_time,
            coordinates,
            fluid,
            pipe,
            grout,
            soil,
        )

        # --------------------------------------------------------------------------

        # Initialize the GHE object
        ghe = GHE(
            v_flow_system,
            b,
            BHType.SINGLEUTUBE,
            fluid,
            borehole,
            pipe,
            grout,
            soil,
            g_function,
            1,
            num_months,
            hourly_extraction_ground_loads,
        )

        # Simulate after computing just one g-function
        max_hp_eft, min_hp_eft = ghe.simulate(method=TimestepType.HYBRID)

        self.log(f"Min EFT: {min_hp_eft:0.3f}\nMax EFT: {max_hp_eft:0.3f}")

        # Compute a range of g-functions for interpolation
        h_values = [24.0, 48.0, 96.0, 192.0, 384.0]
        bh_depth = 2.0

        g_function = calc_g_func_for_multiple_lengths(
            b,
            h_values,
            dia / 2.0,
            bh_depth,
            m_flow_borehole,
            BHType.SINGLEUTUBE,
            log_time,
            coordinates,
            fluid,
            pipe,
            grout,
            soil,
        )

        # Re-Initialize the GHE object
        ghe = GHE(
            v_flow_system,
            b,
            BHType.SINGLEUTUBE,
            fluid,
            borehole,
            pipe,
            grout,
            soil,
            g_function,
            1,
            num_months,
            hourly_extraction_ground_loads,
        )

        ghe.size(method=TimestepType.HYBRID, max_height=384, min_height=24, design_max_eft=35, design_min_eft=5)

        self.log(f"Height of boreholes: {ghe.bhe.borehole.H:0.4f}")
