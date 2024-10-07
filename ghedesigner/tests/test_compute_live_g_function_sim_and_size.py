from ghedesigner.borehole import GHEBorehole
from ghedesigner.coordinates import rectangle
from ghedesigner.enums import BHPipeType, TimestepType
from ghedesigner.gfunction import calc_g_func_for_multiple_lengths
from ghedesigner.ground_heat_exchangers import GHE
from ghedesigner.media import GHEFluid, Grout, Pipe, Soil
from ghedesigner.simulation import SimulationParameters
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
        r_out = d_out / 2.0
        r_in = d_in / 2.0
        s = 0.01856  # Inner-tube to inner-tube Shank spacing (m)
        epsilon = 1.0e-6  # Pipe roughness (m)

        # Pipe positions
        # --------------
        # Single U-tube [(x_in, y_in), (x_out, y_out)]
        pos = Pipe.place_pipes(s, r_out, 1)

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
        pipe = Pipe(pos, r_in, r_out, s, epsilon, k_p, rho_cp_p)
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
        fluid = GHEFluid(fluid_str="Water", percent=0.0)

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
        borehole = GHEBorehole(h, d, dia / 2.0, x=0.0, y=0.0)

        # Simulation start month and end month
        # --------------------------------
        # Simulation start month and end month
        start_month = 1
        n_years = 20
        end_month = n_years * 12

        sim_params = SimulationParameters(
            start_month,
            end_month,
        )

        sim_params.set_design_temps(35, 5)
        sim_params.set_design_heights(384, 24)

        # Process loads from file
        hourly_extraction_ground_loads = self.get_atlanta_loads()

        # Calculate a g-function for uniform inlet fluid temperature with
        g_function = calc_g_func_for_multiple_lengths(
            b,
            [h],
            dia / 2.0,
            d,
            m_flow_borehole,
            BHPipeType.SINGLEUTUBE,
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
            BHPipeType.SINGLEUTUBE,
            fluid,
            borehole,
            pipe,
            grout,
            soil,
            g_function,
            sim_params,
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
            BHPipeType.SINGLEUTUBE,
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
            BHPipeType.SINGLEUTUBE,
            fluid,
            borehole,
            pipe,
            grout,
            soil,
            g_function,
            sim_params,
            hourly_extraction_ground_loads,
        )

        ghe.size(method=TimestepType.HYBRID)

        self.log(f"Height of boreholes: {ghe.bhe.b.H:0.4f}")
