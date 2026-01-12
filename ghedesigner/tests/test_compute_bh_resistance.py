from ghedesigner.enums import DoubleUTubeConnType
from ghedesigner.ghe.boreholes.coaxial_borehole import CoaxialPipe
from ghedesigner.ghe.boreholes.core import Borehole
from ghedesigner.ghe.boreholes.multi_u_borehole import MultipleUTube
from ghedesigner.ghe.boreholes.single_u_borehole import SingleUTube
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.media import Fluid, Grout, Soil
from ghedesigner.tests.test_base_case import GHEBaseTest


class TestBHResistance(GHEBaseTest):
    def test_bh_resistance_coaxial(self):
        # Borehole dimensions
        h = 100.0  # Borehole length (m)
        d = 2.0  # Borehole buried depth (m)
        dia = 150.0 / 1000.0  # Borehole diameter

        # Pipe dimensions
        # Inner pipe radii
        r_in_in = 44.2 / 1000.0 / 2.0
        r_in_out = 50.0 / 1000.0 / 2.0
        # Outer pipe radii
        r_out_in = 97.4 / 1000.0 / 2.0
        r_out_out = 110.0 / 1000.0 / 2.0
        # Pipe radii
        # Note: This convention is different from pygfunction
        epsilon = 1.0e-6  # Pipe roughness (m)

        # Thermal properties
        k_p = (0.4, 0.4)  # Inner and outer pipe thermal conductivity (W/m.K)
        k_s = 2.0  # Ground thermal conductivity (W/m.K)
        k_g = 1.0  # Grout thermal conductivity (W/m.K)

        # Volumetric heat capacities
        rho_cp_p = 1542000.0  # Pipe volumetric heat capacity (J/K.m3)
        rho_cp_s = 2343493.0  # Soil volumetric heat capacity (J/K.m3)
        rho_cp_g = 3901000.0  # Grout volumetric heat capacity (J/K.m3)

        # Thermal properties
        # Pipe
        pipe = Pipe.init_coaxial(
            conductivity=k_p,
            inner_pipe_d_in=r_in_in * 2,
            inner_pipe_d_out=r_in_out * 2,
            outer_pipe_d_in=r_out_in * 2,
            outer_pipe_d_out=r_out_out * 2,
            roughness=epsilon,
            rho_cp=rho_cp_p,
        )
        # Soil
        ugt = 18.3  # Undisturbed ground temperature (degrees Celsius)
        soil = Soil(k_s, rho_cp_s, ugt)
        # Grout
        grout = Grout(k_g, rho_cp_g)

        # Fluid properties
        fluid = Fluid(fluid_name="Water", percent=0.0)
        v_flow_borehole = 0.8  # Volumetric flow rate per borehole (L/s)
        # Total fluid mass flow rate per borehole (kg/s)
        m_flow_borehole = v_flow_borehole / 1000.0 * fluid.rho

        # Define a borehole
        borehole = Borehole(borehole_height=h, burial_depth=d, borehole_radius=dia / 2.0)
        coaxial = CoaxialPipe(m_flow_borehole, fluid, borehole, pipe, grout, soil)
        r_b = coaxial.calc_effective_borehole_resistance()

        assert self.rel_error_within_tol(r_b, 0.1086, 0.01)

    def test_bh_resistance_double_u_tube(self):
        # borehole
        h = 100.0  # borehole length (m)
        d = 2.0  # borehole buried depth (m)
        dia = 0.150  # borehole diameter (m)
        borehole = Borehole(borehole_height=h, burial_depth=d, borehole_radius=dia / 2.0)

        # pipe
        d_out = 0.02667  # pipe outer diameter (m)
        d_in = 0.0216  # pipe inner diameter (m)
        r_in = d_in / 2.0
        s = 0.0323  # shank spacing (m)
        k_p = 0.4  # pipe thermal conductivity (W/m.K)
        rho_cp_p = 1542000.0  # pipe volumetric heat capacity (J/K.m3)
        epsilon = 1.0e-6  # pipe roughness (m)

        pipe = Pipe.init_double_u_tube_series(
            conductivity=k_p,
            inner_diameter=d_in,
            outer_diameter=d_out,
            shank_spacing=s,
            roughness=epsilon,
            rho_cp=rho_cp_p,
        )

        # soil
        k_s = 2.0  # Ground thermal conductivity (W/m.K)
        rho_cp_s = 2343493.0  # Soil volumetric heat capacity (J/K.m3)
        ugt = 18.3  # Undisturbed ground temperature (degrees Celsius)
        soil = Soil(k_s, rho_cp_s, ugt)

        # grout
        k_g = 1.0  # Grout thermal conductivity (W/m.K)
        rho_cp_g = 3901000.0  # Grout volumetric heat capacity (J/K.m3)
        grout = Grout(k_g, rho_cp_g)

        # fluid
        fluid = Fluid(fluid_name="Water", percent=0.0)

        # U-tubes
        v_flow_borehole = 0.2  # volumetric flow rate per borehole (L/s)
        m_flow_borehole = v_flow_borehole / 1000.0 * fluid.rho  # mass flow rate (kg/s)

        # Series
        double_u_tube_series = MultipleUTube(
            m_flow_borehole, fluid, borehole, pipe, grout, soil, DoubleUTubeConnType.SERIES
        )
        r_b_series = double_u_tube_series.calc_effective_borehole_resistance()
        re = MultipleUTube.compute_reynolds(double_u_tube_series.m_flow_pipe, r_in, fluid)

        # TODO: evaluate whether these tests are still needed
        # test values pinned to current performance because GLHEPro doesn't offer a series connection
        assert self.rel_error_within_tol(re, 11744.0, 0.01)
        assert self.rel_error_within_tol(r_b_series, 0.1597, 0.01)

        # Parallel
        double_u_tube_parallel = MultipleUTube(
            m_flow_borehole, fluid, borehole, pipe, grout, soil, DoubleUTubeConnType.PARALLEL
        )
        r_b_parallel = double_u_tube_parallel.calc_effective_borehole_resistance()
        re = MultipleUTube.compute_reynolds(double_u_tube_parallel.m_flow_pipe, r_in, fluid)

        # test values from GLHEPro v5.1
        assert self.rel_error_within_tol(re, 5820.0, 0.01)
        assert self.rel_error_within_tol(r_b_parallel, 0.1591, 0.005)

    def test_bh_resistance_single_u_tube(self):
        # Borehole dimensions
        h = 100.0  # Borehole length (m)
        d = 2.0  # Borehole buried depth (m)
        dia = 150.0 / 1000.0  # Borehole diameter

        # Pipe dimensions
        d_out = 0.02667  # Pipe outer diameter (m)
        d_in = 0.0216  # Pipe inner diameter (m)
        r_in = d_in / 2.0
        s = 0.0323  # Inner-tube to inner-tube Shank spacing (m)
        epsilon = 1.0e-6  # Pipe roughness (m)

        # Pipe positions
        # Single U-tube [(x_in, y_in), (x_out, y_out)]

        # Thermal conductivities
        k_p = 0.4  # Pipe thermal conductivity (W/m.K)
        k_s = 2.0  # Ground thermal conductivity (W/m.K)
        k_g = 1.0  # Grout thermal conductivity (W/m.K)

        # Volumetric heat capacities
        rho_cp_p = 1542000.0  # Pipe volumetric heat capacity (J/K.m3)
        rho_cp_s = 2343493.0  # Soil volumetric heat capacity (J/K.m3)
        rho_cp_g = 3901000.0  # Grout volumetric heat capacity (J/K.m3)

        # Thermal properties
        # Pipe
        pipe = Pipe.init_single_u_tube(
            conductivity=k_p,
            inner_diameter=d_in,
            outer_diameter=d_out,
            shank_spacing=s,
            roughness=epsilon,
            rho_cp=rho_cp_p,
        )
        # Soil
        ugt = 18.3  # Undisturbed ground temperature (degrees Celsius)
        soil = Soil(k_s, rho_cp_s, ugt)
        # Grout
        grout = Grout(k_g, rho_cp_g)

        # Fluid properties
        fluid = Fluid(fluid_name="Water", percent=0.0)
        v_flow_borehole = 0.2  # Volumetric flow rate per borehole (L/s)
        # Total fluid mass flow rate per borehole (kg/s)
        m_flow_borehole = v_flow_borehole / 1000.0 * fluid.rho

        # Define a borehole
        borehole = Borehole(borehole_height=h, burial_depth=d, borehole_radius=dia / 2.0)

        single_u_tube = SingleUTube(m_flow_borehole, fluid, borehole, pipe, grout, soil)

        # Intermediate variables
        re = SingleUTube.compute_reynolds(single_u_tube.m_flow_borehole, r_in, fluid)

        r_b = single_u_tube.calc_effective_borehole_resistance()

        # comparison values from GLHEPro v5.1
        assert self.rel_error_within_tol(re, 11748.0, 0.005)
        assert self.rel_error_within_tol(r_b, 0.2073, 0.005)

    def test_bh_resistance_validation(self):
        # Dictionary for storing PLAT variations
        borehole_values = {"Single U-tube": {}, "Double U-tube": {}, "Coaxial": {}}
        # Borehole dimensions
        h = 100.0  # Borehole length (m)
        d = 2.0  # Borehole buried depth (m)
        dia = 150.0 / 1000.0  # Borehole diameter

        # Pipe dimensions
        d_out = 0.02667  # Pipe outer diameter (m)
        d_in = 0.0216  # Pipe inner diameter (m)
        r_in = d_in / 2.0
        s = 0.0323  # Inner-tube to inner-tube Shank spacing (m)
        epsilon = 1.0e-6  # Pipe roughness (m)

        # Thermal conductivities
        k_p = 0.4  # Pipe thermal conductivity (W/m.K)
        k_s = 2.0  # Ground thermal conductivity (W/m.K)
        k_g = 1.0  # Grout thermal conductivity (W/m.K)

        # Volumetric heat capacities
        rho_cp_p = 1542000.0  # Pipe volumetric heat capacity (J/K.m3)
        rho_cp_s = 2343493.0  # Soil volumetric heat capacity (J/K.m3)
        rho_cp_g = 3901000.0  # Grout volumetric heat capacity (J/K.m3)

        # Thermal properties
        # Pipe
        pipe_s = Pipe.init_single_u_tube(
            conductivity=k_p,
            inner_diameter=d_in,
            outer_diameter=d_out,
            shank_spacing=s,
            roughness=epsilon,
            rho_cp=rho_cp_p,
        )
        pipe_d = Pipe.init_double_u_tube_parallel(
            conductivity=k_p,
            inner_diameter=d_in,
            outer_diameter=d_out,
            shank_spacing=s,
            roughness=epsilon,
            rho_cp=rho_cp_p,
        )
        # Soil
        ugt = 18.3  # Undisturbed ground temperature (degrees Celsius)
        soil = Soil(k_s, rho_cp_s, ugt)
        # Grout
        grout = Grout(k_g, rho_cp_g)

        # Fluid properties
        fluid = Fluid(fluid_name="Water", percent=0.0)

        # A list of volumetric flow rates to check borehole resistances for (L/s)
        v_flow_rates = [0.3, 0.2, 0.18, 0.15, 0.12, 0.1, 0.08, 0.07, 0.06, 0.05]
        v_flow_rates_coaxial = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.29, 0.28, 0.27, 0.26]

        re_glhepro_single = [17621, 11748, 10573, 8811, 7049, 5874, 4699, 4112, 3524, 2937]
        rb_glhepro_single = [0.2021, 0.2073, 0.2094, 0.214, 0.2223, 0.232, 0.2491, 0.2629, 0.2833, 0.3199]

        re_glhepro_double = [8811, 5874, 5286, 4405, 3524, 2937, 2350, 2056, 1762, 1468]
        rb_glhepro_double = [0.1502, 0.1597, 0.1635, 0.1720, 0.1868, 0.2037, 0.2329, 0.2615, 0.2879, 0.3287]

        re_glhepro_coaxial = [8607, 7747, 6886, 6025, 5164, 4304, 3443, 2582, 2496, 2410, 2324, 2238]
        rb_glhepro_coaxial = [
            0.1052,
            0.1064,
            0.1086,
            0.1128,
            0.1211,
            0.1350,
            0.1515,
            0.1673,
            0.1691,
            0.1709,
            0.1768,
            0.1784,
        ]

        # Single and Double U-tubes
        borehole_values["Single U-tube"] = {"V_dot": [], "Rb": [], "Re": []}
        borehole_values["Double U-tube"] = {"V_dot": [], "Rb": [], "Re": []}

        for idx, v_flow_borehole in enumerate(v_flow_rates):
            # Store volumetric flow rates in (L/s)
            borehole_values["Single U-tube"]["V_dot"].append(v_flow_borehole)
            borehole_values["Double U-tube"]["V_dot"].append(v_flow_borehole)

            # Total fluid mass flow rate per borehole (kg/s)
            m_flow_borehole = v_flow_borehole / 1000.0 * fluid.rho

            # Define a borehole
            borehole = Borehole(borehole_height=h, burial_depth=d, borehole_radius=dia / 2.0)
            single_u_tube = SingleUTube(m_flow_borehole, fluid, borehole, pipe_s, grout, soil)

            # check Reynolds numbers
            re = SingleUTube.compute_reynolds(single_u_tube.m_flow_borehole, r_in, fluid)
            borehole_values["Single U-tube"]["Re"].append(re)
            assert self.rel_error_within_tol(re, re_glhepro_single[idx], 0.01)

            # check BH resistance
            resist_bh = single_u_tube.calc_effective_borehole_resistance()
            borehole_values["Single U-tube"]["Rb"].append(resist_bh)
            assert self.rel_error_within_tol(resist_bh, rb_glhepro_single[idx], 0.01)

            # Define a borehole
            double_u_tube_parallel = MultipleUTube(m_flow_borehole, fluid, borehole, pipe_d, grout, soil)

            # check Reynolds numbers
            re = MultipleUTube.compute_reynolds(double_u_tube_parallel.m_flow_pipe, r_in, fluid)
            borehole_values["Double U-tube"]["Re"].append(re)
            assert self.rel_error_within_tol(re, re_glhepro_double[idx], 0.01)

            # check BH resistance
            resist_bh = double_u_tube_parallel.calc_effective_borehole_resistance()
            borehole_values["Double U-tube"]["Rb"].append(resist_bh)
            assert self.rel_error_within_tol(resist_bh, rb_glhepro_double[idx], 0.05)

        # Pipe dimensions
        # Inner pipe radii
        r_in_in = 44.2 / 1000.0 / 2.0
        r_in_out = 50.0 / 1000.0 / 2.0
        # Outer pipe radii
        r_out_in = 97.4 / 1000.0 / 2.0
        r_out_out = 110.0 / 1000.0 / 2.0
        # Pipe radii
        # Note: This convention is different from pygfunction
        epsilon = 1.0e-6  # Pipe roughness (m)

        # Thermal properties
        k_p = (0.4, 0.4)  # Inner and outer pipe thermal conductivity (W/m.K)
        k_s = 2.0  # Ground thermal conductivity (W/m.K)
        k_g = 1.0  # Grout thermal conductivity (W/m.K)

        # Volumetric heat capacities
        rho_cp_p = 1542000.0  # Pipe volumetric heat capacity (J/K.m3)
        rho_cp_s = 2343493.0  # Soil volumetric heat capacity (J/K.m3)
        rho_cp_g = 3901000.0  # Grout volumetric heat capacity (J/K.m3)

        # Thermal properties
        # Pipe
        pipe = Pipe.init_coaxial(
            conductivity=k_p,
            inner_pipe_d_in=r_in_in * 2,
            inner_pipe_d_out=r_in_out * 2,
            outer_pipe_d_in=r_out_in * 2,
            outer_pipe_d_out=r_out_out * 2,
            roughness=epsilon,
            rho_cp=rho_cp_p,
        )

        # Soil
        ugt = 18.3  # Undisturbed ground temperature (degrees Celsius)
        soil = Soil(k_s, rho_cp_s, ugt)
        # Grout
        grout = Grout(k_g, rho_cp_g)

        borehole_values["Coaxial"] = {"V_dot": [], "Rb": [], "Re": []}
        for idx, v_flow_borehole in enumerate(v_flow_rates_coaxial):
            # Total fluid mass flow rate per borehole (kg/s)
            m_flow_borehole = v_flow_borehole / 1000.0 * fluid.rho
            borehole_values["Coaxial"]["V_dot"].append(v_flow_borehole)

            # Define a borehole
            borehole = Borehole(borehole_height=h, burial_depth=d, borehole_radius=dia / 2.0)
            coaxial = CoaxialPipe(m_flow_borehole, fluid, borehole, pipe, grout, soil)

            # check Reynolds number
            re = CoaxialPipe.compute_reynolds_concentric(coaxial.m_flow_borehole, r_in_out, r_out_in, fluid)
            borehole_values["Coaxial"]["Re"].append(re)
            assert self.rel_error_within_tol(re, re_glhepro_coaxial[idx], 0.01)

            resist_bh = coaxial.calc_effective_borehole_resistance()
            borehole_values["Coaxial"]["Rb"].append(resist_bh)

            if v_flow_borehole > 0.4:
                assert self.rel_error_within_tol(resist_bh, rb_glhepro_coaxial[idx], 0.02)
