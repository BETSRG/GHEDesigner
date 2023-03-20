from ghedesigner.borehole import GHEBorehole
from ghedesigner.borehole_heat_exchangers import CoaxialPipe, MultipleUTube, SingleUTube
from ghedesigner.enums import DoubleUTubeConnType
from ghedesigner.media import Pipe, Grout, GHEFluid, Soil
from ghedesigner.tests.ghe_base_case import GHEBaseTest


class TestBHResistance(GHEBaseTest):
    def test_bh_resistance_coaxial(self):
        # Borehole dimensions
        h = 100.0  # Borehole length (m)
        d = 2.0  # Borehole buried depth (m)
        r_b = 150.0 / 1000.0 / 2.0  # Borehole radius

        # Pipe dimensions
        # Inner pipe radii
        r_in_in = 44.2 / 1000.0 / 2.0
        r_in_out = 50.0 / 1000.0 / 2.0
        # Outer pipe radii
        r_out_in = 97.4 / 1000.0 / 2.0
        r_out_out = 110.0 / 1000.0 / 2.0
        # Pipe radii
        # Note: This convention is different from pygfunction
        r_inner = [r_in_in, r_in_out]  # The radii of the inner pipe from in to out
        r_outer = [r_out_in, r_out_out]  # The radii of the outer pipe from in to out
        epsilon = 1.0e-6  # Pipe roughness (m)

        # Pipe positioning
        pos = (0, 0)
        s = 0

        # Thermal properties
        k_p = [0.4, 0.4]  # Inner and outer pipe thermal conductivity (W/m.K)
        k_s = 2.0  # Ground thermal conductivity (W/m.K)
        k_g = 1.0  # Grout thermal conductivity (W/m.K)

        # Volumetric heat capacities
        rho_cp_p = 1542000.0  # Pipe volumetric heat capacity (J/K.m3)
        rho_cp_s = 2343493.0  # Soil volumetric heat capacity (J/K.m3)
        rho_cp_g = 3901000.0  # Grout volumetric heat capacity (J/K.m3)

        # Thermal properties
        # Pipe
        pipe = Pipe(pos, r_inner, r_outer, s, epsilon, k_p, rho_cp_p)
        # Soil
        ugt = 18.3  # Undisturbed ground temperature (degrees Celsius)
        soil = Soil(k_s, rho_cp_s, ugt)
        # Grout
        grout = Grout(k_g, rho_cp_g)

        # Fluid properties
        fluid = GHEFluid(fluid_str="Water", percent=0.0)
        v_flow_borehole = 0.8  # Volumetric flow rate per borehole (L/s)
        # Total fluid mass flow rate per borehole (kg/s)
        m_flow_borehole = v_flow_borehole / 1000.0 * fluid.rho

        # Define a borehole
        borehole = GHEBorehole(h, d, r_b, x=0.0, y=0.0)
        coaxial = CoaxialPipe(m_flow_borehole, fluid, borehole, pipe, grout, soil)
        r_b = coaxial.calc_effective_borehole_resistance()

        self.log(coaxial)
        self.log("Intermediate Variables \n" + "-" * 30)
        self.log(f"Convection coefficient at inside of inner pipe (W/m2.K): {coaxial.h_f_in}")
        self.log(f"Convection coefficient at outside of inner pipe (W/m2.K): {coaxial.h_f_a_in}")
        self.log(f"Convection coefficient at inside of outer pipe (W/m2.K): {coaxial.h_f_a_out}")
        self.log("-" * 30)
        self.log(f"Convective resistance at inside of inner pipe (K/(W/m): {coaxial.R_f_in}")
        self.log(f"Conduction resistance of inner pipe (K/(W/m): {coaxial.R_p_in}")
        self.log(f"Convection resistance at outside of inner pipe (K/(W/m): {coaxial.R_f_a_in}")
        self.log(f"Convection resistance at inside of outer pipe (K/(W/m): {coaxial.R_f_a_out}")
        self.log(f"Conduction resistance of outer pipe (K/(W/m): {coaxial.R_p_out}")
        self.log(f"Inner fluid to inner annulus fluid resistance (K/(W/m): {coaxial.R_ff}")
        self.log(f"Outer annulus fluid to pipe thermal resistance (K/(W/m): {coaxial.R_fp}")

        self.log("-" * 30)
        self.log(f"Borehole thermal resistance: {r_b:0.4f} m.K/W")

        self.assertTrue(self.rel_error_within_tol(coaxial.h_f_in, 2255, 0.01))
        self.assertTrue(self.rel_error_within_tol(coaxial.h_f_a_in, 670, 0.15))
        self.assertTrue(self.rel_error_within_tol(r_b, 0.1078, 0.01))

    def test_bh_resistance_double_u_tube(self):
        # borehole
        h = 100.0  # borehole length (m)
        d = 2.0  # borehole buried depth (m)
        r_b = 0.075  # borehole radius (m)
        borehole = GHEBorehole(h, d, r_b, x=0.0, y=0.0)

        # pipe
        r_out = 0.013335  # pipe outer radius (m)
        r_in = 0.0108  # pipe inner radius (m)
        s = 0.0323  # shank spacing (m)
        pos = Pipe.place_pipes(s, r_out, 2)  # double U-tube # TODO: move from static method to borehole
        k_p = 0.4  # pipe thermal conductivity (W/m.K)
        rho_cp_p = 1542000.0  # pipe volumetric heat capacity (J/K.m3)
        epsilon = 1.0e-6  # pipe roughness (m)
        pipe = Pipe(pos, r_in, r_out, s, epsilon, k_p, rho_cp_p)

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
        fluid = GHEFluid(fluid_str="Water", percent=0.0)

        # U-tubes
        v_flow_borehole = 0.2  # volumetric flow rate per borehole (L/s)
        m_flow_borehole = v_flow_borehole / 1000.0 * fluid.rho  # mass flow rate (kg/s)

        # Series
        double_u_tube_series = MultipleUTube(m_flow_borehole, fluid, borehole, pipe, grout, soil,
                                             DoubleUTubeConnType.SERIES)
        r_b_series = double_u_tube_series.calc_effective_borehole_resistance()
        re = MultipleUTube.compute_reynolds(double_u_tube_series.m_flow_pipe, r_in, fluid)
        m_dot = double_u_tube_series.m_flow_pipe
        r_p = double_u_tube_series.R_p
        h_f = double_u_tube_series.h_f
        r_fp = double_u_tube_series.R_fp

        self.log(double_u_tube_series)
        self.log(f"Reynolds number: {re}")
        self.log(f"Mass flow per pipe: (kg/s): {m_dot}")
        self.log(f"Pipe resistance (K/(W/m)): {r_p}")
        self.log(f"Convection coefficient (W/m2.K): {h_f}")
        self.log(f"Convective resistance (K/(W/m)): {r_fp}")
        self.log(f"Borehole thermal resistance: {r_b_series:0.4f} m.K/W")

        # test values pinned to current performance because GLHEPro doesn't offer a series connection
        self.assertTrue(self.rel_error_within_tol(re, 11744.0, 0.01))
        self.assertTrue(self.rel_error_within_tol(h_f, 2529.0, 0.01))
        self.assertTrue(self.rel_error_within_tol(r_b_series, 0.1624, 0.01))

        # Parallel
        double_u_tube_parallel = MultipleUTube(m_flow_borehole, fluid, borehole, pipe, grout, soil,
                                               DoubleUTubeConnType.PARALLEL)
        r_b_parallel = double_u_tube_parallel.calc_effective_borehole_resistance()
        re = MultipleUTube.compute_reynolds(double_u_tube_parallel.m_flow_pipe, r_in, fluid)
        m_dot = double_u_tube_parallel.m_flow_pipe
        r_p = double_u_tube_parallel.R_p
        h_f = double_u_tube_parallel.h_f
        r_fp = double_u_tube_parallel.R_fp

        self.log(double_u_tube_parallel)
        self.log(f"Reynolds number: {re}")
        self.log(f"Mass flow per pipe: (kg/s): {m_dot}")
        self.log(f"Pipe resistance (K/(W/m)) : {r_p}")
        self.log(f"Convection coefficient (W/m2.K): {h_f}")
        self.log(f"Convective resistance (K/(W/m)): {r_fp}")
        self.log(f"Borehole thermal resistance: {r_b_parallel:0.4f} m.K/W")

        # test values from GLHEPro v5.1
        self.assertTrue(self.rel_error_within_tol(re, 5820.0, 0.01))
        self.assertTrue(self.rel_error_within_tol(h_f, 1288.0, 0.01))
        self.assertTrue(self.rel_error_within_tol(r_b_parallel, 0.1591, 0.005))

    def test_bh_resistance_single_u_tube(self):
        # Borehole dimensions
        h = 100.0  # Borehole length (m)
        d = 2.0  # Borehole buried depth (m)
        r_b = 150.0 / 1000.0 / 2.0  # Borehole radius

        # Pipe dimensions
        r_out = 0.013335  # Pipe outer radius (m)
        r_in = 0.0108  # Pipe inner radius (m)
        s = 0.0323  # Inner-tube to inner-tube Shank spacing (m)
        epsilon = 1.0e-6  # Pipe roughness (m)

        # Pipe positions
        # Single U-tube [(x_in, y_in), (x_out, y_out)]
        pos = Pipe.place_pipes(s, r_out, 1)

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
        pipe = Pipe(pos, r_in, r_out, s, epsilon, k_p, rho_cp_p)
        # Soil
        ugt = 18.3  # Undisturbed ground temperature (degrees Celsius)
        soil = Soil(k_s, rho_cp_s, ugt)
        # Grout
        grout = Grout(k_g, rho_cp_g)

        # Fluid properties
        fluid = GHEFluid(fluid_str="Water", percent=0.0)
        v_flow_borehole = 0.2  # Volumetric flow rate per borehole (L/s)
        # Total fluid mass flow rate per borehole (kg/s)
        m_flow_borehole = v_flow_borehole / 1000.0 * fluid.rho

        # Define a borehole
        borehole = GHEBorehole(h, d, r_b, x=0.0, y=0.0)

        single_u_tube = SingleUTube(m_flow_borehole, fluid, borehole, pipe, grout, soil)

        # Intermediate variables
        re = SingleUTube.compute_reynolds(single_u_tube.m_flow_borehole, r_in, fluid)
        r_p = single_u_tube.R_p
        h_f = single_u_tube.h_f
        r_fp = single_u_tube.R_fp
        r_b = single_u_tube.calc_effective_borehole_resistance()

        # comparison values from GLHEPro v5.1
        self.assertTrue(self.rel_error_within_tol(re, 11748.0, 0.005))
        self.assertTrue(self.rel_error_within_tol(h_f, 2538.0, 0.005))
        self.assertTrue(self.rel_error_within_tol(r_b, 0.2073, 0.005))

        self.log(single_u_tube)
        self.log(f"Reynolds number: {re}")
        self.log(f"Pipe resistance (K/(W/m)) : {r_p}")
        self.log(f"Convection coefficient (W/m2.K): {h_f}")
        self.log(f"Convective resistance (K/(W/m)): {r_fp}")
        self.log(f"Borehole thermal resistance: {r_b:0.4f} m.K/W")

    def test_bh_resistance_validation(self):
        # Dictionary for storing PLAT variations
        borehole_values = {"Single U-tube": {}, "Double U-tube": {}, "Coaxial": {}}
        # Borehole dimensions
        h = 100.0  # Borehole length (m)
        d = 2.0  # Borehole buried depth (m)
        r_b = 150.0 / 1000.0 / 2.0  # Borehole radius

        # Pipe dimensions
        r_out = 0.013335  # Pipe outer radius (m)
        r_in = 0.0108  # Pipe inner radius (m)
        s = 0.0323  # Inner-tube to inner-tube Shank spacing (m)
        epsilon = 1.0e-6  # Pipe roughness (m)

        # Pipe positions
        # Single U-tube [(x_in, y_in), (x_out, y_out)]
        pos_s = Pipe.place_pipes(s, r_out, 1)
        # Pipe positions
        pos_d = Pipe.place_pipes(s, r_out, 2)

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
        pipe_s = Pipe(pos_s, r_in, r_out, s, epsilon, k_p, rho_cp_p)
        pipe_d = Pipe(pos_d, r_in, r_out, s, epsilon, k_p, rho_cp_p)
        # Soil
        ugt = 18.3  # Undisturbed ground temperature (degrees Celsius)
        soil = Soil(k_s, rho_cp_s, ugt)
        # Grout
        grout = Grout(k_g, rho_cp_g)

        # Fluid properties
        fluid = GHEFluid(fluid_str="Water", percent=0.0)

        # A list of volumetric flow rates to check borehole resistances for (L/s)
        v_flow_rates = [0.3, 0.2, 0.18, 0.15, 0.12, 0.1, 0.08, 0.07, 0.06, 0.05]
        v_flow_rates_coaxial = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.29, 0.28, 0.27, 0.26]

        re_glhepro_single = [17621, 11748, 10573, 8811, 7049, 5874, 4699, 4112, 3524, 2937]
        rb_glhepro_single = [0.2021, 0.2073, 0.2094, 0.214, 0.2223, 0.232, 0.2491, 0.2629, 0.2833, 0.3149]

        re_glhepro_double = [8811, 5874, 5286, 4405, 3524, 2937, 2350, 2056, 1762, 1468]
        rb_glhepro_double = [0.1502, 0.1597, 0.1635, 0.1720, 0.1868, 0.2037, 0.2329, 0.2615, 0.2879, 0.3287]

        re_glhepro_coaxial = [8607, 7747, 6886, 6025, 5164, 4304, 3443, 2582, 2496, 2410, 2324, 2238]
        rb_glhepro_coaxial = [0.1055, 0.1066, 0.1081, 0.1102, 0.1131, 0.1176, 0.1249, 0.1386, 0.1406, 0.1428, 0.1452,
                              0.1781]

        # Single and Double U-tubes
        borehole_values["Single U-tube"] = {"V_dot": [], "Rb": [], "Re": []}
        borehole_values["Double U-tube"] = {"V_dot": [], "Rb": [], "Re": []}

        for idx, V_flow_borehole in enumerate(v_flow_rates):
            # Store volumetric flow rates in (L/s)
            borehole_values["Single U-tube"]["V_dot"].append(V_flow_borehole)
            borehole_values["Double U-tube"]["V_dot"].append(V_flow_borehole)

            # Total fluid mass flow rate per borehole (kg/s)
            m_flow_borehole = V_flow_borehole / 1000.0 * fluid.rho

            # Define a borehole
            borehole = GHEBorehole(h, d, r_b, x=0.0, y=0.0)
            single_u_tube = SingleUTube(m_flow_borehole, fluid, borehole, pipe_s, grout, soil)

            # check Reynolds numbers
            re = SingleUTube.compute_reynolds(single_u_tube.m_flow_borehole, r_in, fluid)
            borehole_values["Single U-tube"]["Re"].append(re)
            self.assertTrue(self.rel_error_within_tol(re, re_glhepro_single[idx], 0.01))

            # check BH resistance
            resist_bh = single_u_tube.calc_effective_borehole_resistance()
            borehole_values["Single U-tube"]["Rb"].append(resist_bh)
            self.assertTrue(self.rel_error_within_tol(resist_bh, rb_glhepro_single[idx], 0.01))

            # Define a borehole
            double_u_tube_parallel = MultipleUTube(m_flow_borehole, fluid, borehole, pipe_d, grout, soil)

            # check Reynolds numbers
            re = MultipleUTube.compute_reynolds(double_u_tube_parallel.m_flow_pipe, r_in, fluid)
            borehole_values["Double U-tube"]["Re"].append(re)
            self.assertTrue(self.rel_error_within_tol(re, re_glhepro_double[idx], 0.01))

            # check BH resistance
            resist_bh = double_u_tube_parallel.calc_effective_borehole_resistance()
            borehole_values["Double U-tube"]["Rb"].append(resist_bh)
            self.assertTrue(self.rel_error_within_tol(resist_bh, rb_glhepro_double[idx], 0.05))

        # Pipe dimensions
        # Inner pipe radii
        r_in_in = 44.2 / 1000.0 / 2.0
        r_in_out = 50.0 / 1000.0 / 2.0
        # Outer pipe radii
        r_out_in = 97.4 / 1000.0 / 2.0
        r_out_out = 110.0 / 1000.0 / 2.0
        # Pipe radii
        # Note: This convention is different from pygfunction
        r_inner = [r_in_in, r_in_out]  # The radii of the inner pipe from in to out
        r_outer = [r_out_in, r_out_out]  # The radii of the outer pipe from in to out
        epsilon = 1.0e-6  # Pipe roughness (m)

        # Pipe positioning
        pos = (0, 0)
        s = 0

        # Thermal properties
        k_p = [0.4, 0.4]  # Inner and outer pipe thermal conductivity (W/m.K)
        k_s = 2.0  # Ground thermal conductivity (W/m.K)
        k_g = 1.0  # Grout thermal conductivity (W/m.K)

        # Volumetric heat capacities
        rho_cp_p = 1542000.0  # Pipe volumetric heat capacity (J/K.m3)
        rho_cp_s = 2343493.0  # Soil volumetric heat capacity (J/K.m3)
        rho_cp_g = 3901000.0  # Grout volumetric heat capacity (J/K.m3)

        # Thermal properties
        # Pipe
        pipe = Pipe(pos, r_inner, r_outer, s, epsilon, k_p, rho_cp_p)
        # Soil
        ugt = 18.3  # Undisturbed ground temperature (degrees Celsius)
        soil = Soil(k_s, rho_cp_s, ugt)
        # Grout
        grout = Grout(k_g, rho_cp_g)

        borehole_values["Coaxial"] = {"V_dot": [], "Rb": [], "Re": []}
        for idx, V_flow_borehole in enumerate(v_flow_rates_coaxial):
            # Total fluid mass flow rate per borehole (kg/s)
            m_flow_borehole = V_flow_borehole / 1000.0 * fluid.rho
            borehole_values["Coaxial"]["V_dot"].append(V_flow_borehole)

            # Define a borehole
            borehole = GHEBorehole(h, d, r_b, x=0.0, y=0.0)
            coaxial = CoaxialPipe(m_flow_borehole, fluid, borehole, pipe, grout, soil)

            # check Reynolds number
            re = CoaxialPipe.compute_reynolds_concentric(coaxial.m_flow_borehole, r_in_out, r_out_in, fluid)
            borehole_values["Coaxial"]["Re"].append(re)
            self.assertTrue(self.rel_error_within_tol(re, re_glhepro_coaxial[idx], 0.01))

            resist_bh = coaxial.calc_effective_borehole_resistance()
            borehole_values["Coaxial"]["Rb"].append(resist_bh)

            if V_flow_borehole > 0.4:
                self.assertTrue(self.rel_error_within_tol(resist_bh, rb_glhepro_coaxial[idx], 0.02))

        # Note: DO NOT DELETE THIS SECTION
        #       Uncomment the code and pip install matplotlib if you want to regenerate the plots

        # Comparison plots
        # borehole_validation_values = {
        #     "Single U-tube": {"Re": re_glhepro_single, "Rb": rb_glhepro_single, "V_dot": v_flow_rates},
        #     "Double U-tube": {"Re": re_glhepro_double, "Rb": rb_glhepro_double, "V_dot": v_flow_rates},
        #     "Coaxial": {"Re": re_glhepro_coaxial, "Rb": rb_glhepro_coaxial, "V_dot": v_flow_rates_coaxial}
        # }

        # fig_1, ax_1 = pyplot.subplots(3, sharex='all', sharey='none')
        # fig_2, ax_2 = pyplot.subplots(3, sharex='all', sharey='none')

        # for i, tube in enumerate(borehole_values):
        #     ax_1[i].scatter(
        #         borehole_values[tube]["Re"],
        #         borehole_values[tube]["Rb"],
        #         label=tube + " (GHEDesigner)",
        #     )
        #     ax_1[i].scatter(
        #         borehole_validation_values[tube]["Re"],
        #         borehole_validation_values[tube]["Rb"],
        #         label=tube + " (GLHEPRO)",
        #         marker="x",
        #     )
        #     ax_1[i].grid()
        #     ax_1[i].set_axisbelow(True)
        #     ax_1[i].legend()

        #     ax_2[i].scatter(
        #         borehole_values[tube]["V_dot"],
        #         borehole_values[tube]["Rb"],
        #         label=tube + " (GHEDesigner)",
        #     )
        #     ax_2[i].scatter(
        #         borehole_validation_values[tube]["V_dot"],
        #         borehole_validation_values[tube]["Rb"],
        #         label=tube + " (GLHEPRO)",
        #         marker="x",
        #     )
        #     ax_2[i].grid()
        #     ax_2[i].set_axisbelow(True)
        #     ax_2[i].legend()

        # ax_1[2].set_xlabel("Reynolds number")
        # ax_1[1].set_ylabel("Effective borehole thermal resistance, R$_b^*$ (m.K/W)")

        # ax_2[2].set_xlabel("Volumetric flow rate per borehole (L/s)")
        # ax_2[1].set_ylabel("Effective borehole thermal resistance, R$_b^*$ (m.K/W)")

        # fig_1.tight_layout()
        # fig_2.tight_layout()

        # output_plot = self.test_outputs_directory / 'Rb_vs_Re.png'
        # fig_1.savefig(str(output_plot))
        # output_plot = self.test_outputs_directory / 'Rb_vs_v_dot.png'
        # fig_2.savefig(str(output_plot))
        # pyplot.close(fig_1)
        # pyplot.close(fig_2)
