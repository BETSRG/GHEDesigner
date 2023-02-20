from matplotlib import pyplot
from pandas import ExcelFile, read_excel

from ghedesigner.borehole import GHEBorehole
from ghedesigner.borehole_heat_exchangers import GHEDesignerBoreholeBase, CoaxialPipe, MultipleUTube, SingleUTube
from ghedesigner.enums import FlowConfig
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
        v_flow_borehole = 0.2  # Volumetric flow rate per borehole (L/s)
        # Total fluid mass flow rate per borehole (kg/s)
        m_flow_borehole = v_flow_borehole / 1000.0 * fluid.rho

        # Define a borehole
        borehole = GHEBorehole(h, d, r_b, x=0.0, y=0.0)

        coaxial = CoaxialPipe(m_flow_borehole, fluid, borehole, pipe, grout, soil)

        self.log(coaxial)

        r_b = coaxial.calc_effective_borehole_resistance()

        val = "Intermediate variables"
        self.log(val + "\n" + len(val) * "-")
        re = GHEDesignerBoreholeBase.compute_reynolds(
            coaxial.m_flow_pipe, coaxial.pipe.r_out[1], fluid
        )
        self.log(f"Reynolds number: {re}")
        h_f = coaxial.h_f_a_out
        self.log(f"Convection coefficient (W/m2.K): {h_f}")
        r_fp = coaxial.R_fp
        self.log(f"Convective resistance (K/(W/m)): {r_fp}")
        self.log(f"Borehole thermal resistance: {r_b:0.4f} m.K/W")

        fig = coaxial.visualize_pipes()

        output_plot = self.test_outputs_directory / 'Coaxial.png'
        fig.savefig(str(output_plot))

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
        double_u_tube_series = MultipleUTube(m_flow_borehole, fluid, borehole, pipe, grout, soil)
        double_u_tube_parallel = MultipleUTube(m_flow_borehole, fluid, borehole, pipe, grout, soil, FlowConfig.Parallel)

        self.log(double_u_tube_parallel)

        r_b_series = double_u_tube_series.calc_effective_borehole_resistance()
        r_b_parallel = double_u_tube_parallel.calc_effective_borehole_resistance()

        # Intermediate variables
        re = GHEDesignerBoreholeBase.compute_reynolds(double_u_tube_parallel.m_flow_pipe, r_in, fluid)

        self.log(f"Reynolds number: {re}")
        r_p = double_u_tube_parallel.R_p
        self.log(f"Pipe resistance (K/(W/m)) : {r_p}")
        h_f = double_u_tube_parallel.h_f
        self.log(f"Convection coefficient (W/m2.K): {h_f}")
        r_fp = double_u_tube_parallel.R_fp
        self.log(f"Convective resistance (K/(W/m)): {r_fp}")
        self.log(f"Borehole thermal resistance (series): {r_b_series:0.4f} m.K/W")
        self.log(f"Borehole thermal resistance (parallel): {r_b_parallel:0.4f} m.K/W")

        # Create a borehole top view
        fig = double_u_tube_series.visualize_pipes()

        # Save the figure as a png
        output_plot = self.test_outputs_directory / 'double_u_tube_series.png'
        fig.savefig(str(output_plot))

        # Create a borehole top view
        fig = double_u_tube_parallel.visualize_pipes()

        # Save the figure as a png
        output_plot = self.test_outputs_directory / 'double_u_tube_parallel.png'
        fig.savefig(str(output_plot))

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

        self.log(single_u_tube)

        # Intermediate variables
        re = GHEDesignerBoreholeBase.compute_reynolds(
            single_u_tube.m_flow_borehole, r_in, fluid
        )
        self.log(f"Reynolds number: {re}")
        r_p = single_u_tube.R_p
        self.log(f"Pipe resistance (K/(W/m)) : {r_p}")
        h_f = single_u_tube.h_f
        self.log(f"Convection coefficient (W/m2.K): {h_f}")
        r_fp = single_u_tube.R_fp
        self.log(f"Convective resistance (K/(W/m)): {r_fp}")

        r_b = single_u_tube.calc_effective_borehole_resistance()

        self.log(f"Borehole thermal resistance: {r_b:0.4f} m.K/W")

        # Create a borehole top view
        fig = single_u_tube.visualize_pipes()

        # Save the figure as a png
        output_plot = self.test_outputs_directory / 'single_u_tube.png'
        fig.savefig(str(output_plot))

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

        # Single and Double U-tubes
        borehole_values["Single U-tube"] = {"V_dot": [], "Rb": [], "Re": []}
        borehole_values["Double U-tube"] = {"V_dot": [], "Rb": [], "Re": []}

        for V_flow_borehole in v_flow_rates:
            # Store volumetric flow rates in (L/s)
            borehole_values["Single U-tube"]["V_dot"].append(V_flow_borehole)
            borehole_values["Double U-tube"]["V_dot"].append(V_flow_borehole)

            # Total fluid mass flow rate per borehole (kg/s)
            m_flow_borehole = V_flow_borehole / 1000.0 * fluid.rho

            # Define a borehole
            borehole = GHEBorehole(h, d, r_b, x=0.0, y=0.0)

            single_u_tube = SingleUTube(m_flow_borehole, fluid, borehole, pipe_s, grout, soil)

            re = GHEDesignerBoreholeBase.compute_reynolds(
                single_u_tube.m_flow_pipe, r_in, fluid
            )
            borehole_values["Single U-tube"]["Re"].append(re)

            double_u_tube_parallel = MultipleUTube(m_flow_borehole, fluid, borehole, pipe_d, grout, soil,
                                                   FlowConfig.Parallel)

            re = GHEDesignerBoreholeBase.compute_reynolds(
                double_u_tube_parallel.m_flow_pipe, r_in, fluid
            )
            borehole_values["Double U-tube"]["Re"].append(re)

            r_b = single_u_tube.calc_effective_borehole_resistance()
            borehole_values["Single U-tube"]["Rb"].append(r_b)
            r_b = double_u_tube_parallel.calc_effective_borehole_resistance()
            borehole_values["Double U-tube"]["Rb"].append(r_b)

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

        v_flow_rates = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.29, 0.28, 0.27, 0.26]

        borehole_values["Coaxial"] = {"V_dot": [], "Rb": [], "Re": []}
        for V_flow_borehole in v_flow_rates:
            # Total fluid mass flow rate per borehole (kg/s)
            m_flow_borehole = V_flow_borehole / 1000.0 * fluid.rho
            borehole_values["Coaxial"]["V_dot"].append(V_flow_borehole)

            # Define a borehole
            borehole = GHEBorehole(h, d, r_b, x=0.0, y=0.0)

            coaxial = CoaxialPipe(m_flow_borehole, fluid, borehole, pipe, grout, soil)

            re = GHEDesignerBoreholeBase.compute_reynolds_concentric(
                coaxial.m_flow_pipe, r_in_out, r_out_in, fluid
            )
            borehole_values["Coaxial"]["Re"].append(re)

            r_b = coaxial.calc_effective_borehole_resistance()
            borehole_values["Coaxial"]["Rb"].append(r_b)

        # Open GLHEPRO xlsx file
        glhepro_file = str(self.test_data_directory / 'GLHEPRO.xlsx')
        xlsx = ExcelFile(glhepro_file)
        sheet_names = xlsx.sheet_names
        borehole_validation_values = {}
        for sheet in sheet_names:
            d = read_excel(xlsx, sheet_name=sheet).to_dict("list")
            borehole_validation_values[sheet] = d

        # Comparison plots
        fig_1, ax_1 = pyplot.subplots(3, sharex='all', sharey='none')
        fig_2, ax_2 = pyplot.subplots(3, sharex='all', sharey='none')

        for i, tube in enumerate(borehole_values):
            ax_1[i].scatter(
                borehole_values[tube]["Re"],
                borehole_values[tube]["Rb"],
                label=tube + " (GHEDesigner)",
            )
            ax_1[i].scatter(
                borehole_validation_values[tube]["Re"],
                borehole_validation_values[tube]["Rb"],
                label=tube + " (GLHEPRO)",
                marker="x",
            )
            ax_1[i].grid()
            ax_1[i].set_axisbelow(True)
            ax_1[i].legend()

            ax_2[i].scatter(
                borehole_values[tube]["V_dot"],
                borehole_values[tube]["Rb"],
                label=tube + " (PLAT)",
            )
            ax_2[i].scatter(
                borehole_validation_values[tube]["V_dot"],
                borehole_validation_values[tube]["Rb"],
                label=tube + " (GLHEPRO)",
                marker="x",
            )
            ax_2[i].grid()
            ax_2[i].set_axisbelow(True)
            ax_2[i].legend()

        ax_1[2].set_xlabel("Reynolds number")
        ax_1[1].set_ylabel("Effective borehole thermal resistance, R$_b^*$ (m.K/W)")

        ax_2[2].set_xlabel("Volumetric flow rate per borehole (L/s)")
        ax_2[1].set_ylabel("Effective borehole thermal resistance, R$_b^*$ (m.K/W)")

        fig_1.tight_layout()
        fig_2.tight_layout()

        output_plot = self.test_outputs_directory / 'Rb_vs_Re.png'
        fig_1.savefig(str(output_plot))
        output_plot = self.test_outputs_directory / 'Rb_vs_v_dot.png'
        fig_2.savefig(str(output_plot))
        pyplot.close(fig_1)
        pyplot.close(fig_2)
