from unittest import TestCase, skipIf
from pathlib import Path
import pygfunction as gt
from ghedt.peak_load_analysis_tool import media, borehole_heat_exchangers


try:
    # noinspection PyPackageRequirements
    import matplotlib.pyplot as plt
    import pandas as pd
    glhe_file = Path(__file__).resolve().parent / 'test_data' / 'GLHEPRO.xlsx'
    pd.ExcelFile(glhe_file)
    skip_validation = False
except ImportError:
    plt = None
    pd = None
    skip_validation = True


class TestBHResistance(TestCase):
    def test_bh_resistance_coaxial(self):
        # Borehole dimensions
        H = 100.0  # Borehole length (m)
        D = 2.0  # Borehole buried depth (m)
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
        rhoCp_p = 1542.0 * 1000.0  # Pipe volumetric heat capacity (J/K.m3)
        rhoCp_s = 2343.493 * 1000.0  # Soil volumetric heat capacity (J/K.m3)
        rhoCp_g = 3901.0 * 1000.0  # Grout volumetric heat capacity (J/K.m3)

        # Thermal properties
        # Pipe
        pipe = media.Pipe(pos, r_inner, r_outer, s, epsilon, k_p, rhoCp_p)
        # Soil
        ugt = 18.3  # Undisturbed ground temperature (degrees Celsius)
        soil = media.Soil(k_s, rhoCp_s, ugt)
        # Grout
        grout = media.Grout(k_g, rhoCp_g)

        # Fluid properties
        fluid = gt.media.Fluid(fluid_str="Water", percent=0.0)
        V_flow_borehole = 0.2  # Volumetric flow rate per borehole (L/s)
        # Total fluid mass flow rate per borehole (kg/s)
        m_flow_borehole = V_flow_borehole / 1000.0 * fluid.rho

        # Define a borehole
        borehole = gt.boreholes.Borehole(H, D, r_b, x=0.0, y=0.0)

        Coaxial = borehole_heat_exchangers.CoaxialPipe(
            m_flow_borehole, fluid, borehole, pipe, grout, soil
        )

        print(Coaxial)

        R_b = Coaxial.compute_effective_borehole_resistance()

        val = "Intermediate variables"
        print(val + "\n" + len(val) * "-")
        Re = borehole_heat_exchangers.compute_reynolds(
            Coaxial.m_flow_pipe, Coaxial.pipe.r_out[1], fluid
        )
        print("Reynolds number: {}".format(Re))
        # R_p = Coaxial.R_p
        # print('Pipe resistance (K/(W/m)) : {}'.format(R_p))
        h_f = Coaxial.h_fluid_a_out
        print("Convection coefficient (W/m2.K): {}".format(h_f))
        R_fp = Coaxial.R_fp
        print("Convective resistance (K/(W/m)): {}".format(R_fp))
        print("Borehole thermal resistance: {0:4f} m.K/W".format(R_b))

        fig = Coaxial.visualize_pipes()

        fig.savefig("Coaxial.png")

    def test_bh_resistance_double_u_tube(self):
        # Borehole dimensions
        H = 100.0  # Borehole length (m)
        D = 2.0  # Borehole buried depth (m)
        r_b = 150.0 / 1000.0 / 2.0  # Borehole radius

        # Pipe dimensions
        r_out = 26.67 / 1000.0 / 2.0  # Pipe outer radius (m)
        r_in = 21.6 / 1000.0 / 2.0  # Pipe inner radius (m)
        s = 32.3 / 1000.0  # Inner-tube to inner-tube Shank spacing (m)
        epsilon = 1.0e-6  # Pipe roughness (m)

        # Thermal conductivities
        k_p = 0.4  # Pipe thermal conductivity (W/m.K)
        k_s = 2.0  # Ground thermal conductivity (W/m.K)
        k_g = 1.0  # Grout thermal conductivity (W/m.K)

        # Volumetric heat capacities
        rhoCp_p = 1542.0 * 1000.0  # Pipe volumetric heat capacity (J/K.m3)
        rhoCp_s = 2343.493 * 1000.0  # Soil volumetric heat capacity (J/K.m3)
        rhoCp_g = 3901.0 * 1000.0  # Grout volumetric heat capacity (J/K.m3)

        # Pipe positions
        # Double U-tube [(x_in, y_in), (x_out, y_out), (x_in, y_in), (x_out, y_out)]
        pos = media.Pipe.place_pipes(s, r_out, 2)

        # Thermal properties
        # Pipe
        pipe = media.Pipe(pos, r_in, r_out, s, epsilon, k_p, rhoCp_p)
        # Soil
        ugt = 18.3  # Undisturbed ground temperature (degrees Celsius)
        soil = media.Soil(k_s, rhoCp_s, ugt)
        # Grout
        grout = media.Grout(k_g, rhoCp_g)

        # Fluid properties
        fluid = gt.media.Fluid(fluid_str="Water", percent=0.0)
        V_flow_borehole = 0.2  # Volumetric flow rate per borehole (L/s)
        # Total fluid mass flow rate per borehole (kg/s)
        m_flow_borehole = V_flow_borehole / 1000.0 * fluid.rho

        # Define a borehole
        borehole = gt.boreholes.Borehole(H, D, r_b, x=0.0, y=0.0)

        double_u_tube_series = borehole_heat_exchangers.MultipleUTube(
            m_flow_borehole, fluid, borehole, pipe, grout, soil, config="series"
        )

        double_u_tube_parallel = borehole_heat_exchangers.MultipleUTube(
            m_flow_borehole, fluid, borehole, pipe, grout, soil, config="parallel"
        )

        print(double_u_tube_parallel)

        R_b_series = double_u_tube_series.compute_effective_borehole_resistance()
        R_B_parallel = double_u_tube_parallel.compute_effective_borehole_resistance()

        # Intermediate variables
        Re = borehole_heat_exchangers.compute_reynolds(
            double_u_tube_parallel.m_flow_pipe, r_in, fluid
        )

        print("Reynolds number: {}".format(Re))
        R_p = double_u_tube_parallel.R_p
        print("Pipe resistance (K/(W/m)) : {}".format(R_p))
        h_f = double_u_tube_parallel.h_f
        print("Convection coefficient (W/m2.K): {}".format(h_f))
        R_fp = double_u_tube_parallel.R_fp
        print("Convective resistance (K/(W/m)): {}".format(R_fp))

        print("Borehole thermal resistance (series): {0:.4f} m.K/W".format(R_b_series))
        print("Borehole thermal resistance (parallel): {0:.4f} m.K/W".format(R_B_parallel))

        # Create a borehole top view
        fig = double_u_tube_series.visualize_pipes()

        # Save the figure as a png
        fig.savefig("double_u_tube_series.png")

        # Create a borehole top view
        fig = double_u_tube_parallel.visualize_pipes()

        # Save the figure as a png
        fig.savefig("double_u_tube_parallel.png")

    def test_bh_resistance_single_u_tube(self):
        # Borehole dimensions
        H = 100.0  # Borehole length (m)
        D = 2.0  # Borehole buried depth (m)
        r_b = 150.0 / 1000.0 / 2.0  # Borehole radius

        # Pipe dimensions
        r_out = 26.67 / 1000.0 / 2.0  # Pipe outer radius (m)
        r_in = 21.6 / 1000.0 / 2.0  # Pipe inner radius (m)
        s = 32.3 / 1000.0  # Inner-tube to inner-tube Shank spacing (m)
        epsilon = 1.0e-6  # Pipe roughness (m)

        # Pipe positions
        # Single U-tube [(x_in, y_in), (x_out, y_out)]
        pos = media.Pipe.place_pipes(s, r_out, 1)

        # Thermal conductivities
        k_p = 0.4  # Pipe thermal conductivity (W/m.K)
        k_s = 2.0  # Ground thermal conductivity (W/m.K)
        k_g = 1.0  # Grout thermal conductivity (W/m.K)

        # Volumetric heat capacities
        rhoCp_p = 1542.0 * 1000.0  # Pipe volumetric heat capacity (J/K.m3)
        rhoCp_s = 2343.493 * 1000.0  # Soil volumetric heat capacity (J/K.m3)
        rhoCp_g = 3901.0 * 1000.0  # Grout volumetric heat capacity (J/K.m3)

        # Thermal properties
        # Pipe
        pipe = media.Pipe(pos, r_in, r_out, s, epsilon, k_p, rhoCp_p)
        # Soil
        ugt = 18.3  # Undisturbed ground temperature (degrees Celsius)
        soil = media.Soil(k_s, rhoCp_s, ugt)
        # Grout
        grout = media.Grout(k_g, rhoCp_g)

        # Fluid properties
        fluid = gt.media.Fluid(fluid_str="Water", percent=0.0)
        V_flow_borehole = 0.2  # Volumetric flow rate per borehole (L/s)
        # Total fluid mass flow rate per borehole (kg/s)
        m_flow_borehole = V_flow_borehole / 1000.0 * fluid.rho

        # Define a borehole
        borehole = gt.boreholes.Borehole(H, D, r_b, x=0.0, y=0.0)

        single_u_tube = borehole_heat_exchangers.SingleUTube(
            m_flow_borehole, fluid, borehole, pipe, grout, soil
        )

        print(single_u_tube)

        # Intermediate variables
        Re = borehole_heat_exchangers.compute_reynolds(
            single_u_tube.m_flow_borehole, r_in, fluid
        )
        print("Reynolds number: {}".format(Re))
        R_p = single_u_tube.R_p
        print("Pipe resistance (K/(W/m)) : {}".format(R_p))
        h_f = single_u_tube.h_f
        print("Convection coefficient (W/m2.K): {}".format(h_f))
        R_fp = single_u_tube.R_fp
        print("Convective resistance (K/(W/m)): {}".format(R_fp))

        R_b = single_u_tube.compute_effective_borehole_resistance()

        print("Borehole thermal resistance: {0:.4f} m.K/W".format(R_b))

        # Create a borehole top view
        fig = single_u_tube.visualize_pipes()

        # Save the figure as a png
        fig.savefig("single_u_tube.png")

    @skipIf(skip_validation, "Skipping test_bh_resistance_validation, to run: pip install openpyxl matplotlib pandas")
    def test_bh_resistance_validation(self):
        # Dictionary for storing PLAT variations
        borehole_values = {"Single U-tube": {}, "Double U-tube": {}, "Coaxial": {}}
        # Borehole dimensions
        H = 100.0  # Borehole length (m)
        D = 2.0  # Borehole buried depth (m)
        r_b = 150.0 / 1000.0 / 2.0  # Borehole radius

        # Pipe dimensions
        r_out = 26.67 / 1000.0 / 2.0  # Pipe outer radius (m)
        r_in = 21.6 / 1000.0 / 2.0  # Pipe inner radius (m)
        s = 32.3 / 1000.0  # Inner-tube to inner-tube Shank spacing (m)
        epsilon = 1.0e-6  # Pipe roughness (m)

        # Pipe positions
        # Single U-tube [(x_in, y_in), (x_out, y_out)]
        pos_s = media.Pipe.place_pipes(s, r_out, 1)
        # Pipe positions
        pos_d = media.Pipe.place_pipes(s, r_out, 2)

        # Thermal conductivities
        k_p = 0.4  # Pipe thermal conductivity (W/m.K)
        k_s = 2.0  # Ground thermal conductivity (W/m.K)
        k_g = 1.0  # Grout thermal conductivity (W/m.K)

        # Volumetric heat capacities
        rhoCp_p = 1542.0 * 1000.0  # Pipe volumetric heat capacity (J/K.m3)
        rhoCp_s = 2343.493 * 1000.0  # Soil volumetric heat capacity (J/K.m3)
        rhoCp_g = 3901.0 * 1000.0  # Grout volumetric heat capacity (J/K.m3)

        # Thermal properties
        # Pipe
        pipe_s = media.Pipe(pos_s, r_in, r_out, s, epsilon, k_p, rhoCp_p)
        pipe_d = media.Pipe(pos_d, r_in, r_out, s, epsilon, k_p, rhoCp_p)
        # Soil
        ugt = 18.3  # Undisturbed ground temperature (degrees Celsius)
        soil = media.Soil(k_s, rhoCp_s, ugt)
        # Grout
        grout = media.Grout(k_g, rhoCp_g)

        # Fluid properties
        fluid = gt.media.Fluid(fluid_str="Water", percent=0.0)

        # Define a borehole
        borehole = gt.boreholes.Borehole(H, D, r_b, x=0.0, y=0.0)

        # A list of volumetric flow rates to check borehole resistances for
        # (L/s)
        V_flow_rates = [0.3, 0.2, 0.18, 0.15, 0.12, 0.1, 0.08, 0.07, 0.06, 0.05]

        # Single and Double U-tubes
        borehole_values["Single U-tube"] = {"V_dot": [], "Rb": [], "Re": []}
        borehole_values["Double U-tube"] = {"V_dot": [], "Rb": [], "Re": []}

        for V_flow_borehole in V_flow_rates:
            # Store volumetric flow rates in (L/s)
            borehole_values["Single U-tube"]["V_dot"].append(V_flow_borehole)
            borehole_values["Double U-tube"]["V_dot"].append(V_flow_borehole)

            # Total fluid mass flow rate per borehole (kg/s)
            m_flow_borehole = V_flow_borehole / 1000.0 * fluid.rho

            # Define a borehole
            borehole = gt.boreholes.Borehole(H, D, r_b, x=0.0, y=0.0)

            single_u_tube = borehole_heat_exchangers.SingleUTube(
                m_flow_borehole, fluid, borehole, pipe_s, grout, soil
            )

            Re = borehole_heat_exchangers.compute_reynolds(
                single_u_tube.m_flow_pipe, r_in, fluid
            )
            borehole_values["Single U-tube"]["Re"].append(Re)

            double_u_tube_parallel = borehole_heat_exchangers.MultipleUTube(
                m_flow_borehole, fluid, borehole, pipe_d, grout, soil, config="parallel"
            )

            Re = borehole_heat_exchangers.compute_reynolds(
                double_u_tube_parallel.m_flow_pipe, r_in, fluid
            )
            borehole_values["Double U-tube"]["Re"].append(Re)

            R_b = single_u_tube.compute_effective_borehole_resistance()
            borehole_values["Single U-tube"]["Rb"].append(R_b)
            R_b = double_u_tube_parallel.compute_effective_borehole_resistance()
            borehole_values["Double U-tube"]["Rb"].append(R_b)

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
        rhoCp_p = 1542.0 * 1000.0  # Pipe volumetric heat capacity (J/K.m3)
        rhoCp_s = 2343.493 * 1000.0  # Soil volumetric heat capacity (J/K.m3)
        rhoCp_g = 3901.0 * 1000.0  # Grout volumetric heat capacity (J/K.m3)

        # Thermal properties
        # Pipe
        pipe = media.Pipe(pos, r_inner, r_outer, s, epsilon, k_p, rhoCp_p)
        # Soil
        ugt = 18.3  # Undisturbed ground temperature (degrees Celsius)
        soil = media.Soil(k_s, rhoCp_s, ugt)
        # Grout
        grout = media.Grout(k_g, rhoCp_g)

        V_flow_rates = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.29, 0.28, 0.27, 0.26]

        borehole_values["Coaxial"] = {"V_dot": [], "Rb": [], "Re": []}
        for V_flow_borehole in V_flow_rates:
            # Total fluid mass flow rate per borehole (kg/s)
            m_flow_borehole = V_flow_borehole / 1000.0 * fluid.rho
            borehole_values["Coaxial"]["V_dot"].append(V_flow_borehole)

            # Define a borehole
            borehole = gt.boreholes.Borehole(H, D, r_b, x=0.0, y=0.0)

            Coaxial = borehole_heat_exchangers.CoaxialPipe(
                m_flow_borehole, fluid, borehole, pipe, grout, soil
            )

            Re = borehole_heat_exchangers.compute_reynolds_concentric(
                Coaxial.m_flow_pipe, r_in_out, r_out_in, fluid
            )
            borehole_values["Coaxial"]["Re"].append(Re)

            R_b = Coaxial.compute_effective_borehole_resistance()
            borehole_values["Coaxial"]["Rb"].append(R_b)

        # Open GLHEPRO xlsx file
        test_data_dir = Path(__file__).resolve().parent / 'test_data'
        GLHEPRO_file = str(test_data_dir / 'GLHEPRO.xlsx')
        xlsx = pd.ExcelFile(GLHEPRO_file)
        sheet_names = xlsx.sheet_names
        borehole_validation_values = {}
        for sheet in sheet_names:
            d = pd.read_excel(xlsx, sheet_name=sheet).to_dict("list")
            borehole_validation_values[sheet] = d

        # Comparison plots
        fig_1, ax_1 = plt.subplots(3, sharex=True, sharey=False)
        fig_2, ax_2 = plt.subplots(3, sharex=True, sharey=False)

        for i, tube in enumerate(borehole_values):
            ax_1[i].scatter(
                borehole_values[tube]["Re"],
                borehole_values[tube]["Rb"],
                label=tube + " (GLHEDT)",
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

        fig_1.savefig("Rb_vs_Re.png")
        fig_2.savefig("Rb_vs_Vdot.png")
        plt.close(fig_1)
        plt.close(fig_2)
