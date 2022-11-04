# Jack C. Cook
# Tuesday, October 26, 2021

# Purpose: Show how to compute a live-time g-function that can be simulated and
# sized.

import ghedt as dt
import ghedt.peak_load_analysis_tool as plat
import pygfunction as gt
import pandas as pd


def main():
    # Borehole dimensions
    # -------------------
    H = 96.0  # Borehole length (m)
    D = 2.0  # Borehole buried depth (m)
    r_b = 0.075  # Borehole radius]
    B = 5.0  # Borehole spacing (m)

    # Pipe dimensions
    # ---------------
    r_out = 26.67 / 1000.0 / 2.0  # Pipe outer radius (m)
    r_in = 21.6 / 1000.0 / 2.0  # Pipe inner radius (m)
    s = 32.3 / 1000.0  # Inner-tube to inner-tube Shank spacing (m)
    epsilon = 1.0e-6  # Pipe roughness (m)

    # Pipe positions
    # --------------
    # Single U-tube [(x_in, y_in), (x_out, y_out)]
    pos = plat.media.Pipe.place_pipes(s, r_out, 1)
    # Single U-tube BHE object
    bhe_object = plat.borehole_heat_exchangers.SingleUTube

    # Thermal conductivities
    # ----------------------
    k_p = 0.4  # Pipe thermal conductivity (W/m.K)
    k_s = 2.0  # Ground thermal conductivity (W/m.K)
    k_g = 1.0  # Grout thermal conductivity (W/m.K)

    # Volumetric heat capacities
    # --------------------------
    rhoCp_p = 1542.0 * 1000.0  # Pipe volumetric heat capacity (J/K.m3)
    rhoCp_s = 2343.493 * 1000.0  # Soil volumetric heat capacity (J/K.m3)
    rhoCp_g = 3901.0 * 1000.0  # Grout volumetric heat capacity (J/K.m3)

    # Thermal properties
    # ------------------
    # Pipe
    pipe = plat.media.Pipe(pos, r_in, r_out, s, epsilon, k_p, rhoCp_p)
    # Soil
    ugt = 18.3  # Undisturbed ground temperature (degrees Celsius)
    soil = plat.media.Soil(k_s, rhoCp_s, ugt)
    # Grout
    grout = plat.media.Grout(k_g, rhoCp_g)

    # Eskilson's original ln(t/ts) values
    log_time = dt.utilities.Eskilson_log_times()

    # Inputs related to fluid
    # -----------------------
    # Fluid properties
    fluid = gt.media.Fluid(fluid_str="Water", percent=0.0)

    # Coordinates
    Nx = 12
    Ny = 13
    coordinates = dt.coordinates.rectangle(Nx, Ny, B, B)

    # Fluid properties
    V_flow_borehole = 0.2  # System volumetric flow rate (L/s)
    V_flow_system = V_flow_borehole * float(Nx * Ny)
    # Total fluid mass flow rate per borehole (kg/s)
    m_flow_borehole = V_flow_borehole / 1000.0 * fluid.rho

    # Define a borehole
    borehole = gt.boreholes.Borehole(H, D, r_b, x=0.0, y=0.0)

    # Simulation start month and end month
    # --------------------------------
    # Simulation start month and end month
    start_month = 1
    n_years = 20
    end_month = n_years * 12
    # Maximum and minimum allowable fluid temperatures
    max_EFT_allowable = 35  # degrees Celsius
    min_EFT_allowable = 5  # degrees Celsius
    # Maximum and minimum allowable heights
    max_Height = 150  # in meters
    min_Height = 60  # in meters
    sim_params = plat.media.SimulationParameters(
        start_month,
        end_month,
        max_EFT_allowable,
        min_EFT_allowable,
        max_Height,
        min_Height,
    )

    # Process loads from file
    # -----------------------
    # read in the csv file and convert the loads to a list of length 8760
    hourly_extraction: dict = pd.read_csv(
        "../Atlanta_Office_Building_Loads.csv"
    ).to_dict("list")
    # Take only the first column in the dictionary
    hourly_extraction_ground_loads: list = hourly_extraction[
        list(hourly_extraction.keys())[0]
    ]

    # Calculate a g-function for uniform inlet fluid temperature with
    # 8 unequal segments using the equivalent solver
    nSegments = 8
    segments = "unequal"
    solver = "equivalent"
    boundary = "MIFT"
    end_length_ratio = 0.02
    segment_ratios = gt.utilities.segment_ratios(
        nSegments, end_length_ratio=end_length_ratio
    )
    g_function = dt.gfunction.compute_live_g_function(
        B,
        [H],
        [r_b],
        [D],
        m_flow_borehole,
        bhe_object,
        log_time,
        coordinates,
        fluid,
        pipe,
        grout,
        soil,
        nSegments=nSegments,
        segments=segments,
        solver=solver,
        boundary=boundary,
        segment_ratios=segment_ratios,
    )

    # --------------------------------------------------------------------------

    # Initialize the GHE object
    ghe = dt.ground_heat_exchangers.GHE(
        V_flow_system,
        B,
        bhe_object,
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
    max_HP_EFT, min_HP_EFT = ghe.simulate()

    print("Min EFT: {0:.3f}\nMax EFT: {1:.3f}".format(min_HP_EFT, max_HP_EFT))

    # Compute a range of g-functions for interpolation
    H_values = [24.0, 48.0, 96.0, 192.0, 384.0]
    r_b_values = [r_b] * len(H_values)
    D_values = [2.0] * len(H_values)

    g_function = dt.gfunction.compute_live_g_function(
        B,
        H_values,
        r_b_values,
        D_values,
        m_flow_borehole,
        bhe_object,
        log_time,
        coordinates,
        fluid,
        pipe,
        grout,
        soil,
    )

    # Re-Initialize the GHE object
    ghe = dt.ground_heat_exchangers.GHE(
        V_flow_system,
        B,
        bhe_object,
        fluid,
        borehole,
        pipe,
        grout,
        soil,
        g_function,
        sim_params,
        hourly_extraction_ground_loads,
    )

    ghe.size(method="hybrid")

    print("Height of boreholes: {0:.4f}".format(ghe.bhe.b.H))


if __name__ == "__main__":
    main()
