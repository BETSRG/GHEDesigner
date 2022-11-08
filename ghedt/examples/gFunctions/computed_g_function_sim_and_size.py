from ghedt.peak_load_analysis_tool import media, borehole_heat_exchangers
from pathlib import Path
import pygfunction as gt
from ghedt import utilities, gfunction, ground_heat_exchangers


def main():
    # Borehole dimensions
    # -------------------
    H = 100.0  # Borehole length (m)
    D = 2.0  # Borehole buried depth (m)
    r_b = 150.0 / 1000.0 / 2.0  # Borehole radius]
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
    pos = media.Pipe.place_pipes(s, r_out, 1)
    # Single U-tube BHE object
    bhe_object = borehole_heat_exchangers.SingleUTube

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
    pipe = media.Pipe(pos, r_in, r_out, s, epsilon, k_p, rhoCp_p)
    # Soil
    ugt = 18.3  # Undisturbed ground temperature (degrees Celsius)
    soil = media.Soil(k_s, rhoCp_s, ugt)
    # Grout
    grout = media.Grout(k_g, rhoCp_g)

    # Read in g-functions from GLHEPro
    file = "GLHEPRO_gFunctions_12x13.json"
    data = utilities.js_load(file)

    # Configure the database data for input to the goethermal GFunction object
    geothermal_g_input = gfunction.GFunction.configure_database_file_for_usage(data)

    # Initialize the GFunction object
    g_function = gfunction.GFunction(**geothermal_g_input)

    # Inputs related to fluid
    # -----------------------
    V_flow_system = 31.2  # System volumetric flow rate (L/s)
    # Fluid properties
    fluid = gt.media.Fluid(fluid_str="Water", percent=0.0)

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
    max_Height = 200  # in meters
    min_Height = 60  # in meters
    sim_params = media.SimulationParameters(
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
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    csv_file = project_root / 'examples' / 'data' / 'Atlanta_Office_Building_Loads.csv'
    raw_lines = csv_file.read_text().split('\n')
    hourly_extraction_ground_loads = [float(x) for x in raw_lines[1:] if x.strip() != '']

    # --------------------------------------------------------------------------

    # Initialize GHE object
    ghe = ground_heat_exchangers.GHE(
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

    ghe.size()

    calculation_details = "GLHEPRO_gFunctions_12x13.json".split(".")[0]
    print(calculation_details)
    print("Height of boreholes: {0:.3f}".format(ghe.bhe.b.H))


if __name__ == "__main__":
    main()
