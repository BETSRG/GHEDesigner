from ghedt.peak_load_analysis_tool import media, borehole_heat_exchangers
import pygfunction as gt
from ghedt import gfunction, ground_heat_exchangers
from json import loads
from .ghe_base_case import GHEBaseTest


class TestComputedGFunctionSimAndSize(GHEBaseTest):
    def test_computed_g_function_sim_and_size(self):
        # Borehole dimensions
        # -------------------
        h = 100.0  # Borehole length (m)
        d = 2.0  # Borehole buried depth (m)
        r_b = 150.0 / 1000.0 / 2.0  # Borehole radius]
        b = 5.0  # Borehole spacing (m)

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
        rho_cp_p = 1542.0 * 1000.0  # Pipe volumetric heat capacity (J/K.m3)
        rho_cp_s = 2343.493 * 1000.0  # Soil volumetric heat capacity (J/K.m3)
        rho_cp_g = 3901.0 * 1000.0  # Grout volumetric heat capacity (J/K.m3)

        # Thermal properties
        # ------------------
        # Pipe
        pipe = media.Pipe(pos, r_in, r_out, s, epsilon, k_p, rho_cp_p)
        # Soil
        ugt = 18.3  # Undisturbed ground temperature (degrees Celsius)
        soil = media.Soil(k_s, rho_cp_s, ugt)
        # Grout
        grout = media.Grout(k_g, rho_cp_g)

        # Read in g-functions from GLHEPro
        glhe_json_data = self.test_data_directory / 'GLHEPRO_gFunctions_12x13.json'
        data = loads(glhe_json_data.read_text())

        # Configure the database data for input to the goethermal GFunction object
        geothermal_g_input = gfunction.GFunction.configure_database_file_for_usage(data)

        # Initialize the GFunction object
        g_function = gfunction.GFunction(**geothermal_g_input)

        # Inputs related to fluid
        # -----------------------
        v_flow_system = 31.2  # System volumetric flow rate (L/s)
        # Fluid properties
        fluid = gt.media.Fluid(fluid_str="Water", percent=0.0)

        # Define a borehole
        borehole = gt.boreholes.Borehole(h, d, r_b, x=0.0, y=0.0)

        # Simulation start month and end month
        # --------------------------------
        # Simulation start month and end month
        start_month = 1
        n_years = 20
        end_month = n_years * 12
        # Maximum and minimum allowable fluid temperatures
        max_eft_allowable = 35  # degrees Celsius
        min_eft_allowable = 5  # degrees Celsius
        # Maximum and minimum allowable heights
        max_height = 200  # in meters
        min_height = 60  # in meters
        sim_params = media.SimulationParameters(
            start_month,
            end_month,
            max_eft_allowable,
            min_eft_allowable,
            max_height,
            min_height,
        )

        # Process loads from file
        # -----------------------
        # read in the csv file and convert the loads to a list of length 8760
        glhe_json_data = self.test_data_directory / 'Atlanta_Office_Building_Loads.csv'
        raw_lines = glhe_json_data.read_text().split('\n')
        hourly_extraction_ground_loads = [float(x) for x in raw_lines[1:] if x.strip() != '']

        # --------------------------------------------------------------------------

        # Initialize GHE object
        ghe = ground_heat_exchangers.GHE(
            v_flow_system,
            b,
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
        self.log(calculation_details)
        self.log("Height of boreholes: {0:.3f}".format(ghe.bhe.b.H))
