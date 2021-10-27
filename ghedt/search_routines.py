# Jack C. Cook
# Wednesday, October 27, 2021

import ghedt
import ghedt.PLAT.pygfunction as gt
import ghedt.PLAT as PLAT


class Bisection1D:
    def __init__(self, coordinates_domain: list, V_flow_borehole: float,
                 borehole: gt.boreholes.Borehole,
                 bhe_object: PLAT.borehole_heat_exchangers,
                 fluid: gt.media.Fluid, pipe: PLAT.media.Pipe,
                 grout: PLAT.media.ThermalProperty, soil: PLAT.media.Soil,
                 sim_params: PLAT.media.SimulationParameters,
                 hourly_extraction_ground_loads: list):

        # Take the lowest part of the coordinates domain to be used for the
        # initial setup
        coordinates = coordinates_domain[0]

        V_flow_system = V_flow_borehole * float(len(coordinates))
        # Total fluid mass flow rate per borehole (kg/s)
        m_flow_borehole = V_flow_borehole / 1000. * fluid.rho

        self.log_time = ghedt.utilities.Eskilson_log_times()
        self.sim_params = sim_params
        self.hourly_extraction_ground_loads = hourly_extraction_ground_loads

        B = ghedt.utilities.borehole_spacing(borehole, coordinates)

        # Calculate a g-function for uniform inlet fluid temperature with
        # 8 unequal segments using the equivalent solver
        g_function = ghedt.gfunction.compute_live_g_function(
            B, [borehole.H], [borehole.r_b], [borehole.D], m_flow_borehole,
            bhe_object, self.log_time, coordinates, fluid, pipe, grout, soil)

        # Initialize the GHE object
        self.ghe = ghedt.ground_heat_exchangers.GHE(
            V_flow_system, B, bhe_object, fluid, borehole, pipe, grout, soil,
            g_function, sim_params, hourly_extraction_ground_loads)

        self.calculated_temperatures = {}

    def initialize_ghe(self, coordinates, H):

        self.ghe.bhe.b.H = H
        borehole = self.ghe.bhe.b.H
        m_flow_borehole = self.ghe.bhe.m_flow_borehole
        bhe_object = self.ghe.bhe
        fluid = self.ghe.bhe.fluid
        pipe = self.ghe.bhe.pipe
        grout = self.ghe.bhe.grout
        soil = self.ghe.bhe.soil
        V_flow_borehole = self.ghe.V_flow_borehole
        V_flow_system = V_flow_borehole * float(len(coordinates))

        B = ghedt.utilities.borehole_spacing(borehole, coordinates)

        # Calculate a g-function for uniform inlet fluid temperature with
        # 8 unequal segments using the equivalent solver
        g_function = ghedt.gfunction.compute_live_g_function(
            B, [borehole.H], [borehole.r_b], [borehole.D], m_flow_borehole,
            bhe_object, self.log_time, coordinates, fluid, pipe, grout, soil)

        # Initialize the GHE object
        self.ghe = ghedt.ground_heat_exchangers.GHE(
            V_flow_system, B, bhe_object, fluid, borehole, pipe, grout, soil,
            g_function, self.sim_params, self.hourly_extraction_ground_loads)

    def calculate_excess(self, coordinates):
        # Simulate after computing just one g-function
        max_HP_EFT, min_HP_EFT = self.ghe.simulate()
        T_excess = self.ghe.cost(max_HP_EFT, min_HP_EFT)

        print('Min EFT: {}\nMax EFT: {}'.format(min_HP_EFT, max_HP_EFT))

        return T_excess
