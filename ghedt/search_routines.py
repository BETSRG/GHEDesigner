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
                 ):

        self.coordinates_domain = coordinates_domain
        self.V_flow_borehole = V_flow_borehole

        a = 1

    def calculate_excess(self, coordinates):

        V_flow_system = self.V_flow_borehole * float(len(coordinates))
        # Total fluid mass flow rate per borehole (kg/s)
        m_flow_borehole = self.V_flow_borehole / 1000. * fluid.rho

        log_time = ghedt.utilities.Eskilson_log_times()

        # Calculate a g-function for uniform inlet fluid temperature with
        # 8 unequal segments using the equivalent solver
        g_function = ghedt.gfunction.compute_live_g_function(
            B, H_values, r_b_values, D_values, m_flow_borehole, bhe_object,
            log_time,
            coordinates, fluid, pipe, grout, soil)

        # Initialize the GHE object
        ghe = ghedt.ground_heat_exchangers.GHE(
            V_flow_system, B, bhe_object, fluid, borehole, pipe, grout, soil,
            g_function, sim_params, hourly_extraction_ground_loads)

        # Simulate after computing just one g-function
        max_HP_EFT, min_HP_EFT = ghe.simulate()

        print('Min EFT: {}\nMax EFT: {}'.format(min_HP_EFT, max_HP_EFT))