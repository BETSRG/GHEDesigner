# Jack C. Cook
# Thursday, September 16, 2021
import copy
import warnings

import gFunctionDatabase.Management.application
import scipy.interpolate
import scipy.optimize
import GHEDT.PLAT as PLAT
import GHEDT.PLAT.pygfunction as gt

import numpy as np


class GHEBase:
    def __init__(
            self, V_flow_system: float, B_spacing: float,
                 bhe_function: PLAT.borehole_heat_exchangers,
                 fluid: gt.media.Fluid, borehole: gt.boreholes.Borehole,
                 pipe: PLAT.media.Pipe, grout: PLAT.media.ThermalProperty,
                 soil: PLAT.media.Soil,
                 GFunction: gFunctionDatabase.Management.application.GFunction,
                 sim_params: PLAT.media.SimulationParameters,
                 hourly_extraction_ground_loads: list):

        self.V_flow_system = V_flow_system
        self.B_spacing = B_spacing
        self.nbh = float(len(GFunction.bore_locations))
        self.V_flow_borehole = self.V_flow_system / self.nbh
        m_flow_borehole = self.V_flow_borehole / 1000. * fluid.rho

        # Borehole Heat Exchanger
        self.bhe = bhe_function(
            m_flow_borehole, fluid, borehole, pipe, grout, soil)
        # Equivalent borehole Heat Exchanger
        self.bhe_eq = PLAT.equivalance.compute_equivalent(self.bhe)

        # Radial numerical short time step
        self.radial_numerical = \
            PLAT.radial_numerical_borehole.RadialNumericalBH(self.bhe_eq)
        self.radial_numerical.calc_sts_g_functions(self.bhe_eq)

        # GFunction object
        self.GFunction = GFunction
        # Additional simulation parameters
        self.sim_params = sim_params
        # Hourly ground extraction loads
        # Building cooling is negative, building heating is positive
        self.hourly_extraction_ground_loads = hourly_extraction_ground_loads

    @staticmethod
    def header(text):
        return 50 * '-' + '\n' + '|' + text.center(48) + \
               '|\n' + 50 * '-' + '\n'

    @staticmethod
    def justify(category, value):
        return category.ljust(40) + '= ' + value + '\n'

    def __repr__(self):
        header = self.header
        # Header
        output = 50 * '-' + '\n'
        output += header('GHEDT GLHE Output - Version 0.1')
        output += 50 * '-' + '\n'

        def justify(category, value):
            return category.ljust(40) + '= ' + value + '\n'

        # Detailed information
        output += justify('Number of borheoles',
                          str(len(self.GFunction.bore_locations)))
        output += justify('Depth of the borehole',
                          str(round(self.bhe.b.H, 4)) + ' (m)')
        output += justify('Bore hole spacing',
                          str(round(self.B_spacing, 4)) + ' (m)')
        output += header('Borehole Heat Exchanger')
        output += self.bhe.__repr__()
        output += header('Equivalent Borehole Heat Exchanger')
        output += self.bhe_eq.__repr__()
        output += header('Simulation parameters')
        output += self.sim_params.__repr__()

        return output

    @staticmethod
    def combine_sts_lts(log_time_lts: list, g_lts: list, log_time_sts: list,
                        g_sts: list) -> scipy.interpolate.interp1d:
        # make sure the short time step doesn't overlap with the long time step
        max_log_time_sts = max(log_time_sts)
        min_log_time_lts = min(log_time_lts)

        if max_log_time_sts < min_log_time_lts:
            log_time = log_time_sts + log_time_lts
            g = g_sts + g_lts
        else:
            # find where to stop in sts
            i = 0
            value = log_time_sts[i]
            while value <= min_log_time_lts:
                i += 1
                value = log_time_sts[i]
            log_time = log_time_sts[0:i] + log_time_lts
            g = g_sts[0:i] + g_lts

        g = scipy.interpolate.interp1d(log_time, g)

        return g

    def grab_g_function(self, B_over_H):
        # interpolate for the Long time step g-function
        g_function, rb_value, D_value, H_eq = \
            self.GFunction.g_function_interpolation(B_over_H)
        # correct the long time step for borehole radius
        g_function_corrected = \
            self.GFunction.borehole_radius_correction(g_function,
                                                      rb_value,
                                                      self.bhe.b.r_b)
        # Don't Update the HybridLoad (its dependent on the STS) because
        # it doesn't change the results much and it slows things down a lot
        # combine the short and long time step g-function
        g = self.combine_sts_lts(
            self.GFunction.log_time, g_function_corrected,
            self.radial_numerical.lntts.tolist(),
            self.radial_numerical.g.tolist())

        return g

    def cost(self, max_EFT, min_EFT):
        delta_T_max = max_EFT - self.sim_params.max_EFT_allowable
        delta_T_min = self.sim_params.min_EFT_allowable - min_EFT

        T_excess = max([delta_T_max, delta_T_min])

        return T_excess

    def _simulate_detailed(self, Q_dot: np.ndarray, time_values: np.ndarray,
                           g: scipy.interpolate.interp1d):
        # Perform a detailed simulation based on a numpy array of heat rejection
        # rates, Q_dot (Watts) where each load is applied at the time_value
        # (seconds). The g-function can interpolated.
        # Source: Chapter 2 of Advances in Ground Source Heat Pumps

        n = Q_dot.size

        # Convert the total load applied to the field to the average over
        # borehole wall rejection rate
        # At time t=0, make the heat rejection rate 0.
        Q_dot_b = np.hstack((0., Q_dot / float(self.nbh)))
        time_values = np.hstack((0., time_values))

        Q_dot_b_dt = np.hstack((Q_dot_b[1:] - Q_dot_b[:-1]))

        ts = self.radial_numerical.t_s  # (-)
        two_pi_k = 2. * np.pi * self.bhe.soil.k  # (W/m.K)
        H = self.bhe.b.H  # (meters)
        Tg = self.bhe.soil.ugt  # (Celsius)
        Rb = self.bhe.compute_effective_borehole_resistance()  # (m.K/W)
        m_dot = self.bhe.m_flow_borehole  # (kg/s)
        cp = self.bhe.fluid.cp  # (J/kg.s)

        HPEFT = []
        delta_Tb = []
        for i in range(1, n+1):
            # Take the last i elements of the reversed time array
            _time = time_values[i] - time_values[0:i]
            # _time = time_values_reversed[n - i:n]
            g_values = g(np.log((_time * 3600.) / ts))
            # Tb = Tg + (q_dt * g)  (Equation 2.12)
            delta_Tb_i = (Q_dot_b_dt[0:i] / H / two_pi_k).dot(g_values)
            # Tf = Tb + q_i * R_b^* (Equation 2.13)
            Tb = Tg + delta_Tb_i
            # Bulk fluid temperature
            Tf_bulk = Tb + Q_dot_b[i] / H * Rb
            # T_out = T_f - Q / (2 * mdot cp)  (Equation 2.14)
            Tf_out = Tf_bulk - Q_dot_b[i] / (2 * m_dot * cp)
            HPEFT.append(Tf_out)
            delta_Tb.append(delta_Tb_i)

        return HPEFT, delta_Tb


class GHE(GHEBase):
    def __init__(self, V_flow_system: float, B_spacing: float,
                 bhe_object: PLAT.borehole_heat_exchangers,
                 fluid: gt.media.Fluid, borehole: gt.boreholes.Borehole,
                 pipe: PLAT.media.Pipe, grout: PLAT.media.ThermalProperty,
                 soil: PLAT.media.Soil,
                 GFunction: gFunctionDatabase.Management.application.GFunction,
                 sim_params: PLAT.media.SimulationParameters,
                 hourly_extraction_ground_loads: list
                 ):
        GHEBase.__init__(
            self, V_flow_system, B_spacing, bhe_object, fluid, borehole, pipe,
            grout, soil, GFunction, sim_params, hourly_extraction_ground_loads)

        # Split the extraction loads into heating and cooling for input to the
        # HybridLoad object
        hourly_rejection_loads, hourly_extraction_loads = \
            PLAT.ground_loads.HybridLoad.split_heat_and_cool(
                self.hourly_extraction_ground_loads)

        hybrid_load = PLAT.ground_loads.HybridLoad(
            hourly_rejection_loads, hourly_extraction_loads, self.bhe_eq,
            self.radial_numerical, sim_params)

        # hybrid load object
        self.hybrid_load = hybrid_load

        # List of heat pump exiting fluid temperatures
        self.HPEFT = []
        # list of change in borehole wall temperatures
        self.dTb = []

    def simulate(self, method='hybrid'):
        B = self.B_spacing
        B_over_H = B / self.bhe.b.H

        self.bhe.update_thermal_resistance()

        # Solve for equivalent single U-tube
        self.bhe_eq = PLAT.equivalance.compute_equivalent(self.bhe)
        # Update short time step object with equivalent single u-tube
        self.radial_numerical.calc_sts_g_functions(self.bhe_eq)
        # Combine the short and long-term g-functions. The long term g-function
        # is interpolated for specific B/H and rb/H values.
        g = self.grab_g_function(B_over_H)

        if method == 'hybrid':
            Q_dot = self.hybrid_load.load[2:] * 1000.  # convert to Watts
            time_values = self.hybrid_load.hour[2:]  # convert to seconds

            HPEFT, dTb = self._simulate_detailed(Q_dot, time_values, g)
        elif method == 'hourly':
            n_months = \
                self.sim_params.end_month - self.sim_params.start_month + 1
            n_hours = int(n_months / 12. * 8760.)
            Q_dot = copy.deepcopy(self.hourly_extraction_ground_loads)
            # How many times does q need to be repeated?
            n_years = int(np.ceil(n_hours / 8760))
            Q_dot = Q_dot * n_years
            Q_dot = -1. * np.array(Q_dot)  # Convert loads to rejection
            t = np.arange(1, n_hours + 1, 1)

            HPEFT, dTb = self._simulate_detailed(Q_dot, t, g)
        else:
            raise ValueError('Only hybrid or hourly methods available.')

        self.HPEFT = HPEFT
        self.dTb = dTb

        max_HP_EFT = float(max(HPEFT))
        min_HP_EFT = float(min(HPEFT))
        return max_HP_EFT, min_HP_EFT


class HybridGHE(GHEBase):
    def __init__(self, V_flow_system: float, B_spacing: float,
                 bhe_object: PLAT.borehole_heat_exchangers,
                 fluid: gt.media.Fluid, borehole: gt.boreholes.Borehole,
                 pipe: PLAT.media.Pipe, grout: PLAT.media.ThermalProperty,
                 soil: PLAT.media.Soil,
                 GFunction: gFunctionDatabase.Management.application.GFunction,
                 sim_params: PLAT.media.SimulationParameters,
                 hourly_extraction_ground_loads: list
                 ):
        GHEBase.__init__(
            self, V_flow_system, B_spacing, bhe_object, fluid, borehole, pipe,
            grout, soil, GFunction, sim_params, hourly_extraction_ground_loads)

        # Split the extraction loads into heating and cooling for input to the
        # HybridLoad object
        hourly_rejection_loads, hourly_extraction_loads = \
            PLAT.ground_loads.HybridLoad.split_heat_and_cool(
                self.hourly_extraction_ground_loads)

        hybrid_load = PLAT.ground_loads.HybridLoad(
            hourly_rejection_loads, hourly_extraction_loads, self.bhe_eq,
            self.radial_numerical, sim_params)

        # hybrid load object
        self.hybrid_load = hybrid_load

        self.TBHW: list = []
        self.MFT: list = []
        self.HPEFT: list = []
        self.linehour: list = []
        self.loadperm: list = []
        self.lntts: list = [0, 0]
        self.delta_Tb = [0, 0]

    def __repr__(self):
        output = GHEBase.__repr__(self)
        self.header('Simulation Results')

        max_HP_EFT, min_HP_EFT = self.simulate()
        output += self.justify('Max HP entering temp',
                               str(round(max_HP_EFT, 4)) + ' (degrees Celsius)')
        output += self.justify('Min HP entering temp',
                               str(round(min_HP_EFT, 4)) + ' (degrees Celsius)')
        T_excess = self.cost(max_HP_EFT, min_HP_EFT)
        output += self.justify('Excess fluid temperature',
                               str(round(T_excess, 4)) + ' (degrees Celsius)')

        output += self.header('Peak Load Analysis')
        output += self.hybrid_load.__repr__() + '\n'

        output += self.header('GFunction Information')
        output += 'Coordinates\nx(m)\ty(m)\n'
        for i in range(len(self.GFunction.bore_locations)):
            x, y = self.GFunction.bore_locations[i]
            output += str(x) + '\t' + str(y) + '\n'

        output += 'G-Function\nln(t/ts)\tg\n'
        B_over_H = self.B_spacing / self.bhe.b.H
        g = self.grab_g_function(B_over_H)

        total_g_values = g.x.size
        number_lts_g_values = 27
        number_sts_g_values = 50
        sts_step_size = int(np.floor((total_g_values - number_lts_g_values) /
                                 number_sts_g_values).tolist())
        lntts = []
        g_values = []
        for i in range(0, (total_g_values-number_lts_g_values), sts_step_size):
            lntts.append(g.x[i].tolist())
            g_values.append(g.y[i].tolist())
        lntts += g.x[total_g_values-number_lts_g_values: total_g_values].tolist()
        g_values += g.y[total_g_values-number_lts_g_values: total_g_values].tolist()

        for i in range(len(lntts)):
            output += str(round(lntts[i], 4)) + '\t' + \
                      str(round(g_values[i], 4)) + '\n'

        return output

    def size(self) -> None:
        # Size the ground heat exchanger

        def local_objective(H):
            self.bhe.b.H = H
            max_HP_EFT, min_HP_EFT = self.simulate()
            T_excess = self.cost(max_HP_EFT, min_HP_EFT)
            return T_excess
        # Make the initial guess variable the average of the heights given
        self.bhe.b.H = \
            (self.sim_params.max_Height + self.sim_params.min_Height) / 2.
        # bhe.b.H is updated during sizing
        PLAT.equivalance.solve_root(
            self.bhe.b.H, local_objective, lower=self.sim_params.min_Height,
            upper=self.sim_params.max_Height, xtol=1.0e-6, rtol=1.0e-6,
            maxiter=50)
        if self.bhe.b.H == self.sim_params.min_Height:
            warnings.warn('The minimum height provided to size this ground heat'
                          ' exchanger is not shallow enough. Provide a '
                          'shallower allowable depth or decrease the size of '
                          'the heat exchanger.')
        if self.bhe.b.H == self.sim_params.max_Height:
            warnings.warn('The maximum height provided to size this ground '
                          'heat exchanger is not deep enough. Provide a deeper '
                          'allowable depth or increase the size of the heat '
                          'exchanger.')

        return

    def simulate(self):
        B = self.B_spacing
        B_over_H = B / self.bhe.b.H
        # update borehole thermal resistance values
        self.bhe.update_thermal_resistance()
        g = self.grab_g_function(B_over_H)
        # Apply the g-function to the loads to calculate the
        # alpha = myGHE.k / (1000 * myGHE.rhocp)
        alpha = self.bhe.soil.k / self.bhe.soil.rhoCp
        # tscale = (myGHE.depth ** 2) / (9 * alpha)  # time scale in seconds
        tscale = (self.bhe.b.H ** 2) / (9 * alpha)  # time scale in seconds
        tsh = tscale / 3600  # time scale in hours

        nbh = self.nbh

        Q_dot = self.hybrid_load.load[2:] * 1000.  # convert to Watts
        time_values = self.hybrid_load.hour[2:]  # convert to seconds
        _HPEFT, dTb = self._simulate_detailed(Q_dot, time_values, g)
        time_values = self.hybrid_load.hour[0:]

        Q_dot_b = np.hstack((0., Q_dot / float(self.nbh)))
        Q_dot_b_dt = np.hstack((0., Q_dot_b[1:] - Q_dot_b[:-1]))

        dTb = [0, 0] + dTb
        n = self.hybrid_load.hour.size
        # calculate lntts for each time and then find g-function value
        lntts = []
        g_all = []
        TBHW = [0, 0]  # Borehole wall temperature
        DTBHW = [0, 0]
        MFT = [0, 0]  # Simple mean fluid temperature
        HPEFT = [0, 0]
        HPEFT.extend(_HPEFT)
        # Pi = 3.141592
        Pi = np.pi
        # This outer loop organizes the calculation of DeltaTBHW for the end of
        # the ith period.
        for i in range(2, n):
            DeltaTBHW = 0
            # This inner loop sums the responses of all previous step functions
            for j in range(1, i):
                _time = time_values[i] - time_values[0:i]
                _sftime = _time[j-1]
                # sftime = loads.hour[i] - loads.hour[j]
                sftime = self.hybrid_load.hour[i] - self.hybrid_load.hour[j]  # Spitler's
                # sftime = self.hybrid_load.hour[i] - self.hybrid_load.hour[j-1]  # JCC 2021.10.15
                if sftime > 0:
                    lnttssf = np.log(sftime / tsh)
                    lntts.append(lnttssf)
                    # gsf = myGHE.gi(lnttssf)
                    if lnttssf < g.x[0]:
                        gsf = g.y[0]
                    else:
                        gsf = g(lnttssf)
                    g_all.append(gsf)
                    # convert loads from total on GHE in kW to W/m
                    stepfunctionload = 1000. * self.hybrid_load.sfload[j + 1] / (self.bhe.b.H * nbh)  # Spitler's
                    a = Q_dot_b_dt[j] / self.bhe.b.H
                    assert (a - stepfunctionload) < 1.0e-8
                    # _stepfunctionload = 1000. * self.hybrid_load.sfload[j] / (self.bhe.b.H * nbh)  # JCC 2021.10.15
                    # stepfunctionload = 1000. * (self.hybrid_load.load[j] - self.hybrid_load.load[j-1]) / (self.bhe.b.H * nbh)
                    # assert (_stepfunctionload - stepfunctionload) < 1.0e-10
                    delta_Tb = gsf * stepfunctionload / (2 * Pi * self.bhe.soil.k)
                    _dTb = gsf * a / (2 * Pi * self.bhe.soil.k)
                    DeltaTBHW += delta_Tb
            self.lntts.append(np.log(self.hybrid_load.hour[i] / tsh))
            DTBHW.append(DeltaTBHW)

        measured = np.array(DTBHW)[2:]
        truth = np.array(dTb)[2:]
        diff = measured - truth
        rmse = np.linalg.norm(measured - truth) / len(truth)
        print('RMSE of HPEFT: {}'.format(rmse))

        for i in range(2, n):
            ugt = self.bhe.soil.ugt
            # Peak heating flow rate is the system flow rate in L/s
            peakhtgflowrate = self.bhe.m_flow_borehole / self.bhe.fluid.rho * 1000. * nbh
            # Peak cooling flow rate is the system flow rate in L/s
            peakclgflowrate = self.bhe.m_flow_borehole / self.bhe.fluid.rho * 1000. * nbh
            # Peak heating volumetric capacity is in kJ/K.m3
            peakhtgrhocp = self.bhe.fluid.rhoCp / 1000.
            # Peak cooling volumetric capacity is in kJ/k.m3
            peakclgrhocp = self.bhe.fluid.rhoCp / 1000.

            Q = 1000. * self.hybrid_load.load[i] / float(nbh)
            q = Q / self.bhe.b.H
            _b1 = DTBHW[i]
            _b2 = dTb[i]
            # Tb = DTBHW[i] + (ugt + 273.15)  # borehole wall temperature (Kelvin)
            Tb = dTb[i] + (ugt + 273.15)  # borehole wall temperature (Kelvin)
            # # Equation 2.13 in Chapter 2 of Advances in GSHP
            Rb_star = self.bhe.compute_effective_borehole_resistance()
            Tf = Tb + q * Rb_star
            # # Equation 2.14
            # T_entering = q * self.bhe.b.H / (2 * self.bhe.m_flow_borehole * self.bhe.fluid.cp) + Tf
            T_entering = -Q / (2 * self.bhe.m_flow_borehole * self.bhe.fluid.cp) + Tf
            T_entering_C = T_entering - 273.15
            # HPEFT.append(T_entering_C)

            TBHW.append(DTBHW[i] + ugt)
            _DT_wall_to_fluid = (1000. * self.hybrid_load.load[i] / (self.bhe.b.H * nbh)) * self.bhe.compute_effective_borehole_resistance()
            DT_wall_to_fluid = q * Rb_star
            assert (DT_wall_to_fluid - _DT_wall_to_fluid) <=1.0e-10
            MFT.append(TBHW[i] + DT_wall_to_fluid)
            # if loads.hour[i] > 0:
            if self.hybrid_load.hour[i] > 0:
                mdotrhocp = peakclgflowrate * peakclgrhocp  # Units of flow rate are L/s or 0.001 m3/s
                # Units of rhocp are kJ/m3 K
                # Units of mdotrhocp are then W/K
            else:
                mdotrhocp = peakhtgflowrate * peakhtgrhocp
            _half_fluidDT = (1000. * self.hybrid_load.load[i]) / (mdotrhocp * 2.)
            half_fluidDT = Q / (2 * self.bhe.m_flow_borehole * self.bhe.fluid.cp)
            assert (half_fluidDT - _half_fluidDT) < 1.0e-10
            _T_entering_C = MFT[i] - half_fluidDT
            # _HPEFT = (Tf - 273.15) - half_fluidDT
            # HPEFT.append(MFT[i] - half_fluidDT)
            # assert (_HPEFT - HPEFT[i]) < 1.0e-10
            # load_per_m = 1000 * loads.load[i] / (myGHE.depth * myGHE.nbh)  # to check W/m
            load_per_m = 1000 * self.hybrid_load.load[i] / (self.bhe.b.H * nbh)  # to check W/m
            linehour = self.hybrid_load.hour[i]

        self.TBHW = TBHW
        self.MFT = MFT
        self.HPEFT = HPEFT
        self.linehour = self.hybrid_load.load.tolist()
        # self.lntts = np.log(self.hybrid_load.load / tsh)
        self.delta_Tb = DTBHW
        load_per_m_list = [1000 * self.hybrid_load.load[i] / (self.bhe.b.H * nbh)
                           for i in range(len(self.hybrid_load.load))]
        self.loadperm = load_per_m_list

        max_HP_EFT = float(max(HPEFT[2:n]))
        min_HP_EFT = float(min(HPEFT[2:n]))
        return max_HP_EFT, min_HP_EFT


class HourlyGHE(GHEBase):
    def __init__(self, V_flow_system: float, B_spacing: float,
                 bhe_object: PLAT.borehole_heat_exchangers, fluid: gt.media.Fluid,
                 borehole: gt.boreholes.Borehole, pipe: PLAT.media.Pipe,
                 grout: PLAT.media.ThermalProperty, soil: PLAT.media.Soil,
                 GFunction: gFunctionDatabase.Management.application.GFunction,
                 sim_params: PLAT.media.SimulationParameters,
                 hourly_extraction_ground_loads: list):
        GHEBase.__init__(
            self, V_flow_system, B_spacing, bhe_object, fluid, borehole, pipe,
            grout, soil, GFunction, sim_params, hourly_extraction_ground_loads)

        self.HPEFT = []
        self.delta_Tb = []

    def __repr__(self):
        output = GHEBase.__repr__(self)
        self.header('Simulation Results')

        max_HP_EFT, min_HP_EFT = self.simulate()
        output += self.justify('Max HP entering temp',
                               str(round(max_HP_EFT, 4)) + ' (degrees Celsius)')
        output += self.justify('Min HP entering temp',
                               str(round(min_HP_EFT, 4)) + ' (degrees Celsius)')
        T_excess = self.cost(max_HP_EFT, min_HP_EFT)
        output += self.justify('Excess fluid temperature',
                               str(round(T_excess, 4)) + ' (degrees Celsius)')

        output += self.header('Peak Load Analysis')
        
        output += self.header('GFunction Information')
        output += 'Coordinates\nx(m)\ty(m)\n'
        for i in range(len(self.GFunction.bore_locations)):
            x, y = self.GFunction.bore_locations[i]
            output += str(x) + '\t' + str(y) + '\n'

        output += 'G-Function\nln(t/ts)\tg\n'
        B_over_H = self.B_spacing / self.bhe.b.H
        g = self.grab_g_function(B_over_H)

        total_g_values = g.x.size
        number_lts_g_values = 27
        number_sts_g_values = 50
        sts_step_size = int(np.floor((total_g_values - number_lts_g_values) /
                                 number_sts_g_values).tolist())
        lntts = []
        g_values = []
        for i in range(0, (total_g_values-number_lts_g_values), sts_step_size):
            lntts.append(g.x[i].tolist())
            g_values.append(g.y[i].tolist())
        lntts += g.x[total_g_values-number_lts_g_values: total_g_values].tolist()
        g_values += g.y[total_g_values-number_lts_g_values: total_g_values].tolist()

        for i in range(len(lntts)):
            output += str(round(lntts[i], 4)) + '\t' + \
                      str(round(g_values[i], 4)) + '\n'

        return output

    def simulate_hourly(self, hours, q, g, Rb, two_pi_k, ts, Tg):
        # An hourly simulation for the fluid temperature
        # Chapter 2 of Advances in Ground Source Heat Pumps

        # How many times does q need to be repeated?
        n_years = np.ceil(hours / 8760)

        q = np.array(q)
        # reverse direction of heat flow to be positive for rejection to ground
        q /= -1. * (float(self.nbh) * self.bhe.b.H)  # convert loads to W/m
        q = np.repeat(q, n_years)
        q_dt = np.hstack((q[0], q[1:] - q[:-1]))
        t = np.arange(1, hours, 1)
        t = t[::-1]

        HPEFT = []
        for n in range(1, hours-1):
            _time = t[hours - 1 - n:hours-1]
            g_values = g(np.log((_time * 3600.) / ts))
            # Tb = Tg + sum_{i=1}^{n} (q_{i} - q_{i-1}) /
                #                             (2 pi k) * g((t_n - t_{i-1}) / ts)
            # Please note the loads are considered to be ground extraction
            # loads, but this simulation expects ground rejection loads,
            # therefore we multiply by -1
            summer = (q_dt[0:n] / two_pi_k).dot(g_values)
            # Tf = Tb + q_i * R_b^* (equation 2.13)
            Tb = (Tg + 273.15) + summer
            Tf = Tb + q[n] * Rb
            # T_out = T_f - Q / (2 * mdot cp)  (equation 2.14)
            T_entering \
                = -q[n] * self.bhe.b.H / (2 * self.bhe.m_flow_borehole * self.bhe.fluid.cp) + Tf
            HPEFT.append(T_entering - 273.15)
            self.delta_Tb.append(Tb - 273.15 - self.bhe.soil.ugt)

        return HPEFT

    def simulate(self):
        B = self.B_spacing
        B_over_H = B / self.bhe.b.H

        self.bhe.update_thermal_resistance()

        g = self.grab_g_function(B_over_H)

        ts = self.radial_numerical.t_s
        two_pi_k = 2. * np.pi * self.bhe.soil.k
        Rb = self.bhe.compute_effective_borehole_resistance()

        n_months = self.sim_params.end_month - self.sim_params.start_month + 1
        n_hours = int(n_months / 12. * 8760.)

        Q_dot = copy.deepcopy(self.hourly_extraction_ground_loads)

        # self.HPEFT = self.simulate_hourly(
        #     n_hours, Q_dot, g, Rb, two_pi_k, ts, self.bhe.soil.ugt)

        # How many times does q need to be repeated?
        n_years = int(np.ceil(n_hours / 8760))
        Q_dot = Q_dot * n_years
        Q_dot = -1. * np.array(Q_dot)  # Convert loads to rejection
        # Q_dot = np.repeat(Q_dot, n_years)
        t = np.arange(1, n_hours+1, 1)

        HPEFT, dT_b = self._simulate_detailed(Q_dot, t, g)

        # print(dT_b[0:20])
        print(HPEFT[0:20])
        # assert (HPEFT == self.HPEFT)
        self.HPEFT = HPEFT

        max_HP_EFT = float(max(self.HPEFT))
        min_HP_EFT = float(min(self.HPEFT))
        return max_HP_EFT, min_HP_EFT

    def size(self) -> None:
        # Size the ground heat exchanger

        def local_objective(H):
            self.bhe.b.H = H
            max_HP_EFT, min_HP_EFT = self.simulate()
            T_excess = self.cost(max_HP_EFT, min_HP_EFT)
            return T_excess

        # Make the initial guess variable the average of the heights given
        self.bhe.b.H = \
            (self.sim_params.max_Height + self.sim_params.min_Height) / 2.
        # bhe.b.H is updated during sizing
        PLAT.equivalance.solve_root(
            self.bhe.b.H, local_objective, lower=self.sim_params.min_Height,
            upper=self.sim_params.max_Height, xtol=1.0e-6, rtol=1.0e-6,
            maxiter=50)
        if self.bhe.b.H == self.sim_params.min_Height:
            warnings.warn('The minimum height provided to size this ground heat'
                          ' exchanger is not shallow enough. Provide a '
                          'shallower allowable depth or decrease the size of '
                          'the heat exchanger.')
        if self.bhe.b.H == self.sim_params.max_Height:
            warnings.warn('The maximum height provided to size this ground '
                          'heat exchanger is not deep enough. Provide a deeper '
                          'allowable depth or increase the size of the heat '
                          'exchanger.')

        return
