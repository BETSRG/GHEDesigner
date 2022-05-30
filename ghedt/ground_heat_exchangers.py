# Jack C. Cook
# Thursday, September 16, 2021
import copy
import warnings

import scipy.interpolate
import scipy.optimize
import pygfunction as gt
import ghedt.peak_load_analysis_tool as plat
import ghedt as dt

import numpy as np


class BaseGHE:
    def __init__(
            self, V_flow_system: list, B_spacing: float,
            bhe_function: plat.borehole_heat_exchangers,
            fluid: gt.media.Fluid, boreholes: list,bHTemplate:list,
            pipe: plat.media.Pipe, grout: plat.media.Grout,
            soil: plat.media.Soil, GFunction: dt.gfunction.GFunction,
            sim_params: plat.media.SimulationParameters,
            hourly_extraction_ground_loads: list,fieldType = "N/A",fieldSpecifier = "N/A"):
        self.templateIndices = bHTemplate
        self.boreholes = boreholes
        self.fieldType = fieldType
        self.fieldSpecifier = fieldSpecifier
        self.V_flow_system = np.sum(self.templateIndexer(V_flow_system,bHTemplate))
        self.B_spacing = B_spacing
        self.nbh = float(len(GFunction.bore_locations))
        #print(V_flow_system)
        #print(self.nbh)
        self.V_flow_borehole = self.V_flow_system/self.nbh
        m_flow_borehole = (np.array(self.V_flow_borehole) / 1000. * fluid.rho)
        m_flow_borehole = m_flow_borehole.tolist()
        self.m_flow_borehole = m_flow_borehole

        # Borehole Heat Exchanger
        self.bhe_object = bhe_function
        self.bhes = [bhe_function(
            m_flow_borehole, fluid, borehole, pipe, grout, soil) for borehole in boreholes]
        self.bhe = self.bhes[0]
        # Equivalent borehole Heat Exchanger
        self.bhe_eqs = [plat.equivalance.compute_equivalent(bhe) for bhe in self.bhes]
        self.bhe_eq = self.bhe_eqs[0]

        # Radial numerical short time step
        self.radial_numerical = \
            plat.radial_numerical_borehole.RadialNumericalBH(self.bhe_eq)
        self.radial_numerical.calc_sts_g_functions(self.bhe_eq)

        # GFunction object
        self.GFunction = GFunction
        # Additional simulation parameters
        self.sim_params = sim_params
        # Hourly ground extraction loads
        # Building cooling is negative, building heating is positive
        self.hourly_extraction_ground_loads = hourly_extraction_ground_loads
        self.times = None
        self.loading = None
        self.H = self.averageHeight()

    @staticmethod
    def header(text):
        return 50 * '-' + '\n' + '|' + text.center(48) + \
               '|\n' + 50 * '-' + '\n'

    @staticmethod
    def justify(category, value):
        return category.ljust(40) + '= ' + value + '\n'
    def templateIndexer(self,arrayToIndex,Indices):
        return [arrayToIndex[index] for index in Indices]
    def __repr__(self):
        header = self.header
        # Header
        output = 50 * '-' + '\n'
        output += header('GHEDT GHE Output - Version 0.1')
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

    def averageHeight(self):
        #print("Starting Average Height Calc")
        return np.sum(self.templateIndexer([boreHole.H for boreHole in self.boreholes],self.templateIndices))/(self.nbh)
        #print("Finished Average Height Calc")

    def averageResistance(self,m_flow_borehole = None, fluid = None):
        return self.avgR
    def calcAverageResistance(self,m_flow_borehole = None, fluid = None):
        #print("Calculating Average Resistance...")
        for bhe in self.bhes:
            bhe.update_thermal_resistance()
        self.avgR = np.sum(self.templateIndexer([bhe.compute_effective_borehole_resistance(m_flow_borehole=m_flow_borehole
                                                ,fluid=fluid) for bhe in self.bhes],self.templateIndices))/(self.nbh)
        #print("Done Calculating Average Resistance...")
        return self.avgR

    def setHeights(self,newAverageHeight):
        beta = []
        bhes = self.bhes
        firstBhe = bhes[0]
        for bhe in bhes:
            beta.append(bhe.b.H/firstBhe.b.H)
        nbh = self.nbh
        weightBetaSum = np.sum(self.templateIndexer(beta,self.templateIndices))
        h1 = (nbh*newAverageHeight)/(weightBetaSum)
        for i in range(len(bhes)):
            #print("Height Set:",h1*beta[i])
            if i == 0:
                bhe.b.H = h1*beta[0]
            bhes[i].b.H = h1*beta[i]
        return


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
        #H = self.bhe.b.H  # (meters)
        Tg = self.bhe.soil.ugt  # (Celsius)
        Rb = self.averageResistance()  # (m.K/W)
        m_dot = self.bhe.m_flow_borehole  # (kg/s)
        cp = self.bhe.fluid.cp  # (J/kg.s)
        H = self.H

        HPEFT = []
        delta_Tb = []
        #print("Starting Detailed Simulation...")
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
        #print("Finished Detailed Simulation")

        return HPEFT, delta_Tb

    def compute_g_functions(self):
        # Compute g-functions for a bracketed solution, based on min and max
        # height
        min_height = self.sim_params.min_Height
        max_height = self.sim_params.max_Height
        avg_height = (min_height + max_height) / 2.
        H_values = [min_height, avg_height, max_height]
        r_b_values = [self.bhe.b.r_b] * len(H_values)
        D_values = [self.bhe.b.D] * len(H_values)

        coordinates = self.GFunction.bore_locations
        log_time = self.GFunction.log_time

        g_function = dt.gfunction.compute_live_g_function(
            self.B_spacing, H_values, r_b_values, D_values,
            self.bhe.m_flow_borehole, self.bhe_object, log_time,
            coordinates, self.bhe.fluid, self.bhe.pipe,
            self.bhe.grout, self.bhe.soil)

        self.GFunction = g_function

        return


class GHE(BaseGHE):
    def __init__(self, V_flow_system: list, B_spacing: float,
                 bhe_object: plat.borehole_heat_exchangers,
                 fluid: gt.media.Fluid, boreholes: list,bHTemplate:list,
                 pipe: plat.media.Pipe, grout: plat.media.Grout,
                 soil: plat.media.Soil,
                 GFunction: dt.gfunction.GFunction,
                 sim_params: plat.media.SimulationParameters,
                 hourly_extraction_ground_loads: list,fieldType = "N/A",fieldSpecifier = "N/A"
                 ):
        BaseGHE.__init__(
            self, V_flow_system, B_spacing, bhe_object, fluid, boreholes,bHTemplate, pipe,
            grout, soil, GFunction, sim_params, hourly_extraction_ground_loads,fieldType = fieldType,fieldSpecifier = fieldSpecifier)

        # Split the extraction loads into heating and cooling for input to
        # the HybridLoad object
        #print(self.hourly_extraction_ground_loads)
        hourly_rejection_loads, hourly_extraction_loads = \
            plat.ground_loads.HybridLoad.split_heat_and_cool(
                self.hourly_extraction_ground_loads)

        hybrid_load = plat.ground_loads.HybridLoad(
            hourly_rejection_loads, hourly_extraction_loads, self.bhe_eq,
            self.radial_numerical, sim_params)

        # hybrid load object
        self.hybrid_load = hybrid_load

        # List of heat pump exiting fluid temperatures
        self.HPEFT = []
        # list of change in borehole wall temperatures
        self.dTb = []

    def __repr__(self):
        output = BaseGHE.__repr__(self)

        self.header('Simulation Results')
        if len(self.HPEFT) > 0:
            max_HP_EFT = max(self.HPEFT)
            min_HP_EFT = min(self.HPEFT)
            output += self.justify('Max HP entering temp',
                                   str(round(max_HP_EFT, 4)) +
                                   ' (degrees Celsius)')
            output += self.justify('Min HP entering temp',
                                   str(round(min_HP_EFT, 4)) +
                                   ' (degrees Celsius)')
            T_excess = self.cost(max_HP_EFT, min_HP_EFT)
            output += self.justify('Excess fluid temperature',
                                   str(round(T_excess, 4)) +
                                   ' (degrees Celsius)')
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
        for i in range(0, (total_g_values - number_lts_g_values),
                       sts_step_size):
            lntts.append(g.x[i].tolist())
            g_values.append(g.y[i].tolist())
        lntts += g.x[
                 total_g_values - number_lts_g_values: total_g_values].tolist()
        g_values += g.y[
                    total_g_values - number_lts_g_values: total_g_values].tolist()
        for i in range(len(lntts)):
            output += str(round(lntts[i], 4)) + '\t' + \
                      str(round(g_values[i], 4)) + '\n'
        return output

    def simulate(self, method='hybrid'):
        B = self.B_spacing
        B_over_H = B / self.H

        self.bhe.update_thermal_resistance()
        self.calcAverageResistance()

        # Solve for equivalent single U-tube
        #print("Getting Equivalent Borehole Heat Exchanger")
        self.bhe_eq = plat.equivalance.compute_equivalent(self.bhe)
        # Update short time step object with equivalent single u-tube
        #print("Getting Radial Numerical gfunction values")
        self.radial_numerical.calc_sts_g_functions(self.bhe_eq)
        # Combine the short and long-term g-functions. The long term g-function
        # is interpolated for specific B/H and rb/H values.
        #print("Getting GFunction Values")
        g = self.grab_g_function(B_over_H)

        #print("Getting Loading Values")
        if method == 'hybrid':
            Q_dot = self.hybrid_load.load[2:] * 1000.  # convert to Watts
            time_values = self.hybrid_load.hour[2:]  # convert to seconds
            self.times = time_values
            self.loading = Q_dot

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
            if self.times == None:
                self.times = np.arange(1, n_hours + 1, 1)
            t = self.times

            HPEFT, dTb = self._simulate_detailed(Q_dot, t, g)
        else:
            raise ValueError('Only hybrid or hourly methods available.')

        self.HPEFT = HPEFT
        self.dTb = dTb

        max_HP_EFT = float(max(HPEFT))
        min_HP_EFT = float(min(HPEFT))
        return max_HP_EFT, min_HP_EFT

    def size(self, method='hybrid') -> None:
        # Size the ground heat exchanger

        def local_objective(H):
            self.setHeights(H)
            #print("H Vals: ",H)
            self.H = H
            max_HP_EFT, min_HP_EFT = self.simulate(method=method)
            T_excess = self.cost(max_HP_EFT, min_HP_EFT)
            return T_excess

        # Make the initial guess variable the average of the heights given
        #self.bhe.b.H = \
            #(self.sim_params.max_Height + self.sim_params.min_Height) / 2.
        # bhe.b.H is updated during sizing
        plat.equivalance.solve_root(
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

def multiDepthSquare(averageHeight,Betas,m,n,D,r_b,spacing = 5):
    coordinates = []
    indices = []
    bL = len(Betas)
    for i in range(m):
        for j in range(n):
            xEdgeLength = min(i,m-1-i)
            yEdgeLength = min(j,n-1-j)
            edgeLength=min(xEdgeLength,yEdgeLength)
            if edgeLength > bL-1:
                indices.append(len(Betas)-1)
            else:
                indices.append(edgeLength)
            coordinates.append([i*spacing,j*spacing])
    boreholes = []
    nbh = m*n
    if not nbh == len(coordinates) and not nbh == len(indices):
        print("Length Mismatch in Coordinate Generation")
    nBSum = 0
    #print("Indices: ",indices)
    for index in indices:
        nBSum += Betas[index]
    heights = []
    for b in Betas:
        heights.append(b*(averageHeight*nbh)/(nBSum))
    for h in heights:
        boreholes.append(gt.boreholes.Borehole(h, D, r_b, x=0., y=0.))
    if not len(Betas) == len(boreholes) and not len(Betas) == len(heights):
        print("Length Mismatch in Height Values")
    return [coordinates,indices,boreholes]

