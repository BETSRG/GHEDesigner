# Jack C. Cook
# Thursday, September 16, 2021
import warnings

import scipy.interpolate
import scipy.optimize

import GLHEDT.geothermal
import PLAT.borehole_heat_exchangers
import numpy as np


class HybridGLHE:
    def __init__(self, bhe: PLAT.borehole_heat_exchangers.SingleUTube,
                 radial_numerical: PLAT.radial_numerical_borehole.RadialNumericalBH,
                 hybrid_load: PLAT.ground_loads.HybridLoad,
                 GFunction: GLHEDT.geothermal.GFunction):
        # borehole heat exchanger object
        self.bhe = bhe
        # sts radial numerical object
        self.radial_numerical = radial_numerical
        # hybrid load object
        self.hybrid_load = hybrid_load
        # GFunction object
        self.GFunction = GFunction

        self.TBHW: list = []
        self.MFT: list = []
        self.HPEFT: list = []
        self.linehour: list = []
        self.loadperm: list = []

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
            while value < min_log_time_lts:
                i += 1
            log_time = log_time_sts[0:i] + log_time_lts
            g = g_sts[0:i] + g_lts

        g = scipy.interpolate.interp1d(log_time, g)

        return g

    @staticmethod
    def cost(max_EFT, min_EFT, max_EFT_allowable, min_EFT_allowable):
        delta_T_max = max_EFT - max_EFT_allowable
        delta_T_min = min_EFT_allowable - min_EFT

        T_excess = max([delta_T_max, delta_T_min])

        return T_excess

    def size(self, max_H, min_H, max_EFT_allowable, min_EFT_allowable, B=None) -> None:
        # Size the ground heat exchanger

        def local_objective(H):
            self.bhe.b.H = H
            max_HP_EFT, min_HP_EFT = self.simulate(B=B)
            T_excess = self.cost(max_HP_EFT, min_HP_EFT,
                                      max_EFT_allowable, min_EFT_allowable)
            return T_excess
        # Make the initial guess variable the average of the heights given
        self.bhe.b.H = (max_H + min_H) / 2.
        # bhe.b.H is updated during sizing
        PLAT.equivalance.solve_root(
            self.bhe.b.H, local_objective, lower=min_H, upper=max_H)
        # TODO: look and see if equivalence.solve_root is returning the opposite of what we want when not bracketed
        if self.bhe.b.H == min_H:
            warnings.warn('The maximum height provided to size this ground '
                          'heat exchanger is not deep enough. Provide a deeper '
                          'allowable depth or increase the size of the heat '
                          'exchanger.')
        if self.bhe.b.H == max_H:
            warnings.warn('The minimum height provided to size this ground heat'
                          ' exchanger is not shallow enough. Provide a '
                          'shallower allowable depth or decrease the size of '
                          'the heat exchanger.')

        return

    def simulate(self, B=None):
        if B is None:
            B = self.GFunction.B
        else:
            B = B
        B_over_H = B / self.bhe.b.H
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

        # Apply the g-function to the loads to calculate the
        # alpha = myGHE.k / (1000 * myGHE.rhocp)
        alpha = self.bhe.soil.k / self.bhe.soil.rhoCp
        # tscale = (myGHE.depth ** 2) / (9 * alpha)  # time scale in seconds
        tscale = (self.bhe.b.H ** 2) / (9 * alpha)  # time scale in seconds
        tsh = tscale / 3600  # time scale in hours

        nbh = len(self.GFunction.bore_locations)

        # n = loads.hour.size
        n = self.hybrid_load.hour.size
        # calculate lntts for each time and then find g-function value
        lntts = []
        g_all = []
        TBHW = [0, 0]  # Borehole wall temperature
        DTBHW = [0, 0]
        MFT = [0, 0]  # Simple mean fluid temperature
        HPEFT = [0, 0]
        Pi = 3.141592
        for i in range(2,
                       n):  # This outer loop organizes the calculation of DeltaTBHW for the end of the ith period.
            DeltaTBHW = 0
            for j in range(1,
                           i):  # This inner loop sums the responses of all previous step functions
                # sftime = loads.hour[i] - loads.hour[j]
                sftime = self.hybrid_load.hour[i] - self.hybrid_load.hour[j]
                if sftime > 0:
                    lnttssf = np.log(sftime / tsh)
                    lntts.append(lnttssf)
                    # gsf = myGHE.gi(lnttssf)
                    gsf = g(lnttssf)
                    g_all.append(gsf)
                    # stepfunctionload = 1000 * loads.sfload[j + 1] / (myGHE.depth * myGHE.nbh)  # convert loads from total on GHE
                    stepfunctionload = 1000 * self.hybrid_load.sfload[j + 1] / (self.bhe.b.H * nbh)  # convert loads from total on GHE
                    # in kW to W/m
                    # DeltaTBHW = DeltaTBHW + gsf * stepfunctionload / (2 * Pi * myGHE.k)
                    DeltaTBHW = DeltaTBHW + gsf * stepfunctionload / (2 * Pi * self.bhe.soil.k)
            DTBHW.append(DeltaTBHW)

        for i in range(2, n):
            # TODO: move ugt to soil
            ugt = 18.3
            # Peak heating flow rate is the system flow rate in L/s
            peakhtgflowrate = self.bhe.m_flow_borehole / self.bhe.fluid.rho * 1000. * nbh
            # Peak cooling flow rate is the system flow rate in L/s
            peakclgflowrate = self.bhe.m_flow_borehole / self.bhe.fluid.rho * 1000. * nbh
            # Peak heating volumetric capacity is in kJ/K.m3
            peakhtgrhocp = self.bhe.fluid.rhoCp / 1000.
            # Peak cooling volumetric capacity is in kJ/k.m3
            peakclgrhocp = self.bhe.fluid.rhoCp / 1000.

            # TBHW.append(DTBHW[i] + myGHE.ugt)
            TBHW.append(DTBHW[i] + ugt)
            # DT_wall_to_fluid = (1000. * loads.load[i] / (myGHE.depth * myGHE.nbh)) * myGHE.bhresistance
            DT_wall_to_fluid = (1000. * self.hybrid_load.load[i] / (self.bhe.b.H * nbh)) * self.bhe.compute_effective_borehole_resistance()
            MFT.append(TBHW[i] + DT_wall_to_fluid)
            # if loads.hour[i] > 0:
            if self.hybrid_load.hour[i] > 0:
                # mdotrhocp = myGHE.peakclgflowrate * myGHE.peakclgrhocp  # Units of flow rate are L/s or 0.001 m3/s
                mdotrhocp = peakclgflowrate * peakclgrhocp  # Units of flow rate are L/s or 0.001 m3/s
                # Units of rhocp are kJ/m3 K
                # Units of mdotrhocp are then W/K
            else:
                # mdotrhocp = myGHE.peakhtgflowrate * myGHE.peakhtgrhocp
                mdotrhocp = peakhtgflowrate * peakhtgrhocp
            # half_fluidDT = (1000. * loads.load[i]) / (mdotrhocp * 2.)
            half_fluidDT = (1000. * self.hybrid_load.load[i]) / (mdotrhocp * 2.)
            HPEFT.append(MFT[i] - half_fluidDT)
            # load_per_m = 1000 * loads.load[i] / (myGHE.depth * myGHE.nbh)  # to check W/m
            load_per_m = 1000 * self.hybrid_load.load[i] / (self.bhe.b.H * nbh)  # to check W/m
            linehour = self.hybrid_load.hour[i]

        def key_if_not(dnary, key, Type='list'):
            keys = list(dnary.keys())
            if key in keys:
                return
            elif Type is 'list':
                dnary[key] = []

        self.TBHW = TBHW
        self.MFT = MFT
        self.HPEFT = HPEFT
        self.linehour = self.hybrid_load.load.tolist()
        load_per_m_list = [1000 * self.hybrid_load.load[i] / (self.bhe.b.H * nbh)
                           for i in range(len(self.hybrid_load.load))]
        self.loadperm = load_per_m_list

        max_HP_EFT = float(max(HPEFT[2:n]))
        min_HP_EFT = float(min(HPEFT[2:n]))
        return max_HP_EFT, min_HP_EFT
