import math
import numpy as np
import pandas as pd
import pygfunction as gt
import scipy.interpolate
import matplotlib.pyplot as plt
from EFT import TotalRates
from ghedt import coordinates, utilities, gfunction, ground_heat_exchangers
from ghedt.peak_load_analysis_tool import borehole_heat_exchangers, media, radial_numerical_borehole, ground_loads

def monthindex(mname):
    months = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5,
              'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10,
              'November': 11, 'December': 12}
    mi = months.get(mname)
    return mi

def monthdays(month):
    if month > 12:
        md = month % 12
    else:
        md = month
    ndays = [31, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    monthdays = ndays[md]
    return monthdays

def firstmonthhour(month):
    fmh = 1
    if month > 1:
        for i in range(1, month):
            mi = i % 12
            fmh = fmh + 24 * monthdays(mi)
    return fmh

def lastmonthhour(month):
    lmh = 0
    for i in range(1, month + 1):
        lmh = lmh + monthdays(i) * 24
    if month == 1:
        lmh = 31 * 24
    return lmh

class FindEFT:
    def __init__(self, HourlyLoadFile, HPDataFilie):

        self.HLFile = HourlyLoadFile
        self.HPFile = HPDataFilie
        self.Setup()
        #self.GetTimes()
        self.GetTimes2()
        self.Evaluating_Timesteps()
        self.Data_Output()
        self.plot()

        # final output of class
        # def __repr__(self):
        # return str(self.HTS)

    def Setup(self):
        self.ugt = 17.1  # Undisturbed ground temperature (degrees Celsius)

        # Maximum and minimum allowable fluid temperatures
        self.max_EFT_allowable = self.ugt + 17  # degrees Celsius
        self.min_EFT_allowable = self.ugt - 10  # degrees Celsius

        self.HourlyRej, self.HourlyExt = TotalRates.Rejection_Extraction_Rates(self.HLFile, self.HPFile,
                                                                               self.min_EFT_allowable,
                                                                               self.max_EFT_allowable)

        # Borehole dimensions
        # -------------------
        self.nbh = 96  # number of boreholes
        self.H = 121.92 # Borehole length (m)
        D = 4.5  # Borehole buried depth (m)
        r_b = 0.055  # Borehole radius (m)
        B = 4.572  # Borehole spacing (m)

        # Pipe dimensions
        # ---------------
        # Single U-tube
        r_out = 26.67 / 1000. / 2.  # Pipe outer radius (m)
        r_in = 21.82 / 1000. / 2.  # Pipe inner radius (m)
        s = 18.87 / 1000.  # Inner-tube to inner-tube Shank spacing (m)
        epsilon = 1.0e-6  # Pipe roughness (m)

        # Pipe positions
        # --------------
        # Single U-tube [(x_in, y_in), (x_out, y_out)]
        pos = media.Pipe.place_pipes(s, r_out, 1)
        bhe_object = borehole_heat_exchangers.SingleUTube

        # Thermal conductivities
        # ----------------------
        k_p = 0.389  # Pipe thermal conductivity (W/m.K)
        self.k_s = 2.1499  # Soil thermal conductivity (W/m.K)
        k_g = 0.744  # Grout thermal conductivity (W/m.K)

        # Volumetric heat capacities
        # --------------------------
        rhoCp_p = 1541.985 * 1000.  # Pipe volumetric heat capacity (J/K.m3)
        rhoCp_s = 2343.493 * 1000.  # Soil volumetric heat capacity (J/K.m3)
        rhoCp_g = 3901. * 1000.  # Grout volumetric heat capacity (J/K.m3)
        rhoCp_f = 4173.8  # Fluid volumetric heat capacity (kJ/K.m3)

        # Thermal properties
        # ------------------
        # Pipe
        pipe = media.Pipe(pos, r_in, r_out, s, epsilon, k_p, rhoCp_p)

        # Soil

        soil = media.Soil(self.k_s, rhoCp_s, self.ugt)
        # Grout
        grout = media.Grout(k_g, rhoCp_g)
        # Define a borehole
        borehole = gt.boreholes.Borehole(self.H, D, r_b, x=0., y=0.)

        # Inputs related to fluid
        # -----------------------
        # Fluid properties
        mixer = 'MEG'  # Ethylene glycol mixed with water
        percent = 0.  # Percentage of Propylene Glycol added in
        fluid = gt.media.Fluid(mixer=mixer, percent=percent)

        # Coordinates
        Nx = 6
        Ny = 16
        coordinate = coordinates.rectangle(Nx, Ny, B, B)

        # Fluid flow rate
        #18.432 or  0.192
        V_flow_borehole = 0.192  # Borehole volumetric flow rate (L/s)
        rho_fluid = 998.2  # Fluid Density (kg/m^3)
        m_flow_borehole = V_flow_borehole / 1000. * rho_fluid
        self.mdotrhocp = V_flow_borehole *self.nbh * rhoCp_f  # Units of flow rate are L/s
                                                    # Units of rhocp are kJ/m3 K
                                                    # Units of mdotrhocp are then W/K

        bhe = borehole_heat_exchangers.SingleUTube(m_flow_borehole, fluid, borehole, pipe, grout, soil)
        radial_numerical = radial_numerical_borehole.RadialNumericalBH(bhe)

        # Simulation parameters
        # ---------------------
        # Simulation start month and end month
        start_month = 1
        n_years = 1
        end_month = n_years * 12

        # Maximum and minimum allowable heights
        max_Height = 384.  # in meters
        min_Height = 24  # in meters
        sim_params = media.SimulationParameters(
            start_month, end_month, self.max_EFT_allowable, self.min_EFT_allowable,
            max_Height, min_Height)

        # Getting Hybrid Time Step
        self.HTS = ground_loads.HybridLoad(self.HourlyRej, self.HourlyExt, bhe, radial_numerical, sim_params)

        # g-func variables
        # Values copied from live_g_function_sim_and_size.py
        nSegments = 8
        segments = 'unequal'
        solver = 'equivalent'
        boundary = 'MIFT'
        end_length_ratio = 0.02
        segment_ratios = gt.utilities.segment_ratios(nSegments, end_length_ratio=end_length_ratio)

        # Eskilson's original ln(t/ts) values
        self.lntts_lts = utilities.Eskilson_log_times()  # aka log_time

        g_func = gfunction.compute_live_g_function(B, [self.H], [r_b], [D], m_flow_borehole, bhe_object, self.lntts_lts,
                                                   coordinate, fluid, pipe, grout, soil, nSegments=nSegments,
                                                   segments=segments, solver=solver, boundary=boundary,
                                                   segment_ratios=segment_ratios)

        # converting g_func output into a usable form (list)
        g = g_func.__dict__
        g_dict = g["g_lts"]
        g_items = list(g_dict.items())
        g_items_tuple = g_items[0]
        self.g_lts = g_items_tuple[1]

        self.Rb = bhe.compute_effective_borehole_resistance()

        #g_function_corrected = gfunction.GFunction.borehole_radius_correction(g_lts, r_b, self.Rb)
        # Don't Update the HybridLoad (its dependent on the STS) because
        # it doesn't change the results much and it slows things down a lot
        # combine the short and long time step g-function
        self.lntts_sts, self.g_sts = radial_numerical.calc_sts_g_functions(bhe)




        self.g = ground_heat_exchangers.BaseGHE.combine_sts_lts(
            self.lntts_lts, self.g_lts,
            self.lntts_sts.tolist(),
            self.g_sts.tolist())

        #GLHEPRO = np.genfromtxt('gvals.csv', dtype=None, delimiter=",", encoding=None, unpack=False)
        #self.lntts = GLHEPRO[0]
        #self.g = GLHEPRO[1]

        # Time scale calculation

        alpha = self.k_s / (rhoCp_s)
        tscale = (self.H ** 2) / (9 * alpha)  # time scale in seconds
        self.tsh = tscale / 3600  # time scale in hours


    def GetTimes(self):
        # Convert HTS to strings and split into lines
        HTS = repr(self.HTS)
        df = HTS.split('\n')

        PLATOutput = []  # matrix containing the following values for each month: Month(string), total rej, total ext,
                         # peak rej, peak ext, avg rej, avg ext, peak rej day, peak ext day, peak rej duration, and
                         # peak ext duration
        i = 0
        # isolate cells with actual data in them, not just a space
        for line in df:
            if i >= 3:
                row = line.split(' ')
                NoSpace = []
                for cell in row:
                    if cell != '':
                        NoSpace.append(cell)
                PLATOutput.append(NoSpace)
            i += 1

        self.Start_Stop_Times = [0]  # array for start and stop time(hour) of each timestep
                                     # stop of one timestep is start of the next
        self.PeakDays = []  # array to hold the start time (in hours) of each peak

        # runs through each row in PLATOutput and finds start and stop times for each timestep
        # if the duration of a peak period is 0.000001 it's treated as 0
        for row in range(len(PLATOutput)):
            Month = monthindex(PLATOutput[row][0])
            MonthStart = firstmonthhour(Month) - 1
            Rej_Duration = float(PLATOutput[row][9])
            Ext_Duration = float(PLATOutput[row][10])

            # month has no peak periods
            if Rej_Duration <= 0.000001 and Ext_Duration <= 0.000001:
                pass

            # month only has a peak extraction period
            elif Rej_Duration <= 0.000001:
                Ext_Day = float(PLATOutput[row][8]) * 24 + MonthStart
                self.PeakDays.append(Ext_Day)
                if Ext_Day != MonthStart:  # Checks if peak day occurs on first day of month so that time
                    self.Start_Stop_Times.append(Ext_Day)                  # is not re-added to the array
                self.Start_Stop_Times.append(Ext_Day + Ext_Duration)

            # month only has a peak rejection period
            elif Ext_Duration <= 0.000001:
                Rej_Day = float(PLATOutput[row][7]) * 24 + MonthStart
                self.PeakDays.append(Rej_Day)
                if Rej_Day != MonthStart:  # Checks if peak day occurs on first day of month so that time
                    self.Start_Stop_Times.append(Rej_Day)                  # is not re-added to the array
                self.Start_Stop_Times.append(Rej_Day + Rej_Duration)

            # month has both a peak rejection and peak extraction period
            else:
                Rej_Day = float(PLATOutput[row][7]) * 24 + MonthStart
                Ext_Day = float(PLATOutput[row][8]) * 24 + MonthStart

                # peak rejection occurs first
                if Rej_Day < Ext_Day:
                    self.PeakDays.append(Rej_Day)
                    if Rej_Day != MonthStart:  # Checks if peak day occurs on first day of month so that time
                        self.Start_Stop_Times.append(Rej_Day)                  # is not re-added to the array
                    self.Start_Stop_Times.append(Rej_Day + Rej_Duration)

                    self.PeakDays.append(Ext_Day)
                    self.Start_Stop_Times.append(Ext_Day)
                    self.Start_Stop_Times.append(Ext_Day + Ext_Duration)

                # peak extraction occurs first
                elif Ext_Day < Rej_Day:
                    self.PeakDays.append(Ext_Day)
                    if Ext_Day != MonthStart:  # Checks if peak day occurs on first day of month so that time
                        self.Start_Stop_Times.append(Ext_Day)                  # is not re-added to the array
                    self.Start_Stop_Times.append(Ext_Day + Ext_Duration)

                    self.PeakDays.append(Rej_Day)
                    self.Start_Stop_Times.append(Rej_Day)
                    self.Start_Stop_Times.append(Rej_Day + Rej_Duration)
                # Note
                # Same day is not taken into consideration in this if statement since currently if the peaks are on the
                # same day one has a duration of 0 and doesn't need to be considered

            self.Start_Stop_Times.append(lastmonthhour(Month))
        return self.Start_Stop_Times, self.PeakDays

    def GetTimes2(self):
        # Convert HTS to strings and split into lines
        HTS = repr(self.HTS)
        df = HTS.split('\n')

        PLATOutput = []  # matrix containing the following values for each month: Month(string), total rej, total ext,
                         # peak rej, peak ext, avg rej, avg ext, peak rej day, peak ext day, peak rej duration, and
                         # peak ext duration
        i = 0
        # isolate cells with actual data in them, not just a space
        for line in df:
            if i >= 3:
                row = line.split(' ')
                NoSpace = []
                for cell in row:
                    if cell != '':
                        NoSpace.append(cell)
                PLATOutput.append(NoSpace)
            i += 1

        self.Start_Stop_Times = [0]  # array for start and stop time(hour) of each timestep
                                     # stop of one timestep is start of the next
        self.PeakDays = []  # array to hold the start time (in hours) of each peak

        # runs through each row in PLATOutput and finds start and stop times for each timestep
        # if the duration of a peak period is 0.000001 it's treated as 0
        for row in range(len(PLATOutput)):
            Month = monthindex(PLATOutput[row][0])
            MonthStart = firstmonthhour(Month) - 1
            Rej_Duration = float(PLATOutput[row][9])
            Ext_Duration = float(PLATOutput[row][10])

            # month has no peak periods
            if Rej_Duration <= 0.000001 and Ext_Duration <= 0.000001:
                pass

            # month only has a peak extraction period
            elif Rej_Duration <= 0.000001:
                Ext_Day = float(PLATOutput[row][8]) * 24 + MonthStart
                peak = 0
                for i in range(int(Ext_Day), int(Ext_Day + 24)):
                    val = self.HourlyExt[i]
                    if val > peak:
                        peak = val
                        t2 = i +1
                t1 = t2 - Ext_Duration
                self.Start_Stop_Times.append(t1)
                self.Start_Stop_Times.append(t2)
                self.PeakDays.append(t1)

            # month only has a peak rejection period
            elif Ext_Duration <= 0.000001:
                Rej_Day = float(PLATOutput[row][7]) * 24 + MonthStart
                peak = 0
                for i in range(int(Rej_Day), int(Rej_Day + 24)):
                    val = self.HourlyRej[i]
                    if val > peak:
                        peak = val
                        t2 = i +1
                t1 = t2 - Rej_Duration
                self.Start_Stop_Times.append(t1)
                self.Start_Stop_Times.append(t2)
                self.PeakDays.append(t1)

            # month has both a peak rejection and peak extraction period
            else:
                Rej_Day = float(PLATOutput[row][7]) * 24 + MonthStart
                Ext_Day = float(PLATOutput[row][8]) * 24 + MonthStart

                # peak rejection occurs first
                if Rej_Day < Ext_Day:
                    peak = 0
                    for i in range(int(Rej_Day), int(Rej_Day + 24)):
                        val = self.HourlyRej[i]
                        if val > peak:
                            peak = val
                            t2 = i +1
                    t1 = t2 - Rej_Duration
                    self.Start_Stop_Times.append(t1)
                    self.Start_Stop_Times.append(t2)
                    self.PeakDays.append(t1)

                    peak = 0
                    for i in range(int(Ext_Day), int(Ext_Day + 24)):
                        val = self.HourlyExt[i]
                        if val > peak:
                            peak = val
                            t2 = i +1
                    t1 = t2 - Ext_Duration
                    self.Start_Stop_Times.append(t1)
                    self.Start_Stop_Times.append(t2)
                    self.PeakDays.append(t1)

                # peak extraction occurs first
                elif Ext_Day < Rej_Day:
                    peak = 0
                    for i in range(int(Ext_Day), int(Ext_Day + 24)):
                        val = self.HourlyExt[i]
                        if val > peak:
                            peak = val
                            t2 = i +1
                    t1 = t2 - Ext_Duration
                    self.Start_Stop_Times.append(t1)
                    self.Start_Stop_Times.append(t2)
                    self.PeakDays.append(t1)

                    peak = 0
                    for i in range(int(Rej_Day), int(Rej_Day + 24)):
                        val = self.HourlyRej[i]
                        if val > peak:
                            peak = val
                            t2 = i + 1
                    t1 = t2 - Rej_Duration
                    self.Start_Stop_Times.append(t1)
                    self.Start_Stop_Times.append(t2)
                    self.PeakDays.append(t1)
                # Note
                # Same day is not taken into consideration in this if statement since currently if the peaks are on the
                # same day one has a duration of 0 and doesn't need to be considered

            self.Start_Stop_Times.append(lastmonthhour(Month))
        return self.Start_Stop_Times, self.PeakDays

    def Evaluating_Timesteps(self):
        # adds 0 to beginning to represent time before loads were applied (needed for HPEFT calculations)
        self.HourlyExt = [0] + self.HourlyExt
        self.HourlyRej = [0] + self.HourlyRej

        self.AvgLoads = [0]  # list of the final average load for each timestep
        self.sfLoads = [0]  # list for final sf load in each timestep
        self.EFT = []  # empty list to store final EFT val at end of every timestep
        self.FinalRej = []
        self.FinalExt = []

        for i in range(0, len(self.Start_Stop_Times) - 1):
            Start = self.Start_Stop_Times[i] + 1  # adds 1 to start for indexing purposes
            Stop = self.Start_Stop_Times[i + 1]
            EndEFT, Rej, Ext = self.TimeStep_Iteration(Start, Stop)
            if Start % 1 == 0:
                self.FinalRej = self.FinalRej + Rej[:-1]
                self.FinalExt = self.FinalExt + Ext[:-1]
            else:
                Load2_Rej = Rej[0]
                Load2_Ext = Ext[0]
                P2 = math.ceil(Start)-Start
                CorrectedRej = Load2_Rej*P2 + Load1_Rej*P1
                CorrectedExt = Load2_Ext * P2 + Load1_Ext * P1
                self.FinalRej.append(CorrectedRej)
                self.FinalExt.append(CorrectedExt)
                if len(Rej)>2:
                    self.FinalRej = self.FinalRej + Rej[1:-1]
                    self.FinalExt = self.FinalExt + Ext[1:-1]
            if Stop % 1 == 0:
                self.FinalRej.append(Rej[-1])
                self.FinalExt.append(Ext[-1])
            else:
                Load1_Rej = Rej[-1]
                Load1_Ext = Ext[-1]
                P1 = Stop - math.floor(Stop)
            self.EFT.append(EndEFT)

    def PeakPeriod(self, NetLoads):
        Highest = 0
        # loops through all loads to find highest net load
        # We don't care if start or stop are decimals as we use the entire load to find the highest
        for i in range(len(NetLoads)):
            if abs(NetLoads[i]) > abs(Highest):
                Highest = NetLoads[i]
        return Highest

    def NonPeakPeriod(self, Start, Stop, NetLoads):
        StartLoad = 0
        StopLoad = 0
        Beginning = 0
        End = len(NetLoads)
        if Start % 1 != 0:  # checks if the start value is a decimal
            PartialLoad = math.floor(Start)
            Percent = 1 - (Start - PartialLoad)
            StartLoad = Percent * NetLoads[0]
            Beginning = 1  # set to 1 so we don't add the partial load
        if Stop % 1 != 0:  # checks if the stop value is a decimal
            PartialLoad = math.ceil(Stop)
            Percent = 1 - (PartialLoad - Stop)
            StopLoad = Percent * NetLoads[-1]
            End = len(NetLoads) - 1  # ends one early so partial load isn't included
        Sum = 0
        for i in range(Beginning, End):  # sums all the full loads
            Sum += NetLoads[i]
        k = (StartLoad + StopLoad + Sum)
        z = ((Stop - Start) + 1)  # adds partial loads to sum then divides by time
        Avg = k/z
        return Avg

    def FindAvg(self, mode, Start, Stop, NetLoads):
        if mode == 1:
            Avg = NetLoads[-1]
        elif mode == 0:
            Avg = self.NonPeakPeriod(Start, Stop, NetLoads)

        # replaces last value in list with new values
        self.AvgLoads[-1] = Avg
        self.sfLoads[-1] = (self.AvgLoads[-1] - self.AvgLoads[-2])

    def TimeStep_Iteration(self, Start, Stop):
        OldEFT = None

        # add 0 to following arrays so we can keep replacing the last value in list as we iterate
        self.AvgLoads.append(0)
        self.sfLoads.append(0)

        # rounds start and stop times so we can pull the appropriate values from our hourly ext and rej loads
        Beginning = math.floor(Start)
        End = math.ceil(Stop) + 1
        NetLoads = np.subtract(self.HourlyRej[Beginning:End], self.HourlyExt[Beginning:End]).tolist()

        k = 0
        mode = 0
        # this for loop determines if this is a peak period we're dealing with
        # assumes we're not in a peak period unless this loop says otherwise
        for j in range(len(self.PeakDays)):
            if (Start - 1) == self.PeakDays[j]:
                mode = 1
                break

        while k == 0:
            #print(NetLoads)
            self.FindAvg(mode, Start, Stop, NetLoads)
            NewEFT = self.HPEFT()
            # recalculates loads based on new EFT
            NewRej, NewExt = TotalRates.Rejection_Extraction_Rates(self.HLFile, self.HPFile, NewEFT, NewEFT,
                                                                   Beginning - 1, End - 1)
            NetLoads = np.subtract(NewRej, NewExt).tolist()

            # checks if the new EFT is close enough to the previous EFT to end the iteration process
            if OldEFT == None:
                pass
            else:
                if abs(OldEFT - NewEFT) <= 0.001:
                    k = 1
                    self.FindAvg(mode, Start, Stop, NetLoads)  # calculates final Avg and sf loads for this timestep
            OldEFT = NewEFT
        return OldEFT, NewRej, NewExt

    def gi(self, lnttsi):
        gi = self.g(lnttsi)
        #gi = np.interp(lnttsi, self.lntts, self.g, left=0.001, right=100)
        # we should perhaps prevent going to the right of the
        # max lntts value by calculating an extrapolated large maximum value
        return gi

    def HPEFT(self):
        n = len(self.AvgLoads)
        DeltaTBHW = 0

        # calculate lntts for each time and then find g-function value for that time
        for j in range(1, n):  # This loop sums the responses of all previous step functions
            sftime = self.Start_Stop_Times[n - 1] - self.Start_Stop_Times[j - 1]
            if sftime > 0:
                lnttssf = np.log(sftime / self.tsh)
                gsf = self.gi(lnttssf)
                stepfunctionload = 1000 * self.sfLoads[j] / (self.H * self.nbh)  # convert loads from total on GHE
                                                                                 # in kW to W/m
                DeltaTBHW = DeltaTBHW + ((gsf * stepfunctionload) / (2 * math.pi * self.k_s))
        DTBHW = DeltaTBHW

        TBHW = DTBHW + self.ugt  # TBHW = Borehole wall temperature
        DT_wall_to_fluid = (1000. * self.AvgLoads[n - 1] / (self.H * self.nbh)) * self.Rb

        MFT = TBHW + DT_wall_to_fluid  # MFT = Simple mean fluid temperature

        half_fluidDT = (1000. * self.AvgLoads[n - 1]) / (self.mdotrhocp * 2.)
        HPEFT = MFT - half_fluidDT

        # checks if calculated EFT exceeds the min or max allowable EFT values
        if HPEFT > self.max_EFT_allowable:
            HPEFT = self.max_EFT_allowable
        elif HPEFT < self.min_EFT_allowable:
            HPEFT = self.min_EFT_allowable

        return HPEFT

    def Data_Output(self):
        FinalNet = np.subtract(self.FinalExt, self.FinalRej)
        Rej = [-x for x in self.FinalRej]
        AvgL = self.AvgLoads[1:]
        Avg = [i * -1 for i in AvgL]
        Avg_per_Meter = [i / (self.H*self.nbh) for i in Avg]
        hours = list(range(1, 8761))
        timestep = self.Start_Stop_Times[1:]
        data = {'Hour': hours, 'Rejection Rate (kW)': Rej, 'Extraction Rate (kW)': self.FinalExt,
                'Net Rate (kw)': FinalNet}
        data2 = {'Timestep Hours': timestep, 'Average Load (kW)': Avg, 'Average Load (kW/m)': Avg_per_Meter,
                 'EFT (C)': self.EFT}
        df = pd.DataFrame(data)
        df2 = pd.DataFrame(data2)
        header_list = [' ', 'Timestep Hours', 'Average Load (kW)', 'Average Load (kW/m)', 'EFT (C)']
        df2 = df2.reindex(columns=header_list)
        (pd.concat([df, df2], axis=1)).to_csv('Outputs.csv', index=False)

        hours_Plot = []
        Net_Plot = [0]
        Rej_Plot = [0]
        Ext_Plot = [0]
        timestep_Plot = [0,0]
        Avg_Plot = [0]
        Avg_per_Meter_Plot = [0]

        for j in range(len(hours)):
            hours_Plot.append(hours[j])
            hours_Plot.append(hours[j])
            if j != 0:
                Net_Plot.append(FinalNet[j])
                Net_Plot.append(FinalNet[j])
                Rej_Plot.append(Rej[j])
                Rej_Plot.append(Rej[j])
                Ext_Plot.append(self.FinalExt[j])
                Ext_Plot.append(self.FinalExt[j])
        Net_Plot.append(0)
        Rej_Plot.append(0)
        Ext_Plot.append(0)

        for k in range(len(timestep)):
            timestep_Plot.append(timestep[k])
            timestep_Plot.append(timestep[k])
            Avg_Plot.append(Avg[k])
            Avg_Plot.append(Avg[k])
            Avg_per_Meter_Plot.append(Avg_per_Meter[k])
            Avg_per_Meter_Plot.append(Avg_per_Meter[k])
        Avg_Plot.append(0)
        Avg_per_Meter_Plot.append(0)

        data3 = {'Hour': hours_Plot, 'Rejection Rate (kW)': Rej_Plot, 'Extraction Rate (kW)': Ext_Plot,
                'Net Rate (kw)': Net_Plot}
        data4 = {'Timestep Hours': timestep_Plot, 'Average Load (kW)': Avg_Plot,
                 'Average Load (kW/m)': Avg_per_Meter_Plot}
        data5 = {'Timestep Hours': timestep, 'EFT (C)': self.EFT}
        df3 = pd.DataFrame(data3)
        df4 = pd.DataFrame(data4)
        df5 = pd.DataFrame(data5)
        header_list = [' ', 'Timestep Hours', 'Average Load (kW)', 'Average Load (kW/m)', ' ']
        df4 = df4.reindex(columns=header_list)
        (pd.concat([df3, df4, df5], axis=1)).to_csv('Outputs-Plotting.csv', index=False)

    def plot(self):
        Q = []
        Months = [0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016, 8760]
        Net = np.subtract(self.FinalExt, self.FinalRej)
        for j in range(len(Months)-1):
            rates = Net[Months[j]:Months[j+1]]
            sum = 0
            for k in range(len(rates)):
                sum += rates[k]
            avg = sum/(Months[j+1]-Months[j])
            z = avg * 1000 / (self.H*self.nbh)
            Q.append(z)

        print(Q)
        print(self.EFT)

        fig2 = gt.gfunction._initialize_figure()
        ax1 = fig2.add_subplot(111)
        ax2 = ax1.twinx()
        x = []
        y = [0]

        for i in range(len(self.Start_Stop_Times)):
            x.append(self.Start_Stop_Times[i])
            x.append(self.Start_Stop_Times[i])
            if i != 0:
                y.append(-1*self.AvgLoads[i])
                y.append(-1*self.AvgLoads[i])
        y.append(0)

        ax1.plot(x, y, 'g-')
        ax2.plot(self.Start_Stop_Times[1:], self.EFT, '--b', marker='.')
        ax1.legend(['Average Ext/Rej Load'], loc=3, prop={'size': 8})
        ax2.legend(['End EFT Temperature'], loc=4, prop={'size': 8})
        ax1.set_xlim(xmin=-500, xmax=9260)
        ax1.hlines(y=0, xmin=-500, xmax=9260, linewidth=1, color='k')
        ax1.vlines(x=744, ymin=-6, ymax=10, linewidth=0.5, color='r', linestyles='dashed')
        ax1.vlines(x=1416, ymin=-6, ymax=10, linewidth=0.5, color='r', linestyles='dashed')
        ax1.vlines(x=2160, ymin=-6, ymax=10, linewidth=0.5, color='r', linestyles='dashed')
        ax1.vlines(x=2880, ymin=-6, ymax=10, linewidth=0.5, color='r', linestyles='dashed')
        ax1.vlines(x=3624, ymin=-6, ymax=10, linewidth=0.5, color='r', linestyles='dashed')
        ax1.vlines(x=4344, ymin=-6, ymax=10, linewidth=0.5, color='r', linestyles='dashed')
        ax1.vlines(x=5088, ymin=-6, ymax=10, linewidth=0.5, color='r', linestyles='dashed')
        ax1.vlines(x=5832, ymin=-6, ymax=10, linewidth=0.5, color='r', linestyles='dashed')
        ax1.vlines(x=6552, ymin=-6, ymax=10, linewidth=0.5, color='r', linestyles='dashed')
        ax1.vlines(x=7296, ymin=-6, ymax=10, linewidth=0.5, color='r', linestyles='dashed')
        ax1.vlines(x=8016, ymin=-6, ymax=10, linewidth=0.5, color='r', linestyles='dashed')
        ax1.set_xlabel('Hours')
        ax1.set_ylabel('Kwh')
        ax2.set_ylabel('Celsius', color='b')

        plt.show()

ans = FindEFT('HLTest4.csv', 'HPTest2.csv')

