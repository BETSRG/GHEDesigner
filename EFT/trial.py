import math
import numpy as np
import pygfunction as gt
import matplotlib.pyplot as plt
from MadisonHolberg import TotalRates
from ghedt import coordinates, utilities, gfunction
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

        # Maximum and minimum allowable fluid temperatures
        self.max_EFT_allowable = 30  # degrees Celsius
        self.min_EFT_allowable = -5  # degrees Celsius

        self.HourlyRej, self.HourlyExt = TotalRates.Rejection_Extraction_Rates(self.HLFile, self.HPFile,
                                                                               self.min_EFT_allowable,
                                                                               self.max_EFT_allowable)
        self.PLAT()
        #self.GetTimes()
        #self.Evaluating_Timesteps()
        #self.plot()


    # final output of class
    #def __repr__(self):
        #return str(self.HTS)

    def PLAT(self):

        # Borehole dimensions
        # -------------------
        self.H = 128.  # Borehole length (m)
        self.D = 2.  # Borehole buried depth (m)
        self.r_b = 0.055  # Borehole radius (m)
        self.B = 5.  # Borehole spacing (m)

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
        self.bhe_object = borehole_heat_exchangers.SingleUTube

        # Thermal conductivities
        # ----------------------
        k_p = 0.389  # Pipe thermal conductivity (W/m.K)
        self.k_s = 2.423  # Soil thermal conductivity (W/m.K)
        k_g = 0.744  # Grout thermal conductivity (W/m.K)

        # Volumetric heat capacities
        # --------------------------
        rhoCp_p = 1541.985 * 1000.  # Pipe volumetric heat capacity (J/K.m3)
        self.rhoCp_s = 2343.493 * 1000.  # Soil volumetric heat capacity (J/K.m3)
        rhoCp_g = 3901. * 1000.  # Grout volumetric heat capacity (J/K.m3)

        # Thermal properties
        # ------------------
        # Pipe
        self.pipe = media.Pipe(pos, r_in, r_out, s, epsilon, k_p, rhoCp_p)

        # Soil
        self.ugt = 9.6  # Undisturbed ground temperature (degrees Celsius)
        self.soil = media.Soil(self.k_s, self.rhoCp_s, self.ugt)
        # Grout
        self.grout = media.Grout(k_g, rhoCp_g)
        # Define a borehole
        self.borehole = gt.boreholes.Borehole(self.H, self.D, self.r_b, x=0., y=0.)

        # Inputs related to fluid
        # -----------------------
        # Fluid properties
        mixer = 'MPG'  # Propylene Glycol mixed with water
        percent = 15.  # Percentage of Propylene Glycol added in
        self.fluid = gt.media.Fluid(mixer=mixer, percent=percent)

        # Coordinates
        Nx = 12
        Ny = 13
        self.coordinates = coordinates.rectangle(Nx, Ny, self.B, self.B)

        # Fluid flow rate
        self.V_flow_borehole = 0.38  # Borehole volumetric flow rate (L/s)
        rho_fluid = 1016.22  # Fluid Density (kg/m^3)
        self.m_flow_borehole = self.V_flow_borehole / 1000. * rho_fluid

        self.bhe = borehole_heat_exchangers.SingleUTube(
            self.m_flow_borehole, self.fluid, self.borehole, self.pipe, self.grout, self.soil)
        self.radial_numerical = radial_numerical_borehole.RadialNumericalBH(self.bhe)

        # Simulation parameters
        # ---------------------
        # Simulation start month and end month
        start_month = 1
        n_years = 1
        end_month = n_years * 12

        # Maximum and minimum allowable heights
        max_Height = 384.  # in meters
        min_Height = 24  # in meters
        self.sim_params = media.SimulationParameters(
            start_month, end_month, self.max_EFT_allowable, self.min_EFT_allowable,
            max_Height, min_Height)

        #Getting Hybrid Time Step
        self.HTS = ground_loads.HybridLoad(
            self.HourlyRej, self.HourlyExt, self.bhe, self.radial_numerical, self.sim_params)

        return self.HTS

    def GetTimes(self):
        HTS = repr(self.HTS)
        df = HTS.split('\n')

        PLATOutput = []
        i = 0
        for line in df:
            if i >= 3:
                row = line.split(' ')
                NoSpace = []
                for cell in row:
                    if cell != '':
                        NoSpace.append(cell)
                PLATOutput.append(NoSpace)
            i += 1

        self.Start_Stop_Times = [0]
        self.PeakDays = []
        for row in range(len(PLATOutput)):
            Month = monthindex(PLATOutput[row][0])
            MonthStart = firstmonthhour(Month)-1
            Rej_Duration = float(PLATOutput[row][9])
            Ext_Duration = float(PLATOutput[row][10])

            if Rej_Duration <= 0.000001 and Ext_Duration <=0.000001:
                pass
            elif Rej_Duration <= 0.000001:
                Ext_Day = float(PLATOutput[row][8]) * 24 + MonthStart
                self.PeakDays.append(Ext_Day)
                if Ext_Day != MonthStart:
                    self.Start_Stop_Times.append(Ext_Day)
                self.Start_Stop_Times.append(Ext_Day+Ext_Duration)
            elif Ext_Duration <=0.000001:
                Rej_Day = float(PLATOutput[row][7]) * 24 + MonthStart
                self.PeakDays.append(Rej_Day)
                if Rej_Day != MonthStart:
                    self.Start_Stop_Times.append(Rej_Day)
                self.Start_Stop_Times.append(Rej_Day + Rej_Duration)
            else:
                Rej_Day = float(PLATOutput[row][7]) * 24 + MonthStart
                Ext_Day = float(PLATOutput[row][8]) * 24 + MonthStart
                if Rej_Day < Ext_Day:
                    self.PeakDays.append(Rej_Day)
                    if Rej_Day != MonthStart:
                        self.Start_Stop_Times.append(Rej_Day)
                    self.Start_Stop_Times.append(Rej_Day + Rej_Duration)

                    self.PeakDays.append(Ext_Day)
                    self.Start_Stop_Times.append(Ext_Day)
                    self.Start_Stop_Times.append(Ext_Day + Ext_Duration)

                elif Ext_Day < Rej_Day:
                    self.PeakDays.append(Ext_Day)
                    if Ext_Day != MonthStart:
                        self.Start_Stop_Times.append(Ext_Day)
                    self.Start_Stop_Times.append(Ext_Day + Ext_Duration)

                    self.PeakDays.append(Rej_Day)
                    self.Start_Stop_Times.append(Rej_Day)
                    self.Start_Stop_Times.append(Rej_Day + Rej_Duration)
            self.Start_Stop_Times.append(lastmonthhour(Month))
        return self.Start_Stop_Times, self.PeakDays

    def Evaluating_Timesteps(self):
        # Eskilson's original ln(t/ts) values
        self.lntts = utilities.Eskilson_log_times()  # aka log_time
        #time_values = np.exp(log_time) * ts (time scale)

        nSegments = 8
        segments = 'unequal'
        solver = 'equivalent'
        boundary = 'MIFT'
        end_length_ratio = 0.02
        segment_ratios = gt.utilities.segment_ratios(nSegments, end_length_ratio=end_length_ratio)

        g_func = gfunction.compute_live_g_function(
            self.B, [self.H], [self.r_b], [self.D], self.m_flow_borehole, self.bhe_object, self.lntts,
            self.coordinates, self.fluid, self.pipe, self.grout, self.soil, nSegments=nSegments,
            segments=segments, solver=solver, boundary=boundary,
            segment_ratios=segment_ratios)

        g = g_func.__dict__
        g_dict = g["g_lts"]
        g_items = list(g_dict.items())
        g_items_tuple = g_items[0]
        self.g = g_items_tuple[1]


        self.Rb = self.bhe.compute_effective_borehole_resistance()
        alpha = self.k_s / (self.rhoCp_s)
        tscale = (self.H ** 2) / (9 * alpha)  # time scale in seconds
        self.tsh = tscale / 3600  # time scale in hours
        self.nbh = 1  # number of boreholes
        self.rhoCp_f = 4066.43  # Fluid volumetric heat capacity (kJ/K.m3)
        self.mdotrhocp = self.V_flow_borehole * self.rhoCp_f  # Units of flow rate are L/s
                                                              # Units of rhocp are kJ/m3 K
                                                              # Units of mdotrhocp are then W/K

        self.HourlyExt = [0] + self.HourlyExt
        self.HourlyRej = [0] + self.HourlyRej
        self.AvgLoads = [0]
        self.sfLoads = [0]
        self.EFT = []
        for i in range(0,len(self.Start_Stop_Times)-1):
            Start = self.Start_Stop_Times[i] + 1
            Stop = self.Start_Stop_Times[i+1]
            EndEFT = self.TimeStep_Iteration(Start, Stop)
            self.EFT.append(EndEFT)

    def PeakPeriod(self, NetLoads):
        Highest = 0
        for i in range(len(NetLoads)):
            if abs(NetLoads[i]) > abs(Highest):
                Highest = NetLoads[i]
        return Highest

    def NonPeakPeriod(self, Start, Stop, NetLoads):
        StartLoad = 0
        StopLoad = 0
        Beginning = 0
        End = len(NetLoads)
        if Start %1 != 0:
            PartialLoad = math.floor(Start)
            Percent = 1 - (Start - PartialLoad)
            StartLoad = Percent * NetLoads[0]
            Beginning = 1
        if Stop %1 != 0:
            PartialLoad = math.ceil(Stop)
            Percent = 1-(PartialLoad - Stop)
            StopLoad = Percent * NetLoads[-1]
            End = len(NetLoads)-1
        Sum = 0
        for i in range(Beginning, End):
            Sum += NetLoads[i]
        Avg = (StartLoad + StopLoad + Sum) / ((Stop - Start) + 1)
        return Avg

    def FindAvg(self, mode, Start, Stop, NetLoads):
        if mode == 1:
            Avg = self.PeakPeriod(NetLoads)
        elif mode == 0:
            Avg = self.NonPeakPeriod(Start, Stop, NetLoads)
        self.AvgLoads[-1] = Avg
        self.sfLoads[-1] = (self.AvgLoads[-1] - self.AvgLoads[-2])

    def TimeStep_Iteration(self, Start, Stop):
        OldEFT = None
        self.AvgLoads.append(0)
        self.sfLoads.append(0)
        k = 0
        Beginning = math.floor(Start)
        End = math.ceil(Stop) + 1
        NetLoads = np.subtract(self.HourlyExt[Beginning:End], self.HourlyRej[Beginning:End])
        mode = 0
        for j in range(len(self.PeakDays)):
            if (Start-1) == self.PeakDays[j]:
                mode = 1
                break

        while k == 0:
            #print(NetLoads)
            self.FindAvg(mode, Start, Stop, NetLoads)
            NewEFT = self.HPEFT()
            NewRej, NewExt = TotalRates.Rejection_Extraction_Rates(self.HLFile, self.HPFile, NewEFT, NewEFT, Beginning-1, End-1)
            NetLoads = np.subtract(NewExt, NewRej)
            if OldEFT == None:
                pass
            else:
                if abs(OldEFT-NewEFT) <= 0.001:
                    k = 1
                    self.FindAvg(mode, Start, Stop, NetLoads)
            OldEFT = NewEFT
        return OldEFT

    def gi(self, lnttsi):
        gi = np.interp(lnttsi, self.lntts, self.g, left=0.001,
                       right=100)  # we should perhaps prevent going to the right of the
        # max lntts value by calculating an extrapolated large
        # maximum value
        return gi

    def HPEFT(self):
        n = len(self.AvgLoads)
        # calculate lntts for each time and then find g-function value
        DeltaTBHW = 0
        for j in range(1, n):  # This inner loop sums the responses of all previous step functions
            sftime = self.Start_Stop_Times[n - 1] - self.Start_Stop_Times[j-1]
            if sftime > 0:
                lnttssf = np.log(sftime / self.tsh)
                gsf = self.gi(lnttssf)
                stepfunctionload = 1000 * self.sfLoads[j] / (self.H * self.nbh)  # convert loads from total on GHE
                                                                            # in kW to W/m
                DeltaTBHW = DeltaTBHW + gsf * stepfunctionload / (2 * math.pi * self.k_s)
        DTBHW = DeltaTBHW

        # TBHW = Borehole wall temperature
        TBHW = DTBHW + self.ugt
        DT_wall_to_fluid = (1000. * self.AvgLoads[n - 1] / (self.H * self.nbh)) * self.Rb
        # MFT = Simple mean fluid temperature
        MFT = TBHW + DT_wall_to_fluid

        half_fluidDT = (1000. * self.AvgLoads[n - 1]) / (self.mdotrhocp * 2.)
        HPEFT = MFT - half_fluidDT

        if HPEFT > self.max_EFT_allowable:
            HPEFT = self.max_EFT_allowable
        elif HPEFT < self.min_EFT_allowable:
            HPEFT = self.min_EFT_allowable

        return HPEFT

    def plot(self):
        fig2 = gt.gfunction._initialize_figure()
        ax1 = fig2.add_subplot(111)
        ax2 = ax1.twinx()
        x = []
        y = [self.AvgLoads[0]]
        y2 = [self.AvgLoads[0]]
        for i in range(len(self.Start_Stop_Times)):
            x.append(self.Start_Stop_Times[i])
            x.append(self.Start_Stop_Times[i])
            if i != 0:
                y.append(self.AvgLoads[i])
                y.append(self.AvgLoads[i])
                z = self.AvgLoads[i] / (self.nbh*self.H)
                y2.append(z)
                y2.append(z)
        y.append(self.AvgLoads[0])
        y2.append(self.AvgLoads[0])

        ax1.plot(x, y, 'g-')
        ax2.plot(self.Start_Stop_Times[1:], self.EFT, '--b', marker = '.')
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

HTS = ans.PLAT()
#SST, PD = ans.GetTimes()
HTS = repr(HTS)
print(HTS)
#print(PD)
#print(SST)

