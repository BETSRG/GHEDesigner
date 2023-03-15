import warnings
from calendar import monthrange
from json import dumps
from math import floor

import numpy as np
from scipy.interpolate import interp1d

from ghedesigner.borehole_heat_exchangers import SingleUTube
from ghedesigner.constants import TWO_PI
from ghedesigner.radial_numerical_borehole import RadialNumericalBH
from ghedesigner.simulation import SimulationParameters


class HybridLoad:
    def __init__(
            self,
            raw_loads: list,
            bhe: SingleUTube,
            radial_numerical: RadialNumericalBH,
            sim_params: SimulationParameters,
            years=None,
    ):
        # Split the hourly loads into heating and cooling (kW)
        if years is None:
            years = [2019]

        self.hourly_rejection_loads, self.hourly_extraction_loads = self.split_heat_and_cool(raw_loads)

        # Simulation start and end month
        self.start_month = sim_params.start_month
        self.end_month = sim_params.end_month
        if len(years) <= 1:
            self.peak_retain_start = 12  # use peak loads for first 12 months
            self.peak_retain_end = 12  # use peak loads for last 12 months
        else:
            self.peak_retain_start = len(years) * 6
            self.peak_retain_end = len(years) * 6

        # Store the borehole heat exchanger
        self.bhe = bhe
        # Store the radial numerical g-function value
        # Note: this is intended to be a scipy.interp1d object
        self.radial_numerical = radial_numerical
        self.years = years

        # Get the number of days in each month for a given year (make 0 NULL)
        self.days_in_month = [0]
        for year in years:
            self.days_in_month.extend([monthrange(year, i)[1] for i in range(1, 13)])

        # TODO: verify whether errors are possible here and raise exception if needed
        # assert (len(hourly_rejection_loads) == sum(self.days_in_month) * 24.0
        #         and len(hourly_extraction_loads) == sum(self.days_in_month) * 24.0), (
        #     "The total number of hours in the year are not equal. Is this a leap year?")

        # This block of data holds the compact monthly representation of the
        # loads. The intention is that these loads will usually repeat. It's
        # possible that for validation or design purposes, users may wish to
        # specify loads that differ from year to year. For these arrays,
        # January is the second item (1) and December the last (12)
        # We'll reserve the first item (0) for an annual total or peak

        num_unique_months = len(years) * 12 + 1

        # monthly cooling loads (or heat rejection) in kWh
        self.monthly_cl = [0] * num_unique_months
        # monthly heating loads (or heat extraction) in kWh
        self.monthly_hl = [0] * num_unique_months
        # monthly peak cooling load (or heat rejection) in kW
        self.monthly_peak_cl = [0] * num_unique_months
        # monthly peak heating load (or heat extraction) in kW
        self.monthly_peak_hl = [0] * num_unique_months
        # monthly average cooling load (or heat rejection) in kW
        self.monthly_avg_cl = [0] * num_unique_months
        # monthly average heating load (or heat extraction) in kW
        self.monthly_avg_hl = [0] * num_unique_months
        # day of the month on which peak clg load occurs (e.g. 1-31)
        self.monthly_peak_cl_day = [0] * num_unique_months
        # day of the month on which peak htg load occurs (e.g. 1-31)
        self.monthly_peak_hl_day = [0] * num_unique_months
        # Process the loads by month
        self.split_loads_by_month()

        # 48 hour loads are going to be necessary for the hourly simulation for
        # finding the peak load duration
        # These will be a 2D list, a list of 48 hour loads in each index
        # Make 0 position NULL
        # list of two day (48 hour) cooling loads (or heat rejection) in kWh
        self.two_day_hourly_peak_cl_loads = [[0]]
        # list of two day (48 hour) heating loads (or heat extraction) in kWh
        self.two_day_hourly_peak_hl_loads = [[0]]
        self.process_two_day_loads()

        # Now we need to perform 48-hour simulations to determine the
        # monthly peak load hours
        # Stores two day (48 hour) fluid temperatures for cooling with nominal
        # load
        self.two_day_fluid_temps_cl_nm = [[0]]
        # Stores two day (48 hour) fluid temperatures for cooling with peak load
        self.two_day_fluid_temps_cl_pk = [[0]]
        # Stores two day (48 hour) fluid temperatures for heating with nominal
        # load
        self.two_day_fluid_temps_hl_nm = [[0]]
        # Stores two day (48 hour) fluid temperatures for heating with peak load
        self.two_day_fluid_temps_hl_pk = [[0]]

        # duration of monthly peak clg load in hours
        self.monthly_peak_cl_duration = [0] * num_unique_months
        # duration of monthly peak htg load in hours
        self.monthly_peak_hl_duration = [0] * num_unique_months
        self.find_peak_durations()

        # This block of data holds the sequence of loads. This is an
        # intermediate form, where the load values hold the actual loads,
        # not the de-convoluted loads
        self.load = np.array(0)  # holds the load during the period
        self.hour = np.array(0)  # holds the last hour of a period
        self.step_func_load = np.array(0)  # holds the load in terms of step functions
        self.process_month_loads()

    def as_dict(self) -> dict:
        output = dict()
        output['type'] = str(self.__class__)
        output['results'] = self.create_dataframe_of_peak_analysis()
        return output

    @staticmethod
    def split_heat_and_cool(raw_loads):
        """
        Split the provided loads into heating and cooling.
        Heating is positive, cooling is negative.

        :param raw_loads: raw loads entered by the user, in Watts
        :return: Loads split into heating and cooling
        """
        hourly_extraction_loads = [x / 1000.0 if x >= 0.0 else 0.0 for x in raw_loads]
        hourly_rejection_loads = [abs(x) / 1000.0 if x < 0.0 else 0.0 for x in raw_loads]

        return hourly_rejection_loads, hourly_extraction_loads

    def split_loads_by_month(self) -> None:
        # Split the loads into peak, total and average loads for each month

        hours_in_day = 24
        # Store the index of the last months hours
        hours_in_previous_months = 0
        for i in range(1, len(self.days_in_month)):
            hours_in_month = hours_in_day * self.days_in_month[i]
            # Slice the hours in this current month
            month_rejection_loads = \
                self.hourly_rejection_loads[hours_in_previous_months: hours_in_previous_months + hours_in_month]
            month_extraction_loads = \
                self.hourly_extraction_loads[hours_in_previous_months: hours_in_previous_months + hours_in_month]

            # TODO: verify whether errors are possible here and raise exception if needed
            # assert (len(month_extraction_loads) == hours_in_month and len(month_rejection_loads) == hours_in_month)

            # Sum
            # monthly cooling loads (or heat rejection) in kWh
            self.monthly_cl[i] = sum(month_rejection_loads)
            # monthly heating loads (or heat extraction) in kWh
            self.monthly_hl[i] = sum(month_extraction_loads)

            # Peak
            # monthly peak cooling load (or heat rejection) in kW
            self.monthly_peak_cl[i] = max(month_rejection_loads)
            # monthly peak heating load (or heat extraction) in kW
            self.monthly_peak_hl[i] = max(month_extraction_loads)

            # Average
            # monthly average cooling load (or heat rejection) in kW
            self.monthly_avg_cl[i] = self.monthly_cl[i] / len(month_rejection_loads)
            # monthly average heating load (or heat extraction) in kW
            self.monthly_avg_hl[i] = self.monthly_hl[i] / len(month_extraction_loads)

            # Day of month the peak heating load occurs
            # day of the month on which peak clg load occurs (e.g. 1-31)
            self.monthly_peak_cl_day[i] = floor(month_rejection_loads.index(self.monthly_peak_cl[i]) / hours_in_day)
            # day of the month on which peak clg load occurs (e.g. 1-31)
            self.monthly_peak_hl_day[i] = floor(month_extraction_loads.index(self.monthly_peak_hl[i]) / hours_in_day)
            # print("Monthly Peak HL Hour",month_extraction_loads.index(
            # self.monthly_peak_hl[i]) / hours_in_day)
            # print("Monthly Peak HL Day: ",self.monthly_peak_hl_day[i])
            # print("")

            hours_in_previous_months += hours_in_month

    def process_two_day_loads(self) -> None:
        # The two day (48 hour) two day loads are selected by locating the day
        # the peak load of the month occurs on, and pulling a 48-hour load
        # profile -- the day before and the day of

        hours_in_day = 24
        hours_in_year = len(self.hourly_rejection_loads)

        # Add the last day of the year to the beginning of the loads to account
        # for the possibility that a peak load occurs on the first day of the
        # year

        hourly_rejection_loads = (self.hourly_rejection_loads[
                                  hours_in_year - hours_in_day: hours_in_year] + self.hourly_rejection_loads)
        hourly_extraction_loads = (self.hourly_extraction_loads[
                                   hours_in_year - hours_in_day: hours_in_year] + self.hourly_extraction_loads)

        # Keep track of how many hours are in
        # start at 24 since we added the last day of the year to the beginning
        hours_in_previous_months = hours_in_day
        # loop over all 12 months
        for i in range(1, len(self.days_in_month)):
            hours_in_month = hours_in_day * self.days_in_month[i]

            # day of the month on which peak clg load occurs (e.g. 1-31)
            monthly_peak_cl_day = self.monthly_peak_cl_day[i]
            # day of the month on which peak clg load occurs (e.g. 1-31)
            monthly_peak_hl_day = self.monthly_peak_hl_day[i]
            # Get the starting hour of the day before the peak cooling load day
            monthly_peak_cl_hour_start = (hours_in_previous_months + (monthly_peak_cl_day - 1) * hours_in_day)
            # Get the starting hour of the day before the peak heating load day
            monthly_peak_hl_hour_start = (hours_in_previous_months + (monthly_peak_hl_day - 1) * hours_in_day)

            # monthly cooling loads (or heat rejection) in kWh
            two_day_hourly_peak_cl_load = \
                hourly_rejection_loads[monthly_peak_cl_hour_start: monthly_peak_cl_hour_start + 2 * hours_in_day]
            # monthly heating loads (or heat extraction) in kWh
            two_day_hourly_peak_hl_load = \
                hourly_extraction_loads[monthly_peak_hl_hour_start: monthly_peak_hl_hour_start + 2 * hours_in_day]

            # TODO: verify whether errors are possible here and raise exception if needed
            # assert (len(two_day_hourly_peak_hl_load) == 2 * hours_in_day
            #         and len(two_day_hourly_peak_cl_load) == 2 * hours_in_day)

            # Double check ourselves
            # TODO: verify whether errors are possible here and raise exception if needed
            # monthly_peak_cl_day_start = int((monthly_peak_cl_hour_start - hours_in_day) / hours_in_day)
            # monthly_peak_cl_hour_month = int(monthly_peak_cl_day_start - sum(self.days_in_month[0:i]))
            # assert monthly_peak_cl_hour_month == monthly_peak_cl_day - 1

            # TODO: verify whether errors are possible here and raise exception if needed
            # monthly_peak_hl_day_start = (monthly_peak_hl_hour_start - hours_in_day) / hours_in_day
            # monthly_peak_hl_hour_month = int(monthly_peak_hl_day_start - sum(self.days_in_month[0:i]))
            # assert monthly_peak_hl_hour_month == monthly_peak_hl_day - 1

            # monthly cooling loads (or heat rejection) in kWh
            self.two_day_hourly_peak_cl_loads.append(two_day_hourly_peak_cl_load)
            # monthly heating loads (or heat extraction) in kWh
            self.two_day_hourly_peak_hl_loads.append(two_day_hourly_peak_hl_load)

            hours_in_previous_months += hours_in_month

    @staticmethod
    def simulate_hourly(hour_time, q, g_sts, resist_bh, two_pi_k, ts):
        # An hourly simulation for the fluid temperature
        # Chapter 2 of Advances in Ground Source Heat Pumps

        q_dt = np.hstack((q[1:] - q[:-1]))

        delta_t_fluid = [0]
        for n in range(1, len(hour_time)):
            # Take the last i elements of the reversed time array
            _time = hour_time[n] - hour_time[0:n]
            # _time = time_values_reversed[n - i:n]
            g_values = g_sts(np.log((_time * 3600.0) / ts))
            # Tb = Tg + (q_dt * g)  (Equation 2.12)
            delta_tb_i = (q_dt[0:n] / two_pi_k).dot(g_values)
            # Delta mean heat pump entering fluid temperature
            tf_mean = delta_tb_i + q[n] * resist_bh
            delta_t_fluid.append(tf_mean)

        return delta_t_fluid

    def perform_current_month_simulation(
            self,
            two_day_hourly_peak_load,
            peak_load,
            avg_load,
            two_day_fluid_temps_pk,
            two_day_fluid_temps_nm,
    ):
        ts = self.radial_numerical.t_s
        two_pi_k = TWO_PI * self.bhe.soil.k
        resist_bh_effective = self.bhe.calc_effective_borehole_resistance()
        g_sts = self.radial_numerical.g_sts
        hours_in_day = 24
        hour_time = np.array(list(range(0, 2 * hours_in_day + 1)))
        # Two day peak cooling load scaled down by average (q_max - q_avg)
        q_peak = np.array([0.0] + [peak_load - avg_load] * (2 * hours_in_day))
        # Two day nominal cooling load (q_i - q_avg) / q_max * q_i
        q_nominal = np.array(
            [0.0]
            + [
                (two_day_hourly_peak_load[i] - avg_load)
                / peak_load
                * two_day_hourly_peak_load[i]
                for i in range(1, len(q_peak))
            ]
        )
        # Get peak fluid temperatures using peak load
        delta_t_fluid_peak = self.simulate_hourly(hour_time, q_peak, g_sts, resist_bh_effective, two_pi_k, ts)
        two_day_fluid_temps_pk.append(delta_t_fluid_peak)
        # Get nominal fluid temperatures using nominal load
        delta_t_fluid_nom = self.simulate_hourly(hour_time, q_nominal, g_sts, resist_bh_effective, two_pi_k, ts)
        two_day_fluid_temps_nm.append(delta_t_fluid_nom)

        delta_t_fluid_nom_max = max(delta_t_fluid_nom)

        if delta_t_fluid_nom_max > 0.0:
            f = interp1d(delta_t_fluid_peak, hour_time)
            peak_duration = f(delta_t_fluid_nom_max).tolist()
        else:
            peak_duration = 1.0e-6

        return peak_duration, q_peak, q_nominal

    def find_peak_durations(self) -> None:
        # Find the peak durations using hourly simulations for 2 days

        for i in range(1, len(self.days_in_month)):
            # Scale all the loads by the peak load
            # Perform an hourly simulation with the scaled loads
            # Perform an hourly simulation with a load of 1, or the peak loads
            # divided by the peak

            # two day cooling loads (or heat rejection) in kWh
            current_two_day_cl_load = [0.0] + self.two_day_hourly_peak_cl_loads[i]

            # This tolerance applies to the difference between the current
            # months peak load and the maximum of the two-day load. If the
            # absolute value of the difference between the current months
            # peak load and the current two-day peak load is within this
            # tolerance, then the maximum of the two-day load is equal to the
            # maximum of the current month. If the absolute difference is
            # greater than the tolerance, then the two-day peak load contains
            # a load greater than the current months peak load. The tolerance
            # could ONLY be exceeded when the first 24 hours is located in the
            # previous month.
            tol = 0.1

            # Ensure the peak load for the two-day load profile is the same or
            # greater than the monthly peak load. This check is done in case
            # the previous month contains a higher load than the current month.
            load_diff = self.monthly_peak_cl[i] - max(current_two_day_cl_load)
            # monthly peak cooling load (or heat rejection) in kW
            if abs(load_diff) < tol:
                current_month_peak_cl = self.monthly_peak_cl[i]
            else:
                current_month_peak_cl = max(current_two_day_cl_load)

            # monthly average cooling load (or heat rejection) in kW
            current_month_avg_cl = self.monthly_avg_cl[i]

            if current_month_peak_cl != 0.0:
                peak_duration, q_peak, q_nominal = self.perform_current_month_simulation(
                    current_two_day_cl_load,
                    current_month_peak_cl,
                    current_month_avg_cl,
                    self.two_day_fluid_temps_cl_pk,
                    self.two_day_fluid_temps_cl_nm)
            else:
                peak_duration = 1.0e-6

            # Set the monthly cooling load duration
            self.monthly_peak_cl_duration[i] = peak_duration

            # two day heating loads (or heat extraction) in kWh
            current_two_day_hl_load = [0.0] + self.two_day_hourly_peak_hl_loads[i]

            # Ensure the peak load for the two-day load profile is the same or
            # greater than the monthly peak load. This check is done in case
            # the previous month contains a higher load than the current month.
            load_diff = self.monthly_peak_hl[i] - max(current_two_day_hl_load)
            # monthly peak cooling load (or heat rejection) in kW
            if abs(load_diff) < tol:
                current_month_peak_hl = self.monthly_peak_hl[i]
            else:
                current_month_peak_hl = max(current_two_day_hl_load)

            # monthly average heating load (or heat extraction) in kW
            current_month_avg_hl = self.monthly_avg_hl[i]

            if current_month_peak_hl != 0.0:
                peak_duration, q_peak, q_nominal = self.perform_current_month_simulation(
                    current_two_day_hl_load,
                    current_month_peak_hl,
                    current_month_avg_hl,
                    self.two_day_fluid_temps_hl_pk,
                    self.two_day_fluid_temps_hl_nm,
                )
            else:
                peak_duration = 1.0e-6

            # Set the monthly cooling load duration
            self.monthly_peak_hl_duration[i] = peak_duration

    def create_dataframe_of_peak_analysis(self) -> str:
        # The fields are: sum, peak, avg, peak day, peak duration
        hybrid_time_step_fields = {
            "Total": {},
            "Peak": {},
            "Average": {},
            "Peak Day": {},
            "Peak Duration": {},
        }

        d: dict = {}
        # For all the months, create dictionary of fields
        for i in range(1, 13):
            m_n = number_to_month(i)
            d[m_n] = hybrid_time_step_fields

            # set total
            d[m_n]["Total"]["rejection"] = self.monthly_cl[i]
            d[m_n]["Total"]["extraction"] = self.monthly_hl[i]
            # set peak
            d[m_n]["Peak"]["rejection"] = self.monthly_peak_cl[i]
            d[m_n]["Peak"]["extraction"] = self.monthly_peak_hl[i]
            # set average
            d[m_n]["Average"]["rejection"] = self.monthly_avg_cl[i]
            d[m_n]["Average"]["extraction"] = self.monthly_avg_hl[i]
            # set peak day
            d[m_n]["Peak Day"]["rejection"] = self.monthly_peak_cl_day[i]
            d[m_n]["Peak Day"]["extraction"] = self.monthly_peak_hl_day[i]
            # set peak duration
            d[m_n]["Peak Duration"]["rejection"] = self.monthly_peak_cl_duration[i]
            d[m_n]["Peak Duration"]["extraction"] = self.monthly_peak_hl_duration[i]

        return dumps(d, indent=2)

    def process_month_loads(self):
        # Converts monthly load format to sequence of loads needed for
        # simulation
        # This routine is taking loads applied to the ground NOT to a heat pump.

        warn_msg_neg_timestep = "A negative time step has been generated in the hybrid loading scheme. " \
                                "This will reduce the accuracy of the simulation."

        # First, begin array with zero load before simulation starts.
        self.load = np.append(self.load, 0)
        last_zero_hour = first_month_hour(self.start_month, self.years) - 1
        self.hour = np.append(self.hour, last_zero_hour)
        if len(self.years) <= 1:
            # Second, replicate months. [if we want to add an option where all
            # monthly loads are explicitly given, this code will be in an if block]
            for i in range(self.start_month, self.end_month + 1):
                if i > 12:
                    mi = i % 12
                    if mi == 0:
                        mi = 12
                    self.monthly_cl.append(self.monthly_cl[mi])
                    self.monthly_hl.append(self.monthly_hl[mi])
                    self.monthly_peak_cl.append(self.monthly_peak_cl[mi])
                    self.monthly_peak_hl.append(self.monthly_peak_hl[mi])
                    self.monthly_peak_cl_duration.append(self.monthly_peak_cl_duration[mi])
                    self.monthly_peak_hl_duration.append(self.monthly_peak_hl_duration[mi])
                    self.monthly_peak_cl_day.append(self.monthly_peak_cl_day[mi])
                    self.monthly_peak_hl_day.append(self.monthly_peak_hl_day[mi])
        # Set the ipf (include peak flag)
        if len(self.years) <= 1:
            ipf = [False] * (self.end_month + 1)
            for i in range(self.start_month, self.end_month + 1):
                # set flag that determines if peak load will be included
                if i < self.start_month + self.peak_retain_start:
                    ipf[i] = True
                if i > self.end_month - self.peak_retain_end:
                    ipf[i] = True
        else:
            ipf = [True] * (self.end_month + 1)
        pass
        peak_last_avg_hour = 0.0
        for i in range(self.start_month, (self.end_month + 1)):
            # There may be a more sophisticated way to do this, but I will loop
            # through the lists month duration is the number of hours over which to
            # calculate the average value for the month
            if ipf[i]:
                if len(self.years) <= 1:
                    current_year = self.years[0]
                else:
                    current_year = self.years[(i - 1) // 12]
                month_duration = \
                    monthdays(i, current_year) * 24 - self.monthly_peak_cl_duration[i] - \
                    self.monthly_peak_hl_duration[i]
                # gives htg load pk energy in kWh
                month_peak_hl = self.monthly_peak_hl[i] * self.monthly_peak_hl_duration[i]
                # gives htg load pk energy in kWh
                month_peak_cl = self.monthly_peak_cl[i] * self.monthly_peak_cl_duration[i]
                month_load = self.monthly_cl[i] - self.monthly_hl[i] - month_peak_cl + month_peak_hl
                month_rate = month_load / month_duration
                peak_day_diff = self.monthly_peak_cl_day[i] - self.monthly_peak_hl_day[i]
                # Place the peaks roughly midway through the day they occur on.
                # (In JDS's opinion, this should be amply accurate for the
                # hybrid time step.)
                # Catch the first and last peak hours to make sure they aren't 0
                # Could only be 0 when the first month has no load.
                first_hour_heating_peak = \
                    first_month_hour(i, self.years) + \
                    (self.monthly_peak_hl_day[i]) * 24 + 12 - (self.monthly_peak_hl_duration[i] / 2)
                if first_hour_heating_peak < 0.0:
                    first_hour_heating_peak = 1.0e-6
                last_hour_heating_peak = first_hour_heating_peak + self.monthly_peak_hl_duration[i]
                if last_hour_heating_peak < 0.0:
                    last_hour_heating_peak = 1.0e-6
                first_hour_cooling_peak = \
                    first_month_hour(i, self.years) + (self.monthly_peak_cl_day[i]) * 24 + 12 - \
                    self.monthly_peak_cl_duration[i] / 2
                if first_hour_cooling_peak < 0.0:
                    first_hour_cooling_peak = 1.0e-06
                last_hour_cooling_peak = first_hour_cooling_peak + self.monthly_peak_cl_duration[i]
                if last_hour_cooling_peak < 0.0:
                    last_hour_cooling_peak = 1.0e-06
            else:  # peak load not used this month

                month_duration = monthdays(i, current_year) * 24

                month_load = self.monthly_cl[i] - self.monthly_hl[i]
                month_rate = month_load / month_duration
                peak_day_diff = 0

            last_avg_hour = 0.0
            if peak_day_diff < 0:
                # monthly peak heating day occurs after peak cooling day
                # monthly average conditions before cooling peak
                if self.monthly_peak_cl[i] > 0 and ipf[i]:
                    # last_avg_hour = first_hour_cooling_peak - 1 JDS corrected 20200604
                    last_avg_hour = first_hour_cooling_peak
                    self.load = np.append(self.load, month_rate)
                    self.hour = np.append(self.hour, last_avg_hour)
                    # cooling peak
                    # self.load = np.append(self.load, -self.monthly_peak_cl[i]) JDS corrected 20200604
                    self.load = np.append(self.load, self.monthly_peak_cl[i])
                    self.hour = np.append(self.hour, last_hour_cooling_peak)

                    if last_avg_hour - peak_last_avg_hour < 0.0:
                        warnings.warn(warn_msg_neg_timestep)
                    peak_last_avg_hour = last_avg_hour
                # monthly average conditions between cooling peak and heating peak
                if self.monthly_peak_hl[i] > 0 and ipf[i]:
                    # last_avg_hour = first_hour_heating_peak - 1 JDS corrected 20200604
                    last_avg_hour = first_hour_heating_peak
                    self.load = np.append(self.load, month_rate)
                    self.hour = np.append(self.hour, last_avg_hour)
                    # heating peak
                    # self.load = np.append(self.load, self.monthly_peak_hl[i]) JDS corrected 20200604
                    self.load = np.append(self.load, -self.monthly_peak_hl[i])
                    self.hour = np.append(self.hour, last_hour_heating_peak)

                    if last_avg_hour - peak_last_avg_hour < 0.0:
                        warnings.warn(warn_msg_neg_timestep)
                    peak_last_avg_hour = last_avg_hour
                # rest of month
                last_avg_hour = last_month_hour(i, self.years)
                self.load = np.append(self.load, month_rate)
                self.hour = np.append(self.hour, last_avg_hour)

                if last_avg_hour - peak_last_avg_hour < 0.0:
                    warnings.warn(warn_msg_neg_timestep)
                peak_last_avg_hour = last_avg_hour

            elif peak_day_diff > 0:
                # monthly peak heating day occurs before peak cooling day
                # monthly average conditions before cooling peak
                if self.monthly_peak_hl[i] > 0 and ipf[i]:
                    last_avg_hour = first_hour_heating_peak
                    self.load = np.append(self.load, month_rate)
                    self.hour = np.append(self.hour, last_avg_hour)
                    # heating peak
                    self.load = np.append(self.load, -self.monthly_peak_hl[i])
                    self.hour = np.append(self.hour, last_hour_heating_peak)

                    if last_avg_hour - peak_last_avg_hour < 0.0:
                        warnings.warn(warn_msg_neg_timestep)
                    peak_last_avg_hour = last_avg_hour
                # monthly average conditions between heating peak and cooling peak
                if self.monthly_peak_cl[i] > 0 and ipf[i]:
                    last_avg_hour = first_hour_cooling_peak
                    self.load = np.append(self.load, month_rate)
                    self.hour = np.append(self.hour, last_avg_hour)
                    # cooling peak
                    self.load = np.append(self.load, self.monthly_peak_cl[i])
                    self.hour = np.append(self.hour, last_hour_cooling_peak)

                    if last_avg_hour - peak_last_avg_hour < 0.0:
                        warnings.warn(warn_msg_neg_timestep)
                    peak_last_avg_hour = last_avg_hour
                # rest of month
                last_avg_hour = last_month_hour(i, self.years)
                self.load = np.append(self.load, month_rate)
                self.hour = np.append(self.hour, last_avg_hour)

                if last_avg_hour - peak_last_avg_hour < 0.0:
                    warnings.warn(warn_msg_neg_timestep)
                peak_last_avg_hour = last_avg_hour
            else:
                # monthly peak heating day and cooling day are the same
                # Cooling Load placed before noon, and the heating load is placed after noon
                # Currently the exact times of the heating and cooling peaks are not stored. If further work is done
                # this default can be made to be more accurate.
                if ipf[i]:
                    # monthly average conditions before cooling peak
                    if self.monthly_peak_cl[i] > 0 and ipf[i]:
                        # last_avg_hour = first_hour_cooling_peak - 1 JDS corrected 20200604
                        last_avg_hour = first_hour_cooling_peak - self.monthly_peak_cl_duration[i] / 2
                        self.load = np.append(self.load, month_rate)
                        self.hour = np.append(self.hour, last_avg_hour)
                        # cooling peak
                        # self.load = np.append(self.load, -self.monthly_peak_cl[i]) JDS corrected 20200604
                        self.load = np.append(self.load, self.monthly_peak_cl[i])
                        self.hour = np.append(
                            self.hour,
                            last_hour_cooling_peak
                            - self.monthly_peak_cl_duration[i] / 2,
                        )

                        if last_avg_hour - peak_last_avg_hour < 0.0:
                            warnings.warn(warn_msg_neg_timestep)
                        peak_last_avg_hour = last_avg_hour
                    # monthly average conditions between cooling peak and heating peak
                    if self.monthly_peak_hl[i] > 0 and ipf[i]:

                        # heating peak
                        # self.load = np.append(self.load, self.monthly_peak_hl[i]) JDS corrected 20200604

                        self.load = np.append(self.load, -self.monthly_peak_hl[i])
                        self.hour = np.append(
                            self.hour,
                            last_hour_heating_peak
                            + self.monthly_peak_hl_duration[i] / 2,
                        )

                        if last_avg_hour - peak_last_avg_hour < 0.0:
                            warnings.warn(warn_msg_neg_timestep)
                        peak_last_avg_hour = last_avg_hour
                    # rest of month
                    last_avg_hour = last_month_hour(i, self.years)
                    self.load = np.append(self.load, month_rate)
                    self.hour = np.append(self.hour, last_avg_hour)

                    if last_avg_hour - peak_last_avg_hour < 0.0:
                        warnings.warn(warn_msg_neg_timestep)
                    peak_last_avg_hour = last_avg_hour

                else:
                    last_avg_hour = last_month_hour(i, self.years)
                    self.load = np.append(self.load, month_rate)
                    self.hour = np.append(self.hour, last_avg_hour)

                if last_avg_hour - peak_last_avg_hour < 0.0:
                    warnings.warn(warn_msg_neg_timestep)
                peak_last_avg_hour = last_avg_hour

        #       Now fill array containing step function loads
        #       Note they are paired with the ending hour, so the ith load will start with the (i-1)th time

        n = self.hour.size
        # Note at this point the load and hour np arrays contain zeroes in indices zero and one, then continue from
        # there.
        for i in range(1, n):
            step_load = self.load[i] - self.load[i - 1]
            self.step_func_load = np.append(self.step_func_load, step_load)
        pass


def number_to_month(x):
    return ["NULL", "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"][x]


def monthdays(month, year):
    leap_year = year % 4 == 0
    if month > 12:
        md = month % 12
    else:
        md = month
    if leap_year:
        num_days = [31, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    else:
        num_days = [31, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return num_days[md]


def first_month_hour(month, years):
    fmh = 1
    if month > 1:
        for i in range(1, month):
            if len(years) > 1:
                current_year = years[(month - 1) // 12]
            else:
                current_year = years[0]
            mi = i % 12
            fmh = fmh + 24 * monthdays(mi, current_year)
    return fmh


def last_month_hour(month, years):
    lmh = 0
    for i in range(1, month + 1):
        if len(years) > 1:
            current_year = years[(month - 1) // 12]
        else:
            current_year = years[0]
        lmh = lmh + monthdays(i, current_year) * 24
    if month == 1:
        lmh = 31 * 24
    return lmh
