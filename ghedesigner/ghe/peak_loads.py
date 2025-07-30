from calendar import monthrange
from json import dumps
from math import floor

import numpy as np

from ghedesigner.constants import HRS_IN_DAY, SEC_IN_HR, TWO_PI
from ghedesigner.ghe.boreholes.single_u_borehole import SingleUTube


class HybridLoads2:
    def __init__(
        self,
        loads: list,
        bhe: SingleUTube,
        radial_numerical: SingleUTube,
        # sim_params: SimulationParameters,
        hourly_temps: list,
        years=None,
        start_month=None,
        end_month=None,
    ) -> None:
        if years is None:
            years = [2025]
        self.years = years

        self.loads = loads
        #print(loads[54:70])
        self.hourly_ExFT_temps = hourly_temps

        self.start_month = start_month
        self.end_month = end_month

        # Simulation start and end month

        if len(self.years) <= 1:
            self.peak_retain_start = 12  # use peak loads for first 12 months
            self.peak_retain_end = 12  # use peak loads for last 12 months
        else:
            self.peak_retain_start = len(self.years) * 6  # use peak loads for first n*6 months
            self.peak_retain_end = (
                len(self.years) * 6
            )  # use peak loads for last n*6 months #TODO do we need this still?

        # Get the number of days in each month for a given year (make 0 NULL)
        self.days_in_month = [0]  # first entry reserved for annual total or peak, see line 72
        for year in years:
            self.days_in_month.extend([monthrange(year, i)[1] for i in range(1, 13)])

        # Store the borehole heat exchanger
        self.bhe = bhe

        # Store the radial numerical g-function value
        # Note: this is intended to be a scipy.interp1d object
        self.borehole = radial_numerical
        # self.sim_params = sim_params
        self.years = years

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

        num_unique_months = len(self.years) * 12 + 1

        # monthly cooling loads (or heat rejection) in kWh
        self.monthly_cl = [0.0] * num_unique_months
        # monthly heating loads (or heat extraction) in kWh
        self.monthly_hl = [0.0] * num_unique_months
        # monthly peak cooling load (or heat rejection) in kW
        self.monthly_peak_cl = [0.0] * num_unique_months
        # monthly peak heating load (or heat extraction) in kW
        self.monthly_peak_hl = [0.0] * num_unique_months
        # monthly average cooling load (or heat rejection) in kW
        self.monthly_avg_cl = [0.0] * num_unique_months
        # monthly average heating load (or heat extraction) in kW
        self.monthly_avg_hl = [0.0] * num_unique_months
        # day of the month on which peak clg load occurs (e.g. 1-31)
        self.monthly_peak_cl_day = [0] * num_unique_months
        # day of the month on which peak htg load occurs (e.g. 1-31)
        self.monthly_peak_hl_day = [0] * num_unique_months
        # monthly minimum GHE exiting fluid temperature (EFT for heatpump) in C
        self.monthly_min_ExFT = [0.0] * num_unique_months
        # monthly maximum GHE exiting fluid temperature (EFT for heatpump) in C
        self.monthly_max_ExFT = [0.0] * num_unique_months
        # hour of the month on which min ExFT occurs for cooling
        self.monthly_min_GHE_ExFT_hour = [0.0] * num_unique_months
        # hour of the month on which peak ExFT occurs for heating
        self.monthly_max_GHE_ExFT_hour = [0.0] * num_unique_months

        self.cumulative_hours = [0]
        for i in range(1, len(self.days_in_month)):
            self.cumulative_hours.append(self.cumulative_hours[-1] + self.days_in_month[i] * HRS_IN_DAY)

        # convert bldg loads to ground loads
        # self.cop_c = 2  # TODO I think we can get rid of this now that we're integrated with GHEdesigner?
        # self.cop_h = 2
        # self.ground_loads = self.bldg_to_ground_load()
        # split heating and cooling
        self.split_heat_and_cool()
        # Process the loads and by month
        self.split_loads_by_month()
        # Process the ExFT by month
        self.split_ExFT_by_month()
        # return the highest 4 ExFTs for cooling
        self.max_4_ExFT = self.get_max_4_ExFT_and_time()
        # return the lowest 4ExFTs for heating
        self.min_4_ExFT = self.get_min_4_ExFT_and_time()
        # find peak durations
        self.durations = self.find_peak_durations()

        self.monthly_peak_cl_duration = []
        self.monthly_peak_hl_duration = []

        self.hourly_extraction_loads = []
        self.hourly_rejection_loads = []

        self.hrly_extraction_loads_norm = []
        self.hrly_rejection_loads_norm = []

        # Step 1----------------------------------------

    def bldg_to_ground_load(self) -> list:
        """
        Acts as a constant COP heatpump
        :return: ground loads in Watts
        """
        ground_loads = []

        for i in range(len(self.loads)):
            if self.loads[i] >= 0:
                ground_loads.append((self.cop_h - 1) / self.cop_h * self.loads[i])
            else:
                ground_loads.append((1 + self.cop_c) / self.cop_c * self.loads[i])

        print("bldg loads have been converted to ground loads.")
        return ground_loads

    # Step 2  -----------------------------

    def split_heat_and_cool(self):
        """
        Splits the hourly normalized ground loads into heating and cooling
        Heating is positive, cooling is negative.
        :return: hourly normalized ground loads split into heating and cooling in kilowatts
        :return: hourly ExFTs split into heating and cooling lists in Celsius
        """

        self.hourly_extraction_loads = [x / 1000.0 if x >= 0.0 else 0.0 for x in self.loads]
        self.hourly_rejection_loads = [abs(x) / 1000.0 if x < 0.0 else 0.0 for x in self.loads]

        self.hrly_extraction_loads_norm = self.normalize_loads(self.hourly_extraction_loads)
        self.hrly_rejection_loads_norm = self.normalize_loads(self.hourly_rejection_loads)

        print("split_heat_and_cool has run")
        #print(f" hrly rejection sample  = {self.hrly_rejection_loads_norm[54:70]} kW")
        #print(f" hrly exaction sample = {self.hrly_extraction_loads_norm[54:70]} kW")

        # TODO returning vs self. variables?

    @staticmethod
    def normalize_loads(load) -> list:
        """
        Normalize the ground loads to a peak of 4000 W since we do not know the size or
        configuration of the borehole field
        :return: hourly ground loads normalized to 4000 W peak [Watts]
        """
        max_ground_loads = max(load)
        normalized_loads = [40 * 100 / max_ground_loads * load[i] for i in range(len(load))]

        print("Normalize loads has run.")

        return normalized_loads

    # Part 3 ----------------

    def split_loads_by_month(self) -> None:
        """Split the loads into peak, total, and average loads for each month

        """

        # Store the index of the last month's hours
        hours_in_previous_months: int = 0  # type is integer and set to a value of 0 to start
        i: int  # set i type to integer
        for i in range(1, len(self.days_in_month)):  # cycle through i for values 1 to number of months (ex: 12)
            hours_in_month = HRS_IN_DAY * self.days_in_month[i]  # e.g. 24 * 31, to give hrs in month
            # Slice the hours in this current month
            current_month_reject_norm_loads = self.hrly_rejection_loads_norm[
                hours_in_previous_months : hours_in_previous_months + hours_in_month
            ]  # first slice will be from [0:0+24*31]
            current_month_extract_norm_loads = self.hrly_extraction_loads_norm[
                hours_in_previous_months : hours_in_previous_months + hours_in_month
            ]

            # TODO: verify whether errors are possible here and raise exception if needed
            # assert (len(month_extraction_loads) == hours_in_month and len(month_rejection_loads) == hours_in_month)

            # Sum
            # monthly cooling loads (or heat rejection) in kWh
            self.monthly_cl[i] = sum(current_month_reject_norm_loads)
            # monthly heating loads (or heat extraction) in kWh
            self.monthly_hl[i] = sum(current_month_extract_norm_loads)

            # Peak
            # monthly peak cooling load (or heat rejection) in kW
            self.monthly_peak_cl[i] = max(current_month_reject_norm_loads)
            # monthly peak heating load (or heat extraction) in kW
            self.monthly_peak_hl[i] = max(current_month_extract_norm_loads)

            # Average
            # monthly average cooling load (or heat rejection) in kW
            self.monthly_avg_cl[i] = self.monthly_cl[i] / len(current_month_reject_norm_loads)
            # monthly average heating load (or heat extraction) in kW
            self.monthly_avg_hl[i] = self.monthly_hl[i] / len(current_month_extract_norm_loads)

            # Day of month the peak heating load occurs
            # day of the month on which peak clg load occurs (e.g. 1-31)
            self.monthly_peak_cl_day[i] = floor(
                current_month_reject_norm_loads.index(self.monthly_peak_cl[i]) / HRS_IN_DAY
            )
            # day of the month on which peak clg load occurs (e.g. 1-31)
            self.monthly_peak_hl_day[i] = floor(
                current_month_extract_norm_loads.index(self.monthly_peak_hl[i]) / HRS_IN_DAY
            )
            # print("Monthly Peak HL Hour",month_extraction_loads.index(
            # self.monthly_peak_hl[i]) / HRS_IN_DAY)
            # print("Monthly Peak HL Day: ",self.monthly_peak_hl_day[i])
            # print("")

            hours_in_previous_months += hours_in_month
        #check the first 3 months
        # print(f"Total monthly cooling energy [kWh]:{self.monthly_cl[1:4]} ")
        # print(f"Total monthly heating energy [kWh]:{self.monthly_hl[1:4]} ")
        # print(f"average cooling load per month [kW]= {self.monthly_avg_cl[1:4]} ")
        # print(f"peak cooling load per month[kW]= {self.monthly_peak_cl[1:4]} ")
        # print(f"average heating load per month [kW]= {self.monthly_avg_hl[1:4]} ")
        # print(f"peak heating load per month[kW]= {self.monthly_peak_hl[1:4]} ")
        print("split_loads_by_month has run")

    def split_ExFT_by_month(self) -> None:
        """Slice the hourly ExFT of the GHE into months"""
        # TODO find a way to slice month hours outside of split_ExFT_by_month and split_load_by_month functions

        # Store the index of the last month's hours
        hours_in_previous_months: int = 0
        i: int
        for i in range(1, len(self.days_in_month)):
            hours_in_month = HRS_IN_DAY * self.days_in_month[i]
            # Slice the hours in this current month

            current_month_ExFTs = self.hourly_ExFT_temps[
                hours_in_previous_months : hours_in_previous_months + hours_in_month
            ]

            # Peak
            # monthly peak (min) exiting fluid temperature from the GHE [degrees C]
            self.monthly_min_ExFT[i] = float(min(current_month_ExFTs))
            # monthly peak exiting fluid temperature from the GHE [degrees C]
            self.monthly_max_ExFT[i] = float(max(current_month_ExFTs))

            # Hour of the month the min ExFT for the GHE occurs (EFT for heatpump). critical for heating
            self.monthly_min_GHE_ExFT_hour[i] = int(np.where(current_month_ExFTs == self.monthly_min_ExFT[i])[0][0])
            # Hour of the month the max ExFT for the GHE occurs (EFT for heatpump). critical for cooling
            self.monthly_max_GHE_ExFT_hour[i] = int(np.where(current_month_ExFTs == self.monthly_max_ExFT[i])[0][0])

            hours_in_previous_months += hours_in_month

        #print("Monthly min ExFTs (C):", self.monthly_min_ExFT)
        #print("Monthly max ExFTs (C):", self.monthly_max_ExFT)
        #print(f"min ExFTs occurs at hr {self.monthly_min_GHE_ExFT_hour}")
        #print(f"max ExFTs occurs at hr {self.monthly_max_GHE_ExFT_hour}")

        print("split_ExFT_by_month has run")

    def get_min_4_ExFT_and_time(self):
        """
        Returns the 4 months in which peak ExFT occurs for cooling, the peak ExFT, and the hour of the month.
        Stores the result as a 2D NumPy array with columns: [ExFT value, hour of month, month, hour of year].
        """
        # Step 1: Create cumulative hour lookup for each month

        top_4_min = sorted(
            [
                (
                    min,  # ExFT value
                    int(self.monthly_min_GHE_ExFT_hour[i]),  # hour within month
                    int(i),  # month index
                    int(self.cumulative_hours[i] + self.monthly_min_GHE_ExFT_hour[i]),  # absolute hour of year
                )
                for i, min in enumerate(self.monthly_min_ExFT)
                if i != 0
            ],
            key=lambda x: x[0],  # sort by ExFT value
            reverse=True,
        )[:4]

        min_4_ExFT = np.array(top_4_min)
        print(f"min ExFT, hour of month, month, hour of year ")
        print(f"{min_4_ExFT}")
        return min_4_ExFT

    def get_max_4_ExFT_and_time(self):
        """
        Returns the 4 months in which peak ExFT occurs for cooling, the peak ExFT, and the hour of the month.
        Stores the result as a 2D NumPy array with columns: [hour, month, ExFT value].
        """

        # TODO switch to Kelvin to eliminate error that peak might be 0 deg C
        top_4 = sorted(
            [
                (
                    max,  # ExFT
                    int(self.monthly_max_GHE_ExFT_hour[i]),  # hour within month
                    int(i),  # month
                    int(self.cumulative_hours[i] + self.monthly_min_GHE_ExFT_hour[i]),  # absolute hour of year
                )
                for i, max in enumerate(self.monthly_max_ExFT)
                if i != 0 and max != 0
            ],
            key=lambda x: x[0],  # sort by ExFT value
            reverse=False,
        )[:4]

        max_4_ExFT = np.array(top_4)
        print(f"max ExFT, hour of month, month, hour of year ")
        print(f"{max_4_ExFT}")
        return max_4_ExFT

    # part 4 ----------------------
    def find_peak_durations(self):
        """
        find the peak load durations by using the GHE ExFT and increment peak load duration
        1 hr at a time until ExFT matches hourly simulation
        :return: the 4 peak load durations
        """

        durations = []
        for row in self.max_4_ExFT:  # iterates through the 4 identified peaks
            target_ExFT = row[0]  # our target ExFT for the iteration
            pk_hour_of_month = int(row[1])
            month_index = int(row[2])
            # pk_hour_of_year = int(row[3])
            peak_duration = 1
            hours_in_month = self.days_in_month[month_index] * HRS_IN_DAY

            hybrid_load = self.update_hybrid_loads(pk_hour_of_month, month_index, peak_duration)

            # TODO update average with accounting for peak and increase duration, make own worker funct
            print(
                f"peak load for month {month_index} occurs at {pk_hour_of_month} hour and "
                f"has a value of {self.monthly_peak_cl[month_index]}"
            )

            # here we call on self.perform monthly ExFT simulation to increase the peak_duration
            # by 1 hour each step until ExFT >= target ExFT
            print(f"\nMonth {month_index}: starting search for peak duration.")

            max_hours = hours_in_month
            while peak_duration < max_hours:
                new_max_ExFT = self.perform_monthly_ExFT_simulation(pk_hour_of_month, hybrid_load)

                print(f"Trying duration {peak_duration} hr: max ExFT = {new_max_ExFT:.2f}, target = {target_ExFT:.2f}")

                if new_max_ExFT > target_ExFT:
                    break
                peak_duration += 1

            durations.append((month_index, peak_duration))  # (month, duration)

        print("\nFinal peak durations (month, hours):", durations)
        return durations

    def update_hybrid_loads(self, pk_hour_of_month, month_index, peak_duration):
        hours_in_month = self.days_in_month[month_index] * HRS_IN_DAY

        # reset the average for the month to have energy balance
        ave_old = self.monthly_avg_cl[month_index]
        ave_new = (
            self.monthly_avg_cl[month_index] * (hours_in_month - peak_duration)
            - self.monthly_peak_cl[month_index] * peak_duration
        ) / hours_in_month

        print(f"ave_old = {ave_old}")
        print(f"ave_new = {ave_new}")

        # creates a list that contains the monthly average load for every hour of the
        # month (ex: 31*24 identical entries for Jan)
        hybrid_load = [ave_new] * hours_in_month

        # updates the ave_load hourly list for the month by inserting the peak cooling
        # load at the hour of the month it occurs
        hybrid_load[pk_hour_of_month] = self.monthly_peak_cl[month_index]
        # print(f"avg_load_hrly_w_pks = {avg_load_hrly_w_pks}")

        print(f"sample of avg_load_hrly_w_pks array {hybrid_load[pk_hour_of_month - 2 : pk_hour_of_month + 3]}")
        return hybrid_load

    def perform_monthly_ExFT_simulation(self, pk_hour_of_month, hybrid_load):
        """
        This performs a month long ExFT simulation given the new hybrid loads
        """

        ts = self.borehole.t_s
        two_pi_k = TWO_PI * self.bhe.soil.k
        resist_bh_effective = self.bhe.calc_effective_borehole_resistance()
        print(f"resist_bh_effective = {resist_bh_effective}")
        g = self.borehole.g_sts
        q = hybrid_load

        hrs_in_month = list(range(len(q)))
        delta_t_fluid = self.simulate_hourly(
            hrs_in_month, q, g, resist_bh_effective, two_pi_k, ts
        )  # runs hourly fluid temperature simulation

        new_ExFT_hrly_w_pk = delta_t_fluid[pk_hour_of_month]

        return new_ExFT_hrly_w_pk

    @staticmethod
    def simulate_hourly(hour_time, q, g, resist_bh, two_pi_k, ts):
        # An hourly simulation for the fluid temperature
        # Chapter 2 of Advances in Ground Source Heat Pumps
        q_arr = np.array(q)
        q_dt = np.hstack(q_arr[1:] - q_arr[:-1])
        hour_time_arr = np.array(hour_time)
        delta_t_fluid = [0]
        for n in range(1, len(hour_time)):
            # Take the last i elements of the reversed time array
            _time = hour_time_arr[n] - hour_time_arr[0:n]
            # _time = time_values_reversed[n - i:n]
            g_values = g(np.log((_time * SEC_IN_HR) / ts))
            # Tb = Tg + (q_dt * g)  (Equation 2.12)
            delta_tb_i = (q_dt[0:n] / two_pi_k).dot(g_values)
            # Delta mean heat pump entering fluid temperature
            tf_mean = delta_tb_i + q_arr[n] * resist_bh
            delta_t_fluid.append(tf_mean)

        return delta_t_fluid

    def create_dataframe_of_peak_analysis(self) -> str:
        # The fields are: sum, peak, avg, peak day, peak duration
        hybrid_time_step_fields: dict = {
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


def number_to_month(x):
    return [
        "NULL",
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ][x]
