from calendar import monthrange

from ghedesigner.utilities import simulate_hourly


class HybridLoadsCalc:
    def __init__(
        self,
        years: list[float] | None = None,
        start_month=None,
        end_month=None,
        cop_h: float = 3.8,
        cop_c: float = 4.5,
    ) -> None:
        # (used in step 1) Initialize COP values
        self.cop_h = cop_h  # Heating COP
        self.cop_c = cop_c  # Cooling COP

        # (used in step 1.5) initialize months under study. Currently we only do one year, but in the future, the goal is to create hybrid loads for 10+ years
        if years is None:
            years = [2025]
        self.years = years

        # Get the number of days in each month for a given year (make 0 NULL)
        self.days_in_month = [0]
        for year in years:
            self.days_in_month.extend([monthrange(year, i)[1] for i in range(1, 13)])

        # compute the starting hour (1-based) for each month in the simulation
        self.month_start_hours = [0]
        start_hour = 0
        for days in self.days_in_month:
            self.month_start_hours.append(start_hour + 1)
            start_hour += days * 24

        # compute the end hour (1-based) for each month
        self.month_end_hours = [0]  # list of end hours for each month
        end_hour = self.days_in_month[0] * 24
        for days in self.days_in_month:
            self.month_end_hours.append(end_hour + 1)
            end_hour += days * 24

    def step_1_bldg_to_ground_load(self, building_loads: list[float]) -> list:
        """
        #run heatpump with constant COPs to convert building loads to ground loads.
        #these are hourly loads
        #
        #inputs: building_loads, COPc, COPh
        #output: ground_loads
        Acts as a constant COP heatpump
        :return: ground loads in Watts
        """
        ground_loads = []

        for i in range(len(building_loads)):
            if building_loads[i] >= 0:
                ground_loads.append((self.cop_h - 1) / self.cop_h * building_loads[i])
            else:
                ground_loads.append(-(1 + 1 / self.cop_c) * abs(building_loads[i]))

        return ground_loads

    @staticmethod
    def step_2_normalize_loads(load: list[float]) -> list:
        """
        # since borehole config and size is unknown,
        # Assume a 100m deep single borehole and 40W/m peak
        #  load and normalize to this peak (4000W max)
        #
        # inputs: ground_loads
        # outputs: normalized_ground_loads

        Normalize the ground loads to a peak of 4000 W since we do not know the size or
        configuration of the borehole field
        :return: hourly ground loads normalized to 4000 W peak [Watts]
        """
        # Handle case where all loads are zero
        if not load or all(x == 0.0 for x in load):
            print("Warning: All loads are zero, returning zeros")
            return [0.0] * len(load)

        abs_ground_loads = [abs(x) for x in load]
        max_ground_loads = max(abs_ground_loads)
        scaling_factor = 4000.0 / max_ground_loads

        # normalized to 40W/m for a 100 m borehole
        normalized_loads = [scaling_factor * load_val for load_val in load]

        print("Normalize loads has run.")

        return normalized_loads

    def step_3_calc_monthly_load_metrics(self, normalized_ground_loads: list[float]) -> None:
        """Calculates average, total, peak extraction
         and peak rejection ground loads for each month of the year.
        Inputs:
            normalized_ground_loads: Ground loads in Watts
        Outputs:
            self.monthly_total_ground_load: Monthly total ground energy in Wh
            self.monthly_peak_extraction: Monthly peak heating load (heat extraction) in W
            self.monthly_peak_rejection: Monthly peak cooling load (heat rejection) in W
            self.monthly_ave_ground_load: Monthly average ground load in W
        """

        num_months = len(self.years) * 12 + 1

        # Initialize all monthly arrays at once
        self.monthly_total_ground_load = [0.0] * num_months
        self.monthly_peak_rejection = [0.0] * num_months
        self.monthly_peak_extraction = [0.0] * num_months
        self.monthly_ave_ground_load = [0.0] * num_months

        start_hour = 0

        for month_idx in range(1, len(self.days_in_month)):
            hours_in_month = 24 * self.days_in_month[month_idx]
            end_hour = start_hour + hours_in_month

            # Get current month's load data
            month_loads = normalized_ground_loads[start_hour:end_hour]

            if month_loads:  # Only process if we have data
                # Calculate monthly metrics
                self.monthly_total_ground_load[month_idx] = sum(month_loads)
                self.monthly_peak_rejection[month_idx] = min(month_loads)
                self.monthly_peak_extraction[month_idx] = max(month_loads)
                self.monthly_ave_ground_load[month_idx] = sum(month_loads) / hours_in_month

            start_hour = end_hour

        print("step_3_calc_monthly_load_metrics")

    def step_4_perform_hrly_ExFT_simulation(self, load_profile):
        """
        # take the normalized ground loads and use
        # a gfunction and the other ground heat exchanger parameters to get the hourly
        # exiting fluid temperature from the ghe (same as entering to HP)
        #
        # inputs: normalized ground loads, ground heat exchanger parameters
        # output: hourly exiting fluid temperature (ExFT) from the GHE
        This performs an ExFT simulation given a loads profile
        """
        ts = self.g_funct.t_s
        two_pi_k = TWO_PI * self.ghe.bhe.soil.k
        # resist_bh_effective = 0.13
        resist_bh_effective = self.ghe.bhe.calc_effective_borehole_resistance()
        print(f"resist_bh_effective = {resist_bh_effective}")

        # DEBUG: Check if g_sts is callable
        g = self.g_funct.g_sts
        print(f"DEBUG: g type = {type(g)}")
        print(f"DEBUG: g callable = {callable(g)}")
        print(f"DEBUG: g value = {g}")

        if g is None:
            raise ValueError("g_sts function is None - check setup")
        if not callable(g):
            raise ValueError(f"g_sts is not callable, type: {type(g)}")

        q = load_profile
        hour_indices = list(range(len(q)))
        delta_t_fluid = simulate_hourly(hour_indices, q, g, resist_bh_effective, two_pi_k, ts)
        return delta_t_fluid

    def step_5_get_monthly_ExFT_maxs_mins_and_times(self, hourly_ExFThe: list[float]) -> None:
        """Split the yearly ExFT of the GHE into months.
        Identify the min and max ExFT for each month and hour of the year on which it occurs.

        Args:
            hourly_ExFThe: Hourly exiting fluid temperatures from GHE

        Outputs:
            self.monthly_min_ExFT: Monthly minimum ExFT [degrees C]
            self.monthly_min_ExFT_time: Hour of year when min ExFT occurs
            self.monthly_max_ExFT: Monthly maximum ExFT [degrees C]
            self.monthly_max_ExFT_time: Hour of year when max ExFT occurs
        """

        num_months = len(self.years) * 12 + 1

        # Initialize monthly arrays
        self.monthly_min_ExFT = [0.0] * num_months
        self.monthly_min_ExFT_time = [0] * num_months  # Hours should be integers
        self.monthly_max_ExFT = [0.0] * num_months
        self.monthly_max_ExFT_time = [0] * num_months  # Hours should be integers

        start_hour = 0

        for month_idx in range(1, len(self.days_in_month) + 1):
            hours_in_month = 24 * self.days_in_month[month_idx]
            end_hour = start_hour + hours_in_month

            month_ExFTs = hourly_ExFThe[start_hour:end_hour]

            if month_ExFTs:  # Only process if we have data
                # Find min and max values
                self.monthly_min_ExFT[month_idx] = min(month_ExFTs)
                self.monthly_max_ExFT[month_idx] = max(month_ExFTs)

                # Find the hours when min and max occur within the month slice
                min_hour_idx = month_ExFTs.index(self.monthly_min_ExFT[month_idx])
                max_hour_idx = month_ExFTs.index(self.monthly_max_ExFT[month_idx])

                # Convert to hour of the year (1-based indexing)
                self.monthly_min_ExFT_time[month_idx] = start_hour + min_hour_idx + 1
                self.monthly_max_ExFT_time[month_idx] = start_hour + max_hour_idx + 1

            start_hour = end_hour

        print("step_5_get_monthly_ExFT_maxs_mins_and_times has run")

    def step_6_sort_n_ExFTpks(self, n):
        """
        sort all monthly ExFT peaks from high to low for ExFTmax and low to high for ExFTmin.
        sort hour of the each peak occurs to match
        pull out the top N number of peaks and the corresponding time stamps

        inputs: self.monthly_min_ExFT
                self.monthly_min_ExFT_time
                self.monthly_max_ExFT
                self.monthly_max_ExFT_time
                n = number of peaks you want to pull out (ex: 4 = 4 heating and 4 cooling)

        Outputs: self.n_monthly_min_ExFTs_sorted
                self.n_monthly_min_ExFTs_time_sorted
                self.n_monthly_max_ExFTs_sorted
                self.n_monthly_max_ExFTs_time_sorted
        """
        # pair up temperatures and corresponding hours of year
        sorted_min_pairs = sorted(zip(self.monthly_min_ExFT, self.monthly_min_ExFT_time))
        sorted_max_pairs = sorted(zip(self.monthly_max_ExFT, self.monthly_max_ExFT_time), reverse=True)

        # Separate back into sorted lists
        sorted_min_temps, sorted_min_hours = zip(*sorted_min_pairs)
        sorted_min_temps = list(sorted_min_temps)
        sorted_min_hours = list(sorted_min_hours)

        sorted_max_temps, sorted_max_hours = zip(*sorted_max_pairs)
        sorted_max_temps = list(sorted_max_temps)
        sorted_max_hours = list(sorted_max_hours)

        # pull out top N number of temps and timestamps

        self.n_monthly_min_ExFTs_sorted = sorted_min_temps[: (n - 1)]
        self.n_monthly_min_ExFTs_time_sorted = sorted_min_temps[: (n - 1)]

        self.n_monthly_max_ExFTs_sorted = sorted_max_temps[: (n - 1)]
        self.n_monthly_max_ExFTs_time_sorted = sorted_max_temps[: (n - 1)]

        print("step_6_sort_n_ExFTpks has run")
        print(f" highest {n} max ExFT are {self.n_monthly_max_ExFTs_sorted}")
        print(f" max ExFTs occur on these hours {self.n_monthly_max_ExFTs_time_sorted}")
        print(f" lowest {n} min ExFT are {self.n_monthly_max_ExFTs_sorted}")
        print(f" min ExFTs occur on these hours {self.n_monthly_min_ExFTs_time_sorted}")

    def step_7_create_hybrid_loads(self):
        """
        This function sets the framework for the final hybrid loads values and timestamps.
        It calls on helper functions _get_month_start_and_end_hours, find_peaks_in_month,
        _add_single_period, _add_single_peak_period, _add_double_peak_period,

        inputs: self.n_monthly_min_ExFTs_sorted : list
                self.n_monthly_min_ExFTs_time_sorted : list
                self.n_monthly_max_ExFTs_sorted : list
                self.n_monthly_max_ExFTs_time_sorted : list

        outputs:
                self.hybrid_loads : list
                self.hybrid_time_step_start_hour : list

                Example: for a month w/ one peak:
                self.hybrid_loads                = [100, 170, 300] W
                self.hybrid_time_step_start_hour = [  0, 165, 168] hr
        """

        # Initialize output lists
        self.hybrid_loads = []
        self.hybrid_time_step_start_hour = []
        # TODO g_funct must be itteratively built
        # starting with month #1 and stepping through each month,
        for month_idx in range(1, len(self.days_in_month) + 1):
            # Calculate month boundaries in terms of hours of the year
            month_start_hour = self.month_start_hours[month_idx]

            # Find peaks that occur in this month
            ExFT_peaks_in_month = self._find_ExFT_peaks_in_month(month_idx)

            if len(ExFT_peaks_in_month) == 0:
                # No peak ExFT, append 1 entry of average load for month
                self._add_zero_peak_month(month_start_hour, month_idx)

            elif len(ExFT_peaks_in_month) == 1:
                # One peak ExFT - append up to 3 entries (pre-peak, on-peak, post-peak)
                self._add_single_peak_month(ExFT_peaks_in_month, month_idx)

            elif len(ExFT_peaks_in_month) == 2:
                # Two peak ExFTs - append up to 5 entries (pre-peak 1, on-peak 1, pre-peak 2, on-peak 2, post-peak 2)
                self._add_double_peak_month(ExFT_peaks_in_month, month_idx)

    def _find_ExFT_peaks_in_month(self, month_idx):
        """Find all peak ExFTs (min and max ExFTs) that occur within the given month"""

        ExFT_peaks = []

        month_start_hour = self.month_start_hours[month_idx]
        month_end_hour = self.month_end_hours[month_idx]

        # Check for min ExFT peaks in this month
        for i, hour in enumerate(self.n_monthly_min_ExFTs_time_sorted):
            if hour in range(month_start_hour, month_end_hour + 1):
                ExFT_peaks.append(
                    {"type": "min", "hour": hour, "value": self.n_monthly_min_ExFTs_sorted[i], "month": month_idx}
                )

        # Check for max ExFT peaks in this month
        for i, hour in enumerate(self.n_monthly_max_ExFTs_time_sorted):
            if hour in range(month_start_hour, month_end_hour + 1):
                ExFT_peaks.append(
                    {"type": "max", "hour": hour, "value": self.n_monthly_min_ExFTs_sorted[i], "month": month_idx}
                )

        ExFT_peaks.sort(key=lambda x: x["hour"])
        return ExFT_peaks

    def _add_zero_peak_month(self, start_hour, month_idx):
        """Add a single month with no peaks"""
        # Calculate average load for the entire month
        avg_load = self.monthly_ave_ground_load[month_idx]

        self.hybrid_loads.append(avg_load)
        self.hybrid_time_step_start_hour.append(start_hour)

    def _add_single_peak_month(self, ExFT_peak, month_idx):
        """Add periods for a month with one peak (up to 3 periods)
        args: peak, month_start_hour, month_end_hour, month index
        output: updated self.hybrid_loads and updated self.hybrid_time_step_start_hour
        """
        on_peak_end_hour = ExFT_peak["hour"]

        month_start_hour = self.month_start_hours[month_idx]
        month_end_hour = self.month_end_hours[month_idx]

        # Call helper function to get pre-peak load, peak load, duration
        pre_peak_load, on_peak_load, on_peak_start_hour = self._calc_hybrid_load_durations(ExFT_peak, month_idx)

        # Pre-peak period (if peak doesn't start at beginning of month)
        if on_peak_start_hour > month_start_hour:
            self.hybrid_loads.append(pre_peak_load)
            self.hybrid_time_step_start_hour.append(month_start_hour)

        # On-peak period
        self.hybrid_loads.append(on_peak_load)
        self.hybrid_time_step_start_hour.append(on_peak_start_hour)

        # Post-peak period (if peak doesn't extend to end of month)
        if on_peak_end_hour < month_end_hour:
            post_peak_load = self.monthly_ave_ground_load[month_idx]
            self.hybrid_loads.append(post_peak_load)
            self.hybrid_time_step_start_hour.append(on_peak_end_hour)

    def _add_double_peak_month(self, peaks_in_month, month_idx):
        """Add periods for a month with two peaks (up to 5 periods)
        args: peak, month_start_hour, month_end_hour, month index
        output: updated self.hybrid_loads and updated self.hybrid_time_step_start_hour
        """
        ExFT_peak1, ExFT_peak2 = peaks_in_month[0], peaks_in_month[1]
        on_peak1_end_hour = ExFT_peak1["hour"]
        on_peak2_end_hour = ExFT_peak2["hour"]

        # Get peak loads and durations for both peaks
        pre_peak1_load, on_peak1_load, on_peak1_start_hour = self._calc_hybrid_load_durations(ExFT_peak1, month_idx)
        pre_peak2_load, on_peak2_load, on_peak2_start_hour = self._calc_hybrid_load_durations(ExFT_peak2, month_idx)

        month_start_hour = self.month_start_hours[month_idx]
        month_end_hour = self.month_end_hours[month_idx]

        # Pre-peak 1 period
        if on_peak1_start_hour > month_start_hour:
            self.hybrid_loads.append(pre_peak1_load)
            self.hybrid_time_step_start_hour.append(month_start_hour)

        # On-peak 1 period
        self.hybrid_loads.append(on_peak1_load)
        self.hybrid_time_step_start_hour.append(on_peak1_start_hour)

        # Between peaks period
        if on_peak1_end_hour < on_peak2_start_hour:
            between_peaks_load = self.monthly_ave_ground_load[month_idx]
            self.hybrid_loads.append(pre_peak2_load)
            self.hybrid_time_step_start_hour.append(on_peak1_end_hour)

        # On-peak 2 period
        self.hybrid_loads.append(on_peak2_load)
        self.hybrid_time_step_start_hour.append(on_peak2_start_hour)

        # Post-peak 2 period
        if on_peak2_end_hour < month_end_hour:
            post_peak2_load = self.monthly_ave_ground_load[month_idx]
            self.hybrid_loads.append(post_peak2_load)
            self.hybrid_time_step_start_hour.append(on_peak2_end_hour)

    def _calc_hybrid_load_durations(self, ExFT_peak, month_idx):
        """
        # calculate the on-peak load duration and pre-peak load that allows for a
        # hybrid time step ExFT to match the hourly time step peak (max or min) ExFT
        #
        # Using the peak load and the peak ExFT for the month,
        # first assign the peak load of the month a one hour duration
        # and adjusted pre-peak average for the hours before the peak
        #
        # run ExFT simulation with the hybrid load for the month
        # if ExFT_hybrid < target_ExFT
        # add 1 hour to the duration of the peak load
        # recalculate the average monthly load to maintain energy balance
        # repeat until ExFT_hybrid >= ExFT_target
        # record load quantities and times of occurrence for the month
        #
        # inputs: ExFT_peak['hour'], ExFT_value, peak_load_value
        #
        # outputs: pre_peak_load, peak_start_hour

        """

        ExFT_peak_hour = ExFT_peak["hour"]
        ExFT_value = ExFT_peak["value"]
        ExFT_type = ExFT_peak["type"]
        ExFT_month = ExFT_peak["month"]

        normalized_ground_load = self.step_2_normalize_loads()
        ExFTs_hourly = self.step_4_perform_hrly_ExFT_simulation(normalized_ground_load)

    def _run_ExFT_simulation_hybrid(self):
        """runs the ExFT simulation given a set of hybrid loads
        inputs: self.hybrid_loads
        Output: ExFT_at_end_hour

        """

    # def update_hybrid_rejection_loads_for_month(self, pk_hour_of_month, month_index, peak_duration):
    #     """
    #     inputs: pk_hour_of_month
    #             month_index
    #             peak_duration
    #     output:hybrid_load
    #     """
    #     hours_in_month = self.days_in_month[month_index] * HRS_IN_DAY

    #     # reset the average for the month to have energy balance
    #     ave_old = self.monthly_avg_rejection[month_index]
    #     ave_new = (
    #         self.monthly_avg_rejection[month_index] * (hours_in_month - peak_duration)
    #         - self.monthly_peak_rejection[month_index] * peak_duration
    #     ) / hours_in_month

    #     print(f"ave_old = {ave_old}")
    #     print(f"ave_new = {ave_new}")

    #     # creates a list that contains the monthly average load for every hour of the
    #     # month (ex: 31*24 identical entries for Jan)
    #     hybrid_load_for_month = [ave_new] * hours_in_month

    #     # updates the ave_load hourly list for the month by inserting the peak cooling
    #     # load at the hour of the month it occurs
    #     hybrid_load_for_month[pk_hour_of_month] = self.monthly_peak_cl[month_index]

    #     print(
    #         f"sample of avg_load_hrly_w_pks array {hybrid_load_for_month[pk_hour_of_month - 2 : pk_hour_of_month + 3]}"
    #     )
    #     return hybrid_load_for_month

    def step_8():
        # reformat hybrid loads into something that GHE designer can use
        #
        # inputs:self.hybrid_loads : list
        #        self.hybrid_time_step_start_hour : list
        # outputs: ?

        pass


def main():
    # run step 1
    building_loads = [1.0] * 8760
    HybridLoadsCalc().step_1_bldg_to_ground_load(building_loads)


if __name__ == "__main__":
    main()
