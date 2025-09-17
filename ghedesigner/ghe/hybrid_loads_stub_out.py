from calendar import monthrange

import numpy as np

from ghedesigner.constants import HRS_IN_DAY
from ghedesigner.utilities import simulate_hourly


class HybridLoadsCalc:
    def __init__(
        self,
        years: list[float] | None = None,
        start_month = None,
        end_month = None,
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

        # (used in step 1.5) Get the number of days in each month for a given year (make 0 NULL)
        self.days_in_month = [0]  # first entry reserved for annual total or peak, see line 72
        for year in years:
            self.days_in_month.extend([monthrange(year, i)[1] for i in range(1, 13)])

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

    def step_5_get_monthly_ExFT_maxs_mins_and_times(self) -> None:
        """split the yearly ExFT of the GHE into months. Identify the peak and hour on which it occurs for each month
        inputs: hourly_EXFThe
        outputs:
        Slice the hourly ExFT of the GHE into months
        input:
        output: self.monthly_min_ExFT
                self.monthly_max_ExFT
        """

        # Store the index of the last month's hours
        hours_in_previous_months: int = 0
        i: int
        for i in range(1, len(self.days_in_month)):
            hours_in_month = HRS_IN_DAY * self.days_in_month[i]
            # Slice the hours in this current month

            current_month_ExFTs = self.target_ExFThe_temps[
                hours_in_previous_months : hours_in_previous_months + hours_in_month
            ]

            # Handle empty ExFT arrays
            if not current_month_ExFTs:
                self.monthly_min_ExFT[i] = 0.0
                self.monthly_max_ExFT[i] = 0.0
                self.monthly_min_GHE_ExFT_hour[i] = 0
                self.monthly_max_GHE_ExFT_hour[i] = 0
            else:
                # Convert to numpy array and ensure it's at least 1D
                current_month_ExFTs = np.atleast_1d(np.array(current_month_ExFTs))

                # Peak
                # monthly peak (min) exiting fluid temperature from the GHE [degrees C]
                self.monthly_min_ExFT[i] = float(np.min(current_month_ExFTs))
                # monthly peak exiting fluid temperature from the GHE [degrees C]
                self.monthly_max_ExFT[i] = float(np.max(current_month_ExFTs))

                # Hour of the month the min ExFT for the GHE occurs (EFT for heatpump). critical for heating
                try:
                    min_indices = np.where(current_month_ExFTs == self.monthly_min_ExFT[i])[0]
                    if len(min_indices) > 0:
                        self.monthly_min_GHE_ExFT_hour[i] = int(min_indices[0])
                    else:
                        self.monthly_min_GHE_ExFT_hour[i] = 0
                except (IndexError, ValueError):
                    self.monthly_min_GHE_ExFT_hour[i] = 0

                # Hour of the month the max ExFT for the GHE occurs (EFT for heatpump). critical for cooling
                try:
                    max_indices = np.where(current_month_ExFTs == self.monthly_max_ExFT[i])[0]
                    if len(max_indices) > 0:
                        self.monthly_max_GHE_ExFT_hour[i] = int(max_indices[0])
                    else:
                        self.monthly_max_GHE_ExFT_hour[i] = 0
                except (IndexError, ValueError):
                    self.monthly_max_GHE_ExFT_hour[i] = 0

            hours_in_previous_months += hours_in_month

        print("split_ExFT_by_month has run")

    def step_6_find_peak_rejection_durations(self):
        """
        # figure out the peak load duration that allows for a
        # hybrid time step simulation's ExFT to match the hourly time step's peak ExFT
        # hourly sim's ExFT is called target_ExFT
        # using the peak load for the month and the peak ExFT
        # assign the peak load of that month to a one hour duration and the average
        # monthly load for the rest of the month. adjusted for energy balance
        # run ExFT simulation with the hybrid load for the month
        # if ExFT_hybrid < target_ExFT
        # add 1 hour to the duration of the peak load
        # recalculate the average monthly load to maintain energy balance
        # repeat until ExFT_hybrid >= ExFT_target
        # record load quantities and times of occurrence for the month
        #
        # inputs: peak_load_of_month_n, peak_ExFT_of_month, peak_ExFT_hour,
        #
        # outputs: hybrid load profile for that month(see below for example)
        # hybrid_load_month1 =  [ average, average, average, peak, peak, average,... average]

        find the peak load durations by using the GHE ExFT and increment peak load duration
        1 hr at a time until ExFT matches hourly simulation
        :return: the 4 peak load durations
        """

        durations = []

        # Check if max_4_ExFT is empty
        if len(self.max_4_ExFT) == 0:
            print("No valid max ExFT data found, returning empty durations")
            return durations

        for row in self.max_4_ExFT:  # iterates through the 4 identified peaks
            target_ExFT = row[0]  # our target ExFT for the iteration
            pk_hour_of_month = int(row[1])
            month_index = int(row[2])
            # pk_hour_of_year = int(row[3])
            peak_duration = 1
            hours_in_month = self.days_in_month[month_index] * HRS_IN_DAY

            hybrid_load = self.update_hybrid_rejection_loads_for_month(pk_hour_of_month, month_index, peak_duration)

            print(
                f"peak load for month {month_index} occurs at {pk_hour_of_month} hour and "
                f"has a value of {self.monthly_peak_cl[month_index]}"
            )

            print(f"\nMonth {month_index}: starting search for peak duration.")

            max_hours = hours_in_month
            while peak_duration < max_hours:
                new_max_ExFT = self.perform_hrly_ExFT_simulation(pk_hour_of_month, hybrid_load)

                # print(f"Trying duration {peak_duration} hr: max ExFT = {new_max_ExFT:.2f}, target = {target_ExFT:.2f}")

                if new_max_ExFT > target_ExFT:
                    break
                peak_duration += 1

            durations.append((month_index, peak_duration))  # (month, duration)

        print("\nFinal peak durations (month, hours):", durations)
        return durations

    def update_hybrid_rejection_loads_for_month(self, pk_hour_of_month, month_index, peak_duration):
        """
        inputs: pk_hour_of_month
                month_index
                peak_duration
        output:hybrid_load
        """
        hours_in_month = self.days_in_month[month_index] * HRS_IN_DAY

        # reset the average for the month to have energy balance
        ave_old = self.monthly_avg_rejection[month_index]
        ave_new = (
            self.monthly_avg_rejection[month_index] * (hours_in_month - peak_duration)
            - self.monthly_peak_rejection[month_index] * peak_duration
        ) / hours_in_month

        print(f"ave_old = {ave_old}")
        print(f"ave_new = {ave_new}")

        # creates a list that contains the monthly average load for every hour of the
        # month (ex: 31*24 identical entries for Jan)
        hybrid_load_for_month = [ave_new] * hours_in_month

        # updates the ave_load hourly list for the month by inserting the peak cooling
        # load at the hour of the month it occurs
        hybrid_load_for_month[pk_hour_of_month] = self.monthly_peak_cl[month_index]

        print(
            f"sample of avg_load_hrly_w_pks array {hybrid_load_for_month[pk_hour_of_month - 2 : pk_hour_of_month + 3]}"
        )
        return hybrid_load_for_month

    def step_7():
        # repeat step 6 for remaining heating and cooling peaks
        # run though all 8 months
        #
        # input: 4 min ExFTs, associated month, associated hour of the month,
        #       4 max ExFTs, associated month, associated hour of month
        #
        # outputs: hybridloads_heating_month1, hybrid_timesteps_month1
        #           hybrid_loads_heating_month2, hybrid_timesteps_month2...
        #          ...hybrid_loads_cooling_month8, hybrid_timestep_month8
        pass

    def step_8():
        # compile new hybrid loads
        # use monthly average load for all non-peak months
        # inputs: all 12 month's new hybrid load profiles
        # outputs: hybrid_loads_for_the_year, durations
        # example
        # [320 W, 0:60 hrs] (jan)
        # [1600 W, 60:66 hrs] (jan peak)
        # [320 W, 67:720 hrs] (jan)
        # [270 w, 721:789 hrs] (feb)
        # [...] (feb_peak)

        pass


def main():
    # run step 1
    building_loads = [1.0] * 8760
    HybridLoadsCalc().step_1_bldg_to_ground_load(building_loads)


if __name__ == "__main__":
    main()
