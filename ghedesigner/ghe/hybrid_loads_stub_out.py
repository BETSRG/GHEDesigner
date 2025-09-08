from calendar import monthrange
from json import dumps
from math import floor

import numpy as np

from ghedesigner.constants import HRS_IN_DAY
from ghedesigner.utilities import simulate_hourly


class HybridLoads2:
    def __init__(
        self,
        building_loads: list,
        ghe: GroundHeatExchanger,
        years=None,
        start_month=None,
        end_month=None,
        cop_h: float = 3.8,
        cop_c: float = 4.5,
    ) -> None:
        self.building_loads = building_loads

        #(used in step 1) Initialize COP values
        self.cop_h = cop_h  # Heating COP
        self.cop_c = cop_c  # Cooling COP

        #(used in step 1.5) initialize months under study. Currently we only do one year, but in the future, the goal is to create hybrid loads for 10+ years
        if years is None:
            years = [2025]
        self.years = years

        # (used in step 1.5) Get the number of days in each month for a given year (make 0 NULL)
        self.days_in_month = [0]  # first entry reserved for annual total or peak, see line 72
        for year in years:
            self.days_in_month.extend([monthrange(year, i)[1] for i in range(1, 13)])




    def step_1():
    #run heatpump with constant COPs to convert building loads to ground loads.
    #these are hourly loads
    #
    #inputs: building_loads, COPc, COPh
    #output: ground_loads

        def bldg_to_ground_load(self, bldg_loads) -> list:
            """
            Acts as a constant COP heatpump
            :return: ground loads in Watts
            """
            ground_loads = []

            for i in range(len(bldg_loads)):
                if bldg_loads[i] >= 0:
                    ground_loads.append((self.cop_h - 1) / self.cop_h * bldg_loads[i])
                else:
                    ground_loads.append(-(1 + 1 / self.cop_c) * abs(bldg_loads[i]))

            return ground_loads
    pass



def step_2():
    # since borehole config and size is unknown,
    # Assume a 100m deep single borehole and 40W/m peak
    #  load and normalize to this peak (4000W max)
    #
    # inputs: ground_loads
    #outputs: normalized_ground_loads

    @staticmethod
    def normalize_loads(load) -> list:
        """
        Normalize the ground loads to a peak of 4000 W since we do not know the size or
        configuration of the borehole field
        :return: hourly ground loads normalized to 4000 W peak [Watts]
        """
        # Handle case where all loads are zero
        if not load or all(x == 0 for x in load):
            print("Warning: All loads are zero, returning zeros")
            return [0.0] * len(load)

        max_ground_loads = max(load)
        if max_ground_loads == 0:
            print("Warning: Maximum load is zero, returning zeros")
            return [0.0] * len(load)

        #normalized to 40W/m for a 100 m borehole
        normalized_loads = [4000 / max_ground_loads * load[i] for i in range(len(load))]

        print("Normalize loads has run.")

        return normalized_loads
    pass

def step_3():
    """parse out ground loads into peak and average for each month of the year.
        also calculate total monthly totals for energy balence.
        done for both heating and cooling.
    inputs: ground_loads
    outputs: self.monthly_total_rejection (list of 12),
            self.monthly_total_extraction (list of 12),
            self.monthly_peak_rejection,
            self.monthly_peak_extraction,
            self.monthly_ave_rejection,
            self.monthly_ave_extraction,
    """

    def split_loads_by_month(self):
        """Split the normalized ground loads into peak, total, and average loads for each month"""

        #run split heat and cool. returns normed and unnormed heating and cooling ground loads. hourly for 1 year
        split_heat_and_cool()

        num_unique_months = len(self.years) * 12 + 1

        #set up empty lists
        # monthly total cooling energy (or heat rejection) in kWh
        self.monthly_total_rejection = [0.0] * num_unique_months
        # monthly total heating energy (or heat extraction) in kWh
        self.monthly_total_extraction = [0.0] * num_unique_months

        # monthly peak cooling load (or heat rejection) in kW
        self.monthly_peak_rejection = [0.0] * num_unique_months
        # monthly peak heating load (or heat extraction) in kW
        self.monthly_peak_extraction = [0.0] * num_unique_months

        # monthly average cooling load (or heat rejection) in kW
        self.monthly_avg_rejection = [0.0] * num_unique_months
        # monthly average heating load (or heat extraction) in kW
        self.monthly_avg_extraction = [0.0] * num_unique_months

        # Store the index of the last month's hours
        hours_in_previous_months: int = 0  # type is integer and set to a value of 0 to start
        i: int  # set i type to integer
        #cycle through each month
        for i in range(1, len(self.days_in_month)):

            hours_in_month = HRS_IN_DAY * self.days_in_month[i]  # e.g. 24 * 31, to give hrs in month

            # Slice the hours in this current month
            current_month_reject_norm_loads = self.hrly_rejection_loads_norm[
                hours_in_previous_months : hours_in_previous_months + hours_in_month
            ]  # first slice will be from [0:0+24*31]
            current_month_extract_norm_loads = self.hrly_extraction_loads_norm[
                hours_in_previous_months : hours_in_previous_months + hours_in_month
            ]

            # Handle empty lists
            if not current_month_reject_norm_loads:
                current_month_reject_norm_loads = [0.0]
            if not current_month_extract_norm_loads:
                current_month_extract_norm_loads = [0.0]

            # Sum
            # monthly cooling loads (or heat rejection) in kWh
            self.monthly_total_rejection[i] = sum(current_month_reject_norm_loads)
            # monthly heating loads (or heat extraction) in kWh
            self.monthly_total_extraction[i] = sum(current_month_extract_norm_loads)

            # Peak
            # monthly peak cooling load (or heat rejection) in kW
            self.monthly_peak_rejection[i] = max(current_month_reject_norm_loads) if current_month_reject_norm_loads else 0.0
            # monthly peak heating load (or heat extraction) in kW
            self.monthly_peak_extraction[i] = max(current_month_extract_norm_loads) if current_month_extract_norm_loads else 0.0

            # Average
            # monthly average cooling load (or heat rejection) in kW
            self.monthly_avg_rejection[i] = (
                self.monthly_total_rejection[i] / len(current_month_reject_norm_loads) if current_month_reject_norm_loads else 0.0
            )
            # monthly average heating load (or heat extraction) in kW
            self.monthly_avg_extraction[i] = (
                self.monthly_total_extraction[i] / len(current_month_extract_norm_loads) if current_month_extract_norm_loads else 0.0
            )

            hours_in_previous_months += hours_in_month

        print("split_loads_by_month has run")


    def split_heat_and_cool(self):
        """
        Splits the hourly ground loads into heating and cooling
        Heating is positive, cooling is negative.
        :return: hourly ground loads split into heating and cooling in kilowatts
        :return: hourly ExFTs split into heating and cooling lists in Celsius
        """
        #split hourly ground loads into extraction and rejection and also convert W into kWs
        self.hourly_extraction_loads = [x / 1000.0 if x >= 0.0 else 0.0 for x in self.building_loads]
        self.hourly_rejection_loads = [abs(x) / 1000.0 if x < 0.0 else 0.0 for x in self.building_loads]

        self.hrly_extraction_loads_norm = self.normalize_loads(self.hourly_extraction_loads)
        self.hrly_rejection_loads_norm = self.normalize_loads(self.hourly_rejection_loads)

        print("split_heat_and_cool has run")

    pass

def step_4():
    # take the normalized ground loads and use
    # a gfunction and the other ground heat exchanger parameters to get the hourly
    # exiting fluid temperature from the ghe (same as entering to HP)
    #
    # inputs: normalized ground loads, ground heat exchanger parameters
    #output: hourly exiting fluid temperature (ExFT) from the GHE
    def perform_hrly_ExFT_simulation(self, load_profile):
        """
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
        hours_in_year = list(range(len(q)))
        delta_t_fluid = simulate_hourly(hours_in_year, q, g, resist_bh_effective, two_pi_k, ts)
        return delta_t_fluid

def step_5():
    """ split the yearly ExFT of the GHE into months. Identify the peak and hour on which it occurs for each month
        inputs: hourly_EXFTghe
        outputs:
        """
    def split_ExFT_by_month(self) -> None:
        """Slice the hourly ExFT of the GHE into months
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

            current_month_ExFTs = self.target_ExFTghe_temps[
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




    pass

def step_5a():
    #find the 4 highest peak exiting fluid temperatures for the year for cooling
    # only one peak allowed per month
    # return the time that the peak occurs

    # inputs: hourly_ExFTs
    #outputs: 4 peak ExFTs, associated month, associated hour of month
    def get_max_4_ExFT_and_time(self):
        """
        Returns the 4 months in which peak ExFT occurs for cooling, the peak ExFT, and the hour of the month.
        Stores the result as a 2D NumPy array with columns: [hour, month, ExFT value].
        """

        # Filter out zero/invalid entries and create valid entries
        valid_entries = []
        for i, max_val in enumerate(self.monthly_max_ExFT):
            if i != 0 and max_val != 0:  # Skip index 0 and zero values
                valid_entries.append(
                    (
                        int(max_val),  # ExFT
                        int(self.monthly_max_GHE_ExFT_hour[i]),  # hour within month
                        int(i),  # month
                        int(self.cumulative_hours[i] + self.monthly_max_GHE_ExFT_hour[i]),  # absolute hour of year
                    )
                )

        # Sort and take top 4 (or fewer if not enough valid entries)
        top_4 = sorted(valid_entries, key=lambda x: x[0], reverse=False)[:4]

        if len(top_4) == 0:
            # Return empty array with correct shape if no valid data
            max_4_ExFT = np.array([]).reshape(0, 4)
            print("max ExFT, hour of month, month, hour of year ")
            print("No valid data found")
        else:
            max_4_ExFT = np.array(top_4, dtype=int)  # Ensure integer dtype
            print("max ExFT, hour of month, month, hour of year ")

            # Format each row for clean integer display
            for row in max_4_ExFT:
                print(f"{row[0]:6d} {row[1]:12d} {row[2]:5d} {row[3]:12d}")

        return max_4_ExFT


    pass

def step_5b():
    #repeat step 4 but for heating
    #looking for lowest ExFTs, 4 month peaks and times
    #
    # inputs: hourly_ExFTs
    #outputs: 4 min ExFTs, associated month, associated hour of the month

    def get_min_4_ExFT_and_time(self):
        """
        Returns the 4 months in which peak ExFT occurs for cooling, the peak ExFT, and the hour of the month.
        Stores the result as a 2D NumPy array with columns: [ExFT value, hour of month, month, hour of year].
        """

        # Filter out zero/invalid entries and create valid entries
        valid_entries = []
        for i, min_val in enumerate(self.monthly_min_ExFT):
            if i != 0 and min_val != 0:  # Skip index 0 and zero values
                valid_entries.append(
                    (
                        int(min_val),  # ExFT value
                        int(self.monthly_min_GHE_ExFT_hour[i]),  # hour within month
                        int(i),  # month index
                        int(self.cumulative_hours[i] + self.monthly_min_GHE_ExFT_hour[i]),  # absolute hour of year
                    )
                )

        # Sort and take top 4 (or fewer if not enough valid entries)
        top_4_min = sorted(valid_entries, key=lambda x: x[0], reverse=True)[:4]

        if len(top_4_min) == 0:
            # Return empty array with correct shape if no valid data
            min_4_ExFT = np.array([]).reshape(0, 4)
            print("min ExFT, hour of month, month, hour of year ")
            print("No valid data found")
        else:
            min_4_ExFT = np.array(top_4_min, dtype=int)  # Ensure integer dtype
            print("min ExFT, hour of month, month, hour of year ")

            # Format each row for clean integer display
            for row in min_4_ExFT:
                print(f"{row[0]:6d} {row[1]:12d} {row[2]:5d} {row[3]:12d}")

        return min_4_ExFT

    pass

def step_6():
    # figure out the peak load duration that allows for a
    # hybrid time step simulation's ExFT to match the hourly time step's peak ExFT
    # hourly sim's ExFT will be called ExFT_target
    # using the peak load for the month and the peak ExFT
    # assign the peak load of that month to a one hour duration and the average
    # monthly load for the rest of the month. adjusted for energy balance
    # run ExFT simulation with the hybrid load for the month
    # if ExFT_hybrid < ExFT_target
        # add 1 hour to the duration of the peak load
        #recalculate the average monthly load to maintain energy balence
    #repeat until ExFT_hybrid >= ExFT_target
    #record load quantities and times of occurance for the month
    #
    # inputs: peak_load_of_month_n, peak_ExFT_of_month, peak_ExFT_hour,
    #
    #outputs: load amount and duration for that month(see below for example)
    #           [time,           load]
    #         [hrs 0:300,  average_load_for_month ]
    #         [hrs 301:305, peak_load_for_month]
    #         [hrs 306:720,  average_load_for_month ]




    pass
def step_7():
    # repeat step 6 for remaining heating and cooling peaks
    # run though all 8 months
    #
    #input: 4 min ExFTs, associated month, associated hour of the month,
    #       4 max ExFTs, associated month, associated hour of month
    #
    #outputs: hybridloads_heating_month1, hybrid_timesteps_month1
    #           hybrid_loads_heating_month2, hybrid_timesteps_month2...
    #          ...hybrid_loads_cooling_month8, hybrid_timestep_month8
    pass

def step_8():
    #compile new hybrid loads
    # use monthly average load for all non-peak months
    #inputs: all 12 month's new hybrid load profiles
    #outputs: hybrid_loads_for_the_year, durations
    #example
    # [320 W, 0:60 hrs] (jan)
    # [1600 W, 60:66 hrs] (jan peak)
    # [320 W, 67:720 hrs] (jan)
    # [270 w, 721:789 hrs] (feb)
    # [...] (feb_peak)

    pass

def main():

    #run step 1
    step_1()

if __name__ == "__main__":
    main()
