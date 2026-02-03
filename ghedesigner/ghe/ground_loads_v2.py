from calendar import monthrange

import numpy as np

from ghedesigner.constants import HRS_IN_DAY, SEC_IN_HR, TWO_PI
from ghedesigner.ghe.boreholes.single_u_borehole import SingleUTube


class HybridLoadV2:
    """New hybrid load algorithm based on Spitler (2024) white paper.

    Key differences from HybridLoad:
    - Uses peak *temperatures* (from hourly simulation) rather than peak *loads*
      to identify critical months and determine peak durations.
    - Normalizes loads to a single 100m borehole at 40 W/m for peak analysis.
    - Only the top 4 heating and top 4 cooling months get peak time steps;
      remaining months are single average-load time steps.
    - Peak durations are iteratively calibrated so the hybrid simulation
      reproduces the same peak temperature as the hourly simulation.
    """

    # Normalization target: 40 W/m * 100 m = 4000 W
    NORM_LOAD_W = 40.0 * 100.0  # 4000 W
    NUM_PEAK_MONTHS = 4  # top N months for heating and cooling peaks

    def __init__(
        self,
        raw_loads: list,
        bhe: SingleUTube,
        radial_numerical: SingleUTube,
        start_month: int,
        end_month: int,
    ) -> None:
        self.raw_loads = raw_loads  # 8760 hourly loads in Watts (positive=extraction, negative=rejection)
        self.bhe = bhe
        self.radial_numerical = radial_numerical
        self.start_month = start_month
        self.end_month = end_month

        # Compute years for multi-year simulation and leap year support
        num_years = max(1, (end_month + 11) // 12)
        base_year = 2026 #TODO - in the input file, the user should be able to specify what year, that way if using multiyear loads, leap years are correctly accounted for)
        self.years = [base_year + y for y in range(num_years)]

        # Days in each month for the first year only (index 0 is placeholder).
        # Steps 1-6 operate on a single year of hourly data (8760 or 8784 hours).
        self.days_in_month = [0]
        self.days_in_month.extend([monthrange(self.years[0], i)[1] for i in range(1, 13)])

        # --- Step 1: Normalize the loads ---
        self.normalized_loads = self.normalize_loads(self.raw_loads)

        # --- Step 2: Monthly load metrics (on original loads, for later use) ---
        self.monthly_cl, self.monthly_hl = None, None
        self.monthly_peak_cl, self.monthly_peak_hl = None, None
        self.monthly_avg_cl, self.monthly_avg_hl = None, None
        self.split_loads_by_month()

        # --- Step 3: Hourly simulation on normalized loads ---
        self.hourly_delta_t = self._run_hourly_simulation()

        # --- Step 4: Identify peak temperature months ---
        # Per-month: max delta_T (cooling/rejection peak), min delta_T (heating/extraction peak)
        # and the hour-of-year at which each occurs
        self.monthly_max_dt = [0.0] * 13  # max delta_T per month (index 1-12)
        self.monthly_min_dt = [0.0] * 13  # min delta_T per month
        self.monthly_max_dt_hour = [0] * 13  # hour-of-year of max delta_T
        self.monthly_min_dt_hour = [0] * 13  # hour-of-year of min delta_T
        self._find_monthly_peak_temperatures()

        # --- Step 5: Select top 4 peak months for cooling and heating ---
        self.peak_cooling_months: list[int] = []  # months with highest max delta_T
        self.peak_heating_months: list[int] = []  # months with lowest min delta_T
        self._select_peak_months()

        # --- Step 6: Find peak durations ---
        self.monthly_peak_cl_duration = [0.0] * 13
        self.monthly_peak_hl_duration = [0.0] * 13
        self._find_peak_durations()

        # --- Step 7: Construct hybrid time step arrays ---
        self.load = np.array(0)
        self.hour = np.array(0)
        self.step_func_load = np.array(0)
        self._process_month_loads()

    # -----------------------------------------------------------------
    # Step 1: Normalize the loads
    # -----------------------------------------------------------------
    @staticmethod
    def normalize_loads(raw_loads: list) -> np.ndarray:
        """Normalize hourly loads for single-borehole peak analysis.

        Per Eq. 2 of Spitler (2024):
            Q_net,n,norm = Q_net,n / max(|Q_net|) * (40 W/m * 100 m)

        This scales all loads so the absolute peak equals 4000 W,
        suitable for simulation with a single 100m borehole.

        :param raw_loads: 8760 hourly net loads in Watts
        :return: Normalized loads as numpy array in Watts
        """
        loads = np.array(raw_loads, dtype=float)
        max_abs_load = np.max(np.abs(loads))
        if max_abs_load == 0.0:
            return loads
        return loads / max_abs_load * HybridLoadV2.NORM_LOAD_W

    # -----------------------------------------------------------------
    # Step 2: Monthly load metrics (on original loads)
    # -----------------------------------------------------------------
    def split_loads_by_month(self) -> None:
        """Compute monthly totals, peaks, averages from original loads.

        These are computed on the original (non-normalized) loads in kW,
        and will be used later when constructing the final hybrid time
        step arrays with actual load magnitudes.
        """
        num_months = len(self.days_in_month)

        # All arrays use index 0 as placeholder, 1-12 for months
        self.monthly_cl = [0.0] * num_months  # total cooling (rejection) kWh
        self.monthly_hl = [0.0] * num_months  # total heating (extraction) kWh
        self.monthly_peak_cl = [0.0] * num_months  # peak cooling kW
        self.monthly_peak_hl = [0.0] * num_months  # peak heating kW
        self.monthly_avg_cl = [0.0] * num_months  # average cooling kW
        self.monthly_avg_hl = [0.0] * num_months  # average heating kW

        hours_in_previous_months = 0
        for i in range(1, num_months):
            hours_in_month = HRS_IN_DAY * self.days_in_month[i]
            month_loads = self.raw_loads[hours_in_previous_months:hours_in_previous_months + hours_in_month]

            # Split into rejection (negative raw = cooling) and extraction (positive raw = heating)
            month_rejection = [abs(x) / 1000.0 if x < 0 else 0.0 for x in month_loads]
            month_extraction = [x / 1000.0 if x >= 0 else 0.0 for x in month_loads]

            self.monthly_cl[i] = sum(month_rejection)
            self.monthly_hl[i] = sum(month_extraction)
            self.monthly_peak_cl[i] = max(month_rejection)
            self.monthly_peak_hl[i] = max(month_extraction)
            self.monthly_avg_cl[i] = self.monthly_cl[i] / hours_in_month if hours_in_month > 0 else 0.0
            self.monthly_avg_hl[i] = self.monthly_hl[i] / hours_in_month if hours_in_month > 0 else 0.0

            hours_in_previous_months += hours_in_month

    # -----------------------------------------------------------------
    # Hourly simulation utility
    # -----------------------------------------------------------------
    @staticmethod
    def simulate_hourly(hour_time, q, g_sts, resist_bh, two_pi_k, ts):
        """Hourly fluid temperature simulation using g-function superposition.

        Based on Chapter 2 of Advances in Ground Source Heat Pumps.
        Sign convention: positive q = heat rejection into ground → positive delta_T.

        :param hour_time: array of time values (hours)
        :param q: array of loads at each time step, q[0]=0
        :param g_sts: scipy interp1d for short-time-step g-function
        :param resist_bh: effective borehole resistance (m.K/W)
        :param two_pi_k: 2*pi*k_soil (W/m.K)
        :param ts: characteristic time for STS g-function (s)
        :return: list of delta fluid temperatures for each time step
        """
        q_dt = np.hstack(q[1:] - q[:-1])
        delta_t_fluid = [0.0]
        for n in range(1, len(hour_time)):
            _time = hour_time[n] - hour_time[0:n]
            g_values = g_sts(np.log((_time * SEC_IN_HR) / ts))
            delta_tb_i = (q_dt[0:n] / two_pi_k).dot(g_values)
            tf_mean = delta_tb_i + q[n] * resist_bh
            delta_t_fluid.append(tf_mean)
        return delta_t_fluid

    # -----------------------------------------------------------------
    # Step 3: Run hourly simulation on normalized loads
    # -----------------------------------------------------------------
    def _run_hourly_simulation(self) -> list:
        """Run a full-year hourly simulation using normalized loads.

        Uses the single-borehole STS g-function to compute delta_T_fluid
        at each hour. The normalized loads ensure results are independent
        of borefield size/configuration.

        :return: List of 8761 delta_T values (index 0 = hour 0 = 0.0)
        """
        ts = self.radial_numerical.t_s
        two_pi_k = TWO_PI * self.bhe.soil.k
        resist_bh = self.bhe.calc_effective_borehole_resistance()
        g_sts = self.radial_numerical.g_sts

        n_hours = len(self.normalized_loads)
        hour_time = np.arange(n_hours + 1)  # 0, 1, 2, ..., 8760

        # Prepend 0 load at hour 0
        # Negate: raw convention is positive=extraction, but simulate_hourly
        # expects positive=rejection (heat into ground -> positive delta_T)
        q = np.hstack((0.0, -self.normalized_loads))

        return self.simulate_hourly(hour_time, q, g_sts, resist_bh, two_pi_k, ts)

    # -----------------------------------------------------------------
    # Step 4: Find monthly peak temperatures and their hours
    # -----------------------------------------------------------------
    def _find_monthly_peak_temperatures(self) -> None:
        """For each month, find the max and min delta_T_fluid and the
        hour-of-year at which each occurs.

        Max delta_T corresponds to peak cooling/rejection (hottest fluid).
        Min delta_T corresponds to peak heating/extraction (coldest fluid).
        """
        hours_in_previous_months = 0
        for month_idx in range(1, len(self.days_in_month)):
            hours_in_month = HRS_IN_DAY * self.days_in_month[month_idx]

            # Slice delta_T for this month (hours are 1-indexed in the sim)
            start_hr = hours_in_previous_months + 1
            end_hr = hours_in_previous_months + hours_in_month + 1
            month_dt = self.hourly_delta_t[start_hr:end_hr]

            if len(month_dt) == 0:
                hours_in_previous_months += hours_in_month
                continue

            # Max delta_T (peak rejection/cooling temperature)
            max_idx = int(np.argmax(month_dt))
            self.monthly_max_dt[month_idx] = month_dt[max_idx]
            self.monthly_max_dt_hour[month_idx] = hours_in_previous_months + max_idx + 1

            # Min delta_T (peak extraction/heating temperature)
            min_idx = int(np.argmin(month_dt))
            self.monthly_min_dt[month_idx] = month_dt[min_idx]
            self.monthly_min_dt_hour[month_idx] = hours_in_previous_months + min_idx + 1

            hours_in_previous_months += hours_in_month

    # -----------------------------------------------------------------
    # Step 5: Select top N peak months for cooling and heating
    # -----------------------------------------------------------------
    def _select_peak_months(self) -> None:
        """Identify the top NUM_PEAK_MONTHS months with highest max delta_T
        (cooling peaks) and lowest min delta_T (heating peaks).

        Only these months will get peak time steps in the hybrid simulation;
        all other months are treated as single average-load time steps.
        """
        n = self.NUM_PEAK_MONTHS

        # Months 1-12 with their max delta_T values
        month_max_pairs = [(m, self.monthly_max_dt[m]) for m in range(1, 13)]
        # Sort descending by max delta_T -- highest temperature rises first
        month_max_pairs.sort(key=lambda x: x[1], reverse=True)
        self.peak_cooling_months = [m for m, _ in month_max_pairs[:n]]

        # Months 1-12 with their min delta_T values
        month_min_pairs = [(m, self.monthly_min_dt[m]) for m in range(1, 13)]
        # Sort ascending by min delta_T -- lowest temperature drops first
        month_min_pairs.sort(key=lambda x: x[1])
        self.peak_heating_months = [m for m, _ in month_min_pairs[:n]]

    # -----------------------------------------------------------------
    # Step 6: Find required peak durations
    # -----------------------------------------------------------------
    def _find_peak_durations(self) -> None:
        """Find peak durations for all peak months by iterative matching.

        For each peak month, iteratively increases the peak duration from
        1 hour until the hybrid simulation's peak temperature matches the
        hourly simulation's peak temperature. Then interpolates to get the
        fractional duration.

        Uses normalized loads in simulation convention (positive = rejection).
        """
        # Convert normalized loads to sim convention (positive = rejection)
        sim_loads = -np.array(self.normalized_loads)

        # Precompute cumulative hour offsets for each month boundary
        # monthly_hour_offset[m] = total hours before month m starts (0-indexed into loads)
        # monthly_hour_offset[1] = 0 (January starts at index 0)
        monthly_hour_offset = [0] * 14
        for m in range(1, 13):
            monthly_hour_offset[m + 1] = (
                monthly_hour_offset[m] + HRS_IN_DAY * self.days_in_month[m]
            )
        # Shift so month 1 starts at 0
        # monthly_hour_offset[1] = 0, [2] = 744 (hours in Jan), etc.
        # But currently [0]=0, [1]=0, [2]=Jan_hours... Let me fix the indexing.
        # We want: offset[m] = hours before month m (for m=1..12)
        # offset[1] = 0, offset[2] = hours_in_jan, ..., offset[13] = 8760
        month_hour_offset = [0] * 14
        for m in range(1, 14):
            if m <= 12:
                month_hour_offset[m] = sum(
                    HRS_IN_DAY * self.days_in_month[j] for j in range(1, m)
                )
            else:
                month_hour_offset[13] = sum(
                    HRS_IN_DAY * self.days_in_month[j] for j in range(1, 13)
                )

        # Monthly net average loads (sim convention, positive = rejection)
        monthly_net_avg_sim = [0.0] * 13
        for m in range(1, 13):
            m_start = month_hour_offset[m]
            m_end = month_hour_offset[m + 1]
            if m_end > m_start:
                monthly_net_avg_sim[m] = float(np.mean(sim_loads[m_start:m_end]))

        # Simulation parameters
        ts = self.radial_numerical.t_s
        two_pi_k = TWO_PI * self.bhe.soil.k
        resist_bh = self.bhe.calc_effective_borehole_resistance()
        g_sts = self.radial_numerical.g_sts

        # Process cooling peaks (months with highest max delta_T)
        for m in self.peak_cooling_months:
            peak_hour = self.monthly_max_dt_hour[m]  # hour-of-year (1-indexed)
            target_dt = self.monthly_max_dt[m]

            # Peak rejection load for this month (positive, sim convention)
            m_start = month_hour_offset[m]
            m_end = month_hour_offset[m + 1]
            q_peak = float(np.max(sim_loads[m_start:m_end]))

            self.monthly_peak_cl_duration[m] = self._iterate_duration(
                peak_month=m,
                peak_hour=peak_hour,
                target_dt=target_dt,
                q_peak=q_peak,
                monthly_net_avg_sim=monthly_net_avg_sim,
                month_hour_offset=month_hour_offset,
                sim_loads=sim_loads,
                g_sts=g_sts,
                resist_bh=resist_bh,
                two_pi_k=two_pi_k,
                ts=ts,
                is_cooling=True,
            )

        # Process heating peaks (months with lowest min delta_T)
        for m in self.peak_heating_months:
            peak_hour = self.monthly_min_dt_hour[m]  # hour-of-year (1-indexed)
            target_dt = self.monthly_min_dt[m]

            # Peak extraction load for this month (negative, sim convention)
            m_start = month_hour_offset[m]
            m_end = month_hour_offset[m + 1]
            q_peak = float(np.min(sim_loads[m_start:m_end]))

            self.monthly_peak_hl_duration[m] = self._iterate_duration(
                peak_month=m,
                peak_hour=peak_hour,
                target_dt=target_dt,
                q_peak=q_peak,
                monthly_net_avg_sim=monthly_net_avg_sim,
                month_hour_offset=month_hour_offset,
                sim_loads=sim_loads,
                g_sts=g_sts,
                resist_bh=resist_bh,
                two_pi_k=two_pi_k,
                ts=ts,
                is_cooling=False,
            )

    def _iterate_duration(
        self,
        peak_month: int,
        peak_hour: int,
        target_dt: float,
        q_peak: float,
        monthly_net_avg_sim: list,
        month_hour_offset: list,
        sim_loads: np.ndarray,
        g_sts,
        resist_bh: float,
        two_pi_k: float,
        ts: float,
        is_cooling: bool,
    ) -> float:
        """Iteratively find peak duration for one peak month.

        Builds a hybrid load sequence from the start of the year to the
        peak temperature hour, with:
        - Previous months at their net average load
        - Peak month split into non-peak (adjusted avg) + peak load

        Increases duration by 1 hour each iteration until the hybrid
        simulation's peak temperature matches or exceeds the hourly
        simulation's target. Then interpolates for fractional duration.

        Per Eq. 3-4 of Spitler (2024), the non-peak load is adjusted so
        total energy from month start to peak hour is conserved.

        :return: Peak duration in hours (may be fractional)
        """
        m_start_offset = month_hour_offset[peak_month]
        # peak_hour is 1-indexed hour-of-year; convert to 0-indexed offset
        # into the loads array for energy summation
        peak_hour_0idx = peak_hour - 1  # 0-indexed into sim_loads
        peak_hour_in_month = peak_hour_0idx - m_start_offset + 1  # count of hours from month start to peak

        # Sum of hourly sim-convention loads from month start to peak hour (inclusive)
        hourly_sum_to_peak = float(np.sum(sim_loads[m_start_offset:peak_hour_0idx + 1]))

        prev_predicted_dt = None
        prev_d = 0
        max_d = peak_hour_in_month  # can't extend peak before the month starts

        for d in range(1, max_d + 1):
            # Energy balance per Eq. 4:
            # Q_non_peak = (sum_hourly - d * Q_peak) / non_peak_hours
            non_peak_hours = peak_hour_in_month - d
            peak_energy = d * q_peak

            if non_peak_hours > 0:
                q_non_peak = (hourly_sum_to_peak - peak_energy) / non_peak_hours
            else:
                q_non_peak = 0.0

            # Build hybrid load sequence: [0, prev_months..., non_peak, peak]
            # Times are in hours (matching the hourly simulation's time axis)
            hours_list = [0.0]
            loads_list = [0.0]

            # Previous months: one time step each at net average
            for pm in range(1, peak_month):
                hours_list.append(float(month_hour_offset[pm + 1]))
                loads_list.append(monthly_net_avg_sim[pm])

            # Peak month: non-peak period then peak period
            peak_start_hour = peak_hour - d  # hour when peak period starts
            if non_peak_hours > 0:
                hours_list.append(float(peak_start_hour))
                loads_list.append(q_non_peak)

            hours_list.append(float(peak_hour))
            loads_list.append(q_peak)

            # Simulate this hybrid sequence
            hour_arr = np.array(hours_list)
            load_arr = np.array(loads_list)
            delta_t = self.simulate_hourly(
                hour_arr, load_arr, g_sts, resist_bh, two_pi_k, ts
            )
            predicted_dt = delta_t[-1]

            # Check if predicted peak temp has reached/exceeded the target
            if is_cooling:
                exceeded = predicted_dt >= target_dt
            else:
                exceeded = predicted_dt <= target_dt

            if exceeded:
                # Interpolate between previous and current duration
                if prev_predicted_dt is not None and prev_d > 0:
                    frac = (target_dt - prev_predicted_dt) / (
                        predicted_dt - prev_predicted_dt
                    )
                    return prev_d + frac
                else:
                    return float(d)

            prev_predicted_dt = predicted_dt
            prev_d = d

        # If we exhausted all durations without exceeding, return max tried
        return float(max_d)

    # -----------------------------------------------------------------
    # Step 7: Construct hybrid time step load/hour arrays
    # -----------------------------------------------------------------
    def _process_month_loads(self) -> None:
        """Build the load, hour, and step_func_load arrays for simulation.

        Output format matches HybridLoad so GHE.simulate() works unchanged:
        - self.load: load values in kW (rejection positive, extraction negative)
        - self.hour: cumulative hours from start of simulation
        - self.step_func_load: load step changes (load[i] - load[i-1])

        Non-peak months get a single time step at the net average load.
        Peak months get time steps for: non-peak, peak(s), non-peak.

        Properly handles multi-year simulations and leap years by
        precomputing month boundaries using the actual calendar year
        for each simulated month.
        """
        # Peak hour offsets within their BASE calendar month (0-based from
        # month start). These come from the single-year hourly simulation
        # (steps 1-6) and are reused for every repetition of that month.
        month_hour_offset_base = [0] * 14
        for m in range(1, 14):
            if m <= 12:
                month_hour_offset_base[m] = sum(
                    HRS_IN_DAY * self.days_in_month[j] for j in range(1, m)
                )
            else:
                month_hour_offset_base[13] = sum(
                    HRS_IN_DAY * self.days_in_month[j] for j in range(1, 13)
                )

        peak_cl_hour_in_month = [0] * 13
        peak_hl_hour_in_month = [0] * 13
        for m in range(1, 13):
            if self.monthly_max_dt_hour[m] > 0:
                peak_cl_hour_in_month[m] = self.monthly_max_dt_hour[m] - month_hour_offset_base[m]
            if self.monthly_min_dt_hour[m] > 0:
                peak_hl_hour_in_month[m] = self.monthly_min_dt_hour[m] - month_hour_offset_base[m]

        cooling_peak_set = set(self.peak_cooling_months)
        heating_peak_set = set(self.peak_heating_months)

        # Extend monthly data arrays for multi-year simulation
        for i in range(self.start_month, self.end_month + 1):
            if i > 12:
                mi = ((i - 1) % 12) + 1
                self.monthly_cl.append(self.monthly_cl[mi])
                self.monthly_hl.append(self.monthly_hl[mi])
                self.monthly_peak_cl.append(self.monthly_peak_cl[mi])
                self.monthly_peak_hl.append(self.monthly_peak_hl[mi])
                self.monthly_peak_cl_duration.append(self.monthly_peak_cl_duration[mi])
                self.monthly_peak_hl_duration.append(self.monthly_peak_hl_duration[mi])

        # Precompute month boundaries for all simulated months,
        # using the actual calendar year for leap year correctness.
        total_months = self.end_month
        month_num_hours = [0] * (total_months + 1)  # index 0 unused
        for i in range(1, total_months + 1):
            year_idx = (i - 1) // 12
            cal_month = ((i - 1) % 12) + 1
            year = self.years[year_idx] if year_idx < len(self.years) else self.years[-1]
            month_num_hours[i] = monthrange(year, cal_month)[1] * HRS_IN_DAY

        # Cumulative hour boundaries (fmh is 1-indexed to match existing convention)
        fmh_arr = [0] * (total_months + 1)
        lmh_arr = [0] * (total_months + 1)
        cumulative = 0
        for i in range(1, total_months + 1):
            fmh_arr[i] = cumulative + 1
            cumulative += month_num_hours[i]
            lmh_arr[i] = cumulative

        # Start arrays with zero load before simulation
        self.load = np.append(self.load, 0)
        last_zero_hour = fmh_arr[self.start_month] - 1
        self.hour = np.append(self.hour, last_zero_hour)

        for i in range(self.start_month, self.end_month + 1):
            # Map to base calendar month (1-12) for peak type lookup
            mi = ((i - 1) % 12) + 1

            month_hours = month_num_hours[i]
            fmh = fmh_arr[i]
            lmh = lmh_arr[i]

            has_cooling_peak = mi in cooling_peak_set
            has_heating_peak = mi in heating_peak_set

            if not has_cooling_peak and not has_heating_peak:
                # Non-peak month: single time step at net average
                if month_hours > 0:
                    month_rate = (self.monthly_cl[i] - self.monthly_hl[i]) / month_hours
                else:
                    month_rate = 0.0
                self.load = np.append(self.load, month_rate)
                self.hour = np.append(self.hour, lmh)
            else:
                # Peak month: build time steps around peak(s)
                d_cl = self.monthly_peak_cl_duration[i] if has_cooling_peak else 0.0
                d_hl = self.monthly_peak_hl_duration[i] if has_heating_peak else 0.0

                # Energy balance: net monthly energy = sum of all period energies
                # Convention: rejection positive, extraction negative
                peak_cl_energy = d_cl * self.monthly_peak_cl[i] if has_cooling_peak else 0.0
                peak_hl_energy = d_hl * self.monthly_peak_hl[i] if has_heating_peak else 0.0

                non_peak_hours = month_hours - d_cl - d_hl
                month_net_energy = self.monthly_cl[i] - self.monthly_hl[i]

                if non_peak_hours > 0:
                    # Eq. 4: adjust non-peak load for energy conservation
                    month_rate = (
                        month_net_energy - peak_cl_energy + peak_hl_energy
                    ) / non_peak_hours
                else:
                    month_rate = 0.0

                # Build peak events: (center_hour_from_month_start, type, load, duration)
                events = []
                if has_cooling_peak:
                    center = peak_cl_hour_in_month[mi]
                    events.append((center, "cl", self.monthly_peak_cl[i], d_cl))
                if has_heating_peak:
                    center = peak_hl_hour_in_month[mi]
                    events.append((center, "hl", -self.monthly_peak_hl[i], d_hl))

                # Sort by center hour so peaks are placed chronologically
                events.sort(key=lambda x: x[0])

                # Place peaks within the month
                cursor = fmh  # current hour position
                for center, _peak_type, peak_load, duration in events:
                    # Peak centered on its temperature hour
                    peak_first_hour = fmh + center - duration / 2.0
                    peak_last_hour = peak_first_hour + duration

                    # Clamp to month boundaries
                    if peak_first_hour < fmh:
                        peak_first_hour = fmh
                        peak_last_hour = fmh + duration
                    if peak_last_hour > lmh:
                        peak_last_hour = lmh
                        peak_first_hour = lmh - duration

                    # Non-peak period before this peak
                    if peak_first_hour > cursor:
                        self.load = np.append(self.load, month_rate)
                        self.hour = np.append(self.hour, peak_first_hour)

                    # Peak period
                    self.load = np.append(self.load, peak_load)
                    self.hour = np.append(self.hour, peak_last_hour)

                    cursor = peak_last_hour

                # Non-peak period after last peak (rest of month)
                if cursor < lmh:
                    self.load = np.append(self.load, month_rate)
                    self.hour = np.append(self.hour, lmh)

        # Build step function loads
        n = self.hour.size
        for i in range(1, n):
            step_load = self.load[i] - self.load[i - 1]
            self.step_func_load = np.append(self.step_func_load, step_load)