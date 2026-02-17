from calendar import monthrange

import numpy as np

from ghedesigner.constants import HRS_IN_DAY, MONTHS_IN_YEAR, SEC_IN_HR, TWO_PI
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

    All monthly arrays use 0-based indexing: index 0 = January, 11 = December.
    """

    # Normalization target: 40 W/m * 100 m = 4000 W
    NORM_LOAD_W = 40.0 * 100.0  # 4000 W
    NORM_BOREHOLE_H = 100.0  # meters - borehole length for normalized simulation
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
        num_years = max(1, (end_month + 11) // MONTHS_IN_YEAR)
        # TODO - in the input file, the user should be able to specify what year,
        #  that way if using multiyear loads, leap years are correctly accounted for)
        base_year = 2026
        self.years = [base_year + y for y in range(num_years)]

        # Days in each month for the first year (0-indexed: 0=Jan, 11=Dec).
        # Steps 1-6 operate on a single year of hourly data (8760 or 8784 hours).
        self.days_in_month = [monthrange(self.years[0], m + 1)[1] for m in range(MONTHS_IN_YEAR)]

        # --- Step 1: Normalize the loads ---
        self.normalized_loads = self.normalize_loads(self.raw_loads)

        # --- Step 2: Monthly load metrics (on original loads, for later use) ---
        self.monthly_cl: list[float]
        self.monthly_hl: list[float]
        self.monthly_peak_cl: list[float]
        self.monthly_peak_hl: list[float]
        self.monthly_avg_cl: list[float]
        self.monthly_avg_hl: list[float]
        self.split_loads_by_month()

        # --- Step 3: Hourly simulation on normalized loads ---
        self.hourly_delta_t = self._run_hourly_simulation()

        # --- Step 4: Identify peak temperature months ---
        # Per-month: max delta_T (cooling/rejection peak), min delta_T (heating/extraction peak)
        # and the hour-of-year at which each occurs (0-indexed: 0=Jan, 11=Dec)
        self.monthly_max_dt = [0.0] * MONTHS_IN_YEAR
        self.monthly_min_dt = [0.0] * MONTHS_IN_YEAR
        self.monthly_max_dt_hour = [0] * MONTHS_IN_YEAR
        self.monthly_min_dt_hour = [0] * MONTHS_IN_YEAR
        self._find_monthly_peak_temperatures()

        # --- Step 5: Select top 4 peak months for cooling and heating ---
        self.peak_cooling_months: list[int] = []  # months with highest max delta_T (0-indexed)
        self.peak_heating_months: list[int] = []  # months with lowest min delta_T (0-indexed)
        self._select_peak_months()

        # --- Step 6: Find peak durations ---
        self.monthly_peak_cl_duration = [0.0] * MONTHS_IN_YEAR
        self.monthly_peak_hl_duration = [0.0] * MONTHS_IN_YEAR
        self.monthly_peak_cl_load_factor = [1.0] * MONTHS_IN_YEAR
        self.monthly_peak_hl_load_factor = [1.0] * MONTHS_IN_YEAR
        self._find_peak_durations()

        # --- Step 7: Construct hybrid time step arrays ---
        self.load = np.array(0)
        self.hour = np.array(0)
        self.step_label: list[str] = [""]  # label for each time step (e.g. "CL_Jul", "HL_Jan")
        self.hybrid_dt: np.ndarray  # predicted delta_T at each hybrid time step
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

        These are computed on the original (non-normalized) loads in W,
        and will be used later when constructing the final hybrid time
        step arrays with actual load magnitudes.
        """
        n = MONTHS_IN_YEAR

        self.monthly_cl = [0.0] * n  # total cooling (rejection) Wh
        self.monthly_hl = [0.0] * n  # total heating (extraction) Wh
        self.monthly_peak_cl = [0.0] * n  # peak cooling W
        self.monthly_peak_hl = [0.0] * n  # peak heating W
        self.monthly_avg_cl = [0.0] * n  # average cooling W
        self.monthly_avg_hl = [0.0] * n  # average heating W

        hours_in_previous_months = 0
        for m in range(n):
            hours_in_month = HRS_IN_DAY * self.days_in_month[m]
            month_loads = self.raw_loads[hours_in_previous_months : hours_in_previous_months + hours_in_month]

            # Split into rejection (negative raw = cooling) and extraction (positive raw = heating)
            month_rejection = [abs(x) if x < 0 else 0.0 for x in month_loads]
            month_extraction = [x if x >= 0 else 0.0 for x in month_loads]

            self.monthly_cl[m] = sum(month_rejection)
            self.monthly_hl[m] = sum(month_extraction)
            self.monthly_peak_cl[m] = max(month_rejection)
            self.monthly_peak_hl[m] = max(month_extraction)
            self.monthly_avg_cl[m] = self.monthly_cl[m] / hours_in_month if hours_in_month > 0 else 0.0
            self.monthly_avg_hl[m] = self.monthly_hl[m] / hours_in_month if hours_in_month > 0 else 0.0

            hours_in_previous_months += hours_in_month

    # -----------------------------------------------------------------
    # Hourly simulation utility
    # -----------------------------------------------------------------
    @staticmethod
    def simulate_hourly(hour_time, q, g_sts, resist_bh, two_pi_k, ts, h):
        """Hourly fluid temperature simulation using g-function superposition.

        Based on Chapter 2 of Advances in Ground Source Heat Pumps.
        Sign convention: positive q = heat extraction from ground
        → negative (Tf_ave - Tg0), i.e. fluid colder than ground.

        :param hour_time: array of time values (hours)
        :param q: array of loads at each time step (W), q[0]=0
        :param g_sts: scipy interp1d for short-time-step g-function
        :param resist_bh: effective borehole resistance (m.K/W)
        :param two_pi_k: 2*pi*k_soil (W/m.K)
        :param ts: characteristic time for STS g-function (s)
        :param h: borehole length (m) - loads are divided by this to get W/m
        :return: list of (Tf_ave - Tg0) for each time step
        """
        q_dt = np.hstack(q[1:] - q[:-1])
        delta_t_fluid = [0.0]
        for n in range(1, len(hour_time)):
            _time = hour_time[n] - hour_time[0:n]
            # Mask out zero-time entries to avoid log(0) = -inf.
            # Their q_dt is always 0, so they contribute nothing.
            nonzero = _time > 0
            if np.any(nonzero):
                g_values = np.zeros(n)
                g_values[nonzero] = g_sts(np.log((_time[nonzero] * SEC_IN_HR) / ts))
                delta_tg0_tb_i = (q_dt[0:n] / h / two_pi_k).dot(g_values)
            else:
                delta_tg0_tb_i = 0.0
            delta_tb_tf = q[n] / h * resist_bh
            # Eq 2.12: delta_Tb = (q'/2πk) * g, where q' = q/H (W/m)
            # Eq 2.13: Tf = Tb + q' * Rb
            delta_tf_tg0 = - delta_tg0_tb_i - delta_tb_tf

            delta_t_fluid.append(delta_tf_tg0)
        return delta_t_fluid

    # -----------------------------------------------------------------
    # Step 3: Run hourly simulation on normalized loads
    # -----------------------------------------------------------------
    def _run_hourly_simulation(self) -> list:
        """Run a full-year hourly simulation using normalized loads.

        Uses the single-borehole STS g-function to compute (Tf_ave - Tg0)
        at each hour. The normalized loads ensure results are independent
        of borefield size/configuration.

        :return: List of 8761 (Tf_ave - Tg0) values (index 0 = hour 0 = 0.0)
        """
        ts = self.radial_numerical.t_s
        two_pi_k = TWO_PI * self.bhe.soil.k
        resist_bh = self.bhe.calc_effective_borehole_resistance()
        g_sts = self.radial_numerical.g_sts

        n_hours = len(self.normalized_loads)
        hour_time = np.arange(n_hours + 1)  # 0, 1, 2, ..., 8760

        # Prepend 0 load at hour 0
        # positive = extraction → negative (Tf_ave - Tg0)
        q = np.hstack((0.0, self.normalized_loads))

        return self.simulate_hourly(
            hour_time, q, g_sts, resist_bh, two_pi_k, ts, self.NORM_BOREHOLE_H
        )

    # -----------------------------------------------------------------
    # Step 4: Find monthly peak temperatures and their hours
    # -----------------------------------------------------------------
    def _find_monthly_peak_temperatures(self) -> None:
        """For each month, find the max and min (Tf_ave - Tg0) and the
        hour-of-year at which each occurs.

        Max (Tf_ave - Tg0) corresponds to peak cooling/rejection (hottest fluid).
        Min (Tf_ave - Tg0) corresponds to peak heating/extraction (coldest fluid).
        """
        hours_in_previous_months = 0
        for m in range(MONTHS_IN_YEAR):
            hours_in_month = HRS_IN_DAY * self.days_in_month[m]

            # Slice delta_T for this month (hours are 1-indexed in the sim)
            start_hr = hours_in_previous_months + 1
            end_hr = hours_in_previous_months + hours_in_month + 1
            month_dt = self.hourly_delta_t[start_hr:end_hr]

            if len(month_dt) == 0:
                hours_in_previous_months += hours_in_month
                continue

            # Max (Tf_ave - Tg0): peak rejection/cooling temperature
            # Only record if positive -- fluid warmer than ground means
            # actual rejection activity in this month.
            max_idx = int(np.argmax(month_dt))
            if month_dt[max_idx] > 0.0:
                self.monthly_max_dt[m] = month_dt[max_idx]
                self.monthly_max_dt_hour[m] = hours_in_previous_months + max_idx + 1

            # Min (Tf_ave - Tg0): peak extraction/heating temperature
            # Only record if negative -- fluid colder than ground means
            # actual extraction activity in this month.
            min_idx = int(np.argmin(month_dt))
            if month_dt[min_idx] < 0.0:
                self.monthly_min_dt[m] = month_dt[min_idx]
                self.monthly_min_dt_hour[m] = hours_in_previous_months + min_idx + 1

            hours_in_previous_months += hours_in_month

    # -----------------------------------------------------------------
    # Step 5: Select top N peak months for cooling and heating
    # -----------------------------------------------------------------
    def _select_peak_months(self) -> None:
        """Identify the top NUM_PEAK_MONTHS months with most extreme
        (Tf_ave - Tg0): most positive for cooling, most negative for heating.

        Only months with a positive max (Tf_ave - Tg0) qualify as cooling
        peaks, and only months with a negative min qualify as heating peaks.
        This means fewer than NUM_PEAK_MONTHS may be selected when the load
        profile is predominantly one-sided.

        Only these months will get peak time steps in the hybrid simulation;
        all other months are treated as single average-load time steps.
        """
        n = self.NUM_PEAK_MONTHS

        # Cooling peaks: months with most positive max (Tf_ave - Tg0) = most rejection
        month_max_pairs = [(m, self.monthly_max_dt[m]) for m in range(MONTHS_IN_YEAR) if self.monthly_max_dt[m] > 0.0]
        month_max_pairs.sort(key=lambda x: x[1], reverse=True)
        self.peak_cooling_months = [m for m, _ in month_max_pairs[:n]]

        # Heating peaks: months with most negative min (Tf_ave - Tg0) = most extraction
        month_min_pairs = [(m, self.monthly_min_dt[m]) for m in range(MONTHS_IN_YEAR) if self.monthly_min_dt[m] < 0.0]
        month_min_pairs.sort(key=lambda x: x[1])
        self.peak_heating_months = [m for m, _ in month_min_pairs[:n]]

    # -----------------------------------------------------------------
    # Step 6: Find required peak durations
    # -----------------------------------------------------------------
    def _find_peak_durations(self) -> None:
        """Find peak durations for all peak months by iterative matching.

        For each peak month, iteratively increases the peak duration from
        1 hour until the hybrid simulation's peak (Tf_ave - Tg0) matches the
        hourly simulation's target. Then interpolates to get the
        fractional duration.

        Uses normalized loads (positive = extraction).
        """
        # Normalized loads: positive = extraction
        sim_loads = np.array(self.normalized_loads)

        # Cumulative hour offsets for month boundaries (0-indexed months).
        # month_hour_offset[m] = total hours before month m starts (0-based into loads array)
        # month_hour_offset[0] = 0, month_hour_offset[12] = 8760
        month_hour_offset = [0] * (MONTHS_IN_YEAR + 1)
        for m in range(MONTHS_IN_YEAR):
            month_hour_offset[m + 1] = month_hour_offset[m] + HRS_IN_DAY * self.days_in_month[m]

        # Monthly net average loads (positive = extraction)
        monthly_net_avg_sim = [0.0] * MONTHS_IN_YEAR
        for m in range(MONTHS_IN_YEAR):
            m_start = month_hour_offset[m]
            m_end = month_hour_offset[m + 1]
            if m_end > m_start:
                monthly_net_avg_sim[m] = float(np.mean(sim_loads[m_start:m_end]))

        # Simulation parameters
        ts = self.radial_numerical.t_s
        two_pi_k = TWO_PI * self.bhe.soil.k
        resist_bh = self.bhe.calc_effective_borehole_resistance()
        g_sts = self.radial_numerical.g_sts

        # Process cooling peaks (months with most positive max Tf_ave - Tg0)
        for m in self.peak_cooling_months:
            peak_hour = self.monthly_max_dt_hour[m]  # hour-of-year (1-indexed)
            target_dt = self.monthly_max_dt[m]

            # Peak rejection load for this month (negative = rejection)
            m_start = month_hour_offset[m]
            m_end = month_hour_offset[m + 1]
            q_peak = float(np.min(sim_loads[m_start:m_end]))

            dur, lf = self._iterate_duration(
                peak_month=m,
                peak_temp_hour=peak_hour,
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
            self.monthly_peak_cl_duration[m] = dur
            self.monthly_peak_cl_load_factor[m] = lf

        # Process heating peaks (months with most negative min Tf_ave - Tg0)
        for m in self.peak_heating_months:
            peak_hour = self.monthly_min_dt_hour[m]  # hour-of-year (1-indexed)
            target_dt = self.monthly_min_dt[m]

            # Peak extraction load for this month (positive = extraction)
            m_start = month_hour_offset[m]
            m_end = month_hour_offset[m + 1]
            q_peak = float(np.max(sim_loads[m_start:m_end]))

            dur, lf = self._iterate_duration(
                peak_month=m,
                peak_temp_hour=peak_hour,
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
            self.monthly_peak_hl_duration[m] = dur
            self.monthly_peak_hl_load_factor[m] = lf

    def _iterate_duration(
        self,
        peak_month: int,
        peak_temp_hour: int,
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
    ) -> tuple[float, float]:
        """Iteratively find peak duration for one peak month.

        Builds a hybrid load sequence from the start of the year to the
        peak temperature hour, with:
        - Previous months at their net average load
        - Peak month split into non-peak (adjusted avg) + peak load

        Extends duration of peak back in time by 1 hour each iteration until the hybrid
        simulation's peak temperature matches or exceeds the hourly
        simulation's target. Then interpolates for fractional duration.

        Per Eq. 3-4 of Spitler (2024), the non-peak load is adjusted so
        total energy from month start to peak end is conserved.

        If the pre-peak clamp engages and the target temperature still
        isn't reached, a second stage concentrates the same total energy
        into progressively shorter durations (higher peak load) until the
        target is matched.

        :param peak_month: 0-indexed month (0=Jan, 11=Dec)
        :return: (duration, load_factor) where load_factor is the ratio
                 of adjusted peak load to original peak load (1.0 = unchanged)
        """
        m_start_offset = month_hour_offset[peak_month]
        # peak_temp_hour is 1-indexed hour-of-year the peak dt occurs; convert to 0-indexed offset
        # into the loads array for energy summation
        peak_hour_0idx = peak_temp_hour - 1  # 0-indexed into sim_loads
        peak_hour_in_month = peak_hour_0idx - m_start_offset + 1  # count of hours from month start to peak

        # Sum of hourly sim-convention loads from month start to peak hour (inclusive)
        hourly_sum_to_peak = float(np.sum(sim_loads[m_start_offset : peak_hour_0idx + 1]))

        prev_predicted_dt = None
        prev_d = 0 #previous duration of peak
        max_d = peak_hour_in_month  # can't extend peak before the month starts

        for d in range(1, max_d + 1):
            # Energy balance per Eq. 4:
            # Q_non_peak = (sum_hourly - d * Q_peak) / non_peak_hours
            non_peak_hours = peak_hour_in_month - d
            peak_energy = d * q_peak
            q_non_peak = (hourly_sum_to_peak - peak_energy) / non_peak_hours if non_peak_hours > 0 else 0.0

            # Clamp: don't let the pre-peak load switch sign from the peak load
            # UNLESS the hourly data in the pre-peak window actually contains
            # opposite-signed loads that justify the flip.
            # - Heating peak (q_peak > 0): flip is valid if pre-peak has cooling loads
            # - Cooling peak (q_peak < 0): flip is valid if pre-peak has heating loads
            #
            # When clamped, set pre-peak to zero and solve for the fractional
            # peak duration that conserves energy: d_energy = hourly_sum / q_peak.
            # Since the sign flip first occurs at iteration d (and was fine at
            # d-1), d_energy always falls between d-1 and d.
            clamped = False
            effective_d = float(d)
            if q_peak != 0.0 and q_non_peak * q_peak < 0.0:
                pre_peak_slice = sim_loads[m_start_offset : m_start_offset + non_peak_hours]
                if is_cooling:
                    has_opposite = bool(np.any(pre_peak_slice > 0)) if len(pre_peak_slice) > 0 else False
                else:
                    has_opposite = bool(np.any(pre_peak_slice < 0)) if len(pre_peak_slice) > 0 else False

                if not has_opposite:
                    q_non_peak = 0.0
                    effective_d = hourly_sum_to_peak / q_peak
                    effective_d = max(0.0, min(effective_d, float(peak_hour_in_month)))
                    non_peak_hours = peak_hour_in_month - effective_d
                    clamped = True

            # Build hybrid load sequence: [0, prev_months..., non_peak, peak]
            # Times are in hours (matching the hourly simulation's time axis)
            hours_list = [0.0]
            loads_list = [0.0]

            # Previous months: one time step each at net average TODO check this
            for pm in range(peak_month):
                hours_list.append(float(month_hour_offset[pm + 1]))
                loads_list.append(monthly_net_avg_sim[pm])

            # Peak month: non-peak period then peak period
            peak_start_hour = peak_temp_hour - effective_d
            if non_peak_hours > 0:
                hours_list.append(float(peak_start_hour))
                loads_list.append(q_non_peak)

            hours_list.append(float(peak_temp_hour))
            loads_list.append(q_peak)

            # Simulate this hybrid sequence
            hour_arr = np.array(hours_list)
            load_arr = np.array(loads_list)
            delta_t = self.simulate_hourly(
                hour_arr, load_arr, g_sts, resist_bh, two_pi_k, ts, self.NORM_BOREHOLE_H
            )
            predicted_dt = delta_t[-1]

            # Check if predicted (Tf_ave - Tg0) has reached/exceeded the target
            # Cooling: target is positive, predicted grows more positive → exceeded when >=
            # Heating: target is negative, predicted grows more negative → exceeded when <=
            exceeded = predicted_dt >= target_dt if is_cooling else predicted_dt <= target_dt

            if exceeded:
                # Interpolate between previous and current duration
                if prev_predicted_dt is not None and prev_d > 0:
                    frac = (target_dt - prev_predicted_dt) / (predicted_dt - prev_predicted_dt)
                    return prev_d + frac * (effective_d - prev_d), 1.0
                else:
                    return effective_d, 1.0

            # If clamped, try concentrating energy into shorter duration
            if clamped:
                result = self._concentrate_energy(
                    peak_month=peak_month,
                    peak_temp_hour=peak_temp_hour,
                    peak_hour_in_month=peak_hour_in_month,
                    target_dt=target_dt,
                    total_energy=hourly_sum_to_peak,
                    q_peak=q_peak,
                    start_d=effective_d,
                    start_predicted_dt=predicted_dt,
                    monthly_net_avg_sim=monthly_net_avg_sim,
                    month_hour_offset=month_hour_offset,
                    g_sts=g_sts,
                    resist_bh=resist_bh,
                    two_pi_k=two_pi_k,
                    ts=ts,
                    is_cooling=is_cooling,
                )
                return result

            prev_predicted_dt = predicted_dt
            prev_d = effective_d

        # If we exhausted all durations without exceeding, return max tried
        return float(max_d), 1.0

    def _concentrate_energy(
        self,
        peak_month: int,
        peak_temp_hour: int,
        peak_hour_in_month: int,
        target_dt: float,
        total_energy: float,
        q_peak: float,
        start_d: float,
        start_predicted_dt: float,
        monthly_net_avg_sim: list,
        month_hour_offset: list,
        g_sts,
        resist_bh: float,
        two_pi_k: float,
        ts: float,
        is_cooling: bool,
    ) -> tuple[float, float]:
        """Concentrate energy into shorter peak to hit target temperature.

        Called when the pre-peak clamp has engaged and the target temperature
        wasn't reached. Iteratively reduces peak duration by 1 hour while
        increasing the peak load to conserve total energy:
            concentrated_q = total_energy / reduced_d

        The sharper pulse drives the temperature harder, potentially reaching
        the target that the original peak load couldn't.

        :param total_energy: energy to conserve (hourly_sum_to_peak)
        :param q_peak: original peak load (for computing load_factor)
        :param start_d: clamped duration where concentration begins
        :param start_predicted_dt: delta_T at start_d
        :return: (duration, load_factor) where load_factor = concentrated_q / q_peak
        """
        prev_dt = start_predicted_dt
        prev_d = start_d

        for rd in range(int(start_d) - 1, 0, -1):
            concentrated_q = total_energy / rd
            non_peak_hours = peak_hour_in_month - rd

            # Build hybrid sequence: [0, prev_months..., zero_pre_peak, concentrated_peak]
            hours_list = [0.0]
            loads_list = [0.0]

            for pm in range(peak_month):
                hours_list.append(float(month_hour_offset[pm + 1]))
                loads_list.append(monthly_net_avg_sim[pm])

            peak_start_hour = peak_temp_hour - rd
            if non_peak_hours > 0:
                hours_list.append(float(peak_start_hour))
                loads_list.append(0.0)  # pre-peak stays at zero

            hours_list.append(float(peak_temp_hour))
            loads_list.append(concentrated_q)

            hour_arr = np.array(hours_list)
            load_arr = np.array(loads_list)
            delta_t = self.simulate_hourly(
                hour_arr, load_arr, g_sts, resist_bh, two_pi_k, ts, self.NORM_BOREHOLE_H
            )
            predicted_dt = delta_t[-1]

            exceeded = predicted_dt >= target_dt if is_cooling else predicted_dt <= target_dt

            if exceeded:
                # Interpolate between previous and current duration
                frac = (target_dt - prev_dt) / (predicted_dt - prev_dt)
                final_d = prev_d + frac * (rd - prev_d)
                final_d = max(final_d, 0.1)  # avoid division by zero
                load_factor = (total_energy / final_d) / q_peak
                return final_d, load_factor

            prev_dt = predicted_dt
            prev_d = float(rd)

        # Exhausted all reductions — return best effort at shortest duration
        return max(start_d, 1.0), total_energy / (max(start_d, 1.0) * q_peak) if q_peak != 0 else 1.0

    # -----------------------------------------------------------------
    # Step 7: Construct hybrid time step load/hour arrays
    # -----------------------------------------------------------------
    def _process_month_loads(self) -> None:
        """Build the load and hour arrays for simulation.

        Output format matches HybridLoad so GHE.simulate() works unchanged:
        - self.load: load values in W (extraction positive, rejection negative)
        - self.hour: cumulative hours from start of simulation

        Non-peak months get a single time step at the net average load.
        Peak months have three regions:
        - Pre-peak: load computed to conserve energy from cursor to peak end
        - Peak: the peak load for the calibrated duration
        - Post-peak: monthly average load for remaining hours

        Properly handles multi-year simulations and leap years by
        precomputing month boundaries using the actual calendar year
        for each simulated month.
        """
        # Hourly loads (W, extraction positive, rejection negative).
        # Used for energy conservation in pre-peak load computation.
        net_loads_w = np.array(self.raw_loads, dtype=float)

        # Cumulative hour offsets for the base year (used to compute
        # peak hour positions within each calendar month).
        # month_hour_offset_base[m] = hours before month m starts
        # month_hour_offset_base[0] = 0, [12] = 8760
        month_hour_offset_base = [0] * (MONTHS_IN_YEAR + 1)
        for m in range(MONTHS_IN_YEAR):
            month_hour_offset_base[m + 1] = month_hour_offset_base[m] + HRS_IN_DAY * self.days_in_month[m]

        # Peak hour offsets within their calendar month (0-based from month start).
        # These come from the single-year hourly simulation (steps 1-6)
        # and are reused for every repetition of that month.
        peak_cl_hour_in_month = [0] * MONTHS_IN_YEAR
        peak_hl_hour_in_month = [0] * MONTHS_IN_YEAR
        for m in range(MONTHS_IN_YEAR):
            if self.monthly_max_dt_hour[m] > 0:
                peak_cl_hour_in_month[m] = self.monthly_max_dt_hour[m] - month_hour_offset_base[m]
            if self.monthly_min_dt_hour[m] > 0:
                peak_hl_hour_in_month[m] = self.monthly_min_dt_hour[m] - month_hour_offset_base[m]

        cooling_peak_set = set(self.peak_cooling_months)
        heating_peak_set = set(self.peak_heating_months)

        # Extend monthly data arrays for multi-year simulation.
        # Base arrays have 12 entries (indices 0-11). For simulation months
        # beyond 12, append copies from the corresponding base month.
        for i in range(MONTHS_IN_YEAR, self.end_month):
            mi = i % MONTHS_IN_YEAR  # 0-indexed calendar month
            self.monthly_cl.append(self.monthly_cl[mi])
            self.monthly_hl.append(self.monthly_hl[mi])
            self.monthly_peak_cl.append(self.monthly_peak_cl[mi])
            self.monthly_peak_hl.append(self.monthly_peak_hl[mi])
            self.monthly_peak_cl_duration.append(self.monthly_peak_cl_duration[mi])
            self.monthly_peak_hl_duration.append(self.monthly_peak_hl_duration[mi])
            self.monthly_peak_cl_load_factor.append(self.monthly_peak_cl_load_factor[mi])
            self.monthly_peak_hl_load_factor.append(self.monthly_peak_hl_load_factor[mi])

        # Precompute month boundaries for all simulated months,
        # using the actual calendar year for leap year correctness.
        # Arrays are 0-indexed: index 0 = simulation month 1.
        total_months = self.end_month
        month_num_hours = [0] * total_months
        for i in range(total_months):
            year_idx = i // MONTHS_IN_YEAR
            cal_month_1indexed = (i % MONTHS_IN_YEAR) + 1
            year = self.years[year_idx] if year_idx < len(self.years) else self.years[-1]
            month_num_hours[i] = monthrange(year, cal_month_1indexed)[1] * HRS_IN_DAY

        # Cumulative hour boundaries
        fmh_arr = [0.0] * total_months
        lmh_arr = [0.0] * total_months
        cumulative = 0
        for i in range(total_months):
            fmh_arr[i] = cumulative + 1
            cumulative += month_num_hours[i]
            lmh_arr[i] = cumulative

        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        # Start arrays with zero load before simulation
        # start_month is 1-indexed, convert to 0-indexed for array access
        start_idx = self.start_month - 1
        self.load = np.append(self.load, 0)
        last_zero_hour = fmh_arr[start_idx] - 1
        self.hour = np.append(self.hour, last_zero_hour)
        self.step_label.append("")

        for i in range(start_idx, self.end_month):
            # Calendar month index (0-indexed) for peak type lookup
            mi = i % MONTHS_IN_YEAR

            month_hours = month_num_hours[i]
            fmh = fmh_arr[i]
            lmh = lmh_arr[i]

            has_cooling_peak = mi in cooling_peak_set
            has_heating_peak = mi in heating_peak_set

            if not has_cooling_peak and not has_heating_peak:
                # Non-peak month: single time step at net average (extraction positive)
                month_rate = (self.monthly_hl[i] - self.monthly_cl[i]) / month_hours if month_hours > 0 else 0.0
                self.load = np.append(self.load, month_rate)
                self.hour = np.append(self.hour, lmh)
                self.step_label.append("")
            else:
                # Peak month: build time steps around peak(s)
                d_cl = self.monthly_peak_cl_duration[i] if has_cooling_peak else 0.0
                d_hl = self.monthly_peak_hl_duration[i] if has_heating_peak else 0.0

                month_net_energy = self.monthly_hl[i] - self.monthly_cl[i]

                # Build peak events: (peak_temp_hour_in_month, type, load, duration)
                # Extraction positive: cooling/rejection is negative, heating/extraction is positive
                events = []
                if has_cooling_peak:
                    peak_hr = peak_cl_hour_in_month[mi]
                    cl_load = -self.monthly_peak_cl[i] * self.monthly_peak_cl_load_factor[i]
                    events.append((peak_hr, "cl", cl_load, d_cl))
                if has_heating_peak:
                    peak_hr = peak_hl_hour_in_month[mi]
                    hl_load = self.monthly_peak_hl[i] * self.monthly_peak_hl_load_factor[i]
                    events.append((peak_hr, "hl", hl_load, d_hl))

                # Sort by peak temperature hour so peaks are placed chronologically
                events.sort(key=lambda x: x[0])

                # Base-year offset for this calendar month in the hourly array
                base = month_hour_offset_base[mi]

                # Place peaks within the month, tracking consumed energy
                cursor = fmh  # current hour position
                consumed_energy = 0.0  # total energy (kWh) placed so far

                for peak_temp_hr, _peak_type, peak_load, duration in events:
                    # Peak load ends at the peak temperature hour
                    # (duration extends backward in time from the temperature peak)
                    peak_last_hour = fmh + peak_temp_hr
                    peak_first_hour = peak_last_hour - duration

                    # Clamp to month boundaries
                    if peak_first_hour < fmh:
                        peak_first_hour = fmh
                        peak_last_hour = fmh + duration
                    if peak_last_hour > lmh:
                        peak_last_hour = lmh
                        peak_first_hour = lmh - duration

                    # Prevent overlap with previous peak event
                    if peak_first_hour < cursor:
                        peak_first_hour = cursor
                        peak_last_hour = cursor + duration
                        peak_last_hour = min(peak_last_hour, lmh)

                    # Pre-peak load: conserve energy from cursor to peak end
                    # Sum hourly loads (W) from cursor to peak_last_hour
                    cursor_offset = int(cursor - fmh)
                    peak_end_offset = int(peak_last_hour - fmh)
                    energy_to_peak_end = float(np.sum(net_loads_w[base + cursor_offset : base + peak_end_offset]))

                    pre_peak_hours = peak_first_hour - cursor
                    actual_duration = peak_last_hour - peak_first_hour
                    peak_energy = actual_duration * peak_load

                    if pre_peak_hours > 0:
                        q_pre = (energy_to_peak_end - peak_energy) / pre_peak_hours
                        self.load = np.append(self.load, q_pre)
                        self.hour = np.append(self.hour, peak_first_hour)
                        self.step_label.append("")
                        consumed_energy += q_pre * pre_peak_hours

                    # Peak period
                    peak_label = ("CL_" if _peak_type == "cl" else "HL_") + month_names[mi]
                    self.load = np.append(self.load, peak_load)
                    self.hour = np.append(self.hour, peak_last_hour)
                    self.step_label.append(peak_label)
                    consumed_energy += peak_energy

                    cursor = peak_last_hour

                # Post-peak: conserve remaining monthly energy
                if cursor < lmh:
                    post_peak_hours = lmh - cursor
                    q_post = (month_net_energy - consumed_energy) / post_peak_hours if post_peak_hours > 0 else 0.0
                    self.load = np.append(self.load, q_post)
                    self.hour = np.append(self.hour, lmh)
                    self.step_label.append("")

        # Compute predicted delta_T at each hybrid time step using normalized loads
        # self.load is in W; scale to normalized W (peak=4000W) so that
        # hybrid_dt is comparable to hourly_delta_t from _run_hourly_simulation.
        ts = self.radial_numerical.t_s
        two_pi_k = TWO_PI * self.bhe.soil.k
        resist_bh = self.bhe.calc_effective_borehole_resistance()
        g_sts = self.radial_numerical.g_sts
        norm_scale = self.NORM_LOAD_W / np.max(np.abs(self.raw_loads))
        self.hybrid_q_norm_w = self.load * norm_scale  # W -> normalized W
        self.hybrid_dt = np.array(
            self.simulate_hourly(self.hour, self.hybrid_q_norm_w, g_sts, resist_bh, two_pi_k, ts, self.NORM_BOREHOLE_H)
        )
