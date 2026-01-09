import numpy as np

from ghedesigner.constants import TWO_PI


class SystemSolver:
    def __init__(self, fluid_params, geo_params, pipe_system, beta):
        self.cp = fluid_params['cp']
        self.rho = fluid_params['rho']
        self.L_sup = geo_params['L_sup']
        self.L_ret = geo_params['L_ret']
        self.L_ghx = geo_params['L_ghx']

        # Horizontal Pipe Physics (Linearized)
        self.pipe = pipe_system
        self.two_pi_k = TWO_PI * self.pipe.soil.k
        self.pipe_conductance = 1.0 / beta  # [W/(m.K)]

        # History Storage for Horizontal Pipes
        self.history_T_sup = []
        self.history_T_ret = []

    def solve_timestep(self, flow_rate, t_ground, h_n, c_n, r1, r2, horiz_factors):
        """
        Solves the coupled 6-node system for one time step.
        """
        c_f = flow_rate * self.rho * self.cp  # [W/K]

        # 1. Calculate Horizontal Pipe History Intercepts [W/m]
        b_sup = self._calculate_horiz_history(self.history_T_sup, t_ground, horiz_factors)
        b_ret = self._calculate_horiz_history(self.history_T_ret, t_ground, horiz_factors)

        # 2. Pre-calculate Pipe Conductance Terms (G = L/R) [W/K]
        g_sup = self.L_sup * self.pipe_conductance
        g_ret = self.L_ret * self.pipe_conductance

        # 3. Build Matrix A (6x6) and Vector B (6)
        # Unknown Vector x = [q_hp, EFT, ExFT, T_ghx_in, T_ghx_out, T_mean]
        a = np.zeros((6, 6))
        b = np.zeros(6)

        # Eq 1: Heat Pump Linear Model
        # q_hp - r1*EFT = r2
        a[0, 0] = 1.0
        a[0, 1] = -r1
        b[0] = r2

        # Eq 2: Heat Pump Energy Balance
        # q_hp + C_f*EFT - C_f*ExFT = 0
        a[1, 0] = 1.0
        a[1, 1] = c_f
        a[1, 2] = -c_f

        # Eq 3: Supply Pipe Physics (ExFT -> T_ghx_in)
        # Balance: (C_f - G)*ExFT - C_f*T_in = -G*Tg + L*b
        # --- FIX: Dimensionally correct RHS ---
        a[2, 2] = c_f - g_sup
        a[2, 3] = -c_f
        b[2] = -(g_sup * t_ground) + (self.L_sup * b_sup)

        # Eq 4: GHX Mean Temperature Definition
        # T_mean - 0.5*T_in - 0.5*T_out = 0
        a[3, 3] = -0.5
        a[3, 4] = -0.5
        a[3, 5] = 1.0

        # Eq 5: GHX Physics (Lecture 11)
        # T_mean - (C_n/L)*q_hp = H_n
        coeff_q_hp = c_n / self.L_ghx
        a[4, 0] = -coeff_q_hp
        a[4, 5] = 1.0
        b[4] = h_n

        # Eq 6: Return Pipe Physics (T_ghx_out -> EFT)
        # Balance: -C_f*EFT + (C_f - G)*T_out = -G*Tg + L*b
        # --- FIX: Dimensionally correct RHS ---
        a[5, 1] = -c_f
        a[5, 4] = c_f - g_ret
        b[5] = -(g_ret * t_ground) + (self.L_ret * b_ret)

        # 4. Solve System
        x = np.linalg.solve(a, b)

        # Unpack Results
        exft = x[2]
        t_ghx_out = x[4]

        # Update History for pipes
        self.history_T_sup.append(exft)
        self.history_T_ret.append(t_ghx_out)

        return x

    def _calculate_horiz_history(self, t_history, t_ground, response_factors):
        """
        Convolves temperature history with response factors.
        Returns flux intercept 'b' in [W/m].
        """
        n = len(t_history)
        if n == 0:
            return 0.0

        if len(response_factors) < n:
            n = len(response_factors)
            t_history = t_history[-n:]

        # Convolve Delta_T with Response Factors
        full_t = np.concatenate(([t_ground], t_history))
        delta_ts = np.diff(full_t)
        relevant_q = response_factors[:n]
        history_sum = np.dot(delta_ts[::-1], relevant_q)

        # Calculate 'b' intercept for y = mx + b
        # b = (Flux_History) - (Slope * Delta_T_Last)
        term_1 = self.two_pi_k * history_sum
        term_2 = self.pipe_conductance * (t_history[-1] - t_ground)

        return term_1 - term_2
