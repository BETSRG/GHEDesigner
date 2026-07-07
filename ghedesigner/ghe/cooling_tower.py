from pathlib import Path

import numpy as np


class WeatherProcessor:
    """
    Extracts and calculates psychrometric data from EPW or SVEBY weather files.
    Outputs a structured NumPy array containing [Hour, T_db, P_atm, T_wb, Enthalpy]
    for use in evaporative cooling equipment simulations.
    """

    def __init__(self, file_path: Path | str) -> None:
        self.file_path = Path(file_path)
        self.num_timesteps: int = 8760

        self.t_db = np.zeros(self.num_timesteps, dtype=float)
        self.rh = np.zeros(self.num_timesteps, dtype=float)
        self.p_atm = np.zeros(self.num_timesteps, dtype=float)

        self.t_wb = np.zeros(self.num_timesteps, dtype=float)
        self.enthalpy = np.zeros(self.num_timesteps, dtype=float)

    def process(self) -> np.ndarray:
        """Main execution method to parse the file and calculate psychrometrics."""
        file_ext = self.file_path.suffix.lower()

        if file_ext == ".epw":
            self._parse_epw()
        elif file_ext == ".csv":  # Assuming SVEBY format for .csv files
            self._parse_sveby()
        else:
            raise ValueError(f"Unsupported weather file format: {file_ext}")

        self._calc_psychrometrics()

        hours = np.arange(1, self.num_timesteps + 1, dtype=float)
        return np.column_stack((hours, self.t_db, self.p_atm, self.t_wb, self.enthalpy))

    def _parse_epw(self) -> None:
        """Parses an EnergyPlus Weather (EPW) file."""
        # Dry Bulb is col 6, RH is col 8, Atmospheric Station Pressure is col 9 (0-indexed)
        data = np.genfromtxt(self.file_path, delimiter=",", skip_header=8, usecols=(6, 8, 9), invalid_raise=False)

        if data.shape[0] != self.num_timesteps:
            raise ValueError(f"EPW file does not contain exactly {self.num_timesteps} hours of data.")

        self.t_db = data[:, 0]
        self.rh = data[:, 1]
        self.p_atm = data[:, 2]

    def _parse_sveby(self) -> None:
        """Parses a SVEBY CSV weather file."""
        # Col 6 is T_db and Col 7 is RH (0-indexed).

        data = np.genfromtxt(self.file_path, delimiter=";", skip_header=3, usecols=(6, 7), invalid_raise=False)

        if data.shape[0] != self.num_timesteps:
            raise ValueError(f"SVEBY file contains {data.shape[0]} hours, expected {self.num_timesteps}.")

        self.t_db = data[:, 0]
        self.rh = data[:, 1]

        # Fallback for Atmospheric Pressure
        # SVEBY files omit pressure.
        # Fill the array with standard sea-level pressure (101325 Pa).
        self.p_atm = np.full(self.num_timesteps, 101325.0, dtype=float)

    def _calc_psychrometrics(self) -> None:
        """Vectorized calculation of moist air properties."""

        # 1. Saturation Vapor Pressure (P_ws) in Pascals
        # Using the Magnus-Tetens formula
        p_ws = 610.78 * np.exp((17.27 * self.t_db) / (self.t_db + 237.3))

        # 2. Actual Vapor Pressure (P_w) in Pascals
        # EPW RH is 0-100, so we divide by 100 here
        p_w = (self.rh / 100.0) * p_ws

        # 3. Humidity Ratio (W) in kg_water / kg_dry_air
        # Clip the denominator to prevent division by zero in extreme pressure scenarios
        w = 0.621945 * (p_w / np.clip(self.p_atm - p_w, 1e-5, None))

        # 4. Specific Enthalpy (h) in kJ/kg
        self.enthalpy = 1.006 * self.t_db + w * (2501.0 + 1.86 * self.t_db)

        # 5. Wet-Bulb Temperature (T_wb) in Celsius (Stull 2011 Approximation)
        # Stull requires RH in standard percentage format (0-100)
        self.t_wb = (
            self.t_db * np.arctan(0.151977 * np.sqrt(self.rh + 8.313659))
            + np.arctan(self.t_db + self.rh)
            - np.arctan(self.rh - 1.676331)
            + 0.00391838 * np.power(self.rh, 1.5) * np.arctan(0.023101 * self.rh)
            - 4.686035
        )
