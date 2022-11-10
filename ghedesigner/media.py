import numpy as np


class ThermalProperty:
    def __init__(self, k, rho_cp):
        self.k = k  # Thermal conductivity (W/m.K)
        self.rhoCp = rho_cp  # Volumetric heat capacity (J/K.m3)

    def as_dict(self) -> dict:
        output = dict()
        output['type'] = str(self.__class__)
        output['thermal_conductivity'] = {'value': self.k, 'units': 'W/m-K'}
        output['volumetric_heat_capacity'] = {'value': self.rhoCp, 'units': 'J/K-m3'}
        return output


class Grout(ThermalProperty):
    pass


class Pipe(ThermalProperty):
    def __init__(self, pos, r_in, r_out, s, roughness, k, rho_cp):
        # Make variables from ThermalProperty available to Pipe
        super().__init__(k, rho_cp)

        # Pipe specific parameters
        self.pos = pos  # Pipe positions either a list of tuples or tuple
        self.r_in = r_in  # Pipe inner radius (m) can be a float or list
        self.r_out = r_out  # Pipe outer radius (m) can be a float or list
        self.s = s  # Center pipe to center pipe shank spacing
        self.roughness = roughness  # Pipe roughness (m)
        if type(pos) is list:
            self.n_pipes = int(len(pos) / 2)  # Number of pipes
        else:
            self.n_pipes = 1

    def as_dict(self) -> dict:
        output = dict()
        output['base'] = super().as_dict()
        output['pipe_center_positions'] = str(self.pos)
        radius = 'radius' if type(self.r_in) is float else 'radii'
        output[f"pipe_inner_{radius}"] = str(self.r_in)
        output[f"pipe_outer_{radius}"] = str(self.r_out)
        output['shank_spacing_pipe_to_pipe'] = {'value': self.s, 'units': 'm'}
        output['pipe_roughness'] = {'value': self.roughness, 'units': 'm'}
        output['number_of_pipes'] = self.n_pipes
        return output

    @staticmethod
    def place_pipes(s, r_out, n_pipes):
        """Positions pipes in an axis-symmetric configuration."""
        shank_space = s / 2 + r_out
        pi = np.pi
        dt = pi / float(n_pipes)
        pos = [(0.0, 0.0) for _ in range(2 * n_pipes)]
        for i in range(n_pipes):
            pos[2 * i] = (
                shank_space * np.cos(2.0 * i * dt + pi),
                shank_space * np.sin(2.0 * i * dt + pi),
            )
            pos[2 * i + 1] = (
                shank_space * np.cos(2.0 * i * dt + pi + dt),
                shank_space * np.sin(2.0 * i * dt + pi + dt),
            )
        return pos


class Soil(ThermalProperty):
    def __init__(self, k, rho_cp, ugt):
        # Make variables from ThermalProperty available to Pipe
        ThermalProperty.__init__(self, k, rho_cp)

        # Soil specific parameters
        self.ugt = ugt

    def as_dict(self) -> dict:
        output = super().as_dict()
        output['undisturbed_ground_temperature'] = {'value': self.ugt, 'units': 'C'}
        return output


class SimulationParameters:
    def __init__(
            self,
            start_month,
            end_month,
            max_entering_fluid_temp_allow,
            min_entering_fluid_temp_allow,
            max_height,
            min_height,
    ):
        # Simulation parameters not found in other objects
        # ------------------------------------------------
        # Simulation start month and end month
        self.start_month = start_month
        self.end_month = end_month
        # Maximum and minimum allowable fluid temperatures
        self.max_EFT_allowable = max_entering_fluid_temp_allow  # degrees Celsius
        self.min_EFT_allowable = min_entering_fluid_temp_allow  # degrees Celsius
        # Maximum and minimum allowable heights
        self.max_Height = max_height  # in meters
        self.min_Height = min_height  # in meters

    def as_dict(self) -> dict:
        output = dict()
        output['type'] = str(self.__class__)
        output['start_month'] = self.start_month
        output['end_month'] = self.end_month
        output['max_eft_allowable'] = {'value': self.max_EFT_allowable, 'units': 'C'}
        output['min_eft_allowable'] = {'value': self.min_EFT_allowable, 'units': 'C'}
        output['maximum_height'] = {'value': self.max_Height, 'units': 'm'}
        output['minimum_height'] = {'value': self.min_Height, 'units': 'm'}
        return output
