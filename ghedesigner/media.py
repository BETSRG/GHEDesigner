from math import cos, pi, sin

from pygfunction.media import Fluid

from ghedesigner.enums import FluidType


class GHEFluid(Fluid):

    def __init__(self, fluid_str: str, percent: float, temperature: float = 20):

        fluid_str = fluid_str.upper()
        if fluid_str == FluidType.ETHYLALCOHOL.name:
            self.fluid_type = FluidType.ETHYLALCOHOL
        elif fluid_str == FluidType.ETHYLENEGLYCOL.name:
            self.fluid_type = FluidType.ETHYLENEGLYCOL
        elif fluid_str == FluidType.METHYLALCOHOL.name:
            self.fluid_type = FluidType.METHYLALCOHOL
        elif fluid_str == FluidType.PROPYLENEGLYCOL.name:
            self.fluid_type = FluidType.PROPYLENEGLYCOL
        elif fluid_str == FluidType.WATER.name:
            self.fluid_type = FluidType.WATER
        else:
            raise ValueError(f"FluidType \"{fluid_str}\" not implemented")

        fluid_map = {
            FluidType.ETHYLALCOHOL.name: "MEA",
            FluidType.ETHYLENEGLYCOL.name: "MEG",
            FluidType.METHYLALCOHOL.name: "MMA",
            FluidType.PROPYLENEGLYCOL.name: "MPG",
            FluidType.WATER.name: "WATER"
        }

        super().__init__(fluid_map[fluid_str], percent, temperature)
        self.concentration_percent = percent
        self.temperature = temperature

    def to_input(self):
        return {'fluid_name': self.fluid_type.name,
                'concentration_percent': self.concentration_percent,
                'temperature': self.temperature}


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

    def to_input(self) -> dict:
        return {'conductivity': self.k, 'rho_cp': self.rhoCp}


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
        if type(self.r_in) is float:
            output["pipe_inner_diameter"] = str(self.r_in * 2.0)
            output["pipe_outer_diameter"] = str(self.r_out * 2.0)
        else:
            output["pipe_inner_diameters"] = str([x * 2.0 for x in self.r_in])
            output["pipe_outer_diameters"] = str([x * 2.0 for x in self.r_out])
        output['shank_spacing_pipe_to_pipe'] = {'value': self.s, 'units': 'm'}
        output['pipe_roughness'] = {'value': self.roughness, 'units': 'm'}
        output['number_of_pipes'] = self.n_pipes
        return output

    @staticmethod
    def place_pipes(s, r_out, n_pipes):
        """Positions pipes in an axis-symmetric configuration."""
        shank_space = s / 2 + r_out
        dt = pi / float(n_pipes)
        pos = [(0.0, 0.0) for _ in range(2 * n_pipes)]
        for i in range(n_pipes):
            pos[2 * i] = (shank_space * cos(2.0 * i * dt + pi), shank_space * sin(2.0 * i * dt + pi))
            pos[2 * i + 1] = (shank_space * cos(2.0 * i * dt + pi + dt), shank_space * sin(2.0 * i * dt + pi + dt))
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

    def to_input(self) -> dict:
        return {'conductivity': self.k, 'rho_cp': self.rhoCp, 'undisturbed_temp': self.ugt}
