# Jack C. Cook
# Wednesday, September 8, 2021

import numpy as np


class ThermalProperty:
    def __init__(self, k, rhoCp):
        self.k = k          # Thermal conductivity (W/m.K)
        self.rhoCp = rhoCp  # Volumetric heat capacity (J/K.m3)

    def __repr__(self):
        def justify(category, value):
            return category.ljust(40) + '= ' + value + '\n'

        output = str(self.__class__) + '\n'

        if type(self.k) == float:
            output += justify('Thermal conductivity',
                              str(round(self.k, 4)) + ' (W/m.K)')
        else:
            output += justify('Thermal conductivity',
                              str(self.k) + ' (W/m.K)')
        output += justify('Volumetric heat capacity',
                          str(round(self.rhoCp, 4)) + ' (J/K.m3)')

        return output


class Grout(ThermalProperty):
    def __init__(self, k, rhoCp):
        # Make variables from ThermalProperty available to Grout
        ThermalProperty.__init__(self, k, rhoCp)


class Pipe(ThermalProperty):
    def __init__(self, pos, r_in, r_out, s, eps, k, rhoCp):
        # Make variables from ThermalProperty available to Pipe
        ThermalProperty.__init__(self, k, rhoCp)

        # Pipe specific parameters
        self.pos = pos      # Pipe positions either a list of tuples or tuple
        self.r_in = r_in    # Pipe inner radius (m) can be float or list
        self.r_out = r_out  # Pipe outer radius (m) can be float or list
        self.s = s          # Center pipe to center pipe shank spacing
        self.eps = eps      # Pipe roughness (m)
        if type(pos) is list:
            self.n_pipes = int(len(pos) / 2)  # Number of pipes
        else:
            self.n_pipes = 1

    def __repr__(self):
        def justify(category, value):
            return category.ljust(40) + '= ' + value + '\n'

        output = ThermalProperty.__repr__(self)

        output += justify('Pipe Positions (Center of pipes)', str(self.pos))
        if type(self.r_in) == float:
            output += justify('Pipe inner radius',
                              str(round(self.r_in, 4)) + ' (m)')
            output += justify('Pipe outer radius',
                              str(round(self.r_out, 4)) + ' (m)')
        else:
            output += justify('Pipe inner radii',
                              str(self.r_in) + ' (m)')
            output += justify('Pipe outer radii',
                              str(self.r_out) + ' (m)')
        output += justify('Shank spacing (pipe to pipe)',
                          str(round(self.s, 4)) + ' (m)')
        output += justify('Pipe roughness', str(self.eps) + ' (m)')
        output += justify('Number of pipes', str(self.n_pipes))

        return output

    @staticmethod
    def place_pipes(s, r_out, n_pipes):
        """ Positions pipes in an axisymetric configuration."""
        D_s = s / 2 + r_out
        pi = np.pi
        dt = pi / float(n_pipes)
        pos = [(0., 0.) for i in range(2 * n_pipes)]
        for i in range(n_pipes):
            pos[2 * i] = (
                D_s * np.cos(2.0 * i * dt + pi),
                D_s * np.sin(2.0 * i * dt + pi))
            pos[2 * i + 1] = (D_s * np.cos(2.0 * i * dt + pi + dt),
                              D_s * np.sin(2.0 * i * dt + pi + dt))
        return pos


class Soil(ThermalProperty):
    def __init__(self, k, rhoCp, ugt):
        # Make variables from ThermalProperty available to Pipe
        ThermalProperty.__init__(self, k, rhoCp)

        # Soil specific parameters
        self.ugt = ugt

    def __repr__(self):
        def justify(category, value):
            return category.ljust(40) + '= ' + value + '\n'

        output = ThermalProperty.__repr__(self)

        output += justify('Undisturbed ground temperature',
                          str(round(self.ugt, 4)) + ' (degrees Celsius)')

        return output


class SimulationParameters:
    def __init__(self, start_month, end_month, max_EFT_allowable,
                 min_EFT_allowable, max_Height, min_Height):
        # Simulation parameters not found in other objects
        # ------------------------------------------------
        # Simulation start month and end month
        self.start_month = start_month
        self.end_month = end_month
        # Maximum and minimum allowable fluid temperatures
        self.max_EFT_allowable = max_EFT_allowable  # degrees Celsius
        self.min_EFT_allowable = min_EFT_allowable  # degrees Celsius
        # Maximum and minimum allowable heights
        self.max_Height = max_Height  # in meters
        self.min_Height = min_Height  # in meters

    def __repr__(self):
        def justify(category, value):
            return category.ljust(40) + '= ' + value + '\n'

        output = str(self.__class__) + '\n'

        output += justify('Start month', str(self.start_month) + ' (months)')
        output += justify('End month', str(self.end_month) + '(months)')
        output += justify('Max EFT Allowable',
                          str(self.max_EFT_allowable) + ' (degrees Celsius)')
        output += justify('Min EFT Allowable',
                          str(self.min_EFT_allowable) + ' (degrees Celsius)')
        output += justify('Maximum height', str(self.max_Height) + ' (m)')
        output += justify('Minimum height', str(self.min_Height) + ' (m)')

        return output
