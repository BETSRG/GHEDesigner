from pygfunction.media import Fluid

from ghedesigner.enums import FluidType


class GHEFluid(Fluid):
    def __init__(self, fluid_str: str, percent: float, temperature: float = 20.0) -> None:
        fluid_type_map = {
            FluidType.ETHYLALCOHOL.name: (FluidType.ETHYLALCOHOL, "MEA"),
            FluidType.ETHYLENEGLYCOL.name: (FluidType.ETHYLENEGLYCOL, "MEG"),
            FluidType.METHYLALCOHOL.name: (FluidType.METHYLALCOHOL, "MMA"),
            FluidType.PROPYLENEGLYCOL.name: (FluidType.PROPYLENEGLYCOL, "MPG"),
            FluidType.WATER.name: (FluidType.WATER, "WATER"),
        }

        try:
            self.fluid_type, fluid_shorthand = fluid_type_map[fluid_str.upper()]
        except KeyError:
            raise ValueError(f'FluidType "{fluid_str}" not implemented')

        super().__init__(fluid_shorthand, percent, temperature)
        self.concentration_percent = percent
        self.temperature = temperature

    def to_input(self):
        return {
            "fluid_name": self.fluid_type.name,
            "concentration_percent": self.concentration_percent,
            "temperature": self.temperature,
        }


class ThermalProperty:
    def __init__(self, k, rho_cp) -> None:
        self.k = k  # Thermal conductivity (W/m.K)
        self.rhoCp = rho_cp  # Volumetric heat capacity (J/K.m3)

    def as_dict(self) -> dict:
        output = {
            "type": str(self.__class__),
            "thermal_conductivity": {"value": self.k, "units": "W/m-K"},
            "volumetric_heat_capacity": {"value": self.rhoCp, "units": "J/K-m3"},
        }
        return output

    def to_input(self) -> dict:
        return {"conductivity": self.k, "rho_cp": self.rhoCp}


class Grout(ThermalProperty):
    pass


class Soil(ThermalProperty):
    def __init__(self, k, rho_cp, ugt) -> None:
        # Make variables from ThermalProperty available to Pipe
        ThermalProperty.__init__(self, k, rho_cp)

        # Soil specific parameters
        self.ugt = ugt

    def as_dict(self) -> dict:
        output = super().as_dict()
        output["undisturbed_ground_temperature"] = {"value": self.ugt, "units": "C"}
        return output

    def to_input(self) -> dict:
        return {"conductivity": self.k, "rho_cp": self.rhoCp, "undisturbed_temp": self.ugt}
