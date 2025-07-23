from dataclasses import InitVar, asdict, dataclass, field

from pygfunction.media import Fluid

from ghedesigner.enums import FluidType


@dataclass
class GHEFluid(Fluid):
    fluid_str: InitVar[str]
    percent: InitVar[float]
    temperature: float = 20.0
    concentration_percent: float = field(init=False)
    fluid_type: FluidType = field(init=False)

    def __post_init__(self, fluid_str: str, percent: float) -> None:
        fluid_type_map = {
            FluidType.ETHYLALCOHOL.name: (FluidType.ETHYLALCOHOL, "MEA"),
            FluidType.ETHYLENEGLYCOL.name: (FluidType.ETHYLENEGLYCOL, "MEG"),
            FluidType.METHYLALCOHOL.name: (FluidType.METHYLALCOHOL, "MMA"),
            FluidType.PROPYLENEGLYCOL.name: (FluidType.PROPYLENEGLYCOL, "MPG"),
            FluidType.WATER.name: (FluidType.WATER, "WATER"),
        }

        self.concentration_percent = percent

        try:
            self.fluid_type, fluid_shorthand = fluid_type_map[fluid_str.upper()]
        except KeyError:
            raise ValueError(f'FluidType "{fluid_str}" not implemented')

        super().__init__(fluid_shorthand, percent, self.temperature)

    def to_input(self) -> dict:
        return {
            **asdict(self, dict_factory=lambda d: {k: v for k, v in d if k not in {"fluid_type"}}),
            "fluid_name": self.fluid_type.name,
        }


class ThermalProperty:
    def __init__(self, k, rho_cp: float) -> None:
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
    def __init__(self, k: float, rho_cp: float, ugt: float) -> None:
        # Make variables from ThermalProperty available to Pipe
        super().__init__(k, rho_cp)

        # Soil specific parameters
        self.ugt = ugt
        self.alpha = k / rho_cp

    def as_dict(self) -> dict:
        output = super().as_dict()
        output["undisturbed_ground_temperature"] = {"value": self.ugt, "units": "C"}
        return output

    def to_input(self) -> dict:
        return {"conductivity": self.k, "rho_cp": self.rhoCp, "undisturbed_temp": self.ugt}
