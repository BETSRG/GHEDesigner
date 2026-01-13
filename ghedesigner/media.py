from scp.ethyl_alcohol import EthylAlcohol
from scp.ethylene_glycol import EthyleneGlycol
from scp.methyl_alcohol import MethylAlcohol
from scp.propylene_glycol import PropyleneGlycol
from scp.water import Water

from ghedesigner.enums import FluidType


class CustomFluid:
    def __init__(self, fluid_name: str, density: float, specific_heat: float, conductivity: float, viscosity: float):
        self.name = fluid_name
        self.density = density
        self.specific_heat = specific_heat
        self.conductivity = conductivity
        self.viscosity = viscosity

    def rho(self, _unused):
        return self.density

    def cp(self, _unused):
        return self.specific_heat

    def k(self, _unused):
        return self.conductivity

    def mu(self, _unused):
        return self.viscosity


class Fluid:
    def __init__(
        self,
        fluid_name: str,
        temperature: float = 20,
        percent: float = 0.0,
        density: float = 0.0,
        specific_heat: float = 0.0,
        conductivity: float = 0.0,
        viscosity: float = 0.0,
    ) -> None:
        self.name = fluid_name
        self.fluid_type = self.get_fluid_type(fluid_name)
        self.temperature = temperature
        self.concentration_percent = percent

        # supported props
        self.cp: float = 0.0
        self.k: float = 0.0
        self.mu: float = 0.0
        self.rho: float = 0.0
        self.rho_cp: float = 0.0

        concentration_frac = self.concentration_percent / 100
        if self.fluid_type == FluidType.ETHYLALCOHOL:
            self._fluid = EthylAlcohol(concentration_frac)
        elif self.fluid_type == FluidType.ETHYLENEGLYCOL:
            self._fluid = EthyleneGlycol(concentration_frac)
        elif self.fluid_type == FluidType.METHYLALCOHOL:
            self._fluid = MethylAlcohol(concentration_frac)
        elif self.fluid_type == FluidType.PROPYLENEGLYCOL:
            self._fluid = PropyleneGlycol(concentration_frac)
        elif self.fluid_type == FluidType.WATER:
            self._fluid = Water()
        elif self.fluid_type == FluidType.CUSTOMFLUID:
            self._fluid = CustomFluid(fluid_name, density, specific_heat, conductivity, viscosity)
        else:
            raise NotImplementedError(f"{self.fluid_type.name} not implemented via this calling path.")

        self.update_props_with_new_temp(temperature)

    @staticmethod
    def init_from_dictionary(fluid_inputs: dict) -> "Fluid":
        required_keys_opt_1 = ["fluid_name", "concentration_percent", "temperature"]
        if all(key in fluid_inputs for key in required_keys_opt_1):
            fluid_name = fluid_inputs["fluid_name"]
            percent = fluid_inputs["concentration_percent"]
            temperature = fluid_inputs["temperature"]
            return Fluid(fluid_name, percent, temperature)

        required_keys_opt_2 = ["fluid_name", "density", "specific_heat", "conductivity", "viscosity"]
        if all(key in fluid_inputs for key in required_keys_opt_2):
            return Fluid(**fluid_inputs)

        raise NotImplementedError("This combination of fluid inputs is not implemented.")

    @staticmethod
    def get_fluid_type(fluid_name: str) -> FluidType:
        fluid_name_upper = fluid_name.upper()
        if fluid_name_upper in ["MEA", "ETHYLALCOHOL", "ETHYL ALCOHOL"]:
            return FluidType.ETHYLALCOHOL
        if fluid_name_upper in ["MEG", "ETHYLENEGLYCOL", "ETHYLENE GLYCOL"]:
            return FluidType.ETHYLENEGLYCOL
        if fluid_name_upper in ["MMA", "METHYLALCOHOL", "METHYL ALCOHOL"]:
            return FluidType.METHYLALCOHOL
        if fluid_name_upper in ["MPG", "PROPYLENEGLYCOL", "PROPYLENE GLYCOL"]:
            return FluidType.PROPYLENEGLYCOL
        if fluid_name_upper == "WATER":
            return FluidType.WATER
        if fluid_name_upper == "CUSTOMFLUID":
            return FluidType.CUSTOMFLUID

        raise ValueError(f'Unsupported fluid type "{fluid_name}"')

    def update_props_with_new_temp(self, temperature: float) -> None:
        self.temperature = temperature
        self.cp = self._fluid.cp(self.temperature)
        self.k = self._fluid.k(self.temperature)
        self.mu = self._fluid.mu(self.temperature)
        self.rho = self._fluid.rho(self.temperature)
        self.rho_cp = self.rho * self.cp


class ThermalProperty:
    def __init__(self, k, rho_cp: float) -> None:
        self.k = k  # Thermal conductivity (W/m.K)
        self.rho_cp = rho_cp  # Volumetric heat capacity (J/K.m3)

    def as_dict(self) -> dict:
        output = {
            "type": str(self.__class__),
            "thermal_conductivity": {"value": self.k, "units": "W/m-K"},
            "volumetric_heat_capacity": {"value": self.rho_cp, "units": "J/K-m3"},
        }
        return output

    def to_input(self) -> dict:
        return {"conductivity": self.k, "rho_cp": self.rho_cp}


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
        return {"conductivity": self.k, "rho_cp": self.rho_cp, "undisturbed_temp": self.ugt}
