from ghedesigner.enums import SimCompType
from ghedesigner.utilities import get_loads


class HeatPumpFixedCOP:
    def __init__(self, name: str, data: dict) -> None:
        self.name = name
        self.comp_type = SimCompType.HEAT_PUMP
        self.cop = data["total_load"]["heat_pump_cop"]
        if self.cop < 0:
            raise ValueError("Coefficient of Performance (COP) must be greater than zero.")

        self.loads = get_loads(name, self.comp_type.name, data["total_load"])

    def convert_bldg_load_to_ground_load(self, bldg_load):
        if bldg_load > 0:
            # heating load to heat extraction load
            return bldg_load * (1 - 1 / self.cop)
        else:
            # cooling load to heat rejection load
            return bldg_load * (1 + 1 / self.cop)

    def get_ground_loads(self):
        return list(map(self.convert_bldg_load_to_ground_load, self.loads))
