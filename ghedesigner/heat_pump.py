from collections.abc import Callable
from pathlib import Path
from typing import cast

LoadFunctionType = Callable[[float], float]


class HeatPump:
    def __init__(self, name: str) -> None:
        self.name = name
        self.cop: float | None = None
        self.loads_list: list[float] | None = None
        self.load_function: LoadFunctionType | None = None

    def set_loads_from_file(self, file_path: Path) -> None:
        loads_file_content = file_path.read_text().strip()
        self.loads_list = [float(x) for x in loads_file_content.split()]
        self.load_function = self.get_load_from_list_at_time

    def set_loads_from_lambda(self, load_function: LoadFunctionType) -> None:
        self.load_function = load_function

    def get_load_from_list_at_time(self, time: float) -> float:
        if self.loads_list is None:
            raise ValueError("Loads values are not set.")
        # Assume time in hours? No leap year? No DST?
        hour_of_year = int(time % 8760)
        return self.loads_list[hour_of_year]

    def set_fixed_cop(self, cop: float) -> None:
        if cop <= 0:
            raise ValueError("Coefficient of Performance (COP) must be greater than zero.")
        self.cop = cop

    def do_sizing(self) -> None:
        pass

    def calculate(self, simulation_time_hours: float, loop_inlet_temp: float, flow_rate: float) -> float:
        if not self.load_function:
            raise ValueError("Load function is not set.")
        building_load = self.load_function(simulation_time_hours)
        if flow_rate == 0 or building_load == 0:
            return loop_inlet_temp

        # Check if cop is set to avoid division by zero
        if not self.cop:
            raise ValueError("Coefficient of Performance (COP) is not set.")

        if building_load > 0:
            # Heating the building, taking energy from loop
            loop_load = building_load - (building_load / self.cop)
        else:
            # Cooling the building, adding energy to loop
            loop_load = building_load + (building_load / self.cop)

        cp = 4100
        return loop_inlet_temp + loop_load / (flow_rate * cp)

    def get_ground_loads(self) -> list[float]:
        if self.loads_list is None:
            raise ValueError("Loads values are not set.")
        ground_loads: list[float] = []
        self.load_function = cast(LoadFunctionType, self.load_function)
        self.cop = cast(float, self.cop)
        for h in range(8760):
            building_load = self.load_function(h)
            if building_load > 0:
                # Heating the building, taking energy from loop
                loop_load = building_load - (building_load / self.cop)
            else:
                # Cooling the building, adding energy to loop
                loop_load = building_load + (building_load / self.cop)
            ground_loads.append(loop_load)
        return ground_loads
