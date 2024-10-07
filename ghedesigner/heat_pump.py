from pathlib import Path
from typing import Callable, Optional

load_function_type = Callable[[float], float]


class HeatPump:
    def __init__(self, name: str):
        self.name = name
        self.cop = None
        self.loads_list = None
        self.load_function: Optional[load_function_type] = None

    def set_loads_from_file(self, file_path: Path):
        loads_file_content = file_path.read_text().strip()
        self.loads_list = [float(x) for x in loads_file_content.split()]
        self.load_function = self.get_load_from_list_at_time

    def set_loads_from_lambda(self, load_function: load_function_type):
        self.load_function = load_function

    def get_load_from_list_at_time(self, time: float) -> float:
        # assume time in hours?  no leap year?  no DST?
        hour_of_year = time % 8760
        return self.loads_list[hour_of_year]

    def set_fixed_cop(self, cop: float):
        self.cop = cop

    def do_sizing(self):
        pass

    def calculate(self, simulation_time: float, inlet_temperature: float, flow_rate: float) -> float:
        building_load = self.load_function(simulation_time)
        if flow_rate == 0.0 or building_load == 0.0:
            return inlet_temperature
        if building_load > 0:  # heating the building, taking energy from loop
            loop_load = building_load - (building_load / self.cop)
        else:  # cooling the building, adding energy to loop
            loop_load = building_load + (building_load / self.cop)
        cp = 4100
        return inlet_temperature + loop_load / (flow_rate * cp)
