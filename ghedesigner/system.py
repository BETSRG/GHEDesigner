from ghedesigner.building import Building
from ghedesigner.ghe.ground_heat_exchangers import GHE


class System:  # TODO: This isn't being used except by a unit test, should we keep it?
    def __init__(self) -> None:
        self.num_months = 0
        self.initial_loop_temp = 20.0
        self.building: Building | None = None
        self.ghe: GHE | None = None

    def set_simulation_parameters(self, num_months: int, initial_loop_temp: float = 20):
        self.num_months = num_months
        self.initial_loop_temp = initial_loop_temp

    def set_building(self, building: Building):
        self.building = building

    def set_ghe(self, ghe: GHE):
        self.ghe = ghe

    def simulate(self):
        flow_rate = 3.14
        ghe_exit_temp = self.initial_loop_temp
        num_hours = self.num_months * 30 * 24
        for hour_index in range(num_hours):
            building_outlet_temp = self.building.calculate(hour_index, ghe_exit_temp, flow_rate)
            ghe_exit_temp = self.ghe.calculate(hour_index, building_outlet_temp, flow_rate)
