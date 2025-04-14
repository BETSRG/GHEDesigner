from ghedesigner.heat_pump import HeatPump


class Building:
    def __init__(self, name: str) -> None:
        self.name = name
        self.heat_pumps: list[HeatPump] = []

    def add_heat_pump(self, heat_pump: HeatPump):
        self.heat_pumps.append(heat_pump)

    def calculate(self, simulation_time: int, inlet_temperature: float, flow_rate: float) -> float:
        outlet_temperature = inlet_temperature
        for heat_pump in self.heat_pumps:
            outlet_temperature = heat_pump.calculate(simulation_time, outlet_temperature, flow_rate)
        return outlet_temperature
