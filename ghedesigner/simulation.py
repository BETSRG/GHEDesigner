class SimulationParameters:
    def __init__(
            self,
            start_month,
            end_month,
            max_entering_fluid_temp_allow,
            min_entering_fluid_temp_allow,
            max_height,
            min_height,
    ):
        # Simulation parameters not found in other objects
        # ------------------------------------------------
        # Simulation start month and end month
        self.start_month = start_month
        self.end_month = end_month
        # Maximum and minimum allowable fluid temperatures
        self.max_EFT_allowable = max_entering_fluid_temp_allow  # degrees Celsius
        self.min_EFT_allowable = min_entering_fluid_temp_allow  # degrees Celsius
        # Maximum and minimum allowable heights
        self.max_height = max_height  # in meters
        self.min_height = min_height  # in meters

    def as_dict(self) -> dict:
        output = dict()
        output['type'] = str(self.__class__)
        output['start_month'] = self.start_month
        output['end_month'] = self.end_month
        output['max_eft_allowable'] = {'value': self.max_EFT_allowable, 'units': 'C'}
        output['min_eft_allowable'] = {'value': self.min_EFT_allowable, 'units': 'C'}
        output['maximum_height'] = {'value': self.max_height, 'units': 'm'}
        output['minimum_height'] = {'value': self.min_height, 'units': 'm'}
        return output

    def to_input(self) -> dict:
        return {'num_months': self.end_month}
