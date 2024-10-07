class SimulationParameters:
    def __init__(
        self,
        # max_height,
        # min_height,
        num_months: int,
        max_boreholes=None,
        continue_if_design_unmet=False,
    ):
        # Simulation parameters not found in other objects
        self.start_month = 1
        self.end_month = num_months
        self.max_boreholes = max_boreholes
        self.continue_if_design_unmet = continue_if_design_unmet

    def set_design_temps(self, max_eft, min_eft):
        self.max_EFT_allowable = max_eft  # degrees Celsius
        self.min_EFT_allowable = min_eft  # degrees Celsius

    def set_design_heights(self, max_height, min_height):
        self.max_height = max_height  # in meters
        self.min_height = min_height  # in meters

    def as_dict(self) -> dict:
        output = {}
        output['type'] = str(self.__class__)
        output['max_eft_allowable'] = {'value': self.max_EFT_allowable, 'units': 'C'}
        output['min_eft_allowable'] = {'value': self.min_EFT_allowable, 'units': 'C'}
        output['maximum_height'] = {'value': self.max_height, 'units': 'm'}
        output['minimum_height'] = {'value': self.min_height, 'units': 'm'}

        if self.max_boreholes is not None:
            output['maximum_boreholes'] = {'value': self.max_boreholes, 'units': '-'}
        return output

    # def to_input(self) -> dict:
    #     return {'num_months': self.end_month}
