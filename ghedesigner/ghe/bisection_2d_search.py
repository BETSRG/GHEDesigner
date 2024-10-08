from pygfunction.boreholes import Borehole

from ghedesigner.ghe.bisection_1d_search import Bisection1D
from ghedesigner.enums import BHPipeType, FlowConfigType, TimestepType
from ghedesigner.media import GHEFluid, Grout, Pipe, Soil
from ghedesigner.ghe.simulation import SimulationParameters


class Bisection2D(Bisection1D):
    def __init__(
        self,
        coordinates_domain_nested: list,
        field_descriptors: list,
        v_flow: float,
        borehole: Borehole,
        bhe_type,
        fluid: GHEFluid,
        pipe: Pipe,
        grout: Grout,
        soil: Soil,
        sim_params: SimulationParameters,
        hourly_extraction_ground_loads: list,
        method: TimestepType,
        flow_type: FlowConfigType.BOREHOLE,
        max_iter=15,
        disp=False,
        field_type="N/A",
        load_years=None,
    ):
        if load_years is None:
            load_years = [2019]
        if disp:
            print("Note: This routine requires a nested bisection search.")
        self.load_years = load_years
        # Get a coordinates domain for initialization
        coordinates_domain = coordinates_domain_nested[0]
        super().__init__(
            coordinates_domain,
            field_descriptors[0],
            v_flow,
            borehole,
            bhe_type,
            fluid,
            pipe,
            grout,
            soil,
            sim_params,
            hourly_extraction_ground_loads,
            method=method,
            flow_type=flow_type,
            max_iter=max_iter,
            disp=disp,
            search=False,
            field_type=field_type,
            load_years=load_years,
        )

        self.coordinates_domain_nested = []
        self.calculated_temperatures_nested = []
        # Tack on one borehole at the beginning to provide a high excess temperature
        outer_domain = [coordinates_domain_nested[0][0]]
        for cdn in coordinates_domain_nested:
            outer_domain.append(cdn[-1])

        self.coordinates_domain = outer_domain

        selection_key, _ = self.search()

        self.calculated_temperatures_nested.append(self.calculated_temperatures)

        # We tacked on one borehole to the beginning, so we need to subtract 1
        # on the index
        inner_domain = coordinates_domain_nested[selection_key - 1]
        self.coordinates_domain = inner_domain
        self.fieldDescriptors = field_descriptors[selection_key - 1]

        # Reset calculated temperatures
        self.calculated_temperatures = {}

        self.selection_key, self.selected_coordinates = self.search()