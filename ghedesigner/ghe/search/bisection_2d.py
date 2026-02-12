from pygfunction.boreholes import Borehole

from ghedesigner.enums import FlowConfigType, TimestepType
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.ghe.search.bisection_1d import Bisection1D
from ghedesigner.media import Fluid, Grout, Soil


class Bisection2D(Bisection1D):
    def __init__(
        self,
        coordinates_domain_nested: list,
        field_descriptors: list,
        v_flow: float,
        borehole: Borehole,
        fluid: Fluid,
        pipe: Pipe,
        grout: Grout,
        soil: Soil,
        max_boreholes: int | None,
        min_height: float,
        max_height: float,
        continue_if_design_unmet: bool,
        start_month: int,
        end_month: int,
        min_eft: float,
        max_eft: float,
        hourly_extraction_ground_loads: list,
        method: TimestepType,
        flow_type: FlowConfigType = FlowConfigType.BOREHOLE,
        max_iter=15,
        disp=False,
        field_type="N/A",
        load_years=None,
    ) -> None:
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
            fluid,
            pipe,
            grout,
            soil,
            max_boreholes,
            min_height,
            max_height,
            continue_if_design_unmet,
            start_month,
            end_month,
            min_eft,
            max_eft,
            hourly_extraction_ground_loads,
            method=method,
            flow_type=flow_type,
            max_iter=max_iter,
            disp=disp,
            search=False,
            field_type=field_type,
            load_years=load_years,
        )

        # TODO why is the class variable set to an empty list and not the `coordinates_domain_nested` argument?
        # self.coordinates_domain_nested = []
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
