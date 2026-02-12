import numpy as np
from pygfunction.boreholes import Borehole

from ghedesigner.enums import FlowConfigType, TimestepType
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.ghe.search.bisection_1d import Bisection1D
from ghedesigner.media import Fluid, Grout, Soil


class BisectionZD(Bisection1D):
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
            print("Note: This design routine currently requires several bisection searches.")

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
        self.max_height = max_height
        self.coordinates_domain_nested = coordinates_domain_nested
        self.nested_fieldDescriptors = field_descriptors
        self.calculated_temperatures_nested: dict[int, dict[int, np.float64]] = {}
        # Tack on one borehole at the beginning to provide a high excess
        # temperature
        outer_domain = [coordinates_domain_nested[0][0]]
        outer_descriptors = [field_descriptors[0][0]]
        for cdn, fd in zip(coordinates_domain_nested, field_descriptors):
            outer_domain.append(cdn[-1])
            outer_descriptors.append(fd[-1])

        self.coordinates_domain = outer_domain
        self.fieldDescriptors = outer_descriptors

        self.selection_key_outer, _ = self.search()
        if self.selection_key_outer > 0:
            self.selection_key_outer -= 1
        self.calculated_heights: dict[int, float] = {}

        self.selection_key, self.selected_coordinates = self.search_successive()

    def search_successive(self, max_iter=None):
        if max_iter is None:
            max_iter = self.selection_key_outer + 7

        i = self.selection_key_outer

        old_height = 99999

        while i < len(self.coordinates_domain_nested) and i < max_iter:
            self.coordinates_domain = self.coordinates_domain_nested[i]
            self.fieldDescriptors = self.nested_fieldDescriptors[i]
            self.calculated_temperatures = {}
            try:
                selection_key, selected_coordinates = self.search()
            except ValueError:
                break
            self.calculated_temperatures_nested[i] = self.calculated_temperatures

            self.ghe.compute_g_functions(self.min_height, self.max_height)
            self.ghe.size(
                method=TimestepType.HYBRID,
                max_height=self.max_height,
                min_height=self.min_height,
                design_max_eft=self.max_eft,
                design_min_eft=self.min_eft,
            )

            nbh = len(selected_coordinates)
            total_drilling = nbh * self.ghe.bhe.borehole.H
            self.calculated_heights[i] = total_drilling

            if old_height < total_drilling:
                break
            else:
                old_height = total_drilling

            i += 1

        keys = list(self.calculated_heights.keys())
        values = list(self.calculated_heights.values())

        minimum_total_drilling = min(values)
        idx = values.index(minimum_total_drilling)
        selection_key_outer = keys[idx]
        self.calculated_temperatures = self.calculated_temperatures_nested[selection_key_outer]

        keys = list(self.calculated_temperatures.keys())
        values = list(self.calculated_temperatures.values())

        negative_excess_values = [v for v in values if v <= 0.0]

        excess_of_interest = max(negative_excess_values)
        idx = values.index(excess_of_interest)
        selection_key = keys[idx]
        selected_coordinates = self.coordinates_domain_nested[selection_key_outer][selection_key]

        self.initialize_ghe(
            selected_coordinates,
            self.max_height,
            field_specifier=self.nested_fieldDescriptors[selection_key_outer][selection_key],
        )
        self.ghe.compute_g_functions(self.min_height, self.max_height)
        self.ghe.size(TimestepType.HYBRID, self.max_height, self.min_height, self.max_eft, self.min_eft)

        return selection_key, selected_coordinates
