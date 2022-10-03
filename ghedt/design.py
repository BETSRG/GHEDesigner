import textwrap

import numpy as np
import pygfunction as gt

import ghedt as dt
from ghedt.peak_load_analysis_tool.media import Grout, Pipe, SimulationParameters, Soil


# Common design interface
class Design:
    def __init__(
        self,
        V_flow: float,
        borehole: gt.boreholes.Borehole,
        bhe_object,
        fluid: gt.media.Fluid,
        pipe: Pipe,
        grout: Grout,
        soil: Soil,
        sim_params: SimulationParameters,
        geometric_constraints: dt.media.GeometricConstraints,
        hourly_extraction_ground_loads: list,
        method: str = "hybrid",
        routine: str = "near-square",
        flow: str = "borehole",
        property_boundary=None,
        buildingDescriptions=None,
        load_years=[2019],
    ):
        self.load_years = load_years
        self.V_flow = V_flow  # volumetric flow rate, m3/s
        self.borehole = borehole
        self.bhe_object = bhe_object  # a borehole heat exchanger object
        self.fluid = fluid  # a fluid object
        self.pipe = pipe
        self.grout = grout
        self.soil = soil
        self.sim_params = sim_params
        self.geometric_constraints = geometric_constraints
        self.hourly_extraction_ground_loads = hourly_extraction_ground_loads
        self.method = method
        if self.method == "hourly":
            msg = (
                "Note: It is not recommended to perform a field selection "
                "with the hourly simulation due to computation time. If "
                "the goal is to validate the selected field with the "
                "hourly simulation, the better solution is to utilize the "
                "hybrid simulation to automatically select the field. Then "
                "perform a sizing routine on the selected GHE with the "
                "hourly simulation."
            )
            # Wrap the text to a 50 char line width and print it
            wrapper = textwrap.TextWrapper(width=72)
            word_list = wrapper.wrap(text=msg)
            for element in word_list:
                print(element)
            print("\n")

        # Check the routine parameter
        self.routine = routine
        available_routines = [
            "near-square",
            "rectangle",
            "bi-rectangle",
            "bi-zoned",
            "bi-rectangle_constrained",
            "row-wise",
        ]
        self.geometric_constraints.check_inputs(self.routine)
        gc = self.geometric_constraints
        if routine in available_routines:
            # If a near-square design routine is requested, then we go from a
            # 1x1 to 32x32 at the B-spacing
            # The lower end of the near-square routine is always 1 borehole.
            # There would never be a time that a user would __need__ to give a
            # different lower range. The upper number of boreholes range is
            # calculated based on the spacing and length provided.
            if routine == "near-square":
                number_of_boreholes = dt.utilities.number_of_boreholes(
                    gc.length, gc.B, func=np.floor
                )
                (
                    self.coordinates_domain,
                    self.fieldDescriptors,
                ) = dt.domains.square_and_near_square(
                    1, number_of_boreholes, self.geometric_constraints.B
                )
            elif routine == "rectangle":
                self.coordinates_domain, self.fieldDescriptors = dt.domains.rectangular(
                    gc.length, gc.width, gc.B_min, gc.B_max_x, disp=False
                )
            elif routine == "bi-rectangle":
                (
                    self.coordinates_domain_nested,
                    self.fieldDescriptors,
                ) = dt.domains.bi_rectangle_nested(
                    gc.length, gc.width, gc.B_min, gc.B_max_x, gc.B_max_y, disp=False
                )
            elif routine == "bi-rectangle_constrained":
                (
                    self.coordinates_domain_nested,
                    self.fieldDescriptors,
                ) = dt.domains.polygonal_land_constraint(
                    property_boundary,
                    gc.B_min,
                    gc.B_max_x,
                    gc.B_max_y,
                    building_descriptions=buildingDescriptions,
                )
            elif routine == "bi-zoned":
                (
                    self.coordinates_domain_nested,
                    self.fieldDescriptors,
                ) = dt.domains.bi_rectangle_zoned_nested(
                    gc.length, gc.width, gc.B_min, gc.B_max_x, gc.B_max_y
                )
            elif routine == "row-wise":
                pass
        else:
            raise ValueError(
                "The requested routine is not available. "
                "The currently available routines are: "
                "`near-square`."
            )
        self.flow = flow

    def find_design(
        self,
        disp=False,
        BRPoint=[0.0, 0.0],
        BRRemovalMethod="CloseToCorner",
        exhaustiveFieldsToCheck=10,
        usePerimeter=True,
    ):
        if disp:
            title = "Find {}...".format(self.routine)
            print(title + "\n" + len(title) * "=")
        # Find near-square
        if self.routine == "near-square":
            bisection_search = dt.search_routines.Bisection1D(
                self.coordinates_domain,
                self.fieldDescriptors,
                self.V_flow,
                self.borehole,
                self.bhe_object,
                self.fluid,
                self.pipe,
                self.grout,
                self.soil,
                self.sim_params,
                self.hourly_extraction_ground_loads,
                method=self.method,
                flow=self.flow,
                disp=disp,
                fieldType="near-square",
                load_years=self.load_years,
            )
        # Find a rectangle
        elif self.routine == "rectangle":
            bisection_search = dt.search_routines.Bisection1D(
                self.coordinates_domain,
                self.fieldDescriptors,
                self.V_flow,
                self.borehole,
                self.bhe_object,
                self.fluid,
                self.pipe,
                self.grout,
                self.soil,
                self.sim_params,
                self.hourly_extraction_ground_loads,
                method=self.method,
                flow=self.flow,
                disp=disp,
                fieldType="rectangle",
                load_years=self.load_years,
            )
        # Find a bi-rectangle
        elif self.routine == "bi-rectangle":
            bisection_search = dt.search_routines.Bisection2D(
                self.coordinates_domain_nested,
                self.fieldDescriptors,
                self.V_flow,
                self.borehole,
                self.bhe_object,
                self.fluid,
                self.pipe,
                self.grout,
                self.soil,
                self.sim_params,
                self.hourly_extraction_ground_loads,
                method=self.method,
                flow=self.flow,
                disp=disp,
                fieldType="bi-rectangle",
                load_years=self.load_years,
            )
        elif self.routine == "bi-rectangle_constrained":
            bisection_search = dt.search_routines.Bisection2D(
                self.coordinates_domain_nested,
                self.fieldDescriptors,
                self.V_flow,
                self.borehole,
                self.bhe_object,
                self.fluid,
                self.pipe,
                self.grout,
                self.soil,
                self.sim_params,
                self.hourly_extraction_ground_loads,
                method=self.method,
                flow=self.flow,
                disp=disp,
                fieldType="bi-rectangle_constrained",
                load_years=self.load_years,
            )
        # Find bi-zoned rectangle
        elif self.routine == "bi-zoned":
            bisection_search = dt.search_routines.BisectionZD(
                self.coordinates_domain_nested,
                self.fieldDescriptors,
                self.V_flow,
                self.borehole,
                self.bhe_object,
                self.fluid,
                self.pipe,
                self.grout,
                self.soil,
                self.sim_params,
                self.hourly_extraction_ground_loads,
                method=self.method,
                flow=self.flow,
                disp=disp,
                fieldType="bi-zoned",
            )
        elif self.routine == "row-wise":
            bisection_search = dt.search_routines.RowWiseModifiedBisectionSearch(
                self.V_flow,
                self.borehole,
                self.bhe_object,
                self.fluid,
                self.pipe,
                self.grout,
                self.soil,
                self.sim_params,
                self.hourly_extraction_ground_loads,
                self.geometric_constraints,
                method=self.method,
                flow=self.flow,
                disp=disp,
                fieldType="row-wise",
                load_years=self.load_years,
                BRPoint=BRPoint,
                BRRemovalMethod=BRRemovalMethod,
                exhaustiveFieldsToCheck=exhaustiveFieldsToCheck,
                usePerimeter=usePerimeter,
            )
        else:
            raise ValueError(
                "The requested routine is not available. "
                "The currently available routines are: "
                "`near-square`."
            )

        return bisection_search
