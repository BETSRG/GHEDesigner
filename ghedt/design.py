from abc import abstractmethod

import numpy as np
import pygfunction as gt

from ghedt import domains, geometry, search_routines
from ghedt.media import Grout, Pipe, SimulationParameters, Soil
from ghedt.utilities import DesignMethod


class DesignBase:
    def __init__(
            self,
            v_flow: float,
            borehole: gt.boreholes.Borehole,
            bhe_object,
            fluid: gt.media.Fluid,
            pipe: Pipe,
            grout: Grout,
            soil: Soil,
            sim_params: SimulationParameters,
            geometric_constraints: geometry.GeometricConstraints,
            hourly_extraction_ground_loads: list,
            method: DesignMethod,
            flow: str = "borehole",
            load_years=None
    ):
        if load_years is None:
            load_years = [2019]
        self.load_years = load_years
        self.V_flow = v_flow  # volumetric flow rate, m3/s
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
        self.flow = flow
        if self.method == "hourly":
            msg = (
                "Note: It is not recommended to perform a field selection ",
                "with the hourly simulation due to computation time. If ",
                "the goal is to validate the selected field with the ",
                "hourly simulation, the better solution is to utilize the ",
                "hybrid simulation to automatically select the field. Then ",
                "perform a sizing routine on the selected GHE with the ",
                "hourly simulation."
            )
            # Wrap the text to a 50 char line width and print it
            for element in msg:
                print(element)
            print("\n")

    @abstractmethod
    def find_design(self, disp=False, b_r_point=None, b_r_removal_method="CloseToCorner",
                    exhaustive_fields_to_check=10, use_perimeter=True):
        pass


class DesignNearSquare(DesignBase):
    def __init__(self, v_flow: float, borehole: gt.boreholes.Borehole, bhe_object, fluid: gt.media.Fluid, pipe: Pipe,
                 grout: Grout, soil: Soil, sim_params: SimulationParameters,
                 geometric_constraints: geometry.GeometricConstraints, hourly_extraction_ground_loads: list,
                 method: DesignMethod, flow: str = "borehole", load_years=None):
        super().__init__(v_flow, borehole, bhe_object, fluid, pipe, grout, soil, sim_params, geometric_constraints,
                         hourly_extraction_ground_loads, method, flow, load_years)
        self.routine = "near-square"
        self.geometric_constraints.check_inputs(self.routine)
        # If a near-square design routine is requested, then we go from a
        # 1x1 to 32x32 at the B-spacing
        # The lower end of the near-square routine is always 1 borehole.
        # There would never be a time that a user would __need__ to give a
        # different lower range. The upper number of boreholes range is
        # calculated based on the spacing and length provided.
        n = np.floor(self.geometric_constraints.length / self.geometric_constraints.B) + 1
        number_of_boreholes = int(n)
        (
            self.coordinates_domain,
            self.fieldDescriptors,
        ) = domains.square_and_near_square(
            1, number_of_boreholes, self.geometric_constraints.B
        )

    def find_design(self, disp=False, b_r_point=None, b_r_removal_method="CloseToCorner",
                    exhaustive_fields_to_check=10, use_perimeter=True):
        if disp:
            title = "Find {}...".format(self.routine)
            print(title + "\n" + len(title) * "=")
        return search_routines.Bisection1D(
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
            field_type="near-square",
            load_years=self.load_years,
        )


class DesignRectangle(DesignBase):
    def __init__(self, v_flow: float, borehole: gt.boreholes.Borehole, bhe_object, fluid: gt.media.Fluid, pipe: Pipe,
                 grout: Grout, soil: Soil, sim_params: SimulationParameters,
                 geometric_constraints: geometry.GeometricConstraints, hourly_extraction_ground_loads: list,
                 method: DesignMethod, flow: str = "borehole", load_years=None):
        super().__init__(v_flow, borehole, bhe_object, fluid, pipe, grout, soil, sim_params, geometric_constraints,
                         hourly_extraction_ground_loads, method, flow, load_years)
        self.routine = "rectangle"
        self.geometric_constraints.check_inputs(self.routine)
        self.coordinates_domain, self.fieldDescriptors = domains.rectangular(
            self.geometric_constraints.length, self.geometric_constraints.width,
            self.geometric_constraints.B_min, self.geometric_constraints.B_max_x, disp=False
        )

    def find_design(self, disp=False, b_r_point=None, b_r_removal_method="CloseToCorner",
                    exhaustive_fields_to_check=10, use_perimeter=True):
        if disp:
            title = "Find {}...".format(self.routine)
            print(title + "\n" + len(title) * "=")
        return search_routines.Bisection1D(
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
            field_type="rectangle",
            load_years=self.load_years,
        )


class DesignBiRectangle(DesignBase):
    def __init__(self, v_flow: float, borehole: gt.boreholes.Borehole, bhe_object, fluid: gt.media.Fluid, pipe: Pipe,
                 grout: Grout, soil: Soil, sim_params: SimulationParameters,
                 geometric_constraints: geometry.GeometricConstraints, hourly_extraction_ground_loads: list,
                 method: DesignMethod, flow: str = "borehole", load_years=None):
        super().__init__(v_flow, borehole, bhe_object, fluid, pipe, grout, soil, sim_params, geometric_constraints,
                         hourly_extraction_ground_loads, method, flow, load_years)
        self.routine = "bi-rectangle"
        self.geometric_constraints.check_inputs(self.routine)
        (
            self.coordinates_domain_nested,
            self.fieldDescriptors,
        ) = domains.bi_rectangle_nested(
            self.geometric_constraints.length, self.geometric_constraints.width, self.geometric_constraints.B_min,
            self.geometric_constraints.B_max_x, self.geometric_constraints.B_max_y, disp=False
        )

    def find_design(self, disp=False, b_r_point=None, b_r_removal_method="CloseToCorner",
                    exhaustive_fields_to_check=10, use_perimeter=True):
        if disp:
            title = "Find {}...".format(self.routine)
            print(title + "\n" + len(title) * "=")
        return search_routines.Bisection2D(
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
            field_type="bi-rectangle",
            load_years=self.load_years,
        )


class DesignBiZoned(DesignBase):
    def __init__(self, v_flow: float, borehole: gt.boreholes.Borehole, bhe_object, fluid: gt.media.Fluid, pipe: Pipe,
                 grout: Grout, soil: Soil, sim_params: SimulationParameters,
                 geometric_constraints: geometry.GeometricConstraints, hourly_extraction_ground_loads: list,
                 method: DesignMethod, flow: str = "borehole", load_years=None):
        super().__init__(v_flow, borehole, bhe_object, fluid, pipe, grout, soil, sim_params, geometric_constraints,
                         hourly_extraction_ground_loads, method, flow, load_years)
        self.routine = "bi-zoned"
        self.geometric_constraints.check_inputs(self.routine)
        (
            self.coordinates_domain_nested,
            self.fieldDescriptors,
        ) = domains.bi_rectangle_zoned_nested(
            self.geometric_constraints.length, self.geometric_constraints.width, self.geometric_constraints.B_min,
            self.geometric_constraints.B_max_x, self.geometric_constraints.B_max_y
        )

    def find_design(self, disp=False, b_r_point=None, b_r_removal_method="CloseToCorner",
                    exhaustive_fields_to_check=10, use_perimeter=True):
        if disp:
            title = "Find {}...".format(self.routine)
            print(title + "\n" + len(title) * "=")
        return search_routines.BisectionZD(
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
            field_type="bi-zoned",
        )


class DesignBiRectangleConstrained(DesignBase):
    def __init__(self, v_flow: float, borehole: gt.boreholes.Borehole, bhe_object, fluid: gt.media.Fluid, pipe: Pipe,
                 grout: Grout, soil: Soil, sim_params: SimulationParameters,
                 geometric_constraints: geometry.GeometricConstraints, hourly_extraction_ground_loads: list,
                 method: DesignMethod, flow: str = "borehole", load_years=None,
                 property_boundary=None, building_descriptions=None):
        super().__init__(v_flow, borehole, bhe_object, fluid, pipe, grout, soil, sim_params, geometric_constraints,
                         hourly_extraction_ground_loads, method, flow, load_years)
        self.routine = "bi-rectangle_constrained"
        self.geometric_constraints.check_inputs(self.routine)
        (
            self.coordinates_domain_nested,
            self.fieldDescriptors,
        ) = domains.polygonal_land_constraint(
            property_boundary,
            self.geometric_constraints.B_min,
            self.geometric_constraints.B_max_x,
            self.geometric_constraints.B_max_y,
            building_descriptions=building_descriptions,
        )

    def find_design(self, disp=False, b_r_point=None, b_r_removal_method="CloseToCorner",
                    exhaustive_fields_to_check=10, use_perimeter=True):
        if disp:
            title = "Find {}...".format(self.routine)
            print(title + "\n" + len(title) * "=")
        return search_routines.Bisection2D(
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
            field_type="bi-rectangle_constrained",
            load_years=self.load_years,
        )


class DesignRowWise(DesignBase):
    def __init__(self, v_flow: float, borehole: gt.boreholes.Borehole, bhe_object, fluid: gt.media.Fluid, pipe: Pipe,
                 grout: Grout, soil: Soil, sim_params: SimulationParameters,
                 geometric_constraints: geometry.GeometricConstraints, hourly_extraction_ground_loads: list,
                 method: DesignMethod, flow: str = "borehole", load_years=None):
        super().__init__(v_flow, borehole, bhe_object, fluid, pipe, grout, soil, sim_params, geometric_constraints,
                         hourly_extraction_ground_loads, method, flow, load_years)
        self.routine = "row-wise"
        self.geometric_constraints.check_inputs(self.routine)

    def find_design(self, disp=False, b_r_point=None, b_r_removal_method="CloseToCorner",
                    exhaustive_fields_to_check=10, use_perimeter=True):
        if b_r_point is None:
            b_r_point = [0.0, 0.0]
        if disp:
            title = "Find {}...".format(self.routine)
            print(title + "\n" + len(title) * "=")
        return search_routines.RowWiseModifiedBisectionSearch(
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
            field_type="row-wise",
            load_years=self.load_years,
            b_r_point=b_r_point,
            b_r_removal_method=b_r_removal_method,
            exhaustive_fields_to_check=exhaustive_fields_to_check,
            use_perimeter=use_perimeter,
        )
