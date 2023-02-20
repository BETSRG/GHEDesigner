from abc import abstractmethod
from math import floor
from typing import Union

from ghedesigner.borehole import GHEBorehole
from ghedesigner.domains import polygonal_land_constraint, bi_rectangle_nested
from ghedesigner.domains import square_and_near_square, rectangular, bi_rectangle_zoned_nested
from ghedesigner.enums import BHPipeType
from ghedesigner.geometry import GeometricConstraints, GeometricConstraintsBiRectangle
from ghedesigner.geometry import GeometricConstraintsBiZoned, GeometricConstraintsBiRectangleConstrained
from ghedesigner.geometry import GeometricConstraintsNearSquare, GeometricConstraintsRectangle
from ghedesigner.geometry import GeometricConstraintsRowWise
from ghedesigner.media import Grout, Pipe, Soil, GHEFluid
from ghedesigner.search_routines import Bisection1D, Bisection2D, BisectionZD, RowWiseModifiedBisectionSearch
from ghedesigner.simulation import SimulationParameters
from ghedesigner.utilities import DesignMethodTimeStep

AnyBisectionType = Union[Bisection1D, Bisection2D, BisectionZD, RowWiseModifiedBisectionSearch]


class DesignBase:
    def __init__(
            self,
            v_flow: float,
            _borehole: GHEBorehole,
            bhe_type: BHPipeType,
            fluid: GHEFluid,
            pipe: Pipe,
            grout: Grout,
            soil: Soil,
            sim_params: SimulationParameters,
            geometric_constraints: GeometricConstraints,
            hourly_extraction_ground_loads: list,
            method: DesignMethodTimeStep,
            flow: str = "borehole",
            load_years=None
    ):
        if load_years is None:
            load_years = [2019]
        self.load_years = load_years
        self.V_flow = v_flow  # volumetric flow rate, m3/s
        self.borehole = _borehole
        self.bhe_type = bhe_type  # a borehole heat exchanger object
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
    def find_design(self, disp=False, use_perimeter=True) -> AnyBisectionType:
        pass


class DesignNearSquare(DesignBase):
    def __init__(self, v_flow: float, _borehole: GHEBorehole, bhe_type: BHPipeType,
                 fluid: GHEFluid, pipe: Pipe, grout: Grout, soil: Soil, sim_params: SimulationParameters,
                 geometric_constraints: GeometricConstraintsNearSquare, hourly_extraction_ground_loads: list,
                 method: DesignMethodTimeStep, flow: str = "borehole", load_years=None):
        super().__init__(v_flow, _borehole, bhe_type, fluid, pipe, grout, soil, sim_params, geometric_constraints,
                         hourly_extraction_ground_loads, method, flow, load_years)
        self.geometric_constraints = geometric_constraints
        # If a near-square design routine is requested, then we go from a
        # 1x1 to 32x32 at the B-spacing
        # The lower end of the near-square routine is always 1
        # There would never be a time that a user would __need__ to give a
        # different lower range. The upper number of boreholes range is
        # calculated based on the spacing and length provided.
        n = floor(self.geometric_constraints.length / self.geometric_constraints.B) + 1
        number_of_boreholes = int(n)
        self.coordinates_domain, self.fieldDescriptors = square_and_near_square(1, number_of_boreholes,
                                                                                self.geometric_constraints.B)

    def find_design(self, disp=False, use_perimeter=True) -> Bisection1D:
        if disp:
            title = "Find near-square.."
            print(title + "\n" + len(title) * "=")
        return Bisection1D(
            self.coordinates_domain,
            self.fieldDescriptors,
            self.V_flow,
            self.borehole,
            self.bhe_type,
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
    def __init__(self, v_flow: float, _borehole: GHEBorehole, bhe_type: BHPipeType,
                 fluid: GHEFluid, pipe: Pipe, grout: Grout, soil: Soil, sim_params: SimulationParameters,
                 geometric_constraints: GeometricConstraintsRectangle, hourly_extraction_ground_loads: list,
                 method: DesignMethodTimeStep, flow: str = "borehole", load_years=None):
        super().__init__(v_flow, _borehole, bhe_type, fluid, pipe, grout, soil, sim_params, geometric_constraints,
                         hourly_extraction_ground_loads, method, flow, load_years)
        self.geometric_constraints = geometric_constraints
        self.coordinates_domain, self.fieldDescriptors = rectangular(
            self.geometric_constraints.length, self.geometric_constraints.width,
            self.geometric_constraints.B_min, self.geometric_constraints.B_max_x)

    def find_design(self, disp=False, use_perimeter=True) -> Bisection1D:
        if disp:
            title = "Find rectangle..."
            print(title + "\n" + len(title) * "=")
        return Bisection1D(
            self.coordinates_domain,
            self.fieldDescriptors,
            self.V_flow,
            self.borehole,
            self.bhe_type,
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
    def __init__(self, v_flow: float, _borehole: GHEBorehole, bhe_type: BHPipeType,
                 fluid: GHEFluid, pipe: Pipe,
                 grout: Grout, soil: Soil, sim_params: SimulationParameters,
                 geometric_constraints: GeometricConstraintsBiRectangle, hourly_extraction_ground_loads: list,
                 method: DesignMethodTimeStep, flow: str = "borehole", load_years=None):
        super().__init__(v_flow, _borehole, bhe_type, fluid, pipe, grout, soil, sim_params, geometric_constraints,
                         hourly_extraction_ground_loads, method, flow, load_years)
        self.geometric_constraints = geometric_constraints
        self.coordinates_domain_nested, self.fieldDescriptors = bi_rectangle_nested(
            self.geometric_constraints.length, self.geometric_constraints.width, self.geometric_constraints.B_min,
            self.geometric_constraints.B_max_x, self.geometric_constraints.B_max_y, disp=False
        )

    def find_design(self, disp=False, use_perimeter=True) -> Bisection2D:
        if disp:
            title = "Find bi-rectangle..."
            print(title + "\n" + len(title) * "=")
        return Bisection2D(
            self.coordinates_domain_nested,
            self.fieldDescriptors,
            self.V_flow,
            self.borehole,
            self.bhe_type,
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
    def __init__(self, v_flow: float, _borehole: GHEBorehole, bhe_type: BHPipeType,
                 fluid: GHEFluid, pipe: Pipe,
                 grout: Grout, soil: Soil, sim_params: SimulationParameters,
                 geometric_constraints: GeometricConstraintsBiZoned, hourly_extraction_ground_loads: list,
                 method: DesignMethodTimeStep, flow: str = "borehole", load_years=None):
        super().__init__(v_flow, _borehole, bhe_type, fluid, pipe, grout, soil, sim_params, geometric_constraints,
                         hourly_extraction_ground_loads, method, flow, load_years)
        self.geometric_constraints = geometric_constraints
        self.coordinates_domain_nested, self.fieldDescriptors = bi_rectangle_zoned_nested(
            self.geometric_constraints.length, self.geometric_constraints.width, self.geometric_constraints.B_min,
            self.geometric_constraints.B_max_x, self.geometric_constraints.B_max_y
        )

    def find_design(self, disp=False, use_perimeter=True) -> BisectionZD:
        if disp:
            title = "Find bi-zoned..."
            print(title + "\n" + len(title) * "=")
        return BisectionZD(
            self.coordinates_domain_nested,
            self.fieldDescriptors,
            self.V_flow,
            self.borehole,
            self.bhe_type,
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
    def __init__(self, v_flow: float, _borehole: GHEBorehole, bhe_type: BHPipeType,
                 fluid: GHEFluid, pipe: Pipe, grout: Grout, soil: Soil, sim_params: SimulationParameters,
                 geometric_constraints: GeometricConstraintsBiRectangleConstrained,
                 hourly_extraction_ground_loads: list,
                 method: DesignMethodTimeStep, flow: str = "borehole", load_years=None,
                 property_boundary=None, building_descriptions=None):
        super().__init__(v_flow, _borehole, bhe_type, fluid, pipe, grout, soil, sim_params, geometric_constraints,
                         hourly_extraction_ground_loads, method, flow, load_years)
        self.geometric_constraints = geometric_constraints
        self.coordinates_domain_nested, self.fieldDescriptors = polygonal_land_constraint(
            property_boundary,
            self.geometric_constraints.B_min,
            self.geometric_constraints.B_max_x,
            self.geometric_constraints.B_max_y,
            building_descriptions=building_descriptions,
        )

    def find_design(self, disp=False, use_perimeter=True) -> Bisection2D:
        if disp:
            title = "Find bi-rectangle_constrained..."
            print(title + "\n" + len(title) * "=")
        return Bisection2D(
            self.coordinates_domain_nested,
            self.fieldDescriptors,
            self.V_flow,
            self.borehole,
            self.bhe_type,
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
    def __init__(self, v_flow: float, _borehole: GHEBorehole, bhe_type: BHPipeType,
                 fluid: GHEFluid, pipe: Pipe,
                 grout: Grout, soil: Soil, sim_params: SimulationParameters,
                 geometric_constraints: GeometricConstraintsRowWise, hourly_extraction_ground_loads: list,
                 method: DesignMethodTimeStep, flow: str = "borehole", load_years=None):
        super().__init__(v_flow, _borehole, bhe_type, fluid, pipe, grout, soil, sim_params, geometric_constraints,
                         hourly_extraction_ground_loads, method, flow, load_years)
        self.geometric_constraints = geometric_constraints

    def find_design(self, disp=False, use_perimeter=True) -> RowWiseModifiedBisectionSearch:
        if disp:
            title = "Find row-wise..."
            print(title + "\n" + len(title) * "=")
        return RowWiseModifiedBisectionSearch(
            self.V_flow,
            self.borehole,
            self.bhe_type,
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
            use_perimeter=use_perimeter,
        )
