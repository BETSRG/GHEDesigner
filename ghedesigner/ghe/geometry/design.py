from __future__ import annotations

from abc import abstractmethod
from math import floor

from pygfunction.boreholes import Borehole

from ghedesigner.enums import BHPipeType, FlowConfigType, TimestepType
from ghedesigner.ghe.bisection_1d_search import Bisection1D
from ghedesigner.ghe.bisection_2d_search import Bisection2D
from ghedesigner.ghe.bisection_zd_search import BisectionZD
from ghedesigner.ghe.geometry.domains import (
    bi_rectangle_nested,
    bi_rectangle_zoned_nested,
    polygonal_land_constraint,
    rectangular,
    square_and_near_square,
)
from ghedesigner.ghe.geometry.geometry import (
    GeometricConstraints,
    GeometricConstraintsBiRectangle,
    GeometricConstraintsBiRectangleConstrained,
    GeometricConstraintsBiZoned,
    GeometricConstraintsNearSquare,
    GeometricConstraintsRectangle,
    GeometricConstraintsRowWise,
)
from ghedesigner.ghe.rowwise_search import RowWiseModifiedBisectionSearch
from ghedesigner.ghe.simulation import SimulationParameters
from ghedesigner.media import GHEFluid, Grout, Pipe, Soil

AnyBisectionType = Bisection1D | Bisection2D | BisectionZD | RowWiseModifiedBisectionSearch


class DesignBase:
    def __init__(
        self,
        v_flow: float,
        _borehole: Borehole,
        bhe_type: BHPipeType,
        fluid: GHEFluid,
        pipe: Pipe,
        grout: Grout,
        soil: Soil,
        sim_params: SimulationParameters,
        geometric_constraints: GeometricConstraints,
        hourly_extraction_ground_loads: list,
        method: TimestepType,
        flow_type: FlowConfigType = FlowConfigType.BOREHOLE,
        load_years=None,
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
        self.flow_type = flow_type
        if self.method == "hourly":
            msg = (
                "Note: It is not recommended to perform a field selection \n",
                "with the hourly simulation due to computation time. If \n",
                "the goal is to validate the selected field with the \n",
                "hourly simulation, the better solution is to utilize the \n",
                "hybrid simulation to automatically select the field. Then \n",
                "perform a sizing routine on the selected GHE with the \n",
                "hourly simulation.",
            )
            # Wrap the text to a 50 char line width and print it
            for element in msg:
                print(element)
            print("\n")

    @abstractmethod
    def find_design(self, disp=False) -> AnyBisectionType:
        pass

    def to_input(self) -> dict:
        return {"flow_rate": self.V_flow, "flow_type": self.flow_type.name}


class DesignNearSquare(DesignBase):
    def __init__(
        self,
        v_flow: float,
        _borehole: Borehole,
        bhe_type: BHPipeType,
        fluid: GHEFluid,
        pipe: Pipe,
        grout: Grout,
        soil: Soil,
        sim_params: SimulationParameters,
        geometric_constraints: GeometricConstraintsNearSquare,
        hourly_extraction_ground_loads: list,
        method: TimestepType,
        flow_type: FlowConfigType = FlowConfigType.BOREHOLE,
        load_years=None,
    ):
        super().__init__(
            v_flow,
            _borehole,
            bhe_type,
            fluid,
            pipe,
            grout,
            soil,
            sim_params,
            geometric_constraints,
            hourly_extraction_ground_loads,
            method,
            flow_type,
            load_years,
        )
        self.geometric_constraints = geometric_constraints
        # If a near-square design routine is requested, then we go from a
        # 1x1 to 32x32 at the B-spacing
        # The lower end of the near-square routine is always 1
        # There would never be a time that a user would __need__ to give a
        # different lower range. The upper number of boreholes range is
        # calculated based on the spacing and length provided.
        n = floor(self.geometric_constraints.length / self.geometric_constraints.b) + 1
        number_of_boreholes = int(n)
        self.coordinates_domain, self.fieldDescriptors = square_and_near_square(
            1, number_of_boreholes, self.geometric_constraints.b
        )

    def find_design(self, disp=False) -> Bisection1D:
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
            flow_type=self.flow_type,
            disp=disp,
            field_type="near-square",
            load_years=self.load_years,
        )


class DesignRectangle(DesignBase):
    def __init__(
        self,
        v_flow: float,
        _borehole: Borehole,
        bhe_type: BHPipeType,
        fluid: GHEFluid,
        pipe: Pipe,
        grout: Grout,
        soil: Soil,
        sim_params: SimulationParameters,
        geometric_constraints: GeometricConstraintsRectangle,
        hourly_extraction_ground_loads: list,
        method: TimestepType,
        flow_type: FlowConfigType = FlowConfigType.BOREHOLE,
        load_years=None,
    ):
        super().__init__(
            v_flow,
            _borehole,
            bhe_type,
            fluid,
            pipe,
            grout,
            soil,
            sim_params,
            geometric_constraints,
            hourly_extraction_ground_loads,
            method,
            flow_type,
            load_years,
        )
        self.geometric_constraints = geometric_constraints
        self.coordinates_domain, self.fieldDescriptors = rectangular(
            self.geometric_constraints.length,
            self.geometric_constraints.width,
            self.geometric_constraints.b_min,
            self.geometric_constraints.b_max_x,
        )

    def find_design(self, disp=False) -> Bisection1D:
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
            flow_type=self.flow_type,
            disp=disp,
            field_type="rectangle",
            load_years=self.load_years,
        )


class DesignBiRectangle(DesignBase):
    def __init__(
        self,
        v_flow: float,
        _borehole: Borehole,
        bhe_type: BHPipeType,
        fluid: GHEFluid,
        pipe: Pipe,
        grout: Grout,
        soil: Soil,
        sim_params: SimulationParameters,
        geometric_constraints: GeometricConstraintsBiRectangle,
        hourly_extraction_ground_loads: list,
        method: TimestepType,
        flow_type: FlowConfigType = FlowConfigType.BOREHOLE,
        load_years=None,
    ):
        super().__init__(
            v_flow,
            _borehole,
            bhe_type,
            fluid,
            pipe,
            grout,
            soil,
            sim_params,
            geometric_constraints,
            hourly_extraction_ground_loads,
            method,
            flow_type,
            load_years,
        )
        self.geometric_constraints = geometric_constraints
        self.coordinates_domain_nested, self.fieldDescriptors = bi_rectangle_nested(
            self.geometric_constraints.length,
            self.geometric_constraints.width,
            self.geometric_constraints.b_min,
            self.geometric_constraints.b_max_x,
            self.geometric_constraints.b_max_y,
            disp=False,
        )

    def find_design(self, disp=False) -> Bisection2D:
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
            flow_type=self.flow_type,
            disp=disp,
            field_type="bi-rectangle",
            load_years=self.load_years,
        )


class DesignBiZoned(DesignBase):
    def __init__(
        self,
        v_flow: float,
        _borehole: Borehole,
        bhe_type: BHPipeType,
        fluid: GHEFluid,
        pipe: Pipe,
        grout: Grout,
        soil: Soil,
        sim_params: SimulationParameters,
        geometric_constraints: GeometricConstraintsBiZoned,
        hourly_extraction_ground_loads: list,
        method: TimestepType,
        flow_type: FlowConfigType = FlowConfigType.BOREHOLE,
        load_years=None,
    ):
        super().__init__(
            v_flow,
            _borehole,
            bhe_type,
            fluid,
            pipe,
            grout,
            soil,
            sim_params,
            geometric_constraints,
            hourly_extraction_ground_loads,
            method,
            flow_type,
            load_years,
        )
        self.geometric_constraints = geometric_constraints
        self.coordinates_domain_nested, self.fieldDescriptors = bi_rectangle_zoned_nested(
            self.geometric_constraints.length,
            self.geometric_constraints.width,
            self.geometric_constraints.b_min,
            self.geometric_constraints.b_max_x,
            self.geometric_constraints.b_max_y,
        )

    def find_design(self, disp=False) -> BisectionZD:
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
            flow_type=self.flow_type,
            disp=disp,
            field_type="bi-zoned",
        )


class DesignBiRectangleConstrained(DesignBase):
    def __init__(
        self,
        v_flow: float,
        _borehole: Borehole,
        bhe_type: BHPipeType,
        fluid: GHEFluid,
        pipe: Pipe,
        grout: Grout,
        soil: Soil,
        sim_params: SimulationParameters,
        geometric_constraints: GeometricConstraintsBiRectangleConstrained,
        hourly_extraction_ground_loads: list,
        method: TimestepType,
        flow_type: FlowConfigType = FlowConfigType.BOREHOLE,
        load_years=None,
        keep_contour=[True, False],
    ):
        super().__init__(
            v_flow,
            _borehole,
            bhe_type,
            fluid,
            pipe,
            grout,
            soil,
            sim_params,
            geometric_constraints,
            hourly_extraction_ground_loads,
            method,
            flow_type,
            load_years,
        )
        self.geometric_constraints = geometric_constraints
        self.coordinates_domain_nested, self.fieldDescriptors = polygonal_land_constraint(
            self.geometric_constraints.b_min,
            self.geometric_constraints.b_max_x,
            self.geometric_constraints.b_max_y,
            self.geometric_constraints.property_boundary,
            self.geometric_constraints.no_go_boundaries,
            keep_contour=keep_contour,
        )

    def find_design(self, disp=False) -> BisectionZD:
        if disp:
            title = "Find bi-rectangle_constrained..."
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
            flow_type=self.flow_type,
            disp=disp,
            field_type="bi-rectangle_constrained",
            load_years=self.load_years,
        )


class DesignRowWise(DesignBase):
    def __init__(
        self,
        v_flow: float,
        _borehole: Borehole,
        bhe_type: BHPipeType,
        fluid: GHEFluid,
        pipe: Pipe,
        grout: Grout,
        soil: Soil,
        sim_params: SimulationParameters,
        geometric_constraints: GeometricConstraintsRowWise,
        hourly_extraction_ground_loads: list,
        method: TimestepType,
        flow_type: FlowConfigType = FlowConfigType.BOREHOLE,
        load_years=None,
    ):
        super().__init__(
            v_flow,
            _borehole,
            bhe_type,
            fluid,
            pipe,
            grout,
            soil,
            sim_params,
            geometric_constraints,
            hourly_extraction_ground_loads,
            method,
            flow_type,
            load_years,
        )
        self.geometric_constraints = geometric_constraints

    def find_design(self, disp=False) -> RowWiseModifiedBisectionSearch:
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
            flow_type=self.flow_type,
            disp=disp,
            field_type="row-wise",
            load_years=self.load_years,
        )
