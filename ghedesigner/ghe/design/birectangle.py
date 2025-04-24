from pygfunction.boreholes import Borehole

from ghedesigner.enums import DesignGeomType, FlowConfigType, TimestepType
from ghedesigner.ghe.design.base import DesignBase, GeometricConstraints
from ghedesigner.ghe.domains import bi_rectangle_nested
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.ghe.search.bisection_2d import Bisection2D
from ghedesigner.media import GHEFluid, Grout, Soil


class GeometricConstraintsBiRectangle(GeometricConstraints):
    """
    Geometric constraints for bi-rectangle design algorithm
    """

    def __init__(self, width: float, length: float, b_min: float, b_max_x: float, b_max_y: float) -> None:
        super().__init__()
        self.width = width
        self.length = length
        self.b_min = b_min
        self.b_max_x = b_max_x
        self.b_max_y = b_max_y
        self.type = DesignGeomType.BIRECTANGLE

    def to_input(self) -> dict:
        return {
            "length": self.length,
            "width": self.width,
            "b_min": self.b_min,
            "b_max_x": self.b_max_x,
            "b_max_y": self.b_max_y,
            "method": DesignGeomType.BIRECTANGLE.name,
        }


class DesignBiRectangle(DesignBase):
    def __init__(
        self,
        v_flow: float,
        _borehole: Borehole,
        fluid: GHEFluid,
        pipe: Pipe,
        grout: Grout,
        soil: Soil,
        start_month: int,
        end_month: int,
        max_eft: float,
        min_eft: float,
        max_height: float,
        min_height: float,
        continue_if_design_unmet: bool,
        max_boreholes: int | None,
        geometric_constraints: GeometricConstraintsBiRectangle,
        hourly_extraction_ground_loads: list,
        method: TimestepType,
        flow_type: FlowConfigType = FlowConfigType.BOREHOLE,
        load_years=None,
    ) -> None:
        super().__init__(
            v_flow,
            _borehole,
            fluid,
            pipe,
            grout,
            soil,
            start_month,
            end_month,
            max_eft,
            min_eft,
            max_height,
            min_height,
            continue_if_design_unmet,
            max_boreholes,
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
            self.fluid,
            self.pipe,
            self.grout,
            self.soil,
            self.max_boreholes,
            self.min_height,
            self.max_height,
            self.continue_if_design_unmet,
            self.start_month,
            self.end_month,
            self.min_EFT_allowable,
            self.max_EFT_allowable,
            self.hourly_extraction_ground_loads,
            method=self.method,
            flow_type=self.flow_type,
            disp=disp,
            field_type="bi-rectangle",
            load_years=self.load_years,
        )
