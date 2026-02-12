from dataclasses import asdict, dataclass, field
from typing import TypeGuard, cast

from pygfunction.boreholes import Borehole

from ghedesigner.enums import DesignGeomType, FlowConfigType, TimestepType
from ghedesigner.ghe.design.base import DesignBase, GeometricConstraints
from ghedesigner.ghe.domains import polygonal_land_constraint
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.ghe.search.bisection_zd import BisectionZD
from ghedesigner.media import Fluid, Grout, Soil


def is_2d(property_boundary: list[list[float]] | list[list[list[float]]]) -> TypeGuard[list[list[float]]]:
    return bool(property_boundary) and isinstance(property_boundary[0][0], float)


@dataclass
class GeometricConstraintsBiRectangleConstrained(GeometricConstraints):
    """
    Geometric constraints for bi-rectangle constrained design algorithm
    """

    b_min: float
    b_max_x: float
    b_max_y: float
    property_boundary: list[list[list[float]]] = field(init=False)
    no_go_boundaries: list[list[list[float]]] | None = None
    type: DesignGeomType = field(default=DesignGeomType.BIRECTANGLECONSTRAINED, init=False, repr=False)

    def __init__(
        self,
        b_min: float,
        b_max_x: float,
        b_max_y: float,
        property_boundary: list[list[float]] | list[list[list[float]]],
        no_go_boundaries: list[list[list[float]]] | None = None,
    ) -> None:
        self.b_min = b_min
        self.b_max_x = b_max_x
        self.b_max_y = b_max_y
        self.no_go_boundaries = no_go_boundaries

        if is_2d(property_boundary):
            self.property_boundary = [property_boundary]
        else:
            self.property_boundary = cast(list[list[list[float]]], property_boundary)

    def to_input(self) -> dict:
        return {
            **asdict(self, dict_factory=lambda d: {k: v for k, v in d if k != "type"}),
            "method": self.type.name,
        }


class DesignBiRectangleConstrained(DesignBase):
    def __init__(
        self,
        v_flow: float,
        borehole: Borehole,
        fluid: Fluid,
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
        geometric_constraints: GeometricConstraintsBiRectangleConstrained,
        hourly_extraction_ground_loads: list,
        method: TimestepType,
        flow_type: FlowConfigType = FlowConfigType.BOREHOLE,
        load_years=None,
        keep_contour: tuple[bool, bool] | None = None,
    ) -> None:
        super().__init__(
            v_flow,
            borehole,
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
        if keep_contour is None:
            keep_contour = cast(tuple[bool, bool], [True, False])
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
            self.v_flow,
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
            field_type="bi-rectangle_constrained",
            load_years=self.load_years,
        )
