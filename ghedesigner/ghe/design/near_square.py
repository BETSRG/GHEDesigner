from dataclasses import asdict, dataclass, field
from math import floor

from pygfunction.boreholes import Borehole

from ghedesigner.enums import DesignGeomType, FlowConfigType, TimestepType
from ghedesigner.ghe.design.base import DesignBase, GeometricConstraints
from ghedesigner.ghe.domains import square_and_near_square
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.ghe.search.bisection_1d import Bisection1D
from ghedesigner.media import Fluid, Grout, Soil


@dataclass
class GeometricConstraintsNearSquare(GeometricConstraints):
    """
    Geometric constraints for near square design algorithm
    """

    b: float
    length: float
    type: DesignGeomType = field(default=DesignGeomType.NEARSQUARE, init=False, repr=False)

    def to_input(self) -> dict:
        return {
            **asdict(self, dict_factory=lambda d: {k: v for k, v in d if k != "type"}),
            "method": self.type.name,
        }


class DesignNearSquare(DesignBase):
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
        geometric_constraints: GeometricConstraintsNearSquare,
        hourly_extraction_ground_loads: list,
        method: TimestepType,
        flow_type: FlowConfigType = FlowConfigType.BOREHOLE,
        load_years=None,
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
            field_type="near-square",
            load_years=self.load_years,
        )
