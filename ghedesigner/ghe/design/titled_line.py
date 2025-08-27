from dataclasses import asdict, dataclass, field
from math import floor

from pygfunction.boreholes import Borehole

from ghedesigner.enums import DesignGeomType, FlowConfigType, TimestepType
from ghedesigner.ghe.design.base import DesignBase, GeometricConstraints
from ghedesigner.ghe.domains import straight_line
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.ghe.search.bisection_1d_tilt_search import Bisection1DTilt
from ghedesigner.media import GHEFluid, Grout, Soil


@dataclass
class GeometricConstraintsTiltedLine(GeometricConstraints):
    """
    Geometric constraints for near square design algorithm
    """

    b: float
    length: float
    tilt: float
    type: DesignGeomType = field(default=DesignGeomType.TILTEDLINE, init=False, repr=False)

    def to_input(self) -> dict:
        return {
            **asdict(self, dict_factory=lambda d: {k: v for k, v in d if k != "type"}),
            "method": self.type.name,
        }


class DesignTiltedLine(DesignBase):
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
        geometric_constraints: GeometricConstraintsTiltedLine,
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

        n = floor(self.geometric_constraints.length / self.geometric_constraints.b) + 1
        number_of_boreholes = int(n)

        (
            self.tilted_coordinates_domain,
            self.tilted_fieldDescriptors,
            self.staggered_coordinates_domain,
            self.staggered_fieldDescriptors,
        ) = straight_line(
            1, number_of_boreholes, self.geometric_constraints.b, self.geometric_constraints.tilt, self.max_height
        )

    def find_design(self, disp=False) -> Bisection1DTilt:
        if disp:
            title = "Find near-square.."
            print(title + "\n" + len(title) * "=")
        return Bisection1DTilt(
            self.tilted_coordinates_domain,
            self.tilted_fieldDescriptors,
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
            field_type="Tilted Line",
            load_years=self.load_years,
            staggered_coordinates_domain=self.staggered_coordinates_domain,
            staggered_field_descriptors=self.staggered_fieldDescriptors,
        )
