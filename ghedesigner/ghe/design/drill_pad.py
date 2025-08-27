from dataclasses import asdict, dataclass, field

from pygfunction.boreholes import Borehole

from ghedesigner.enums import DesignGeomType, FlowConfigType, TimestepType
from ghedesigner.ghe.design.base import DesignBase, GeometricConstraints
from ghedesigner.ghe.domains import drill_pad
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.ghe.search.bisection_1d_tilt_drill_pad_search import Bisection1DTiltDrillPad
from ghedesigner.media import GHEFluid, Grout, Soil


@dataclass
class GeometricConstraintsDrillPad(GeometricConstraints):
    """
    Geometric constraints for drill pad design algorithm
    """

    nbh: int
    radius: float
    tilt: float
    ndp_min: int
    ndp_max: int
    type: DesignGeomType = field(default=DesignGeomType.DRILLPAD, init=False, repr=False)

    def to_input(self) -> dict:
        return {
            **asdict(self, dict_factory=lambda d: {k: v for k, v in d if k != "type"}),
            "method": self.type.name,
        }


class DesignDrillPad(DesignBase):
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
        geometric_constraints: GeometricConstraintsDrillPad,
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
        gc = geometric_constraints
        self.min_eft = min_eft
        self.max_eft = max_eft
        self.ndp_min = geometric_constraints.ndp_min
        self.ndp_max = geometric_constraints.ndp_max
        # always *one* pad layout
        self.coordinates_domain, self.fieldDescriptors = drill_pad(
            nbh=gc.nbh,
            tilt=gc.tilt,
            radius=gc.radius,
            ndp_min=gc.ndp_min,
            ndp_max=gc.ndp_max,
        )

    def find_design(self, disp=False) -> Bisection1DTiltDrillPad:
        if disp:
            title = "Find drill pad design.."
            print(title + "\n" + "=" * len(title))

        return Bisection1DTiltDrillPad(
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
            self.ndp_min,
            self.ndp_max,
            self.continue_if_design_unmet,
            self.start_month,
            self.end_month,
            self.min_eft,
            self.max_eft,
            self.hourly_extraction_ground_loads,
            method=self.method,
            flow_type=self.flow_type,
            disp=disp,
            field_type="Drill Pad",
            load_years=self.load_years,
        )
