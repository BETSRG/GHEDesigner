from dataclasses import asdict, dataclass, field

from pygfunction.boreholes import Borehole

from ghedesigner.constants import RAD_TO_DEG
from ghedesigner.enums import DesignGeomType, FlowConfigType, TimestepType
from ghedesigner.ghe.design.base import DesignBase, GeometricConstraints
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.ghe.search.rowwise import RowWiseModifiedBisectionSearch
from ghedesigner.media import Fluid, Grout, Soil


@dataclass
class GeometricConstraintsRowWise(GeometricConstraints):
    """
    Geometric constraints for rowwise design algorithm
    """

    perimeter_spacing_ratio: float | None
    min_spacing: float
    max_spacing: float
    spacing_step: float | None
    min_rotation: float
    max_rotation: float
    rotate_step: float
    property_boundary: list[list[float]]
    no_go_boundaries: list[list[list[float]]] | None
    type: DesignGeomType = field(default=DesignGeomType.ROWWISE, init=False, repr=False)

    def to_input(self) -> dict:
        return {
            **asdict(self, dict_factory=lambda d: {k: v for k, v in d if k != "type"}),
            "min_rotation": self.min_rotation * RAD_TO_DEG,
            "max_rotation": self.max_rotation * RAD_TO_DEG,
            "method": self.type.name,
        }


class DesignRowWise(DesignBase):
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
        geometric_constraints: GeometricConstraintsRowWise,
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

    def find_design(self, disp=False) -> RowWiseModifiedBisectionSearch:
        if disp:
            title = "Find row-wise..."
            print(title + "\n" + len(title) * "=")
        return RowWiseModifiedBisectionSearch(
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
            self.geometric_constraints,
            method=self.method,
            flow_type=self.flow_type,
            disp=disp,
            field_type="row-wise",
            load_years=self.load_years,
        )
