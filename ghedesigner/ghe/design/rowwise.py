from pygfunction.boreholes import Borehole

from ghedesigner.constants import RAD_TO_DEG
from ghedesigner.enums import BHPipeType, DesignGeomType, FlowConfigType, TimestepType
from ghedesigner.ghe.design.base import DesignBase, GeometricConstraints
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.ghe.search.rowwise import RowWiseModifiedBisectionSearch
from ghedesigner.media import GHEFluid, Grout, Soil


class GeometricConstraintsRowWise(GeometricConstraints):
    """
    Geometric constraints for rowwise design algorithm
    """

    def __init__(
        self,
        perimeter_spacing_ratio: float,
        min_spacing: float,
        max_spacing: float,
        spacing_step: float,
        min_rotation: float,
        max_rotation: float,
        rotate_step: float,
        property_boundary,
        no_go_boundaries,
    ) -> None:
        super().__init__()
        self.perimeter_spacing_ratio = perimeter_spacing_ratio
        self.min_spacing = min_spacing
        self.max_spacing = max_spacing
        self.spacing_step = spacing_step
        self.min_rotation = min_rotation
        self.max_rotation = max_rotation
        self.rotate_step = rotate_step
        self.property_boundary = property_boundary
        self.no_go_boundaries = no_go_boundaries
        self.type = DesignGeomType.ROWWISE

    def to_input(self) -> dict:
        return {
            "perimeter_spacing_ratio": self.perimeter_spacing_ratio,
            "min_spacing": self.min_spacing,
            "max_spacing": self.max_spacing,
            "spacing_step": self.spacing_step,
            "min_rotation": self.min_rotation * RAD_TO_DEG,
            "max_rotation": self.max_rotation * RAD_TO_DEG,
            "rotate_step": self.rotate_step,
            "property_boundary": self.property_boundary,
            "no_go_boundaries": self.no_go_boundaries,
            "method": DesignGeomType.ROWWISE.name,
        }


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
            _borehole,
            bhe_type,
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
            self.V_flow,
            self.borehole,
            self.bhe_type,
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
