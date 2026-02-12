from abc import abstractmethod
from dataclasses import dataclass, field

from pygfunction.boreholes import Borehole

from ghedesigner.enums import DesignGeomType, FlowConfigType, TimestepType
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.ghe.search.bisection_1d import Bisection1D
from ghedesigner.ghe.search.bisection_2d import Bisection2D
from ghedesigner.ghe.search.bisection_zd import BisectionZD
from ghedesigner.ghe.search.rowwise import RowWiseModifiedBisectionSearch
from ghedesigner.media import Fluid, Grout, Soil

AnyBisectionType = Bisection1D | Bisection2D | BisectionZD | RowWiseModifiedBisectionSearch


@dataclass(init=False)
class GeometricConstraints:
    type: DesignGeomType = field(default=DesignGeomType.NONE, init=False, repr=False)

    @abstractmethod
    def to_input(self) -> dict:
        pass


@dataclass
class DesignBase:
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
        geometric_constraints: GeometricConstraints,
        hourly_extraction_ground_loads: list,
        method: TimestepType,
        flow_type: FlowConfigType = FlowConfigType.BOREHOLE,
        load_years=None,
    ) -> None:
        if load_years is None:
            load_years = [2019]
        self.load_years = load_years
        self.v_flow = v_flow  # volumetric flow rate, m3/s
        self.borehole = borehole
        self.fluid = fluid  # a fluid object
        self.pipe = pipe
        self.grout = grout
        self.soil = soil
        self.geometric_constraints = geometric_constraints
        self.hourly_extraction_ground_loads = hourly_extraction_ground_loads
        self.method = method
        self.flow_type = flow_type
        self.start_month = start_month
        self.end_month = end_month
        self.max_EFT_allowable = max_eft
        self.min_EFT_allowable = min_eft
        self.max_height = max_height
        self.min_height = min_height
        self.continue_if_design_unmet = continue_if_design_unmet
        self.max_boreholes = max_boreholes
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
            for element in msg:
                print(element)
            print("\n")

    @abstractmethod
    def find_design(self, disp=False) -> AnyBisectionType:
        pass

    def to_input(self) -> dict:
        return {"flow_rate": self.v_flow, "flow_type": self.flow_type.name}
