from abc import abstractmethod

from pygfunction.boreholes import Borehole

from ghedesigner.constants import PI, TWO_PI
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.media import Fluid, Grout, Soil


class GHEDesignerBoreholeBase:
    def __init__(
        self,
        m_flow_borehole: float,
        fluid: Fluid,
        borehole: Borehole,
        pipe: Pipe,
        grout: Grout,
        soil: Soil,
    ) -> None:
        self.m_flow_borehole = m_flow_borehole
        self.pipe = pipe
        self.soil = soil
        self.grout = grout
        self.fluid = fluid
        self.borehole = borehole

    @abstractmethod
    def calc_effective_borehole_resistance(self) -> float:
        pass

    @staticmethod
    def compute_fluid_resistance(h_conv: float, radius: float) -> float:
        return 1 / (h_conv * TWO_PI * radius)

    @staticmethod
    def compute_reynolds(m_flow_pipe: float, r_in: float, fluid: Fluid) -> float:
        # Hydraulic diameter
        dia_hydraulic = 2.0 * r_in
        # Fluid velocity
        vol_flow_rate = m_flow_pipe / fluid.rho
        area_cr_inner = PI * r_in**2
        velocity = vol_flow_rate / area_cr_inner
        # Reynolds number
        return fluid.rho * velocity * dia_hydraulic / fluid.mu
