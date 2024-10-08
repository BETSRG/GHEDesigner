from abc import abstractmethod

from ghedesigner.constants import TWO_PI, pi
from ghedesigner.media import GHEFluid, Pipe, Grout, Soil
from ghedesigner.ghe.borehole import GHEBorehole


class GHEDesignerBoreholeBase:
    def __init__(
        self,
        m_flow_borehole: float,
        fluid: GHEFluid,
        _borehole: GHEBorehole,
        pipe: Pipe,
        grout: Grout,
        soil: Soil,
    ):
        self.m_flow_borehole = m_flow_borehole
        self.borehole = _borehole
        self.pipe = pipe
        self.soil = soil
        self.grout = grout
        self.fluid = fluid
        self.b = _borehole

    @abstractmethod
    def calc_fluid_pipe_resistance(self) -> float:
        pass

    @abstractmethod
    def calc_effective_borehole_resistance(self) -> float:
        pass

    @staticmethod
    def compute_fluid_resistance(h_conv: float, radius: float) -> float:
        return 1 / (h_conv * TWO_PI * radius)

    @staticmethod
    def compute_reynolds(m_flow_pipe: float, r_in: float, fluid: GHEFluid) -> float:
        # Hydraulic diameter
        dia_hydraulic = 2.0 * r_in
        # Fluid velocity
        vol_flow_rate = m_flow_pipe / fluid.rho
        area_cr_inner = pi * r_in**2
        velocity = vol_flow_rate / area_cr_inner
        # Reynolds number
        return fluid.rho * velocity * dia_hydraulic / fluid.mu