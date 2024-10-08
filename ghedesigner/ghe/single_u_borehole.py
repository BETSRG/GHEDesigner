from ghedesigner.ghe.borehole_base import GHEDesignerBoreholeBase
from ghedesigner.media import GHEFluid, Pipe, Grout, Soil
from ghedesigner.ghe.borehole import GHEBorehole
import pygfunction as gt


class SingleUTube(gt.pipes.SingleUTube, GHEDesignerBoreholeBase):
    def __init__(
        self,
        m_flow_borehole: float,
        fluid: GHEFluid,
        _borehole: GHEBorehole,
        pipe: Pipe,
        grout: Grout,
        soil: Soil,
    ):
        GHEDesignerBoreholeBase.__init__(self, m_flow_borehole, fluid, _borehole, pipe, grout, soil)
        self.R_p = 0.0
        self.R_f = 0.0
        self.R_fp = 0.0
        self.h_f = 0.0
        self.fluid = fluid
        self.m_flow_borehole = m_flow_borehole
        self.borehole = _borehole
        self.pipe = pipe
        self.soil = soil
        self.grout = grout

        # compute resistances required to construct inherited class
        self.calc_fluid_pipe_resistance()

        # Initialize pygfunction SingleUTube base class
        super().__init__(
            self.pipe.pos,
            self.pipe.r_in,
            self.pipe.r_out,
            self.borehole,
            self.soil.k,
            self.grout.k,
            self.R_fp,
        )

        # these methods must be called after inherited class construction
        self.update_thermal_resistances(self.R_fp)
        self.calc_effective_borehole_resistance()

    def calc_fluid_pipe_resistance(self) -> float:
        self.h_f = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
            self.m_flow_borehole,
            self.pipe.r_in,
            self.fluid.mu,
            self.fluid.rho,
            self.fluid.k,
            self.fluid.cp,
            self.pipe.roughness,
        )
        self.R_f = self.compute_fluid_resistance(self.h_f, self.pipe.r_in)
        self.R_p = gt.pipes.conduction_thermal_resistance_circular_pipe(self.pipe.r_in, self.pipe.r_out, self.pipe.k)
        self.R_fp = self.R_f + self.R_p
        return self.R_fp

    def calc_effective_borehole_resistance(self) -> float:
        # TODO: should this be here?
        self._initialize_stored_coefficients()
        resist_bh_effective = self.effective_borehole_thermal_resistance(self.m_flow_borehole, self.fluid.cp)
        return resist_bh_effective

    def to_single(self):
        return self

    def as_dict(self) -> dict:
        return {'type': str(self.__class__)}
