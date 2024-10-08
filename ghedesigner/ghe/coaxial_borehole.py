
from typing import Tuple

import numpy as np
import pygfunction as gt
from numpy import log, pi

from ghedesigner.ghe.multi_u_borehole import GHEDesignerBoreholeWithMultiplePipes, MultipleUTube
from ghedesigner.ghe.single_u_borehole import SingleUTube
from ghedesigner.ghe.borehole import GHEBorehole
from ghedesigner.constants import TWO_PI
from ghedesigner.enums import BHPipeType, DoubleUTubeConnType
from ghedesigner.media import GHEFluid, Grout, Pipe, Soil


class CoaxialPipe(gt.pipes.Coaxial, GHEDesignerBoreholeWithMultiplePipes):
    def __init__(
        self, m_flow_borehole: float, fluid: GHEFluid, _borehole: GHEBorehole, pipe: Pipe, grout: Grout, soil: Soil
    ):
        self.m_flow_borehole = m_flow_borehole
        # Store Thermal properties
        self.soil = soil
        self.grout = grout
        self.pipe = pipe
        # Store fluid properties
        self.fluid = fluid
        # Store pipe roughness
        self.roughness = self.pipe.roughness

        self.r_inner = pipe.r_in
        self.r_outer = pipe.r_out

        # Pipe naming nomenclature
        # <var>_<inner/outer pipe>_<inner/outer surface>
        # e.g. r_in_in is inner radius of the inner pipe

        # Unpack the radii to reduce confusion in the future
        self.r_in_in, self.r_in_out = self.r_inner
        self.r_out_in, self.r_out_out = self.r_outer

        self.borehole = _borehole  # pygfunction borehole

        # Declare variables that are computed in compute_resistance()
        self.R_p_in = 0.0
        self.R_p_out = 0.0
        self.R_grout = 0.0
        self.h_f_in = 0.0
        self.h_f_a_in = 0.0
        self.h_f_a_out = 0.0
        self.R_f_a_in = 0.0
        self.R_f_a_out = 0.0
        self.R_f_in = 0.0
        self.R_fp = 0.0

        # Store Thermal properties
        self.soil = soil
        self.grout = grout
        self.pipe = pipe
        # Store fluid properties
        self.fluid = fluid
        # Store borehole
        self.borehole = _borehole

        # compute resistances required to construct inherited class
        self.calc_fluid_pipe_resistance()

        # Vectors of inner and outer pipe radii
        # Note: The dimensions of the inlet pipe are the first elements of the vectors.
        # In this example, the inlet pipe is the inside pipe.
        # TODO: fix this
        r_inner_p = np.array([pipe.r_in[0], pipe.r_out[0]])  # Inner pipe radii (m)
        r_outer_p = np.array([pipe.r_in[1], pipe.r_out[1]])  # Outer pipe radii (m)

        gt.pipes.Coaxial.__init__(
            self, pipe.pos, r_inner_p, r_outer_p, _borehole, self.soil.k, self.grout.k, self.R_ff, self.R_fp
        )

        # these methods must be called after inherited class construction
        self.update_thermal_resistances(self.R_ff, self.R_fp)
        self.calc_effective_borehole_resistance()

    def calc_fluid_pipe_resistance(self) -> None:
        # inner pipe convection resistance
        self.h_f_in = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
            self.m_flow_borehole,
            self.r_in_in,
            self.fluid.mu,
            self.fluid.rho,
            self.fluid.k,
            self.fluid.cp,
            self.pipe.roughness,
        )
        self.R_f_in = self.compute_fluid_resistance(self.h_f_in, self.r_in_in)

        # inner pipe conduction resistance
        self.R_p_in = gt.pipes.conduction_thermal_resistance_circular_pipe(self.r_in_in, self.r_in_out, self.pipe.k[0])

        # annulus convection resistances
        self.h_f_a_in, self.h_f_a_out = gt.pipes.convective_heat_transfer_coefficient_concentric_annulus(
            self.m_flow_borehole,
            self.r_in_out,
            self.r_out_in,
            self.fluid.mu,
            self.fluid.rho,
            self.fluid.k,
            self.fluid.cp,
            self.roughness,
        )

        self.R_f_a_in = self.compute_fluid_resistance(self.h_f_a_in, self.r_in_out)
        self.R_f_a_out = self.compute_fluid_resistance(self.h_f_a_out, self.r_out_in)

        # inner pipe conduction resistance
        self.R_p_out = gt.pipes.conduction_thermal_resistance_circular_pipe(
            self.r_out_in, self.r_out_out, self.pipe.k[1]
        )

        # inner fluid to inner annulus fluid resistance
        self.R_ff = self.R_f_in + self.R_p_in + self.R_f_a_in

        # outer annulus fluid to pipe thermal resistance
        self.R_fp = self.R_p_out + self.R_f_a_out

    def calc_effective_borehole_resistance(self) -> float:
        # TODO: should this be here?
        self._initialize_stored_coefficients()
        resist_bh_effective = self.effective_borehole_thermal_resistance(self.m_flow_borehole, self.fluid.cp)
        return resist_bh_effective

    def to_single(self) -> SingleUTube:
        # Find an equivalent single U-tube given a coaxial heat exchanger
        vol_fluid, vol_pipe, resist_conv, resist_pipe = self.concentric_tube_volumes()

        preliminary = self.equivalent_single_u_tube(vol_fluid, vol_pipe, resist_conv, resist_pipe)

        # Vary grout thermal conductivity to match effective borehole thermal
        # resistance
        new_single_u_tube = self.match_effective_borehole_resistance(preliminary)

        return new_single_u_tube

    @staticmethod
    def compute_reynolds_concentric(m_flow_pipe: float, r_a_in: float, r_a_out: float, fluid: GHEFluid) -> float:
        # Hydraulic diameter and radius for concentric tube annulus region
        dia_hydraulic = 2 * (r_a_out - r_a_in)
        # r_h = dia_hydraulic / 2
        # Cross-sectional area of the annulus region
        area_cr_annular = pi * ((r_a_out**2) - (r_a_in**2))
        # Volume flow rate
        vol_flow_rate = m_flow_pipe / fluid.rho
        # Average velocity
        velocity = vol_flow_rate / area_cr_annular
        # Reynolds number
        return fluid.rho * velocity * dia_hydraulic / fluid.mu

    def as_dict(self) -> dict:
        blob = {}
        blob['type'] = str(self.__class__)
        blob['mass_flow_borehole'] = {'value': self.m_flow_borehole, 'units': 'kg/s'}
        blob['mass_flow_pipe'] = {'value': self.m_flow_borehole, 'units': 'kg/s'}
        # blob['borehole'] = self.as_dict()
        blob['soil'] = self.soil.as_dict()
        blob['grout'] = self.grout.as_dict()
        blob['pipe'] = self.pipe.as_dict()
        # blob['fluid'] = self.fluid.as_dict()
        reynold_no = self.compute_reynolds_concentric(
            self.m_flow_borehole, self.pipe.r_in, self.pipe.roughness, self.fluid
        )
        blob['reynolds'] = {'value': reynold_no, 'units': ''}
        # blob['convection_coefficient'] = {'value': self.h_f, 'units': 'W/m2-K'}
        # blob['pipe_resistance'] = {'value': self.R_p, 'units': 'm-K/W'}
        # blob['fluid_resistance'] = {'value': self.R_f, 'units': 'm-K/W'}
        return blob

    def concentric_tube_volumes(self) -> Tuple[float, float, float, float]:
        # Unpack the radii to reduce confusion in the future
        r_in_in, r_in_out = self.r_inner
        r_out_in, r_out_out = self.r_outer
        # Compute volumes for concentric ghe geometry
        vol_fluid = pi * ((r_in_in**2) + (r_out_in**2) - (r_in_out**2))
        vol_pipe = pi * ((r_in_out**2) - (r_in_in**2) + (r_out_out**2) - (r_out_in**2))
        # V_grout = pi * ((coaxial.b.r_b**2) - (r_out_out**2))
        area_surf_outer = pi * 2 * r_out_in
        resist_conv = 1 / (self.h_f_a_in * area_surf_outer)
        resist_pipe = log(r_out_out / r_out_in) / (TWO_PI * self.pipe.k[1])
        return vol_fluid, vol_pipe, resist_conv, resist_pipe


def get_bhe_object(
    bhe_type: BHPipeType,
    m_flow_borehole: float,
    fluid: GHEFluid,
    _borehole: GHEBorehole,
    pipe: Pipe,
    grout: Grout,
    soil: Soil,
):
    if bhe_type == BHPipeType.SINGLEUTUBE:
        return SingleUTube(m_flow_borehole, fluid, _borehole, pipe, grout, soil)
    elif bhe_type == BHPipeType.DOUBLEUTUBEPARALLEL:
        return MultipleUTube(m_flow_borehole, fluid, _borehole, pipe, grout, soil, config=DoubleUTubeConnType.PARALLEL)
    elif bhe_type == BHPipeType.DOUBLEUTUBESERIES:
        return MultipleUTube(m_flow_borehole, fluid, _borehole, pipe, grout, soil, config=DoubleUTubeConnType.SERIES)
    elif bhe_type == BHPipeType.COAXIAL:
        return CoaxialPipe(m_flow_borehole, fluid, _borehole, pipe, grout, soil)
    else:
        raise TypeError("BHE type not implemented")
