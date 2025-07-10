from typing import cast

import pygfunction as gt
from bhr.borehole import Borehole as BHRBorehole
from pygfunction.boreholes import Borehole

from ghedesigner.constants import PI
from ghedesigner.ghe.boreholes.multi_u_borehole import GHEDesignerBoreholeWithMultiplePipes
from ghedesigner.ghe.boreholes.single_u_borehole import SingleUTube
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.media import GHEFluid, Grout, Soil


class CoaxialPipe(gt.pipes.Coaxial, GHEDesignerBoreholeWithMultiplePipes):
    def __init__(
        self, m_flow_borehole: float, fluid: GHEFluid, _borehole: Borehole, pipe: Pipe, grout: Grout, soil: Soil
    ) -> None:
        # Ensure r_in and r_out are list[float] rather than floats
        if not isinstance(pipe.r_out, list) or not isinstance(pipe.r_in, list):
            raise TypeError("pipe r_in and r_out must be list[float]")

        # Unpack the radii to reduce confusion in the future
        self.r_inner = cast(tuple[float, float], pipe.r_in)
        self.r_outer = cast(tuple[float, float], pipe.r_out)
        self.r_in_in, self.r_in_out = self.r_inner
        self.r_out_in, self.r_out_out = self.r_outer

        self.m_flow_borehole = m_flow_borehole
        # Store Thermal properties
        self.soil = soil
        self.grout = grout
        self.pipe = pipe
        # Store fluid properties
        self.fluid = fluid
        # Store pipe roughness
        self.roughness = self.pipe.roughness

        self.bhr_borehole = BHRBorehole()
        self.bhr_borehole.init_coaxial_borehole(
            borehole_diameter=_borehole.r_b * 2,
            outer_pipe_outer_diameter=(2.0 * self.r_out_out),
            outer_pipe_dimension_ratio=(2.0 * self.r_out_out) / (self.r_out_out - self.r_out_in),
            outer_pipe_conductivity=pipe.k[1],
            inner_pipe_outer_diameter=(2.0 * self.r_in_out),
            inner_pipe_dimension_ratio=(2.0 * self.r_in_out) / (self.r_in_out - self.r_in_in),
            inner_pipe_conductivity=pipe.k[0],
            length=_borehole.H,
            grout_conductivity=grout.k,
            soil_conductivity=soil.k,
            fluid_type=self.fluid.name,
            fluid_concentration=self.fluid.concentration_percent / 100,
            boundary_condition="UNIFORM_BOREHOLE_WALL_TEMP",
        )

        # Pipe naming nomenclature
        # <var>_<inner/outer pipe>_<inner/outer surface>
        # e.g. r_in_in is inner radius of the inner pipe

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

        # # compute resistances required to construct inherited class
        # self.calc_fluid_pipe_resistance()

        # Vectors of inner and outer pipe radii
        # Note: The dimensions of the inlet pipe are the first elements of the vectors.
        # In this example, the inlet pipe is the inside pipe.
        # TODO: fix this
        # r_in = cast(tuple[float, float], pipe.r_in)
        # r_out = cast(tuple[float, float], pipe.r_out)
        # r_inner_p = np.array([r_in[0], r_out[0]])  # Inner pipe radii (m)
        # r_outer_p = np.array([r_in[1], r_out[1]])  # Outer pipe radii (m)

        # gt.pipes.Coaxial.__init__(
        #     self, pipe.pos, r_inner_p, r_outer_p, _borehole, self.soil.k, self.grout.k, self.R_ff, self.R_fp
        # )
        #
        # # these methods must be called after inherited class construction
        # self.update_thermal_resistances(self.R_ff, self.R_fp)
        self.calc_effective_borehole_resistance()

    # def calc_fluid_pipe_resistance(self):
    #     # inner pipe convection resistance
    #     self.h_f_in = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
    #         self.m_flow_borehole,
    #         self.r_in_in,
    #         self.fluid.mu,
    #         self.fluid.rho,
    #         self.fluid.k,
    #         self.fluid.cp,
    #         self.pipe.roughness,
    #     )
    #     self.R_f_in = self.compute_fluid_resistance(self.h_f_in, self.r_in_in)
    #
    #     # inner pipe conduction resistance
    #     self.R_p_in = gt.pipes.conduction_thermal_resistance_circular_pipe(self.r_in_in, self.r_in_out,
    #     self.pipe.k[0])
    #
    #     # annulus convection resistances
    #     self.h_f_a_in, self.h_f_a_out = gt.pipes.convective_heat_transfer_coefficient_concentric_annulus(
    #         self.m_flow_borehole,
    #         self.r_in_out,
    #         self.r_out_in,
    #         self.fluid.mu,
    #         self.fluid.rho,
    #         self.fluid.k,
    #         self.fluid.cp,
    #         self.roughness,
    #     )
    #
    #     self.R_f_a_in = self.compute_fluid_resistance(self.h_f_a_in, self.r_in_out)
    #     self.R_f_a_out = self.compute_fluid_resistance(self.h_f_a_out, self.r_out_in)
    #
    #     # inner pipe conduction resistance
    #     self.R_p_out = gt.pipes.conduction_thermal_resistance_circular_pipe(
    #         self.r_out_in, self.r_out_out, self.pipe.k[1]
    #     )
    #
    #     # inner fluid to inner annulus fluid resistance
    #     self.R_ff = self.R_f_in + self.R_p_in + self.R_f_a_in
    #
    #     # outer annulus fluid to pipe thermal resistance
    #     self.R_fp = self.R_p_out + self.R_f_a_out
    #
    #     return self.R_fp

    def calc_effective_borehole_resistance(self) -> float:
        resist_bh_effective = self.bhr_borehole.calc_bh_resist(self.m_flow_borehole, self.soil.ugt)
        return resist_bh_effective

    def to_single(self) -> SingleUTube:
        # Find an equivalent single U-tube given a coaxial heat exchanger
        vol_fluid, vol_pipe = self.concentric_tube_volumes()

        # TODO: fix this
        resist_conv = 0.002
        resist_pipe = 0.015
        preliminary = self.equivalent_single_u_tube(vol_fluid, vol_pipe, resist_conv, resist_pipe, self.pipe.rhoCp)

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
        area_cr_annular = PI * ((r_a_out**2) - (r_a_in**2))
        # Volume flow rate
        vol_flow_rate = m_flow_pipe / fluid.rho
        # Average velocity
        velocity = vol_flow_rate / area_cr_annular
        # Reynolds number
        return fluid.rho * velocity * dia_hydraulic / fluid.mu

    def as_dict(self) -> dict:
        # TODO: This is actually a tuple of radii, right?
        r_in = cast(float, self.pipe.r_in)
        # TODO: This is passing roughness, but shouldn't, right?
        reynold_no = self.compute_reynolds_concentric(self.m_flow_borehole, r_in, self.pipe.roughness, self.fluid)
        blob = {
            "type": str(self.__class__),
            "mass_flow_borehole": {"value": self.m_flow_borehole, "units": "kg/s"},
            "mass_flow_pipe": {"value": self.m_flow_borehole, "units": "kg/s"},
            "soil": self.soil.as_dict(),
            "grout": self.grout.as_dict(),
            "pipe": self.pipe.as_dict(),
            "reynolds": {"value": reynold_no, "units": ""},
        }
        return blob

    def concentric_tube_volumes(self) -> tuple[float, float]:
        # Unpack the radii to reduce confusion in the future
        r_inner = cast(tuple[float, float], self.r_inner)
        r_outer = cast(tuple[float, float], self.r_outer)
        r_in_in, r_in_out = r_inner
        r_out_in, r_out_out = r_outer
        # Compute volumes for concentric ghe geometry
        vol_fluid = PI * ((r_in_in**2) + (r_out_in**2) - (r_in_out**2))
        vol_pipe = PI * ((r_in_out**2) - (r_in_in**2) + (r_out_out**2) - (r_out_in**2))

        return vol_fluid, vol_pipe
