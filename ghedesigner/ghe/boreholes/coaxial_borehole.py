from typing import cast

from bhr.borehole import Borehole as BHRBorehole
from pygfunction.boreholes import Borehole

from ghedesigner.constants import PI
from ghedesigner.ghe.boreholes.multi_u_borehole import GHEDesignerBoreholeWithMultiplePipes
from ghedesigner.ghe.boreholes.single_u_borehole import SingleUTube
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.media import Fluid, Grout, Soil


class CoaxialPipe(GHEDesignerBoreholeWithMultiplePipes):
    def __init__(
        self, m_flow_borehole: float, fluid: Fluid, borehole: Borehole, pipe: Pipe, grout: Grout, soil: Soil
    ) -> None:
        super().__init__(m_flow_borehole, fluid, borehole, pipe, grout, soil)

        # Ensure r_in and r_out are list[float] rather than floats
        if not isinstance(pipe.r_out, list) or not isinstance(pipe.r_in, list):
            raise TypeError("pipe r_in and r_out must be list[float]")

        # Unpack the radii to reduce confusion in the future
        # Pipe naming nomenclature
        # <var>_<inner/outer pipe>_<inner/outer surface>
        # e.g. r_in_in is inner radius of the inner pipe
        self.r_inner = cast(tuple[float, float], pipe.r_in)
        self.r_outer = cast(tuple[float, float], pipe.r_out)
        self.r_in_in, self.r_in_out = self.r_inner
        self.r_out_in, self.r_out_out = self.r_outer

        # Store pipe roughness
        self.roughness = self.pipe.roughness

        self.bhr_borehole = BHRBorehole()
        self.bhr_borehole.init_coaxial_borehole(
            borehole_diameter=borehole.r_b * 2,
            outer_pipe_outer_diameter=(2.0 * self.r_out_out),
            outer_pipe_dimension_ratio=(2.0 * self.r_out_out) / (self.r_out_out - self.r_out_in),
            outer_pipe_conductivity=pipe.k[1],
            inner_pipe_outer_diameter=(2.0 * self.r_in_out),
            inner_pipe_dimension_ratio=(2.0 * self.r_in_out) / (self.r_in_out - self.r_in_in),
            inner_pipe_conductivity=pipe.k[0],
            length=borehole.H,
            grout_conductivity=grout.k,
            soil_conductivity=soil.k,
            fluid_type=self.fluid.name,
            fluid_concentration=self.fluid.concentration_percent / 100,
            boundary_condition="UNIFORM_BOREHOLE_WALL_TEMP",
        )

    def calc_effective_borehole_resistance(self) -> float:
        resist_bh_effective = self.bhr_borehole.calc_bh_resist(self.m_flow_borehole, self.soil.ugt)
        return resist_bh_effective

    def to_single(self) -> SingleUTube:
        # Find an equivalent single U-tube given a coaxial heat exchanger
        vol_fluid, vol_pipe = self.concentric_tube_volumes()

        resist_conv = self.bhr_borehole.calc_fluid_resist(self.m_flow_borehole, self.soil.ugt)
        resist_pipe = self.bhr_borehole.calc_pipe_cond_resist()
        preliminary = self.equivalent_single_u_tube(vol_fluid, vol_pipe, resist_conv, resist_pipe, self.pipe.rho_cp)

        # Vary grout thermal conductivity to match effective borehole thermal
        # resistance
        new_single_u_tube = self.match_effective_borehole_resistance(preliminary)

        return new_single_u_tube

    @staticmethod
    def compute_reynolds_concentric(m_flow_pipe: float, r_a_in: float, r_a_out: float, fluid: Fluid) -> float:
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
        reynold_no = self.compute_reynolds_concentric(self.m_flow_borehole, self.r_in_out, self.r_out_in, self.fluid)
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
        # Compute volumes for concentric ghe geometry
        vol_fluid = PI * ((self.r_in_in**2) + (self.r_out_in**2) - (self.r_in_out**2))
        vol_pipe = PI * ((self.r_in_out**2) - (self.r_in_in**2) + (self.r_out_out**2) - (self.r_out_in**2))
        return vol_fluid, vol_pipe
