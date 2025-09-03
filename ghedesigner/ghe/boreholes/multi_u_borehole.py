from copy import deepcopy
from typing import cast

from bhr.borehole import Borehole as BHRBorehole
from numpy import log, sqrt
from pygfunction.boreholes import Borehole

from ghedesigner.constants import PI, TWO_PI
from ghedesigner.enums import DoubleUTubeConnType
from ghedesigner.ghe.boreholes.base import GHEDesignerBoreholeBase
from ghedesigner.ghe.boreholes.single_u_borehole import SingleUTube
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.media import Fluid, Grout, Soil
from ghedesigner.utilities import solve_root


class GHEDesignerBoreholeWithMultiplePipes(GHEDesignerBoreholeBase):
    @staticmethod
    def calc_mass_flow_pipe(m_flow_borehole: float, config: DoubleUTubeConnType | None = None) -> float:
        if config == DoubleUTubeConnType.SERIES or config is None:
            return m_flow_borehole
        elif config == DoubleUTubeConnType.PARALLEL:
            return m_flow_borehole / 2.0
        else:
            raise ValueError(f"Invalid flow configuration: {config!s}")

    def equivalent_single_u_tube(
        self, vol_fluid: float, vol_pipe: float, resist_conv: float, resist_pipe: float, rho_cp_p: float
    ) -> SingleUTube:
        # Note: BHE can be double U-tube or coaxial heat exchanger

        # Compute equivalent single U-tube geometry
        n = 2
        r_p_i_prime = sqrt(vol_fluid / (n * PI))
        r_p_o_prime = sqrt((vol_fluid + vol_pipe) / (n * PI))
        k_p_prime = log(r_p_o_prime / r_p_i_prime) / (TWO_PI * n * resist_pipe)

        # Place single u-tubes at a B-spacing
        # Total horizontal space (m)
        # TODO: investigate why this deepcopy is required
        borehole = deepcopy(self.borehole)
        spacing = borehole.r_b * 2 - (n * r_p_o_prime * 2)
        # If the spacing is negative, then the borehole is not large enough,
        # therefore, the borehole will be increased if necessary
        if spacing <= 0.0:
            borehole.r_b -= spacing  # Add on the necessary spacing to fit
            spacing = (borehole.r_b * 2.0) / 10.0  # make spacing 1/10th of diameter
            borehole.r_b += spacing
        s = spacing / 3  # outer tube-to-tube shank spacing (m)

        # New pipe geometry
        roughness = self.pipe.roughness
        pipe = Pipe.init_single_u_tube(k_p_prime, rho_cp_p, r_p_i_prime * 2, r_p_o_prime * 2, s, roughness, 1)

        # Don't tie together the original and equivalent BHEs
        m_flow_borehole = self.m_flow_borehole
        fluid = self.fluid

        grout = Grout(self.grout.k, self.grout.rho_cp)
        soil = self.soil

        # Maintain the same mass flow rate so that the Rb/Rb* is not diverged from
        eq_single_u_tube = SingleUTube(m_flow_borehole, fluid, borehole, pipe, grout, soil)

        # The thermal conductivity of the pipe must now be varied such that R_fp is
        # equivalent to R_fp_prime
        def objective_pipe_conductivity(pipe_k: float):
            eq_single_u_tube.pipe.k = pipe_k
            eq_single_u_tube.calc_fluid_pipe_resistance()
            return eq_single_u_tube.R_fp - (resist_conv + resist_pipe)

        # Use Brent Quadratic to find the root
        # Define a lower and upper for pipe thermal conductivities
        k_p_lower = eq_single_u_tube.pipe.k / 100.0
        k_p_upper = eq_single_u_tube.pipe.k * 10.0

        # Solve for the pipe conductivity to make the resistance equal
        solve_root(
            eq_single_u_tube.pipe.k,
            objective_pipe_conductivity,
            lower=k_p_lower,
            upper=k_p_upper,
        )

        return eq_single_u_tube

    def match_effective_borehole_resistance(self, preliminary_new_single_u_tube: SingleUTube) -> SingleUTube:
        # Find the thermal conductivity that makes the borehole resistances equal

        # Define objective function for varying the grout thermal conductivity
        def objective_resistance(k_g_in: float):
            # update new tubes grout thermal conductivity and relevant parameters
            preliminary_new_single_u_tube.grout.k = k_g_in
            # Update Delta-circuit thermal resistances
            # Initialize stored_coefficients
            resist_bh_prime = preliminary_new_single_u_tube.calc_effective_borehole_resistance()
            resist_bh = self.calc_effective_borehole_resistance()
            return resist_bh - resist_bh_prime

        # Use Brent Quadratic to find the root
        # Define a lower and upper for thermal conductivities
        kg_lower = 1e-02
        kg_upper = 7.0
        k_g = solve_root(preliminary_new_single_u_tube.grout.k, objective_resistance, lower=kg_lower, upper=kg_upper)
        # Ensure the grout thermal conductivity is updated
        preliminary_new_single_u_tube.grout.k = k_g

        return preliminary_new_single_u_tube


class MultipleUTube(GHEDesignerBoreholeWithMultiplePipes):
    def __init__(
        self,
        m_flow_borehole: float,
        fluid: Fluid,
        borehole: Borehole,
        pipe: Pipe,
        grout: Grout,
        soil: Soil,
        config=DoubleUTubeConnType.PARALLEL,
    ) -> None:
        # Ensure r_in and r_out are floats rather than list[float]
        if not isinstance(pipe.r_out, float) or not isinstance(pipe.r_in, float):
            raise TypeError("pipe r_in and r_out must be floats")

        super().__init__(m_flow_borehole, fluid, borehole, pipe, grout, soil)

        self.R_fp = 0.0
        self.fluid = fluid
        self.m_flow_borehole = m_flow_borehole
        self.m_flow_pipe = self.calc_mass_flow_pipe(self.m_flow_borehole, config)
        self.borehole = borehole
        self.pipe = pipe
        self.soil = soil
        self.grout = grout
        self.flow_config = config
        self.bhr_borehole = BHRBorehole()
        self.n_pipes = 2

        self.bhr_borehole.init_double_u_borehole(
            borehole_diameter=borehole.r_b * 2,
            pipe_outer_diameter=(2.0 * pipe.r_out),
            pipe_dimension_ratio=(2.0 * pipe.r_out) / (pipe.r_out - pipe.r_in),
            length=borehole.H,
            shank_space=(pipe.s / 2.0 + pipe.r_out),
            pipe_conductivity=pipe.k,
            grout_conductivity=grout.k,
            soil_conductivity=soil.k,
            fluid_type=self.fluid.name,
            fluid_concentration=self.fluid.concentration_percent / 100,
            boundary_condition="UNIFORM_BOREHOLE_WALL_TEMP",
            pipe_inlet_arrangement="ADJACENT",
        )

        self.calc_fluid_pipe_resistance()

    def calc_fluid_pipe_resistance(self) -> float:
        self.R_fp = self.bhr_borehole.calc_fluid_pipe_resist(self.m_flow_borehole, self.soil.ugt)
        return self.R_fp

    def calc_effective_borehole_resistance(self) -> float:
        resist_bh_effective = self.bhr_borehole.calc_bh_resist(self.m_flow_borehole, self.soil.ugt)
        return resist_bh_effective

    def u_tube_volumes(self) -> tuple[float, float]:
        # Compute volumes for U-tube geometry
        # Effective parameters
        n = self.n_pipes * 2  # Total number of tubes

        # Volumes
        vol_fluid = n * PI * (cast(float, self.pipe.r_in) ** 2)
        vol_pipe = n * PI * (cast(float, self.pipe.r_out) ** 2) - vol_fluid

        return vol_fluid, vol_pipe

    def to_single(self) -> SingleUTube:
        # Find an equivalent single U-tube given multiple U-tube geometry

        # Get effective parameters for the multiple u-tube
        vol_fluid, vol_pipe = self.u_tube_volumes()

        # TODO: check that the usage here for converting to an equivalent single u-tube is accurate
        resist_pipe = self.bhr_borehole.calc_pipe_cond_resist()
        resist_conv = self.R_fp - resist_pipe

        single_u_tube = self.equivalent_single_u_tube(vol_fluid, vol_pipe, resist_conv, resist_pipe, self.pipe.rho_cp)

        # Vary grout thermal conductivity to match effective borehole thermal resistance
        self.match_effective_borehole_resistance(single_u_tube)

        return single_u_tube
