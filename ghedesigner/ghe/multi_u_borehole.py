from copy import deepcopy

import pygfunction as gt
from numpy import log, sqrt
from pygfunction.boreholes import Borehole

from ghedesigner.constants import TWO_PI, pi
from ghedesigner.enums import DoubleUTubeConnType
from ghedesigner.ghe.borehole_base import GHEDesignerBoreholeBase
from ghedesigner.ghe.single_u_borehole import SingleUTube
from ghedesigner.media import GHEFluid, Grout, Pipe, Soil
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
        self, vol_fluid: float, vol_pipe: float, resist_conv: float, resist_pipe: float
    ) -> SingleUTube:
        # Note: BHE can be double U-tube or coaxial heat exchanger

        # Compute equivalent single U-tube geometry
        n = 2
        r_p_i_prime = sqrt(vol_fluid / (n * pi))
        r_p_o_prime = sqrt((vol_fluid + vol_pipe) / (n * pi))
        # A_s_prime = n * pi * ((r_p_i_prime * 2) ** 2)
        # h_prime = 1 / (R_conv * A_s_prime)
        k_p_prime = log(r_p_o_prime / r_p_i_prime) / (TWO_PI * n * resist_pipe)

        # Place single u-tubes at a B-spacing
        # Total horizontal space (m)
        # TODO: investigate why this deepcopy is required
        _borehole = deepcopy(self.b)
        spacing = _borehole.r_b * 2 - (n * r_p_o_prime * 2)
        # If the spacing is negative, then the borehole is not large enough,
        # therefore, the borehole will be increased if necessary
        if spacing <= 0.0:
            _borehole.r_b -= spacing  # Add on the necessary spacing to fit
            spacing = (_borehole.r_b * 2.0) / 10.0  # make spacing 1/10th of diameter
            _borehole.r_b += spacing
        s = spacing / 3  # outer tube-to-tube shank spacing (m)
        pos = Pipe.place_pipes(s, r_p_o_prime, 1)  # Place single u-tube pipe

        # New pipe geometry
        roughness = self.pipe.roughness
        rho_cp = self.pipe.rhoCp
        pipe = Pipe(pos, r_p_i_prime, r_p_o_prime, s, roughness, k_p_prime, rho_cp)

        # Don't tie together the original and equivalent BHEs
        m_flow_borehole = self.m_flow_borehole
        fluid = self.fluid

        # TODO: investigate why this deepcopy is required
        grout = deepcopy(self.grout)
        soil = self.soil

        # Maintain the same mass flow rate so that the Rb/Rb* is not diverged from
        eq_single_u_tube = SingleUTube(m_flow_borehole, fluid, _borehole, pipe, grout, soil)

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

        # Solve for the mass flow rate to make the convection values equal
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
            preliminary_new_single_u_tube.k_g = k_g_in
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
        preliminary_new_single_u_tube.k_g = k_g
        preliminary_new_single_u_tube.grout.k = k_g

        return preliminary_new_single_u_tube


class MultipleUTube(gt.pipes.MultipleUTube, GHEDesignerBoreholeWithMultiplePipes):
    def __init__(
        self,
        m_flow_borehole: float,
        fluid: GHEFluid,
        _borehole: Borehole,
        pipe: Pipe,
        grout: Grout,
        soil: Soil,
        config=DoubleUTubeConnType.PARALLEL,
    ) -> None:
        self.R_p = 0.0
        self.R_f = 0.0
        self.R_fp = 0.0
        self.h_f = 0.0
        self.fluid = fluid
        self.m_flow_borehole = m_flow_borehole
        self.m_flow_pipe = self.calc_mass_flow_pipe(self.m_flow_borehole, config)
        self.borehole = _borehole
        self.pipe = pipe
        self.soil = soil
        self.grout = grout
        self.flow_config = config

        # Get number of pipes from positions
        self.resist_delta = None
        self.n_pipes = len(pipe.pos) / 2

        # compute resistances required to construct inherited class
        self.calc_fluid_pipe_resistance()

        gt.pipes.MultipleUTube.__init__(
            self,
            self.pipe.pos,
            self.pipe.r_in,
            self.pipe.r_out,
            self.borehole,
            self.soil.k,
            self.grout.k,
            self.R_fp,
            self.pipe.n_pipes,
            config=config.name,
        )

        # these methods must be called after inherited class construction
        self.update_thermal_resistances(self.R_fp)
        self.calc_effective_borehole_resistance()

    def calc_fluid_pipe_resistance(self) -> float:
        self.h_f = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
            self.m_flow_pipe,
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

    def u_tube_volumes(self) -> tuple[float, float, float, float]:
        # Compute volumes for U-tube geometry
        # Effective parameters
        n = self.nPipes * 2  # Total number of tubes
        # Total inside surface area (m^2)
        area_surf_inner = n * pi * (self.r_in * 2.0) ** 2
        resist_conv = 1 / (self.h_f * area_surf_inner)  # Convection resistance (m.K/W)
        # Volumes
        vol_fluid = n * pi * (self.r_in**2)
        vol_pipe = n * pi * (self.r_out**2) - vol_fluid
        # V_grout = pi * (u_tube.b.r_b**2) - vol_pipe - vol_fluid
        resist_pipe = log(self.r_out / self.r_in) / (n * TWO_PI * self.pipe.k)
        return vol_fluid, vol_pipe, resist_conv, resist_pipe

    def to_single(self) -> SingleUTube:
        # Find an equivalent single U-tube given multiple U-tube geometry

        # Get effective parameters for the multiple u-tube
        vol_fluid, vol_pipe, resist_conv, resist_pipe = self.u_tube_volumes()

        single_u_tube = self.equivalent_single_u_tube(vol_fluid, vol_pipe, resist_conv, resist_pipe)

        # Vary grout thermal conductivity to match effective borehole thermal resistance
        self.match_effective_borehole_resistance(single_u_tube)

        return single_u_tube
