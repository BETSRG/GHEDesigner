from copy import deepcopy
import numpy as np
import pygfunction as gt
from pygfunction.pipes import _BasePipe as bp

from typing import Tuple

from numpy import pi, log, sqrt

from ghedesigner import media, utilities, borehole


class GHEDesignerBoreholeBase:
    def __init__(
            self,
            m_flow_borehole: float,
            fluid: media.GHEFluid,
            _borehole: borehole.GHEBorehole,
            pipe: media.Pipe,
            grout: media.Grout,
            soil: media.Soil,
    ):
        self.m_flow_pipe = m_flow_borehole
        self.m_flow_borehole = m_flow_borehole
        self.borehole = _borehole
        self.pipe = pipe
        self.soil = soil
        self.grout = grout
        self.fluid = fluid
        self.b = _borehole

    def compute_effective_borehole_resistance(self, m_flow_borehole: float = None,
                                              fluid: media.GHEFluid = None) -> float:
        pass


class SingleUTube(gt.pipes.SingleUTube, GHEDesignerBoreholeBase):
    def __init__(
            self,
            m_flow_borehole: float,
            fluid: media.GHEFluid,
            _borehole: borehole.GHEBorehole,
            pipe: media.Pipe,
            grout: media.Grout,
            soil: media.Soil,
    ):
        GHEDesignerBoreholeBase.__init__(self, m_flow_borehole, fluid, _borehole, pipe, grout, soil)
        self.R_p = 0.0
        self.R_f = 0.0
        self.R_fp = 0.0
        self.h_f = 0.0
        self.fluid = fluid
        self.m_flow_borehole = m_flow_borehole

        # TODO: May need to revisit this for series/parallel connections
        self.m_flow_pipe = m_flow_borehole
        self.borehole = _borehole
        self.pipe = pipe
        self.soil = soil
        self.grout = grout

        # Compute Single pipe and fluid thermal resistance
        self.update_pipe_fluid_resist(m_flow_borehole)

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

        self.resist_delta = self.update_thermal_resistance(m_flow_borehole)

    def update_pipe_fluid_resist(self, m_flow_borehole: float) -> None:
        self.h_f = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(m_flow_borehole,
                                                                               self.pipe.r_in,
                                                                               self.fluid.mu,
                                                                               self.fluid.rho,
                                                                               self.fluid.k,
                                                                               self.fluid.cp,
                                                                               self.pipe.roughness)
        self.R_f = compute_fluid_resistance(self.h_f, self.pipe.r_in)
        self.R_p = gt.pipes.conduction_thermal_resistance_circular_pipe(self.pipe.r_in, self.pipe.r_out, self.pipe.k)
        self.R_fp = self.R_f + self.R_p

    def update_thermal_resistance(self, m_flow_borehole: float = None) -> float:

        # if the mass flow rate has changed, then update it and use new value
        if m_flow_borehole is None:
            m_flow_borehole = self.m_flow_borehole
        else:
            self.m_flow_borehole = m_flow_borehole

        self.update_pipe_fluid_resist(self.m_flow_borehole)

        # Delta-circuit thermal resistances
        self.resist_delta = gt.pipes.thermal_resistances(
            self.pipe.pos, self.pipe.r_out, self.borehole.r_b, self.soil.k, self.grout.k, self.R_fp)[1]

        # Initialize stored_coefficients
        # TODO: Figure out why this is here...
        self._initialize_stored_coefficients()

        # Note: This is the local borehole resistance
        # # Evaluate borehole thermal resistance
        # R_b = 1 / np.trace(1 / self.resist_delta)

        # Compute and return effective borehole resistance
        resist_bh_effective = self.effective_borehole_thermal_resistance(m_flow_borehole, self.fluid.cp)

        return resist_bh_effective

    def compute_effective_borehole_resistance(self, m_flow_borehole: float = None,
                                              fluid: media.GHEFluid = None) -> float:
        return self.update_thermal_resistance(m_flow_borehole)

    def to_single(self):
        return self

    def as_dict(self) -> dict:
        return {'type': str(self.__class__)}


class GHEDesignerBoreholeWithMultiplePipes(GHEDesignerBoreholeBase):

    def equivalent_single_u_tube(self, vol_fluid: float, vol_pipe: float, resist_conv: float,
                                 resist_pipe: float) -> SingleUTube:
        # Note: BHE can be double U-tube or coaxial heat exchanger

        # Compute equivalent single U-tube geometry
        n = 2
        r_p_i_prime = sqrt(vol_fluid / (n * pi))
        r_p_o_prime = sqrt((vol_fluid + vol_pipe) / (n * pi))
        # A_s_prime = n * pi * ((r_p_i_prime * 2) ** 2)
        # h_prime = 1 / (R_conv * A_s_prime)
        k_p_prime = log(r_p_o_prime / r_p_i_prime) / (2 * pi * n * resist_pipe)

        # Place single u-tubes at a B-spacing
        # Total horizontal space (m)
        _borehole = deepcopy(self.b)
        spacing = _borehole.r_b * 2 - (n * r_p_o_prime * 2)
        # If the spacing is negative, then the borehole is not large enough,
        # therefore, the borehole will be increased if necessary
        if spacing <= 0.0:
            _borehole.r_b -= spacing  # Add on the necessary spacing to fit
            spacing = (_borehole.r_b * 2.0) / 10.0  # make spacing 1/10th of diameter
            _borehole.r_b += spacing
        s = spacing / 3  # outer tube-to-tube shank spacing (m)
        pos = media.Pipe.place_pipes(s, r_p_o_prime, 1)  # Place single u-tube pipe

        # New pipe geometry
        roughness = deepcopy(self.pipe.roughness)
        rho_cp = deepcopy(self.pipe.rhoCp)
        pipe = media.Pipe(pos, r_p_i_prime, r_p_o_prime, s, roughness, k_p_prime, rho_cp)

        # Don't tie together the original and equivalent BHEs
        m_flow_borehole = deepcopy(self.m_flow_borehole)
        fluid = deepcopy(self.fluid)
        grout = deepcopy(self.grout)
        soil = deepcopy(self.soil)

        # Maintain the same mass flow rate so that the Rb/Rb* is not diverged from
        eq_single_u_tube = SingleUTube(
            m_flow_borehole, fluid, _borehole, pipe, grout, soil
        )

        # The thermal conductivity of the pipe must now be varied such that R_fp is
        # equivalent to R_fp_prime
        def objective_pipe_conductivity(pipe_k):
            eq_single_u_tube.pipe.k = pipe_k
            eq_single_u_tube.update_thermal_resistance()
            return eq_single_u_tube.R_fp - (resist_conv + resist_pipe)

        # Use Brent Quadratic to find the root
        # Define a lower and upper for pipe thermal conductivities
        k_p_lower = eq_single_u_tube.pipe.k / 100.0
        k_p_upper = eq_single_u_tube.pipe.k * 10.0

        # Solve for the mass flow rate to make the convection values equal
        utilities.solve_root(
            eq_single_u_tube.pipe.k,
            objective_pipe_conductivity,
            lower=k_p_lower,
            upper=k_p_upper,
        )

        eq_single_u_tube.update_thermal_resistance()

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
            resist_bh_prime = preliminary_new_single_u_tube.update_thermal_resistance(m_flow_borehole=None)
            resist_bh = self.compute_effective_borehole_resistance()
            return resist_bh - resist_bh_prime

        # Use Brent Quadratic to find the root
        # Define a lower and upper for thermal conductivities
        kg_lower = 1e-02
        kg_upper = 7.0
        k_g = utilities.solve_root(
            preliminary_new_single_u_tube.grout.k, objective_resistance, lower=kg_lower, upper=kg_upper
        )
        # Ensure the grout thermal conductivity is updated
        preliminary_new_single_u_tube.k_g = k_g
        preliminary_new_single_u_tube.grout.k = k_g
        preliminary_new_single_u_tube.update_thermal_resistance()

        return preliminary_new_single_u_tube


class MultipleUTube(gt.pipes.MultipleUTube, GHEDesignerBoreholeWithMultiplePipes):
    def __init__(
            self,
            m_flow_borehole: float,
            fluid: media.GHEFluid,
            borehole: borehole.GHEBorehole,
            pipe: media.Pipe,
            grout: media.Grout,
            soil: media.Soil,
            config="parallel",
    ):

        self.R_p = 0.0
        self.R_f = 0.0
        self.R_fp = 0.0
        self.h_f = 0.0
        self.fluid = fluid
        self.m_flow_borehole = m_flow_borehole

        # TODO: May need to revisit this for series/parallel connections
        self.m_flow_pipe = m_flow_borehole
        self.borehole = borehole
        self.pipe = pipe
        self.soil = soil
        self.grout = grout

        # Get number of pipes from positions
        self.resist_delta = None
        self.n_pipes = int(len(pipe.pos) / 2)

        # Compute Single pipe and fluid thermal resistance
        self.update_pipe_fluid_resist(m_flow_borehole)

        super().__init__(
            self.pipe.pos,
            self.pipe.r_in,
            self.pipe.r_out,
            self.borehole,
            self.soil.k,
            self.grout.k,
            self.R_fp,
            self.pipe.n_pipes,
            config=config,
        )

    def update_pipe_fluid_resist(self, m_flow_borehole: float) -> None:
        self.h_f = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(m_flow_borehole,
                                                                               self.pipe.r_in,
                                                                               self.fluid.mu,
                                                                               self.fluid.rho,
                                                                               self.fluid.k,
                                                                               self.fluid.cp,
                                                                               self.pipe.roughness)
        self.R_f = compute_fluid_resistance(self.h_f, self.pipe.r_in)
        self.R_p = gt.pipes.conduction_thermal_resistance_circular_pipe(self.pipe.r_in, self.pipe.r_out, self.pipe.k)
        self.R_fp = self.R_f + self.R_p

    def update_thermal_resistance(self, m_flow_borehole: float = None) -> float:

        # if the mass flow rate has changed, then update it and use new value
        if m_flow_borehole is None:
            m_flow_borehole = self.m_flow_borehole
        else:
            self.m_flow_borehole = m_flow_borehole

        self.update_pipe_fluid_resist(self.m_flow_borehole)

        # Delta-circuit thermal resistances
        self.resist_delta = gt.pipes.thermal_resistances(
            self.pos, self.r_out, self.b.r_b, self.k_s, self.k_g, self.R_fp, J=self.J
        )[1]

        # Initialize stored_coefficients
        self._initialize_stored_coefficients()

        # Note: This is the local borehole resistance
        # # Evaluate borehole thermal resistance
        # R_b = 1 / np.trace(1 / self.resist_delta)

        # Compute and return effective borehole resistance
        resist_bh_effective = self.effective_borehole_thermal_resistance(m_flow_borehole, self.fluid.cp)

        return resist_bh_effective

    def compute_effective_borehole_resistance(self, m_flow_borehole: float = None,
                                              fluid: media.GHEFluid = None) -> float:
        return self.update_thermal_resistance(m_flow_borehole)

    def u_tube_volumes(self) -> Tuple[float, float, float, float]:
        # Compute volumes for U-tube geometry
        # Effective parameters
        n = int(self.nPipes * 2)  # Total number of tubes
        # Total inside surface area (m^2)
        area_surf_inner = n * pi * (self.r_in * 2.0) ** 2
        resist_conv = 1 / (self.h_f * area_surf_inner)  # Convection resistance (m.K/W)
        # Volumes
        vol_fluid = n * pi * (self.r_in ** 2)
        vol_pipe = n * pi * (self.r_out ** 2) - vol_fluid
        # V_grout = pi * (u_tube.b.r_b**2) - vol_pipe - vol_fluid
        resist_pipe = log(self.r_out / self.r_in) / (n * 2 * pi * self.pipe.k)
        return vol_fluid, vol_pipe, resist_conv, resist_pipe

    def to_single(self) -> SingleUTube:
        # Find an equivalent single U-tube given multiple U-tube geometry

        # Get effective parameters for the multiple u-tube
        vol_fluid, vol_pipe, resist_conv, resist_pipe = self.u_tube_volumes()

        single_u_tube = self.equivalent_single_u_tube(
            vol_fluid, vol_pipe, resist_conv, resist_pipe
        )

        # Vary grout thermal conductivity to match effective borehole thermal
        # resistance
        self.match_effective_borehole_resistance(single_u_tube)

        return single_u_tube


class CoaxialPipe(gt.pipes.Coaxial, GHEDesignerBoreholeWithMultiplePipes):
    def __init__(
            self,
            m_flow_borehole: float,
            fluid: media.GHEFluid,  # TODO: Use ghedesigner.media.Fluid
            borehole: borehole.GHEBorehole,
            pipe: media.Pipe,
            grout: media.Grout,
            soil: media.Soil,
            config=None,
    ):
        self.m_flow_borehole = m_flow_borehole
        self.m_flow_pipe = m_flow_borehole
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

        # Unpack the radii to reduce confusion in the future
        self.r_in_in, self.r_in_out = self.r_inner
        self.r_out_in, self.r_out_out = self.r_outer

        self.b = borehole  # pygfunction borehole

        # Declare variables that are computed in compute_resistance()
        self.R_p_in = 0.0
        self.R_p_out = 0.0
        self.R_grout = 0.0
        self.h_fluid_a_in = 0.0
        self.h_fluid_a_out = 0.0
        self.R_f_a_in = 0.0
        self.R_f_a_out = 0.0
        self.h_fluid_in = 0.0
        self.R_f_in = 0.0
        self.R_ff = 0.0
        self.R_fp = 0.0

        self.compute_resistances()

        # borehole mass flow rate
        self.m_flow_borehole = m_flow_borehole
        # mass flow rate to each pipe
        self.m_flow_pipe = self.compute_mass_flow_rate_pipe(m_flow_borehole, config)
        # Store Thermal properties
        self.soil = soil
        self.grout = grout
        self.pipe = pipe
        # Store fluid properties
        self.fluid = fluid
        # Store borehole
        self.borehole = borehole

        # # Declare variables that are computed in compute_resistance()
        self.h_f = 0.0
        self.R_p = 0.0
        self.R_f = 0.0

        self.compute_resistances()

        # Vectors of inner and outer pipe radii
        # Note : The dimensions of the inlet pipe are the first elements of
        #       the vectors. In this example, the inlet pipe is the inside pipe.
        self.resist_delta = None
        r_inner_p = np.array([pipe.r_in[0], pipe.r_out[0]])  # Inner pipe radii (m)
        r_outer_p = np.array([pipe.r_in[1], pipe.r_out[1]])  # Outer pip radii (m)

        # Compute resistances that are stored by pygfunction
        # Inner fluid to inner annulus fluid resistance
        resist_ff = self.R_f_in + self.R_p_in + self.R_f_a_in
        # Outer annulus fluid to pipe thermal resistance
        resist_fp = self.R_p_out + self.R_f_a_out

        gt.pipes.Coaxial.__init__(
            self,
            pipe.pos,
            r_inner_p,
            r_outer_p,
            borehole,
            self.soil.k,
            self.grout.k,
            resist_ff,
            resist_fp,
            J=2,
        )

    def update_thermal_resistance(self, m_flow_borehole: float = None, fluid: media.GHEFluid = None) -> float:

        # if the mass flow rate has changed, then update it and use new value
        if m_flow_borehole is None:
            m_flow_borehole = self.m_flow_borehole
        else:
            self.m_flow_borehole = m_flow_borehole

        # if the mass flow rate has changed, then update it and use new value
        if fluid is None:
            fluid = self.fluid
        else:
            self.fluid = fluid

        self.compute_resistances()

        # Compute resistances that are stored by pygfunction
        # Inner fluid to inner annulus fluid resistance
        self.R_ff = self.R_f_in + self.R_p_in + self.R_f_a_in
        # Outer annulus fluid to pipe thermal resistance
        self.R_fp = self.R_p_out + self.R_f_a_out

        # Determine the indexes of the inner and outer pipes
        idx_inner = self.r_out.argmin()
        idx_outer = self.r_out.argmax()
        # Outer pipe to borehole wall thermal resistance
        resist_fg = gt.pipes.thermal_resistances(
            self.pos,
            self.r_out[idx_outer],
            self.b.r_b,
            self.soil.k,
            self.grout.k,
            self.R_fp,
            J=self.J,
        )[1][0]

        # Delta-circuit thermal resistances
        self.resist_delta = np.zeros((2 * self.nPipes, 2 * self.nPipes))
        self.resist_delta[idx_inner, idx_inner] = np.inf
        self.resist_delta[idx_inner, idx_outer] = self.R_ff
        self.resist_delta[idx_outer, idx_inner] = self.R_ff
        self.resist_delta[idx_outer, idx_outer] = resist_fg

        # Initialize stored_coefficients
        self._initialize_stored_coefficients()

        # Note: This is the local borehole resistance
        # # Evaluate borehole thermal resistance
        # R_b = 1 / np.trace(1 / self.resist_delta)

        # Compute and return effective borehole resistance
        resist_bh_effective = bp.effective_borehole_thermal_resistance(
            self, m_flow_borehole, fluid.cp
        )

        return resist_bh_effective

    def to_single(self) -> SingleUTube:
        # Find an equivalent single U-tube given a coaxial heat exchanger
        vol_fluid, vol_pipe, resist_conv, resist_pipe = self.concentric_tube_volumes()

        preliminary = self.equivalent_single_u_tube(
            vol_fluid, vol_pipe, resist_conv, resist_pipe
        )

        # Vary grout thermal conductivity to match effective borehole thermal
        # resistance
        new_single_u_tube = self.match_effective_borehole_resistance(preliminary)

        return new_single_u_tube

    def as_dict(self) -> dict:
        blob = dict()
        blob['type'] = str(self.__class__)
        blob['mass_flow_borehole'] = {'value': self.m_flow_borehole, 'units': 'kg/s'}
        blob['mass_flow_pipe'] = {'value': self.m_flow_pipe, 'units': 'kg/s'}
        # blob['borehole'] = self.borehole.as_dict()
        blob['soil'] = self.soil.as_dict()
        blob['grout'] = self.grout.as_dict()
        blob['pipe'] = self.pipe.as_dict()
        # blob['fluid'] = self.fluid.as_dict()
        reynold_no = compute_reynolds_concentric(
            self.m_flow_pipe, self.pipe.r_in, self.pipe.roughness, self.fluid
        )
        blob['reynolds'] = {'value': reynold_no, 'units': ''}
        blob['convection_coefficient'] = {'value': self.h_f, 'units': 'W/m2-K'}
        blob['pipe_resistance'] = {'value': self.R_p, 'units': 'm-K/W'}
        blob['fluid_resistance'] = {'value': self.R_f, 'units': 'm-K/W'}
        return blob

    # def compute_resistances(self):
    #     # Convection heat transfer coefficient of a single pipe
    #     self.h_f = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
    #         self.m_flow_pipe,
    #         self.pipe.r_in,
    #         self.fluid.mu,
    #         self.fluid.rho,
    #         self.fluid.k,
    #         self.fluid.cp,
    #         self.pipe.roughness,
    #     )
    #
    #     # Single pipe thermal resistance
    #     self.R_p = gt.pipes.conduction_thermal_resistance_circular_pipe(
    #         self.pipe.r_in, self.pipe.r_out, self.pipe.k
    #     )
    #     # Single pipe fluid thermal resistance
    #     self.R_f = 1.0 / (self.h_f * 2 * pi * self.pipe.r_in)
    #
    #     return

    @staticmethod
    def compute_mass_flow_rate_pipe(m_flow_borehole: float, config: str) -> float:
        if config == "series" or config is None:
            m_flow_pipe = m_flow_borehole
        elif config == "parallel":
            m_flow_pipe = m_flow_borehole / 2.0
        else:
            raise ValueError("No such configuration exists.")
        return m_flow_pipe

    def compute_effective_borehole_resistance(self, m_flow_borehole: float = None,
                                              fluid: media.GHEFluid = None) -> float:
        # Compute the effective borehole thermal resistance

        # if the mass flow rate has changed, then update it and use new value
        if m_flow_borehole is None:
            m_flow_borehole = self.m_flow_borehole
        else:
            self.m_flow_borehole = m_flow_borehole

        # if the mass flow rate has changed, then update it and use new value
        if fluid is None:
            fluid = self.fluid
        else:
            self.fluid = fluid

        resist_bh_effective = bp.effective_borehole_thermal_resistance(
            self, m_flow_borehole, fluid.cp
        )

        return resist_bh_effective

    # def as_dict(self) -> dict:
    #     output = dict()
    #     output['type'] = str(self.__class__)
    #     output['mass_flow_borehole'] = {'value': self.m_flow_borehole, 'units': 'kg/s'}
    #     output['mass_flow_pipe'] = {'value': self.m_flow_pipe, 'units': 'kg/s'}
    #     output['b'] = self.b.as_dict()
    #     output['soil'] = self.soil.as_dict()
    #     output['grout'] = self.grout.as_dict()
    #     output['pipe'] = self.pipe.as_dict()
    #     output['fluid'] = self.fluid.as_dict()
    #     output['inner_pipe_resistance'] = {'value': self.R_p_in, 'units': 'm-K/W'}
    #     output['outer_pipe_resistance'] = {'value': self.R_p_out, 'units': 'm-K/W'}
    #     output['grout_resistance'] = {'value': self.R_grout, 'units': 'm-K/W'}
    #     reynolds_annulus = compute_reynolds_concentric(
    #         self.m_flow_pipe, self.r_in_out, self.r_out_in, self.fluid
    #     )
    #     output['reynolds_annulus'] = {'value': reynolds_annulus, 'units': ''}
    #     output['inner_annulus_convection_coefficient'] = {'value': self.h_fluid_a_in, 'units': 'W/m2-K'}
    #     output['outer_annulus_convection_coefficient'] = {'value': self.h_fluid_a_out, 'units': 'W/m2-K'}
    #     output['inner_annulus_resistance'] = {'value': self.R_f_a_in, 'units': 'm-K/W'}
    #     output['outer_annulus_resistance'] = {'value': self.R_f_a_out, 'units': 'm-K/W'}
    #     reynolds = compute_reynolds(self.m_flow_pipe, self.r_in_in, self.fluid)
    #     output['reynolds_inner'] = {'value': reynolds, 'units': ''}
    #     output['inner_convection_coefficient'] = {'value': self.h_fluid_in, 'units': 'W/m2-K'}
    #     output['fluid_to_fluid_resistance'] = {'value': self.R_ff, 'units': 'm-K/W'}
    #     output['fluid_to_pipe_resistance'] = {'value': self.R_fp, 'units': 'm-K/W'}
    #     resit_bh_effective = bp.effective_borehole_thermal_resistance(
    #         self, self.m_flow_borehole, self.fluid.cp
    #     )
    #     output['effective_borehole_resistance'] = {'value': resit_bh_effective, 'units': 'm-K/W'}
    #     return output

    def compute_resistances(self) -> None:
        # Inner pipe thermal resistance
        self.R_p_in = gt.pipes.conduction_thermal_resistance_circular_pipe(
            self.r_in_in, self.r_in_out, self.pipe.k[0]
        )
        # Outer pipe thermal resistance
        self.R_p_out = gt.pipes.conduction_thermal_resistance_circular_pipe(
            self.r_out_in, self.r_out_out, self.pipe.k[1]
        )
        # Grout thermal resistance
        self.R_grout = gt.pipes.conduction_thermal_resistance_circular_pipe(
            self.r_out_out, self.b.r_b, self.grout.k
        )

        (
            h_fluid_a_in,
            h_fluid_a_out,
        ) = gt.pipes.convective_heat_transfer_coefficient_concentric_annulus(
            self.m_flow_borehole,
            self.r_in_out,
            self.r_out_in,
            self.fluid.mu,
            self.fluid.rho,
            self.fluid.k,
            self.fluid.cp,
            self.roughness,
        )

        self.h_fluid_a_in = h_fluid_a_in
        self.h_fluid_a_out = h_fluid_a_out

        # Inner fluid convective resistance
        self.R_f_a_in = 1.0 / (self.h_fluid_a_in * 2 * pi * self.r_in_out)
        # Outer fluid convective resistance
        self.R_f_a_out = 1.0 / (h_fluid_a_out * 2 * pi * self.r_out_in)
        self.h_fluid_in = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
            self.m_flow_borehole,
            self.r_in_in,
            self.fluid.mu,
            self.fluid.rho,
            self.fluid.k,
            self.fluid.cp,
            self.roughness,
        )
        self.R_f_in = 1.0 / (self.h_fluid_in * 2 * pi * self.r_in_in)

    def concentric_tube_volumes(self) -> Tuple[float, float, float, float]:
        # Unpack the radii to reduce confusion in the future
        r_in_in, r_in_out = self.r_inner
        r_out_in, r_out_out = self.r_outer
        # Compute volumes for concentric ghe geometry
        vol_fluid = pi * ((r_in_in ** 2) + (r_out_in ** 2) - (r_in_out ** 2))
        vol_pipe = pi * ((r_in_out ** 2) - (r_in_in ** 2) + (r_out_out ** 2) - (r_out_in ** 2))
        # V_grout = pi * ((coaxial.b.r_b**2) - (r_out_out**2))
        area_surf_outer = pi * 2 * r_out_in
        resist_conv = 1 / (self.h_fluid_a_in * area_surf_outer)
        resist_pipe = log(r_out_out / r_out_in) / (2 * pi * self.pipe.k[1])
        return vol_fluid, vol_pipe, resist_conv, resist_pipe


def compute_fluid_resistance(h_conv: float, radius: float) -> float:
    return 1 / (h_conv * 2 * pi * radius)


def compute_reynolds(m_flow_pipe: float, r_in: float, fluid: media.GHEFluid) -> float:
    # Hydraulic diameter
    dia_hydraulic = 2.0 * r_in
    # Fluid velocity
    vol_flow_rate = m_flow_pipe / fluid.rho
    area_cr_inner = pi * r_in ** 2
    velocity = vol_flow_rate / area_cr_inner
    # Reynolds number
    return fluid.rho * velocity * dia_hydraulic / fluid.mu


def compute_reynolds_concentric(m_flow_pipe: float, r_a_in: float, r_a_out: float, fluid: media.GHEFluid) -> float:
    # Hydraulic diameter and radius for concentric tube annulus region
    dia_hydraulic = 2 * (r_a_out - r_a_in)
    # r_h = dia_hydraulic / 2
    # Cross-sectional area of the annulus region
    area_cr_annular = pi * ((r_a_out ** 2) - (r_a_in ** 2))
    # Volume flow rate
    vol_flow_rate = m_flow_pipe / fluid.rho
    # Average velocity
    velocity = vol_flow_rate / area_cr_annular
    # Reynolds number
    return fluid.rho * velocity * dia_hydraulic / fluid.mu
