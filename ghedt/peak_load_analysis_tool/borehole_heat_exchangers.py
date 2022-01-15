# Jack C. Cook
# Friday, August 20, 2021

import pygfunction as gt
import ghedt.peak_load_analysis_tool as plat
import numpy as np
from numpy import pi


class BasePipe(object):
    def __init__(self, m_flow_borehole, fluid, borehole, soil, grout, pipe,
                 config=None):
        # borehole mass flow rate
        self.m_flow_borehole = m_flow_borehole
        # mass flow rate to each pipe
        self.m_flow_pipe = self.compute_mass_flow_rate_pipe(m_flow_borehole,
                                                            config)
        # Store Thermal properties
        self.soil = soil
        self.grout = grout
        self.pipe = pipe
        # Store fluid properties
        self.fluid = fluid
        # Store borehole
        self.borehole = borehole

        # Declare variables that are computed in compute_resistance()
        self.h_f = None
        self.R_p = None
        self.R_f = None

        self.compute_resistances()

    @staticmethod
    def justify(category, value):
        return category.ljust(40) + '= ' + value + '\n'

    def __repr__(self):
        justify = self.justify

        output = str(self.__class__) + '\n'
        output += 50 * '-' + '\n'

        output += justify('Mass flow borehole',
                          str(round(self.m_flow_borehole, 4)) + ' (kg/s)')
        output += justify('Mass flow pipe',
                          str(round(self.m_flow_pipe, 4)) + ' (kg/s)')
        output += self.borehole.__repr__() + '\n'
        output += self.soil.__repr__()
        output += self.grout.__repr__()
        output += self.pipe.__repr__()
        output += self.fluid.__repr__()
        output += '\n'

        Re = compute_Reynolds_concentric(
            self.m_flow_pipe, self.pipe.r_in, self.pipe.eps, self.fluid)

        output += justify('Reynolds number', str(round(Re, 4)))
        output += justify('Convection coefficient',
                          str(round(self.h_f, 4)) + ' (W/m2.K)')
        output += justify('Pipe resistance',
                          str(round(self.R_p, 4)) + ' (m.K/W)')
        output += justify('Fluid resistance',
                          str(round(self.R_f, 4)) + ' (m.K/W)')

        return output

    def compute_resistances(self):
        # Convection heat transfer coefficient of a single pipe
        self.h_f = \
            gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
                self.m_flow_pipe, self.pipe.r_in, self.fluid.mu, self.fluid.rho,
                self.fluid.k, self.fluid.cp, self.pipe.eps)

        # Single pipe thermal resistance
        self.R_p = gt.pipes.conduction_thermal_resistance_circular_pipe(
            self.pipe.r_in, self.pipe.r_out, self.pipe.k)
        # Single pipe fluid thermal resistance
        self.R_f = 1.0 / (self.h_f * 2 * pi * self.pipe.r_in)

        return

    @staticmethod
    def compute_mass_flow_rate_pipe(m_flow_borehole, config):
        if config == 'series' or config is None:
            m_flow_pipe = m_flow_borehole
        elif config == 'parallel':
            m_flow_pipe = m_flow_borehole / 2.
        else:
            raise ValueError('No such configuration exists.')
        return m_flow_pipe

    def compute_effective_borehole_resistance(
            self, m_flow_borehole=None, fluid=None):
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

        R_b_star = gt.pipes.borehole_thermal_resistance(
            self, m_flow_borehole, fluid.cp)

        return R_b_star


class SingleUTube(BasePipe, gt.pipes.SingleUTube):
    def __init__(self, m_flow_borehole: float,
                 fluid: gt.media.Fluid, borehole: gt.boreholes.Borehole,
                 pipe: plat.media.Pipe, grout: plat.media.Grout,
                 soil: plat.media.Soil):
        # Initialize base pipe class
        BasePipe.__init__(
            self, m_flow_borehole, fluid, borehole, soil, grout, pipe)
        # Compute Single pipe and fluid thermal resistance
        R_fp = self.R_f + self.R_p
        # Initialize pygfunction SingleUTube base class
        gt.pipes.SingleUTube.__init__(
            self, pipe.pos, pipe.r_in, pipe.r_out, borehole, self.soil.k,
            self.grout.k, R_fp)

    def __repr__(self):
        justify = self.justify

        output = BasePipe.__repr__(self)

        Rb_star = gt.pipes.borehole_thermal_resistance(
            self, self.m_flow_borehole, self.fluid.cp)

        output += justify('Effective borehole resistance',
                          str(round(Rb_star, 4)) + ' (m.K/W)')

        return output

    def update_thermal_resistance(self, m_flow_borehole=None, fluid=None):

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

        self.R_fp = self.R_f + self.R_p

        # Delta-circuit thermal resistances
        self._Rd = gt.pipes.thermal_resistances(
            self.pos, self.r_out, self.b.r_b, self.k_s, self.k_g, self.R_fp,
            J=self.J)[1]

        # Initialize stored_coefficients
        self._initialize_stored_coefficients()

        # Note: This is the local borehole resistance
        # # Evaluate borehole thermal resistance
        # R_b = 1 / np.trace(1 / self._Rd)

        # Compute and return effective borehole resistance
        R_b_star = gt.pipes.borehole_thermal_resistance(
            self, m_flow_borehole, fluid.cp)

        return R_b_star

    def compute_convection_coefficient(self, m_flow_borehole):
        h = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
            m_flow_borehole, self.pipe.r_in, self.fluid.mu, self.fluid.rho,
            self.fluid.k, self.fluid.cp, self.pipe.eps)
        return h


class MultipleUTube(BasePipe, gt.pipes.MultipleUTube):
    def __init__(self, m_flow_borehole: float, fluid: gt.media.Fluid,
                 borehole: gt.boreholes.Borehole,
                 pipe: plat.media.Pipe, grout: plat.media.Grout,
                 soil: plat.media.Soil, config='parallel'):
        # Initialize base pipe class
        BasePipe.__init__(
            self, m_flow_borehole, fluid, borehole, soil, grout, pipe,
            config=config)

        # Get number of pipes from positions
        self.n_pipes = int(len(pipe.pos) / 2)
        # Compute Single pipe and fluid thermal resistance
        R_fp = self.R_f + self.R_p
        gt.pipes.MultipleUTube.__init__(
            self, pipe.pos, pipe.r_in, pipe.r_out, borehole, self.soil.k,
            self.grout.k, R_fp, pipe.n_pipes, config=config)

    def update_thermal_resistance(self, m_flow_borehole=None, fluid=None):

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

        self.R_fp = self.R_f + self.R_p

        # Delta-circuit thermal resistances
        self._Rd = gt.pipes.thermal_resistances(
            self.pos, self.r_out, self.b.r_b, self.k_s, self.k_g, self.R_fp,
            J=self.J)[1]

        # Initialize stored_coefficients
        self._initialize_stored_coefficients()

        # Note: This is the local borehole resistance
        # # Evaluate borehole thermal resistance
        # R_b = 1 / np.trace(1 / self._Rd)

        # Compute and return effective borehole resistance
        R_b_star = gt.pipes.borehole_thermal_resistance(
            self, m_flow_borehole, fluid.cp)

        return R_b_star

    def __repr__(self):
        justify = self.justify

        output = BasePipe.__repr__(self)

        Rb_star = gt.pipes.borehole_thermal_resistance(
            self, self.m_flow_borehole, self.fluid.cp)

        output += justify('Effective borehole resistance',
                          str(round(Rb_star, 4)) + ' (m.K/W)')

        output += justify('Config', self.config)

        return output


class CoaxialBase(object):
    def __init__(self, m_flow_borehole, fluid, borehole, pipe, soil, grout,
                 config=None):
        self.m_flow_borehole = m_flow_borehole
        self.m_flow_pipe = m_flow_borehole
        # Store Thermal properties
        self.soil = soil
        self.grout = grout
        self.pipe = pipe
        # Store fluid properties
        self.fluid = fluid
        # Store pipe roughness
        self.eps = self.pipe.eps

        self.r_inner = pipe.r_in
        self.r_outer = pipe.r_out

        # Unpack the radii to reduce confusion in the future
        self.r_in_in, self.r_in_out = self.r_inner
        self.r_out_in, self.r_out_out = self.r_outer

        self.b = borehole  # pygfunction borehole

        # Declare variables that are computed in compute_resistance()
        self.R_p_in = None
        self.R_p_out = None
        self.R_grout = None
        self.h_fluid_a_in = None
        self.h_fluid_a_out = None
        self.R_f_a_in = None
        self.R_f_a_out = None
        self.h_fluid_in = None
        self.R_f_in = None
        self.R_ff = None
        self.R_fp = None

        self.compute_resistances()

        return

    @staticmethod
    def justify(category, value):
        return category.ljust(40) + '= ' + value + '\n'

    def __repr__(self):
        justify = self.justify

        output = str(self.__class__) + '\n'
        output += 50 * '-' + '\n'

        output += justify('Mass flow borehole',
                          str(round(self.m_flow_borehole, 4)) + ' (kg/s)')
        output += justify('Mass flow pipe',
                          str(round(self.m_flow_pipe, 4)) + ' (kg/s)')
        output += self.b.__repr__() + '\n'
        output += self.soil.__repr__()
        output += self.grout.__repr__()
        output += self.pipe.__repr__()
        output += self.fluid.__repr__()
        output += '\n'

        output += justify('Inner pipe resistance',
                          str(round(self.R_p_in, 4)) + ' (m.K/W)')
        output += justify('Outer pipe resistance',
                          str(round(self.R_p_out, 4)) + ' (m.K/W)')
        output += justify('Grout resistance',
                          str(round(self.R_grout, 4)) + ' (m.K/W)')
        Re_annulus = compute_Reynolds_concentric(
            self.m_flow_pipe, self.r_in_out, self.r_out_in, self.fluid
        )
        output += justify('Reynolds annulus', str(round(Re_annulus, 4)))
        output += justify('Inner annulus convection coef',
                          str(round(self.h_fluid_a_in, 4)) + ' (W/m2.K)')
        output += justify('Outer annulus convection coef',
                          str(round(self.h_fluid_a_out, 4)) + ' (W/m2.K)')
        output += justify('Inner annulus resistance',
                          str(round(self.R_f_a_in, 4)) + ' (m.K/W)')
        output += justify('Outer annulus resistance',
                          str(round(self.R_f_a_out, 4)) + ' (m.K/W)')
        Re = compute_Reynolds(
            self.m_flow_pipe, self.r_in_in, self.eps, self.fluid)
        output += justify('Reynolds inner', str(round(Re, 4)))
        output += justify('Inner Convection coefficient',
                          str(round(self.h_fluid_in, 4)) + ' (W/m2.K)')
        output += justify('Fluid-to-fluid resistance',
                          str(round(self.R_ff, 4)) + ' (m.K/W)')
        output += justify('Fluid-to-pipe resistance',
                          str(round(self.R_fp, 4)) + ' (m.K/W)')

        Rb_star = gt.pipes.borehole_thermal_resistance(
            self, self.m_flow_borehole, self.fluid.cp)

        output += justify('Effective borehole resistance',
                          str(round(Rb_star, 4)) + ' (m.K/W)')

        return output

    def compute_resistances(self):
        # Inner pipe thermal resistance
        self.R_p_in = gt.pipes. \
            conduction_thermal_resistance_circular_pipe(self.r_in_in,
                                                        self.r_in_out,
                                                        self.pipe.k[0])
        # Outer pipe thermal resistance
        self.R_p_out = gt.pipes. \
            conduction_thermal_resistance_circular_pipe(self.r_out_in,
                                                        self.r_out_out,
                                                        self.pipe.k[1])
        # Grout thermal resistance
        self.R_grout = gt.pipes.conduction_thermal_resistance_circular_pipe(
            self.r_out_out, self.b.r_b, self.grout.k)

        h_fluid_a_in, h_fluid_a_out = \
            gt.pipes.convective_heat_transfer_coefficient_concentric_annulus(
                self.m_flow_borehole, self.r_in_out, self.r_out_in,
                self.fluid.mu, self.fluid.rho, self.fluid.k, self.fluid.cp,
                self.eps)

        self.h_fluid_a_in = h_fluid_a_in
        self.h_fluid_a_out = h_fluid_a_out

        # Inner fluid convective resistance
        self.R_f_a_in = 1. / (self.h_fluid_a_in * 2 * pi * self.r_in_out)
        # Outer fluid convective resistance
        self.R_f_a_out = 1. / (h_fluid_a_out * 2 * pi * self.r_out_in)
        self.h_fluid_in = \
            gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
                self.m_flow_borehole, self.r_in_in, self.fluid.mu,
                self.fluid.rho, self.fluid.k, self.fluid.cp, self.eps)
        self.R_f_in = 1. / (self.h_fluid_in * 2 * pi * self.r_in_in)

        return


class CoaxialPipe(CoaxialBase, gt.pipes.Coaxial, BasePipe):
    def __init__(self, m_flow_borehole: float, fluid: gt.media.Fluid,
                 borehole: gt.boreholes.Borehole,
                 pipe: plat.media.Pipe, grout: plat.media.Grout,
                 soil: plat.media.Soil, config=None):
        CoaxialBase.__init__(self, m_flow_borehole, fluid, borehole, pipe,
                             soil, grout, config=config)

        # Vectors of inner and outer pipe radii
        # Note : The dimensions of the inlet pipe are the first elements of
        #       the vectors. In this example, the inlet pipe is the inside pipe.
        r_inner_p = np.array([pipe.r_in[0], pipe.r_out[0]])  # Inner pipe radii (m)
        r_outer_p = np.array([pipe.r_in[1], pipe.r_out[1]])  # Outer pip radii (m)

        # Compute resistances that are stored by pygfunction
        # Inner fluid to inner anulus fluid resistance
        R_ff = self.R_f_in + self.R_p_in + self.R_f_a_in
        # Outer annulus fluid to pipe thermal resistance
        R_fp = self.R_p_out + self.R_f_a_out

        gt.pipes.Coaxial.__init__(
            self, pipe.pos, r_inner_p, r_outer_p,  borehole, self.soil.k,
            self.grout.k, R_ff, R_fp, J=2)

    def update_thermal_resistance(self, m_flow_borehole=None, fluid=None):

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
        # Inner fluid to inner anulus fluid resistance
        self.R_ff = self.R_f_in + self.R_p_in + self.R_f_a_in
        # Outer annulus fluid to pipe thermal resistance
        self.R_fp = self.R_p_out + self.R_f_a_out

        # Determine the indexes of the inner and outer pipes
        iInner = self.r_out.argmin()
        iOuter = self.r_out.argmax()
        # Outer pipe to borehole wall thermal resistance
        R_fg = gt.pipes.thermal_resistances(
            self.pos, self.r_out[iOuter], self.b.r_b, self.soil.k,
            self.grout.k, self.R_fp, J=self.J)[1][0]

        # Delta-circuit thermal resistances
        self._Rd = np.zeros((2 * self.nPipes, 2 * self.nPipes))
        self._Rd[iInner, iInner] = np.inf
        self._Rd[iInner, iOuter] = self.R_ff
        self._Rd[iOuter, iInner] = self.R_ff
        self._Rd[iOuter, iOuter] = R_fg

        # Initialize stored_coefficients
        self._initialize_stored_coefficients()

        # Note: This is the local borehole resistance
        # # Evaluate borehole thermal resistance
        # R_b = 1 / np.trace(1 / self._Rd)

        # Compute and return effective borehole resistance
        R_b_star = gt.pipes.borehole_thermal_resistance(
            self, m_flow_borehole, fluid.cp)

        return R_b_star


def compute_Reynolds(m_flow_pipe, r_in, epsilon, fluid):
    # Hydraulic diameter
    D = 2.*r_in
    # Relative roughness
    E = epsilon / D
    # Fluid velocity
    V_flow = m_flow_pipe / fluid.rho
    A_cs = pi * r_in**2
    V = V_flow / A_cs
    # Reynolds number
    Re = fluid.rho * V * D / fluid.mu
    return Re


def compute_Reynolds_concentric(m_flow_pipe, r_a_in, r_a_out, fluid):
    # Hydraulic diameter and radius for concentric tube annulus region
    D_h = 2 * (r_a_out - r_a_in)
    r_h = D_h / 2
    # Cross-sectional area of the annulus region
    A_c = pi * ((r_a_out ** 2) - (r_a_in ** 2))
    # Volume flow rate
    V_dot = m_flow_pipe / fluid.rho
    # Average velocity
    V = V_dot / A_c
    # Reynolds number
    Re = fluid.rho * V * D_h / fluid.mu
    return Re
