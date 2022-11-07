import numpy as np
import pygfunction as gt
from numpy import pi
from pygfunction.pipes import _BasePipe as bp

from ghedt.peak_load_analysis_tool import media


class BasePipe(object):
    def __init__(
            self, m_flow_borehole, fluid, borehole, soil, grout, pipe, config=None
    ):
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

        # Declare variables that are computed in compute_resistance()
        self.h_f = None
        self.R_p = None
        self.R_f = None

        self.compute_resistances()

    @staticmethod
    def justify(category, value):
        return category.ljust(40) + "= " + value + "\n"

    def __repr__(self):
        justify = self.justify

        output = str(self.__class__) + "\n"
        output += 50 * "-" + "\n"

        output += justify(
            "Mass flow borehole", str(round(self.m_flow_borehole, 4)) + " (kg/s)"
        )
        output += justify("Mass flow pipe", str(round(self.m_flow_pipe, 4)) + " (kg/s)")
        output += self.borehole.__repr__() + "\n"
        output += self.soil.__repr__()
        output += self.grout.__repr__()
        output += self.pipe.__repr__()
        output += self.fluid.__repr__()
        output += "\n"

        reynold_no = compute_reynolds_concentric(
            self.m_flow_pipe, self.pipe.r_in, self.pipe.eps, self.fluid
        )

        output += justify("Reynolds number", str(round(reynold_no, 4)))
        output += justify(
            "Convection coefficient", str(round(self.h_f, 4)) + " (W/m2.K)"
        )
        output += justify("Pipe resistance", str(round(self.R_p, 4)) + " (m.K/W)")
        output += justify("Fluid resistance", str(round(self.R_f, 4)) + " (m.K/W)")

        return output

    def compute_resistances(self):
        # Convection heat transfer coefficient of a single pipe
        self.h_f = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
            self.m_flow_pipe,
            self.pipe.r_in,
            self.fluid.mu,
            self.fluid.rho,
            self.fluid.k,
            self.fluid.cp,
            self.pipe.eps,
        )

        # Single pipe thermal resistance
        self.R_p = gt.pipes.conduction_thermal_resistance_circular_pipe(
            self.pipe.r_in, self.pipe.r_out, self.pipe.k
        )
        # Single pipe fluid thermal resistance
        self.R_f = 1.0 / (self.h_f * 2 * pi * self.pipe.r_in)

        return

    @staticmethod
    def compute_mass_flow_rate_pipe(m_flow_borehole, config):
        if config == "series" or config is None:
            m_flow_pipe = m_flow_borehole
        elif config == "parallel":
            m_flow_pipe = m_flow_borehole / 2.0
        else:
            raise ValueError("No such configuration exists.")
        return m_flow_pipe

    def compute_effective_borehole_resistance(self, m_flow_borehole=None, fluid=None):
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


class SingleUTube(BasePipe, gt.pipes.SingleUTube):
    def __init__(
            self,
            m_flow_borehole: float,
            fluid: gt.media.Fluid,
            borehole: gt.boreholes.Borehole,
            pipe: media.Pipe,
            grout: media.Grout,
            soil: media.Soil,
    ):
        # Initialize base pipe class
        BasePipe.__init__(self, m_flow_borehole, fluid, borehole, soil, grout, pipe)
        # Compute Single pipe and fluid thermal resistance
        resist_fluid_pipe = self.R_f + self.R_p
        # Initialize pygfunction SingleUTube base class
        gt.pipes.SingleUTube.__init__(
            self,
            pipe.pos,
            pipe.r_in,
            pipe.r_out,
            borehole,
            self.soil.k,
            self.grout.k,
            resist_fluid_pipe,
        )

    def __repr__(self):
        justify = self.justify

        output = BasePipe.__repr__(self)

        resist_bh_effective = bp.effective_borehole_thermal_resistance(
            self, self.m_flow_borehole, self.fluid.cp
        )

        output += justify(
            "Effective borehole resistance", str(round(resist_bh_effective, 4)) + " (m.K/W)"
        )

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
            self.pos, self.r_out, self.b.r_b, self.k_s, self.k_g, self.R_fp, J=self.J
        )[1]

        # Initialize stored_coefficients
        self._initialize_stored_coefficients()

        # Note: This is the local borehole resistance
        # # Evaluate borehole thermal resistance
        # R_b = 1 / np.trace(1 / self._Rd)

        # Compute and return effective borehole resistance
        resist_bh_effective = bp.effective_borehole_thermal_resistance(
            self, m_flow_borehole, fluid.cp
        )

        return resist_bh_effective


class MultipleUTube(BasePipe, gt.pipes.MultipleUTube):
    def __init__(
            self,
            m_flow_borehole: float,
            fluid: gt.media.Fluid,
            borehole: gt.boreholes.Borehole,
            pipe: media.Pipe,
            grout: media.Grout,
            soil: media.Soil,
            config="parallel",
    ):
        # Initialize base pipe class
        BasePipe.__init__(
            self, m_flow_borehole, fluid, borehole, soil, grout, pipe, config=config
        )

        # Get number of pipes from positions
        self.n_pipes = int(len(pipe.pos) / 2)
        # Compute Single pipe and fluid thermal resistance
        resist_fp = self.R_f + self.R_p
        gt.pipes.MultipleUTube.__init__(
            self,
            pipe.pos,
            pipe.r_in,
            pipe.r_out,
            borehole,
            self.soil.k,
            self.grout.k,
            resist_fp,
            pipe.n_pipes,
            config=config,
        )

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
            self.pos, self.r_out, self.b.r_b, self.k_s, self.k_g, self.R_fp, J=self.J
        )[1]

        # Initialize stored_coefficients
        self._initialize_stored_coefficients()

        # Note: This is the local borehole resistance
        # # Evaluate borehole thermal resistance
        # R_b = 1 / np.trace(1 / self._Rd)

        # Compute and return effective borehole resistance
        resist_bh_effective = bp.effective_borehole_thermal_resistance(
            self, m_flow_borehole, fluid.cp
        )

        return resist_bh_effective

    def __repr__(self):
        justify = self.justify

        output = BasePipe.__repr__(self)

        resist_bh_effective = bp.effective_borehole_thermal_resistance(
            self, self.m_flow_borehole, self.fluid.cp
        )

        output += justify(
            "Effective borehole resistance", str(round(resist_bh_effective, 4)) + " (m.K/W)"
        )

        output += justify("Config", self.config)

        return output


class CoaxialBase(object):
    def __init__(self, m_flow_borehole, fluid, borehole, pipe, soil, grout):
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
        return category.ljust(40) + "= " + value + "\n"

    def __repr__(self):
        justify = self.justify

        output = str(self.__class__) + "\n"
        output += 50 * "-" + "\n"

        output += justify(
            "Mass flow borehole", str(round(self.m_flow_borehole, 4)) + " (kg/s)"
        )
        output += justify("Mass flow pipe", str(round(self.m_flow_pipe, 4)) + " (kg/s)")
        output += self.b.__repr__() + "\n"
        output += self.soil.__repr__()
        output += self.grout.__repr__()
        output += self.pipe.__repr__()
        output += self.fluid.__repr__()
        output += "\n"

        output += justify(
            "Inner pipe resistance", str(round(self.R_p_in, 4)) + " (m.K/W)"
        )
        output += justify(
            "Outer pipe resistance", str(round(self.R_p_out, 4)) + " (m.K/W)"
        )
        output += justify("Grout resistance", str(round(self.R_grout, 4)) + " (m.K/W)")
        reynolds_annulus = compute_reynolds_concentric(
            self.m_flow_pipe, self.r_in_out, self.r_out_in, self.fluid
        )
        output += justify("Reynolds annulus", str(round(reynolds_annulus, 4)))
        output += justify(
            "Inner annulus convection coefficient",
            str(round(self.h_fluid_a_in, 4)) + " (W/m2.K)",
        )
        output += justify(
            "Outer annulus convection coefficient",
            str(round(self.h_fluid_a_out, 4)) + " (W/m2.K)",
        )
        output += justify(
            "Inner annulus resistance", str(round(self.R_f_a_in, 4)) + " (m.K/W)"
        )
        output += justify(
            "Outer annulus resistance", str(round(self.R_f_a_out, 4)) + " (m.K/W)"
        )
        reynolds = compute_reynolds(self.m_flow_pipe, self.r_in_in, self.eps, self.fluid)
        output += justify("Reynolds inner", str(round(reynolds, 4)))
        output += justify(
            "Inner Convection coefficient", str(round(self.h_fluid_in, 4)) + " (W/m2.K)"
        )
        output += justify(
            "Fluid-to-fluid resistance", str(round(self.R_ff, 4)) + " (m.K/W)"
        )
        output += justify(
            "Fluid-to-pipe resistance", str(round(self.R_fp, 4)) + " (m.K/W)"
        )

        resit_bh_effective = bp.effective_borehole_thermal_resistance(
            self, self.m_flow_borehole, self.fluid.cp
        )

        output += justify(
            "Effective borehole resistance", str(round(resit_bh_effective, 4)) + " (m.K/W)"
        )

        return output

    def compute_resistances(self):
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
            self.eps,
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
            self.eps,
        )
        self.R_f_in = 1.0 / (self.h_fluid_in * 2 * pi * self.r_in_in)

        return


class CoaxialPipe(CoaxialBase, gt.pipes.Coaxial, BasePipe):
    def __init__(
            self,
            m_flow_borehole: float,
            fluid: gt.media.Fluid,
            borehole: gt.boreholes.Borehole,
            pipe: media.Pipe,
            grout: media.Grout,
            soil: media.Soil
    ):
        CoaxialBase.__init__(
            self, m_flow_borehole, fluid, borehole, pipe, soil, grout)

        # Vectors of inner and outer pipe radii
        # Note : The dimensions of the inlet pipe are the first elements of
        #       the vectors. In this example, the inlet pipe is the inside pipe.
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
        self._Rd = np.zeros((2 * self.nPipes, 2 * self.nPipes))
        self._Rd[idx_inner, idx_inner] = np.inf
        self._Rd[idx_inner, idx_outer] = self.R_ff
        self._Rd[idx_outer, idx_inner] = self.R_ff
        self._Rd[idx_outer, idx_outer] = resist_fg

        # Initialize stored_coefficients
        self._initialize_stored_coefficients()

        # Note: This is the local borehole resistance
        # # Evaluate borehole thermal resistance
        # R_b = 1 / np.trace(1 / self._Rd)

        # Compute and return effective borehole resistance
        resist_bh_effective = bp.effective_borehole_thermal_resistance(
            self, m_flow_borehole, fluid.cp
        )

        return resist_bh_effective


def compute_reynolds(m_flow_pipe, r_in, fluid):
    # Hydraulic diameter
    dia_hydraulic = 2.0 * r_in
    # Fluid velocity
    vol_flow_rate = m_flow_pipe / fluid.rho
    area_cr_inner = pi * r_in ** 2
    velocity = vol_flow_rate / area_cr_inner
    # Reynolds number
    return fluid.rho * velocity * dia_hydraulic / fluid.mu


def compute_reynolds_concentric(m_flow_pipe, r_a_in, r_a_out, fluid):
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
