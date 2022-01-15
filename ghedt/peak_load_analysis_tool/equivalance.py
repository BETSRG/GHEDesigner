# Jack C. Cook
# Monday, August 16, 2021

import pygfunction as gt
from copy import deepcopy
from numpy import pi, log, sqrt
from scipy.optimize import brentq
from . import borehole_heat_exchangers
import ghedt.peak_load_analysis_tool as plat


def compute_equivalent(bhe):
    # Compute an equivalent borehole heat exchanger based on the type
    if type(bhe) == plat.borehole_heat_exchangers.SingleUTube:
        _bhe = bhe
    elif type(bhe) == plat.borehole_heat_exchangers.MultipleUTube:
        _bhe = multiple_to_single(bhe)
    elif type(bhe) == plat.borehole_heat_exchangers.CoaxialPipe:
        _bhe = coaxial_to_single(bhe)
    else:
        raise ValueError('Not an acceptable BHE.')

    return _bhe


def solve_root(x, objective_function, lower=None, upper=None,
               xtol=1.0e-6, rtol=1.0e-6, maxiter=50):
    # Vary flow rate to match the convective resistance

    # Use Brent Quadratic to find the root
    # Define a lower and upper for thermal conductivities
    if lower is None:
        lower = x / 100.
    else:
        lower = lower
    if upper is None:
        upper = x * 10.
    else:
        upper = upper
    # Check objective function upper and lower bounds to make sure the root is
    # bracketed
    minus = objective_function(lower)
    plus = objective_function(upper)
    # get signs of upper and lower thermal conductivity bounds
    kg_minus_sign = int(minus / abs(minus))
    kg_plus_sign = int(plus / abs(plus))

    # Solve the root if we can, if not, take the higher value
    if kg_plus_sign != kg_minus_sign:
        x = brentq(objective_function, lower, upper,
                   xtol=xtol, rtol=rtol, maxiter=maxiter)
    elif kg_plus_sign == -1 and kg_minus_sign == -1:
        x = upper
    elif kg_plus_sign == 1 and kg_minus_sign == 1:
        x = lower

    return x


def equivalent_single_u_tube(bhe, V_fluid, V_pipe, R_conv, R_pipe):
    # Note: BHE can be double U-tube or coaxial heat exchanger

    # Compute equivalent single U-tube geometry
    n = 2
    r_p_i_prime = sqrt(V_fluid / (n * pi))
    r_p_o_prime = sqrt((V_fluid + V_pipe) / (n * pi))
    A_s_prime = n * pi * ((r_p_i_prime * 2) ** 2)
    h_prime = 1 / (R_conv * A_s_prime)
    k_p_prime = log(r_p_o_prime / r_p_i_prime) / (2 * pi * n * R_pipe)

    # Place single u-tubes at a B-spacing
    # Total horizontal space (m)
    borehole = deepcopy(bhe.b)
    spacing = borehole.r_b * 2 - (n * r_p_o_prime * 2)
    # If the spacing is negative, then the borehole is not large enough,
    # therefore, the borehole will be increased if necessary
    if spacing <= 0.0:
        borehole.r_b -= spacing  # Add on the necessary spacing to fit
        spacing = (borehole.r_b * 2.) / 10.  # make spacing 1/10th of diameter
        borehole.r_b += spacing
    s = spacing / 3  # outer tube-to-tube shank spacing (m)
    pos = plat.media.Pipe.place_pipes(s, r_p_o_prime, 1)  # Place single u-tube pipe

    # New pipe geometry
    eps = deepcopy(bhe.pipe.eps)
    rhoCp = deepcopy(bhe.pipe.rhoCp)
    pipe = \
        plat.media.Pipe(pos, r_p_i_prime, r_p_o_prime, s, eps, k_p_prime, rhoCp)

    # Don't tie together the original and equivalent BHE's
    m_flow_borehole = deepcopy(bhe.m_flow_borehole)
    fluid = deepcopy(bhe.fluid)
    grout = deepcopy(bhe.grout)
    soil = deepcopy(bhe.soil)

    # Maintain the same mass flow rate so that the Rb/Rb* is not diverged from
    eq_single_u_tube = borehole_heat_exchangers.SingleUTube(
        m_flow_borehole, fluid, borehole, pipe, grout, soil)

    # The thermal conductivity of the pipe must now be varied such that R_fp is
    # equivalent to R_fp_prime
    def objective_pipe_conductivity(pipe_k):
        eq_single_u_tube.pipe.k = pipe_k
        eq_single_u_tube.update_thermal_resistance()
        return eq_single_u_tube.R_fp - (R_conv + R_pipe)

    # Use Brent Quadratic to find the root
    # Define a lower and upper for pipe thermal conductivities
    k_p_lower = eq_single_u_tube.pipe.k / 100.
    k_p_upper = eq_single_u_tube.pipe.k * 10.

    # Solve for the mass flow rate to make the convection values equal
    solve_root(eq_single_u_tube.pipe.k, objective_pipe_conductivity,
               lower=k_p_lower, upper=k_p_upper)

    eq_single_u_tube.update_thermal_resistance()

    return eq_single_u_tube


def u_tube_volumes(u_tube):
    # Compute volumes for U-tube geometry
    # Effective parameters
    n = int(u_tube.nPipes * 2)  # Total number of tubes
    # Total inside surface area (m^2)
    A_s = n * pi * (u_tube.r_in * 2.) ** 2
    R_conv = 1 / (u_tube.h_f * A_s)  # Convection resistance (m.K/W)
    # Volumes
    V_fluid = n * pi * (u_tube.r_in ** 2)
    V_pipe = n * pi * (u_tube.r_out ** 2) - V_fluid
    V_grout = pi * (u_tube.b.r_b ** 2) - V_pipe - V_fluid
    R_pipe = \
        log(u_tube.r_out / u_tube.r_in) / \
        (n * 2 * pi * u_tube.pipe.k)
    return V_fluid, V_pipe, R_conv, R_pipe


def concentric_tube_volumes(coaxial):
    # Unpack the radii to reduce confusion in the future
    r_in_in, r_in_out = coaxial.r_inner
    r_out_in, r_out_out = coaxial.r_outer
    # Compute volumes for concentric ghe geometry
    V_fluid = pi * ((r_in_in ** 2) + (r_out_in ** 2) -
                    (r_in_out ** 2))
    V_pipe = pi * ((r_in_out ** 2) - (r_in_in ** 2) +
                   (r_out_out ** 2) - (r_out_in ** 2))
    V_grout = pi * ((coaxial.b.r_b ** 2) - (r_out_out ** 2))
    A_s = pi * 2 * r_out_in
    R_conv = 1 / (coaxial.h_fluid_a_in * A_s)
    R_pipe = log(r_out_out / r_out_in) / (2 * pi * coaxial.pipe.k[1])
    return V_fluid, V_pipe, R_conv, R_pipe


def match_effective_borehole_resistance(tube_ref, new_tube):
    # Find the thermal conductivity that makes the borehole resistances equal

    # Define objective function for varying the grout thermal conductivity
    def objective_resistance(k_g):
        # update new tubes grout thermal conductivity and relevant parameters
        new_tube.k_g = k_g
        new_tube.grout.k = k_g
        # Update Delta-circuit thermal resistances
        # Initialize stored_coefficients
        Rb_prime = new_tube.update_thermal_resistance(m_flow_borehole=None,
                                                      fluid=None)
        Rb = tube_ref.compute_effective_borehole_resistance()
        return Rb - Rb_prime

    # Use Brent Quadratic to find the root
    # Define a lower and upper for thermal conductivities
    kg_lower = 1e-02
    kg_upper = 7.
    k_g = solve_root(
        new_tube.grout.k, objective_resistance, lower=kg_lower, upper=kg_upper)
    # Ensure the grout thermal conductivity is updated
    new_tube.k_g = k_g
    new_tube.grout.k = k_g
    new_tube.update_thermal_resistance()

    return


def multiple_to_single(multiple_u_tube):
    # Find an equivalent single U-tube given multiple U-tube geometry

    # Get effective parameters for the multiple u-tube
    V_fluid, V_pipe, R_conv, R_pipe = u_tube_volumes(multiple_u_tube)

    single_u_tube = equivalent_single_u_tube(
        multiple_u_tube, V_fluid, V_pipe, R_conv, R_pipe)

    # Vary grout thermal conductivity to match effective borehole thermal
    # resistance
    match_effective_borehole_resistance(multiple_u_tube, single_u_tube)

    return single_u_tube


def coaxial_to_single(coaxial_tube):
    # Find an equivalent single U-tube given a coaxial heat exchanger
    V_fluid, V_pipe, R_conv, R_pipe = concentric_tube_volumes(coaxial_tube)

    single_u_tube = equivalent_single_u_tube(
        coaxial_tube, V_fluid, V_pipe, R_conv, R_pipe)

    # Vary grout thermal conductivity to match effective borehole thermal
    # resistance
    match_effective_borehole_resistance(coaxial_tube, single_u_tube)

    return single_u_tube
