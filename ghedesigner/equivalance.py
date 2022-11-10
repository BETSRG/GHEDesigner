from copy import deepcopy

from numpy import pi, log, sqrt

from ghedesigner import borehole_heat_exchangers, media, utilities


def equivalent_single_u_tube(bhe, vol_fluid, vol_pipe, resist_conv, resist_pipe):
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
    borehole = deepcopy(bhe.b)
    spacing = borehole.r_b * 2 - (n * r_p_o_prime * 2)
    # If the spacing is negative, then the borehole is not large enough,
    # therefore, the borehole will be increased if necessary
    if spacing <= 0.0:
        borehole.r_b -= spacing  # Add on the necessary spacing to fit
        spacing = (borehole.r_b * 2.0) / 10.0  # make spacing 1/10th of diameter
        borehole.r_b += spacing
    s = spacing / 3  # outer tube-to-tube shank spacing (m)
    pos = media.Pipe.place_pipes(s, r_p_o_prime, 1)  # Place single u-tube pipe

    # New pipe geometry
    roughness = deepcopy(bhe.pipe.roughness)
    rho_cp = deepcopy(bhe.pipe.rhoCp)
    pipe = media.Pipe(pos, r_p_i_prime, r_p_o_prime, s, roughness, k_p_prime, rho_cp)

    # Don't tie together the original and equivalent BHEs
    m_flow_borehole = deepcopy(bhe.m_flow_borehole)
    fluid = deepcopy(bhe.fluid)
    grout = deepcopy(bhe.grout)
    soil = deepcopy(bhe.soil)

    # Maintain the same mass flow rate so that the Rb/Rb* is not diverged from
    eq_single_u_tube = borehole_heat_exchangers.SingleUTube(
        m_flow_borehole, fluid, borehole, pipe, grout, soil
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


def match_effective_borehole_resistance(tube_ref, new_tube):
    # Find the thermal conductivity that makes the borehole resistances equal

    # Define objective function for varying the grout thermal conductivity
    def objective_resistance(k_g_in):
        # update new tubes grout thermal conductivity and relevant parameters
        new_tube.k_g = k_g_in
        new_tube.grout.k = k_g_in
        # Update Delta-circuit thermal resistances
        # Initialize stored_coefficients
        resist_bh_prime = new_tube.update_thermal_resistance(m_flow_borehole=None)
        resist_bh = tube_ref.compute_effective_borehole_resistance()
        return resist_bh - resist_bh_prime

    # Use Brent Quadratic to find the root
    # Define a lower and upper for thermal conductivities
    kg_lower = 1e-02
    kg_upper = 7.0
    k_g = utilities.solve_root(
        new_tube.grout.k, objective_resistance, lower=kg_lower, upper=kg_upper
    )
    # Ensure the grout thermal conductivity is updated
    new_tube.k_g = k_g
    new_tube.grout.k = k_g
    new_tube.update_thermal_resistance()

    return
