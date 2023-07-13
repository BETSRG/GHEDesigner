from enum import auto, IntEnum
from math import exp, log, pi, sqrt

import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg.lapack import dgtsv

from ghedesigner.borehole_heat_exchangers import SingleUTube
from ghedesigner.constants import TWO_PI


class CellProps(IntEnum):
    R_IN = 0
    R_CENTER = auto()
    R_OUT = auto()
    K = auto()
    RHO_CP = auto()
    TEMP = auto()
    VOL = auto()


class RadialNumericalBH(object):
    """
    X. Xu and Jeffrey D. Spitler. 2006. 'Modeling of Vertical Ground Loop Heat
    Exchangers with Variable Convective Resistance and Thermal Mass of the
    Fluid.' in Proceedings of the 10th International Conference on Thermal
    Energy Storage-EcoStock. Pomona, NJ, May 31-June 2.
    """

    def __init__(self, single_u_tube: SingleUTube):
        self.single_u_tube = single_u_tube

        # "The one dimensional model has a fluid core, an equivalent convective
        # resistance layer, a tube layer, a grout layer and is surrounded by the
        # ground."

        # cell numbers
        self.num_fluid_cells = 3
        self.num_conv_cells = 1
        self.num_pipe_cells = 4
        self.num_grout_cells = 27
        self.num_soil_cells = 500

        self.num_cells = self.num_fluid_cells + self.num_conv_cells + self.num_fluid_cells
        self.num_cells += self.num_grout_cells + self.num_soil_cells + 1

        self.bh_wall_idx = self.num_fluid_cells + self.num_conv_cells + self.num_pipe_cells + self.num_grout_cells

        # Geometry and grid procedure

        # far-field radius is set to 10m; the soil region is represented by
        # 500 cells
        far_field_radius = 10  # soil radius (in meters)
        self.r_far_field = far_field_radius - single_u_tube.b.r_b

        # borehole radius is set to the actual radius of the borehole
        self.r_borehole = single_u_tube.b.r_b

        # outer tube radius is set to sqrt(2) * r_p_o, tube region has 4 cells
        self.r_out_tube = sqrt(2) * single_u_tube.pipe.r_out

        # inner tube radius is set to r_out_tube-t_p
        self.thickness_pipe = single_u_tube.pipe.r_out - single_u_tube.pipe.r_in
        self.r_in_tube = self.r_out_tube - self.thickness_pipe

        # r_in_convection is set to r_in_tube - 1/4 * t
        self.r_in_convection = self.r_in_tube - self.thickness_pipe / 4.0

        # r_fluid is set to r_in_convection - 3/4 * t
        self.r_fluid = self.r_in_convection - (3.0 / 4.0 * self.thickness_pipe)

        # Thicknesses of the grid regions
        self.thickness_soil = (self.r_far_field - self.r_borehole) / self.num_soil_cells
        self.thickness_grout = (self.r_borehole - self.r_out_tube) / self.num_grout_cells
        # pipe thickness is equivalent to original tube thickness
        self.thickness_conv = (self.r_in_tube - self.r_in_convection) / self.num_conv_cells
        self.thickness_fluid = (self.r_in_convection - self.r_fluid) / self.num_fluid_cells

        # other
        self.init_temp = 20

        # other
        self.g = np.array([], dtype=np.double)
        self.g_bhw = np.array([], dtype=np.double)
        self.lntts = np.array([], dtype=np.double)
        self.c_0 = TWO_PI * single_u_tube.soil.k
        soil_diffusivity = single_u_tube.k_s / single_u_tube.soil.rhoCp
        self.t_s = single_u_tube.b.H ** 2 / (9 * soil_diffusivity)
        # default is at least 49 hours, or up to -8.6 log time
        self.calc_time_in_sec = max([self.t_s * exp(-8.6), 49.0 * 3600.0])
        self.g_sts = None

    def partial_init(self, single_u_tube: SingleUTube):
        # TODO: unravel how to eliminate this.
        # - It was calling the full class ctor "self.__init__()" which is just plain wrong...
        # - Now we're calling a stripped down version with only the most essential
        #   variables which are required.
        # - This is here partially because equivalent boreholes are generated.
        self.single_u_tube = single_u_tube
        soil_diffusivity = single_u_tube.k_s / single_u_tube.soil.rhoCp
        self.t_s = single_u_tube.b.H ** 2 / (9 * soil_diffusivity)
        self.calc_time_in_sec = max([self.t_s * exp(-8.6), 49.0 * 3600.0])

    def fill_radial_cell(self, radial_cell, resist_p_eq, resist_f_eq, resist_tg_eq):

        num_fluid_cells = self.num_fluid_cells
        num_conv_cells = self.num_conv_cells
        num_pipe_cells = self.num_pipe_cells
        num_grout_cells = self.num_grout_cells
        num_soil_cells = self.num_soil_cells

        cell_summation = 0

        # load fluid cells
        for idx in range(cell_summation, num_fluid_cells + cell_summation):
            center_radius = self.r_fluid + idx * self.thickness_fluid

            if idx == 0:
                inner_radius = center_radius
            else:
                inner_radius = center_radius - self.thickness_fluid / 2.0

            outer_radius = center_radius + self.thickness_fluid / 2.0

            # The equivalent thermal mass of the fluid can be calculated from
            # equation (2)
            # pi (r_in_conv ** 2 - r_f **2) C_eq_f = 2pi r_p_in**2 * C_f
            rho_cp_eq = (2.0
                         * (self.single_u_tube.pipe.r_in ** 2)
                         * self.single_u_tube.fluid.rhoCp
                         ) / ((self.r_in_convection ** 2) - (self.r_fluid ** 2))

            k_eq = rho_cp_eq / self.single_u_tube.fluid.cp

            volume = pi * (outer_radius ** 2 - inner_radius ** 2)
            radial_cell[:, idx] = np.array(
                [
                    inner_radius,
                    center_radius,
                    outer_radius,
                    k_eq,
                    rho_cp_eq,
                    self.init_temp,
                    volume,
                ],
                dtype=np.double,
            )
        cell_summation += num_fluid_cells

        # TODO: verify whether errors are possible here and raise exception if needed
        # assert cell_summation == num_fluid_cells

        # load convection cells
        for idx in range(cell_summation, num_conv_cells + cell_summation):
            j = idx - cell_summation
            inner_radius = self.r_in_convection + j * self.thickness_conv
            center_radius = inner_radius + self.thickness_conv / 2.0
            outer_radius = inner_radius + self.thickness_conv
            k_eq = log(self.r_in_tube / self.r_in_convection) / (TWO_PI * resist_f_eq)
            rho_cp = 1.0
            volume = pi * (outer_radius ** 2 - inner_radius ** 2)
            radial_cell[:, idx] = np.array(
                [
                    inner_radius,
                    center_radius,
                    outer_radius,
                    k_eq,
                    rho_cp,
                    self.init_temp,
                    volume,
                ],
                dtype=np.double,
            )
        cell_summation += num_conv_cells

        # TODO: verify whether errors are possible here and raise exception if needed
        # assert cell_summation == (num_fluid_cells + num_conv_cells)

        # load pipe cells
        for idx in range(cell_summation, num_pipe_cells + cell_summation):
            j = idx - cell_summation
            inner_radius = self.r_in_tube + j * self.thickness_pipe
            center_radius = inner_radius + self.thickness_pipe / 2.0
            outer_radius = inner_radius + self.thickness_pipe
            conductivity = log(self.r_borehole / self.r_in_tube) / (TWO_PI * resist_p_eq)
            rho_cp = self.single_u_tube.pipe.rhoCp
            volume = pi * (outer_radius ** 2 - inner_radius ** 2)
            radial_cell[:, idx] = np.array(
                [
                    inner_radius,
                    center_radius,
                    outer_radius,
                    conductivity,
                    rho_cp,
                    self.init_temp,
                    volume,
                ],
                dtype=np.double,
            )
        cell_summation += num_pipe_cells

        # TODO: verify whether errors are possible here and raise exception if needed
        # assert cell_summation == (num_fluid_cells + num_conv_cells + num_pipe_cells)

        # load grout cells
        for idx in range(cell_summation, num_grout_cells + cell_summation):
            j = idx - cell_summation
            inner_radius = self.r_out_tube + j * self.thickness_grout
            center_radius = inner_radius + self.thickness_grout / 2.0
            outer_radius = inner_radius + self.thickness_grout
            conductivity = log(self.r_borehole / self.r_in_tube) / (TWO_PI * resist_tg_eq)
            rho_cp = self.single_u_tube.grout.rhoCp
            volume = pi * (outer_radius ** 2 - inner_radius ** 2)
            radial_cell[:, idx] = np.array(
                [
                    inner_radius,
                    center_radius,
                    outer_radius,
                    conductivity,
                    rho_cp,
                    self.init_temp,
                    volume,
                ],
                dtype=np.double,
            )
        cell_summation += num_grout_cells

        # TODO: verify whether errors are possible here and raise exception if needed
        # assert cell_summation == (num_fluid_cells + num_conv_cells + num_pipe_cells + num_grout_cells)

        # load soil cells
        for idx in range(cell_summation, num_soil_cells + cell_summation):
            j = idx - cell_summation
            inner_radius = self.r_borehole + j * self.thickness_soil
            center_radius = inner_radius + self.thickness_soil / 2.0
            outer_radius = inner_radius + self.thickness_soil
            conductivity = self.single_u_tube.soil.k
            rho_cp = self.single_u_tube.soil.rhoCp
            volume = pi * (outer_radius ** 2 - inner_radius ** 2)
            radial_cell[:, idx] = np.array(
                [
                    inner_radius,
                    center_radius,
                    outer_radius,
                    conductivity,
                    rho_cp,
                    self.init_temp,
                    volume,
                ],
                dtype=np.double,
            )
        cell_summation += num_soil_cells

    def calc_sts_g_functions(self, single_u_tube, final_time=None) -> tuple:

        self.partial_init(single_u_tube)

        resist_bh_effective = self.single_u_tube.calc_effective_borehole_resistance()

        resist_f_eq = self.single_u_tube.R_f / 2.0
        resist_p_eq = self.single_u_tube.R_p / 2.0
        resist_tg_eq = resist_bh_effective - resist_f_eq

        # Pass radial cell by reference and fill here so that it can be
        # destroyed when this method returns
        radial_cell = np.zeros(shape=(len(CellProps), self.num_cells), dtype=np.double)
        self.fill_radial_cell(radial_cell, resist_p_eq, resist_f_eq, resist_tg_eq)

        if final_time is None:
            final_time = self.calc_time_in_sec

        g = []
        g_bhw = []
        lntts = []

        _dl = np.zeros(self.num_cells - 1)
        _d = np.zeros(self.num_cells)
        _du = np.zeros(self.num_cells - 1)
        _b = np.zeros(self.num_cells)

        heat_flux = 1.0
        init_temp = self.init_temp

        time = 1e-12 - 120
        time_step = 120

        _fe_1 = np.zeros(shape=(self.num_cells - 2), dtype=np.double)
        _fe_2 = np.zeros_like(_fe_1)
        _ae = np.zeros_like(_fe_2)
        _fw_1 = np.zeros_like(_ae)
        _fw_2 = np.zeros_like(_fw_1)
        _aw = np.zeros_like(_fw_2)
        _ad = np.zeros_like(_aw)

        _west_cell = radial_cell[:, 0: self.num_cells - 2]
        _center_cell = radial_cell[:, 1: self.num_cells - 1]
        _east_cell = radial_cell[:, 2: self.num_cells - 0]

        fe_1 = log(radial_cell[CellProps.R_OUT, 0] / radial_cell[CellProps.R_CENTER, 0])
        fe_1 /= (TWO_PI * radial_cell[CellProps.K, 0])

        fe_2 = log(radial_cell[CellProps.R_CENTER, 1] / radial_cell[CellProps.R_IN, 1])
        fe_2 /= (TWO_PI * radial_cell[CellProps.K, 1])

        ae = 1 / (fe_1 + fe_2)
        ad = radial_cell[CellProps.RHO_CP, 0] * radial_cell[CellProps.VOL, 0] / time_step
        _d[0] = -ae / ad - 1
        _du[0] = ae / ad

        def fill_f1(fx_1, cell):
            fx_1[:] = np.log(cell[CellProps.R_OUT, :] / cell[CellProps.R_CENTER, :]) / (TWO_PI * cell[CellProps.K, :])

        def fill_f2(fx_2, cell):
            fx_2[:] = np.log(cell[CellProps.R_CENTER, :] / cell[CellProps.R_IN, :]) / (TWO_PI * cell[CellProps.K, :])

        fill_f1(_fe_1, _center_cell)
        fill_f2(_fe_2, _east_cell)
        _ae[:] = 1.0 / (_fe_1 + _fe_2)

        fill_f1(_fw_1, _west_cell)
        fill_f2(_fw_2, _center_cell)
        _aw[:] = -1.0 / (_fw_1 + _fw_2)

        _ad[:] = (_center_cell[CellProps.RHO_CP, :] * _center_cell[CellProps.VOL, :] / time_step)
        _dl[0: self.num_cells - 2] = -_aw / _ad
        _d[1: self.num_cells - 1] = _aw / _ad - _ae / _ad - 1.0
        _du[1: self.num_cells - 1] = _ae / _ad

        while True:

            time += time_step

            # For the idx == 0 case:

            _b[0] = -radial_cell[CellProps.TEMP, 0] - heat_flux / ad

            # For the idx == n-1 case

            _dl[self.num_cells - 2] = 0.0
            _d[self.num_cells - 1] = 1.0
            _b[self.num_cells - 1] = radial_cell[CellProps.TEMP, self.num_cells - 1]

            # Now handle the 1 to n-2 cases with numpy slicing and vectorization
            _b[1: self.num_cells - 1] = -radial_cell[CellProps.TEMP, 1: self.num_cells - 1]

            # Tri-diagonal matrix solver
            # High level interface to LAPACK routine
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lapack.dgtsv.html#scipy.linalg.lapack.dgtsv
            dgtsv(_dl, _d, _du, _b, overwrite_b=1)  # TODO: Do we really need lapack just to do a TDMA solution?

            radial_cell[CellProps.TEMP, :] = _b

            # compute standard g-functions
            g.append(self.c_0 * ((radial_cell[CellProps.TEMP, 0] - init_temp) / heat_flux - resist_bh_effective))

            # compute g-functions at bh wall
            bh_wall_temp = radial_cell[CellProps.TEMP, self.bh_wall_idx]
            g_bhw.append(self.c_0 * ((bh_wall_temp - init_temp) / heat_flux))

            lntts.append(log(time / self.t_s))

            if time >= final_time - time_step:
                break

        # quickly chop down the total values to a more manageable set
        num_intervals = 30
        g_tmp = interp1d(lntts, g)
        uniform_lntts_vals = np.linspace(lntts[0], lntts[-1], num_intervals)
        uniform_g_vals = g_tmp(uniform_lntts_vals)

        g_bhw_tmp = interp1d(lntts, g_bhw)
        uniform_g_bhw_vals = g_bhw_tmp(uniform_lntts_vals)

        # set the final arrays and interpolator objects
        self.lntts = np.array(uniform_lntts_vals)
        self.g = np.array(uniform_g_vals)
        self.g_bhw = np.array(uniform_g_bhw_vals)
        self.g_sts = interp1d(self.lntts, self.g)

        return self.lntts, self.g
