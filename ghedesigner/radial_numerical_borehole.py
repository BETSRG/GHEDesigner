from math import exp, log, pi, sqrt

import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg.lapack import dgtsv

from ghedesigner.borehole_heat_exchangers import SingleUTube
from ghedesigner.constants import TWO_PI


class RadialCellType(object):
    FLUID = 1
    CONVECTION = 2
    PIPE = 3
    GROUT = 4
    SOIL = 5


class RadialNumericalBH(object):
    """
    X. Xu and Jeffrey D. Spitler. 2006. 'Modeling of Vertical Ground Loop Heat
    Exchangers with Variable Convective Resistance and Thermal Mass of the
    Fluid.' in Proceedings of the 10th International Conference on Thermal
    Energy Storage-EcoStock. Pomona, NJ, May 31-June 2.
    """

    def __init__(
            self,
            single_u_tube: SingleUTube,
            ground_init_temp: float = 20.0,
            data_type: np.dtype = np.double,
    ):
        self.single_u_tube = single_u_tube
        self.dtype = data_type

        self.cell_inputs = {
            "type": 0,
            "inner-radius": 1,
            "center-radius": 2,
            "outer-radius": 3,
            "thickness": 4,
            "conductivity": 5,
            "heat-capacity": 6,
            "initial-temperature": 7,
            "volume": 8,
        }

        # "The one dimensional model has a fluid core, an equivalent convective
        # resistance layer, a tube layer, a grout layer and is surrounded by the
        # ground."

        # cell numbers
        self.num_fluid_cells = 3
        self.num_conv_cells = 1
        self.num_pipe_cells = 4
        self.num_grout_cells = 27
        self.num_soil_cells = 500

        self.num_cells = \
            self.num_fluid_cells + self.num_conv_cells + self.num_fluid_cells + self.num_grout_cells + \
            self.num_soil_cells + 1

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
        self.init_temp = ground_init_temp

        # other
        self.g = np.array([], dtype=self.dtype)
        self.lntts = np.array([], dtype=self.dtype)
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
            cell_type = RadialCellType.FLUID
            thickness = self.thickness_fluid
            center_radius = self.r_fluid + idx * thickness

            if idx == 0:
                inner_radius = center_radius
            else:
                inner_radius = center_radius - thickness / 2.0

            outer_radius = center_radius + thickness / 2.0

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
                    cell_type,
                    inner_radius,
                    center_radius,
                    outer_radius,
                    thickness,
                    k_eq,
                    rho_cp_eq,
                    self.init_temp,
                    volume,
                ],
                dtype=self.dtype,
            )
        cell_summation += num_fluid_cells

        # TODO: verify whether errors are possible here and raise exception if needed
        # assert cell_summation == num_fluid_cells

        # load convection cells
        for idx in range(cell_summation, num_conv_cells + cell_summation):
            j = idx - cell_summation
            cell_type = RadialCellType.CONVECTION
            thickness = self.thickness_conv
            inner_radius = self.r_in_convection + j * thickness
            center_radius = inner_radius + thickness / 2.0
            outer_radius = inner_radius + thickness
            k_eq = log(self.r_in_tube / self.r_in_convection) / (TWO_PI * resist_f_eq)
            rho_cp = 1.0
            volume = pi * (outer_radius ** 2 - inner_radius ** 2)
            radial_cell[:, idx] = np.array(
                [
                    cell_type,
                    inner_radius,
                    center_radius,
                    outer_radius,
                    thickness,
                    k_eq,
                    rho_cp,
                    self.init_temp,
                    volume,
                ],
                dtype=self.dtype,
            )
        cell_summation += num_conv_cells

        # TODO: verify whether errors are possible here and raise exception if needed
        # assert cell_summation == (num_fluid_cells + num_conv_cells)

        # load pipe cells
        for idx in range(cell_summation, num_pipe_cells + cell_summation):
            j = idx - cell_summation
            cell_type = RadialCellType.PIPE
            thickness = self.thickness_pipe
            inner_radius = self.r_in_tube + j * thickness
            center_radius = inner_radius + thickness / 2.0
            outer_radius = inner_radius + thickness
            conductivity = log(self.r_borehole / self.r_in_tube) / (TWO_PI * resist_p_eq)
            rho_cp = self.single_u_tube.pipe.rhoCp
            volume = pi * (outer_radius ** 2 - inner_radius ** 2)
            radial_cell[:, idx] = np.array(
                [
                    cell_type,
                    inner_radius,
                    center_radius,
                    outer_radius,
                    thickness,
                    conductivity,
                    rho_cp,
                    self.init_temp,
                    volume,
                ],
                dtype=self.dtype,
            )
        cell_summation += num_pipe_cells

        # TODO: verify whether errors are possible here and raise exception if needed
        # assert cell_summation == (num_fluid_cells + num_conv_cells + num_pipe_cells)

        # load grout cells
        for idx in range(cell_summation, num_grout_cells + cell_summation):
            j = idx - cell_summation
            cell_type = RadialCellType.GROUT
            thickness = self.thickness_grout
            inner_radius = self.r_out_tube + j * thickness
            center_radius = inner_radius + thickness / 2.0
            outer_radius = inner_radius + thickness
            conductivity = log(self.r_borehole / self.r_in_tube) / (TWO_PI * resist_tg_eq)
            rho_cp = self.single_u_tube.grout.rhoCp
            volume = pi * (outer_radius ** 2 - inner_radius ** 2)
            radial_cell[:, idx] = np.array(
                [
                    cell_type,
                    inner_radius,
                    center_radius,
                    outer_radius,
                    thickness,
                    conductivity,
                    rho_cp,
                    self.init_temp,
                    volume,
                ],
                dtype=self.dtype,
            )
        cell_summation += num_grout_cells

        # TODO: verify whether errors are possible here and raise exception if needed
        # assert cell_summation == (num_fluid_cells + num_conv_cells + num_pipe_cells + num_grout_cells)

        # load soil cells
        for idx in range(cell_summation, num_soil_cells + cell_summation):
            j = idx - cell_summation
            cell_type = RadialCellType.SOIL
            thickness = self.thickness_soil
            inner_radius = self.r_borehole + j * thickness
            center_radius = inner_radius + thickness / 2.0
            outer_radius = inner_radius + thickness
            conductivity = self.single_u_tube.soil.k
            rho_cp = self.single_u_tube.soil.rhoCp
            volume = pi * (outer_radius ** 2 - inner_radius ** 2)
            radial_cell[:, idx] = np.array(
                [
                    cell_type,
                    inner_radius,
                    center_radius,
                    outer_radius,
                    thickness,
                    conductivity,
                    rho_cp,
                    self.init_temp,
                    volume,
                ],
                dtype=self.dtype,
            )
        cell_summation += num_soil_cells

    def calc_sts_g_functions(self, single_u_tube, final_time=None, calculate_at_bh_wall=False) -> tuple:

        self.partial_init(single_u_tube)

        resist_bh_effective = self.single_u_tube.calc_effective_borehole_resistance()

        resist_f_eq = self.single_u_tube.R_f / 2.0
        resist_p_eq = self.single_u_tube.R_p / 2.0
        resist_tg_eq = resist_bh_effective - resist_f_eq

        # Pass radial cell by reference and fill here so that it can be
        # destroyed when this method returns
        radial_cell = np.zeros(
            shape=(len(self.cell_inputs), self.num_cells), dtype=self.dtype
        )
        self.fill_radial_cell(radial_cell, resist_p_eq, resist_f_eq, resist_tg_eq)

        if final_time is None:
            final_time = self.calc_time_in_sec

        g = []
        lntts = []

        _dl = np.zeros(self.num_cells - 1)
        _d = np.zeros(self.num_cells)
        _du = np.zeros(self.num_cells - 1)
        _b = np.zeros(self.num_cells)

        heat_flux = 1.0
        init_temp = self.init_temp

        time = 1e-12 - 120
        time_step = 120

        # type_idx = self.cell_inputs["type"]
        r_in_idx = self.cell_inputs["inner-radius"]
        r_center_idx = self.cell_inputs["center-radius"]
        r_out_idx = self.cell_inputs["outer-radius"]
        # thickness_idx = self.cell_inputs["thickness"]
        k_idx = self.cell_inputs["conductivity"]
        rho_cp_idx = self.cell_inputs["heat-capacity"]
        temperature_idx = self.cell_inputs["initial-temperature"]
        previous_temperature_idx = self.cell_inputs["initial-temperature"]
        volume_idx = self.cell_inputs["volume"]

        _fe_1 = np.zeros(shape=(self.num_cells - 2), dtype=self.dtype)
        _fe_2 = np.zeros_like(_fe_1)
        _ae = np.zeros_like(_fe_2)
        _fw_1 = np.zeros_like(_ae)
        _fw_2 = np.zeros_like(_fw_1)
        _aw = np.zeros_like(_fw_2)
        _ad = np.zeros_like(_aw)

        _west_cell = radial_cell[:, 0: self.num_cells - 2]
        _center_cell = radial_cell[:, 1: self.num_cells - 1]
        _east_cell = radial_cell[:, 2: self.num_cells - 0]

        fe_1 = log(radial_cell[r_out_idx, 0] / radial_cell[r_center_idx, 0]) / (TWO_PI * radial_cell[k_idx, 0])
        fe_2 = log(radial_cell[r_center_idx, 1] / radial_cell[r_in_idx, 1]) / (TWO_PI * radial_cell[k_idx, 1])
        ae = 1 / (fe_1 + fe_2)
        ad = radial_cell[rho_cp_idx, 0] * radial_cell[volume_idx, 0] / time_step
        _d[0] = -ae / ad - 1
        _du[0] = ae / ad

        def fill_f1(fx_1, cell):
            fx_1[:] = np.log(cell[r_out_idx, :] / cell[r_center_idx, :]) / (TWO_PI * cell[k_idx, :])

        def fill_f2(fx_2, cell):
            fx_2[:] = np.log(cell[r_center_idx, :] / cell[r_in_idx, :]) / (TWO_PI * cell[k_idx, :])

        fill_f1(_fe_1, _center_cell)
        fill_f2(_fe_2, _east_cell)
        _ae[:] = 1.0 / (_fe_1 + _fe_2)

        fill_f1(_fw_1, _west_cell)
        fill_f2(_fw_2, _center_cell)
        _aw[:] = -1.0 / (_fw_1 + _fw_2)

        _ad[:] = (_center_cell[rho_cp_idx, :] * _center_cell[volume_idx, :] / time_step)
        _dl[0: self.num_cells - 2] = -_aw / _ad
        _d[1: self.num_cells - 1] = _aw / _ad - _ae / _ad - 1.0
        _du[1: self.num_cells - 1] = _ae / _ad

        while True:

            time += time_step

            # For the idx == 0 case:

            _b[0] = -radial_cell[previous_temperature_idx, 0] - heat_flux / ad

            # For the idx == n-1 case

            _dl[self.num_cells - 2] = 0.0
            _d[self.num_cells - 1] = 1.0
            _b[self.num_cells - 1] = radial_cell[previous_temperature_idx, self.num_cells - 1]

            # Now handle the 1 to n-2 cases with numpy slicing and vectorization
            _b[1: self.num_cells - 1] = -radial_cell[previous_temperature_idx, 1: self.num_cells - 1]

            # Tri-diagonal matrix solver
            # High level interface to LAPACK routine
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lapack.dgtsv.html#scipy.linalg.lapack.dgtsv
            dgtsv(_dl, _d, _du, _b, overwrite_b=1)  # TODO: Do we really need lapack just to do a TDMA solution?

            radial_cell[previous_temperature_idx, :] = _b
            radial_cell[temperature_idx, :] = _b

            if calculate_at_bh_wall:
                raise ValueError(
                    "This portion of the code does not currently run with this "
                    "vectorized radial numerical implementation."
                )
                # bh_wall_temp = 0
                #
                # # calculate bh wall temp
                # for idx, _ in enumerate(self.cells):
                #     west_cell = self.cells[idx]
                #     east_cell = self.cells[idx + 1]
                #
                #     if (
                #         west_cell.type == RadialCellType.GROUT
                #         and east_cell.type == RadialCellType.SOIL
                #     ):
                #         west_conductance_num = 2 * pi * west_cell.conductivity
                #         west_conductance_den = log(
                #             west_cell.outer_radius / west_cell.inner_radius
                #         )
                #         west_conductance = west_conductance_num / west_conductance_den
                #
                #         east_conductance_num = 2 * pi * east_cell.conductivity
                #         east_conductance_den = log(
                #             east_cell.center_radius / west_cell.inner_radius
                #         )
                #         east_conductance = east_conductance_num / east_conductance_den
                #
                #         bh_wall_temp_num_1 = west_conductance * west_cell.temperature
                #         bh_wall_temp_num_2 = east_conductance * east_cell.temperature
                #         bh_wall_temp_num = bh_wall_temp_num_1 + bh_wall_temp_num_2
                #         bh_wall_temp_den = west_conductance + east_conductance
                #         bh_wall_temp = bh_wall_temp_num / bh_wall_temp_den
                #
                #         break
                #
                # g.append(self.c_0 * ((bh_wall_temp - init_temp) / heat_flux))
            else:
                g.append(
                    self.c_0
                    * ((radial_cell[temperature_idx, 0] - init_temp) / heat_flux - resist_bh_effective)
                )

            lntts.append(log(time / self.t_s))

            if time >= final_time - time_step:
                break

        # quickly chop down the total values to a more manageable set
        num_intervals = 30
        g_sts_temp = interp1d(lntts, g)
        uniform_lntts_vals = np.linspace(lntts[0], lntts[-1], num_intervals)
        uniform_g_vals = g_sts_temp(uniform_lntts_vals)

        # set the final arrays and interpolator objects
        self.lntts = np.array(uniform_lntts_vals)
        self.g = np.array(uniform_g_vals)
        self.g_sts = interp1d(self.lntts, self.g)

        return self.lntts, self.g
