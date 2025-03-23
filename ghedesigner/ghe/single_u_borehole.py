from enum import IntEnum, auto
from math import exp, log, pi, sqrt

import numpy as np
import pygfunction as gt
from pygfunction.boreholes import Borehole
from scipy.interpolate import interp1d
from scipy.linalg.lapack import dgtsv

from ghedesigner.constants import SEC_IN_HR, TWO_PI
from ghedesigner.media import GHEFluid, Grout, Pipe, Soil
from bhr.borehole import Borehole as bhr


class CellProps(IntEnum):
    R_IN = 0
    R_CENTER = auto()
    R_OUT = auto()
    K = auto()
    RHO_CP = auto()
    TEMP = auto()
    VOL = auto()


class SingleUTube(gt.pipes.SingleUTube):
    def __init__(
        self,
        m_flow_borehole: float,
        fluid: GHEFluid,
        _borehole: Borehole,
        pipe: Pipe,
        grout: Grout,
        soil: Soil,
    ) -> None:
        # GHEDesignerBoreholeBase.__init__(self, m_flow_borehole, fluid, _borehole, pipe, grout, soil)
        # self.R_p = 0.0
        # self.R_f = 0.0
        # self.R_fp = 0.0
        # self.h_f = 0.0
        self.fluid = fluid
        self.m_flow_borehole = m_flow_borehole
        self.borehole = _borehole
        self.pipe = pipe
        self.soil = soil
        self.grout = grout


        self.bhr = bhr()
        self.bhr.init_single_u_borehole(
            borehole_diameter=_borehole.r_b * 2.0,
            pipe_outer_diameter=pipe.r_out * 2.0,
            pipe_dimension_ratio=(pipe.r_out * 2.0) / (pipe.r_out - pipe.r_in),
            length=_borehole.H,
            shank_space=pipe.ss_from_center,
            pipe_conductivity=pipe.k,
            grout_conductivity=grout.k,
            soil_conductivity=soil.k,
            fluid_type=fluid.name,
            fluid_concentration=fluid.concentration_percent,
        )

        # # compute resistances required to construct inherited class
        # self.calc_fluid_pipe_resistance()

        # # Initialize pygfunction SingleUTube base class
        # gt.pipes.SingleUTube.__init__(
        #     self,
        #     self.pipe.pos,
        #     self.pipe.r_in,
        #     self.pipe.r_out,
        #     self.borehole,
        #     self.soil.k,
        #     self.grout.k,
        #     self.R_fp,
        # )

        # # these methods must be called after inherited class construction
        # self.update_thermal_resistances(self.R_fp)
        # self.calc_effective_borehole_resistance()

        # radial numerical initialization
        # "The one dimensional model has a fluid core, an equivalent convective
        # resistance layer, a tube layer, a grout layer and is surrounded by the
        # ground."

        # cell numbers
        self.num_fluid_cells = 3
        self.num_conv_cells = 1
        self.num_pipe_cells = 4
        self.num_grout_cells = 27
        self.num_soil_cells = 500

        self.num_cells = sum(
            (self.num_fluid_cells, self.num_conv_cells, self.num_pipe_cells, self.num_grout_cells, self.num_soil_cells)
        )
        self.bh_wall_idx = sum((self.num_fluid_cells, self.num_conv_cells, self.num_pipe_cells, self.num_grout_cells))

        # Geometry and grid procedure

        # far-field radius is set to 10m
        self.r_far_field = 10

        # borehole radius is set to the actual radius of the borehole
        self.r_borehole = self.borehole.r_b

        # outer tube radius is set to sqrt(2) * r_p_o, tube region has 4 cells
        self.r_out_tube = sqrt(2) * self.pipe.r_out

        # inner tube radius is set to r_out_tube - t_p
        self.t_pipe_wall_actual = self.pipe.r_out - self.pipe.r_in
        self.r_in_tube = self.r_out_tube - self.t_pipe_wall_actual

        # r_convection is set to r_in_tube - 1/4 * t_p
        self.r_convection = self.r_in_tube - self.t_pipe_wall_actual / 4.0

        # r_fluid is set to r_in_convection - 3/4 * t_p
        self.r_fluid = self.r_convection - (3.0 / 4.0 * self.t_pipe_wall_actual)

        # Thicknesses of the grid regions
        self.thickness_soil_cell = (self.r_far_field - self.r_borehole) / self.num_soil_cells
        self.thickness_grout_cell = (self.r_borehole - self.r_out_tube) / self.num_grout_cells
        self.thickness_pipe_cell = (self.r_out_tube - self.r_in_tube) / self.num_pipe_cells
        self.thickness_conv_cell = (self.r_in_tube - self.r_convection) / self.num_conv_cells
        self.thickness_fluid_cell = (self.r_convection - self.r_fluid) / self.num_fluid_cells

        # other
        self.init_temp = 20

        # other
        self.g = np.array([], dtype=np.double)
        self.g_bhw = np.array([], dtype=np.double)
        self.lntts = np.array([], dtype=np.double)
        self.c_0 = TWO_PI * self.soil.k
        soil_diffusivity = self.soil.k / self.soil.rhoCp
        self.t_s = self.bhr._bh.bh_length**2 / (9 * soil_diffusivity)
        # default is at least 49 hours, or up to -8.6 log time
        self.calc_time_in_sec = max([self.t_s * exp(-8.6), 49.0 * SEC_IN_HR])
        self.g_sts = None

    # def calc_fluid_pipe_resistance(self) -> float:
    #     self.h_f = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
    #         self.m_flow_borehole,
    #         self.pipe.r_in,
    #         self.fluid.mu,
    #         self.fluid.rho,
    #         self.fluid.k,
    #         self.fluid.cp,
    #         self.pipe.roughness,
    #     )
    #     self.R_f = self.compute_fluid_resistance(self.h_f, self.pipe.r_in)
    #     self.R_p = gt.pipes.conduction_thermal_resistance_circular_pipe(self.pipe.r_in, self.pipe.r_out, self.pipe.k)
    #     self.R_fp = self.R_f + self.R_p
    #     return self.R_fp

    # def calc_effective_borehole_resistance(self) -> float:
    #     # TODO: should this be here?
    #     self._initialize_stored_coefficients()
    #     resist_bh_effective = self.effective_borehole_thermal_resistance(self.m_flow_borehole, self.fluid.cp)
    #     return resist_bh_effective

    def to_single(self):
        return self

    def as_dict(self) -> dict:
        return {"type": str(self.__class__)}

    def partial_init(self):
        # TODO: unravel how to eliminate this.
        # - It was calling the full class ctor "self.__init__()" which is just plain wrong...
        # - Now we're calling a stripped down version with only the most essential
        #   variables which are required.
        # - This is here partially because equivalent boreholes are generated.
        soil_diffusivity = self.k_s / self.soil.rhoCp
        self.t_s = self.b.H**2 / (9 * soil_diffusivity)
        self.calc_time_in_sec = max([self.t_s * exp(-8.6), 49.0 * SEC_IN_HR])

    def fill_radial_cells(self, resist_f_effective, resist_pg_effective):
        radial_cells = np.zeros(shape=(len(CellProps), self.num_cells), dtype=np.double)

        cell_summation = 0

        def fill_single_cell(inner_radius, thickness, conductivity, rho_cp):
            center_radius = inner_radius + thickness / 2.0
            outer_radius = inner_radius + thickness
            volume = pi * (outer_radius**2 - inner_radius**2)
            return np.array(
                [inner_radius, center_radius, outer_radius, conductivity, rho_cp, self.init_temp, volume],
                dtype=np.double,
            )

        # load fluid cells
        # The equivalent thermal mass of the fluid can be calculated from
        # equation (2)
        # pi (r_in_conv ** 2 - r_f **2) C_eq_f = 2pi r_p_in**2 * C_f
        rho_cp_eq_fluid = 2.0 * (self.pipe.r_in**2) * self.fluid.rhoCp
        rho_cp_eq_fluid /= (self.r_convection**2) - (self.r_fluid**2)
        conductivity_fluid = 200
        for idx in range(cell_summation, self.num_fluid_cells + cell_summation):
            inner_radius_fluid_cell = self.r_fluid + idx * self.thickness_fluid_cell
            radial_cells[:, idx] = fill_single_cell(
                inner_radius_fluid_cell, self.thickness_fluid_cell, conductivity_fluid, rho_cp_eq_fluid
            )

        cell_summation += self.num_fluid_cells

        # load convection cells
        conductivity_conv = log(self.r_in_tube / self.r_convection) / (TWO_PI * resist_f_effective)
        rho_cp_conv = 1.0
        for j, idx in enumerate(range(cell_summation, self.num_conv_cells + cell_summation)):
            inner_radius_conv_cell = self.r_convection + j * self.thickness_conv_cell
            radial_cells[:, idx] = fill_single_cell(
                inner_radius_conv_cell, self.thickness_conv_cell, conductivity_conv, rho_cp_conv
            )

        cell_summation += self.num_conv_cells

        # load pipe cells
        conductivity_pipe_grout = log(self.r_borehole / self.r_in_tube) / (TWO_PI * resist_pg_effective)
        rho_cp_pipe = self.pipe.rhoCp
        for j, idx in enumerate(range(cell_summation, self.num_pipe_cells + cell_summation)):
            inner_radius_pipe_cell = self.r_in_tube + j * self.thickness_pipe_cell
            radial_cells[:, idx] = fill_single_cell(
                inner_radius_pipe_cell, self.thickness_pipe_cell, conductivity_pipe_grout, rho_cp_pipe
            )

        cell_summation += self.num_pipe_cells

        # load grout cells
        rho_cp_grout = self.grout.rhoCp
        for j, idx in enumerate(range(cell_summation, self.num_grout_cells + cell_summation)):
            inner_radius_grout_cell = self.r_out_tube + j * self.thickness_grout_cell
            radial_cells[:, idx] = fill_single_cell(
                inner_radius_grout_cell, self.thickness_grout_cell, conductivity_pipe_grout, rho_cp_grout
            )

        cell_summation += self.num_grout_cells

        # load soil cells
        conductivity_soil = self.soil.k
        rho_cp_soil = self.soil.rhoCp
        for j, idx in enumerate(range(cell_summation, self.num_soil_cells + cell_summation)):
            inner_radius_soil_cell = self.r_borehole + j * self.thickness_soil_cell
            radial_cells[:, idx] = fill_single_cell(
                inner_radius_soil_cell, self.thickness_soil_cell, conductivity_soil, rho_cp_soil
            )

        cell_summation += self.num_soil_cells

        return radial_cells

    def calc_sts_g_functions(self, final_time=None) -> tuple:
        self.partial_init()

        # effective borehole resistance
        resist_bh_effective = self.calc_effective_borehole_resistance()

        # effective convection resistance, assumes 2 pipes
        resist_f_effective = self.R_f / 2.0

        # effective combined pipe-grout resistance. assumes Rees 2016, eq. 3.6 applies
        resist_pg_effective = resist_bh_effective - resist_f_effective

        radial_cells = self.fill_radial_cells(resist_f_effective, resist_pg_effective)

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

        _west_cell = radial_cells[:, 0 : self.num_cells - 2]
        _center_cell = radial_cells[:, 1 : self.num_cells - 1]
        _east_cell = radial_cells[:, 2 : self.num_cells - 0]

        fe_1 = log(radial_cells[CellProps.R_OUT, 0] / radial_cells[CellProps.R_CENTER, 0])
        fe_1 /= TWO_PI * radial_cells[CellProps.K, 0]

        fe_2 = log(radial_cells[CellProps.R_CENTER, 1] / radial_cells[CellProps.R_IN, 1])
        fe_2 /= TWO_PI * radial_cells[CellProps.K, 1]

        ae = 1 / (fe_1 + fe_2)
        ad = radial_cells[CellProps.RHO_CP, 0] * radial_cells[CellProps.VOL, 0] / time_step
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

        _ad[:] = _center_cell[CellProps.RHO_CP, :] * _center_cell[CellProps.VOL, :] / time_step
        _dl[0 : self.num_cells - 2] = -_aw / _ad
        _d[1 : self.num_cells - 1] = _aw / _ad - _ae / _ad - 1.0
        _du[1 : self.num_cells - 1] = _ae / _ad

        while True:
            time += time_step

            # For the idx == 0 case:

            _b[0] = -radial_cells[CellProps.TEMP, 0] - heat_flux / ad

            # For the idx == n-1 case

            _dl[self.num_cells - 2] = 0.0
            _d[self.num_cells - 1] = 1.0
            _b[self.num_cells - 1] = radial_cells[CellProps.TEMP, self.num_cells - 1]

            # Now handle the 1 to n-2 cases with numpy slicing and vectorization
            _b[1 : self.num_cells - 1] = -radial_cells[CellProps.TEMP, 1 : self.num_cells - 1]

            # Tri-diagonal matrix solver
            # High level interface to LAPACK routine
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lapack.dgtsv.html#scipy.linalg.lapack.dgtsv
            dgtsv(_dl, _d, _du, _b, overwrite_b=1)  # TODO: Do we really need lapack just to do a TDMA solution?

            radial_cells[CellProps.TEMP, :] = _b

            # compute standard g-functions
            g.append(self.c_0 * ((radial_cells[CellProps.TEMP, 0] - init_temp) / heat_flux - resist_bh_effective))

            # compute g-functions at bh wall
            bh_wall_temp = radial_cells[CellProps.TEMP, self.bh_wall_idx]
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
