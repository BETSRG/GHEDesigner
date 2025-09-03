import logging
import warnings
from math import log

import numpy as np
import pygfunction as gt
from pygfunction.boreholes import Borehole
from pygfunction.enums import PipeType as PyPipeType
from pygfunction.gfunction import gFunction
from scipy.interpolate import interp1d, lagrange

from ghedesigner.enums import PipeType

logging.basicConfig(level=logging.WARN, format="%(message)s", datefmt="[%X]")
logger = logging.getLogger(__name__)

pyg_pipe_type_map = {
    PipeType.SINGLEUTUBE.name: PyPipeType.SINGLE_UTUBE.name,
    PipeType.DOUBLEUTUBESERIES.name: PyPipeType.DOUBLE_UTUBE_SERIES.name,
    PipeType.DOUBLEUTUBEPARALLEL.name: PyPipeType.DOUBLE_UTUBE_PARALLEL.name,
    PipeType.COAXIAL.name: PyPipeType.COAXIAL_ANNULAR_OUT.name,
}


def calculate_g_function(
    m_flow_borehole,
    bhe_type: PipeType,
    time_values,
    coordinates,
    borehole,
    fluid,
    pipe,
    grout,
    soil,
    boundary_condition="MIFT",
):
    match bhe_type:
        case PipeType.SINGLEUTUBE | PipeType.DOUBLEUTUBESERIES | PipeType.DOUBLEUTUBEPARALLEL:
            r_inner = pipe.r_in
            r_outer = pipe.r_out
        case PipeType.COAXIAL:
            # converting to pygfunction coaxial pipe conventions
            # this assumes the pipe (not annulus) is the inlet
            r_inner = [pipe.r_in[0], pipe.r_out[0]]
            r_outer = [pipe.r_in[1], pipe.r_out[1]]
        case _:
            raise ValueError(f"bhe_type {bhe_type} is not supported")

    # setup options
    # none of these were ever used or even exposed for users to access them. hardcoding them here until needed.
    solver = "equivalent"
    disp = False
    n_segments = 8
    end_length_ratio = 0.02

    options = {
        "nSegments": n_segments,
        "segment_ratios": gt.utilities.segment_ratios(nSegments=n_segments, end_length_ratio=end_length_ratio),
        "disp": disp,
    }

    nbh = len(coordinates)
    m_flow_network = nbh * m_flow_borehole

    g_func = gFunction.from_static_params(
        H=borehole.H,
        D=borehole.D,
        r_b=borehole.r_b,
        x=[x for x, _ in coordinates],
        y=[y for _, y in coordinates],
        alpha=soil.alpha,
        options=options,
        method=solver,
        boundary_condition=boundary_condition,
        k_p=pipe.k,
        k_s=soil.k,
        k_g=grout.k,
        epsilon=pipe.roughness,
        fluid_str=fluid.name,
        fluid_concentration_pct=fluid.concentration_percent,
        pos=pipe.pos,
        r_in=r_inner,
        r_out=r_outer,
        pipe_type_str=PyPipeType[pyg_pipe_type_map[bhe_type.name]].name,
        m_flow_network=m_flow_network,
    )

    g_func_vals = g_func.evaluate_g_function(time_values)

    return g_func_vals


def calc_g_func_for_multiple_lengths(
    b: float,
    h_values: list,
    r_b,
    depth,
    m_flow_borehole,
    bhe_type: PipeType,
    log_time,
    coordinates,
    fluid,
    pipe,
    grout,
    soil,
):
    r_b_values = dict.fromkeys(h_values, r_b)
    g_lts_values = {}

    alpha = soil.k / soil.rho_cp

    for h in h_values:
        borehole = Borehole(h, depth, r_b, 0.0, 0.0)

        ts = h**2 / (9.0 * alpha)  # Bore field characteristic time
        time_values = np.exp(log_time) * ts

        g_lts_values[h] = calculate_g_function(
            m_flow_borehole,
            bhe_type,
            time_values,
            coordinates,
            borehole,
            fluid,
            pipe,
            grout,
            soil,
        ).tolist()

    # Initialize the gFunction object
    return GFunction(
        b=b, r_b_values=r_b_values, d=depth, g_lts=g_lts_values, log_time=log_time, bore_locations=coordinates
    )


class GFunction:
    def __init__(
        self,
        b: float,
        d: float,
        r_b_values: dict,
        g_lts: dict,
        log_time: list,
        bore_locations: list,
    ) -> None:
        self.B: float = b  # a B spacing in the borefield
        # r_b (borehole radius) value keyed by height
        self.r_b_values: dict = r_b_values
        self.d: float = d  # burial depth
        self.g_lts: dict = g_lts  # g-functions (LTS) keyed by height
        # ln(t/ts) values that apply to all the heights
        self.log_time: list = log_time
        # (x, y) coordinates of boreholes
        self.bore_locations: list = bore_locations
        # self.time: dict = {}  # the time values in years

        # an interpolation table for B/H ratios, D, r_b (used in the method
        # g_function_interpolation)
        self.interpolation_table: dict = {}

    def g_function_interpolation(self, b_over_h: float, kind="default"):
        # the g-functions are stored in a dictionary based on heights, so an
        # equivalent height can be found
        h_eq = 1 / b_over_h * self.B

        tolerance = 0.001

        # Determine if we are out of range and need to extrapolate
        height_values = list(self.g_lts.keys())
        # If we are close to the outer bounds, then set H_eq as outer bounds
        close_tolerance = 1.0e-6
        if abs(h_eq - max(height_values)) < close_tolerance:
            h_eq = max(height_values)
        if abs(h_eq - min(height_values)) < close_tolerance:
            h_eq = min(height_values)

        if min(height_values) <= h_eq <= max(height_values) or abs(min(height_values) - h_eq) < tolerance:
            fill_value = ""
        else:
            fill_value = "extrapolate"
            warnings.warn("Extrapolation is being used.")

        # if the interpolation kind is default, use what we know about the
        # accuracy of interpolation to choose a technique

        if kind == "default":
            num_curves = len(height_values)
            if num_curves >= 5:  # noqa: PLR2004
                kind = "cubic"
            elif num_curves >= 3:  # noqa: PLR2004
                kind = "quadratic"
            elif num_curves == 2:  # noqa: PLR2004
                kind = "linear"
            elif (h_eq - height_values[0]) / height_values[0] < tolerance or min(height_values) - h_eq < tolerance:
                g_function = self.g_lts[height_values[0]]
                rb = self.r_b_values[height_values[0]]
                return g_function, rb, self.d, h_eq
            else:
                raise ValueError(
                    "The interpolation requires two g-function curves if the requested B/H is not already computed."
                )

        # Automatically adjust interpolation if necessary
        # Lagrange also needs 2
        interpolation_kinds = {"linear": 2, "quadratic": 3, "cubic": 4, "lagrange": 2}
        curves_by_kind = {2: "linear", 3: "quadratic", 4: "cubic"}
        if len(height_values) < 2:  # noqa: PLR2004
            raise ValueError("Interpolation requires two g-function curves.")
        else:
            # If the number of required curves for the interpolation type is
            # greater than what is available, reduce the interpolation type
            required_curves = interpolation_kinds[kind]
            if required_curves > len(height_values):
                kind = curves_by_kind[len(height_values)]

        # if the interpolation table is not yet know, build it
        if len(self.interpolation_table) == 0:
            # create an interpolation for the g-function which takes the height
            # (or equivalent height) as an input the g-function needs to be
            # interpolated at each point in dimensionless time
            self.interpolation_table["g"] = []
            for i, _ in enumerate(self.log_time):
                x = []
                y = []
                for key in self.g_lts:
                    height_value = float(key)
                    g_value = self.g_lts[key][i]
                    x.append(height_value)
                    y.append(g_value)
                f = lagrange(x, y) if kind == "lagrange" else interp1d(x, y, kind=kind, fill_value=fill_value)
                self.interpolation_table["g"].append(f)
            # create interpolation tables for 'D' and 'r_b' by height
            keys = list(self.r_b_values.keys())
            height_values_2 = []
            rb_values: list = []
            for h in keys:
                height_values_2.append(float(h))
                rb_values.append(self.r_b_values[h])
            if kind == "lagrange":
                rb_f = lagrange(height_values_2, rb_values)
            else:
                # interpolation function for rb values by H equivalent
                rb_f = interp1d(height_values_2, rb_values, kind=kind, fill_value=fill_value)
            self.interpolation_table["rb"] = rb_f

        # create the g-function by interpolating at each ln(t/ts) value
        rb_value = self.interpolation_table["rb"](h_eq)
        g_function_l = []
        for i in range(len(self.log_time)):
            f = self.interpolation_table["g"][i]
            g = f(h_eq).tolist()
            g_function_l.append(g)
        return g_function_l, rb_value, self.d, h_eq

    @staticmethod
    def borehole_radius_correction(g_function: list, rb: float, rb_star: float):
        r"""
        Correct the borehole radius. From paper 3 of Eskilson 1987.
        .. math::
            g(\dfrac{t}{t_s}, \dfrac{r_b^*}{H}) =
            g(\dfrac{t}{t_s}, \dfrac{r_b}{H}) - ln(\dfrac{r_b^*}{r_b})
        Parameters
        ----------
        g_function: list
            A g-function
        rb: float
            The current borehole radius
        rb_star: float
            The borehole radius that is being corrected to
        Returns
        -------
        g_function_corrected: list
            A corrected g_function
        """
        g_function_corrected = [g - log(rb_star / rb) for g in g_function]
        return g_function_corrected
