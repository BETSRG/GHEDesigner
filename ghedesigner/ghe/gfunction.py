import logging
import warnings
from math import log
from copy import deepcopy
import numpy as np
import pygfunction as gt
from pygfunction.boreholes import Borehole
from scipy.interpolate import interp1d, lagrange

from ghedesigner.enums import BHPipeType
from ghedesigner.ghe.boreholes.factory import get_bhe_object

logging.basicConfig(level=logging.WARN, format="%(message)s", datefmt="[%X]")
logger = logging.getLogger(__name__)


def calculate_g_function(
    m_flow_borehole,
    bhe_type: BHPipeType,
    time_values,
    coordinates,
    borehole,
    fluid,
    pipe,
    grout,
    soil,
    n_segments=8,
    end_length_ratio=0.02,
    segments="unequal",
    solver="equivalent",
    boundary="MIFT",
    segment_ratios=None,
    disp=False,
    tilts=None,
    orientations=None
):
    bore_field = []
    bhe_objects = []

    h = borehole.H
    r_b = borehole.r_b
    d = borehole.D
    # tilt = borehole.tilt
    # orientation = borehole.orientation

    for i, (x, y) in enumerate(coordinates):
        if tilts is not None and orientations is not None:
            _borehole = Borehole(h, d, r_b, x, y, tilts[i], orientations[i])
        else:
            _borehole = Borehole(h, d, r_b, x, y)
        bore_field.append(_borehole)
        # Initialize pipe model
        if boundary == "MIFT":
            bhe = get_bhe_object(bhe_type, m_flow_borehole, fluid, _borehole, pipe, grout, soil)
            bhe_objects.append(bhe)

    alpha = soil.k / soil.rhoCp

    # setup options
    segments = segments.lower()
    if segments == "equal":
        options = {"nSegments": n_segments, "disp": disp}
    elif segments == "unequal":
        if segment_ratios is None:
            segment_ratios = gt.utilities.segment_ratios(n_segments, end_length_ratio=end_length_ratio)
        options = {
            "nSegments": n_segments,
            "segment_ratios": segment_ratios,
            "disp": disp,
        }
    else:
        raise ValueError("Equal or Unequal are acceptable options for segments.")

    if boundary in ("UHTR", "UBWT"):
        gfunc = gt.gfunction.gFunction(
            bore_field,
            alpha,
            time=time_values,
            boundary_condition=boundary,
            options=options,
            method=solver,
        )
    elif boundary == "MIFT":
        m_flow_network = len(bore_field) * m_flow_borehole
        network = gt.networks.Network(bore_field, bhe_objects, m_flow_network=m_flow_network, cp_f=fluid.cp)
        gfunc = gt.gfunction.gFunction(
            network,
            alpha,
            time=time_values,
            boundary_condition=boundary,
            options=options,
            method=solver,
        )
    else:
        raise ValueError("UHTR, UBWT or MIFT are accepted boundary conditions.")

    return gfunc


def calc_g_func_for_multiple_lengths(
    b: float,
    h_values: list,
    r_b,
    depth,
    m_flow_borehole,
    bhe_type: BHPipeType,
    log_time,
    coordinates,
    fluid,
    pipe,
    grout,
    soil,
    n_segments=8,
    segments="unequal",
    solver="equivalent",
    boundary="MIFT",
    segment_ratios=None,
    tilts=None,
    orientations=None
):

    if tilts is not None and orientations is not None and solver == "equivalent":
        raise Warning("pygfunction's equivalent solver cannot use tilted boreholes. Using similarities solver instead.")

    r_b_values = dict.fromkeys(h_values, r_b)
    g_lts_values = {}

    alpha = soil.k / soil.rhoCp

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
            n_segments=n_segments,
            segments=segments,
            solver=solver,
            boundary=boundary,
            segment_ratios=segment_ratios,
            tilts=tilts,
            orientations=orientations
        ).gFunc.tolist()

    # Initialize the gFunction object
    return GFunction(
        b=b, r_b_values=r_b_values, d=depth, g_lts=g_lts_values, log_time=log_time, bore_locations=coordinates,
        bore_tilts=tilts, bore_orientations=orientations
    )

def merge_g_functions(g_func_mid, g_func_max):
    if g_func_mid.bore_locations != g_func_max.bore_locations:
        raise ValueError(f"Borehole coordinates do not match, unable to merge")
    if g_func_mid.bore_tilts != g_func_max.bore_tilts:
        raise ValueError(f"Borehole tilts do not match, unable to merge")
    if g_func_mid.bore_orientations != g_func_max.bore_orientations:
        raise ValueError(f"Borehole orientations do not match, unable to merge")
    if  g_func_mid.B != g_func_max.B:
        raise ValueError(f"Borehole spacings do not match, unable to merge")
    if g_func_mid.d != g_func_max.d:
        raise ValueError(f"Borehole depths do not match, unable to merge")
    if g_func_mid.log_time != g_func_max.log_time:
        raise ValueError(f"Borehole log times do not match, unable to merge")

    new_r_b_values = deepcopy(g_func_mid.r_b_values)
    new_g_lts = deepcopy(g_func_mid.g_lts)

    for h, rb in g_func_max.r_b_values.items():
        if h not in new_r_b_values:
            new_r_b_values[h] = rb
    for h, lts in g_func_max.g_lts.items():
        if h not in new_g_lts:
            new_g_lts[h] = lts

    return GFunction(
        b=g_func_mid.B, r_b_values=new_r_b_values, d=g_func_mid.d, g_lts=new_g_lts, log_time=g_func_mid.log_time, bore_locations=g_func_mid.bore_locations,
        bore_tilts=g_func_mid.bore_tilts, bore_orientations=g_func_mid.bore_orientations
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
        bore_tilts: list,
        bore_orientations: list,
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
        self.bore_tilts = bore_tilts
        self.bore_orientations = bore_orientations
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
