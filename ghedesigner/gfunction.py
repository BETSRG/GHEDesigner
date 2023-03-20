import warnings
from math import log

import numpy as np
import pygfunction as gt
from scipy.interpolate import interp1d, lagrange

from ghedesigner.borehole import GHEBorehole
from ghedesigner.borehole_heat_exchangers import get_bhe_object
from ghedesigner.enums import BHPipeType


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
):
    bore_field = []
    bhe_objects = []

    h = borehole.H
    r_b = borehole.r_b
    d = borehole.D
    tilt = borehole.tilt
    orientation = borehole.orientation

    for x, y in coordinates:
        _borehole = GHEBorehole(h, d, r_b, x, y, tilt, orientation)
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
        else:
            segment_ratios = segment_ratios
        options = {
            "nSegments": n_segments,
            "segment_ratios": segment_ratios,
            "disp": disp,
        }
    else:
        raise ValueError("Equal or Unequal are acceptable options " "for segments.")

    if boundary == "UHTR" or boundary == "UBWT":
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
):
    d = {"g": {}, "bore_locations": coordinates, "logtime": log_time}

    for h in h_values:
        _borehole = GHEBorehole(h, depth, r_b, 0.0, 0.0)

        alpha = soil.k / soil.rhoCp

        ts = h ** 2 / (9.0 * alpha)  # Bore field characteristic time
        time_values = np.exp(log_time) * ts

        gfunc = calculate_g_function(
            m_flow_borehole,
            bhe_type,
            time_values,
            coordinates,
            _borehole,
            fluid,
            pipe,
            grout,
            soil,
            n_segments=n_segments,
            segments=segments,
            solver=solver,
            boundary=boundary,
            segment_ratios=segment_ratios,
        )

        key = f"{b}_{h}_{r_b}_{d}"

        d["g"][key] = gfunc.gFunc.tolist()

    geothermal_g_input = GFunction.configure_database_file_for_usage(d)
    # Initialize the gFunction object
    g_function = GFunction(**geothermal_g_input)

    return g_function


class GFunction:
    def __init__(
            self,
            b: float,
            r_b_values: dict,
            d_values: dict,
            g_lts: dict,
            log_time: list,
            bore_locations: list,
    ):
        self.B: float = b  # a B spacing in the borefield
        # r_b (borehole radius) value keyed by height
        self.r_b_values: dict = r_b_values
        self.D_values: dict = d_values  # D (burial depth) value keyed by height
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

        # Determine if we are out of range and need to extrapolate
        height_values = list(self.g_lts.keys())
        # If we are close to the outer bounds, then set H_eq as outer bounds
        if abs(h_eq - max(height_values)) < 1.0e-6:
            h_eq = max(height_values)
        if abs(h_eq - min(height_values)) < 1.0e-6:
            h_eq = min(height_values)

        if min(height_values) <= h_eq <= max(height_values):
            fill_value = ""
        elif abs(min(height_values) - h_eq) < 0.001:
            fill_value = ""
        else:
            fill_value = "extrapolate"
            warnings.warn("Extrapolation is being used.")

        # if the interpolation kind is default, use what we know about the
        # accuracy of interpolation to choose a technique
        if kind == "default":
            num_curves = len(height_values)
            if num_curves >= 5:
                kind = "cubic"
            elif num_curves >= 3:
                kind = "quadratic"
            elif num_curves == 2:
                kind = "linear"
            else:
                if (h_eq - height_values[0]) / height_values[0] < 0.001:
                    g_function = self.g_lts[height_values[0]]
                    rb = self.r_b_values[height_values[0]]
                    d = self.D_values[height_values[0]]
                    return g_function, rb, d, h_eq
                elif min(height_values) - h_eq < 0.001:
                    g_function = self.g_lts[height_values[0]]
                    rb = self.r_b_values[height_values[0]]
                    d = self.D_values[height_values[0]]
                    return g_function, rb, d, h_eq
                else:
                    raise ValueError(
                        "The interpolation requires two g-function curves "
                        "if the requested B/H is not already computed.")

        # Automatically adjust interpolation if necessary
        # Lagrange also needs 2
        interpolation_kinds = {"linear": 2, "quadratic": 3, "cubic": 4, "lagrange": 2}
        curves_by_kind = {2: "linear", 3: "quadratic", 4: "cubic"}
        if len(height_values) < 2:
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
                if kind == "lagrange":
                    f = lagrange(x, y)
                else:
                    f = interp1d(x, y, kind=kind, fill_value=fill_value)
                self.interpolation_table["g"].append(f)
            # create interpolation tables for 'D' and 'r_b' by height
            keys = list(self.r_b_values.keys())
            height_values: list = []
            rb_values: list = []
            d_values: list = []
            for h in keys:
                height_values.append(float(h))
                rb_values.append(self.r_b_values[h])
                try:
                    d_values.append(self.D_values[h])
                except:
                    pass
            if kind == "lagrange":
                rb_f = lagrange(height_values, rb_values)
            else:
                # interpolation function for rb values by H equivalent
                rb_f = interp1d(height_values, rb_values, kind=kind, fill_value=fill_value)
            self.interpolation_table["rb"] = rb_f
            try:
                if kind == "lagrange":
                    d_f = lagrange(height_values, d_values)
                else:
                    d_f = interp1d(height_values, d_values, kind=kind, fill_value=fill_value)
                self.interpolation_table["D"] = d_f
            except:
                pass

        # create the g-function by interpolating at each ln(t/ts) value
        rb_value = self.interpolation_table["rb"](h_eq)
        try:
            d_value = self.interpolation_table["D"](h_eq)
        except:
            d_value = None
        g_function: list = []
        for i in range(len(self.log_time)):
            f = self.interpolation_table["g"][i]
            g = f(h_eq).tolist()
            g_function.append(g)
        return g_function, rb_value, d_value, h_eq

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
        g_function_corrected = []
        for g in g_function:
            g_function_corrected.append(g - log(rb_star / rb))
        return g_function_corrected

    @staticmethod
    def configure_database_file_for_usage(data) -> dict:
        """
        This method is called upon initialization of the object.
        Read the cpgfunction output dictionary into the borefield class for easy
        access of information
        Parameters
        ----------
        data: dict
            A dictionary which is in the output format of cpgfunction.
        Returns
        -------
            a dictionary for input to this object
        """
        log_time = data["logtime"]

        bore_locations = data[
            "bore_locations"
        ]  # store the bore locations in the object
        g_values: dict = data["g"]  # pull the g-functions into the g_values
        # a temporary g-function dictionary that might be out of order
        g_tmp: dict = {}
        # a temporary burial depth dictionary that may be out of order
        ds_tmp: dict = {}
        # the borehole radius dictionary that may be out of order
        r_bs_tmp: dict = {}

        for key in g_values:
            # do the g-function dictionary
            key_split = key.split("_")
            # get the current height
            height = float((key_split[1]))
            # create a g-function list associated with this height key
            g_tmp[height] = g_values[key]
            # create an r_b value associated with this height key
            r_b = float(key_split[2])
            r_bs_tmp[height] = r_b
            # the D value is recently added to the key value for the saved
            # g-functions computed
            try:
                d = float(key_split[3])
                ds_tmp[height] = d
            except:
                pass

        # every B-spacing should be the same for each file
        b = float(list(g_values.keys())[0].split("_")[0])

        keys = sorted(list(g_tmp.keys()), key=int)  # sort the heights in order
        # fill the g-function dictionary with sorted heights
        g = {key: g_tmp[key] for key in keys}
        # fill the burial depth dictionary with sorted heights
        try:
            ds = {key: ds_tmp[key] for key in keys}
        except:
            # if there's no D provided, make it 2
            ds = {key: 2.0 for key in keys}
        r_bs = {key: r_bs_tmp[key] for key in keys}

        geothermal_g_input = {
            "b": b,
            "r_b_values": r_bs,
            "d_values": ds,
            "g_lts": g,
            "log_time": log_time,
            "bore_locations": bore_locations
        }

        return geothermal_g_input
