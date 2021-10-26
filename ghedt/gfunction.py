# Jack C. Cook
# Monday, October 25, 2021
import copy
import warnings

from scipy.interpolate import interp1d, lagrange
import ghedt
import ghedt.PLAT.pygfunction as gt
import numpy as np


def calculate_g_function(
        m_flow_borehole, bhe_object, time_values, coordinates, borehole,
        nSegments, fluid, pipe, grout, soil, segments='unequal',
        solver='equivalent', boundary='MIFT', disp=False):

    boreField = []
    BHEs = []

    H = copy.deepcopy(borehole.H)
    r_b = copy.deepcopy(borehole.r_b)
    D = copy.deepcopy(borehole.D)
    tilt = copy.deepcopy(borehole.tilt)
    orientation = copy.deepcopy(borehole.orientation)

    for i in range(len(coordinates)):
        x, y = coordinates[i]
        _borehole = gt.boreholes.Borehole(H, D, r_b, x, y, tilt, orientation)
        boreField.append(_borehole)
        # Initialize pipe model
        if boundary == 'MIFT':
            bhe = \
                bhe_object(m_flow_borehole, fluid, _borehole, pipe, grout, soil)
            BHEs.append(bhe)

    alpha = soil.k / soil.rhoCp

    # setup options
    segments = segments.lower()
    if segments == 'equal':
        options = {'nSegments': nSegments, 'disp': disp}
    elif segments == 'unequal':
        segment_ratios = gt.utilities.segment_ratios(nSegments)
        options = {'nSegments': nSegments, 'segment_ratios': segment_ratios,
                   'disp': disp}
    else:
        raise ValueError('Equal or Unequal are acceptable options '
                         'for segments.')

    if boundary == 'UHTR' or boundary == 'UBWT':
        gfunc = gt.gfunction.gFunction(
            boreField, alpha, time=time_values, boundary_condition=boundary,
            options=options, method=solver
        )
    elif boundary == 'MIFT':
        m_flow_network = len(boreField) * m_flow_borehole
        network = gt.networks.Network(
            boreField, BHEs, m_flow_network=m_flow_network, cp_f=fluid.cp)
        gfunc = gt.gfunction.gFunction(
            network, alpha, time=time_values,
            boundary_condition=boundary, options=options, method=solver)
    else:
        raise ValueError('UHTR, UBWT or MIFT are accepted boundary conditions.')

    return gfunc


class GFunction:
    def __init__(self, B: float, r_b_values: dict, D_values: dict,
                 g_lts: dict, log_time: list, bore_locations: list):
        self.B: float = B  # a B spacing in the borefield
        # r_b (borehole radius) value keyed by height
        self.r_b_values: dict = r_b_values
        self.D_values: dict = D_values  # D (burial depth) value keyed by height
        self.g_lts: dict = g_lts  # g-functions (LTS) keyed by height
        # ln(t/ts) values that apply to all the heights
        self.log_time: list = log_time
        # (x, y) coordinates of boreholes
        self.bore_locations: list = bore_locations
        # self.time: dict = {}  # the time values in years

        # an interpolation table for B/H ratios, D, r_b (used in the method
        # g_function_interpolation)
        self.interpolation_table: dict = {}

    def g_function_interpolation(self, B_over_H: float, kind='default'):
        """
        Interpolate a range of g-functions for a specific B/H ratio
        Parameters
        ----------
        B_over_H: float
            A B/H ratio
        kind: str
            Could be 'linear', 'quadratic', 'cubic', etc.
            default: 'cubic'
        Returns
        -------
        **g-function: list**
            A list of the g-function values for each ln(t/ts)
        **rb: float**
            A borehole radius value that is interpolated for
        **D: float**
            A burial depth that is interpolated for
        **H_eq: float**
            An equivalent height
            .. math::
                H_{eq} = \dfrac{B_{field}}{B/H}
        """
        # the g-functions are stored in a dictionary based on heights, so an
        # equivalent height can be found
        H_eq = 1 / B_over_H * self.B

        # Determine if we are out of range and need to extrapolate
        height_values = list(self.g_lts.keys())
        if min(height_values) <= H_eq <= max(height_values):
            fill_value = ''
        else:
            fill_value = 'extrapolate'
            warnings.warn('Extrapolation is being used.')

        # if the interpolation kind is default, use what we know about the
        # accuracy of interpolation to choose a technique
        if kind == 'default':
            num_curves = len(height_values)
            if num_curves >= 5:
                kind = 'cubic'
            elif num_curves >= 3:
                kind = 'quadratic'
            elif num_curves == 2:
                kind = 'linear'
            else:
                raise ValueError(
                    'Interpolation requires two g-function curves.')

        # Automatically adjust interpolation if necessary
        # Lagrange also needs 2
        interpolation_kinds = {'linear': 2, 'quadratic': 3, 'cubic': 4,
                               'lagrange': 2}
        curves_by_kind = {2: 'linear', 3: 'quadratic', 4: 'cubic'}
        if len(height_values) < 2:
            raise ValueError('Interpolation requires two g-function curves.')
        else:
            # If the number of required curves for the interpolation type is
            # greater than what is available, reduce the interpolation type
            required_curves = interpolation_kinds[kind]
            if required_curves > len(height_values):
                kind = curves_by_kind[len(height_values)]

        # if the interpolation table is not yet know, build it
        if len(self.interpolation_table) == 0:
            # create an interpolation for the g-function which takes the height
            # (or equivilant height) as an input the g-function needs to be
            # interpolated at each point in dimensionless time
            self.interpolation_table['g'] = []
            for i, lntts in enumerate(self.log_time):
                x = []
                y = []
                for key in self.g_lts:
                    height_value = float(key)
                    g_value = self.g_lts[key][i]
                    x.append(height_value)
                    y.append(g_value)
                if kind == 'lagrange':
                    f = lagrange(x, y)
                else:
                    f = interp1d(x, y, kind=kind, fill_value=fill_value)
                self.interpolation_table['g'].append(f)
            # create interpolation tables for 'D' and 'r_b' by height
            keys = list(self.r_b_values.keys())
            height_values: list = []
            rb_values: list = []
            D_values: list = []
            for h in keys:
                height_values.append(float(h))
                rb_values.append(self.r_b_values[h])
                try:
                    D_values.append(self.D_values[h])
                except:
                    pass
            if kind == 'lagrange':
                rb_f = lagrange(height_values, rb_values)
            else:
                # interpolation function for rb values by H equivalent
                rb_f = interp1d(height_values, rb_values, kind=kind, fill_value=fill_value)
            self.interpolation_table['rb'] = rb_f
            try:
                if kind == 'lagrange':
                    D_f = lagrange(height_values, D_values)
                else:
                    D_f = interp1d(height_values, D_values, kind=kind, fill_value=fill_value)
                self.interpolation_table['D'] = D_f
            except:
                pass

        # create the g-function by interpolating at each ln(t/ts) value
        rb_value = self.interpolation_table['rb'](H_eq)
        try:
            D_value = self.interpolation_table['D'](H_eq)
        except:
            D_value = None
        g_function: list = []
        for i in range(len(self.log_time)):
            f = self.interpolation_table['g'][i]
            g = f(H_eq).tolist()
            g_function.append(g)
        return g_function, rb_value, D_value, H_eq

    @staticmethod
    def borehole_radius_correction(g_function: list, rb: float, rb_star: float):
        """
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
        for i in range(len(g_function)):
            g_function_corrected.append(g_function[i] - np.log(rb_star / rb))
        return g_function_corrected

    def correct_coordinates(self, B: float):
        # scale the field by the ratio
        scale = B / self.B

        new_locations = []
        for i in range(len(self.bore_locations)):
            _x = self.bore_locations[i][0]
            _y = self.bore_locations[i][1]
            if _x == 0.0 and _y == 0.0:
                new_locations.append([_x, _y])
            elif _x == 0.0:
                new_locations.append([_x, _y*scale])
            else:
                x = _x * scale
                y = _y * scale
                new_locations.append([x, y])

        return new_locations

    def visualize_g_functions(self):
        """
        Visualize the g-functions.
        Returns
        -------
        **fig, ax**
            Figure and axes information.
        """
        fig = gt.utilities._initialize_figure()
        ax = fig.add_subplot(111)

        ax.set_xlim([-8.8, 3.9])
        ax.set_ylim([-2, 139])

        ax.text(2.75, 135, 'B/H')

        keys = reversed(list(self.g_lts.keys()))

        for key in keys:
            ax.plot(self.log_time, self.g_lts[key],
                    label=str(int(self.B)) + '/' + str(int(key)))
            x_n = self.log_time[-1]
            y_n = self.g_lts[key][-1]
            if key == 8:
                ax.annotate(str(round(float(self.B) / float(key), 4)),
                            xy=(x_n - .4, y_n - 5))
            else:
                ax.annotate(str(round(float(self.B) / float(key), 4)),
                            xy=(x_n-.4, y_n+1))

        handles, labels = ax.get_legend_handles_labels()

        # legend = fig.legend(handles=handles, labels=labels,
        #                     title='B/H'.rjust(5) + '\nLibrary',
        #                     bbox_to_anchor=(1, 1.0))
        legend = fig.legend(handles=handles, labels=labels,
                            title='B/H'.rjust(5), bbox_to_anchor=(1, 1.0))
        fig.gca().add_artist(legend)

        ax.set_ylabel('g')
        ax.set_xlabel('ln(t/t$_s$)')
        ax.grid()
        ax.set_axisbelow(True)
        fig.subplots_adjust(left=0.09, right=0.835, bottom=0.1, top=.99)

        return fig, ax

    def visualize_borefield(self, bore_locations=None):
        """
        Visualize the (x,y) coordinates.
        Returns
        -------
        **fig, ax**
            Figure and axes information.
        """
        if bore_locations is None:
            bore_locations = self.bore_locations
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(3.5, 5))

        x, y = list(zip(*bore_locations))

        ax.scatter(x, y)

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')

        ax.set_aspect('equal')
        fig.tight_layout()

        return fig, ax

    @staticmethod
    def visualize_area_and_constraints(perimeter: list, coordinates: list,
                                       no_go=None):
        """
        Visualize the (x,y) coordinates, and no go zone.
        Returns
        -------
        **fig, ax**
            Figure and axes information.
        """
        fig = gt.utilities._initialize_figure()
        ax = fig.add_subplot(111)

        perimeter = perimeter + [perimeter[0]]

        x, y = list(zip(*coordinates))

        ax.scatter(x, y)
        _x, _y = list(zip(*perimeter))
        ax.plot(_x, _y, 'g')
        if no_go is not None:
            no_go = no_go + [no_go[0]]
            _x, _y = list(zip(*no_go))
            ax.plot(_x, _y, 'r')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')

        fig.tight_layout()

        return fig, ax

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
        log_time = data['logtime']

        bore_locations = data[
            'bore_locations']  # store the bore locations in the object
        g_values: dict = data['g']  # pull the g-functions into the g_values
        g_tmp: dict = {}  # a temporary g-function dictionary that might be out of order
        Ds_tmp: dict = {}  # a temporary burial depth dictionary that may be out of order
        r_bs_tmp: dict = {}  # the borehole radius dictionary that may be out of order
        t_tmp: dict = {}
        for key in g_values:
            # do the g-function dictionary
            key_split = key.split('_')
            # get the current height
            height = float((key_split[1]))
            # create a g-function list associated with this height key
            g_tmp[height] = g_values[key]
            # create a r_b value associated with this height key
            r_b = float(key_split[2])
            r_bs_tmp[height] = r_b
            try:  # the D value is recently added to the key value for the saved g-functions computed
                D = float(key_split[3])
                Ds_tmp[height] = D
            except:
                pass

            # do the time dictionary
            time_arr: list = []
            for _, lntts in enumerate(log_time):
                alpha = 1.0e-06
                t_seconds = height ** 2 / 9 / alpha * np.exp(lntts)
                t_year = t_seconds / 60 / 24 / 365
                time_arr.append(t_year)
            t_tmp[height] = time_arr

        B = float(list(g_values.keys())[0].split('_')[
                           0])  # every B-spacing should be the same for each file

        keys = sorted(list(g_tmp.keys()), key=int)  # sort the heights in order

        g = {key: g_tmp[key] for key in
                  keys}  # fill the g-function dictionary with sorted heights
        try:
            Ds = {key: Ds_tmp[key] for key in
                       keys}  # fill the burial depth dictionary with sorted heights
        except:
            # if there's no D provided, make it 2
            Ds = {key: 2. for key in keys}
        r_bs = {key: r_bs_tmp[key] for key in keys}
        time = {key: t_tmp[key] for key in
                     keys}  # fill the time array for yearly points

        geothermal_g_input = {}
        geothermal_g_input['B'] = B
        geothermal_g_input['r_b_values'] = r_bs
        geothermal_g_input['D_values'] = Ds
        geothermal_g_input['g_lts'] = g
        geothermal_g_input['log_time'] = log_time
        geothermal_g_input['bore_locations'] = bore_locations

        return geothermal_g_input