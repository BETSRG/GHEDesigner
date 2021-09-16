# Jack C. Cook
# Thursday, September 16, 2021

from scipy.interpolate import interp1d, lagrange
import matplotlib.pyplot as plt
import numpy as np


class GFunction:
    def __init__(self, B: float, r_b_values: dict, D_values: dict,
                 g_lts: dict, log_time: dict, bore_locations: list):
        self.B: float = B  # a B spacing in the borefield
        # r_b (borehole radius) value keyed by height
        self.r_b_values: dict = {}
        self.D_values: dict = {}  # D (burial depth) value keyed by height
        self.g_lts: dict = {}  # g-functions (LTS) keyed by height
        # ln(t/ts) values that apply to all the heights
        self.log_time: list = []
        self.bore_locations: list = []  # (x, y) coordinates of boreholes
        # self.time: dict = {}  # the time values in years

        # an interpolation table for B/H ratios, D, r_b (used in the method
        # g_function_interpolation)
        self.interpolation_table: dict = {}

    def g_function_interpolation(self, B_over_H: float, kind='cubic'):
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
                    f = interp1d(x, y, kind=kind)
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
                rb_f = interp1d(height_values, rb_values, kind=kind)
            self.interpolation_table['rb'] = rb_f
            try:
                if kind == 'lagrange':
                    D_f = lagrange(height_values, D_values)
                else:
                    D_f = interp1d(height_values, D_values, kind=kind)
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

    def visualize_g_functions(self):
        """
        Visualize the g-functions.
        Returns
        -------
        **fig, ax**
            Figure and axes information.
        """
        fig, ax = plt.subplots()

        ax.set_xlim([-8.8, 3.9])
        ax.set_ylim([-2, 139])

        ax.text(2.75, 135, 'B/H')

        keys = reversed(list(self.g_lts.keys()))

        for key in keys:
            ax.plot(self.log_time, self.g_lts[key],
                    label=str(int(self.B)) + '/' + str(key))
            x_n = self.log_time[-1]
            y_n = self.g_lts[key][-1]
            if key == 8:
                ax.annotate(str(round(float(self.B) / float(key), 4)),
                            xy=(x_n - .4, y_n - 5))
            else:
                ax.annotate(str(round(float(self.B) / float(key), 4)),
                            xy=(x_n-.4, y_n+1))

        handles, labels = ax.get_legend_handles_labels()

        legend = fig.legend(handles=handles, labels=labels,
                            title='B/H'.rjust(5) + '\nLibrary',
                            bbox_to_anchor=(1, 1.0))
        fig.gca().add_artist(legend)

        ax.set_ylabel('g')
        ax.set_xlabel('ln(t/t$_s$)')
        ax.grid()
        ax.set_axisbelow(True)
        fig.subplots_adjust(left=0.09, right=0.835, bottom=0.1, top=.99)

        return fig, ax

    def visualize_borefield(self):
        """
        Visualize the (x,y) coordinates.
        Returns
        -------
        **fig, ax**
            Figure and axes information.
        """
        fig, ax = plt.subplots(figsize=(3.5,5))

        x, y = list(zip(*self.bore_locations))

        ax.scatter(x, y)

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')

        ax.set_aspect('equal')
        fig.tight_layout()

        return fig, ax
