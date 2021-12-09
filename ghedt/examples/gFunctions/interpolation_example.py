# Jack C. Cook
# Sunday, November 28, 2021


import ghedt.PLAT.pygfunction as gt
import ghedt
import ghedt.PLAT as PLAT
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, lagrange
import numpy as np


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
    gt.utilities._format_axes(ax)

    ax.set_xlim([-8.8, 4.5])
    ax.set_ylim([-2, 139])

    ax.text(3.3, 132, 'B/H')

    keys = reversed(list(self.g_lts.keys()))

    for key in keys:
        ax.plot(self.log_time, self.g_lts[key], marker='o',
                label=str(int(self.B)) + '/' + str(int(key)))
        x_n = self.log_time[-1]
        y_n = self.g_lts[key][-1]
        if key == 8:
            ax.annotate(str(round(float(self.B) / float(key), 4)),
                        xy=(x_n + .1, y_n - 5))
        else:
            ax.annotate(str(round(float(self.B) / float(key), 4)),
                        xy=(x_n + .1, y_n + 1))

    handles, labels = ax.get_legend_handles_labels()

    legend = fig.legend(handles=handles, labels=labels,
                        title='B/H'.rjust(5), bbox_to_anchor=(1, 1.0))
    fig.gca().add_artist(legend)

    ax.set_ylabel('g-function')
    ax.set_xlabel('ln(t/t$_s$)')
    ax.grid()
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.09, right=0.835, bottom=0.1, top=.99)

    return fig, ax


def main():
    H_values = [24., 48., 96., 192., 384.]
    r_b_values = [0.012, 0.024, 0.048, 0.096, 0.192]
    D_values = [.5, 1., 2., 4., 8.]
    B = 5

    # Pipe dimensions
    # ---------------
    r_out = 26.67 / 1000. / 2.  # Pipe outer radius (m)
    r_in = 21.6 / 1000. / 2.  # Pipe inner radius (m)
    s = 32.3 / 1000.  # Inner-tube to inner-tube Shank spacing (m)
    epsilon = 1.0e-6  # Pipe roughness (m)

    # Pipe positions
    # --------------
    # Single U-tube [(x_in, y_in), (x_out, y_out)]
    pos = PLAT.media.Pipe.place_pipes(s, r_out, 1)
    # Single U-tube BHE object
    bhe_object = PLAT.borehole_heat_exchangers.SingleUTube

    # Thermal conductivities
    # ----------------------
    k_p = 0.4  # Pipe thermal conductivity (W/m.K)
    k_s = 2.0  # Ground thermal conductivity (W/m.K)
    k_g = 1.0  # Grout thermal conductivity (W/m.K)

    # Volumetric heat capacities
    # --------------------------
    rhoCp_p = 1542. * 1000.  # Pipe volumetric heat capacity (J/K.m3)
    rhoCp_s = 2343.493 * 1000.  # Soil volumetric heat capacity (J/K.m3)
    rhoCp_g = 3901. * 1000.  # Grout volumetric heat capacity (J/K.m3)

    # Thermal properties
    # ------------------
    # Pipe
    pipe = PLAT.media.Pipe(pos, r_in, r_out, s, epsilon, k_p, rhoCp_p)
    # Soil
    ugt = 18.3  # Undisturbed ground temperature (degrees Celsius)
    soil = PLAT.media.Soil(k_s, rhoCp_s, ugt)
    # Grout
    grout = PLAT.media.ThermalProperty(k_g, rhoCp_g)

    # Eskilson's original ln(t/ts) values
    log_time = ghedt.utilities.Eskilson_log_times()

    # Inputs related to fluid
    # -----------------------
    # Fluid properties
    mixer = 'MEG'  # Ethylene glycol mixed with water
    percent = 0.  # Percentage of ethylene glycol added in
    fluid = gt.media.Fluid(mixer=mixer, percent=percent)

    # Coordinates
    Nx = 7
    Ny = 10
    coordinates = ghedt.coordinates.rectangle(Nx, Ny, B, B)

    # Fluid properties
    V_flow_borehole = 0.2  # System volumetric flow rate (L/s)
    V_flow_system = V_flow_borehole * float(Nx * Ny)
    # Total fluid mass flow rate per borehole (kg/s)
    m_flow_borehole = V_flow_borehole / 1000. * fluid.rho

    g_function = ghedt.gfunction.compute_live_g_function(
        B, H_values, r_b_values, D_values, m_flow_borehole, bhe_object,
        log_time, coordinates, fluid, pipe, grout, soil, nSegments=12,
        segments='equal', solver='equivalent', boundary='UBWT')

    fig, ax = visualize_g_functions(g_function)
    for i in range(len(g_function.log_time)):
        log_times = [g_function.log_time[i]] * len(H_values)
        g_values = []
        for j in range(len(H_values)):
            g = g_function.g_lts[H_values[j]][i]
            g_values.append(g)
        ax.plot(log_times, g_values, color='k', linestyle='--')

    fig.savefig('linear_interpolation.png')
    plt.close(fig)

    # Plot different interpolation schemes
    fig = gt.utilities._initialize_figure()
    ax = fig.add_subplot()
    gt.utilities._format_axes(ax)

    H_range = np.arange(H_values[0], H_values[-1], 0.1)

    g_values = []
    for i in range(len(H_values)):
        g = g_function.g_lts[H_values[i]][-1]
        g_values.append(g)

    # Linear interpolation
    lin = interp1d(H_values, g_values, kind='linear')
    lin_values = lin(H_range)
    # Quadratic interpolation
    quad = interp1d(H_values, g_values, kind='quadratic')
    quad_values = quad(H_range)
    # Cubic spline
    cubic = interp1d(H_values, g_values, kind='cubic')
    cubic_values = cubic(H_range)
    # Lagrange polynomial
    lagr = lagrange(H_values, g_values)
    lagr_values = lagr(H_range)

    ax.scatter(H_values, g_values, zorder=2,
               label='g-Function values at ln(t/t$_s$)=3.003')
    ax.plot(H_range, lin_values, linestyle='-', color='b',
            label='Linear spline', zorder=0)
    ax.plot(H_range, quad_values, linestyle='dotted', color='g',
            label='Quadratic spline', zorder=2)
    ax.plot(H_range, cubic_values, linestyle='dashed', color='r',
            label='Cubic spline', zorder=1)
    ax.plot(H_range, lagr_values, linestyle='-.', color='c',
            label='Lagrange polynomail', zorder=1)

    ax.set_xlabel('Height value (m)')
    ax.set_ylabel('g-value')
    ax.grid()
    ax.set_axisbelow(True)

    fig.legend(bbox_to_anchor=(0.55, 0.95))

    fig.tight_layout()

    fig.savefig('spline_methodology.png')

    rb_over_H = 0.0005
    D_over_H = 0.0208

    H_avg_values = []
    rb_avg_values = []
    D_avg_values = []
    for i in range(len(H_values)-1):
        H_lst = list(range(int(H_values[i]), int(H_values[i+1]), 6))
        for j in range(len(H_lst)):
            H_avg_values.append(H_lst[j])
            rb_avg_values.append(rb_over_H * H_lst[j])
            D_avg_values.append(D_over_H * H_lst[j])

    H_avg_values.append(384)
    rb_avg_values.append(384 * rb_over_H)
    D_avg_values.append(384 * D_over_H)

    interpolation_kinds = ['Linear', 'Quadratic', 'Cubic', 'Lagrange']
    print('Height (m)\tLinear\tQuadratic\tCubic\tLagrange')

    error_data = {'Linear': [],
                  'Quadratic': [],
                  'Cubic': [],
                  'Lagrange': []}

    for i in range(len(H_avg_values)):
        H = H_avg_values[i]
        D = D_avg_values[i]
        rb = rb_avg_values[i]
        _borehole = gt.boreholes.Borehole(H, D, rb, x=0, y=0)
        alpha = soil.k / soil.rhoCp
        ts = H ** 2 / (9. * alpha)  # Bore field characteristic time
        time_values = np.exp(log_time) * ts
        g_ref = ghedt.gfunction.calculate_g_function(
            m_flow_borehole, bhe_object, time_values, coordinates, _borehole,
            fluid, pipe, grout, soil, nSegments=12, segments='equal',
            solver='equivalent', boundary='UBWT')
        B_over_H = B / H
        mpe_values = []
        for k in range(len(interpolation_kinds)):
            g_func, rb_value, D_value, H_eq = \
                g_function.g_function_interpolation(
                    B_over_H, kind=interpolation_kinds[k].lower())
            g_function.interpolation_table = {}  # reset interpolation table
            mpe = ghedt.utilities.compute_mpe(g_ref.gFunc.tolist(), g_func)
            error_data[interpolation_kinds[k]].append(mpe)
            mpe_values.append(mpe)

        print('{}\t{}\t{}\t{}\t{}'.format(
            H, mpe_values[0], mpe_values[1], mpe_values[2],
            mpe_values[3]))

    fig = gt.utilities._initialize_figure()
    ax = fig.add_subplot(111)

    line_styles = ['-', '--', '-.', ':']
    count = 0
    for key in error_data:
        if key != 'Lagrange':
            ax.plot(H_avg_values, error_data[key],
                    label=key + ' spline', linestyle=line_styles[count])
        else:
            ax.plot(H_avg_values, error_data[key],
                    label=key + ' polynomail', linestyle=line_styles[count])
        count += 1

    ax.set_xlabel('Borehole height, $H$ (m)')
    ax.set_ylabel('Mean error of g-function (%)')

    fig.legend()

    ax.grid()
    ax.set_axisbelow(True)

    fig.tight_layout()

    fig.savefig('interpolation_errors_by_height.png')
    plt.close(fig)

    # Check the error of interpolation
    B = 5.
    H_value = 288
    rb_value = 0.144
    D_value = 5.9904
    _borehole = gt.boreholes.Borehole(H_value, D_value, rb_value, x=0, y=0)
    alpha = soil.k / soil.rhoCp
    ts = H_value ** 2 / (9. * alpha)  # Bore field characteristic time
    time_values = np.exp(log_time) * ts
    g_ref = ghedt.gfunction.calculate_g_function(
        m_flow_borehole, bhe_object, time_values, coordinates, _borehole,
        fluid, pipe, grout, soil, nSegments=12, segments='equal',
        solver='equivalent', boundary='UBWT')
    print(g_ref.gFunc.tolist())

    H = 288.
    B_over_H = B / H
    print('# of g-Functions\tLinear\tQuadratic\tCubic\tLagrange')
    for j in range(4):
        _H_values = H_values[j:len(H_values)]
        g_function = ghedt.gfunction.compute_live_g_function(
            B, _H_values, r_b_values, D_values, m_flow_borehole, bhe_object,
            log_time, coordinates, fluid, pipe, grout, soil, nSegments=12,
            segments='equal', solver='equivalent', boundary='UBWT')
        mpe_values = []
        for i in range(len(interpolation_kinds)):
            g_func, rb_value, D_value, H_eq = \
                g_function.g_function_interpolation(
                    B_over_H, kind=interpolation_kinds[i].lower())
            g_function.interpolation_table = {}  # reset interpolation table
            mpe = ghedt.utilities.compute_mpe(g_ref.gFunc.tolist(), g_func)
            mpe_values.append(mpe)

        print('{}\t{}\t{}\t{}\t{}'.format(
            5-j, mpe_values[0], mpe_values[1], mpe_values[2],
            mpe_values[3]))


if __name__ == '__main__':
    main()
