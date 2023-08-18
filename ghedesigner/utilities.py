from json import loads
from math import sqrt
from pathlib import Path

from scipy.optimize import brentq

import numpy as np


# Time functions
# --------------
def eskilson_log_times():
    # Return a list of Eskilson's original 27 dimensionless points in time
    return [-8.5, -7.8, -7.2, -6.5, -5.9, -5.2, -4.5, -3.963, -3.27, -2.864, -2.577, -2.171, -1.884, -1.191,
            -0.497, -0.274, -0.051, 0.196, 0.419, 0.642, 0.873, 1.112, 1.335, 1.679, 2.028, 2.275, 3.003]


# Spatial functions
# -----------------
def borehole_spacing(borehole, coordinates):
    # Use the distance between the first pair of coordinates as the B-spacing
    x_0, y_0 = coordinates[0]
    if len(coordinates) == 1:
        # Set the spacing to be the borehole radius if there's just one borehole
        return borehole.r_b
    elif len(coordinates) > 1:
        x_1, y_1 = coordinates[1]
        return max(borehole.r_b, sqrt((x_1 - x_0) ** 2 + (y_1 - y_0) ** 2))
    else:
        raise ValueError("The coordinates_domain needs to contain a positive number of (x, y) pairs.")


def length_of_side(n, b):
    return (n - 1) * b


# Design oriented functions
# -------------------------
def sign(x: float) -> int:
    """
    Determine the sign of a value, pronounced "sig-na"
    :param x: the input value
    :type x: float
    :return: a 1 or a -1
    """
    return int(abs(x) / x)


def check_bracket(sign_x_l, sign_x_r) -> bool:
    if sign_x_l < 0 < sign_x_r:
        # Bracketed the root
        return True
    elif sign_x_r < 0 < sign_x_l:
        # Bracketed the root
        return True
    else:
        # The root has not been bracketed, this method will return false.
        return False


def solve_root(x, objective_function, lower=None, upper=None, abs_tol=1.0e-6, rel_tol=1.0e-6, max_iter=50):
    # Vary flow rate to match the convective resistance

    # Use Brent Quadratic to find the root
    # Define a lower and upper for thermal conductivities
    if lower is None:
        lower = x / 100.0
    else:
        lower = lower
    if upper is None:
        upper = x * 10.0
    else:
        upper = upper
    # Check objective function upper and lower bounds to make sure the root is
    # bracketed
    minus = objective_function(lower)
    plus = objective_function(upper)
    # get signs of upper and lower thermal conductivity bounds
    kg_minus_sign = int(minus / abs(minus))
    kg_plus_sign = int(plus / abs(plus))

    # Solve the root if we can, if not, take the higher value
    if kg_plus_sign != kg_minus_sign:
        x = brentq(objective_function, lower, upper, xtol=abs_tol, rtol=rel_tol, maxiter=max_iter)
    elif kg_plus_sign == -1 and kg_minus_sign == -1:
        x = lower
    elif kg_plus_sign == 1 and kg_minus_sign == 1:
        x = upper

    return x


def write_idf_object(data: list):
    s = ''
    num_leading_pad_spaces = 4
    leading_pad = ' ' * num_leading_pad_spaces
    num_fields = len(data)

    for idx, (val, comment) in enumerate(data):
        len_val = len(val)
        comment_col_no = 30
        num_mid_pad = comment_col_no - num_leading_pad_spaces - len_val

        if num_mid_pad < 2:
            num_mid_pad = 2

        mid_pad = ' ' * num_mid_pad

        if idx == 0:
            # writing object header
            s += f'{val},\n'
        elif (idx + 1) == num_fields:
            # writing last field. conclude with semicolon
            s += f'{leading_pad}{val};{mid_pad}!- {comment}\n'
        else:
            s += f'{leading_pad}{val},{mid_pad}!- {comment}\n'

    return s


def write_idf(summary_path: Path):

    data = loads(summary_path.read_text())

    # assuming the g-function file lives next to the summary file path...
    root_dir = summary_path.parent
    g_function_path = root_dir / 'Gfunction.csv'
    g_function_arr = np.genfromtxt(g_function_path, delimiter=',')

    nbh = data['ghe_system']['number_of_boreholes']
    fluid_density = data['ghe_system']['fluid_density']['value']
    fluid_mdot = data['ghe_system']['fluid_mass_flow_rate_per_borehole']['value']
    des_vdot = fluid_mdot * nbh / fluid_density

    soil_k = data['ghe_system']['soil_thermal_conductivity']['value']
    soil_rho_cp = data['ghe_system']['soil_volumetric_heat_capacity']['value'] * 1000

    system = [
        ('GroundHeatExchanger:System', ''),
        ('GHE System Name', 'Name'),
        ('Inlet Node Name', 'Inlet Node Name'),
        ('Outlet Node Name', 'Outlet Node Name'),
        (f'{des_vdot:0.3e}', 'Design Flow Rate {m3/s}'),
        ('Ground Temp Obj Type', 'Undisturbed Ground Temperature Model Type'),
        ('Ground Temp Obj Name', 'Undisturbed Ground Temperature Model Name'),
        (f'{soil_k:0.3f}', 'Ground Thermal Conductivity {W/m-K}'),
        (f'{soil_rho_cp:0.3e}', 'Ground Thermal Heat Capacity {J/m3-K}'),
        ('g-functions Obj Name', 'GHE:Vertical:ResponseFactors Object Name')
    ]

    bh_depth = data['ghe_system']['borehole_buried_depth']['value']
    bh_length = data['ghe_system']['active_borehole_length']['value']
    bh_dia = data['ghe_system']['borehole_diameter']['value']
    grout_k = data['ghe_system']['grout_thermal_conductivity']['value']
    grout_rho_cp = data['ghe_system']['grout_volumetric_heat_capacity']['value'] * 1000
    pipe_k = data['ghe_system']['pipe_thermal_conductivity']['value']
    pipe_rho_cp = data['ghe_system']['pipe_volumetric_heat_capacity']['value'] * 1000
    pipe_outer_dia = data['ghe_system']['pipe_geometry']['pipe_outer_diameter']['value']
    pipe_inner_dia = data['ghe_system']['pipe_geometry']['pipe_inner_diameter']['value']
    pipe_thickness = (pipe_outer_dia - pipe_inner_dia) / 2.0
    shank_space = data['ghe_system']['shank_spacing']['value'] + pipe_outer_dia

    prpoerties = [
        ('GroundHeatExchanger:Vertical:Properties', ''),
        ('Vert Props Name', 'Name'),
        (f'{bh_depth:0.2f}', 'Depth of Top of Borehole {m}'),
        (f'{bh_length:0.2f}', 'Borehole Length {m}'),
        (f'{bh_dia:0.4f}', 'Borehole Diameter {m}'),
        (f'{grout_k:0.2f}', 'Grout Thermal Conductivity {W/m-K}'),
        (f'{grout_rho_cp:0.3e}', 'Grout Thermal Heat Capacity {J/m3-K}'),
        (f'{pipe_k:0.2f}', 'Pipe Thermal Conductivity {W/m-K}'),
        (f'{pipe_rho_cp:0.3e}', 'Pipe Thermal Heat Capacity {J/m3-K}'),
        (f'{pipe_outer_dia:0.3e}', 'Pipe Outer Diameter {m}'),
        (f'{pipe_thickness:0.3e}', 'Pipe Thickness {m}'),
        (f'{shank_space:0.3e}', 'U-Tube Distance {m}'),
    ]

    soil_density = 2500
    soil_cp = soil_rho_cp / soil_density
    ugt = data['ghe_system']['soil_undisturbed_ground_temp']['value']

    ground_temps = [
        ('Site:GroundTemperature:Undisturbed:KusudaAchenbach', ''),
        ('GTM Name', 'Name'),
        (f'{soil_k:0.2f}', 'Soil Thermal Conductivity {W/m-K}'),
        (f'{soil_density:0.2f}', 'Soil Density {kg/m3}'),
        (f'{soil_cp:0.2f}', 'Soil Specific Heat {J/kg-K}'),
        (f'{ugt}', 'Average Soil Surface Temperature {C}'),
        ('0', 'Average Amplitude of Surface Temperature {deltaC}'),
        ('0', 'Phase Shift of Minimum Surface Temperature {days}')
    ]

    ref_ratio = (bh_dia / 2.0) / bh_length
    lntts_vals = g_function_arr[1:, 0]
    g_vals = g_function_arr[1:, 1]

    resp_factors = [
        ('GroundHeatExchanger:ResponseFactors', ''),
        ('Response Factors Name', 'Name'),
        ('Vert Props Name', 'GHE:Vertical:Properties Object Name'),
        (f'{int(nbh)}', 'Number of Boreholes'),
        (f'{ref_ratio:0.2e}', 'G-Function Reference Ratio {dimensionless}'),
    ]

    for idx in range(len(g_vals)):
        resp_factors.append((f'{lntts_vals[idx]:0.3f}', f'g-Function Ln(T/Ts) Value {idx+1}'))
        resp_factors.append((f'{g_vals[idx]:0.3f}', f'g-Function g Value {idx+1}'))

    s = ''
    s += write_idf_object(system)
    s += '\n'
    s += write_idf_object(prpoerties)
    s += '\n'
    s += write_idf_object(ground_temps)
    s += '\n'
    s += write_idf_object(resp_factors)

    idf_path = root_dir / 'out.idf'
    idf_path.write_text(s)
