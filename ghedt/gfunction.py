# Jack C. Cook
# Monday, October 25, 2021
import copy

import ghedt
import ghedt.PLAT.pygfunction as gt


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
