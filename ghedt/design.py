# Jack C. Cook
# Friday, December 10, 2021

import ghedt as dt
import ghedt.PLAT as PLAT
import ghedt.PLAT.pygfunction as gt


# Common design interface
class Design:
    def __init__(self, V_flow: float, borehole: gt.boreholes.Borehole,
                 bhe_object: PLAT.borehole_heat_exchangers,
                 fluid: gt.media.Fluid, pipe: PLAT.media.Pipe,
                 grout: PLAT.media.ThermalProperty, soil: PLAT.media.Soil,
                 sim_params: PLAT.media.SimulationParameters,
                 geometric_constraints: dt.media.GeometricConstraints,
                 coordinates_domain: list, hourly_extraction_ground_loads: list,
                 routine: str = 'near-square', flow: str = 'borehole'):
        self.V_flow = V_flow  # volumetric flow rate, m3/s
        self.borehole = borehole
        self.bhe_object = bhe_object  # a borehole heat exchanger object
        self.fluid = fluid  # a fluid object
        self.pipe = pipe
        self.grout = grout
        self.soil = soil
        self.sim_params = sim_params
        self.geometric_constraints = geometric_constraints
        self.coordinates_domain = coordinates_domain
        self.hourly_extraction_ground_loads = hourly_extraction_ground_loads

        # Check the routine parameter
        self.routine = routine
        available_routines = ['near-square']
        if routine in available_routines:
            # If a near-square design routine is requested, then we go from a
            # 1x1 to 32x32 at the B-spacing
            if routine == 'near-square':
                self.coordinates_domain = \
                    dt.domains.square_and_near_square(
                        1, 32, self.geometric_constraints.B_max_x)
        else:
            raise ValueError('The requested routine is not available. '
                             'The currently available routines are: '
                             '`near-square`.')

        # Check the flow rate parameter
        if flow == 'borehole':
            self.V_flow_borehole = V_flow  # borehole volumetric flow rate, m3/s
        elif flow == 'system':
            self.V_flow_system = V_flow  # system volumetric flow rate, m3/s
        else:
            raise ValueError('The flow rate should be on a `borehole` or '
                             '`system` basis.')

    def find_design(self, disp=False):
        # Find near-square
        if self.routine == 'near-square':
            bisection_search = dt.search_routines.Bisection1D(
                self.coordinates_domain, self.V_flow_borehole, self.borehole,
                self.bhe_object, self.fluid, self.pipe, self.grout,
                self.soil, self.sim_params, self.hourly_extraction_ground_loads,
                disp=disp)
        else:
            raise ValueError('The requested routine is not available. '
                             'The currently available routines are: '
                             '`near-square`.')

        return bisection_search

    def create_input_file(self, file_name='ghedt_input'):
        # Track the version to keep up to date with compatibility. Only change
        # this version number when the previous one is incompatible.
        version = 0.1
        import pickle

        file_handler = open(file_name + '.obj', 'wb')
        pickle.dump(self, file_handler)
        file_handler.close()

        return


def read_input_file(path_to_file):
    import pickle
    file = open(path_to_file, 'rb')
    object_file = pickle.load(file)
    file.close()

    return object_file


def oak_ridge_export(bisection_search, file_name='ghedt_output'):
    # Dictionary for export
    d = {}
    d['number_of_boreholes'] = len(bisection_search.selected_coordinates)
    d['g_function_pairs'] = []

    ghe = bisection_search.ghe
    H = ghe.bhe.b.H
    B_over_H = ghe.B_spacing / H
    g = ghe.grab_g_function(B_over_H)

    lntts = []
    g_values = []
    for i in range(len(g.y)):
        lntts.append(g.x[i].tolist())
        g_values.append(g.y[i].tolist())

    for i in range(len(lntts)):
        d['g_function_pairs'].append({'ln_tts': lntts[i],
                                      'g_value': g_values[i]})

    dt.utilities.js_dump(file_name, d, indent=4)

    return
