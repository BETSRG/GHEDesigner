# Jack C. Cook
# Wednesday, October 27, 2021

import ghedt as dt
import pygfunction as gt
import ghedt.peak_load_analysis_tool as plat
from ghedt.utilities import sign, check_bracket
import numpy as np
import copy
from ghedt.RowWise.RowWiseGeneration import fieldOptimizationWPSpac_FR

class Bisection1D:
    def __init__(self, coordinates_domain: list,fieldDescriptors: list, V_flow: float,
                 borehole: gt.boreholes.Borehole,
                 bhe_object: plat.borehole_heat_exchangers,
                 fluid: gt.media.Fluid, pipe: plat.media.Pipe,
                 grout: plat.media.Grout, soil: plat.media.Soil,
                 sim_params: plat.media.SimulationParameters,
                 hourly_extraction_ground_loads: list, method: str = 'hybrid',
                 flow: str = 'borehole', max_iter=15, disp=False, search=True,fieldType="N/A"):

        # Take the lowest part of the coordinates domain to be used for the
        # initial setup
        self.searchTracker = []
        coordinates = coordinates_domain[0]
        currentField = fieldDescriptors[0]
        self.fieldType = fieldType
        # Flow rate tracking
        self.V_flow = V_flow
        self.flow = flow
        V_flow_system, m_flow_borehole = \
            self.retrieve_flow(coordinates, fluid.rho)
        self.method = method

        self.log_time = dt.utilities.Eskilson_log_times()
        self.bhe_object = bhe_object
        self.sim_params = sim_params
        self.hourly_extraction_ground_loads = hourly_extraction_ground_loads
        self.coordinates_domain = coordinates_domain
        self.fieldDescriptors = fieldDescriptors
        self.max_iter = max_iter
        self.disp = disp

        B = dt.utilities.borehole_spacing(borehole, coordinates)

        # Calculate a g-function for uniform inlet fluid temperature with
        # 8 unequal segments using the equivalent solver
        g_function = dt.gfunction.compute_live_g_function(
            B, [borehole.H], [borehole.r_b], [borehole.D], m_flow_borehole,
            self.bhe_object, self.log_time, coordinates, fluid, pipe, grout,
            soil)

        # Initialize the GHE object
        self.ghe = dt.ground_heat_exchangers.GHE(
            V_flow_system, B, bhe_object, fluid, borehole, pipe, grout,
            soil, g_function, sim_params, hourly_extraction_ground_loads,fieldSpecifier=currentField,fieldType=fieldType)

        self.calculated_temperatures = {}

        if search:
            self.selection_key, self.selected_coordinates = self.search()

    def retrieve_flow(self, coordinates, rho):
        if self.flow == 'borehole':
            V_flow_system = self.V_flow * float(len(coordinates))
            # Total fluid mass flow rate per borehole (kg/s)
            m_flow_borehole = self.V_flow / 1000. * rho
        elif self.flow == 'system':
            V_flow_system = self.V_flow
            V_flow_borehole = self.V_flow / float(len(coordinates))
            m_flow_borehole = V_flow_borehole / 1000. * rho
        else:
            raise ValueError('The flow argument should be either `borehole`'
                             'or `system`.')
        return V_flow_system, m_flow_borehole

    def initialize_ghe(self, coordinates, H,fieldSpecifier="N/A"):
        V_flow_system, m_flow_borehole = \
            self.retrieve_flow(coordinates, self.ghe.bhe.fluid.rho)

        self.ghe.bhe.b.H = H
        borehole = self.ghe.bhe.b
        fluid = self.ghe.bhe.fluid
        pipe = self.ghe.bhe.pipe
        grout = self.ghe.bhe.grout
        soil = self.ghe.bhe.soil

        B = dt.utilities.borehole_spacing(borehole, coordinates)

        # Calculate a g-function for uniform inlet fluid temperature with
        # 8 unequal segments using the equivalent solver
        g_function = dt.gfunction.compute_live_g_function(
            B, [borehole.H], [borehole.r_b], [borehole.D], m_flow_borehole,
            self.bhe_object, self.log_time, coordinates, fluid, pipe, grout,
            soil)

        # Initialize the GHE object
        self.ghe = dt.ground_heat_exchangers.GHE(
            V_flow_system, B, self.bhe_object, fluid, borehole, pipe, grout,
            soil, g_function, self.sim_params,
            self.hourly_extraction_ground_loads,fieldType=self.fieldType,fieldSpecifier=fieldSpecifier)

    def calculate_excess(self, coordinates, H,fieldSpecifier="N/A"):
        self.initialize_ghe(coordinates, H,fieldSpecifier=fieldSpecifier)
        # Simulate after computing just one g-function
        max_HP_EFT, min_HP_EFT = self.ghe.simulate(method=self.method)
        T_excess = self.ghe.cost(max_HP_EFT, min_HP_EFT)
        self.searchTracker.append([fieldSpecifier, '{.2f}'.format(T_excess),'{.2f}'.format(max_HP_EFT)
                                      ,'{.2f}'.format(min_HP_EFT)])

        # This is more of a debugging statement. May remove it in the future.
        # Perhaps there becomes a debug: bool option in the API.
        # if self.disp:
        #     print('Min EFT: {}\nMax EFT: {}'.format(min_HP_EFT, max_HP_EFT))

        return T_excess

    def search(self):

        xL_idx = 0
        xR_idx = len(self.coordinates_domain) - 1
        if self.disp:
            print('Do some initial checks before searching.')
        # Get the lowest possible excess temperature from minimum height at the
        # smallest location in the domain
        T_0_lower = self.calculate_excess(self.coordinates_domain[xL_idx],
                                          self.sim_params.min_Height,fieldSpecifier=self.fieldDescriptors[xL_idx])
        T_0_upper = self.calculate_excess(self.coordinates_domain[xL_idx],
                                          self.sim_params.max_Height,fieldSpecifier=self.fieldDescriptors[xL_idx])
        T_m1 = \
            self.calculate_excess(
                self.coordinates_domain[xR_idx],
                self.sim_params.max_Height,fieldSpecifier=self.fieldDescriptors[xR_idx])

        self.calculated_temperatures[xL_idx] = T_0_upper
        self.calculated_temperatures[xR_idx] = T_m1

        if check_bracket(sign(T_0_lower), sign(T_0_upper)):
            if self.disp:
                print('Size between min and max of lower bound in domain.')
            self.initialize_ghe(self.coordinates_domain[0],
                                self.sim_params.max_Height)
            return 0, self.coordinates_domain[0]
        elif check_bracket(sign(T_0_upper), sign(T_m1)):
            if self.disp:
                print('Perform the integer bisection search routine.')
            pass
        else:
            # This domain does not bracked the solution
            if T_0_upper < 0.0 and T_m1 < 0.0:
                msg = 'Based on the loads provided, the excess temperatures ' \
                      'for the minimum and maximum number of boreholes falls ' \
                      'below 0. This means that the loads are "miniscule" or ' \
                      'that the lower end of the domain needs to contain ' \
                      'less boreholes.'.format()
                raise ValueError(msg)
            if T_0_upper > 0.0 and T_m1 > 0.0:
                msg = 'Based on the loads provided, the excess temperatures ' \
                      'for the minimum and maximum number of boreholes falls ' \
                      'above 0. This means that the loads are "astronomical" ' \
                      'or that the higher end of the domain needs to contain ' \
                      'more boreholes. Consider increasing the available land' \
                      ' area, or decreasing the minimum allowable borehole ' \
                      'spacing.'
                raise ValueError(msg)
            return None, None

        if self.disp:
            print('Beginning bisection search...')

        xL_sign = sign(T_0_upper)
        xR_sign = sign(T_m1)
        
        i = 0

        while i < self.max_iter:
            c_idx = int(np.ceil((xL_idx + xR_idx) / 2))
            # if the solution is no longer making progress break the while
            if c_idx == xL_idx or c_idx == xR_idx:
                break

            c_T_excess = self.calculate_excess(self.coordinates_domain[c_idx],
                                               self.sim_params.max_Height,fieldSpecifier=self.fieldDescriptors[c_idx])

            self.calculated_temperatures[c_idx] = c_T_excess
            c_sign = sign(c_T_excess)

            if c_sign == xL_sign:
                xL_idx = copy.deepcopy(c_idx)
            else:
                xR_idx = copy.deepcopy(c_idx)

            i += 1

        coordinates = self.coordinates_domain[i]

        H = self.sim_params.max_Height

        self.calculate_excess(coordinates, H,fieldSpecifier=self.fieldDescriptors[i])
        # Make sure the field being returned pertains to the index which is the
        # closest to 0 but also negative (the maximum of all 0 or negative
        # excess temperatures)
        keys = list(self.calculated_temperatures.keys())
        values = list(self.calculated_temperatures.values())

        negative_excess_values = [values[i] for i in range(len(values))
                                  if values[i] <= 0.0]

        excess_of_interest = max(negative_excess_values)
        idx = values.index(excess_of_interest)
        selection_key = keys[idx]
        selected_coordinates = self.coordinates_domain[selection_key]

        self.initialize_ghe(selected_coordinates, H,fieldSpecifier=self.fieldDescriptors[selection_key])

        return selection_key, selected_coordinates

#This is the search algorithm used for finding RowWise fields
class RowWiseModifiedBisectionSearch:
    def __init__(self, V_flow: float,
                 borehole: gt.boreholes.Borehole,
                 bhe_object: plat.borehole_heat_exchangers,
                 fluid: gt.media.Fluid, pipe: plat.media.Pipe,
                 grout: plat.media.Grout, soil: plat.media.Soil,
                 sim_params: plat.media.SimulationParameters,
                 hourly_extraction_ground_loads: list,geometricConstraints, method: str = 'hybrid',
                 flow: str = 'borehole', max_iter=10, disp=False, search=True,advanced_tracking=False, fieldType="N/A"):

        # Take the lowest part of the coordinates domain to be used for the
        # initial setup
        self.excess = None
        self.fluid = fluid
        self.pipe = pipe
        self.grout = grout
        self.soil = soil
        self.borehole = borehole
        self.geometricConstraints = geometricConstraints
        self.searchTracker = []
        self.RowWiseTracking = []
        self.fieldType = fieldType
        # Flow rate tracking
        self.V_flow = V_flow
        self.flow = flow
        self.method = method
        self.log_time = dt.utilities.Eskilson_log_times()
        self.bhe_object = bhe_object
        self.sim_params = sim_params
        self.hourly_extraction_ground_loads = hourly_extraction_ground_loads
        self.max_iter = max_iter
        self.disp = disp
        self.ghe = None
        self.calculated_temperatures = {}
        self.advanced_tracking = advanced_tracking
        if advanced_tracking:
            self.excessTemperatures = []
            self.checkedFields = []
        if search:
            self.selected_coordinates,self.selected_specifier = self.search()
            self.initialize_ghe(self.selected_coordinates,self.sim_params.max_Height,fieldSpecifier=self.selected_specifier)

    def retrieve_flow(self, coordinates, rho):
        if self.flow == 'borehole':
            V_flow_system = self.V_flow * float(len(coordinates))
            # Total fluid mass flow rate per borehole (kg/s)
            m_flow_borehole = self.V_flow / 1000. * rho
        elif self.flow == 'system':
            V_flow_system = self.V_flow
            V_flow_borehole = self.V_flow / float(len(coordinates))
            m_flow_borehole = V_flow_borehole / 1000. * rho
        else:
            raise ValueError('The flow argument should be either `borehole`'
                             'or `system`.')
        return V_flow_system, m_flow_borehole

    def initialize_ghe(self, coordinates, H, fieldSpecifier="N/A"):
        V_flow_system, m_flow_borehole = \
            self.retrieve_flow(coordinates, self.fluid.rho)

        self.borehole.H = h
        borehole = self.borehole
        fluid = self.fluid
        pipe = self.pipe
        grout = self.grout
        soil = self.soil

        B = dt.utilities.borehole_spacing(borehole, coordinates)

        # Calculate a g-function for uniform inlet fluid temperature with
        # 8 unequal segments using the equivalent solver
        g_function = dt.gfunction.compute_live_g_function(
            B, [borehole.H], [borehole.r_b], [borehole.D], m_flow_borehole,
            self.bhe_object, self.log_time, coordinates, fluid, pipe, grout,
            soil)

        # Initialize the GHE object
        self.ghe = dt.ground_heat_exchangers.GHE(
            V_flow_system, B, self.bhe_object, fluid, borehole, pipe, grout,
            soil, g_function, self.sim_params,
            self.hourly_extraction_ground_loads, fieldType=self.fieldType, fieldSpecifier=fieldSpecifier)

    def calculate_excess(self, coordinates, H, fieldSpecifier="N/A"):
        self.initialize_ghe(coordinates, H, fieldSpecifier=fieldSpecifier)
        # Simulate after computing just one g-function
        max_HP_EFT, min_HP_EFT = self.ghe.simulate(method=self.method)
        T_excess = self.ghe.cost(max_HP_EFT, min_HP_EFT)
        self.searchTracker.append([fieldSpecifier, T_excess, max_HP_EFT, min_HP_EFT])

        # This is more of a debugging statement. May remove it in the future.
        # Perhaps there becomes a debug: bool option in the API.
        # if self.disp:
        #     print('Min EFT: {}\nMax EFT: {}'.format(min_HP_EFT, max_HP_EFT))

        return T_excess

    def search(self,eT = 1e-10):
        #Copy all of the geometric constraints to local variables
        spacStart = self.geometricConstraints.spacStart
        spacStop = self.geometricConstraints.spacStop
        spacStep = self.geometricConstraints.spacStep
        rotateStep = self.geometricConstraints.rotateStep
        propBound = self.geometricConstraints.propBound
        ngZones = self.geometricConstraints.ngZones
        rotateStart = self.geometricConstraints.rotateStart
        rotateStop = self.geometricConstraints.rotateStop
        pSpac = self.geometricConstraints.pSpac

        selected_coordinates = None
        selected_specifier = None
        selected_temp_excess = None

        #Check The Upper and Lower Bounds

        #Generate Fields
        upperField,upperFieldSpecifier = fieldOptimizationWPSpac_FR([pSpac],spacStart,rotateStep,propBound,ngZones=ngZones
                                                ,rotateStart=rotateStart,rotateStop=rotateStop)
        lowerField,lowerFieldSpecifier = fieldOptimizationWPSpac_FR([pSpac], spacStop, rotateStep, propBound, ngZones=ngZones,
                                                rotateStart=rotateStart, rotateStop=rotateStop)
        #Get Excess Temperatures
        T_upper = self.calculate_excess(upperField,self.sim_params.max_Height,fieldSpecifier=upperFieldSpecifier)
        T_lower = self.calculate_excess(lowerField,self.sim_params.max_Height,fieldSpecifier=lowerFieldSpecifier)

        if self.advanced_tracking:
            self.excessTemperatures.append(T_upper)
            self.excessTemperatures.append(T_lower)
            self.checkedFields.append(upperField)
            self.checkedFields.append(lowerField)

        #If the excess temperature is >0 utilizing the largest field and largest depth, then notify the user that
        #the given contrants cannot find a satisfactory field.
        if T_upper > 0.0 and T_lower > 0.0:
            msg = 'Based on the loads provided, the excess temperatures for the minimum and maximum number of boreholes' \
                  'fall above 0. This means that the loads are too large for the corresponding simulation parameters.' \
                  'Please double check the loadings or adjust those parameters.'
            raise ValueError(msg)
        #If the excess temperature is > 0 when utilizing the largest field and depth but < 0 when using the largest
        # depth and smallest field, then fields should be searched between the two target depths.
        elif T_upper < 0.0 and T_lower > 0.0:
            #This search currently works by doing a slightly modified bisection search where the "steps" are the set by the
            #"spacStep" variable. The steps are used to check fields on either side of the field found by doing a normal
            #bisection search. These extra fields are meant to help prevent falling into local minima (although this
            # will still happen sometimes).
            i = 0
            spacHigh= spacStart
            spacLow = spacStop
            lowE = T_upper
            highE = T_lower
            spacM = (spacStop+spacStart)*0.5
            selected_coordinates=upperField
            while (i < self.max_iter):
                print("Bisection Search Iteration: ",i)
                #Getting Three Middle Field
                f1,f1Specifier = fieldOptimizationWPSpac_FR([pSpac],spacM,rotateStep,propBound,ngZones=ngZones
                                                ,rotateStart=rotateStart,rotateStop=rotateStop)

                #Getting the three field's excess temperature
                T_e1 = self.calculate_excess(f1,self.sim_params.max_Height,fieldSpecifier=f1Specifier)

                if self.advanced_tracking:
                    self.excessTemperatures.append(T_e1)
                    self.checkedFields.append(f1)
                if T_e1 <= 0.0:
                    spacHigh = spacM
                    highE = T_e1
                    selected_coordinates = f1
                    selected_specifier = f1Specifier
                    selected_temp_excess = T_e1
                else:
                    spacLow = spacM
                    lowE = T_e1

                spacM = (spacLow+spacHigh)*0.5
                if abs(lowE-highE) < eT:
                    break

                i += 1

            #Now Check fields to the left and right:
            fR,fRS = fieldOptimizationWPSpac_FR([pSpac],spacM-spacStep,rotateStep,propBound,ngZones=ngZones
                                            ,rotateStart=rotateStart,rotateStop=rotateStop)
            fL,fLS = fieldOptimizationWPSpac_FR([pSpac],spacM+spacStep,rotateStep,propBound,ngZones=ngZones
                                            ,rotateStart=rotateStart,rotateStop=rotateStop)
            TR = self.calculate_excess(fR,self.sim_params.max_Height,fieldSpecifier=fRS)
            TL = self.calculate_excess(fL,self.sim_params.max_Height,fieldSpecifier=fLS)
            fieldsToCheck = [selected_coordinates]
            fSTC = [selected_specifier]
            excessTemps = [selected_temp_excess]
            if TR < 0.0:
                fieldsToCheck.append(fR)
                fSTC.append(fRS)
                excessTemps.append(TR)
                if self.advanced_tracking:
                    self.checkedFields.append(fR)
                    self.excessTemperatures.append(TR)
            if TL < 0.0:
                fieldsToCheck.append(fL)
                fSTC.append(fLS)
                excessTemps.append(TL)
                if self.advanced_tracking:
                    self.checkedFields.append(fL)
                    self.excessTemperatures.append(TL)

            #Now look at the three fields and choose the one that has the least drilling with a satisfactory
            #excess temperature.
            bestField = None
            bestDrilling = float('inf')
            bestExcess = None
            for i in range(len(fieldsToCheck)):
                field = fieldsToCheck[i]
                fS = fSTC[i]
                self.initialize_ghe(field,self.sim_params.max_Height,fieldSpecifier=fS)
                self.ghe.compute_g_functions()
                self.ghe.size()
                H = self.ghe.averageHeight()
                totalDrilling = H*len(field)
                if bestField is None:
                    bestField = field
                    bestDrilling = totalDrilling
                    bestExcess=excessTemps[i]
                else:
                    if totalDrilling<bestDrilling:
                        bestDrilling = totalDrilling
                        bestField = field
                        bestExcess = excessTemps[i]
            selected_coordinates = bestField
            selected_temp_excess = bestExcess

        #If the excess temperature is < 0 when utilizing the largest depth and the smallest field, it is most likely
        # in the user's best interest to return a field smaller than the smallest one. This is done by removing
        #boreholes from the field.
        elif T_lower < 0.0 and T_upper < 0.0:
            #The most efficient field is most likely the one utilizing the maximum target depth. As such, boreholes will
            #be removed until the height used is as close to the maximum as possible.
            currentField = lowerField
            lastField = lowerField
            lastExcess = 0
            continueLoop = True
            selected_specifier = lowerFieldSpecifier
            while continueLoop: #Loop will terminate from if statements inside of loop.
                currentField = self.RemoveRow(currentField)
                T_excess = self.calculate_excess(currentField,self.sim_params.max_Height,fieldSpecifier=lowerFieldSpecifier)
                if self.advanced_tracking:
                    self.excessTemperatures.append(T_excess)
                    self.checkedFields.append(currentField)
                #If the excess temperature for this new field is > 0.0, that means that the previous field is the best
                #option available.
                if T_excess > 0.0:
                    selected_coordinates = lastField
                    selected_temp_excess = lastExcess
                    continueLoop = False
                #If there is no more progress being made, check the excess temperature utilizing the minimum height
                #and either return the current field or notify the user that there is an issue with the smallest field.
                elif len(currentField) == len(lastField):
                    T_excess_minimum = self.calculate_excess(currentField,
                                                        self.sim_params.min_Height,fieldSpecifier=lowerFieldSpecifier)
                    if T_excess_minimum > 0.0:
                        msg = 'Based on the loads provided, the excess temperatures for the minimum and maximum number of boreholes' \
                        'fall below 0. This means that the loads are too small for the corresponding simulation parameters.' \
                        'Please double check the loadings or adjust those parameters.'
                        continueLoop = False
                    else:
                        selected_coordinates=currentField
                        selected_temp_excess = T_excess
                        continueLoop = False
                lastField = currentField
                lastExcess = T_excess
        #If none of the options above have been true, then there is most likely an issue with the excess temperature
        #calculation.
        else:
            msg = 'There seems to be an issue calculating excess temperatures. Check that you have the correct' \
                  'package version. If this is a recurring issue, please contact the current package management for ' \
                  'assistance.'
            raise ValueError(msg)
        self.excess = selected_temp_excess
        return selected_coordinates,selected_specifier

#This is a somewhat depreciated function for RowWise searching. It could still be used, but does not seem optimal based
#on the current testing as of 5/30.
class Bisection1D_modified:
    def __init__(self, coordinates_domain: list, fieldDescriptors: list, V_flow: float,
                 borehole: gt.boreholes.Borehole,
                 bhe_object: plat.borehole_heat_exchangers,
                 fluid: gt.media.Fluid, pipe: plat.media.Pipe,
                 grout: plat.media.Grout, soil: plat.media.Soil,
                 sim_params: plat.media.SimulationParameters,
                 hourly_extraction_ground_loads: list, method: str = 'hybrid',
                 flow: str = 'borehole', max_iter=15, disp=False, search=True, fieldType="N/A"):

        # Take the lowest part of the coordinates domain to be used for the
        # initial setup
        self.searchTracker = []
        coordinates = coordinates_domain[0]
        currentField = fieldDescriptors[0]
        self.fieldType = fieldType
        # Flow rate tracking
        self.V_flow = V_flow
        self.flow = flow
        V_flow_system, m_flow_borehole = \
            self.retrieve_flow(coordinates, fluid.rho)
        self.method = method

        self.log_time = dt.utilities.Eskilson_log_times()
        self.bhe_object = bhe_object
        self.sim_params = sim_params
        self.hourly_extraction_ground_loads = hourly_extraction_ground_loads
        self.coordinates_domain = coordinates_domain
        self.fieldDescriptors = fieldDescriptors
        self.max_iter = max_iter
        self.disp = disp

        B = dt.utilities.borehole_spacing(borehole, coordinates)

        # Calculate a g-function for uniform inlet fluid temperature with
        # 8 unequal segments using the equivalent solver
        g_function = dt.gfunction.compute_live_g_function(
            B, [borehole.H], [borehole.r_b], [borehole.D], m_flow_borehole,
            self.bhe_object, self.log_time, coordinates, fluid, pipe, grout,
            soil)

        # Initialize the GHE object
        self.ghe = dt.ground_heat_exchangers.GHE(
            V_flow_system, B, bhe_object, fluid, borehole, pipe, grout,
            soil, g_function, sim_params, hourly_extraction_ground_loads)

        self.calculated_temperatures = {}

        if search:
            self.selection_key, self.selected_coordinates = self.search()

    def retrieve_flow(self, coordinates, rho):
        if self.flow == 'borehole':
            V_flow_system = self.V_flow * float(len(coordinates))
            # Total fluid mass flow rate per borehole (kg/s)
            m_flow_borehole = self.V_flow / 1000. * rho
        elif self.flow == 'system':
            V_flow_system = self.V_flow
            V_flow_borehole = self.V_flow / float(len(coordinates))
            m_flow_borehole = V_flow_borehole / 1000. * rho
        else:
            raise ValueError('The flow argument should be either `borehole`'
                             'or `system`.')
        return V_flow_system, m_flow_borehole

    def initialize_ghe(self, coordinates, H, fieldSpecifier="N/A"):
        V_flow_system, m_flow_borehole = \
            self.retrieve_flow(coordinates, self.ghe.bhe.fluid.rho)

        self.ghe.bhe.b.H = H
        borehole = self.ghe.bhe.b
        fluid = self.ghe.bhe.fluid
        pipe = self.ghe.bhe.pipe
        grout = self.ghe.bhe.grout
        soil = self.ghe.bhe.soil

        B = dt.utilities.borehole_spacing(borehole, coordinates)

        # Calculate a g-function for uniform inlet fluid temperature with
        # 8 unequal segments using the equivalent solver
        g_function = dt.gfunction.compute_live_g_function(
            B, [borehole.H], [borehole.r_b], [borehole.D], m_flow_borehole,
            self.bhe_object, self.log_time, coordinates, fluid, pipe, grout,
            soil)

        # Initialize the GHE object
        self.ghe = dt.ground_heat_exchangers.GHE(
            V_flow_system, B, self.bhe_object, fluid, borehole, pipe, grout,
            soil, g_function, self.sim_params,
            self.hourly_extraction_ground_loads, fieldType=self.fieldType, fieldSpecifier=fieldSpecifier)

    def calculate_excess(self, coordinates, H, fieldSpecifier="N/A"):
        self.initialize_ghe(coordinates, H, fieldSpecifier=fieldSpecifier)
        # Simulate after computing just one g-function
        max_HP_EFT, min_HP_EFT = self.ghe.simulate(method=self.method)
        T_excess = self.ghe.cost(max_HP_EFT, min_HP_EFT)
        self.searchTracker.append([fieldSpecifier, T_excess, max_HP_EFT, min_HP_EFT])

        # This is more of a debugging statement. May remove it in the future.
        # Perhaps there becomes a debug: bool option in the API.
        # if self.disp:
        #     print('Min EFT: {}\nMax EFT: {}'.format(min_HP_EFT, max_HP_EFT))

        return T_excess

    def search(self):

        xL_idx = 0
        xR_idx = len(self.coordinates_domain) - 1
        if self.disp:
            print('Do some initial checks before searching.')
        # Get the lowest possible excess temperature from minimum height at the
        # smallest location in the domain
        T_0_lower = self.calculate_excess(self.coordinates_domain[xL_idx],
                                          self.sim_params.min_Height, fieldSpecifier=self.fieldDescriptors[xL_idx])
        T_0_upper = self.calculate_excess(self.coordinates_domain[xL_idx],
                                          self.sim_params.max_Height, fieldSpecifier=self.fieldDescriptors[xL_idx])
        T_m1 = \
            self.calculate_excess(
                self.coordinates_domain[xR_idx],
                self.sim_params.max_Height, fieldSpecifier=self.fieldDescriptors[xR_idx])

        self.calculated_temperatures[xL_idx] = T_0_upper
        self.calculated_temperatures[xR_idx] = T_m1

        if check_bracket(sign(T_0_lower), sign(T_0_upper)):
            if self.disp:
                print('Size between min and max of lower bound in domain.')
            self.initialize_ghe(self.coordinates_domain[0],
                                self.sim_params.max_Height)
            return 0, self.coordinates_domain[0]
        elif check_bracket(sign(T_0_upper), sign(T_m1)):
            if self.disp:
                print('Perform the integer bisection search routine.')
            pass
        else:
            # This domain does not bracked the solution
            if T_0_upper < 0.0 and T_m1 < 0.0:
                msg = 'Based on the loads provided, the excess temperatures ' \
                      'for the minimum and maximum number of boreholes falls ' \
                      'below 0. This means that the loads are "miniscule" or ' \
                      'that the lower end of the domain needs to contain ' \
                      'less boreholes.'.format()
                raise ValueError(msg)
            if T_0_upper > 0.0 and T_m1 > 0.0:
                msg = 'Based on the loads provided, the excess temperatures ' \
                      'for the minimum and maximum number of boreholes falls ' \
                      'above 0. This means that the loads are "astronomical" ' \
                      'or that the higher end of the domain needs to contain ' \
                      'more boreholes. Consider increasing the available land' \
                      ' area, or decreasing the minimum allowable borehole ' \
                      'spacing.'
                raise ValueError(msg)
            return None, None

        if self.disp:
            print('Beginning bisection search...')

        xL_sign = sign(T_0_upper)
        xR_sign = sign(T_m1)

        i = 0

        while i < self.max_iter:
            c_idx = int(np.ceil((xL_idx + xR_idx) / 2))
            # if the solution is no longer making progress break the while
            if c_idx == xL_idx or c_idx == xR_idx:
                break

            c_T_excess = self.calculate_excess(self.coordinates_domain[c_idx],
                                               self.sim_params.max_Height, fieldSpecifier=self.fieldDescriptors[c_idx])

            self.calculated_temperatures[c_idx] = c_T_excess
            c_sign = sign(c_T_excess)

            if c_sign == xL_sign:
                xL_idx = copy.deepcopy(c_idx)
            else:
                xR_idx = copy.deepcopy(c_idx)

            i += 1

        self.calculated_temperatures[c_idx+1] = self.calculate_excess(self.coordinates_domain[c_idx+1],
                                               self.sim_params.max_Height, fieldSpecifier=self.fieldDescriptors[c_idx+1])
        self.calculated_temperatures[c_idx - 1] = self.calculate_excess(self.coordinates_domain[c_idx - 1],
                                                                        self.sim_params.max_Height,
                                                                        fieldSpecifier=self.fieldDescriptors[c_idx - 1])

        coordinates = self.coordinates_domain[i]

        H = self.sim_params.max_Height

        self.calculate_excess(coordinates, H, fieldSpecifier=self.fieldDescriptors[i])
        # Make sure the field being returned pertains to the index which is the
        # closest to 0 but also negative (the maximum of all 0 or negative
        # excess temperatures)
        keys = list(self.calculated_temperatures.keys())
        values = list(self.calculated_temperatures.values())

        negative_excess_values = [values[i] for i in range(len(values))
                                  if values[i] <= 0.0]

        excess_of_interest = max(negative_excess_values)
        idx = values.index(excess_of_interest)
        selection_key = keys[idx]
        selected_coordinates = self.coordinates_domain[selection_key]

        self.initialize_ghe(selected_coordinates, H, fieldSpecifier=self.fieldDescriptors[selection_key])

        return selection_key, selected_coordinates

class Bisection2D(Bisection1D):
    def __init__(self, coordinates_domain_nested: list,fieldDescriptors: list, V_flow: float,
                 borehole: gt.boreholes.Borehole,
                 bhe_object: plat.borehole_heat_exchangers,
                 fluid: gt.media.Fluid, pipe: plat.media.Pipe,
                 grout: plat.media.Grout, soil: plat.media.Soil,
                 sim_params: plat.media.SimulationParameters,
                 hourly_extraction_ground_loads: list, method: str = 'hybrid',
                 flow: str = 'borehole', max_iter=15, disp=False,fieldType="N/A"):
        if disp:
            print('Note: This routine requires a nested bisection search.')

        # Get a coordinates domain for initialization
        coordinates_domain = coordinates_domain_nested[0]
       # print("Coordinate Dimensions",len(coordinates_domain),len(coordinates_domain[0]))
        Bisection1D.__init__(
            self, coordinates_domain,fieldDescriptors[0], V_flow, borehole, bhe_object,
            fluid, pipe, grout, soil, sim_params,
            hourly_extraction_ground_loads, method=method, flow=flow,
            max_iter=max_iter, disp=disp, search=False,fieldType=fieldType)

        self.coordinates_domain_nested = []
        self.calculated_temperatures_nested = []
        # Tack on one borehole at the beginning to provide a high excess
        # temperature
        outer_domain = [coordinates_domain_nested[0][0]]
        for i in range(len(coordinates_domain_nested)):
            outer_domain.append(coordinates_domain_nested[i][-1])

        self.coordinates_domain = outer_domain

        selection_key, selected_coordinates = self.search()

        self.calculated_temperatures_nested.append(
            copy.deepcopy(self.calculated_temperatures))

        # We tacked on one borehole to the beginning, so we need to subtract 1
        # on the index
        inner_domain = coordinates_domain_nested[selection_key-1]
        self.coordinates_domain = inner_domain
        self.fieldDescriptors = fieldDescriptors[selection_key-1]

        # Reset calculated temperatures
        self.calculated_temperatures = {}

        self.selection_key, self.selected_coordinates = self.search()


class BisectionZD(Bisection1D):
    def __init__(self, coordinates_domain_nested: list,fieldDescriptors: list, V_flow: float,
                 borehole: gt.boreholes.Borehole,
                 bhe_object: plat.borehole_heat_exchangers,
                 fluid: gt.media.Fluid, pipe: plat.media.Pipe,
                 grout: plat.media.Grout, soil: plat.media.Soil,
                 sim_params: plat.media.SimulationParameters,
                 hourly_extraction_ground_loads: list, method: str = 'hybrid',
                 flow: str = 'borehole', max_iter=15, disp=False,fieldType="N/A"):
        if disp:
            print('Note: This design routine currently requires several '
                  'bisection searches.')

        # Get a coordinates domain for initialization
        coordinates_domain = coordinates_domain_nested[0]
        Bisection1D.__init__(
            self, coordinates_domain,fieldDescriptors[0], V_flow, borehole, bhe_object,
            fluid, pipe, grout, soil, sim_params,
            hourly_extraction_ground_loads, method=method, flow=flow,
            max_iter=max_iter, disp=disp, search=False,fieldType=fieldType)

        self.coordinates_domain_nested = coordinates_domain_nested
        self.nested_fieldDescriptors = fieldDescriptors
        self.calculated_temperatures_nested = {}
        # Tack on one borehole at the beginning to provide a high excess
        # temperature
        outer_domain = [coordinates_domain_nested[0][0]]
        outerDescriptors = [fieldDescriptors[0][0]]
        for i in range(len(coordinates_domain_nested)):
            outer_domain.append(coordinates_domain_nested[i][-1])
            outerDescriptors.append(fieldDescriptors[i][-1])

        self.coordinates_domain = outer_domain
        self.fieldDescriptors = outerDescriptors

        self.selection_key_outer, self.selected_coordinates_outer = \
            self.search()
        if self.selection_key_outer > 0:
            self.selection_key_outer -= 1
        self.calculated_heights = {}

        self.selection_key, self.selected_coordinates = self.search_successive()

    def search_successive(self, max_iter=None):
        if max_iter is None:
            max_iter = self.selection_key_outer + 7

        i = self.selection_key_outer

        old_height = 99999

        while i < len(self.coordinates_domain_nested) and i < max_iter:

            self.coordinates_domain = self.coordinates_domain_nested[i]
            self.fieldDescriptors = self.nested_fieldDescriptors[i]
            self.calculated_temperatures = {}
            try:
                selection_key, selected_coordinates = self.search()
            except ValueError:
                break
            self.calculated_temperatures_nested[i] = \
                copy.deepcopy(self.calculated_temperatures)

            self.ghe.compute_g_functions()
            self.ghe.size(method='hybrid')

            nbh = len(selected_coordinates)
            total_drilling = float(nbh) * self.ghe.bhe.b.H
            self.calculated_heights[i] = total_drilling

            if old_height < total_drilling:
                break
            else:
                old_height = copy.deepcopy(total_drilling)

            i += 1

        keys = list(self.calculated_heights.keys())
        values = list(self.calculated_heights.values())

        minimum_total_drilling = min(values)
        idx = values.index(minimum_total_drilling)
        selection_key_outer = keys[idx]
        self.calculated_temperatures = \
            copy.deepcopy(self.calculated_temperatures_nested[
                              selection_key_outer])

        keys = list(self.calculated_temperatures.keys())
        values = list(self.calculated_temperatures.values())

        negative_excess_values = [values[i] for i in range(len(values))
                                  if values[i] <= 0.0]

        excess_of_interest = max(negative_excess_values)
        idx = values.index(excess_of_interest)
        selection_key = keys[idx]
        selected_coordinates = \
            self.coordinates_domain_nested[selection_key_outer][selection_key]

        self.initialize_ghe(selected_coordinates, self.sim_params.max_Height,fieldSpecifier=self.nested_fieldDescriptors[selection_key_outer][selection_key])
        self.ghe.compute_g_functions()
        self.ghe.size(method='hybrid')

        return selection_key, selected_coordinates


# The following functions are utility functions specific to search_routines.py
# ------------------------------------------------------------------------------
def oak_ridge_export(bisection_search, file_name='ghedt_output'):
    # Dictionary for export
    d = {}
    d['borehole_length'] = bisection_search.ghe.bhe.b.H
    d['number_of_boreholes'] = len(bisection_search.selected_coordinates)
    d['g_function_pairs'] = []
    d['single_u_tube'] = {}

    # create a local single U-tube object
    bhe_eq = bisection_search.ghe.bhe_eq
    d['single_u_tube']['r_b'] = bhe_eq.borehole.r_b  # Borehole radius
    d['single_u_tube']['r_in'] = bhe_eq.r_in  # Inner pipe radius
    d['single_u_tube']['r_out'] = bhe_eq.r_out  # Outer pipe radius
    # Note: Shank spacing or center pipe positions could be used
    d['single_u_tube']['s'] = bhe_eq.pipe.s  # Shank spacing (tube-to-tube)
    d['single_u_tube']['pos'] = bhe_eq.pos  # Center of the pipes
    d['single_u_tube']['m_flow_borehole'] = \
        bhe_eq.m_flow_borehole  # mass flow rate of the borehole
    d['single_u_tube']['k_g'] = bhe_eq.grout.k  # Grout thermal conductivity
    d['single_u_tube']['k_s'] = bhe_eq.soil.k  # Soil thermal conductivity
    d['single_u_tube']['k_p'] = bhe_eq.pipe.k  # Pipe thermal conductivity

    # create a local ghe object
    ghe = bisection_search.ghe
    H = ghe.bhe.b.H
    B_over_H = ghe.B_spacing / H
    g = ghe.grab_g_function(B_over_H)

    total_g_values = g.x.size
    number_lts_g_values = 27
    number_sts_g_values = 30
    sts_step_size = int(np.floor((total_g_values - number_lts_g_values) /
                                 number_sts_g_values).tolist())
    lntts = []
    g_values = []
    for i in range(1, (total_g_values - number_lts_g_values),
                   sts_step_size):
        lntts.append(g.x[i].tolist())
        g_values.append(g.y[i].tolist())
    lntts += g.x[
             total_g_values - number_lts_g_values: total_g_values].tolist()
    g_values += g.y[
                total_g_values - number_lts_g_values: total_g_values].tolist()

    for i in range(len(lntts)):
        d['g_function_pairs'].append({'ln_tts': lntts[i],
                                      'g_value': g_values[i]})

    dt.utilities.js_dump(file_name, d, indent=4)

    return
