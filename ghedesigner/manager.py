from json import loads, dumps
from pathlib import Path
from sys import exit, stderr
from time import time
from typing import List, Optional, Union

import click

from jsonschema import ValidationError

from ghedesigner import VERSION
from ghedesigner.borehole import GHEBorehole
from ghedesigner.constants import DEG_TO_RAD
from ghedesigner.design import AnyBisectionType, DesignBase, DesignNearSquare, DesignRectangle, DesignBiRectangle
from ghedesigner.design import DesignBiZoned, DesignBiRectangleConstrained, DesignRowWise
from ghedesigner.enums import BHPipeType, TimestepType, DesignGeomType, FlowConfigType
from ghedesigner.geometry import GeometricConstraints, GeometricConstraintsRectangle, GeometricConstraintsNearSquare
from ghedesigner.geometry import GeometricConstraintsBiRectangle, GeometricConstraintsBiZoned
from ghedesigner.geometry import GeometricConstraintsBiRectangleConstrained, GeometricConstraintsRowWise
from ghedesigner.media import GHEFluid, Grout, Pipe, Soil
from ghedesigner.output import OutputManager
from ghedesigner.simulation import SimulationParameters
from ghedesigner.validate import validate_input_file


class GHEManager:

    def __init__(self):
        self._fluid: Optional[GHEFluid] = None
        self._grout: Optional[Grout] = None
        self._soil: Optional[Soil] = None
        self._pipe: Optional[Pipe] = None
        self.pipe_type: Optional[BHPipeType] = None
        self._borehole: Optional[GHEBorehole] = None
        self._simulation_parameters: Optional[SimulationParameters] = None
        self._ground_loads: Optional[List[float]] = None
        # OK so geometric_constraints is tricky.  We have base classes, yay!
        # Unfortunately, the functionality between the child classes is not actually
        # collapsed into a base class function ... yet.  So there will be complaints
        # about types temporarily.  It's going in the right direction though.
        self.geom_type: Optional[DesignGeomType] = None
        self._geometric_constraints: Optional[GeometricConstraints] = None
        self._design: Optional[DesignBase] = None
        self._search: Optional[AnyBisectionType] = None
        self.results: Optional[OutputManager] = None

        # some things for results
        self._search_time: int = 0
        self.summary_results: dict = {}

    def set_design_geometry_type(self, design_geometry_str: str, throw: bool = True) -> int:
        """
        Sets the design type.

        :param design_geometry_str: design geometry input string.
        :param throw: By default, function will raise an exception on error, override to false to not raise exception
        :returns: Zero if successful, nonzero if failure
        :rtype: int
        """
        design_geometry_str = str(design_geometry_str).upper()
        if design_geometry_str == DesignGeomType.BIRECTANGLE.name:
            self.geom_type = DesignGeomType.BIRECTANGLE
        elif design_geometry_str == DesignGeomType.BIRECTANGLECONSTRAINED.name:
            self.geom_type = DesignGeomType.BIRECTANGLECONSTRAINED
        elif design_geometry_str == DesignGeomType.BIZONEDRECTANGLE.name:
            self.geom_type = DesignGeomType.BIZONEDRECTANGLE
        elif design_geometry_str == DesignGeomType.NEARSQUARE.name:
            self.geom_type = DesignGeomType.NEARSQUARE
        elif design_geometry_str == DesignGeomType.RECTANGLE.name:
            self.geom_type = DesignGeomType.RECTANGLE
        elif design_geometry_str == DesignGeomType.ROWWISE.name:
            self.geom_type = DesignGeomType.ROWWISE
        else:
            message = "Geometry constraint method not supported."
            print(message, file=stderr)
            if throw:
                raise ValueError(message)
            return 1

        return 0

    def set_pipe_type(self, bh_pipe_str: str, throw: bool = True) -> int:
        """
        Sets the borehole pipe type.

        :param bh_pipe_str: pipe type input string.
        :param throw: By default, function will raise an exception on error, override to false to not raise exception
        :returns: Zero if successful, nonzero if failure
        :rtype: int
        """
        bh_pipe_str = str(bh_pipe_str).upper()
        if bh_pipe_str == BHPipeType.SINGLEUTUBE.name:
            self.pipe_type = BHPipeType.SINGLEUTUBE
        elif bh_pipe_str == BHPipeType.DOUBLEUTUBEPARALLEL.name:
            self.pipe_type = BHPipeType.DOUBLEUTUBEPARALLEL
        elif bh_pipe_str == BHPipeType.DOUBLEUTUBESERIES.name:
            self.pipe_type = BHPipeType.DOUBLEUTUBESERIES
        elif bh_pipe_str == BHPipeType.COAXIAL.name:
            self.pipe_type = BHPipeType.COAXIAL
        else:
            message = f"Borehole pipe type \"{bh_pipe_str}\" not supported."
            print(message, file=stderr)
            if throw:
                raise ValueError(message)
            return 1

        return 0

    def set_fluid(self, fluid_name: str = "Water", concentration_percent: float = 0.0, temperature: float = 20.0,
                  throw: bool = True) -> int:
        """
        Sets the fluid instance.

        :param fluid_name: fluid name input string.
        :param concentration_percent: concentration percent of antifreeze mixture.
        :param temperature: design fluid temperature, in C.
        :param throw: By default, function will raise an exception on error, override to false to not raise exception
        :returns: Zero if successful, nonzero if failure
        :rtype: int
        """
        try:
            self._fluid = GHEFluid(fluid_str=fluid_name,
                                   percent=concentration_percent,
                                   temperature=temperature)
            return 0
        except ValueError:
            message = "Invalid fluid property input data."
            print(message, file=stderr)
            if throw:
                raise ValueError(message)
            return 1

    def set_grout(self, conductivity: float, rho_cp: float) -> int:
        """
        Sets the grout instance.

        :param conductivity: thermal conductivity, in W/m-K.
        :param rho_cp: volumetric heat capacity, in J/m^3-K.
        :returns: Zero if successful, nonzero if failure
        :rtype: int
        """
        self._grout = Grout(conductivity, rho_cp)
        return 0

    def set_soil(self, conductivity: float, rho_cp: float, undisturbed_temp: float) -> int:
        """
        Sets the soil instance.

        :param conductivity: thermal conductivity, in W/m-K.
        :param rho_cp: volumetric heat capacity, in J/m^3-K.
        :param undisturbed_temp: undisturbed soil temperature, in C.
        :returns: Zero if successful, nonzero if failure
        :rtype: int
        """
        self._soil = Soil(conductivity, rho_cp, undisturbed_temp)
        return 0

    def set_single_u_tube_pipe(self, inner_diameter: float, outer_diameter: float, shank_spacing: float,
                               roughness: float, conductivity: float, rho_cp: float) -> int:
        """
        Sets the pipe instance for a single u-tube pipe.

        :param inner_diameter: inner pipe diameter, in m.
        :param outer_diameter: outer pipe diameter, in m.
        :param shank_spacing: shank spacing between the u-tube legs, in m, as measured edge-to-edge.
        :param roughness: pipe surface roughness, in m.
        :param conductivity: thermal conductivity, in W/m-K.
        :param rho_cp: volumetric heat capacity, in J/m^3-K.
        :returns: Zero if successful, nonzero if failure
        :rtype: int
        """

        r_in = inner_diameter / 2.0
        r_out = outer_diameter / 2.0

        self.pipe_type = BHPipeType.SINGLEUTUBE
        pipe_positions = Pipe.place_pipes(shank_spacing, r_out, 1)
        self._pipe = Pipe(pipe_positions, r_in, r_out, shank_spacing, roughness, conductivity, rho_cp)
        return 0

    def set_double_u_tube_pipe_parallel(self, inner_diameter: float, outer_diameter: float, shank_spacing: float,
                                        roughness: float, conductivity: float, rho_cp: float) -> int:
        """
        Sets the pipe instance for a double u-tube pipe in a parallel configuration.

        :param inner_diameter: inner pipe diameter, in m.
        :param outer_diameter: outer pipe diameter, in m.
        :param shank_spacing: shank spacing between the u-tube legs, in m, as measured edge-to-edge.
        :param roughness: pipe surface roughness, in m.
        :param conductivity: thermal conductivity, in W/m-K.
        :param rho_cp: volumetric heat capacity, in J/m^3-K.
        :returns: Zero if successful, nonzero if failure
        :rtype: int
        """

        r_in = inner_diameter / 2.0
        r_out = outer_diameter / 2.0

        self.pipe_type = BHPipeType.DOUBLEUTUBEPARALLEL
        pipe_positions = Pipe.place_pipes(shank_spacing, r_out, 2)
        self._pipe = Pipe(pipe_positions, r_in, r_out, shank_spacing, roughness, conductivity, rho_cp)
        return 0

    def set_double_u_tube_pipe_series(self, inner_diameter: float, outer_diameter: float, shank_spacing: float,
                                      roughness: float, conductivity: float, rho_cp: float) -> int:
        """
        Sets the pipe instance for a double u-tube pipe in a series configuration.

        :param inner_diameter: inner pipe diameter, in m.
        :param outer_diameter: outer pipe diameter, in m.
        :param shank_spacing: shank spacing between the u-tube legs, in m, as measured edge-to-edge.
        :param roughness: pipe surface roughness, in m.
        :param conductivity: thermal conductivity, in W/m-K.
        :param rho_cp: volumetric heat capacity, in J/m^3-K.
        :returns: Zero if successful, nonzero if failure
        :rtype: int
        """

        r_in = inner_diameter / 2.0
        r_out = outer_diameter / 2.0

        self.pipe_type = BHPipeType.DOUBLEUTUBESERIES
        pipe_positions = Pipe.place_pipes(shank_spacing, r_out, 2)
        self._pipe = Pipe(pipe_positions, r_in, r_out, shank_spacing, roughness, conductivity, rho_cp)
        return 0

    def set_coaxial_pipe(self, inner_pipe_d_in: float, inner_pipe_d_out: float, outer_pipe_d_in: float,
                         outer_pipe_d_out: float, roughness: float, conductivity_inner: float,
                         conductivity_outer: float,
                         rho_cp: float) -> int:
        """
        Sets the pipe instance for a coaxial pipe.

        :param inner_pipe_d_in: inner pipe inner diameter, in m.
        :param inner_pipe_d_out: inner pipe outer diameter, in m.
        :param outer_pipe_d_in: outer pipe inner diameter, in m.
        :param outer_pipe_d_out: outer pipe outer diameter, in m.
        :param roughness: pipe surface roughness, in m.
        :param conductivity_inner: thermal conductivity of inner pipe, in W/m-K.
        :param conductivity_outer: thermal conductivity of outer pipe, in W/m-K.
        :param rho_cp: volumetric heat capacity, in J/m^3-K.
        :returns: Zero if successful, nonzero if failure
        :rtype: int
        """

        self.pipe_type = BHPipeType.COAXIAL

        # Note: This convention is different from pygfunction
        r_inner = [inner_pipe_d_in / 2.0, inner_pipe_d_out / 2.0]  # The radii of the inner pipe from in to out
        r_outer = [outer_pipe_d_in / 2.0, outer_pipe_d_out / 2.0]  # The radii of the outer pipe from in to out
        k_p = [conductivity_inner, conductivity_outer]
        self._pipe = Pipe((0, 0), r_inner, r_outer, 0, roughness, k_p, rho_cp)
        return 0

    def set_borehole(self, height: float, buried_depth: float, diameter: float) -> int:
        """
        Sets the borehole instance

        :param height: height, or active length, of the borehole, in m.
        :param buried_depth: depth of top of borehole below the ground surface, in m.
        :param diameter: diameter of the borehole, in m.
        :returns: Zero if successful, nonzero if failure
        :rtype: int
        """
        radius = diameter / 2.0
        self._borehole = GHEBorehole(height, buried_depth, radius, x=0.0, y=0.0)
        return 0

    def set_simulation_parameters(
            self, num_months: int, max_eft: float, min_eft: float, max_height: float, min_height: float
    ) -> int:
        """
        Sets the simulation parameters

        :param num_months: number of months in simulation.
        :param max_eft: maximum heat pump entering fluid temperature, in C.
        :param min_eft: minimum heat pump entering fluid temperature, in C.
        :param max_height: maximum height of borehole, in m.
        :param min_height: minimum height of borehole, in m.
        :returns: Zero if successful, nonzero if failure
        :rtype: int
        """
        self._simulation_parameters = SimulationParameters(
            1,
            num_months,
            max_eft,
            min_eft,
            max_height,
            min_height,
        )
        return 0

    def set_ground_loads_from_hourly_list(self, hourly_ground_loads: List[float]) -> int:
        """
        Sets the ground loads based on a list input.

        :param hourly_ground_loads: annual, hourly ground loads, in W.
         positive values indicate heat extraction, negative values indicate heat rejection.
        :returns: Zero if successful, nonzero if failure
        :rtype: int
        """
        # TODO: Add API methods for different load inputs
        self._ground_loads = hourly_ground_loads
        return 0

    def set_geometry_constraints_near_square(self, b: float, length: float) -> int:
        """
        Sets the geometry constraints for the near-square design method.

        :param b: borehole-to-borehole spacing, in m.
        :param length: side length of the sizing domain, in m.
        :returns: Zero if successful, nonzero if failure
        :rtype: int
        """
        self._geometric_constraints = GeometricConstraintsNearSquare(b, length)
        return 0

    def set_geometry_constraints_rectangle(self, length: float, width: float, b_min: float, b_max: float) -> int:
        """
        Sets the geometry constraints for the rectangle design method.

        :param length: side length of the sizing domain, in m.
        :param width: side width of the sizing domain, in m.
        :param b_min: minimum borehole-to-borehole spacing, in m.
        :param b_max: maximum borehole-to-borehole spacing, in m.
        :returns: Zero if successful, nonzero if failure
        :rtype: int
        """
        self.geom_type = DesignGeomType.RECTANGLE
        self._geometric_constraints = GeometricConstraintsRectangle(width, length, b_min, b_max)
        return 0

    def set_geometry_constraints_bi_rectangle(self, length: float, width: float, b_min: float,
                                              b_max_x: float, b_max_y: float) -> int:
        """
        Sets the geometry constraints for the bi-rectangle design method.

        :param length: side length of the sizing domain, in m.
        :param width: side width of the sizing domain, in m.
        :param b_min: minimum borehole-to-borehole spacing, in m.
        :param b_max_x: maximum borehole-to-borehole spacing in the x-direction, in m.
        :param b_max_y: maximum borehole-to-borehole spacing in the y-direction, in m.
        :returns: Zero if successful, nonzero if failure
        :rtype: int
        """
        self.geom_type = DesignGeomType.BIRECTANGLE
        self._geometric_constraints = GeometricConstraintsBiRectangle(width, length, b_min, b_max_x, b_max_y)
        return 0

    def set_geometry_constraints_bi_zoned_rectangle(self, length: float, width: float, b_min: float,
                                                    b_max_x: float, b_max_y: float) -> int:
        """
        Sets the geometry constraints for the bi-zoned rectangle design method.

        :param length: side length of the sizing domain, in m.
        :param width: side width of the sizing domain, in m.
        :param b_min: minimum borehole-to-borehole spacing, in m.
        :param b_max_x: maximum borehole-to-borehole spacing in the x-direction, in m.
        :param b_max_y: maximum borehole-to-borehole spacing in the y-direction, in m.
        :returns: Zero if successful, nonzero if failure
        :rtype: int
        """
        self.geom_type = DesignGeomType.BIZONEDRECTANGLE
        self._geometric_constraints = GeometricConstraintsBiZoned(width, length, b_min, b_max_x, b_max_y)
        return 0

    def set_geometry_constraints_bi_rectangle_constrained(self, b_min: float, b_max_x: float, b_max_y: float,
                                                          property_boundary: list, no_go_boundaries: list) -> int:
        """
        Sets the geometry constraints for the bi-rectangle constrained design method.

        :param b_min: minimum borehole-to-borehole spacing, in m.
        :param b_max_x: maximum borehole-to-borehole spacing in the x-direction, in m.
        :param b_max_y: maximum borehole-to-borehole spacing in the y-direction, in m.
        :param property_boundary: property boundary points, in m.
        :param no_go_boundaries: boundary points for no-go zones, in m.
        :returns: Zero if successful, nonzero if failure
        :rtype: int
        """
        self.geom_type = DesignGeomType.BIRECTANGLECONSTRAINED
        self._geometric_constraints = GeometricConstraintsBiRectangleConstrained(b_min, b_max_x, b_max_y,
                                                                                 property_boundary, no_go_boundaries)
        return 0

    def set_geometry_constraints_rowwise(self, perimeter_spacing_ratio: Union[float, None],
                                         max_spacing: float, min_spacing: float, spacing_step: float,
                                         max_rotation: float, min_rotation: float, rotate_step: float,
                                         property_boundary: list, no_go_boundaries: list) -> int:
        """
        Sets the geometry constraints for the row-wise design method.

        :param perimeter_spacing_ratio: the ratio between the minimum spacing between
            boreholes placed along the property and no-go zones and the standard borehole-to-borehole
            spacing used for internal boreholes.
        :param max_spacing: the largest minimum spacing that will be used to generate a RowWise field.
        :param min_spacing: the smallest minimum spacing that will be used to generate a RowWise field.
        :param spacing_step: the distance in spacing from the design found in the first part of first
            search to exhaustively check in the second part.
        :param max_rotation: the maximum rotation of the rows of each field relative to horizontal that
            will be used in the search.
        :param min_rotation: the minimum rotation of the rows of each field relative to horizontal that
            will be used in the search.
        :param rotate_step: step size for field rotation search.
        :param property_boundary: property boundary points.
        :param no_go_boundaries: boundary points for no-go zones.
        :returns: Zero if successful, nonzero if failure
        :rtype: int
        """

        # convert from degrees to radians
        max_rotation = max_rotation * DEG_TO_RAD
        min_rotation = min_rotation * DEG_TO_RAD

        self.geom_type = DesignGeomType.ROWWISE
        self._geometric_constraints = GeometricConstraintsRowWise(perimeter_spacing_ratio,
                                                                  min_spacing, max_spacing, spacing_step,
                                                                  min_rotation, max_rotation, rotate_step,
                                                                  property_boundary, no_go_boundaries)
        return 0

    def set_design(self, flow_rate: float, flow_type_str: str, throw: bool = True) -> int:
        """
        Set the design method.

        :param flow_rate: design flow rate, in lps.
        :param flow_type_str: flow type string input.
        :param throw: By default, function will raise an exception on error, override to false to not raise exception
        :returns: Zero if successful, nonzero if failure
        :rtype: int
        """

        flow_type_str = flow_type_str.upper()
        if flow_type_str == FlowConfigType.SYSTEM.name:
            flow_type = FlowConfigType.SYSTEM
        elif flow_type_str == FlowConfigType.BOREHOLE.name:
            flow_type = FlowConfigType.BOREHOLE
        else:
            message = f"FlowConfig \"{flow_type_str}\" is not implemented."
            print(message, file=stderr)
            if throw:
                raise ValueError(message)
            return 1

        if self._geometric_constraints.type is None:
            message = "Geometric constraints must be set before `set_design` is called."
            print(message, file=stderr)
            if throw:
                raise ValueError(message)
            return 1

        if self._geometric_constraints.type == DesignGeomType.NEARSQUARE:
            # temporary disable of the type checker because of the _geometric_constraints member
            # noinspection PyTypeChecker
            self._design = DesignNearSquare(
                flow_rate,
                self._borehole,
                self.pipe_type,
                self._fluid,
                self._pipe,
                self._grout,
                self._soil,
                self._simulation_parameters,
                self._geometric_constraints,
                self._ground_loads,
                flow_type=flow_type,
                method=TimestepType.HYBRID,
            )
        elif self._geometric_constraints.type == DesignGeomType.RECTANGLE:
            # temporary disable of the type checker because of the _geometric_constraints member
            # noinspection PyTypeChecker
            self._design = DesignRectangle(
                flow_rate,
                self._borehole,
                self.pipe_type,
                self._fluid,
                self._pipe,
                self._grout,
                self._soil,
                self._simulation_parameters,
                self._geometric_constraints,
                self._ground_loads,
                flow_type=flow_type,
                method=TimestepType.HYBRID,
            )
        elif self._geometric_constraints.type == DesignGeomType.BIRECTANGLE:
            # temporary disable of the type checker because of the _geometric_constraints member
            # noinspection PyTypeChecker
            self._design = DesignBiRectangle(
                flow_rate,
                self._borehole,
                self.pipe_type,
                self._fluid,
                self._pipe,
                self._grout,
                self._soil,
                self._simulation_parameters,
                self._geometric_constraints,
                self._ground_loads,
                flow_type=flow_type,
                method=TimestepType.HYBRID,
            )
        elif self._geometric_constraints.type == DesignGeomType.BIZONEDRECTANGLE:
            # temporary disable of the type checker because of the _geometric_constraints member
            # noinspection PyTypeChecker
            self._design = DesignBiZoned(
                flow_rate,
                self._borehole,
                self.pipe_type,
                self._fluid,
                self._pipe,
                self._grout,
                self._soil,
                self._simulation_parameters,
                self._geometric_constraints,
                self._ground_loads,
                flow_type=flow_type,
                method=TimestepType.HYBRID,
            )
        elif self._geometric_constraints.type == DesignGeomType.BIRECTANGLECONSTRAINED:
            # temporary disable of the type checker because of the _geometric_constraints member
            # noinspection PyTypeChecker
            self._design = DesignBiRectangleConstrained(
                flow_rate,
                self._borehole,
                self.pipe_type,
                self._fluid,
                self._pipe,
                self._grout,
                self._soil,
                self._simulation_parameters,
                self._geometric_constraints,
                self._ground_loads,
                flow_type=flow_type,
                method=TimestepType.HYBRID,
            )
        elif self._geometric_constraints.type == DesignGeomType.ROWWISE:
            # temporary disable of the type checker because of the _geometric_constraints member
            # noinspection PyTypeChecker
            self._design = DesignRowWise(
                flow_rate,
                self._borehole,
                self.pipe_type,
                self._fluid,
                self._pipe,
                self._grout,
                self._soil,
                self._simulation_parameters,
                self._geometric_constraints,
                self._ground_loads,
                flow_type=flow_type,
                method=TimestepType.HYBRID,
            )
        else:
            message = "This design method has not been implemented"
            print(message, file=stderr)
            if throw:
                raise ValueError(message)
            return 1
        return 0

    def find_design(self, throw: bool = True) -> int:
        """
        Calls design methods to execute sizing.

        :param throw: By default, function will raise an exception on error, override to false to not raise exception
        :returns: Zero if successful, nonzero if failure
        :rtype: int
        """

        if any([x is None for x in [
            self._fluid,
            self._grout,
            self._soil,
            self._pipe,
            self._borehole,
            self._simulation_parameters,
            self._ground_loads,
            self._geometric_constraints,
            self._design,
        ]]):
            message = "All GHE properties must be set before GHEManager.find_design is called."
            print(message, file=stderr)
            if throw:
                raise ValueError(message)
            return 1

        start_time = time()
        self._search = self._design.find_design()
        self._search.ghe.compute_g_functions()
        self._search_time = time() - start_time
        self._search.ghe.size(method=TimestepType.HYBRID)
        return 0

    def prepare_results(self, project_name: str, note: str, author: str, iteration_name: str):
        """
        Prepares the output results.
        """
        self.results = OutputManager(
            self._search,
            self._search_time,
            project_name,
            note,
            author,
            iteration_name,
            load_method=TimestepType.HYBRID,
        )

    def write_output_files(self, output_directory: Path, output_file_suffix: str = ""):
        """
        Writes the output files.

        :param output_directory: output directory for output files.
        :param output_file_suffix: adds a string suffix to the output files.
        """
        self.results.write_all_output_files(output_directory=output_directory, file_suffix=output_file_suffix)

    def write_input_file(self, output_file_path: Path, throw: bool = True) -> int:
        """
        Writes an input file based on current simulation configuration.

        :param output_file_path: output directory to write input file.
        :param throw: By default, function will raise an exception on error, override to false to not raise exception
        :returns: Zero if successful, nonzero if failure
        :rtype: int
        """

        # TODO: geometric constraints are currently held in two places
        #       SimulationParameters and GeometricConstraints
        #       these should be consolidated
        d_geo = self._geometric_constraints.to_input()
        d_geo['max_height'] = self._simulation_parameters.max_height
        d_geo['min_height'] = self._simulation_parameters.min_height

        # TODO: data held in different places
        d_des = self._design.to_input()
        d_des['max_eft'] = self._simulation_parameters.max_EFT_allowable
        d_des['min_eft'] = self._simulation_parameters.min_EFT_allowable

        # pipe data
        d_pipe = {'rho_cp': self._pipe.rhoCp, 'roughness': self._pipe.roughness}

        if self.pipe_type in [BHPipeType.SINGLEUTUBE, BHPipeType.DOUBLEUTUBEPARALLEL, BHPipeType.DOUBLEUTUBESERIES]:
            d_pipe['inner_diameter'] = self._pipe.r_in * 2.0
            d_pipe['outer_diameter'] = self._pipe.r_out * 2.0
            d_pipe['shank_spacing'] = self._pipe.s
            d_pipe['conductivity'] = self._pipe.k
        elif self.pipe_type == BHPipeType.COAXIAL:
            d_pipe['inner_pipe_d_in'] = self._pipe.r_in[0] * 2.0
            d_pipe['inner_pipe_d_out'] = self._pipe.r_in[1] * 2.0
            d_pipe['outer_pipe_d_in'] = self._pipe.r_out[0] * 2.0
            d_pipe['outer_pipe_d_out'] = self._pipe.r_out[1] * 2.0
            d_pipe['conductivity_inner'] = self._pipe.k[0]
            d_pipe['conductivity_outer'] = self._pipe.k[1]
        else:
            message = 'Invalid pipe type'
            print(message, file=stderr)
            if throw:
                raise ValueError(message)
            return 1

        if self.pipe_type == BHPipeType.SINGLEUTUBE:
            d_pipe['arrangement'] = BHPipeType.SINGLEUTUBE.name
        elif self.pipe_type == BHPipeType.DOUBLEUTUBEPARALLEL:
            d_pipe['arrangement'] = BHPipeType.DOUBLEUTUBEPARALLEL.name
        elif self.pipe_type == BHPipeType.DOUBLEUTUBESERIES:
            d_pipe['arrangement'] = BHPipeType.DOUBLEUTUBESERIES.name
        elif self.pipe_type == BHPipeType.COAXIAL:
            d_pipe['arrangement'] = BHPipeType.COAXIAL.name
        else:
            message = 'Invalid pipe type'
            print(message, file=stderr)
            if throw:
                raise ValueError(message)
            return 1

        d = {
            'version': VERSION,
            'fluid': self._fluid.to_input(),
            'grout': self._grout.to_input(),
            'soil': self._soil.to_input(),
            'pipe': d_pipe,
            'borehole': self._borehole.to_input(),
            'simulation': self._simulation_parameters.to_input(),
            'geometric_constraints': d_geo,
            'design': d_des,
            'loads': {'ground_loads': self._ground_loads}
        }

        with open(output_file_path, 'w') as f:
            f.write(dumps(d, sort_keys=True, indent=2, separators=(',', ': ')))
        return 0


def run_manager_from_cli_worker(input_file_path: Path, output_directory: Path) -> int:
    """
    Worker function to run simulation.

    :param input_file_path: path to input file.
    :param output_directory: path to write output files.
    """

    if not input_file_path.exists():
        print(f"No input file found at {input_file_path}, aborting", file=stderr)
        return 1

    # validate inputs against schema before doing anything
    if validate_input_file(input_file_path) != 0:
        return 1

    inputs = loads(input_file_path.read_text())

    ghe = GHEManager()

    version = inputs['version']

    if version != VERSION:
        print("Mismatched version, could be a problem", file=stderr)

    fluid_props = inputs['fluid']  # type: dict
    grout_props = inputs['grout']  # type: dict
    soil_props = inputs['soil']  # type: dict
    pipe_props = inputs['pipe']  # type: dict
    borehole_props = inputs['borehole']  # type: dict
    sim_props = inputs['simulation']  # type: dict
    constraint_props = inputs['geometric_constraints']  # type: dict
    design_props = inputs['design']  # type: dict
    ground_load_props = inputs['loads']['ground_loads']  # type: list

    ghe.set_fluid(**fluid_props, throw=False)
    ghe.set_grout(**grout_props)
    ghe.set_soil(**soil_props)
    ghe.set_pipe_type(pipe_props["arrangement"], throw=False)

    if ghe.pipe_type == BHPipeType.SINGLEUTUBE:
        ghe.set_single_u_tube_pipe(
            inner_diameter=pipe_props["inner_diameter"],
            outer_diameter=pipe_props["outer_diameter"],
            shank_spacing=pipe_props["shank_spacing"],
            roughness=pipe_props["roughness"],
            conductivity=pipe_props["conductivity"],
            rho_cp=pipe_props["rho_cp"]
        )
    elif ghe.pipe_type == BHPipeType.DOUBLEUTUBEPARALLEL:
        ghe.set_double_u_tube_pipe_parallel(
            inner_diameter=pipe_props["inner_diameter"],
            outer_diameter=pipe_props["outer_diameter"],
            shank_spacing=pipe_props["shank_spacing"],
            roughness=pipe_props["roughness"],
            conductivity=pipe_props["conductivity"],
            rho_cp=pipe_props["rho_cp"]
        )
    elif ghe.pipe_type == BHPipeType.DOUBLEUTUBESERIES:
        ghe.set_double_u_tube_pipe_series(
            inner_diameter=pipe_props["inner_diameter"],
            outer_diameter=pipe_props["outer_diameter"],
            shank_spacing=pipe_props["shank_spacing"],
            roughness=pipe_props["roughness"],
            conductivity=pipe_props["conductivity"],
            rho_cp=pipe_props["rho_cp"]
        )
    elif ghe.pipe_type == BHPipeType.COAXIAL:
        ghe.set_coaxial_pipe(
            inner_pipe_d_in=pipe_props["inner_pipe_d_in"],
            inner_pipe_d_out=pipe_props["inner_pipe_d_out"],
            outer_pipe_d_in=pipe_props["outer_pipe_d_in"],
            outer_pipe_d_out=pipe_props["outer_pipe_d_out"],
            roughness=pipe_props["roughness"],
            conductivity_inner=pipe_props["conductivity_inner"],
            conductivity_outer=pipe_props["conductivity_outer"],
            rho_cp=pipe_props["rho_cp"]
        )

    ghe.set_borehole(
        height=constraint_props["max_height"],
        buried_depth=borehole_props["buried_depth"],
        diameter=borehole_props["diameter"]
    )

    ghe.set_ground_loads_from_hourly_list(ground_load_props)
    ghe.set_simulation_parameters(
        num_months=sim_props["num_months"],
        max_eft=design_props["max_eft"],
        min_eft=design_props["min_eft"],
        max_height=constraint_props["max_height"],
        min_height=constraint_props["min_height"]
    )

    if ghe.set_design_geometry_type(constraint_props["method"], throw=False) != 0:
        return 1

    if ghe.geom_type == DesignGeomType.RECTANGLE:
        ghe.set_geometry_constraints_rectangle(
            length=constraint_props["length"],
            width=constraint_props["width"],
            b_min=constraint_props["b_min"],
            b_max=constraint_props["b_max"],
        )
    elif ghe.geom_type == DesignGeomType.NEARSQUARE:
        ghe.set_geometry_constraints_near_square(
            b=constraint_props["b"],
            length=constraint_props["length"]
        )
    elif ghe.geom_type == DesignGeomType.BIRECTANGLE:
        ghe.set_geometry_constraints_bi_rectangle(
            length=constraint_props["length"],
            width=constraint_props["width"],
            b_min=constraint_props["b_min"],
            b_max_x=constraint_props["b_max_x"],
            b_max_y=constraint_props["b_max_y"]
        )
    elif ghe.geom_type == DesignGeomType.BIZONEDRECTANGLE:
        ghe.set_geometry_constraints_bi_zoned_rectangle(
            length=constraint_props["length"],
            width=constraint_props["width"],
            b_min=constraint_props["b_min"],
            b_max_x=constraint_props["b_max_x"],
            b_max_y=constraint_props["b_max_y"]
        )
    elif ghe.geom_type == DesignGeomType.BIRECTANGLECONSTRAINED:
        ghe.set_geometry_constraints_bi_rectangle_constrained(
            b_min=constraint_props["b_min"],
            b_max_x=constraint_props["b_max_x"],
            b_max_y=constraint_props["b_max_y"],
            property_boundary=constraint_props["property_boundary"],
            no_go_boundaries=constraint_props["no_go_boundaries"]
        )
    elif ghe.geom_type == DesignGeomType.ROWWISE:
        # We are using perimeter calculations if present
        if "perimeter_spacing_ratio" in constraint_props.keys():
            perimeter_spacing_ratio = constraint_props["perimeter_spacing_ratio"]
        else:
            perimeter_spacing_ratio = None

        ghe.set_geometry_constraints_rowwise(
            perimeter_spacing_ratio=perimeter_spacing_ratio,
            max_spacing=constraint_props["max_spacing"],
            min_spacing=constraint_props["min_spacing"],
            spacing_step=constraint_props["spacing_step"],
            max_rotation=constraint_props["max_rotation"],
            min_rotation=constraint_props["min_rotation"],
            rotate_step=constraint_props["rotate_step"],
            property_boundary=constraint_props["property_boundary"],
            no_go_boundaries=constraint_props["no_go_boundaries"]
        )
    else:
        print("Geometry constraint method not supported.", file=stderr)
        return 1

    ghe.set_design(
        flow_rate=design_props["flow_rate"],
        flow_type_str=design_props["flow_type"],
        throw=False
    )

    ghe.find_design(throw=False)
    ghe.prepare_results("GHEDesigner Run from CLI", "Notes", "Author", "Iteration Name")
    ghe.write_output_files(output_directory)

    return 0


@click.command(name="GHEDesignerCommandLine")
@click.argument("input-path", type=click.Path(exists=True))
@click.argument("output-directory", type=click.Path(exists=True), required=False)
@click.version_option(VERSION)
@click.option(
    "--validate",
    default=False,
    is_flag=True,
    show_default=False,
    help="Validate input and exit."
)
def run_manager_from_cli(input_path, output_directory, validate):
    input_path = Path(input_path).resolve()

    if validate:
        try:
            validate_input_file(input_path)
            print("Valid input file.")
            return 0
        except ValidationError:
            print("Schema validation error. See previous error message for details.", file=stderr)
            return 1

    output_path = Path(output_directory).resolve()

    if not input_path.exists():
        print(f'Input file does not exist. Input file path: "{str(input_path)}"', file=stderr)

    return run_manager_from_cli_worker(input_path, output_path)


if __name__ == "__main__":
    exit(run_manager_from_cli())
