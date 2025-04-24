from pygfunction.boreholes import Borehole

from ghedesigner.enums import BHPipeType
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.media import GHEFluid, Grout, Soil


class GroundHeatExchanger:
    def __init__(
        self,
        grout_conductivity: float,
        grout_rho_cp: float,
        soil_conductivity: float,
        soil_rho_cp: float,
        soil_undisturbed_temperature: float,
        borehole_buried_depth: float,
        borehole_radius: float,
        pipe_arrangement_type: BHPipeType,
        pipe_parameters: dict,
        fluid_name: str = "Water",
        fluid_concentration_percent: float = 0.0,
        fluid_temperature: float = 20.0,
    ) -> None:
        self.fluid = GHEFluid(fluid_name, fluid_concentration_percent, fluid_temperature)
        self.grout = Grout(grout_conductivity, grout_rho_cp)
        self.soil = Soil(soil_conductivity, soil_rho_cp, soil_undisturbed_temperature)
        if pipe_arrangement_type == BHPipeType.SINGLEUTUBE:
            params = ["conductivity", "inner_diameter", "outer_diameter", "shank_spacing", "roughness"]
            if not all([x in pipe_parameters for x in params]):
                raise ValueError(f"pipe_arrangement_type of {pipe_arrangement_type!s} requires these inputs: {params}")
            pipe_parameters["num_pipes"] = 1
            self.pipe = Pipe.init_single_u_tube(**pipe_parameters)
            # pipe = Pipe.init_single_u_tube(
            #     inner_diameter=0.03404,
            #     outer_diameter=0.04216,
            #     shank_spacing=0.01856,
            #     roughness=1.0e-6,
            #     conductivity=0.4,
            #     rho_cp=1542000.0,
            #     num_pipes=1,
            # )
        elif pipe_arrangement_type == BHPipeType.DOUBLEUTUBESERIES:
            params = ["conductivity", "rho_cp", "inner_diameter", "outer_diameter", "shank_spacing", "roughness"]
            if not all([x in pipe_parameters for x in params]):
                raise ValueError(f"pipe_arrangement_type of {pipe_arrangement_type!s} requires these inputs: {params}")
            self.pipe = Pipe.init_double_u_tube_series(**pipe_parameters)
            # pipe = Pipe.init_double_u_tube_series(
            #     inner_diameter=0.03404,
            #     outer_diameter=0.04216,
            #     shank_spacing=0.01856,
            #     roughness=1.0e-6,
            #     conductivity=0.4,
            #     rho_cp=1542000.0,
            # )
        elif pipe_arrangement_type == BHPipeType.DOUBLEUTUBEPARALLEL:
            params = ["conductivity", "rho_cp", "inner_diameter", "outer_diameter", "shank_spacing", "roughness"]
            if not all([x in pipe_parameters for x in params]):
                raise ValueError(f"pipe_arrangement_type of {pipe_arrangement_type!s} requires these inputs: {params}")
            self.pipe = Pipe.init_double_u_tube_parallel(**pipe_parameters)
            # pipe = Pipe.init_double_u_tube_parallel(
            #     inner_diameter=0.03404,
            #     outer_diameter=0.04216,
            #     shank_spacing=0.01856,
            #     roughness=1.0e-6,
            #     conductivity=0.4,
            #     rho_cp=1542000.0,
            # )
        else:  # Assuming coaxial
            params = [
                "conductivity_inner",
                "rho_cp",
                "conductivity_outer",
                "inner_pipe_d_in",
                "inner_pipe_d_out",
                "outer_pipe_d_in",
                "outer_pipe_d_out",
            ]
            if not all([x in pipe_parameters for x in params]):
                raise ValueError(f"pipe_arrangement_type of {BHPipeType.COAXIAL!s} requires these inputs: {params}")
            pipe_parameters["conductivity"] = (
                pipe_parameters["conductivity_inner"],
                pipe_parameters["conductivity_outer"],
            )
            del pipe_parameters["conductivity_inner"]
            del pipe_parameters["conductivity_outer"]
            self.pipe = Pipe.init_coaxial(**pipe_parameters)
            # pipe = Pipe.init_coaxial(
            #     inner_pipe_d_in=0.0442,
            #     inner_pipe_d_out=0.050,
            #     outer_pipe_d_in=0.0974,
            #     outer_pipe_d_out=0.11,
            #     roughness=1.0e-6,
            #     conductivity=(0.4, 0.4),
            #     rho_cp=1542000.0,
            # )
        # self.pipe = Pipe(pipe_arrangement_type, pipe_parameters["conductivity"], pipe_parameters["rho_cp"])
        self.pygfunction_borehole = Borehole(100, borehole_buried_depth, borehole_radius, x=0.0, y=0.0)

    @classmethod
    def init_from_dictionary(cls, ghe_dict: dict, fluid_inputs: dict | None = None) -> "GroundHeatExchanger":
        """
        Initialize a GroundHeatExchanger object from input dictionaries, performing validation and ultimately calling
        the main object constructor.
        :param ghe_dict: Dictionary of ground heat exchanger parameters, see the input schema specification for required
                         inputs in the ground-heat-exchanger schema field.
        :param fluid_inputs: Optional dictionary of fluid input parameters, see the input schema fluid spec for details.
        :return: GroundHeatExchanger object.
        # TODO: Add validation back in to the input fields
        """
        grout_parameters: dict = ghe_dict["grout"]
        g_c: float = grout_parameters["conductivity"]
        g_rho_cp: float = grout_parameters["rho_cp"]

        soil_parameters: dict = ghe_dict["soil"]
        s_k: float = soil_parameters["conductivity"]
        s_rho_cp: float = soil_parameters["rho_cp"]
        s_temp: float = soil_parameters["undisturbed_temp"]

        borehole_parameters: dict = ghe_dict["borehole"]
        buried_depth: float = borehole_parameters["buried_depth"]
        diameter: float = borehole_parameters["diameter"]
        radius: float = diameter / 2.0

        fluid_name: str = "Water"
        concentration_percent: float = 0.0
        temperature: float = 20.0
        if fluid_inputs is not None:
            fluid_name = fluid_inputs.get("fluid_name", "Water")
            concentration_percent = fluid_inputs.get("concentration_percent", 0.0)
            temperature = fluid_inputs.get("temperature", 20.0)

        pipe_parameters: dict = ghe_dict["pipe"]
        pipe_type: BHPipeType = BHPipeType(pipe_parameters["arrangement"].upper())
        del pipe_parameters["arrangement"]

        ghe: GroundHeatExchanger = cls(
            g_c,
            g_rho_cp,
            s_k,
            s_rho_cp,
            s_temp,
            buried_depth,
            radius,
            pipe_type,
            pipe_parameters,
            fluid_name,
            concentration_percent,
            temperature,
        )
        return ghe

    # def write_input_file(self, output_file_path: Path, simulation_parameters: SimulationParameters) -> None:
    #     """
    #     Writes an input file based on current simulation configuration.
    #
    #     :param output_file_path: output directory to write input file.
    #     :raises AttributeError: If necessary class attributes are not set.
    #     :raises ValueError: If the pipe type is not supported.
    #     """
    #     # TODO: geometric constraints are currently held in two places
    #     #       SimulationParameters and GeometricConstraints
    #     #       these should be consolidated
    #     d_geo = self._geometric_constraints.to_input()
    #     d_geo["max_height"] = simulation_parameters.max_height
    #     d_geo["min_height"] = simulation_parameters.min_height
    #
    #     # TODO: data held in different places
    #     d_des = self._design.to_input()
    #     d_des["max_eft"] = simulation_parameters.max_EFT_allowable
    #     d_des["min_eft"] = simulation_parameters.min_EFT_allowable
    #
    #     if simulation_parameters.max_boreholes is not None:
    #         d_des["max_boreholes"] = simulation_parameters.max_boreholes
    #     if simulation_parameters.continue_if_design_unmet is True:
    #         d_des["continue_if_design_unmet"] = simulation_parameters.continue_if_design_unmet
    #
    #     # pipe data
    #     d_pipe = {"rho_cp": self.pipe.rhoCp, "roughness": self.pipe.roughness}
    #
    #     if self.pipe.type in [BHPipeType.SINGLEUTUBE, BHPipeType.DOUBLEUTUBEPARALLEL, BHPipeType.DOUBLEUTUBESERIES]:
    #         d_pipe["inner_diameter"] = self.pipe.r_in * 2.0
    #         d_pipe["outer_diameter"] = self.pipe.r_out * 2.0
    #         d_pipe["shank_spacing"] = self.pipe.s
    #         d_pipe["conductivity"] = self.pipe.k
    #     elif self.pipe.type == BHPipeType.COAXIAL:
    #         d_pipe["inner_pipe_d_in"] = self.pipe.r_in[0] * 2.0
    #         d_pipe["inner_pipe_d_out"] = self.pipe.r_in[1] * 2.0
    #         d_pipe["outer_pipe_d_in"] = self.pipe.r_out[0] * 2.0
    #         d_pipe["outer_pipe_d_out"] = self.pipe.r_out[1] * 2.0
    #         d_pipe["conductivity_inner"] = self.pipe.k[0]
    #         d_pipe["conductivity_outer"] = self.pipe.k[1]
    #     else:
    #         raise ValueError(f"Invalid pipe type '{self.pipe.type.name if self.pipe.type else 'None'}'")
    #
    #     d_pipe["arrangement"] = self.pipe.type.name
    #
    #     d = {
    #         "fluid": self.fluid.to_input(),
    #         "grout": self.grout.to_input(),
    #         "soil": self.soil.to_input(),
    #         "pipe": d_pipe,
    #         # "borehole": self._borehole.to_input(),
    #         # "simulation": self._simulation_parameters.to_input(),
    #         "geometric_constraints": d_geo,
    #         "design": d_des,
    #         "loads": {"ground_loads": self._ground_loads},
    #     }
    #
    #     output_file_path.parent.mkdir(parents=True, exist_ok=True)
    #     output_file_path.write_text(dumps(d, sort_keys=True, indent=2, separators=(",", ": ")))
