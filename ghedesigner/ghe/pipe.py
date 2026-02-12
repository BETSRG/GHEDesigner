from math import cos, pi, sin
from typing import cast

from ghedesigner.enums import PipeType
from ghedesigner.media import ThermalProperty
from ghedesigner.utilities import check_arg_bounds

FloatOrFloats = float | list[float]
TupleOrTuples = tuple | list[tuple]


class Pipe(ThermalProperty):
    def __init__(self, pipe_type: PipeType, conductivity: float | tuple[float, float], rho_cp: float) -> None:
        super().__init__(conductivity, rho_cp)
        self.type = pipe_type

    @classmethod
    def init_from_dict(cls, pipe_type: PipeType, pipe_props: dict) -> "Pipe":
        rho_cp = pipe_props["rho_cp"]

        match pipe_type:
            case PipeType.SINGLEUTUBE:
                k = pipe_props["conductivity"]
                return Pipe.init_single_u_tube(
                    conductivity=k,
                    rho_cp=rho_cp,
                    inner_diameter=pipe_props["inner_diameter"],
                    outer_diameter=pipe_props["outer_diameter"],
                    shank_spacing=pipe_props["shank_spacing"],
                    roughness=pipe_props["roughness"],
                    num_pipes=pipe_props.get("num_pipes", 1),
                )
            case PipeType.DOUBLEUTUBESERIES:
                k = pipe_props["conductivity"]
                return Pipe.init_double_u_tube_series(
                    conductivity=k,
                    rho_cp=rho_cp,
                    inner_diameter=pipe_props["inner_diameter"],
                    outer_diameter=pipe_props["outer_diameter"],
                    shank_spacing=pipe_props["shank_spacing"],
                    roughness=pipe_props["roughness"],
                )
            case PipeType.DOUBLEUTUBEPARALLEL:
                k = pipe_props["conductivity"]
                return Pipe.init_double_u_tube_parallel(
                    conductivity=k,
                    rho_cp=rho_cp,
                    inner_diameter=pipe_props["inner_diameter"],
                    outer_diameter=pipe_props["outer_diameter"],
                    shank_spacing=pipe_props["shank_spacing"],
                    roughness=pipe_props["roughness"],
                )
            case PipeType.COAXIAL:
                k = pipe_props["conductivity_inner"], pipe_props["conductivity_outer"]
                return Pipe.init_coaxial(
                    conductivity=k,
                    rho_cp=rho_cp,
                    inner_pipe_d_in=pipe_props["inner_pipe_d_in"],
                    inner_pipe_d_out=pipe_props["inner_pipe_d_out"],
                    outer_pipe_d_in=pipe_props["outer_pipe_d_in"],
                    outer_pipe_d_out=pipe_props["outer_pipe_d_out"],
                    roughness=pipe_props["roughness"],
                )
            case _:
                raise ValueError(f"Pipe type {pipe_type} not recognized.")

    @classmethod
    def init_single_u_tube(
        cls,
        conductivity: float,
        rho_cp: float,
        inner_diameter: float,
        outer_diameter: float,
        shank_spacing: float,
        roughness: float,
        num_pipes: int = 1,
    ) -> "Pipe":
        check_arg_bounds(inner_diameter, outer_diameter, "inner_diameter", "outer_diameter")
        p = cls(PipeType.SINGLEUTUBE, conductivity, rho_cp)  # TODO: Untangle this a little more
        r_in = inner_diameter / 2.0
        r_out = outer_diameter / 2.0
        pipe_positions = Pipe.place_pipes(shank_spacing, r_out, num_pipes)
        p._finalize(pipe_positions, r_in, r_out, shank_spacing, roughness)
        return p

    @classmethod
    def init_double_u_tube_series(
        cls,
        conductivity: float,
        rho_cp: float,
        inner_diameter: float,
        outer_diameter: float,
        shank_spacing: float,
        roughness: float,
    ) -> "Pipe":
        check_arg_bounds(inner_diameter, outer_diameter, "inner_diameter", "outer_diameter")
        p = cls(PipeType.DOUBLEUTUBESERIES, conductivity, rho_cp)
        r_in = inner_diameter / 2.0
        r_out = outer_diameter / 2.0
        pipe_positions = Pipe.place_pipes(shank_spacing, r_out, 2)
        p._finalize(pipe_positions, r_in, r_out, shank_spacing, roughness)
        return p

    @classmethod
    def init_double_u_tube_parallel(
        cls,
        conductivity: float,
        rho_cp: float,
        inner_diameter: float,
        outer_diameter: float,
        shank_spacing: float,
        roughness: float,
    ) -> "Pipe":
        check_arg_bounds(inner_diameter, outer_diameter, "inner_diameter", "outer_diameter")
        p = cls(PipeType.DOUBLEUTUBEPARALLEL, conductivity, rho_cp)
        r_in = inner_diameter / 2.0
        r_out = outer_diameter / 2.0
        pipe_positions = Pipe.place_pipes(shank_spacing, r_out, 2)
        p._finalize(pipe_positions, r_in, r_out, shank_spacing, roughness)
        return p

    @classmethod
    def init_coaxial(
        cls,
        conductivity: tuple[float, float],
        rho_cp: float,
        inner_pipe_d_in: float,
        inner_pipe_d_out: float,
        outer_pipe_d_in: float,
        outer_pipe_d_out: float,
        roughness: float,
    ) -> "Pipe":
        check_arg_bounds(inner_pipe_d_in, inner_pipe_d_out, "inner_pipe_d_in", "inner_pipe_d_out")
        check_arg_bounds(outer_pipe_d_in, outer_pipe_d_out, "outer_pipe_d_in", "outer_pipe_d_out")
        p = cls(PipeType.COAXIAL, conductivity, rho_cp)
        # Note: This convention is different from pygfunction
        r_inner = [inner_pipe_d_in / 2.0, inner_pipe_d_out / 2.0]  # The radii of the inner pipe from in to out
        r_outer = [outer_pipe_d_in / 2.0, outer_pipe_d_out / 2.0]  # The radii of the outer pipe from in to out
        p._finalize((0, 0), r_inner, r_outer, 0, roughness)
        return p

    def _finalize(
        self,
        pos: TupleOrTuples,
        r_in: FloatOrFloats,
        r_out: FloatOrFloats,
        s: float,
        roughness: float,
    ) -> None:
        self.pos = pos  # Pipe positions either a list of tuples or tuple
        self.r_in = r_in  # Pipe inner radius (m) can be a float or list
        self.r_out = r_out  # Pipe outer radius (m) can be a float or list
        self.s = s  # Center pipe to center pipe shank spacing
        self.roughness = roughness  # Pipe roughness (m)
        self.n_pipes = int(len(pos) / 2) if isinstance(pos, list) else 1

    def as_dict(self) -> dict:
        output = {
            "base": super().as_dict(),
            "pipe_center_positions": str(self.pos),
            "shank_spacing_pipe_to_pipe": {"value": self.s, "units": "m"},
            "pipe_roughness": {"value": self.roughness, "units": "m"},
            "number_of_pipes": self.n_pipes,
        }
        if isinstance(self.r_in, float):
            r_in = cast(float, self.r_in)
            r_out = cast(float, self.r_out)
            output["pipe_inner_diameter"] = str(r_in * 2.0)
            output["pipe_outer_diameter"] = str(r_out * 2.0)
        else:
            r_in_list = cast(list[float], self.r_in)
            r_out_list = cast(list[float], self.r_out)
            output["pipe_inner_diameters"] = str([x * 2.0 for x in r_in_list])
            output["pipe_outer_diameters"] = str([x * 2.0 for x in r_out_list])
        return output

    @staticmethod
    def place_pipes(s, r_out, n_pipes):
        """Positions pipes in an axis-symmetric configuration."""
        shank_space = s / 2 + r_out
        dt = pi / float(n_pipes)
        pos = [(0.0, 0.0) for _ in range(2 * n_pipes)]
        for i in range(n_pipes):
            pos[2 * i] = (shank_space * cos(2.0 * i * dt + pi), shank_space * sin(2.0 * i * dt + pi))
            pos[2 * i + 1] = (shank_space * cos(2.0 * i * dt + pi + dt), shank_space * sin(2.0 * i * dt + pi + dt))
        return pos
