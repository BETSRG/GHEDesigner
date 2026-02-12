# This is just a very thin wrapper over the pygfunction borehole class
# TODO: Think about how this class plays with the other actual configured borehole classes, could just be a rename

from typing import Any

from pygfunction.boreholes import Borehole as PyG_Borehole


class Borehole(PyG_Borehole):
    def __init__(
        self,
        burial_depth: float,
        borehole_radius: float,
        borehole_height: float = 100,  # TODO: Not sure this value is ever used, but we set it to 100 everywhere
        tilt: float = 0.0,
        orientation: float = 0.0,
        x: Any = 0.0,
        y: Any = 0.0,
    ):
        super().__init__(
            H=borehole_height, D=burial_depth, r_b=borehole_radius, x=x, y=y, tilt=tilt, orientation=orientation
        )
