from enum import Enum, auto


class PipeType(Enum):
    COAXIAL = "COAXIAL"
    DOUBLEUTUBEPARALLEL = "DOUBLEUTUBEPARALLEL"
    DOUBLEUTUBESERIES = "DOUBLEUTUBESERIES"
    SINGLEUTUBE = "SINGLEUTUBE"


class DoubleUTubeConnType(Enum):
    PARALLEL = auto()
    SERIES = auto()


class TimestepType(Enum):
    HOURLY = auto()
    HYBRID = auto()


class DesignGeomType(Enum):
    BIRECTANGLE = auto()
    BIRECTANGLECONSTRAINED = auto()
    BIZONEDRECTANGLE = auto()
    NEARSQUARE = auto()
    RECTANGLE = auto()
    ROWWISE = auto()
    NONE = auto()  # TODO: Check this won't break anything


class FlowConfigType(Enum):
    BOREHOLE = "BOREHOLE"
    SYSTEM = "SYSTEM"


class FluidType(Enum):
    ETHYLALCOHOL = auto()
    ETHYLENEGLYCOL = auto()
    METHYLALCOHOL = auto()
    PROPYLENEGLYCOL = auto()
    WATER = auto()
