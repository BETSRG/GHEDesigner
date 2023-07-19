from enum import Enum, auto


class BHPipeType(Enum):
    COAXIAL = auto()
    DOUBLEUTUBEPARALLEL = auto()
    DOUBLEUTUBESERIES = auto()
    SINGLEUTUBE = auto()


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


class FlowConfigType(Enum):
    BOREHOLE = auto()
    SYSTEM = auto()


class FluidType(Enum):
    ETHYLALCOHOL = auto()
    ETHYLENEGLYCOL = auto()
    METHYLALCOHOL = auto()
    PROPYLENEGLYCOL = auto()
    WATER = auto()
