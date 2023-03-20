from enum import Enum, auto


class BHPipeType(Enum):
    COAXIAL = auto()
    DOUBLEUTUBEPARALLEL = auto()
    DOUBLEUTUBESERIES = auto()
    SINGLEUTUBE = auto()


class DoubleUTubeConnType(Enum):
    PARALLEL = auto()
    SERIES = auto()


class DesignMethodTimeStep(Enum):
    HOURLY = auto()
    HYBRID = auto()


class DesignGeomType(Enum):
    BIRECTANGLE = auto()
    BIRECTANGLECONSTRAINED = auto()
    BIZONEDRECTANGLE = auto()
    NEARSQUARE = auto()
    RECTANGLE = auto()
    ROWWISE = auto()


class FlowConfig(Enum):
    BOREHOLE = auto()
    SYSTEM = auto()
