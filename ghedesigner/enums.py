from enum import Enum, auto


class BHPipeType(Enum):
    SINGLEUTUBE = auto()
    DOUBLEUTUBEPARALLEL = auto()
    DOUBLEUTUBESERIES = auto()
    COAXIAL = auto()


class FlowConfig(Enum):
    PARALLEL = auto()
    SERIES = auto()


class DesignMethodTimeStep(Enum):
    HYBRID = auto()
    HOURLY = auto()
