from enum import Enum, auto


class BHPipeType(Enum):
    SingleUType = auto()
    DoubleUTypeParallel = auto()
    DoubleUTypeSeries = auto()
    CoaxialType = auto()


class FlowConfig(Enum):
    Parallel = auto()
    Series = auto()


class DesignMethodTimeStep(Enum):
    Hybrid = auto()
    Hourly = auto()
