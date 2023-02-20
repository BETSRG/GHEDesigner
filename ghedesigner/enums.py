from enum import Enum, auto


class BHPipeType(Enum):
    SingleUType = auto()
    DoubleUType = auto()
    CoaxialType = auto()


class FlowConfig(Enum):
    Parallel = auto()
    Series = auto()
