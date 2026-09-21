from enum import Enum, StrEnum, auto


class BHType(Enum):
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


class FluidType(Enum):
    ETHYLALCOHOL = auto()
    ETHYLENEGLYCOL = auto()
    METHYLALCOHOL = auto()
    PROPYLENEGLYCOL = auto()
    WATER = auto()


class SimCompType(Enum):
    BUILDING = auto()
    GROUND_HEAT_EXCHANGER = auto()
    SOURCE_SINK_HEAT_EXCHANGER = auto()
    HEAT_PUMP = auto()
    ISOLATED_HORIZONTAL_PIPE = auto()
    COUPLED_HORIZONTAL_PIPE = auto()


class CentralLoopType(Enum):
    ONEPIPE = auto()
    TWOPIPE = auto()
    NETWORK = auto()


class SourceSinkOpMode(Enum):
    SOURCE = auto()
    SINK = auto()


class ParametricStudyParameters(StrEnum):
    MAX_EFT_MODIFICATION = auto()
    MIN_EFT_MODIFICATION = auto()
    GROUT_CONDUCTIVITIES = auto()
    PIPE_SIZES = auto()
    BOREHOLE_HEIGHTS = auto()
    UPDATED_TOPOLOGY = auto()
