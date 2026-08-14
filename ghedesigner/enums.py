"""Enumerations shared by GHEDesigner inputs and numerical workflows."""

from enum import Enum, auto


class BHType(Enum):
    """Supported vertical borehole heat-exchanger pipe arrangements."""

    COAXIAL = "COAXIAL"
    DOUBLEUTUBEPARALLEL = "DOUBLEUTUBEPARALLEL"
    DOUBLEUTUBESERIES = "DOUBLEUTUBESERIES"
    SINGLEUTUBE = "SINGLEUTUBE"


class DoubleUTubeConnType(Enum):
    """Hydraulic connection choices for a double U-tube borehole."""

    PARALLEL = auto()
    SERIES = auto()


class TimestepType(Enum):
    """Time discretizations supported by GHE simulations."""

    HOURLY = auto()
    HYBRID = auto()


class DesignGeomType(Enum):
    """Available borefield geometry search strategies."""

    BIRECTANGLE = auto()
    BIRECTANGLECONSTRAINED = auto()
    BIZONEDRECTANGLE = auto()
    NEARSQUARE = auto()
    RECTANGLE = auto()
    ROWWISE = auto()
    NONE = auto()  # TODO: Check this won't break anything


class FlowConfigType(Enum):
    """Whether the configured volumetric flow applies per borehole or per system."""

    BOREHOLE = "BOREHOLE"
    SYSTEM = "SYSTEM"


class FluidType(Enum):
    """Heat-transfer fluids supported by the property library."""

    ETHYLALCOHOL = auto()
    ETHYLENEGLYCOL = auto()
    METHYLALCOHOL = auto()
    PROPYLENEGLYCOL = auto()
    WATER = auto()


class SimCompType(Enum):
    """Component categories that can appear in a district simulation topology."""

    BUILDING = auto()
    GROUND_HEAT_EXCHANGER = auto()
    SOURCE_SINK_HEAT_EXCHANGER = auto()
    HEAT_PUMP = auto()
    ISOLATED_HORIZONTAL_PIPE = auto()
    COUPLED_HORIZONTAL_PIPE = auto()


class CentralLoopType(Enum):
    """Supported district central-loop pipe configurations."""

    ONEPIPE = auto()
    TWOPIPE = auto()


class SourceSinkOpMode(Enum):
    """Operating modes for a source/sink heat exchanger."""

    SOURCE = auto()
    SINK = auto()
