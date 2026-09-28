import importlib.metadata
from math import pi, tau

VERSION = importlib.metadata.version("ghedesigner")
INPUT_VERSION = 3

PI = pi
DEG_TO_RAD = pi / 180.0
RAD_TO_DEG = 180.0 / pi
PI_OVER_2 = pi / 2.0
TWO_PI = tau
HRS_IN_DAY = 24
SEC_IN_HR = 3600
DAYS_IN_YEAR = 365
SEC_IN_DAY = SEC_IN_HR * HRS_IN_DAY
SEC_IN_YEAR = SEC_IN_DAY * DAYS_IN_YEAR
MONTHS_IN_YEAR = 12
HOURS_IN_YEAR = HRS_IN_DAY * DAYS_IN_YEAR
LPS_TO_M3S = 1 / 1000
BOREHOLES_PER_SQUARE_METER = 0.0494  # This is the tightest boreholes can be infinitely
# tessellated with a minimum 4.5m spacing (to my knowledge).
IDX_COMPARISON_OFFSET_1 = 1  # Used for offsets in history term calculation
IDX_COMPARISON_OFFSET_2 = 2  # Used for offsets in history term calculation
SIMULATION_CONSTANT_COP_COOLING_OFFSET = 15.0  # (°C) Used to estimate constant COP if temperature bounds are not given.
SIMULATION_CONSTANT_COP_HEATING_OFFSET = 10.0  # Also used to set HP fit curve temperature limits if not given.
SIMULATION_OPERATING_TEMPERATURE_DIFFERENCE = 10.0  # (°C) Used to determine flowrate if only COP is given for HP model.
DLA_EXPANSION_RATE = 1.62
DLA_BINS_PER_LEVEL = 9
HORZ_LIBRARY_FILENAME = "unified_horizontal_library.json"
