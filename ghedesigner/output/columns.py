"""Canonical column names for public CSV output files."""

from dataclasses import dataclass

SIMULATION = "Simulation"
NETWORK = "Network"
SEARCH = "Search"
SYSTEM = "System"
PARAMETRIC_STUDY = "Parametric Study"


@dataclass(frozen=True)
class OutputVariable:
    """A descriptive variable name and its output unit."""

    name: str
    unit: str

    def for_object(self, object_name: str) -> str:
        """Return ``<Object Name>: <Variable Name> [<Units>]``."""
        if not object_name.strip():
            raise ValueError("CSV output object names must not be empty.")
        return f"{object_name}: {self.name} [{self.unit}]"


def output_column(object_name: str, variable_name: str, unit: str) -> str:
    """Create a canonical public CSV column name."""
    return OutputVariable(variable_name, unit).for_object(object_name)


ELAPSED_TIME = OutputVariable("Elapsed Time", "h")
ENTERING_FLUID_TEMPERATURE = OutputVariable("Entering Fluid Temperature", "C")
EXITING_FLUID_TEMPERATURE = OutputVariable("Exiting Fluid Temperature", "C")
MIXED_LOOP_EXITING_FLUID_TEMPERATURE = OutputVariable("Mixed-Loop Exiting Fluid Temperature", "C")
MEAN_FLUID_TEMPERATURE = OutputVariable("Mean Fluid Temperature", "C")
CONTROL_ENTERING_FLUID_TEMPERATURE = OutputVariable("Control Entering Fluid Temperature", "C")
HEATING_LOAD = OutputVariable("Heating Load", "W")
COOLING_LOAD = OutputVariable("Cooling Load", "W")
NET_BUILDING_LOAD = OutputVariable("Net Building Load", "W")
MASS_FLOW_RATE = OutputVariable("Mass Flow Rate", "kg/s")
HEATING_HEAT_PUMP_POWER = OutputVariable("Heating Heat Pump Power", "W")
COOLING_HEAT_PUMP_POWER = OutputVariable("Cooling Heat Pump Power", "W")
TOTAL_HEAT_PUMP_POWER = OutputVariable("Total Heat Pump Power", "W")
CIRCULATION_PUMP_POWER = OutputVariable("Circulation Pump Power", "W")
SOURCE_SIDE_HEATING_HEAT_TRANSFER_RATE = OutputVariable("Source-Side Heating Heat Transfer Rate", "W")
SOURCE_SIDE_COOLING_HEAT_TRANSFER_RATE = OutputVariable("Source-Side Cooling Heat Transfer Rate", "W")
SOURCE_SIDE_NET_HEAT_TRANSFER_RATE = OutputVariable("Source-Side Net Heat Transfer Rate", "W")
HEAT_TRANSFER_RATE_PER_LENGTH = OutputVariable("Heat Transfer Rate", "W/m")
HEAT_TRANSFER_RATE = OutputVariable("Heat Transfer Rate", "W")
PUMP_MASS_FLOW_RATE = OutputVariable("Pump Mass Flow Rate", "kg/s")
LOCAL_RECIRCULATION_FLOW_RATE = OutputVariable("Local Recirculation Flow Rate", "kg/s")
OPERATING_STATUS = OutputVariable("Operating Status", "-")


def segment_exiting_fluid_temperature(segment_number: int) -> OutputVariable:
    return OutputVariable(f"Segment {segment_number} Exiting Fluid Temperature", "C")


def segment_mean_fluid_temperature(segment_number: int) -> OutputVariable:
    return OutputVariable(f"Segment {segment_number} Mean Fluid Temperature", "C")


def segment_heat_transfer_rate(segment_number: int) -> OutputVariable:
    return OutputVariable(f"Segment {segment_number} Heat Transfer Rate", "W/m")
