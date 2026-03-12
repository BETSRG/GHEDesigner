"""
Extract scalar and timeseries quantities from sized GHE + heat-pump objects
for use by the LCOE bridge.

All energy values are expressed in MWh per project step (steps_per_year=1
means one year per step).  For V1 the per-step value is uniform across the
entire project horizon (year-1 loads repeated for every year).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import TYPE_CHECKING

from ghedesigner.enums import BHType

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ghedesigner.ghe.ground_heat_exchangers import GHE
    from ghedesigner.heat_pump_fixed_cop import HeatPumpFixedCOP


@dataclass
class GHEQuantities:
    """All scalar and per-step energy quantities needed by the lCOE bridge file."""

    # --- GHE field geometry (aggregated across all GHEs) ---
    total_drilling_m: float       # total active borehole length  [m]
    n_boreholes: int              # total number of boreholes
    vertical_pipe_m: float        # total pipe length inside boreholes [m]
    grout_volume_m3: float        # total grout fill volume [m³]
    n_heat_pumps: int             # number of heat-pump units

    # --- Year-1 annual energy totals (MWh) ---
    # These are repeated for every year of the LCOE horizon (V1 assumption).
    heat_MWh_yr1: float           # building heating energy delivered
    cool_MWh_yr1: float           # building cooling energy delivered
    elec_heat_MWh_yr1: float      # HP electricity for heating  (heat / COP)
    elec_cool_MWh_yr1: float      # HP electricity for cooling  (cool / COP)


def _grout_volume_for_ghe(ghe: GHE) -> float:
    """
    Compute grout fill volume [m³] for a single GHE.

    Uses the borehole cross-section minus the pipe cross-section(s).
    """
    r_b = ghe.bhe.borehole.r_b        # borehole radius [m]
    r_out = ghe.bhe.pipe.r_out

    bh_area = pi * r_b**2

    match ghe.bhe_type:
        case BHType.SINGLEUTUBE:
            pipe_area = 2.0 * pi * r_out**2
        case BHType.DOUBLEUTUBEPARALLEL | BHType.DOUBLEUTUBESERIES:
            pipe_area = 4.0 * pi * r_out**2
        case _:  # coaxial
            pipe_area = pi * max(r_out) ** 2

    annulus_area = max(bh_area - pipe_area, 0.0)
    return annulus_area * ghe.bhe.borehole.H * ghe.nbh


def _pipe_legs_per_borehole(ghe: GHE) -> int:
    """Return the number of pipe legs (flow paths) per borehole."""
    match ghe.bhe_type:
        case BHType.DOUBLEUTUBEPARALLEL | BHType.DOUBLEUTUBESERIES:
            return 4
        case _:
            return 2  # single U-tube or coaxial


def extract_quantities(
    ghe_objects: Sequence[GHE],
    hp_objects: Sequence[HeatPumpFixedCOP],
) -> GHEQuantities:
    """
    Aggregate GHE geometry and building energy across all GHE / HP objects.

    Parameters
    ----------
    ghe_objects:
        Sized GHE instances (post design_and_size_ghe).
    hp_objects:
        HeatPumpFixedCOP instances whose hourly load series drove the sizing.
        Loads are in Watts (positive = heating, negative = cooling) at hourly
        resolution (8 760 values for one year).
    """
    # --- GHE geometry ---
    total_drilling_m = sum(ghe.bhe.borehole.H * ghe.nbh for ghe in ghe_objects)
    n_boreholes = sum(ghe.nbh for ghe in ghe_objects)

    vertical_pipe_m = sum(
        _pipe_legs_per_borehole(ghe) * ghe.bhe.borehole.H * ghe.nbh
        for ghe in ghe_objects
    )

    grout_volume_m3 = sum(_grout_volume_for_ghe(ghe) for ghe in ghe_objects)

    # --- Building energy (year 1, aggregated across all HP objects) ---
    # hp.loads is an hourly timeseries in W; dividing by 1e6 gives MWh per hour.
    heat_mwh = 0.0
    cool_mwh = 0.0
    elec_heat_mwh = 0.0
    elec_cool_mwh = 0.0

    for hp in hp_objects:
        cop = hp.cop
        for load_w in hp.loads:
            if load_w > 0.0:
                h = load_w / 1e6           # W·h → MWh
                heat_mwh += h
                elec_heat_mwh += h / cop
            elif load_w < 0.0:
                c = abs(load_w) / 1e6
                cool_mwh += c
                elec_cool_mwh += c / cop

    return GHEQuantities(
        total_drilling_m=total_drilling_m,
        n_boreholes=n_boreholes,
        vertical_pipe_m=vertical_pipe_m,
        grout_volume_m3=grout_volume_m3,
        n_heat_pumps=len(hp_objects),
        heat_MWh_yr1=heat_mwh,
        cool_MWh_yr1=cool_mwh,
        elec_heat_MWh_yr1=elec_heat_mwh,
        elec_cool_MWh_yr1=elec_cool_mwh,
    )
