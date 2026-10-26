"""Shared setup for the Modelica borefield comparison tests."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ghedesigner.enums import BHType, TimestepType
from ghedesigner.ghe.boreholes.core import Borehole
from ghedesigner.ghe.coordinates import rectangle
from ghedesigner.ghe.gfunction import calc_g_func_for_multiple_lengths
from ghedesigner.ghe.ground_heat_exchangers import GHE
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.media import Fluid, Grout, Soil
from ghedesigner.utilities import eskilson_log_times

REFERENCE_DATA_DIR = Path(__file__).parent / "test_data" / "modelica_ghe"

BOREHOLE_HEIGHT_M = 150.0
BOREHOLE_BURIAL_DEPTH_M = 1.0
BOREHOLE_RADIUS_M = 0.075
BOREHOLE_SPACING_M = 5
TOTAL_MASS_FLOW_KG_S = 5.0

RMSE_LIMIT_C = 0.20
MEAN_ERROR_LIMIT_C = 0.15
MAX_ABSOLUTE_ERROR_LIMIT_C = 0.40


@dataclass(frozen=True)
class ComparisonMetrics:
    rmse_c: float
    mean_error_c: float
    max_absolute_error_c: float


def rectangular_4_by_5_coordinates() -> list[tuple[float, float]]:
    return rectangle(4, 5, BOREHOLE_SPACING_M, BOREHOLE_SPACING_M)


def l_shaped_11_by_10_coordinates() -> list[tuple[float, float]]:
    horizontal_leg = [(float(x), 0.0) for x in range(0, 55, BOREHOLE_SPACING_M)]
    vertical_leg = [(0.0, float(y)) for y in range(5, 50, BOREHOLE_SPACING_M)]
    return horizontal_leg + vertical_leg


def build_comparison_ghe(coordinates: list[tuple[float, float]], heat_rejection_w: np.ndarray) -> GHE:
    """Build the documented 20-borehole comparison field."""
    fluid = Fluid("Water", temperature=15.0)
    pipe = Pipe.init_single_u_tube(
        conductivity=0.5,
        # Pipe heat capacity was not specified in the comparison document;
        # use GHEDesigner's standard HDPE value.
        rho_cp=1_542_000.0,
        inner_diameter=0.036,
        outer_diameter=0.040,
        # The document locates each pipe center 0.05 m from the borehole
        # center. GHEDesigner's input is the clear spacing between the pipes.
        shank_spacing=0.060,
        roughness=1.0e-6,
    )
    grout = Grout(k=1.15, rho_cp=800.0 * 1600.0)
    soil = Soil(k=2.5, rho_cp=1200.0 * 1800.0, ugt=10.0)
    borehole = Borehole(
        borehole_height=BOREHOLE_HEIGHT_M,
        burial_depth=BOREHOLE_BURIAL_DEPTH_M,
        borehole_radius=BOREHOLE_RADIUS_M,
    )

    number_of_boreholes = len(coordinates)
    mass_flow_per_borehole = TOTAL_MASS_FLOW_KG_S / number_of_boreholes
    volume_flow_system_l_s = TOTAL_MASS_FLOW_KG_S / fluid.rho * 1000.0
    g_function = calc_g_func_for_multiple_lengths(
        BOREHOLE_SPACING_M,
        [BOREHOLE_HEIGHT_M],
        BOREHOLE_RADIUS_M,
        BOREHOLE_BURIAL_DEPTH_M,
        mass_flow_per_borehole,
        BHType.SINGLEUTUBE,
        eskilson_log_times(),
        coordinates,
        fluid,
        pipe,
        grout,
        soil,
    )

    # GHE.simulate() accepts extraction-positive loads and converts them to
    # rejection-positive internally. The Modelica heater convention is the
    # opposite, hence the sign change here.
    extraction_load_w = -np.asarray(heat_rejection_w, dtype=float)
    return GHE(
        volume_flow_system_l_s,
        BOREHOLE_SPACING_M,
        BHType.SINGLEUTUBE,
        fluid,
        borehole,
        pipe,
        grout,
        soil,
        g_function,
        start_month=1,
        end_month=12,
        hourly_extraction_ground_loads=extraction_load_w.tolist(),
    )


def compare_against_modelica(reference_filename: str, coordinates: list[tuple[float, float]]) -> ComparisonMetrics:
    reference = pd.read_csv(REFERENCE_DATA_DIR / reference_filename)
    expected_hours = np.arange(1, 8761, dtype=float)
    np.testing.assert_array_equal(reference["hour"].to_numpy(dtype=float), expected_hours)

    heat_rejection_w = reference["heat_rejection_w"].to_numpy(dtype=float)
    expected_outlet_c = reference["modelica_ghe_outlet_c"].to_numpy(dtype=float)
    assert np.all(np.isfinite(heat_rejection_w))
    assert np.all(np.isfinite(expected_outlet_c))

    ghe = build_comparison_ghe(coordinates, heat_rejection_w)
    ghe.simulate(TimestepType.HOURLY)
    actual_outlet_c = np.asarray(ghe.hp_eft, dtype=float)
    assert actual_outlet_c.shape == expected_outlet_c.shape

    error_c = actual_outlet_c - expected_outlet_c
    metrics = ComparisonMetrics(
        rmse_c=float(np.sqrt(np.mean(error_c**2))),
        mean_error_c=float(np.mean(error_c)),
        max_absolute_error_c=float(np.max(np.abs(error_c))),
    )

    assert metrics.rmse_c <= RMSE_LIMIT_C, (
        f"GHEDesigner/Modelica outlet-temperature RMSE is {metrics.rmse_c:.4f} C; limit is {RMSE_LIMIT_C:.4f} C"
    )
    assert abs(metrics.mean_error_c) <= MEAN_ERROR_LIMIT_C, (
        "GHEDesigner/Modelica absolute mean outlet-temperature error is "
        f"{abs(metrics.mean_error_c):.4f} C; limit is {MEAN_ERROR_LIMIT_C:.4f} C"
    )
    assert metrics.max_absolute_error_c <= MAX_ABSOLUTE_ERROR_LIMIT_C, (
        "GHEDesigner/Modelica maximum outlet-temperature error is "
        f"{metrics.max_absolute_error_c:.4f} C; limit is {MAX_ABSOLUTE_ERROR_LIMIT_C:.4f} C"
    )
    return metrics
